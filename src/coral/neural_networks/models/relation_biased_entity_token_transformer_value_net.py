"""Relation-biased transformer value model for padded entity tokens."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from coral.neural_networks.models.entity_token_transformer_value_net import (
    EntityTokenTransformerValueNet,
    EntityTokenTransformerValueNetArgs,
)
from coral.neural_networks.nn_model_type import NNModelType


@dataclass(frozen=True)
class RelationBiasedEntityTokenTransformerValueNetArgs(
    EntityTokenTransformerValueNetArgs
):
    """Arguments for a relation-biased entity-token transformer."""

    type: Literal[NNModelType.RELATION_BIASED_ENTITY_TOKEN_TRANSFORMER_VALUE_NET] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        NNModelType.RELATION_BIASED_ENTITY_TOKEN_TRANSFORMER_VALUE_NET  # type: ignore[assignment]
    )
    num_relation_types: int = 0
    relation_bias_scale: float = 1.0

    def __post_init__(self) -> None:
        """Validate ordinary and relation-specific hyperparameters."""
        super().__post_init__()
        if self.num_relation_types < 2:
            raise ValueError
        if not math.isfinite(self.relation_bias_scale) or self.relation_bias_scale < 0:
            raise ValueError

    def __str__(self) -> str:
        """Return a readable summary including the relation vocabulary size."""
        base = (
            super()
            .__str__()
            .removeprefix("EntityTokenTransformerValueNetArgs(")
            .removesuffix(")")
        )
        return (
            f"RelationBiasedEntityTokenTransformerValueNetArgs({base}, "
            f"num_relation_types={self.num_relation_types}, "
            f"relation_bias_scale={self.relation_bias_scale})"
        )

    def filename(self) -> str:
        """Generate a stable filename including the relation vocabulary size."""
        return f"{super().filename()}_{self.num_relation_types}relationtypes"


class RelationBiasedEntityTokenTransformerValueNet(EntityTokenTransformerValueNet):
    """Entity-token transformer with shared per-head relation score biases."""

    args: RelationBiasedEntityTokenTransformerValueNetArgs

    def __init__(self, args: RelationBiasedEntityTokenTransformerValueNetArgs) -> None:
        """Initialize the base transformer and zero relation-bias table."""
        super().__init__(args)
        self.relation_bias = nn.Embedding(
            num_embeddings=args.num_relation_types,
            embedding_dim=args.n_head,
            padding_idx=0,
        )
        nn.init.zeros_(self.relation_bias.weight)

    def _normalize_relation_triples(
        self,
        relation_triples: torch.Tensor,
        *,
        batch_size: int,
        was_unbatched: bool,
        expected_device: torch.device,
    ) -> torch.Tensor:
        """Validate relation rank and dtype, then return batched long triples."""
        if relation_triples.device != expected_device:
            raise ValueError(  # noqa: TRY003
                "relation_triples must be on the same device as entity tokens: "
                f"expected {expected_device}, got {relation_triples.device}."
            )
        if (
            relation_triples.dtype == torch.bool
            or relation_triples.is_floating_point()
            or relation_triples.is_complex()
        ):
            raise ValueError

        expected_rank = 2 if was_unbatched else 3
        if relation_triples.dim() != expected_rank:
            raise ValueError
        if relation_triples.shape[-1] != 3:
            raise ValueError
        if not was_unbatched and relation_triples.shape[0] != batch_size:
            raise ValueError

        normalized = relation_triples.to(dtype=torch.long)
        if was_unbatched:
            normalized = normalized.unsqueeze(0)
        return normalized

    def _build_relation_attention_mask(
        self,
        *,
        relation_triples: torch.Tensor,
        valid_entity_tokens: torch.Tensor,
        sequence_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build additive ``[B * H, S, S]`` biases from directed triples.

        Triples are ``[source entity, destination entity, relation type]``.
        Type zero rows are ignored. For active rows, the source is the attention
        query and the destination is the key. Entity indices are offset when the
        learned value token is present. The dense mask uses quadratic sequence
        memory and is intended as the minimal correctness-first implementation.
        """
        if torch.any(relation_triples[..., 2] < 0):
            raise ValueError

        active = relation_triples[..., 2] > 0
        if not torch.any(active):
            return None

        active_batch_indices = (
            torch.arange(relation_triples.shape[0], device=device)
            .unsqueeze(1)
            .expand(active.shape)[active]
        )
        active_triples = relation_triples[active]
        source_indices = active_triples[:, 0]
        destination_indices = active_triples[:, 1]

        if torch.any(active_triples[:, 2] >= self.args.num_relation_types):
            raise ValueError
        if torch.any(source_indices < 0) or torch.any(
            source_indices >= valid_entity_tokens.shape[1]
        ):
            raise ValueError
        if torch.any(destination_indices < 0) or torch.any(
            destination_indices >= valid_entity_tokens.shape[1]
        ):
            raise ValueError
        if torch.any(~valid_entity_tokens[active_batch_indices, source_indices]):
            raise ValueError
        if torch.any(~valid_entity_tokens[active_batch_indices, destination_indices]):
            raise ValueError

        if self.args.relation_bias_scale == 0:
            return None
        bias_per_relation = (
            self.relation_bias(active_triples[:, 2]) * self.args.relation_bias_scale
        )
        entity_index_offset = int(self.args.use_value_token)
        # Broadcast relation rows against heads, retaining the original row-major
        # index_add order (including duplicate relations and their gradients).
        flat_attention_indices = (
            (
                (
                    active_batch_indices.unsqueeze(1) * self.args.n_head
                    + torch.arange(self.args.n_head, device=device)
                )
                * sequence_length
                + source_indices.unsqueeze(1)
                + entity_index_offset
            )
            * sequence_length
            + destination_indices.unsqueeze(1)
            + entity_index_offset
        ).reshape(-1)
        flat_attention_bias = torch.zeros(
            relation_triples.shape[0]
            * self.args.n_head
            * sequence_length
            * sequence_length,
            dtype=dtype,
            device=device,
        )
        flat_attention_bias = flat_attention_bias.index_add(
            0,
            flat_attention_indices,
            bias_per_relation.to(dtype=dtype).reshape(-1),
        )
        return flat_attention_bias.reshape(
            relation_triples.shape[0] * self.args.n_head,
            sequence_length,
            sequence_length,
        )

    def forward(
        self, x: torch.Tensor, relation_triples: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Evaluate tokens and sparse directed ``[source, destination, type]`` rows."""
        if relation_triples is None:
            raise TypeError
        hidden, safe_padding_mask, valid_tokens, was_unbatched = (
            self._prepare_encoder_inputs(x)
        )
        normalized_relations = self._normalize_relation_triples(
            relation_triples,
            batch_size=hidden.shape[0],
            was_unbatched=was_unbatched,
            expected_device=hidden.device,
        )
        attention_mask = self._build_relation_attention_mask(
            relation_triples=normalized_relations,
            valid_entity_tokens=valid_tokens,
            sequence_length=hidden.shape[1],
            dtype=hidden.dtype,
            device=hidden.device,
        )
        encoded = self._encode(
            hidden,
            safe_padding_mask,
            attention_mask=attention_mask,
        )
        return self._finalize_output(encoded, valid_tokens, was_unbatched)
