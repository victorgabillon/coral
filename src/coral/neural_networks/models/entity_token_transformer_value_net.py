"""Generic transformer value model for padded entity-token inputs."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from coral.chi_nn import ChiNN
from coral.neural_networks.nn_model_type import NNModelType
from coral.utils.logger import coral_logger


@dataclass(frozen=True)
class EntityTokenTransformerValueNetArgs:
    """Arguments for a generic padded-token transformer value network."""

    type: Literal[NNModelType.ENTITY_TOKEN_TRANSFORMER_VALUE_NET] = (
        NNModelType.ENTITY_TOKEN_TRANSFORMER_VALUE_NET
    )
    input_feature_dim: int = 0
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 4
    dim_feedforward: int = 256
    dropout_ratio: float = 0.0
    use_validity_feature: bool = True
    validity_feature_index: int = -1
    use_value_token: bool = True
    pooling: Literal["value_token", "masked_mean"] = "value_token"
    output_tanh: bool = True

    def __post_init__(self) -> None:
        """Validate model hyperparameters."""
        if self.input_feature_dim <= 0:
            raise ValueError
        if self.d_model <= 0:
            raise ValueError
        if self.n_head <= 0:
            raise ValueError
        if self.n_layer < 0:
            raise ValueError
        if self.dim_feedforward <= 0:
            raise ValueError
        if self.dropout_ratio < 0.0:
            raise ValueError
        if self.d_model % self.n_head != 0:
            raise ValueError
        if self.pooling not in {"value_token", "masked_mean"}:
            raise ValueError
        if self.pooling == "value_token" and not self.use_value_token:
            raise ValueError
        if self.use_validity_feature:
            if self.validity_feature_index >= self.input_feature_dim:
                raise ValueError
            if self.validity_feature_index < -self.input_feature_dim:
                raise ValueError

    def __str__(self) -> str:
        """Return a readable summary of the model arguments."""
        return (
            f"EntityTokenTransformerValueNetArgs("
            f"input_feature_dim={self.input_feature_dim}, "
            f"d_model={self.d_model}, "
            f"n_head={self.n_head}, "
            f"n_layer={self.n_layer}, "
            f"dim_feedforward={self.dim_feedforward}, "
            f"dropout_ratio={self.dropout_ratio}, "
            f"pooling={self.pooling}, "
            f"use_validity_feature={self.use_validity_feature}, "
            f"output_tanh={self.output_tanh})"
        )

    def filename(self) -> str:
        """Generate a stable filename component for this model architecture."""
        validity = "validity" if self.use_validity_feature else "novalidity"
        output = "tanh" if self.output_tanh else "linear"
        return (
            f"{self.type.value}_{self.input_feature_dim}features_"
            f"{self.d_model}dmodel_{self.n_head}head_{self.n_layer}layer_"
            f"{self.dim_feedforward}ff_{self.dropout_ratio:.2f}dropout_"
            f"{self.pooling}_{validity}_{output}"
        )

    @property
    def normalized_validity_feature_index(self) -> int:
        """Return the non-negative validity feature index."""
        return self.validity_feature_index % self.input_feature_dim


class EntityTokenTransformerValueNet(ChiNN):
    """Game-agnostic transformer from padded entity tokens to a scalar value."""

    def __init__(self, args: EntityTokenTransformerValueNetArgs) -> None:
        """Initialize the transformer value network."""
        super().__init__()
        self.args = args

        projected_feature_dim = args.input_feature_dim
        if args.use_validity_feature:
            projected_feature_dim -= 1
        self.input_projection = nn.Linear(projected_feature_dim, args.d_model)

        self.value_token: nn.Parameter | None = None
        if args.use_value_token:
            self.value_token = nn.Parameter(torch.zeros(1, 1, args.d_model))

        if args.n_layer > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.n_head,
                dim_feedforward=args.dim_feedforward,
                dropout=args.dropout_ratio,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder: nn.Module = nn.TransformerEncoder(
                encoder_layer,
                num_layers=args.n_layer,
            )
        else:
            self.encoder = nn.Identity()

        self.value_head = nn.Sequential(
            nn.LayerNorm(args.d_model),
            nn.Linear(args.d_model, args.d_model),
            nn.GELU(),
            nn.Linear(args.d_model, 1),
        )
        self.output_activation = nn.Tanh() if args.output_tanh else nn.Identity()

        self.apply(self._init_weights)
        if self.value_token is not None:
            nn.init.normal_(self.value_token, mean=0.0, std=0.02)

    def init_weights(self) -> None:
        """Initialize model weights."""
        return

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize supported module weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _normalize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Normalize input to batched shape and report whether it was unbatched."""
        if x.dim() == 2:
            normalized_x = x.unsqueeze(0)
            was_unbatched = True
        elif x.dim() == 3:
            normalized_x = x
            was_unbatched = False
        else:
            raise ValueError

        if normalized_x.shape[-1] != self.args.input_feature_dim:
            raise ValueError
        return normalized_x, was_unbatched

    def _validity_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return a bool mask with True for real tokens and False for padding."""
        if not self.args.use_validity_feature:
            return torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        validity_index = self.args.normalized_validity_feature_index
        return x[..., validity_index] > 0.5

    def _remove_validity_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the validity feature column when it is enabled."""
        if not self.args.use_validity_feature:
            return x
        validity_index = self.args.normalized_validity_feature_index
        return torch.cat(
            [x[..., :validity_index], x[..., validity_index + 1 :]],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass for input shaped T x F or B x T x F."""
        x, was_unbatched = self._normalize_input(x)
        valid_tokens = self._validity_mask(x)
        features = self._remove_validity_feature(x)

        hidden = self.input_projection(features)
        hidden = hidden * valid_tokens.unsqueeze(-1).to(hidden.dtype)

        padding_mask = ~valid_tokens
        batch_size = hidden.shape[0]

        if self.args.use_value_token:
            if self.value_token is None:
                raise RuntimeError
            value_token = self.value_token.expand(batch_size, 1, -1)
            hidden = torch.cat([value_token, hidden], dim=1)
            value_padding = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([value_padding, padding_mask], dim=1)

        # PyTorch attention can emit NaN when every position in a row is masked.
        safe_padding_mask = padding_mask
        all_masked = safe_padding_mask.all(dim=1)
        if torch.any(all_masked):
            safe_padding_mask = safe_padding_mask.clone()
            safe_padding_mask[all_masked, 0] = False

        if self.args.n_layer > 0:
            encoded = self.encoder(hidden, src_key_padding_mask=safe_padding_mask)
        else:
            encoded = self.encoder(hidden)

        if self.args.pooling == "value_token":
            pooled = encoded[:, 0, :]
        elif self.args.pooling == "masked_mean":
            entity_encoded = encoded[:, 1:, :] if self.args.use_value_token else encoded
            weights = valid_tokens.to(entity_encoded.dtype).unsqueeze(-1)
            denominator = weights.sum(dim=1).clamp_min(1.0)
            pooled = (entity_encoded * weights).sum(dim=1) / denominator
        else:
            raise RuntimeError

        out = self.output_activation(self.value_head(pooled))
        if was_unbatched:
            return out.squeeze(0)
        return out

    def _parameter_summary(self) -> str:
        """Return a compact parameter summary."""
        lines: list[str] = []
        total_parameters = 0
        for name, parameter in self.named_parameters():
            parameter_count = parameter.numel()
            total_parameters += parameter_count
            shape = "x".join(str(size) for size in parameter.shape)
            lines.append(f"{name}: shape=({shape}), numel={parameter_count}")
        lines.append(f"total_parameters: {total_parameters}")
        return "\n".join(lines)

    def log_readable_model_weights_to_file(self, file_path: str) -> None:
        """Write parameter names, shapes, counts, and total count to a text file."""
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(self._parameter_summary())
            file.write("\n")
        coral_logger.info("Model parameter summary written to %s", file_path)

    def print_param(self) -> None:
        """Log a compact parameter summary."""
        coral_logger.info(self._parameter_summary())
