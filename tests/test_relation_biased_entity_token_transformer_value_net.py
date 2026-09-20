"""Tests for the relation-biased entity-token transformer value network."""

from typing import Any

import pytest
import torch

from coral.neural_networks.models.entity_token_transformer_value_net import (
    EntityTokenTransformerValueNet,
    EntityTokenTransformerValueNetArgs,
)
from coral.neural_networks.models.relation_biased_entity_token_transformer_value_net import (
    RelationBiasedEntityTokenTransformerValueNet,
    RelationBiasedEntityTokenTransformerValueNetArgs,
)


def _small_args(
    **kwargs: Any,
) -> RelationBiasedEntityTokenTransformerValueNetArgs:
    defaults: dict[str, Any] = {
        "input_feature_dim": 5,
        "d_model": 16,
        "n_head": 4,
        "n_layer": 1,
        "dim_feedforward": 32,
        "dropout_ratio": 0.0,
        "num_relation_types": 6,
    }
    defaults.update(kwargs)
    return RelationBiasedEntityTokenTransformerValueNetArgs(**defaults)


def _valid_tokens(*shape: int) -> torch.Tensor:
    tokens = torch.randn(*shape, 5)
    tokens[..., -1] = 1.0
    return tokens


@pytest.mark.parametrize("num_relation_types", [0, 1])
def test_too_few_relation_types_are_rejected(num_relation_types: int) -> None:
    """The vocabulary reserves zero and requires at least one real type."""
    with pytest.raises(ValueError):
        _small_args(num_relation_types=num_relation_types)


def test_two_relation_types_are_accepted() -> None:
    """A padding type and one active relation type are sufficient."""
    assert _small_args(num_relation_types=2).num_relation_types == 2


@pytest.mark.parametrize("scale", [-0.1, float("inf"), float("nan")])
def test_invalid_relation_scale_is_rejected(scale: float) -> None:
    """Persisted relation scaling must be finite and nonnegative."""
    with pytest.raises(ValueError):
        _small_args(relation_bias_scale=scale)


def test_relation_scale_matches_scaled_weights_and_scales_gradients() -> None:
    """The 0.25 configuration scales actual attention, including its gradient."""
    torch.manual_seed(3)
    scaled = RelationBiasedEntityTokenTransformerValueNet(
        _small_args(relation_bias_scale=0.25)
    )
    reference = RelationBiasedEntityTokenTransformerValueNet(_small_args())
    with torch.no_grad():
        scaled.relation_bias.weight[1].copy_(torch.tensor([0.3, -0.5, 0.8, 0.2]))
    reference.load_state_dict(scaled.state_dict())
    with torch.no_grad():
        reference.relation_bias.weight.mul_(0.25)
    tokens = _valid_tokens(1, 5)
    relations = torch.tensor([[[0, 1, 1]]])
    actual = scaled(tokens, relations)
    expected = reference(tokens, relations)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    assert scaled.relation_bias.weight.grad is not None
    assert reference.relation_bias.weight.grad is not None
    torch.testing.assert_close(
        scaled.relation_bias.weight.grad, reference.relation_bias.weight.grad * 0.25
    )


def test_eval_preserves_additive_bias_magnitudes() -> None:
    """Inference must agree with the unfused training computation at dropout zero."""
    torch.manual_seed(31)
    model = RelationBiasedEntityTokenTransformerValueNet(
        _small_args(relation_bias_scale=0.25)
    )
    with torch.no_grad():
        model.relation_bias.weight[1].copy_(torch.tensor([0.5, 1.0, -0.5, -1.0]))
    tokens = _valid_tokens(2, 5)
    relations = torch.tensor([[[0, 1, 1]], [[2, 3, 1]]])
    with torch.inference_mode():
        model.train()
        expected = model(tokens, relations)
        model.eval()
        actual = model(tokens, relations)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_scaling_preserves_legacy_weight_schema_and_zero_disables_bias() -> None:
    """Scaling adds no weight keys and zero ignores nonzero learned relation biases."""
    legacy = RelationBiasedEntityTokenTransformerValueNet(_small_args())
    disabled = RelationBiasedEntityTokenTransformerValueNet(
        _small_args(relation_bias_scale=0.0)
    )
    with torch.no_grad():
        legacy.relation_bias.weight[1].fill_(0.5)
    disabled.load_state_dict(legacy.state_dict(), strict=True)
    assert set(legacy.state_dict()) == set(disabled.state_dict())
    disabled.eval()
    tokens = _valid_tokens(1, 5)
    with torch.inference_mode():
        actual = disabled(tokens, torch.tensor([[[0, 1, 1]]]))
        expected = disabled(tokens, torch.empty(1, 0, 3, dtype=torch.long))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_additive_encoder_matches_standard_unfused_training_computation() -> None:
    """The inference correction preserves the original public encoder computation."""
    torch.manual_seed(11)
    model = RelationBiasedEntityTokenTransformerValueNet(_small_args())
    model.train()
    hidden = torch.randn(2, 6, 16)
    padding = torch.zeros(2, 6, dtype=torch.bool)
    padding[:, -1] = True
    additive_padding = torch.zeros(2, 6).masked_fill(padding, float("-inf"))
    attention_bias = torch.randn(8, 6, 6) * 0.25

    expected = model.encoder(
        hidden, mask=attention_bias, src_key_padding_mask=additive_padding
    )
    actual = model._encode(hidden, padding, attention_mask=attention_bias)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_args_filename_and_string_include_relation_count() -> None:
    """Architecture descriptions distinguish relation vocabulary sizes."""
    args = _small_args(pooling="masked_mean", output_tanh=False)

    assert args.filename() == (
        "relation_biased_entity_token_transformer_value_net_5features_16dmodel_"
        "4head_1layer_32ff_0.00dropout_masked_mean_validity_linear_6relationtypes"
    )
    assert "num_relation_types=6" in str(args)


def test_forward_accepts_unbatched_tokens_and_relations() -> None:
    """Unbatched token and relation inputs return one scalar."""
    model = RelationBiasedEntityTokenTransformerValueNet(_small_args())
    output = model(_valid_tokens(5), torch.tensor([[0, 1, 1]]))

    assert output.shape == (1,)


def test_forward_accepts_batched_tokens_and_relations() -> None:
    """Batched token and matching relation inputs preserve the batch size."""
    model = RelationBiasedEntityTokenTransformerValueNet(_small_args())
    relations = torch.tensor([[[0, 1, 1]], [[2, 3, 2]]])

    output = model(_valid_tokens(2, 5), relations)

    assert output.shape == (2, 1)


def test_empty_relations_work_for_both_batch_conventions() -> None:
    """Empty sparse relation lists do not require a dense attention mask."""
    model = RelationBiasedEntityTokenTransformerValueNet(_small_args())

    unbatched = model(_valid_tokens(5), torch.empty((0, 3), dtype=torch.long))
    batched = model(_valid_tokens(2, 5), torch.empty((2, 0, 3), dtype=torch.long))

    assert unbatched.shape == (1,)
    assert batched.shape == (2, 1)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_relation_and_token_devices_must_match() -> None:
    """Relation tensors must already use the entity-token device."""
    model = RelationBiasedEntityTokenTransformerValueNet(_small_args()).cuda()
    tokens = _valid_tokens(5).cuda()
    relations = torch.tensor([[0, 1, 1]], device="cpu")

    with pytest.raises(ValueError, match="same device"):
        model(tokens, relations)


def test_type_zero_rows_are_ignored_before_index_validation() -> None:
    """Padding relation rows may contain arbitrary endpoint indices."""
    torch.manual_seed(0)
    model = RelationBiasedEntityTokenTransformerValueNet(_small_args())
    tokens = _valid_tokens(5)
    empty = torch.empty((0, 3), dtype=torch.long)
    padded = torch.tensor([[0, 0, 0], [999, -999, 0]])

    torch.testing.assert_close(model(tokens, empty), model(tokens, padded))


@pytest.mark.parametrize(
    "relations",
    [
        torch.zeros((1, 3), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.bool),
        torch.zeros(3, dtype=torch.long),
        torch.zeros((1, 1, 3), dtype=torch.long),
        torch.zeros((1, 4), dtype=torch.long),
    ],
)
def test_unbatched_relation_shape_and_dtype_validation(
    relations: torch.Tensor,
) -> None:
    """Unbatched relations require an integer R x 3 tensor."""
    model = RelationBiasedEntityTokenTransformerValueNet(_small_args())

    with pytest.raises(ValueError):
        model(_valid_tokens(5), relations)


@pytest.mark.parametrize(
    "relations",
    [
        torch.zeros((1, 3), dtype=torch.long),
        torch.zeros((2, 1, 4), dtype=torch.long),
        torch.zeros((3, 1, 3), dtype=torch.long),
    ],
)
def test_batched_relation_shape_validation(relations: torch.Tensor) -> None:
    """Batched relations require B x R x 3 with matching batch size."""
    model = RelationBiasedEntityTokenTransformerValueNet(_small_args())

    with pytest.raises(ValueError):
        model(_valid_tokens(2, 5), relations)


@pytest.mark.parametrize(
    "triple",
    [
        [-1, 1, 1],
        [0, -1, 1],
        [5, 1, 1],
        [0, 5, 1],
        [0, 1, -1],
        [0, 1, 6],
    ],
)
def test_active_relation_bounds_are_validated(triple: list[int]) -> None:
    """Every active endpoint and type must be in range."""
    model = RelationBiasedEntityTokenTransformerValueNet(_small_args())

    with pytest.raises(ValueError):
        model(_valid_tokens(5), torch.tensor([triple]))


@pytest.mark.parametrize("triple", [[2, 0, 1], [0, 2, 1]])
def test_relations_cannot_reference_padded_tokens(triple: list[int]) -> None:
    """Active relation endpoints must refer to valid entity rows."""
    model = RelationBiasedEntityTokenTransformerValueNet(_small_args())
    tokens = _valid_tokens(4)
    tokens[2, -1] = 0.0

    with pytest.raises(ValueError):
        model(tokens, torch.tensor([triple]))


def test_value_token_offset_is_applied_to_relation_mask() -> None:
    """Entity indices are shifted past the learned value token."""
    model = RelationBiasedEntityTokenTransformerValueNet(_small_args())
    with torch.no_grad():
        model.relation_bias.weight[1].copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    mask = model._build_relation_attention_mask(
        relation_triples=torch.tensor([[[0, 1, 1]]]),
        valid_entity_tokens=torch.ones((1, 3), dtype=torch.bool),
        sequence_length=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert mask is not None
    dense = mask.reshape(1, 4, 4, 4)
    torch.testing.assert_close(dense[0, :, 1, 2], torch.arange(1.0, 5.0))
    torch.testing.assert_close(dense[0, :, 0, 1], torch.zeros(4))


def test_relation_mask_has_no_offset_without_value_token() -> None:
    """Entity indices remain unchanged when no value token is configured."""
    model = RelationBiasedEntityTokenTransformerValueNet(
        _small_args(use_value_token=False, pooling="masked_mean")
    )
    with torch.no_grad():
        model.relation_bias.weight[1].copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    mask = model._build_relation_attention_mask(
        relation_triples=torch.tensor([[[0, 1, 1]]]),
        valid_entity_tokens=torch.ones((1, 3), dtype=torch.bool),
        sequence_length=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert mask is not None
    dense = mask.reshape(1, 4, 3, 3)
    torch.testing.assert_close(dense[0, :, 0, 1], torch.arange(1.0, 5.0))


def test_duplicate_relations_accumulate() -> None:
    """Repeated source-destination pairs sum their per-head biases."""
    model = RelationBiasedEntityTokenTransformerValueNet(_small_args())
    with torch.no_grad():
        model.relation_bias.weight[1].copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    mask = model._build_relation_attention_mask(
        relation_triples=torch.tensor([[[0, 1, 1], [0, 1, 1]]]),
        valid_entity_tokens=torch.ones((1, 3), dtype=torch.bool),
        sequence_length=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert mask is not None
    dense = mask.reshape(1, 4, 4, 4)
    torch.testing.assert_close(dense[0, :, 1, 2], 2.0 * torch.arange(1.0, 5.0))


def test_zero_bias_is_strict_extension_of_ordinary_model() -> None:
    """Zero relation biases reproduce the ordinary transformer's output."""
    torch.manual_seed(7)
    ordinary_args = EntityTokenTransformerValueNetArgs(
        input_feature_dim=5,
        d_model=16,
        n_head=4,
        n_layer=1,
        dim_feedforward=32,
        dropout_ratio=0.0,
        output_tanh=False,
    )
    relational_args = _small_args(output_tanh=False)
    ordinary = EntityTokenTransformerValueNet(ordinary_args)
    relational = RelationBiasedEntityTokenTransformerValueNet(relational_args)
    incompatible = relational.load_state_dict(ordinary.state_dict(), strict=False)

    assert incompatible.missing_keys == ["relation_bias.weight"]
    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(
        relational.relation_bias.weight,
        torch.zeros_like(relational.relation_bias.weight),
    )

    tokens = _valid_tokens(2, 5)
    relations = torch.tensor([[[0, 1, 1]], [[2, 3, 2]]])
    torch.testing.assert_close(
        ordinary(tokens),
        relational(tokens, relations),
        rtol=1e-5,
        atol=1e-6,
    )


def test_only_used_nonzero_relation_bias_changes_output() -> None:
    """Attention responds to a used relation row but not an unused row."""
    torch.manual_seed(11)
    model = RelationBiasedEntityTokenTransformerValueNet(
        _small_args(output_tanh=False, pooling="masked_mean")
    )
    tokens = _valid_tokens(2, 5)
    relations = torch.tensor([[[0, 1, 1]], [[2, 3, 1]]])
    zero_output = model(tokens, relations)

    with torch.no_grad():
        model.relation_bias.weight[2].copy_(torch.tensor([20.0, -10.0, 5.0, -2.0]))
    unused_output = model(tokens, relations)
    torch.testing.assert_close(zero_output, unused_output)

    with torch.no_grad():
        model.relation_bias.weight[1].copy_(torch.tensor([20.0, -10.0, 5.0, -2.0]))
    used_output = model(tokens, relations)

    assert not torch.allclose(zero_output, used_output)


def test_gradient_reaches_used_relation_bias_only() -> None:
    """Indexed dense-mask accumulation preserves relation-table gradients."""
    torch.manual_seed(13)
    model = RelationBiasedEntityTokenTransformerValueNet(
        _small_args(output_tanh=False, pooling="masked_mean")
    )
    relations = torch.tensor([[0, 1, 1], [0, 1, 1]])

    model(_valid_tokens(5), relations).sum().backward()

    gradient = model.relation_bias.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient[1]).all()
    assert gradient[1].abs().sum() > 0
    torch.testing.assert_close(gradient[0], torch.zeros_like(gradient[0]))


def test_parameter_count_increase_is_relation_types_times_heads() -> None:
    """The relation model adds exactly one scalar per type and head."""
    ordinary = EntityTokenTransformerValueNet(
        EntityTokenTransformerValueNetArgs(
            input_feature_dim=5,
            d_model=16,
            n_head=4,
            n_layer=1,
            dim_feedforward=32,
        )
    )
    relational = RelationBiasedEntityTokenTransformerValueNet(_small_args())

    ordinary_count = sum(parameter.numel() for parameter in ordinary.parameters())
    relational_count = sum(parameter.numel() for parameter in relational.parameters())
    assert relational_count - ordinary_count == 6 * 4


def test_zero_layer_forward_is_finite() -> None:
    """Relations are accepted even when there is no attention layer."""
    model = RelationBiasedEntityTokenTransformerValueNet(_small_args(n_layer=0))

    output = model(_valid_tokens(2, 5), torch.tensor([[[0, 1, 1]], [[1, 2, 2]]]))

    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()
