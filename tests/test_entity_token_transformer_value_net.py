"""Tests for the generic entity-token transformer value network."""

from typing import Any

import pytest
import torch

from coral.neural_networks.models.entity_token_transformer_value_net import (
    EntityTokenTransformerValueNet,
    EntityTokenTransformerValueNetArgs,
)


def _small_args(**kwargs: Any) -> EntityTokenTransformerValueNetArgs:
    defaults: dict[str, Any] = {
        "input_feature_dim": 5,
        "d_model": 16,
        "n_head": 4,
        "n_layer": 1,
        "dim_feedforward": 32,
        "dropout_ratio": 0.0,
    }
    defaults.update(kwargs)
    return EntityTokenTransformerValueNetArgs(**defaults)


def test_args_filename_is_stable_and_contains_key_fields() -> None:
    """Filename includes enough fields to identify the architecture."""
    args = _small_args(pooling="masked_mean", output_tanh=False)

    assert args.filename() == (
        "entity_token_transformer_value_net_5features_16dmodel_4head_"
        "1layer_32ff_0.00dropout_masked_mean_validity_linear"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_feature_dim": 0},
        {"d_model": 10, "n_head": 4},
        {"pooling": "bad_pooling"},
        {"pooling": "value_token", "use_value_token": False},
        {"validity_feature_index": 5},
        {"validity_feature_index": -6},
    ],
)
def test_invalid_args_are_rejected(kwargs: dict[str, Any]) -> None:
    """Invalid model args fail during construction."""
    with pytest.raises(ValueError):
        _small_args(**kwargs)


def test_forward_accepts_unbatched_input() -> None:
    """Unbatched T x F input returns shape (1,)."""
    model = EntityTokenTransformerValueNet(_small_args())
    x = torch.randn(7, 5)
    x[:, -1] = 1.0

    out = model(x)

    assert out.shape == (1,)


def test_forward_accepts_batched_input() -> None:
    """Batched B x T x F input returns shape B x 1."""
    model = EntityTokenTransformerValueNet(_small_args())
    x = torch.randn(3, 7, 5)
    x[:, :, -1] = 1.0

    out = model(x)

    assert out.shape == (3, 1)


def test_padding_token_content_does_not_affect_output() -> None:
    """Changing invalid token features does not change the model output."""
    torch.manual_seed(0)
    model = EntityTokenTransformerValueNet(_small_args())
    model.eval()
    x1 = torch.randn(2, 6, 5)
    x1[:, :3, -1] = 1.0
    x1[:, 3:, -1] = 0.0
    x2 = x1.clone()
    x2[:, 3:, :-1] = torch.randn(2, 3, 4) * 100_000.0

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_masked_mean_pooling_works() -> None:
    """Masked mean pooling returns a batched scalar value."""
    model = EntityTokenTransformerValueNet(_small_args(pooling="masked_mean"))
    x = torch.randn(2, 4, 5)
    x[:, :2, -1] = 1.0
    x[:, 2:, -1] = 0.0

    out = model(x)

    assert out.shape == (2, 1)


def test_use_validity_feature_false_works() -> None:
    """The model can treat every token as valid."""
    args = _small_args(input_feature_dim=4, use_validity_feature=False)
    model = EntityTokenTransformerValueNet(args)
    x = torch.randn(2, 4, 4)

    out = model(x)

    assert out.shape == (2, 1)


def test_all_padding_input_does_not_nan() -> None:
    """A batch row with no valid entity tokens still returns finite output."""
    model = EntityTokenTransformerValueNet(_small_args(pooling="masked_mean"))
    x = torch.randn(2, 4, 5)
    x[:, :, -1] = 0.0

    out = model(x)

    assert torch.isfinite(out).all()


def test_wrong_rank_raises_value_error() -> None:
    """Only T x F and B x T x F inputs are accepted."""
    model = EntityTokenTransformerValueNet(_small_args())

    with pytest.raises(ValueError):
        model(torch.randn(5))


def test_wrong_feature_dimension_raises_value_error() -> None:
    """Feature dimension must match args.input_feature_dim."""
    model = EntityTokenTransformerValueNet(_small_args())

    with pytest.raises(ValueError):
        model(torch.randn(2, 4))


def test_backward_works() -> None:
    """Gradients flow through the value network."""
    model = EntityTokenTransformerValueNet(_small_args())
    x = torch.randn(2, 4, 5)
    x[:, :, -1] = 1.0

    loss = model(x).sum()
    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
