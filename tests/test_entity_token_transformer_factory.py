"""Factory and architecture parsing tests for the entity-token transformer."""

from enum import Enum

import dacite
import torch

from coral.board_evaluation import PointOfView
from coral.neural_networks.factory import create_nn
from coral.neural_networks.models.entity_token_transformer_value_net import (
    EntityTokenTransformerValueNet,
    EntityTokenTransformerValueNetArgs,
)
from coral.neural_networks.models.relation_biased_entity_token_transformer_value_net import (
    RelationBiasedEntityTokenTransformerValueNet,
    RelationBiasedEntityTokenTransformerValueNetArgs,
)
from coral.neural_networks.neural_net_architecture_args import (
    NeuralNetArchitectureArgs,
)
from coral.neural_networks.nn_model_type import NNModelType
from coral.neural_networks.output_converters.model_output_type import (
    ModelOutputType,
)


def test_factory_creates_entity_token_transformer_value_net() -> None:
    """The main neural-network factory can instantiate the new model."""
    args = EntityTokenTransformerValueNetArgs(
        input_feature_dim=5,
        d_model=16,
        n_head=4,
        n_layer=1,
        dim_feedforward=32,
    )

    net = create_nn(args)

    assert isinstance(net, EntityTokenTransformerValueNet)


def test_neural_net_architecture_args_filename_includes_model_and_point_of_view() -> (
    None
):
    """Architecture filenames compose model type details and output point of view."""
    args = NeuralNetArchitectureArgs(
        model_type_args=EntityTokenTransformerValueNetArgs(
            input_feature_dim=5,
            d_model=16,
            n_head=4,
            n_layer=1,
            dim_feedforward=32,
        ),
        model_output_type=ModelOutputType(
            point_of_view=PointOfView.PLAYER_TO_MOVE,
        ),
    )

    filename = args.filename()

    assert "entity_token_transformer_value_net" in filename
    assert "player_to_move" in filename


def test_dacite_parses_yaml_like_architecture_dict() -> None:
    """Dacite can parse the new args through the architecture union."""
    data = {
        "model_type_args": {
            "type": "entity_token_transformer_value_net",
            "input_feature_dim": 5,
            "d_model": 16,
            "n_head": 4,
            "n_layer": 1,
            "dim_feedforward": 32,
            "dropout_ratio": 0.0,
            "pooling": "masked_mean",
        },
        "model_output_type": {"point_of_view": "player_to_move"},
    }

    args = dacite.from_dict(
        data_class=NeuralNetArchitectureArgs,
        data=data,
        config=dacite.Config(cast=[Enum]),
    )

    assert isinstance(args.model_type_args, EntityTokenTransformerValueNetArgs)
    assert args.model_type_args.type == NNModelType.ENTITY_TOKEN_TRANSFORMER_VALUE_NET
    assert args.model_output_type.point_of_view == PointOfView.PLAYER_TO_MOVE


def test_factory_created_model_forward() -> None:
    """A factory-created entity-token transformer can run a forward pass."""
    model = create_nn(
        EntityTokenTransformerValueNetArgs(
            input_feature_dim=5,
            d_model=16,
            n_head=4,
            n_layer=1,
            dim_feedforward=32,
        )
    )
    x = torch.randn(2, 5, 5)
    x[:, :, -1] = 1.0

    y = model(x)

    assert y.shape == (2, 1)


def test_factory_creates_relation_biased_entity_token_transformer() -> None:
    """The relational subclass is matched before its ordinary parent class."""
    args = RelationBiasedEntityTokenTransformerValueNetArgs(
        input_feature_dim=5,
        d_model=16,
        n_head=4,
        n_layer=1,
        dim_feedforward=32,
        num_relation_types=6,
    )

    net = create_nn(args)

    assert isinstance(net, RelationBiasedEntityTokenTransformerValueNet)


def test_dacite_parses_relation_biased_architecture_dict() -> None:
    """Dacite selects relational args from the architecture union."""
    data = {
        "model_type_args": {
            "type": "relation_biased_entity_token_transformer_value_net",
            "input_feature_dim": 5,
            "d_model": 16,
            "n_head": 4,
            "n_layer": 1,
            "dim_feedforward": 32,
            "dropout_ratio": 0.0,
            "pooling": "value_token",
            "num_relation_types": 6,
        },
        "model_output_type": {"point_of_view": "player_to_move"},
    }

    args = dacite.from_dict(
        data_class=NeuralNetArchitectureArgs,
        data=data,
        config=dacite.Config(cast=[Enum]),
    )

    assert isinstance(
        args.model_type_args,
        RelationBiasedEntityTokenTransformerValueNetArgs,
    )
    assert (
        args.model_type_args.type
        == NNModelType.RELATION_BIASED_ENTITY_TOKEN_TRANSFORMER_VALUE_NET
    )
    assert args.model_type_args.num_relation_types == 6


def test_factory_created_relational_model_forward() -> None:
    """A factory-created relational transformer accepts both model inputs."""
    model = create_nn(
        RelationBiasedEntityTokenTransformerValueNetArgs(
            input_feature_dim=5,
            d_model=16,
            n_head=4,
            n_layer=1,
            dim_feedforward=32,
            num_relation_types=6,
        )
    )
    tokens = torch.randn(2, 5, 5)
    tokens[:, :, -1] = 1.0
    relations = torch.tensor([[[0, 1, 1]], [[2, 3, 2]]])

    output = model(tokens, relations)

    assert output.shape == (2, 1)
