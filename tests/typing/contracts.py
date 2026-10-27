"""Positive tensor-converter variance and evaluator state contracts."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from valanga import Color, HasTurn

if TYPE_CHECKING:
    import torch

    from coral.neural_networks.input_converters.content_to_input import (
        ContentToInput,
        ContentToInputFunction,
    )
    from coral.neural_networks.nn_state_evaluator import (
        NNBWStateEvaluator,
        NNStateEvaluator,
    )


@dataclass(frozen=True)
class Position:
    """Concrete adversarial evaluator input."""

    turn: Color


def callable_contravariance(
    converter: ContentToInputFunction[HasTurn],
) -> ContentToInputFunction[Position]:
    """Accept concrete states through a converter for every acting state."""
    return converter


def method_contravariance(
    converter: ContentToInput[HasTurn],
) -> ContentToInput[Position]:
    """Infer the same contravariance for method-based converters."""
    return converter


def tensor_output(
    converter: ContentToInputFunction[Position], state: Position
) -> torch.Tensor:
    """Expose actual tensors from converters."""
    return converter(state=state)


def evaluator_contract(evaluator: NNBWStateEvaluator[Position]) -> torch.Tensor:
    """Retain the concrete input type in evaluator converters."""
    return evaluator.content_to_input_convert(state=Position(Color.WHITE))


def protocol_input(evaluator: NNStateEvaluator[Position], state: Position) -> float:
    """Couple evaluation and conversion to one state type."""
    return evaluator.evaluate(state).score
