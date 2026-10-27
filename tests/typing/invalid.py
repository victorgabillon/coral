"""Deliberately invalid converter and evaluator contracts."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from valanga import HasTurn

    from coral.neural_networks.input_converters.content_to_input import (
        ContentToInput,
        ContentToInputFunction,
    )
    from coral.neural_networks.nn_state_evaluator import (
        NNBWStateEvaluator,
        NNStateEvaluator,
    )

    from .contracts import Position

bad_converter: ContentToInputFunction[object]  # expect-error
bad_input: ContentToInputFunction[Position, str]  # expect-error
bad_method: ContentToInput[object]  # expect-error
bad_evaluator: NNBWStateEvaluator[object]  # expect-error
bad_protocol: NNStateEvaluator[object]  # expect-error


def reversed_converter(
    converter: ContentToInputFunction[Position],
) -> ContentToInputFunction[HasTurn]:
    """Reject a specialized converter where every state is accepted."""
    return converter  # expect-error


def wrong_result(state: Position) -> str:
    """Reject a string-producing callable as a tensor converter."""
    return str(state.turn)


wrong_tensor: ContentToInputFunction[Position] = wrong_result  # expect-error
