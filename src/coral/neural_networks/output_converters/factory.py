"""Factory helpers for output value converters."""

from coral.board_evaluation import PointOfView
from coral.neural_networks.output_converters.model_output_type import (
    ModelOutputType,
)
from coral.neural_networks.output_converters.output_value_converter import (
    IdentityConverter,
    PlayerToMoveValueToValueWhiteConverter,
    TurnOutputValueConverter,
)


def create_output_converter(
    model_output_type: ModelOutputType,
) -> TurnOutputValueConverter:
    """Create an output converter for the given model output type."""
    output_value_converter: TurnOutputValueConverter
    point_of_view: PointOfView = model_output_type.point_of_view
    match point_of_view:
        case PointOfView.WHITE:
            output_value_converter = IdentityConverter()
        case PointOfView.PLAYER_TO_MOVE:
            output_value_converter = PlayerToMoveValueToValueWhiteConverter()
        case other:
            raise Exception(f"Not a valid output converter: {other} in file{__name__}")

    return output_value_converter
