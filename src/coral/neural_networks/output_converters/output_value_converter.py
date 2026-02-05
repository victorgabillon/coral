"""Module for converting the output of the neural network to a board evaluation."""

from abc import ABC, abstractmethod

import torch
from valanga import Color, FloatyStateEvaluation, HasTurn, State


class OutputValueConverter(ABC):
    """Convert a neural network output to a board evaluation.

    Convert a board evaluation back into a neural network output.
    """

    @abstractmethod
    def to_content_evaluation(
        self, output_nn: torch.Tensor, state: State
    ) -> FloatyStateEvaluation:
        """Convert the output of the neural network to a content evaluation.

        Args:
            output_nn (torch.Tensor): The output of the neural network.
            state (State): The state used for the evaluation.

        Returns:
            FloatyBoardEvaluation: The converted board evaluation.

        """
        ...

    @abstractmethod
    def from_value_white_to_model_output(
        self, content_value_white: float, content_with_turn: State
    ) -> torch.Tensor:
        """Convert the white value to the corresponding NN model output.

        Use this conversion in training when comparing value_white against float targets.
        """


class TurnOutputValueConverter(ABC):
    """Convert a neural network output to a board evaluation.

    Convert a board evaluation back into a neural network output.
    """

    @abstractmethod
    def to_content_evaluation(
        self, output_nn: torch.Tensor, state: HasTurn
    ) -> FloatyStateEvaluation:
        """Convert the output of the neural network to a content evaluation.

        Args:
            output_nn (torch.Tensor): The output of the neural network.
            state (HasTurn): The state used for the evaluation.

        Returns:
            FloatyBoardEvaluation: The converted board evaluation.

        """
        ...

    @abstractmethod
    def from_value_white_to_model_output(
        self, content_value_white: float, content_with_turn: HasTurn
    ) -> torch.Tensor:
        """Convert the white value to the corresponding NN model output.

        Use this conversion in training when comparing value_white against float targets.
        """


# TODO(victor): This part should be reformated/improved as two concept are mixed See  issue #24 for more details
#  the convertion from the model output to  value white and the conversion to the final object used in pytorch as FloatyBoardEvaluation
#  maybe just rewrite a bit to make it less confusing
#  and the naming is not great as now we have multi option.


class PlayerToMoveValueToValueWhiteConverter(TurnOutputValueConverter):
    """Converting from a NN that outputs a 1D value from the point of view of the player to move."""

    def convert_value_from_mover_viewpoint_to_value_white(
        self, turn: Color, value_from_mover_view_point: float
    ) -> float:
        """Convert the value from the mover's viewpoint to the value from the white player's viewpoint.

        Args:
            turn (Color): The color of the player to move.
            value_from_mover_view_point (float): The value from the mover's viewpoint.

        Returns:
            float: The value from the white player's viewpoint.

        """
        if turn == Color.BLACK:
            value_white = -value_from_mover_view_point
        else:
            value_white = value_from_mover_view_point
        return value_white

    def to_content_evaluation(
        self, output_nn: torch.Tensor, state: HasTurn
    ) -> FloatyStateEvaluation:
        """Convert the output of the neural network to a content evaluation.

        Args:
            output_nn (torch.Tensor): The output of the neural network.
            state (HasTurn): The state used for the evaluation.

        Returns:
            FloatyBoardEvaluation: The converted board evaluation.

        """
        value: float = output_nn.item()
        value_white: float = self.convert_value_from_mover_viewpoint_to_value_white(
            turn=state.turn, value_from_mover_view_point=value
        )
        state_evaluation: FloatyStateEvaluation = FloatyStateEvaluation(
            value_white=value_white
        )
        return state_evaluation

    def from_value_white_to_model_output(
        self, content_value_white: float, content_with_turn: HasTurn
    ) -> torch.Tensor:
        """Convert the white value to the corresponding NN model output.

        Use this conversion in training when comparing value_white against float targets.
        """
        value_from_mover_view_point: float
        if content_with_turn.turn == Color.BLACK:
            value_from_mover_view_point = -content_value_white
        else:
            value_from_mover_view_point = content_value_white
        return torch.tensor([value_from_mover_view_point])


class IdentityConverter(TurnOutputValueConverter):
    """Converting from a NN that outputs a 1D value from the point of view of the player to move."""

    def to_content_evaluation(
        self, output_nn: torch.Tensor, state: HasTurn
    ) -> FloatyStateEvaluation:
        """Convert the output of the neural network to a content evaluation.

        Args:
            output_nn (torch.Tensor): The output of the neural network.
            state (HasTurn): The state used for the evaluation.

        Returns:
            FloatyBoardEvaluation: The converted board evaluation.

        """
        value_white: float = output_nn.item()
        state_evaluation: FloatyStateEvaluation = FloatyStateEvaluation(
            value_white=value_white
        )
        return state_evaluation

    def from_value_white_to_model_output(
        self, content_value_white: float, content_with_turn: HasTurn
    ) -> torch.Tensor:
        """Convert the white value to the corresponding NN model output.

        Use this conversion in training when comparing value_white against float targets.
        """
        return torch.tensor([content_value_white])
