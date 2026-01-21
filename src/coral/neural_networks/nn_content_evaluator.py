"""
Module for the Neural Network Board Evaluator
"""

from typing import Protocol

import torch
from valanga import (
    FloatyStateEvaluation,
    HasTurn,
    State,
    TurnState,
)

from coral.chi_nn import ChiNN
from coral.neural_networks.input_converters.content_to_input import (
    ContentToInputFunction,
)
from coral.neural_networks.output_converters.output_value_converter import (
    OutputValueConverter,
    TurnOutputValueConverter,
)


class NNContentEvaluator(Protocol):
    """
    Protocol for Neural Network Content Evaluator
    """

    net: ChiNN
    output_and_value_converter: OutputValueConverter
    content_to_input_convert: ContentToInputFunction

    def value_white(self, bw_content: State) -> float:
        """Return the white player value for a given state."""
        ...


class NNBWContentEvaluator:
    """
    The Generic Neural network class for board evaluation

    Attributes:
        net (ChiNN): The neural network model
        output_and_value_converter (OutputValueConverter): The converter for output values
        content_to_input_converter (BoardToInputFunction): The converter for board to input tensor
    """

    net: ChiNN
    output_and_value_converter: TurnOutputValueConverter
    content_to_input_convert: ContentToInputFunction

    def __init__(
        self,
        net: ChiNN,
        output_and_value_converter: TurnOutputValueConverter,
        content_to_input_convert: ContentToInputFunction,
        script: bool = True,
    ) -> None:
        """
        Initialize the NNBoardEvaluator

        Args:
            net (ChiNN): The neural network model
            output_and_value_converter (OutputValueConverter): The converter for output values
            content_to_input_converter (BoardToInputFunction): The converter for board to input tensor
        """
        self.model = torch.jit.script(net) if script else net
        self.model.eval()  # do once
        self.output_and_value_converter = output_and_value_converter
        self.content_to_input_convert = content_to_input_convert

    def value_white(self, state: TurnState) -> float:
        """
        Evaluate the value for the white player

        Args:
            board (BoardChi): The chess board

        Returns:
            float: The value for the white player
        """
        input_layer: torch.Tensor = self.content_to_input_convert(state=state)
        with torch.no_grad():
            output_layer = self.model(input_layer)

        content_evaluation: FloatyStateEvaluation = (
            self.output_and_value_converter.to_content_evaluation(
                output_nn=output_layer, state=state
            )
        )
        value_white: float | None = content_evaluation.value_white
        assert value_white is not None
        return value_white

    def evaluate(
        self, input_layer: torch.Tensor, state: HasTurn
    ) -> FloatyStateEvaluation:
        """
        Evaluate the board position

        Args:
            input_layer (torch.Tensor): The input tensor representing the board position
            color_to_play (chess.Color): The color to play

        Returns:
            FloatyBoardEvaluation: The evaluation of the board position
        """
        self.model.eval()
        with torch.no_grad():
            # run the batch of input_converters into the NN and get the batch of output_converters
            output_layer = self.model(input_layer)

            # translate the NN output batch into a proper Board Evaluations classes in a list
            state_evaluations = self.output_and_value_converter.to_content_evaluation(
                output_nn=output_layer, state=state
            )

        return state_evaluations
