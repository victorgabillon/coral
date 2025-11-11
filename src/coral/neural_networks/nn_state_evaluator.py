"""
Module for the Neural Network Board Evaluator
"""

import torch
from valanga import (
    Color,
    FloatyBoardEvaluation,
    HasTurn,
)

from coral.chi_nn import ChiNN
from coral.neural_networks.input_converters.content_to_input import (
    ContentToInputFunction,
)
from coral.neural_networks.output_converters.output_value_converter import (
    OutputValueConverter,
)


class NNBWStateEvaluator:
    """
    The Generic Neural network class for board evaluation

    Attributes:
        net (ChiNN): The neural network model
        output_and_value_converter (OutputValueConverter): The converter for output values
        content_to_input_converter (BoardToInputFunction): The converter for board to input tensor
    """

    net: ChiNN
    output_and_value_converter: OutputValueConverter
    content_to_input_convert: ContentToInputFunction

    def __init__(
        self,
        net: ChiNN,
        output_and_value_converter: OutputValueConverter,
        content_to_input_convert: ContentToInputFunction,
    ) -> None:
        """
        Initialize the NNBoardEvaluator

        Args:
            net (ChiNN): The neural network model
            output_and_value_converter (OutputValueConverter): The converter for output values
            content_to_input_converter (BoardToInputFunction): The converter for board to input tensor
        """
        self.net = net
        self.my_scripted_model = torch.jit.script(net)
        self.output_and_value_converter = output_and_value_converter
        self.content_to_input_convert = content_to_input_convert

    def value_white(self, bw_content: HasTurn) -> float:
        """
        Evaluate the value for the white player

        Args:
            board (BoardChi): The chess board

        Returns:
            float: The value for the white player
        """
        self.my_scripted_model.eval()
        input_layer: torch.Tensor = self.content_to_input_convert(
            content_with_turn=bw_content
        )
        torch.no_grad()
        output_layer: torch.Tensor = self.my_scripted_model(input_layer)
        torch.no_grad()
        content_evaluation: FloatyBoardEvaluation = (
            self.output_and_value_converter.to_content_evaluation(
                output_nn=output_layer, color_to_play=bw_content.turn
            )
        )
        value_white: float | None = content_evaluation.value_white
        assert value_white is not None
        return value_white

    def evaluate(
        self, input_layer: torch.Tensor, color_to_play: Color
    ) -> FloatyBoardEvaluation:
        """
        Evaluate the board position

        Args:
            input_layer (torch.Tensor): The input tensor representing the board position
            color_to_play (chess.Color): The color to play

        Returns:
            FloatyBoardEvaluation: The evaluation of the board position
        """
        self.my_scripted_model.eval()
        torch.no_grad()

        # run the batch of input_converters into the NN and get the batch of output_converters
        output_layer = self.my_scripted_model(input_layer)

        # translate the NN output batch into a proper Board Evaluations classes in a list
        board_evaluations = self.output_and_value_converter.to_content_evaluation(
            output_nn=output_layer, color_to_play=color_to_play
        )

        return board_evaluations
