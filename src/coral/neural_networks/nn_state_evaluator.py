"""Module for the Neural Network Board Evaluator."""

from abc import abstractmethod
from asyncio import Protocol

import torch
from valanga import (
    FloatyStateEvaluation,
    HasTurn,
)

from coral.chi_nn import ChiNN
from coral.neural_networks.input_converters.content_to_input import (
    ContentToInputFunction,
)
from coral.neural_networks.output_converters.output_value_converter import (
    OutputValueConverter,
    TurnOutputValueConverter,
)


class NNStateEvaluator[StateT](Protocol):
    """Protocol for Neural Network Content Evaluator."""

    net: ChiNN
    output_and_value_converter: OutputValueConverter
    content_to_input_convert: ContentToInputFunction[StateT]

    @abstractmethod
    def value_white(self, state: StateT) -> float:  # pylint: disable=unused-argument
        """Return the white player value for a given state."""
        ...


class NNBWStateEvaluator[StateT: HasTurn]:
    """The Generic Neural network class for board evaluation.

    Attributes:
        net (ChiNN): The neural network model
        output_and_value_converter (OutputValueConverter): The converter for output values
        content_to_input_converter (ContentToInputFunction): The converter for board to input tensor

    """

    net: ChiNN
    output_and_value_converter: TurnOutputValueConverter
    content_to_input_convert: ContentToInputFunction[StateT]

    def __init__(
        self,
        net: ChiNN,
        output_and_value_converter: TurnOutputValueConverter,
        content_to_input_convert: ContentToInputFunction[StateT],
        script: bool = True,
    ) -> None:
        """Initialize the NNBoardEvaluator.

        Args:
            net (ChiNN): The neural network model.
            output_and_value_converter (OutputValueConverter): The converter for output values.
            content_to_input_convert (ContentToInputFunction): The converter for board to input tensor.
            script (bool): Whether to torchscript the model.

        """
        self.net = net
        self.scripted_net = torch.jit.script(net) if script else net
        self.net.eval()  # do once
        self.output_and_value_converter = output_and_value_converter
        self.content_to_input_convert = content_to_input_convert

    def value_white(self, state: StateT) -> float:
        """Evaluate the value for the white player.

        Args:
            state (StateT): The state to evaluate.

        Returns:
            float: The value for the white player.

        """
        input_layer: torch.Tensor = self.content_to_input_convert(state=state)
        with torch.no_grad():
            output_layer = self.scripted_net(input_layer)

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
        """Evaluate the board position.

        Args:
            input_layer (torch.Tensor): The input tensor representing the board position
            state (HasTurn): The state with turn information

        Returns:
            FloatyBoardEvaluation: The evaluation of the board position

        """
        self.scripted_net.eval()
        with torch.no_grad():
            # run the batch of input_converters into the NN and get the batch of output_converters
            output_layer = self.scripted_net(input_layer)

            # translate the NN output batch into a proper Board Evaluations classes in a list
            return self.output_and_value_converter.to_content_evaluation(
                output_nn=output_layer, state=state
            )
