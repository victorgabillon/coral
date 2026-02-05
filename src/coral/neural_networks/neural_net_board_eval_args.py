"""Module that contains the NeuralNetBoardEvalArgs class."""

from dataclasses import dataclass, field
from typing import Literal

from coral.board_evaluation import (
    PointOfView,
)
from coral.neural_networks.factory import (
    NeuralNetModelsAndArchitecture,
)
from coral.neural_networks.models.multi_layer_perceptron import (
    MultiLayerPerceptronArgs,
)
from coral.neural_networks.neural_net_architecture_args import (
    NeuralNetArchitectureArgs,
)
from coral.neural_networks.nn_model_type import (
    ActivationFunctionType,
    NNModelType,
)
from coral.neural_networks.output_converters.model_output_type import (
    ModelOutputType,
)

NN_NET_EVAL_LITERAL_STRING: Literal["neural_network"] = "neural_network"
NN_NET_EVAL_STRING: str = NN_NET_EVAL_LITERAL_STRING


class NeuralNetBoardEvalArgsError(ValueError):
    """Invalid neural net board eval args."""


class UnexpectedBoardEvalTypeError(NeuralNetBoardEvalArgsError):
    def __init__(self) -> None:
        super().__init__("Expecting neural_network as name")


class MissingPathToNnFolderError(NeuralNetBoardEvalArgsError):
    def __init__(self, module_name: str) -> None:
        super().__init__(f"Expecting a path_to_nn_folder in {module_name}")


@dataclass
class NeuralNetBoardEvalArgs:
    """NeuralNetBoardEvalArgs encapsulates the configuration arguments required for evaluating board positions
    using a neural network-based node evaluator.

    Attributes:
        neural_nets_model_and_architecture (NeuralNetModelsAndArchitecture):
            Specifies the neural network model, its architecture, and associated parameters.
            Defaults to a MultiLayerPerceptron with predefined layer sizes and activation functions.
        type (Literal[NodeEvaluatorTypes.NeuralNetwork]):
            The type of node evaluator, which must be set to 'NeuralNetwork'.

    """

    type: Literal["neural_network"] = NN_NET_EVAL_LITERAL_STRING
    neural_nets_model_and_architecture: NeuralNetModelsAndArchitecture = field(
        default_factory=lambda: NeuralNetModelsAndArchitecture(
            model_weights_file_name="*default*",
            nn_architecture_args=NeuralNetArchitectureArgs(
                model_type_args=MultiLayerPerceptronArgs(
                    type=NNModelType.MULTI_LAYER_PERCEPTRON,
                    number_neurons_per_layer=[5, 1],
                    list_of_activation_functions=[
                        ActivationFunctionType.TANGENT_HYPERBOLIC
                    ],
                ),
                model_output_type=ModelOutputType(
                    point_of_view=PointOfView.PLAYER_TO_MOVE
                ),
            ),
        )
    )

    def __post_init__(self) -> None:
        """Performs additional initialization after the object is created.

        Raises:
            ValueError: If the type is not NodeEvaluatorTypes.NeuralNetwork.

        """
        if self.type != NN_NET_EVAL_LITERAL_STRING:
            raise UnexpectedBoardEvalTypeError
        if (
            self.neural_nets_model_and_architecture.model_weights_file_name
            == "*default*"
        ):
            raise MissingPathToNnFolderError(__name__)
