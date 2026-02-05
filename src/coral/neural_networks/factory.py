"""Module for creating neural networks and neural network board evaluators."""

import os.path
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import dacite
from valanga import HasTurn

from coral.chi_nn import ChiNN
from coral.neural_networks.input_converters.content_to_input import (
    ContentToInputFunction,
)
from coral.neural_networks.models.multi_layer_perceptron import (
    MultiLayerPerceptron,
    MultiLayerPerceptronArgs,
)
from coral.neural_networks.models.transformer_one import (
    TransformerArgs,
    TransformerOne,
)
from coral.neural_networks.neural_net_architecture_args import (
    NeuralNetArchitectureArgs,
)
from coral.neural_networks.nn_model_type_args import (
    NNModelTypeArgs,
)
from coral.neural_networks.nn_state_evaluator import (
    NNBWStateEvaluator,
)
from coral.neural_networks.output_converters.factory import (
    create_output_converter,
)
from coral.utils.small_tools import MyPath, yaml_fetch_args_in_file

if TYPE_CHECKING:
    from coral.neural_networks.output_converters.output_value_converter import (
        TurnOutputValueConverter,
    )


class ArchitectureFileNotFoundError(FileNotFoundError):
    """Raise when the architecture file is missing."""

    def __init__(self, filename: str) -> None:
        """Initialize the error with the missing filename."""
        super().__init__(f"this is not a file {filename}")


@dataclass
class NeuralNetModelsAndArchitecture:
    """Class to hold the neural network models and architecture.

    Attributes:
        model_weights_file_name (path): The file name of the model weights.
        nn_architecture_args (NeuralNetArchitectureArgs): The neural network architecture arguments.

    """

    model_weights_file_name: MyPath
    nn_architecture_args: NeuralNetArchitectureArgs

    @classmethod
    def build_from_folder_path(
        cls, folder_path: MyPath
    ) -> "NeuralNetModelsAndArchitecture":
        """Build an instance of NeuralNetModelsAndArchitecture from the given folder path.

        Args:
            folder_path (Path): Path to the folder containing 'architecture.yaml' and model weights.

        Returns:
            NeuralNetModelsAndArchitecture: An initialized instance.

        """
        nn_args = get_architecture_args_from_folder(folder_path=folder_path)
        model_file = os.path.join(folder_path, nn_args.filename() + ".pt")

        return cls(model_weights_file_name=model_file, nn_architecture_args=nn_args)


def get_nn_param_file_path_from(
    folder_path: MyPath, file_name: str | None = None
) -> tuple[str, str]:
    """Get the file path for the neural network parameters.

    Args:
        folder_path (str): The folder path for the neural network parameters.
        file_name (str | None): Optional base filename for the parameter files.

    Returns:
        str: The file path for the neural network parameters.

    """
    nn_param_file_path: str
    if file_name is None:
        nn_param_file_path = os.path.join(folder_path, "param")
    else:
        nn_param_file_path = os.path.join(folder_path, file_name)
    return nn_param_file_path + ".pt", nn_param_file_path + ".yaml"


def get_nn_architecture_file_path_from(folder_path: MyPath) -> str:
    """Get the file path for the architecture parameters.

    Args:
        folder_path (str): The folder path for the architecture parameters.

    Returns:
        str: The file path for the architecture parameters.

    """
    nn_param_file_path: str = os.path.join(folder_path, "architecture.yaml")
    return nn_param_file_path


def create_nn(nn_type_args: NNModelTypeArgs) -> ChiNN:
    """Create a neural network."""
    net: ChiNN
    match nn_type_args:
        case MultiLayerPerceptronArgs():
            net = MultiLayerPerceptron(args=nn_type_args)
        case TransformerArgs():
            net = TransformerOne(args=nn_type_args)
        case _:
            sys.exit(f"Create NN: can not find {nn_type_args} in file {__name__}")
    return net


# TODO: probably dead code, check!
def get_architecture_args_from_file(
    architecture_file_name: MyPath,
) -> NeuralNetArchitectureArgs:
    """Get the architecture arguments from a YAML file.

    Args:
        architecture_file_name (path): The path to the architecture YAML file.

    Returns:
        NeuralNetArchitectureArgs: The architecture arguments.

    """
    args_dict: dict[Any, Any] = yaml_fetch_args_in_file(
        path_file=architecture_file_name
    )
    nn_architecture_args: NeuralNetArchitectureArgs = dacite.from_dict(
        data_class=NeuralNetArchitectureArgs,
        data=args_dict,
        config=dacite.Config(cast=[Enum]),
    )
    return nn_architecture_args


def get_architecture_args_from_folder(folder_path: MyPath) -> NeuralNetArchitectureArgs:
    """Get the architecture arguments from a folder.

    Args:
        folder_path (path): The path to the folder containing the architecture file.

    Returns:
        NeuralNetArchitectureArgs: The architecture arguments.

    """
    architecture_file_name: MyPath = get_nn_architecture_file_path_from(
        folder_path=folder_path
    )
    if not os.path.isfile(architecture_file_name):
        raise ArchitectureFileNotFoundError(str(architecture_file_name))

    nn_architecture_args: NeuralNetArchitectureArgs = get_architecture_args_from_file(
        architecture_file_name=architecture_file_name
    )

    return nn_architecture_args


def create_nn_from_param_path_and_architecture_args(
    model_weights_file_name: MyPath, nn_architecture_args: NeuralNetArchitectureArgs
) -> tuple[ChiNN, NeuralNetArchitectureArgs]:
    """Create a neural network from a parameter file and architecture arguments.

    Args:
        model_weights_file_name (path): The path to the model weights file.
        nn_architecture_args (NeuralNetArchitectureArgs): The architecture arguments.

    Returns:
        tuple[ChiNN, NeuralNetArchitectureArgs]: The created neural network and the architecture arguments.

    """
    net: ChiNN = create_nn(nn_type_args=nn_architecture_args.model_type_args)
    net.load_weights_from_file(path_to_param_file=model_weights_file_name)
    return net, nn_architecture_args


def create_nn_from_folder_path_and_existing_model(
    folder_path: MyPath,
) -> tuple[ChiNN, NeuralNetArchitectureArgs]:
    """Create a neural network from a folder path and existing model.

    Args:
        folder_path (path): The path to the folder containing the model files.

    Returns:
        tuple[ChiNN, NeuralNetArchitectureArgs]: _description_

    """
    nn_architecture_args: NeuralNetArchitectureArgs = get_architecture_args_from_folder(
        folder_path=folder_path
    )
    model_weights_file_name: MyPath = os.path.join(folder_path, "param.pt")

    return create_nn_from_param_path_and_architecture_args(
        model_weights_file_name=model_weights_file_name,
        nn_architecture_args=nn_architecture_args,
    )


def create_nn_content_eval_from_folder_path_and_existing_model[StateT: HasTurn](
    path_to_nn_folder: MyPath,
    content_to_input_convert: ContentToInputFunction[StateT],
) -> tuple[NNBWStateEvaluator[StateT], NeuralNetArchitectureArgs]:
    """Create a neural network content evaluator.

    Args:
        path_to_nn_folder (path): the path to the folder where the model is defined.
        content_to_input_convert (ContentToInputFunction): Converter from content to NN input.

    Returns:
        NNContentEvaluator: The created neural network content evaluator.

    """
    net: ChiNN
    nn_architecture_args: NeuralNetArchitectureArgs
    net, nn_architecture_args = create_nn_from_folder_path_and_existing_model(
        folder_path=path_to_nn_folder
    )

    nn_state_evaluator = create_nn_state_eval_from_nn_and_architecture_args(
        nn=net,
        nn_architecture_args=nn_architecture_args,
        content_to_input_convert=content_to_input_convert,
    )
    return nn_state_evaluator, nn_architecture_args


def create_nn_state_eval_from_nn_and_architecture_args[StateT: HasTurn](
    nn_architecture_args: NeuralNetArchitectureArgs,
    content_to_input_convert: ContentToInputFunction[StateT],
    nn: ChiNN,
) -> NNBWStateEvaluator[StateT]:
    """Create a neural network state evaluator from architecture arguments.

    Args:
        nn_architecture_args (NeuralNetArchitectureArgs): The architecture arguments.
        content_to_input_convert (ContentToInputFunction): The content to input converter.
        nn (ChiNN): The neural network.

    Returns:
        NNBWStateEvaluator: The created neural network state evaluator.

    """
    output_and_value_converter: TurnOutputValueConverter = create_output_converter(
        model_output_type=nn_architecture_args.model_output_type
    )

    return NNBWStateEvaluator(
        net=nn,
        output_and_value_converter=output_and_value_converter,
        content_to_input_convert=content_to_input_convert,
    )


def create_nn_state_eval_from_architecture_args[StateT: HasTurn](
    nn_architecture_args: NeuralNetArchitectureArgs,
    content_to_input_convert: ContentToInputFunction[StateT],
) -> NNBWStateEvaluator[StateT]:
    """Create a neural network state evaluator from architecture arguments.

    Args:
        nn_architecture_args (NeuralNetArchitectureArgs): The architecture arguments.
        content_to_input_convert (ContentToInputFunction): The content to input converter.

    Returns:
        NNBWStateEvaluator: The created neural network state evaluator.

    """
    nn = create_nn(nn_type_args=nn_architecture_args.model_type_args)
    nn.init_weights()

    return create_nn_state_eval_from_nn_and_architecture_args(
        nn_architecture_args=nn_architecture_args,
        nn=nn,
        content_to_input_convert=content_to_input_convert,
    )


def create_nn_state_eval_from_nn_parameters_file_and_existing_model[StateT: HasTurn](
    model_weights_file_name: MyPath,
    nn_architecture_args: NeuralNetArchitectureArgs,
    content_to_input_convert: ContentToInputFunction[StateT],
) -> NNBWStateEvaluator[StateT]:
    """Create a neural network state evaluator from a parameters file and architecture arguments.

    Args:
        model_weights_file_name (path): The path to the model weights file.
        nn_architecture_args (NeuralNetArchitectureArgs): The architecture arguments.
        content_to_input_convert (ContentToInputFunction): The content to input converter.

    Returns:
        NNBWStateEvaluator: The created neural network state evaluator.

    """
    net: ChiNN
    net, nn_architecture_args = create_nn_from_param_path_and_architecture_args(
        model_weights_file_name=model_weights_file_name,
        nn_architecture_args=nn_architecture_args,
    )

    return create_nn_state_eval_from_nn_and_architecture_args(
        nn_architecture_args=nn_architecture_args,
        nn=net,
        content_to_input_convert=content_to_input_convert,
    )
