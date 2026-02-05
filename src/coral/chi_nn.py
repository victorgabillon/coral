"""Module for the ChiNN class."""

import sys
import traceback

import torch
from torch import nn

from coral.utils.logger import coral_logger
from coral.utils.small_tools import MyPath, resolve_package_path


class ChiNN(nn.Module):
    """The Generic Neural network class of chipiron."""

    def __init__(self) -> None:
        """Initialize an instance of the ChiNN class."""
        super().__init__()

    def __getstate__(self) -> dict[str, object]:
        """Get the state of the object for pickling.

        Returns:
            dict: The state dictionary of the object.

        """
        return self.__dict__.copy()

    def init_weights(self) -> None:
        """Initialize the weights of the model."""
        raise NotImplementedError("not implemented in base class")

    def load_weights_from_file(self, path_to_param_file: MyPath) -> None:
        """Load the neural network weights from a file.

        Args:
            path_to_param_file (MyPath): The path to the parameter file.

        """
        coral_logger.info("load_or_init_weights from %s", path_to_param_file)
        try:  # load
            resolved_path = resolve_package_path(str(path_to_param_file))
            with open(resolved_path, "rb") as file_nnr:
                coral_logger.info("loading the existing param file %s", resolved_path)
                if torch.cuda.is_available():
                    self.load_state_dict(torch.load(file_nnr))
                else:
                    self.load_state_dict(
                        torch.load(file_nnr, map_location=torch.device("cpu"))
                    )

        except OSError:  # init
            # Print the full traceback to stderr
            traceback.print_exc()

            resolved_path = resolve_package_path(str(path_to_param_file))
            coral_logger.error("no file %s at %s", path_to_param_file, resolved_path)
            sys.exit(
                f"Error: no NN weights file and no rights to create it for file {path_to_param_file}"
            )

    def log_readable_model_weights_to_file(self, file_path: str) -> None:
        """Log the model weights to a file in a human-readable format.

        Args:
            file_path (str): The path to the file where the weights will be logged.

        Raises:
            Exception: If the logging fails.

        """
        raise NotImplementedError("not implemented in base class")
