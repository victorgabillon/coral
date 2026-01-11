"""
Module for the BoardToInput protocol and BoardToInputFunction protocol.
"""

from typing import Any, Protocol, runtime_checkable

import torch
from valanga import State


@runtime_checkable
class ContentToInputFunction(Protocol):
    """
    Protocol for a callable object that converts a chess board to a tensor input for a neural network.
    """

    def __call__(self, state: State) -> Any:
        """
        Converts the given chess board to a tensor input.

        Args:
            state (State): The state with turn information to convert.

        Returns:
            torch.Tensor: The tensor input representing the chess board.
        """
        ...


class ContentToInput(Protocol):
    """
    Protocol for converting a chess board to a tensor input for a neural network.
    """

    def convert(self, state: State) -> torch.Tensor:
        """
        Converts the given chess board to a tensor input.

        Args:
            state (State): The state with turn information to convert.

        Returns:
            torch.Tensor: The tensor input representing the chess board.
        """
        ...
