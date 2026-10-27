"""Module for the BoardToInput protocol and BoardToInputFunction protocol."""

from typing import Protocol, runtime_checkable

import torch
from valanga import HasTurn


@runtime_checkable
class ContentToInputFunction[StateT: HasTurn](Protocol):
    """Protocol for a callable object that converts a chess board to a tensor input for a neural network."""

    def __call__(self, state: StateT) -> torch.Tensor:
        """Convert the given chess board to a tensor input.

        Args:
            state (StateT): The state with turn information to convert.

        Returns:
            torch.Tensor: The tensor input representing the chess board.

        """
        ...


class ContentToInput[StateT: HasTurn](Protocol):
    """Protocol for converting a chess board to a tensor input for a neural network."""

    def convert(self, content_with_turn: StateT) -> torch.Tensor:
        """Convert the given chess board to a tensor input.

        Args:
            content_with_turn (StateT): The state with turn information to convert.

        Returns:
            torch.Tensor: The tensor input representing the chess board.

        """
        ...
