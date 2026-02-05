"""Module for the BoardToInput protocol and BoardToInputFunction protocol."""

from typing import Any, Protocol, TypeVar, runtime_checkable

import torch
from valanga import HasTurn

StateT_contra = TypeVar("StateT_contra", bound=HasTurn, contravariant=True)


@runtime_checkable
class ContentToInputFunction[StateT_contra](Protocol):
    """Protocol for a callable object that converts a chess board to a tensor input for a neural network."""

    def __call__(self, state: StateT_contra) -> Any:
        """Converts the given chess board to a tensor input.

        Args:
            state (State): The state with turn information to convert.

        Returns:
            torch.Tensor: The tensor input representing the chess board.

        """
        ...


class ContentToInput[StateT_contra](Protocol):
    """Protocol for converting a chess board to a tensor input for a neural network."""

    def convert(self, content_with_turn: StateT_contra) -> torch.Tensor:
        """Converts the given chess board to a tensor input.

        Args:
            state (State): The state with turn information to convert.

        Returns:
            torch.Tensor: The tensor input representing the chess board.

        """
        ...
