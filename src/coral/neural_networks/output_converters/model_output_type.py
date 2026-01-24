"""Data structures describing neural network output types."""

from dataclasses import dataclass

from coral.board_evaluation import (
    PointOfView,
)


# this type is about the type of representation input fed to NN models
@dataclass
class ModelOutputType:
    """Represents the output type and point of view for NN models."""

    point_of_view: PointOfView
