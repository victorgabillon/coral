from dataclasses import dataclass

from coral.board_evaluation import (
    PointOfView,
)


# this type is about the type of representation input fed to NN models
@dataclass
class ModelOutputType:
    point_of_view: PointOfView
