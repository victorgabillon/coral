from typing import TypeAlias

from coral.neural_networks.models.multi_layer_perceptron import (
    MultiLayerPerceptronArgs,
)
from coral.neural_networks.models.transformer_one import (
    TransformerArgs,
)

NNModelTypeArgs: TypeAlias = MultiLayerPerceptronArgs | TransformerArgs
