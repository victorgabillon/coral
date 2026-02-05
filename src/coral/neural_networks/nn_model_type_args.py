"""Type aliases for neural network model argument variants."""

from coral.neural_networks.models.multi_layer_perceptron import (
    MultiLayerPerceptronArgs,
)
from coral.neural_networks.models.transformer_one import (
    TransformerArgs,
)

type NNModelTypeArgs = MultiLayerPerceptronArgs | TransformerArgs
