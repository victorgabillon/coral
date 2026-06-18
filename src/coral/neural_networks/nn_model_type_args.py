"""Type aliases for neural network model argument variants."""

from coral.neural_networks.models.entity_token_transformer_value_net import (
    EntityTokenTransformerValueNetArgs,
)
from coral.neural_networks.models.multi_layer_perceptron import (
    MultiLayerPerceptronArgs,
)
from coral.neural_networks.models.transformer_one import (
    TransformerArgs,
)

NNModelTypeArgs = (
    EntityTokenTransformerValueNetArgs | MultiLayerPerceptronArgs | TransformerArgs
)
