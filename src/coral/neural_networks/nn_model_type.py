"""Defines the neural network model and activation function types."""

from collections.abc import Callable
from enum import StrEnum

from torch import nn


class NNModelType(StrEnum):
    """Defines the types of neural network models."""

    MULTI_LAYER_PERCEPTRON = "multi_layer_perceptron"
    TRANSFORMER = "transformer"


class ActivationFunctionType(StrEnum):
    """Defines the types of activation functions."""

    TANGENT_HYPERBOLIC = "hyperbolic_tangent"
    PARAMETRIC_RELU = "parametric_relu"
    RELU = "relu"


activation_map: dict[ActivationFunctionType, Callable[[], nn.Module]] = {
    ActivationFunctionType.RELU: nn.ReLU,
    ActivationFunctionType.TANGENT_HYPERBOLIC: nn.Tanh,
    ActivationFunctionType.PARAMETRIC_RELU: nn.PReLU,
}
