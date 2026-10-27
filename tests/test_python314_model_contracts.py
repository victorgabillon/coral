"""Runtime annotation, model artifact and evaluator contracts on Python 3.14."""

from typing import TYPE_CHECKING, get_type_hints

import pytest
import torch
from valanga import Color

from coral.neural_networks import factory
from coral.neural_networks.input_converters.content_to_input import (
    ContentToInputFunction,
)
from coral.neural_networks.models.multi_layer_perceptron import (
    MultiLayerPerceptron,
    MultiLayerPerceptronArgs,
)
from coral.neural_networks.neural_net_architecture_args import NeuralNetArchitectureArgs
from coral.neural_networks.nn_model_type import ActivationFunctionType, NNModelType
from coral.neural_networks.nn_state_evaluator import NNBWStateEvaluator
from coral.neural_networks.output_converters.model_output_type import ModelOutputType
from coral.neural_networks.output_converters.output_value_converter import (
    IdentityConverter,
)

if TYPE_CHECKING:
    from pathlib import Path


def _model() -> MultiLayerPerceptron:
    return MultiLayerPerceptron(
        MultiLayerPerceptronArgs(
            type=NNModelType.MULTI_LAYER_PERCEPTRON,
            number_neurons_per_layer=[5, 8, 1],
            list_of_activation_functions=[
                ActivationFunctionType.PARAMETRIC_RELU,
                ActivationFunctionType.TANGENT_HYPERBOLIC,
            ],
        )
    )


def test_deferred_public_annotations_resolve_at_runtime() -> None:
    """Keep dacite and evaluator introspection working with native annotations."""
    assert get_type_hints(NeuralNetArchitectureArgs)["model_type_args"] is not None
    assert get_type_hints(ModelOutputType)["point_of_view"] is not None
    assert get_type_hints(ContentToInputFunction.__call__)["return"] is torch.Tensor
    for target in (
        factory.create_nn,
        factory.create_nn_from_param_path_and_architecture_args,
        factory.create_nn_from_folder_path_and_existing_model,
        factory.create_nn_content_eval_from_folder_path_and_existing_model,
        factory.create_nn_state_eval_from_nn_and_architecture_args,
        factory.create_nn_state_eval_from_architecture_args,
        factory.create_nn_state_eval_from_nn_parameters_file_and_existing_model,
        NNBWStateEvaluator,
        NNBWStateEvaluator.__init__,
        NNBWStateEvaluator.evaluate,
    ):
        assert get_type_hints(target)


@pytest.mark.parametrize("legacy", [False, True])
def test_existing_state_dictionary_and_torchscript_roundtrip(
    tmp_path: Path, legacy: bool
) -> None:
    """Load existing weights on CPU and retain the scripted MLP evaluation path."""
    torch.manual_seed(314)
    original = _model().eval()
    inputs = torch.randn(4, 5)
    weights = tmp_path / "weights.pt"
    torch.save(
        original.state_dict(), weights, _use_new_zipfile_serialization=not legacy
    )
    restored = _model().eval()
    restored.load_weights_from_file(weights)
    scripted = torch.jit.script(restored)
    with torch.no_grad():
        expected = original(inputs)
        torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
        torch.testing.assert_close(scripted(inputs), expected, rtol=0, atol=0)
    assert list(restored.state_dict()) == list(original.state_dict())


class _Position:
    turn = Color.WHITE


def test_evaluator_inference_retains_values_and_input_gradients() -> None:
    """Return the same scalar without changing the caller's autograd context."""
    model = _model()
    inputs = torch.randn(5, requires_grad=True)

    def convert(state: _Position) -> torch.Tensor:
        assert state.turn == Color.WHITE
        return inputs

    evaluator = NNBWStateEvaluator(model, IdentityConverter(), convert, script=False)
    with torch.no_grad():
        expected = model(inputs).item()
    assert evaluator.evaluate(_Position()).score == expected
    assert evaluator.evaluate_tensor(inputs, _Position()).score == expected
    assert torch.is_grad_enabled()
    assert inputs.requires_grad
    assert inputs.grad is None


def test_standard_module_pickling_omits_transient_compiler_state(
    tmp_path: Path,
) -> None:
    """Retain normal full-model serialization after an opt-in compile wrapper."""
    model = _model().eval()
    inputs = torch.randn(2, 5)
    with torch.no_grad():
        expected = model(inputs)
    keys = list(model.state_dict())
    model.compile(backend="eager")
    artifact = tmp_path / "compiled-model.pt"
    torch.save(model, artifact)
    restored = torch.load(artifact, weights_only=False, map_location="cpu")
    assert list(restored.state_dict()) == keys
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
