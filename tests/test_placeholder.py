"""Basic tests for coral package."""


def test_board_evaluation_imports():
    """Verify board evaluation module can be imported."""
    from coral.board_evaluation import PointOfView

    assert PointOfView.WHITE == "white"
    assert PointOfView.BLACK == "black"
    assert PointOfView.PLAYER_TO_MOVE == "player_to_move"
    assert PointOfView.NOT_PLAYER_TO_MOVE == "not_player_to_move"


def test_point_of_view_enum():
    """Test PointOfView enum values."""
    from coral.board_evaluation import PointOfView

    # Check all enum values exist
    pov_values = [pov.value for pov in PointOfView]
    assert "white" in pov_values
    assert "black" in pov_values
    assert "player_to_move" in pov_values
    assert "not_player_to_move" in pov_values

    # Check enum has exactly 4 values
    assert len(pov_values) == 4


def test_neural_networks_module_imports():
    """Verify neural network modules can be imported."""
    from coral.neural_networks import factory
    from coral.neural_networks.NNModelType import NNModelType

    assert factory is not None
    assert NNModelType is not None
