"""
Unit tests for Robot API classes
"""

import pytest
from robot_environment.robot.robot_api import RobotAPI


class TestRobotAPI:
    """Test suite for RobotAPI abstract class"""

    def test_is_abstract(self):
        """Test that RobotAPI cannot be instantiated directly"""
        with pytest.raises(TypeError):
            RobotAPI()

    def test_has_required_methods(self):
        """Test that RobotAPI defines required abstract methods"""
        required_methods = ["pick_place_object", "pick_object", "place_object", "push_object", "move2observation_pose"]

        for method in required_methods:
            assert hasattr(RobotAPI, method)

    def test_pick_place_object_signature(self):
        """Test pick_place_object method signature"""
        import inspect

        sig = inspect.signature(RobotAPI.pick_place_object)
        params = list(sig.parameters.keys())

        assert "object_name" in params
        assert "pick_coordinate" in params
        assert "place_coordinate" in params
        assert "location" in params

    def test_pick_object_signature(self):
        """Test pick_object method signature"""
        import inspect

        sig = inspect.signature(RobotAPI.pick_object)
        params = list(sig.parameters.keys())

        assert "object_name" in params
        assert "pick_coordinate" in params

    def test_place_object_signature(self):
        """Test place_object method signature"""
        import inspect

        sig = inspect.signature(RobotAPI.place_object)
        params = list(sig.parameters.keys())

        assert "place_coordinate" in params
        assert "location" in params

    def test_push_object_signature(self):
        """Test push_object method signature"""
        import inspect

        sig = inspect.signature(RobotAPI.push_object)
        params = list(sig.parameters.keys())

        assert "object_name" in params
        assert "push_coordinate" in params
        assert "direction" in params
        assert "distance" in params

    def test_move2observation_pose_signature(self):
        """Test move2observation_pose method signature"""
        import inspect

        sig = inspect.signature(RobotAPI.move2observation_pose)
        params = list(sig.parameters.keys())

        assert "workspace_id" in params
