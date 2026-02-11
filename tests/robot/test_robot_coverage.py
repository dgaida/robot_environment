"""
Coverage-focused tests for Robot class
"""

import pytest
from unittest.mock import Mock, patch
from robot_environment.robot.robot import Robot
from robot_workspace import Location
from robot_workspace import PoseObjectPNP
from robot_workspace import Object
from robot_workspace import Objects


@pytest.fixture
def mock_environment():
    """Create mock environment"""
    env = Mock()
    env.verbose.return_value = False
    env.get_robot_controller.return_value = Mock()
    env.get_observation_pose.return_value = PoseObjectPNP(0.2, 0.0, 0.3)
    env.oralcom_call_text2speech_async.return_value = Mock()
    env.get_workspace_home_id.return_value = "niryo_ws_left"
    return env


@pytest.fixture
def mock_robot_controller():
    """Create mock robot controller"""
    with patch("robot_environment.robot.robot.NiryoRobotController") as mock:
        controller = Mock()
        controller.get_pose.return_value = PoseObjectPNP(0.2, 0.0, 0.3)
        controller.get_target_pose_from_rel.return_value = PoseObjectPNP(0.25, 0.05, 0.01)
        controller.move2observation_pose = Mock()
        controller.robot_pick_object.return_value = True
        controller.robot_place_object.return_value = True
        controller.robot_push_object.return_value = True
        controller.is_in_motion.return_value = False
        mock.return_value = controller
        yield mock


def create_mock_object(label="cube", x=0.25, y=0.05):
    """Helper to create a properly mocked object"""
    obj = Mock(spec=Object)
    obj.label.return_value = label
    obj.x_com.return_value = x
    obj.y_com.return_value = y
    obj.coordinate.return_value = [x, y]
    obj.width_m.return_value = 0.05
    obj.height_m.return_value = 0.05
    obj.pose_com.return_value = PoseObjectPNP(x, y, 0.01)
    obj.pose_center.return_value = PoseObjectPNP(x, y, 0.01)
    return obj


class TestRobotCoverage:
    """Additional tests to increase coverage of robot.py"""

    def test_pick_place_object_across_workspaces(self, mock_environment, mock_robot_controller):
        """Test picking from one workspace and placing in another"""
        robot = Robot(mock_environment, robot_id="niryo")

        # Setup mock objects for each workspace
        mock_pick_obj = create_mock_object("cube", 0.1, 0.1)
        mock_place_ref_obj = create_mock_object("box", 0.3, 0.3)

        # Mock environment methods for multi-workspace
        mock_environment.get_detected_objects_from_workspace.side_effect = [
            Objects([mock_pick_obj]),
            Objects([mock_place_ref_obj]),
        ]
        mock_environment.get_workspace_by_id.return_value = Mock()

        success = robot.pick_place_object_across_workspaces(
            object_name="cube",
            pick_workspace_id="ws_left",
            pick_coordinate=[0.1, 0.1],
            place_workspace_id="ws_right",
            place_coordinate=[0.3, 0.3],
            location=Location.RIGHT_NEXT_TO,
        )

        assert success is True
        assert mock_robot_controller.return_value.move2observation_pose.call_count == 2
        mock_environment.update_object_in_workspace.assert_called_once()

    def test_pick_object_from_workspace_not_found(self, mock_environment, mock_robot_controller):
        """Test pick_object_from_workspace when object is missing"""
        robot = Robot(mock_environment, robot_id="niryo")
        mock_environment.get_detected_objects_from_workspace.return_value = Objects()

        success = robot.pick_object_from_workspace("cube", "ws_test", [0.1, 0.1])

        assert success is False

    def test_place_object_in_workspace_no_workspace(self, mock_environment, mock_robot_controller):
        """Test place_object_in_workspace when workspace ID is invalid"""
        robot = Robot(mock_environment, robot_id="niryo")
        mock_environment.get_workspace_by_id.return_value = None

        success = robot.place_object_in_workspace("invalid_ws", [0.1, 0.1])

        assert success is False

    def test_push_object_directions(self, mock_environment, mock_robot_controller):
        """Test all directions for push_object to cover lines 391-413"""
        robot = Robot(mock_environment, robot_id="niryo")
        mock_obj = create_mock_object("large_box")

        with patch.object(robot, "get_detected_objects", return_value=Objects([mock_obj])):
            for direction in ["up", "down", "left", "right"]:
                success = robot.push_object("large_box", [0.25, 0.05], direction, 50.0)
                assert success is True

            # Test invalid direction
            success = robot.push_object("large_box", [0.25, 0.05], "invalid", 50.0)
            assert success is True  # Still returns True because it falls through to robot_push_object

    def test_get_nearest_object_in_workspace_no_coords(self, mock_environment, mock_robot_controller):
        """Test _get_nearest_object_in_workspace without target coordinates"""
        robot = Robot(mock_environment, robot_id="niryo")
        mock_obj = create_mock_object("cube")
        mock_environment.get_detected_objects_from_workspace.return_value = Objects([mock_obj])

        obj = robot._get_nearest_object_in_workspace("cube", "ws_test", [])

        assert obj == mock_obj

    def test_handle_object_detection(self, mock_environment, mock_robot_controller):
        """Test handle_object_detection to cover line 62"""
        robot = Robot(mock_environment, robot_id="niryo")
        mock_environment.get_workspace.return_value = Mock()

        objects_dict_list = [{"label": "cube", "x": 0.2, "y": 0.1, "width": 0.05, "height": 0.05, "yaw": 0.0}]

        with patch("robot_workspace.Objects.dict_list_to_objects") as mock_conv:
            mock_conv.return_value = [create_mock_object()]
            robot.handle_object_detection(objects_dict_list)
            mock_conv.assert_called_once()

    def test_calibrate(self, mock_environment, mock_robot_controller):
        """Test calibrate method"""
        robot = Robot(mock_environment, robot_id="niryo")
        mock_robot_controller.return_value.calibrate.return_value = True

        assert robot.calibrate() is True

    def test_widowx_initialization(self, mock_environment):
        """Test initialization with non-niryo robot_id"""
        robot = Robot(mock_environment, robot_id="widowx")
        assert robot._robot is None

    def test_pick_object_no_object_found(self, mock_environment, mock_robot_controller):
        """Test pick_object when no object is found at coordinates"""
        robot = Robot(mock_environment, robot_id="niryo")
        with patch.object(robot, "get_detected_objects", return_value=Objects()):
            success = robot.pick_object("pencil", [0.25, 0.05])
            assert success is False

    def test_place_object_no_picked_object(self, mock_environment, mock_robot_controller):
        """Test place_object when no object was previously picked"""
        robot = Robot(mock_environment, robot_id="niryo")
        robot._object_last_picked = None
        success = robot.place_object([0.3, 0.1], location=Location.NONE)
        assert success is True  # Still returns True if robot_place_object succeeds

    def test_place_object_in_workspace_with_reference(self, mock_environment, mock_robot_controller):
        """Test place_object_in_workspace with a reference object for relative placement"""
        robot = Robot(mock_environment, robot_id="niryo")
        mock_picked = create_mock_object("cube_picked")
        robot._object_last_picked = mock_picked

        mock_ref = create_mock_object("box_ref", 0.3, 0.3)
        mock_environment.get_workspace_by_id.return_value = Mock()
        mock_environment.get_detected_objects_from_workspace.return_value = Objects([mock_ref])

        # Test ON_TOP_OF
        success = robot.place_object_in_workspace("ws_test", [0.3, 0.3], location=Location.ON_TOP_OF)
        assert success is True

        # Test BELOW
        success = robot.place_object_in_workspace("ws_test", [0.3, 0.3], location=Location.BELOW)
        assert success is True

        # Test ABOVE
        success = robot.place_object_in_workspace("ws_test", [0.3, 0.3], location=Location.ABOVE)
        assert success is True

    def test_get_nearest_object_fallback_logic(self, mock_environment, mock_robot_controller):
        """Test the fallback logic in _get_nearest_object (lines 677-687)"""
        robot = Robot(mock_environment, robot_id="niryo")
        with patch.object(robot, "get_detected_objects", return_value=Objects()):
            # Mock get_most_similar_object if it existed, but it's commented out in code.
            # We want to hit the case where nearest_object_name is None
            obj = robot._get_nearest_object("pencil", [0.25, 0.05])
            assert obj is None

    def test_pick_place_across_workspaces_pick_fails(self, mock_environment, mock_robot_controller):
        """Test pick_place_object_across_workspaces when pick operation fails"""
        robot = Robot(mock_environment, robot_id="niryo")
        mock_environment.get_detected_objects_from_workspace.return_value = Objects()

        success = robot.pick_place_object_across_workspaces("cube", "ws_pick", [0.1, 0.1], "ws_place", [0.2, 0.2])
        assert success is False

    def test_get_nearest_object_empty_coords(self, mock_environment, mock_robot_controller):
        """Test _get_nearest_object with empty target coordinates (lines 638-639)"""
        robot = Robot(mock_environment, robot_id="niryo")
        mock_obj = create_mock_object("cube")
        with patch.object(robot, "get_detected_objects", return_value=Objects([mock_obj])):
            obj = robot._get_nearest_object("cube", [])
            assert obj == mock_obj
