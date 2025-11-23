"""
Unit tests for Robot class
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
        mock.return_value = controller
        yield mock


class TestRobot:
    """Test suite for Robot class"""

    def test_initialization(self, mock_environment, mock_robot_controller):
        """Test robot initialization"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        assert robot.environment() == mock_environment
        assert robot.verbose() is False

    def test_initialization_creates_controller(self, mock_environment, mock_robot_controller):
        """Test that robot controller is created"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        assert robot.robot() is not None
        mock_robot_controller.assert_called_once()

    def test_get_pose(self, mock_environment, mock_robot_controller):
        """Test getting robot pose"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        pose = robot.get_pose()

        assert isinstance(pose, PoseObjectPNP)

    def test_get_target_pose_from_rel(self, mock_environment, mock_robot_controller):
        """Test getting target pose from relative coordinates"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        pose = robot.get_target_pose_from_rel("test_ws", 0.5, 0.5, 0.0)

        assert isinstance(pose, PoseObjectPNP)

    def test_move2observation_pose(self, mock_environment, mock_robot_controller):
        """Test moving to observation pose"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        robot.move2observation_pose("test_ws")

        mock_robot_controller.return_value.move2observation_pose.assert_called_once_with("test_ws")

    def test_pick_object(self, mock_environment, mock_robot_controller):
        """Test picking an object"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        # Mock detected objects
        mock_obj = Mock(spec=Object)
        mock_obj.label.return_value = "pencil"
        mock_obj.x_com.return_value = 0.25
        mock_obj.y_com.return_value = 0.05
        mock_obj.coordinate.return_value = [0.25, 0.05]

        with patch.object(robot, "get_detected_objects", return_value=Objects([mock_obj])):
            success = robot.pick_object("pencil", [0.25, 0.05])

        assert success is True
        mock_robot_controller.return_value.robot_pick_object.assert_called_once()

    def test_pick_object_not_found(self, mock_environment, mock_robot_controller):
        """Test picking object that doesn't exist"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        with patch.object(robot, "get_detected_objects", return_value=Objects()):
            success = robot.pick_object("nonexistent", [0.25, 0.05])

        assert success is False

    def test_place_object(self, mock_environment, mock_robot_controller):
        """Test placing an object"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        # Set up a picked object
        mock_obj = Mock(spec=Object)
        mock_obj.label.return_value = "pencil"
        mock_obj.width_m.return_value = 0.02
        mock_obj.height_m.return_value = 0.15
        robot._object_last_picked = mock_obj

        success = robot.place_object([0.3, 0.1], location=Location.NONE)

        assert success is True
        mock_robot_controller.return_value.robot_place_object.assert_called_once()

    def test_place_object_right_next_to(self, mock_environment, mock_robot_controller):
        """Test placing object right next to another"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        # Mock picked object
        mock_picked = Mock(spec=Object)
        mock_picked.label.return_value = "pencil"
        mock_picked.width_m.return_value = 0.02
        mock_picked.height_m.return_value = 0.15
        robot._object_last_picked = mock_picked

        # Mock reference object
        mock_ref_obj = Mock(spec=Object)
        mock_ref_obj.width_m.return_value = 0.05
        mock_ref_obj.height_m.return_value = 0.05
        mock_ref_obj.pose_center.return_value = PoseObjectPNP(0.3, 0.1, 0.01)

        with patch.object(robot, "get_detected_objects", return_value=Objects([mock_ref_obj])):
            success = robot.place_object([0.3, 0.1], location=Location.RIGHT_NEXT_TO)

        assert success is True

    def test_place_object_left_next_to(self, mock_environment, mock_robot_controller):
        """Test placing object left next to another"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        mock_picked = Mock(spec=Object)
        mock_picked.label.return_value = "pencil"
        mock_picked.width_m.return_value = 0.02
        mock_picked.height_m.return_value = 0.15
        robot._object_last_picked = mock_picked

        mock_ref_obj = Mock(spec=Object)
        mock_ref_obj.width_m.return_value = 0.05
        mock_ref_obj.height_m.return_value = 0.05
        mock_ref_obj.pose_center.return_value = PoseObjectPNP(0.3, 0.1, 0.01)

        with patch.object(robot, "get_detected_objects", return_value=Objects([mock_ref_obj])):
            success = robot.place_object([0.3, 0.1], location=Location.LEFT_NEXT_TO)

        assert success is True

    def test_place_object_above(self, mock_environment, mock_robot_controller):
        """Test placing object above another"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        mock_picked = Mock(spec=Object)
        mock_picked.label.return_value = "pencil"
        mock_picked.width_m.return_value = 0.02
        mock_picked.height_m.return_value = 0.15
        robot._object_last_picked = mock_picked

        mock_ref_obj = Mock(spec=Object)
        mock_ref_obj.width_m.return_value = 0.05
        mock_ref_obj.height_m.return_value = 0.05
        mock_ref_obj.pose_center.return_value = PoseObjectPNP(0.3, 0.1, 0.01)

        with patch.object(robot, "get_detected_objects", return_value=Objects([mock_ref_obj])):
            success = robot.place_object([0.3, 0.1], location=Location.ABOVE)

        assert success is True

    def test_place_object_below(self, mock_environment, mock_robot_controller):
        """Test placing object below another"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        mock_picked = Mock(spec=Object)
        mock_picked.label.return_value = "pencil"
        mock_picked.width_m.return_value = 0.02
        mock_picked.height_m.return_value = 0.15
        robot._object_last_picked = mock_picked

        mock_ref_obj = Mock(spec=Object)
        mock_ref_obj.width_m.return_value = 0.05
        mock_ref_obj.height_m.return_value = 0.05
        mock_ref_obj.pose_center.return_value = PoseObjectPNP(0.3, 0.1, 0.01)

        with patch.object(robot, "get_detected_objects", return_value=Objects([mock_ref_obj])):
            success = robot.place_object([0.3, 0.1], location=Location.BELOW)

        assert success is True

    def test_place_object_on_top_of(self, mock_environment, mock_robot_controller):
        """Test placing object on top of another"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        mock_picked = Mock(spec=Object)
        mock_picked.label.return_value = "pencil"
        mock_picked.width_m.return_value = 0.02
        mock_picked.height_m.return_value = 0.15
        robot._object_last_picked = mock_picked

        mock_ref_obj = Mock(spec=Object)
        mock_ref_obj.width_m.return_value = 0.05
        mock_ref_obj.height_m.return_value = 0.05
        mock_ref_obj.pose_center.return_value = PoseObjectPNP(0.3, 0.1, 0.01)

        with patch.object(robot, "get_detected_objects", return_value=Objects([mock_ref_obj])):
            success = robot.place_object([0.3, 0.1], location=Location.ON_TOP_OF)

        assert success is True

    def test_place_object_inside(self, mock_environment, mock_robot_controller):
        """Test placing object inside another"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        mock_picked = Mock(spec=Object)
        mock_picked.label.return_value = "pencil"
        mock_picked.width_m.return_value = 0.02
        mock_picked.height_m.return_value = 0.15
        robot._object_last_picked = mock_picked

        mock_ref_obj = Mock(spec=Object)
        mock_ref_obj.width_m.return_value = 0.05
        mock_ref_obj.height_m.return_value = 0.05
        mock_ref_obj.pose_center.return_value = PoseObjectPNP(0.3, 0.1, 0.01)

        with patch.object(robot, "get_detected_objects", return_value=Objects([mock_ref_obj])):
            success = robot.place_object([0.3, 0.1], location=Location.INSIDE)

        assert success is True

    def test_push_object(self, mock_environment, mock_robot_controller):
        """Test pushing an object"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        mock_obj = Mock(spec=Object)
        mock_obj.label.return_value = "cube"
        mock_obj.pose_com.return_value = PoseObjectPNP(0.25, 0.05, 0.01)
        mock_obj.width_m.return_value = 0.05
        mock_obj.height_m.return_value = 0.05

        with patch.object(robot, "get_detected_objects", return_value=Objects([mock_obj])):
            success = robot.push_object("cube", [0.25, 0.05], "right", 50.0)

        assert success is True
        mock_robot_controller.return_value.robot_push_object.assert_called_once()

    def test_push_object_different_directions(self, mock_environment, mock_robot_controller):
        """Test pushing in different directions"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        mock_obj = Mock(spec=Object)
        mock_obj.label.return_value = "cube"
        mock_obj.pose_com.return_value = PoseObjectPNP(0.25, 0.05, 0.01)
        mock_obj.width_m.return_value = 0.05
        mock_obj.height_m.return_value = 0.05

        directions = ["up", "down", "left", "right"]

        for direction in directions:
            with patch.object(robot, "get_detected_objects", return_value=Objects([mock_obj])):
                success = robot.push_object("cube", [0.25, 0.05], direction, 50.0)
                assert success is True

    def test_pick_place_object(self, mock_environment, mock_robot_controller):
        """Test complete pick and place operation"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        # Mock picked object
        mock_pick_obj = Mock(spec=Object)
        mock_pick_obj.label.return_value = "pencil"
        mock_pick_obj.x_com.return_value = 0.15
        mock_pick_obj.y_com.return_value = -0.05
        mock_pick_obj.coordinate.return_value = [0.15, -0.05]
        mock_pick_obj.width_m.return_value = 0.02
        mock_pick_obj.height_m.return_value = 0.15
        mock_pick_obj.pose_com.return_value = PoseObjectPNP(0.15, -0.05, 0.01)

        # Mock place reference object
        mock_place_obj = Mock(spec=Object)
        mock_place_obj.width_m.return_value = 0.05
        mock_place_obj.height_m.return_value = 0.05
        mock_place_obj.pose_center.return_value = PoseObjectPNP(0.3, 0.1, 0.01)

        with patch.object(robot, "get_detected_objects") as mock_get:
            # First call returns pick object, second call returns place object
            mock_get.side_effect = [Objects([mock_pick_obj]), Objects([mock_place_obj])]

            success = robot.pick_place_object("pencil", [0.15, -0.05], [0.3, 0.1], location=Location.RIGHT_NEXT_TO)

        assert success is True

    def test_pick_place_object_pick_fails(self, mock_environment, mock_robot_controller):
        """Test pick_place when pick fails"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        with patch.object(robot, "get_detected_objects", return_value=Objects()):
            success = robot.pick_place_object("nonexistent", [0.15, -0.05], [0.3, 0.1])

        assert success is False

    def test_robot_in_motion_property(self, mock_environment, mock_robot_controller):
        """Test robot_in_motion property"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        assert robot.robot_in_motion() is False

    def test_environment_property(self, mock_environment, mock_robot_controller):
        """Test environment property"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        assert robot.environment() == mock_environment

    def test_robot_property(self, mock_environment, mock_robot_controller):
        """Test robot property"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        assert robot.robot() is not None

    def test_verbose_property(self, mock_environment, mock_robot_controller):
        """Test verbose property"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo", verbose=True)

        assert robot.verbose() is True


class TestRobotCommandParsing:
    """Test command parsing for safe execution"""

    def test_parse_command_pick_place(self, mock_environment, mock_robot_controller):
        """Test parsing pick_place_object command"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        command = 'robot.pick_place_object(object_name="pencil", pick_coordinate=[0.1, 0.0], place_coordinate=[0.2, 0.0], location="right next to")'

        target, method, pos_args, kw_args = robot._parse_command(command)

        assert target == "robot"
        assert method == "pick_place_object"
        assert kw_args["object_name"] == "pencil"
        assert kw_args["location"] == Location.RIGHT_NEXT_TO

    def test_parse_command_pick(self, mock_environment, mock_robot_controller):
        """Test parsing pick_object command"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        command = 'robot.pick_object("pencil", [0.1, 0.0])'

        target, method, pos_args, kw_args = robot._parse_command(command)

        assert target == "robot"
        assert method == "pick_object"
        assert len(pos_args) == 2

    def test_parse_command_place(self, mock_environment, mock_robot_controller):
        """Test parsing place_object command"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        command = 'robot.place_object([0.2, 0.0], location="above")'

        target, method, pos_args, kw_args = robot._parse_command(command)

        assert target == "robot"
        assert method == "place_object"
        assert kw_args["location"] == Location.ABOVE

    def test_parse_command_invalid(self, mock_environment, mock_robot_controller):
        """Test parsing invalid command"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        command = "invalid command"

        target, method, pos_args, kw_args = robot._parse_command(command)

        assert target is None
        assert method is None

    def test_execute_command_pick_place(self, mock_environment, mock_robot_controller):
        """Test executing pick_place command"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        mock_obj = Mock(spec=Object)
        mock_obj.label.return_value = "pencil"
        mock_obj.coordinate.return_value = [0.15, -0.05]
        mock_obj.width_m.return_value = 0.02
        mock_obj.height_m.return_value = 0.15
        mock_obj.pose_com.return_value = PoseObjectPNP(0.15, -0.05, 0.01)

        with patch.object(robot, "get_detected_objects", return_value=Objects([mock_obj])):
            success = robot._execute_command(
                "robot",
                "pick_place_object",
                [],
                {
                    "object_name": "pencil",
                    "pick_coordinate": [0.15, -0.05],
                    "place_coordinate": [0.3, 0.1],
                    "location": Location.NONE,
                },
            )

        assert success is True

    def test_execute_command_unknown_method(self, mock_environment, mock_robot_controller):
        """Test executing unknown method"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        success = robot._execute_command("robot", "unknown_method", [], {})

        assert success is False

    def test_execute_python_code_safe(self, mock_environment, mock_robot_controller):
        """Test safe Python code execution"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        code = 'robot.move2observation_pose("test_ws")'

        result, success = robot.execute_python_code_safe(code)

        assert success is True


class TestRobotGetNearestObject:
    """Test _get_nearest_object private method"""

    def test_get_nearest_object_found(self, mock_environment, mock_robot_controller):
        """Test getting nearest object when it exists"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        mock_obj = Mock(spec=Object)
        mock_obj.label.return_value = "pencil"
        mock_obj.x_com.return_value = 0.25
        mock_obj.y_com.return_value = 0.05

        with patch.object(robot, "get_detected_objects", return_value=Objects([mock_obj])):
            obj = robot._get_nearest_object("pencil", [0.26, 0.06])

        assert obj is not None

    def test_get_nearest_object_not_found(self, mock_environment, mock_robot_controller):
        """Test getting nearest object when it doesn't exist"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        with patch.object(robot, "get_detected_objects", return_value=Objects()):
            obj = robot._get_nearest_object("nonexistent", [0.25, 0.05])

        assert obj is None

    def test_get_nearest_object_no_target_coords(self, mock_environment, mock_robot_controller):
        """Test getting object without target coordinates"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        mock_obj = Mock(spec=Object)
        mock_obj.label.return_value = "pencil"

        with patch.object(robot, "get_detected_objects", return_value=Objects([mock_obj])):
            obj = robot._get_nearest_object("pencil", [])

        assert obj is not None


class TestRobotStringConversions:
    """Test string conversion for Location enum"""

    def test_location_string_conversion(self, mock_environment, mock_robot_controller):
        """Test that string locations are converted to enum"""
        robot = Robot(mock_environment, use_simulation=False, robot_id="niryo")

        mock_picked = Mock(spec=Object)
        mock_picked.label.return_value = "pencil"
        mock_picked.width_m.return_value = 0.02
        mock_picked.height_m.return_value = 0.15
        robot._object_last_picked = mock_picked

        # Test with string location
        success = robot.place_object([0.3, 0.1], location="right next to")

        # Should still work (converted to enum internally)
        assert success is True
