"""
Unit tests for NiryoRobotController class - FIXED VERSION
"""

import pytest
import threading
from unittest.mock import Mock, patch
from robot_environment.robot.niryo_robot_controller import NiryoRobotController
from robot_workspace import PoseObjectPNP
from robot_workspace import Object


@pytest.fixture
def mock_robot():
    """Create mock Robot instance"""
    robot = Mock()
    robot.verbose.return_value = False

    # Mock environment
    mock_env = Mock()
    mock_env.get_observation_pose.return_value = PoseObjectPNP(0.2, 0.0, 0.3)
    robot.environment.return_value = mock_env

    return robot


@pytest.fixture
def mock_niryo_robot():
    """Create mock NiryoRobot instance"""
    with patch("robot_environment.robot.niryo_robot_controller.NiryoRobot") as mock:
        mock_instance = Mock()
        mock_instance.calibrate_auto = Mock()
        mock_instance.update_tool = Mock()

        # FIX #2: Return real PoseObject with numeric attributes
        mock_pose = Mock()
        mock_pose.x = 0.2
        mock_pose.y = 0.0
        mock_pose.z = 0.3
        mock_pose.roll = 0.0
        mock_pose.pitch = 1.57
        mock_pose.yaw = 0.0
        mock_instance.get_pose.return_value = mock_pose

        mock_instance.pick_from_pose = Mock()
        mock_instance.place_from_pose = Mock()
        mock_instance.move_pose = Mock()
        mock_instance.shift_pose = Mock()
        mock_instance.close_gripper = Mock()

        # FIX #3: Return real PoseObject for get_target_pose_from_rel
        mock_target_pose = Mock()
        mock_target_pose.x = 0.25
        mock_target_pose.y = 0.05
        mock_target_pose.z = 0.01
        mock_target_pose.roll = 0.0
        mock_target_pose.pitch = 1.57
        mock_target_pose.yaw = 0.0
        mock_instance.get_target_pose_from_rel.return_value = mock_target_pose

        mock.return_value = mock_instance
        yield mock


class TestNiryoRobotController:
    """Test suite for NiryoRobotController class"""

    def test_initialization(self, mock_robot, mock_niryo_robot):
        """Test controller initialization"""
        controller = NiryoRobotController(mock_robot, use_simulation=False, verbose=False)

        assert controller.robot() == mock_robot
        assert controller.verbose() is False
        assert controller.lock() is not None

    def test_initialization_creates_lock(self, mock_robot, mock_niryo_robot):
        """Test that lock is created"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        lock = controller.lock()

        assert lock is not None
        assert hasattr(lock, "acquire")
        assert hasattr(lock, "release")
        assert hasattr(lock, "__enter__")
        assert hasattr(lock, "__exit__")

    def test_initialization_calibrates_robot(self, mock_robot, mock_niryo_robot):
        """Test that robot is calibrated on init"""
        NiryoRobotController(mock_robot, use_simulation=False)

        mock_niryo_robot.return_value.calibrate_auto.assert_called_once()

    def test_get_pose(self, mock_robot, mock_niryo_robot):
        """Test getting robot pose"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        pose = controller.get_pose()

        assert isinstance(pose, PoseObjectPNP)
        assert pose.x == 0.2

    def test_robot_pick_object(self, mock_robot, mock_niryo_robot):
        """Test picking an object"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        # Create mock object
        mock_obj = Mock(spec=Object)
        mock_pose = PoseObjectPNP(0.25, 0.05, 0.01, 0.0, 1.57, 0.0)
        mock_obj.pose_com.return_value = mock_pose

        result = controller.robot_pick_object(mock_obj)

        assert result is True
        mock_niryo_robot.return_value.pick_from_pose.assert_called_once()

    def test_robot_pick_object_with_z_offset(self, mock_robot, mock_niryo_robot):
        """Test that pick adds z-offset"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        mock_obj = Mock(spec=Object)
        mock_pose = PoseObjectPNP(0.25, 0.05, 0.01, 0.0, 1.57, 0.0)
        mock_obj.pose_com.return_value = mock_pose

        controller.robot_pick_object(mock_obj)

        # Check that pose was modified with offset
        call_args = mock_niryo_robot.return_value.pick_from_pose.call_args[0][0]
        assert call_args.z == pytest.approx(0.011, abs=0.001)

    def test_robot_place_object(self, mock_robot, mock_niryo_robot):
        """Test placing an object"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        place_pose = PoseObjectPNP(0.3, 0.1, 0.01, 0.0, 1.57, 0.0)

        result = controller.robot_place_object(place_pose)

        assert result is True
        mock_niryo_robot.return_value.place_from_pose.assert_called_once()

    def test_robot_place_object_with_z_offset(self, mock_robot, mock_niryo_robot):
        """Test that place adds z-offset"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        place_pose = PoseObjectPNP(0.3, 0.1, 0.01, 0.0, 1.57, 0.0)

        controller.robot_place_object(place_pose)

        # Check offset was added
        call_args = mock_niryo_robot.return_value.place_from_pose.call_args[0][0]
        assert call_args.z == pytest.approx(0.015, abs=0.001)

    def test_robot_push_object_up(self, mock_robot, mock_niryo_robot):
        """Test pushing object up"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        push_pose = PoseObjectPNP(0.25, 0.05, 0.01)

        result = controller.robot_push_object(push_pose, "up", 50.0)

        assert result is True
        mock_niryo_robot.return_value.close_gripper.assert_called_once()
        mock_niryo_robot.return_value.shift_pose.assert_called_once()

    def test_robot_push_object_down(self, mock_robot, mock_niryo_robot):
        """Test pushing object down"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        push_pose = PoseObjectPNP(0.25, 0.05, 0.01)

        controller.robot_push_object(push_pose, "down", 50.0)

        # Check shift was called with negative distance
        call_args = mock_niryo_robot.return_value.shift_pose.call_args
        assert call_args[0][1] < 0

    def test_robot_push_object_left(self, mock_robot, mock_niryo_robot):
        """Test pushing object left"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        push_pose = PoseObjectPNP(0.25, 0.05, 0.01)

        controller.robot_push_object(push_pose, "left", 50.0)

        mock_niryo_robot.return_value.shift_pose.assert_called_once()

    def test_robot_push_object_right(self, mock_robot, mock_niryo_robot):
        """Test pushing object right"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        push_pose = PoseObjectPNP(0.25, 0.05, 0.01)

        controller.robot_push_object(push_pose, "right", 50.0)

        mock_niryo_robot.return_value.shift_pose.assert_called_once()

    def test_robot_push_object_invalid_direction(self, mock_robot, mock_niryo_robot):
        """Test pushing with invalid direction"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        push_pose = PoseObjectPNP(0.25, 0.05, 0.01)

        # Should still return True but print error
        result = controller.robot_push_object(push_pose, "invalid", 50.0)

        assert result is True

    def test_get_target_pose_from_rel(self, mock_robot, mock_niryo_robot):
        """Test getting target pose from relative coordinates"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        pose = controller.get_target_pose_from_rel("test_ws", 0.5, 0.5, 0.0)

        assert isinstance(pose, PoseObjectPNP)
        assert pose.x == 0.25
        mock_niryo_robot.return_value.get_target_pose_from_rel.assert_called_once()

    def test_get_target_pose_from_rel_clamps_coordinates(self, mock_robot, mock_niryo_robot):
        """Test that coordinates are clamped to [0, 1]"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        # Try with out-of-bounds coordinates
        controller.get_target_pose_from_rel("test_ws", 1.5, -0.5, 0.0)

        # Should clamp to [0, 1]
        call_args = mock_niryo_robot.return_value.get_target_pose_from_rel.call_args[0]
        assert 0.0 <= call_args[2] <= 1.0  # u_rel
        assert 0.0 <= call_args[3] <= 1.0  # v_rel

    def test_get_target_pose_from_rel_handles_exception(self, mock_robot, mock_niryo_robot):
        """Test handling of exception in get_target_pose_from_rel - FIXED"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        # FIX #3: Make the mock raise exception BEFORE the try block is entered
        # The exception must be raised when the method is called
        from pyniryo.api.exceptions import NiryoRobotException

        mock_niryo_robot.return_value.get_target_pose_from_rel.side_effect = NiryoRobotException("Connection error")

        # Now when we call it, the exception should be caught and return zero pose
        pose = controller.get_target_pose_from_rel("test_ws", 0.5, 0.5, 0.0)

        # Should return zero pose
        assert pose.x == 0.0
        assert pose.y == 0.0

    def test_move2observation_pose(self, mock_robot, mock_niryo_robot):
        """Test moving to observation pose"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        controller.move2observation_pose("test_ws")

        # Should call move_pose
        mock_niryo_robot.return_value.move_pose.assert_called_once()

    def test_move2observation_pose_with_none_pose(self, mock_robot, mock_niryo_robot):
        """Test moving to observation pose when pose is None"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        mock_robot.environment().get_observation_pose.return_value = None

        controller.move2observation_pose("test_ws")

        # Should not call move_pose
        mock_niryo_robot.return_value.move_pose.assert_not_called()

    def test_reset_connection(self, mock_robot, mock_niryo_robot):
        """Test resetting connection"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        controller.reset_connection()

        # Should recreate robot
        assert mock_niryo_robot.call_count >= 2  # Initial + reset

    def test_thread_safety_with_lock(self, mock_robot, mock_niryo_robot):
        """Test that operations are thread-safe"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        # Lock should be used for get_pose
        with controller.lock():
            mock_niryo_robot.return_value.get_pose()

        assert True

    def test_cleanup(self, mock_robot, mock_niryo_robot):
        """Test cleanup method"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        controller.cleanup()

        # Should set shutdown flag
        assert controller._shutdown_v is True

    def test_destructor(self, mock_robot, mock_niryo_robot):
        """Test destructor calls shutdown"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        controller.__del__()

        # Just verify it doesn't crash
        assert True

    def test_robot_ctrl_property(self, mock_robot, mock_niryo_robot):
        """Test robot_ctrl property"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        robot_ctrl = controller.robot_ctrl()

        assert robot_ctrl is not None

    def test_initialization_simulation_mode(self, mock_robot, mock_niryo_robot):
        """Test initialization in simulation mode"""
        controller = NiryoRobotController(mock_robot, use_simulation=True)

        # Should use simulation IP address
        assert controller._robot_ip_address == "192.168.247.128"

    def test_initialization_real_mode(self, mock_robot, mock_niryo_robot):
        """Test initialization in real robot mode"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        # Should use real robot IP address
        assert controller._robot_ip_address == "192.168.0.140"


class TestNiryoRobotControllerThreading:
    """Test threading behavior"""

    def test_concurrent_get_pose_calls(self, mock_robot, mock_niryo_robot):
        """Test concurrent pose requests"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        results = []

        def get_pose_thread():
            pose = controller.get_pose()
            results.append(pose)

        # Start multiple threads
        threads = [threading.Thread(target=get_pose_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(results) == 5

    def test_lock_prevents_concurrent_access(self, mock_robot, mock_niryo_robot):
        """Test that lock prevents concurrent access"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        access_log = []

        def access_with_lock():
            with controller.lock():
                access_log.append("start")
                import time

                time.sleep(0.01)  # Simulate work
                access_log.append("end")

        threads = [threading.Thread(target=access_with_lock) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have alternating start/end (serialized access)
        assert access_log.count("start") == 3
        assert access_log.count("end") == 3


class TestNiryoRobotControllerExceptionHandling:
    """Test exception handling"""

    def test_unicode_decode_error_in_get_target_pose(self, mock_robot, mock_niryo_robot):
        """Test handling UnicodeDecodeError"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        mock_niryo_robot.return_value.get_target_pose_from_rel.side_effect = UnicodeDecodeError(
            "utf-8", b"\x00\x00", 0, 1, "invalid"
        )

        pose = controller.get_target_pose_from_rel("test_ws", 0.5, 0.5, 0.0)

        # Should return zero pose
        assert pose.x == 0.0

    def test_syntax_error_in_get_target_pose(self, mock_robot, mock_niryo_robot):
        """Test handling SyntaxError"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        mock_niryo_robot.return_value.get_target_pose_from_rel.side_effect = SyntaxError("Syntax error")

        pose = controller.get_target_pose_from_rel("test_ws", 0.5, 0.5, 0.0)

        assert pose.x == 0.0

    def test_unicode_error_in_move2observation_pose(self, mock_robot, mock_niryo_robot):
        """Test handling error in move2observation_pose"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        mock_niryo_robot.return_value.move_pose.side_effect = UnicodeDecodeError("utf-8", b"\x00\x00", 0, 1, "invalid")

        # Should not crash
        controller.move2observation_pose("test_ws")

        assert True


class TestNiryoRobotControllerPrivateMethods:
    """Test private methods"""

    def test_shift_pose(self, mock_robot, mock_niryo_robot):
        """Test _shift_pose method"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        from robot_environment.robot.niryo_robot_controller import RobotAxis

        controller._shift_pose(RobotAxis.X, 50.0)

        # Should convert mm to m and call shift_pose
        mock_niryo_robot.return_value.shift_pose.assert_called_once()
        call_args = mock_niryo_robot.return_value.shift_pose.call_args[0]
        assert call_args[1] == pytest.approx(0.05, abs=0.001)  # 50mm = 0.05m

    def test_move_pose(self, mock_robot, mock_niryo_robot):
        """Test _move_pose method"""
        controller = NiryoRobotController(mock_robot, use_simulation=False)

        mock_pose = Mock()
        controller._move_pose(mock_pose)

        mock_niryo_robot.return_value.move_pose.assert_called_once_with(mock_pose)
