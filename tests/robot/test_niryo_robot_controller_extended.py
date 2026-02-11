"""
Extended unit tests for NiryoRobotController
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from robot_environment.robot.niryo_robot_controller import NiryoRobotController
from robot_workspace import PoseObjectPNP
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError


@pytest.fixture
def mock_robot():
    """Mock Robot instance"""
    robot = Mock()
    robot.environment.return_value = Mock()
    return robot


class TestNiryoRobotControllerExtended:
    """Test suite for NiryoRobotController coverage"""

    @patch("robot_environment.robot.niryo_robot_controller.NiryoRobot")
    def test_get_camera_intrinsics_pyniryo2(self, mock_niryo, mock_robot):
        """Test get_camera_intrinsics with pyniryo2 branch (line 92)"""
        with patch("robot_environment.robot.niryo_robot_controller.pyniryo_v", "pyniryo2"):
            # Mock _init_robot to avoid connection
            with patch.object(NiryoRobotController, "_init_robot"):
                controller = NiryoRobotController(mock_robot, False)
                controller._robot_ctrl = MagicMock()
                controller._robot_ctrl.vision.get_camera_intrinsics.return_value = ("mtx", "dist")

                mtx, dist = controller.get_camera_intrinsics()
                assert mtx == "mtx"
                assert dist == "dist"

    @patch("robot_environment.robot.niryo_robot_controller.NiryoRobot")
    def test_get_img_compressed_pyniryo2(self, mock_niryo, mock_robot):
        """Test get_img_compressed with pyniryo2 branch (lines 100-106)"""
        with patch("robot_environment.robot.niryo_robot_controller.pyniryo_v", "pyniryo2"):
            with patch.object(NiryoRobotController, "_init_robot"):
                controller = NiryoRobotController(mock_robot, False)
                controller._robot_ctrl = MagicMock()
                controller._robot_ctrl.vision.get_img_compressed.return_value = np.zeros(10)

                img = controller.get_img_compressed()
                assert len(img) == 10

    @patch("robot_environment.robot.niryo_robot_controller.NiryoRobot")
    def test_reset_connection(self, mock_niryo, mock_robot):
        """Test reset_connection coverage (lines 126-149)"""
        with patch.object(NiryoRobotController, "_init_robot"):
            controller = NiryoRobotController(mock_robot, False)
            controller._robot_ctrl = MagicMock()

            with patch.object(controller, "_shutdown") as mock_shut, \
                 patch.object(controller, "_create_robot") as mock_create:
                controller.reset_connection()
                mock_shut.assert_called_once()
                mock_create.assert_called_once()

    @patch("robot_environment.robot.niryo_robot_controller.NiryoRobot")
    def test_reset_connection_error(self, mock_niryo, mock_robot):
        """Test reset_connection error handling"""
        with patch.object(NiryoRobotController, "_init_robot"):
            controller = NiryoRobotController(mock_robot, False)
            controller._robot_ctrl = MagicMock()

            with patch.object(controller, "_shutdown", side_effect=Exception("Shut error")):
                controller.reset_connection() # Should log error but continue

    @patch("robot_environment.robot.niryo_robot_controller.NiryoRobot")
    def test_robot_pick_object_error(self, mock_niryo, mock_robot):
        """Test robot_pick_object error handling (lines 178-180)"""
        with patch.object(NiryoRobotController, "_init_robot"):
            controller = NiryoRobotController(mock_robot, False)
            controller._robot_ctrl = MagicMock()
            controller._robot_ctrl.pick_from_pose.side_effect = Exception("Pick error")

            success = controller.robot_pick_object(PoseObjectPNP(0, 0, 0))
            assert success is False

    @patch("robot_environment.robot.niryo_robot_controller.NiryoRobot")
    def test_robot_place_object_error(self, mock_niryo, mock_robot):
        """Test robot_place_object error handling (lines 211-213)"""
        with patch.object(NiryoRobotController, "_init_robot"):
            controller = NiryoRobotController(mock_robot, False)
            controller._robot_ctrl = MagicMock()
            controller._robot_ctrl.place_from_pose.side_effect = Exception("Place error")

            success = controller.robot_place_object(PoseObjectPNP(0, 0, 0))
            assert success is False

    @patch("robot_environment.robot.niryo_robot_controller.NiryoRobot")
    def test_robot_push_object_pyniryo2(self, mock_niryo, mock_robot):
        """Test robot_push_object with pyniryo2 branch (line 238)"""
        with patch("robot_environment.robot.niryo_robot_controller.pyniryo_v", "pyniryo2"):
            with patch.object(NiryoRobotController, "_init_robot"):
                controller = NiryoRobotController(mock_robot, False)
                controller._robot_ctrl = MagicMock()

                success = controller.robot_push_object(PoseObjectPNP(0,0,0), "up", 10.0)
                assert success is True
                controller._robot_ctrl.tool.close_gripper.assert_called_once()

    @patch("robot_environment.robot.niryo_robot_controller.NiryoRobot")
    def test_get_target_pose_from_rel_error(self, mock_niryo, mock_robot):
        """Test get_target_pose_from_rel error handling (lines 335-341)"""
        from pyniryo.api.exceptions import NiryoRobotException
        with patch.object(NiryoRobotController, "_init_robot"):
            controller = NiryoRobotController(mock_robot, False)
            controller._robot_ctrl = MagicMock()
            controller._robot_ctrl.get_target_pose_from_rel.side_effect = NiryoRobotException("Vision error")

            pose = controller.get_target_pose_from_rel("ws", 0.5, 0.5, 0.0)
            assert pose.x == 0.0
            assert pose.y == 0.0

    @patch("robot_environment.robot.niryo_robot_controller.NiryoRobot")
    def test_get_target_pose_from_rel_timeout_success(self, mock_niryo, mock_robot):
        """Test get_target_pose_from_rel_timeout success path (lines 311-341)"""
        with patch.object(NiryoRobotController, "_init_robot"):
            controller = NiryoRobotController(mock_robot, False)
            controller._robot_ctrl = MagicMock()
            mock_niryo_pose = MagicMock()
            mock_niryo_pose.x = 0.1
            mock_niryo_pose.y = 0.2
            mock_niryo_pose.z = 0.3
            mock_niryo_pose.roll = 0
            mock_niryo_pose.pitch = 0
            mock_niryo_pose.yaw = 0

            controller._robot_ctrl.get_target_pose_from_rel.return_value = mock_niryo_pose

            pose = controller.get_target_pose_from_rel_timeout("ws", 0.5, 0.5, 0.0)
            assert pose.x == 0.1

    @patch("robot_environment.robot.niryo_robot_controller.NiryoRobot")
    def test_get_target_pose_from_rel_timeout_lock_fail(self, mock_niryo, mock_robot):
        """Test get_target_pose_from_rel_timeout when lock fails"""
        with patch.object(NiryoRobotController, "_init_robot"):
            controller = NiryoRobotController(mock_robot, False)
            controller._lock = MagicMock()
            controller._lock.acquire.return_value = False

            pose = controller.get_target_pose_from_rel_timeout("ws", 0.5, 0.5, 0.0)
            assert pose.x == 0.0

    @patch("robot_environment.robot.niryo_robot_controller.NiryoRobot")
    def test_get_target_pose_from_rel_timeout_expired(self, mock_niryo, mock_robot):
        """Test get_target_pose_from_rel_timeout when future times out (line 377)"""
        with patch.object(NiryoRobotController, "_init_robot"):
            controller = NiryoRobotController(mock_robot, False)
            controller._robot_ctrl = MagicMock()

            # We need to mock the executor to return a future that times out
            mock_future = MagicMock(spec=Future)
            mock_future.result.side_effect = FuturesTimeoutError()
            controller._executor = MagicMock()
            controller._executor.submit.return_value = mock_future

            pose = controller.get_target_pose_from_rel_timeout("ws", 0.5, 0.5, 0.0)
            assert pose.x == 0.0
            mock_future.cancel.assert_called_once()

    @patch("robot_environment.robot.niryo_robot_controller.NiryoRobot")
    def test_init_robot_simulation(self, mock_niryo, mock_robot):
        """Test _init_robot with simulation=True (lines 421-422)"""
        with patch.object(NiryoRobotController, "_create_robot"):
            controller = NiryoRobotController(mock_robot, True)
            assert controller._robot_ip_address == "192.168.247.128"

    @patch("robot_environment.robot.niryo_robot_controller.NiryoRobot")
    def test_move2observation_pose_unicode_error(self, mock_niryo, mock_robot):
        """Test move2observation_pose with UnicodeDecodeError (line 402)"""
        with patch.object(NiryoRobotController, "_init_robot"):
            controller = NiryoRobotController(mock_robot, False)
            controller._robot_ctrl = MagicMock()
            mock_robot.environment().get_observation_pose.return_value = PoseObjectPNP(0.2, 0, 0.3)

            with patch.object(controller, "_move_pose", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "error")):
                controller.move2observation_pose("ws")
                # Should not crash
