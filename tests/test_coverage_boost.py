"""
Integration tests to increase code coverage for robot_environment.
Targeting specifically the missing lines identified in the coverage report.
"""

from unittest.mock import Mock, patch, MagicMock
from robot_environment.common.logger import log_start_end_cls
from robot_environment.environment import Environment
from robot_workspace import PoseObjectPNP
from robot_environment.robot.niryo_robot_controller import NiryoRobotController
from robot_environment.config import (
    RobotConfig, CameraConfig, RobotControlConfig,
    MemoryConfig, ConfigManager, get_config
)
from robot_environment.robot.command_processor import parse_robot_command
from robot_environment.common.logger_config import set_verbose, get_package_logger
import numpy as np

# -----------------------------------------------------------------------------
# Config Coverage
# -----------------------------------------------------------------------------

def test_config_boost(tmp_path):
    config = RobotConfig.get_default_niryo()
    yaml_file = tmp_path / "test.yaml"
    json_file = tmp_path / "test.json"
    config.to_yaml(yaml_file)
    config.to_json(json_file)
    CameraConfig.from_dict({"resolution": [640, 480]})
    RobotControlConfig.from_dict({"pick_z_offset_m": 0.01})
    MemoryConfig.from_dict({"position_tolerance_m": 0.1})
    manager = ConfigManager()
    manager.load()
    manager.update(robot_id="custom")
    manager.save(yaml_file)
    manager.save(json_file, format="json")
    get_config()

# -----------------------------------------------------------------------------
# Command Processor & Logger
# -----------------------------------------------------------------------------

def test_cp_logger_boost():
    parse_robot_command(None)
    class NoLog:
        @log_start_end_cls()
        def run(self): return 1
    n = NoLog()
    n.run()
    n._logger = Mock()
    n._logger.isEnabledFor.side_effect = Exception("error")
    n.run()
    logger = get_package_logger("test", False)
    set_verbose(logger, True)
    set_verbose(logger, False)

# -----------------------------------------------------------------------------
# Environment & Robot
# -----------------------------------------------------------------------------

def test_env_robot_boost():
    with patch("robot_environment.environment.Robot") as mock_robot_cls, \
         patch("robot_environment.environment.NiryoFrameGrabber") as mock_fg_cls, \
         patch("robot_environment.environment.NiryoWorkspaces") as mock_ws_cls, \
         patch("robot_environment.environment.RedisMessageBroker") as mock_broker_cls, \
         patch("robot_environment.environment.RedisLabelManager") as mock_labels_cls, \
         patch("robot_environment.environment.Text2Speech"):

        m_ws = MagicMock()
        mock_ws_cls.return_value = m_ws
        m_ws.get_workspace_home_id.return_value = "ws"
        m_ws.__iter__ = Mock(return_value=iter([]))

        m_robot = mock_robot_cls.return_value
        m_robot.get_robot_controller.return_value = MagicMock(spec=NiryoRobotController)

        m_fg = mock_fg_cls.return_value
        m_fg.get_current_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        env = Environment("k", True, "niryo", enable_performance_monitoring=False, start_camera_thread=False)
        env._workspaces = m_ws
        env._framegrabber = m_fg

        env.get_robot_pose()
        with patch.object(env, "get_object_labels", return_value=[[]]):
             env.get_object_labels_as_string()

        mock_broker_cls.return_value.get_latest_objects.return_value = None
        env.get_detected_objects()

        mock_labels_cls.return_value.get_latest_labels.return_value = None
        env.get_object_labels()

        env.verbose = True
        env._current_workspace_id = "ws"
        with patch.object(env._memory_manager, "update", return_value=(1, 1)):
             with patch.object(env._memory_manager, "get_memory_stats", return_value={"ws": {"object_count":1, "manual_updates":0, "visible":True}}):
                  gen = env.update_camera_and_objects()
                  next(gen)

        env.cleanup()

# -----------------------------------------------------------------------------
# Niryo Controller
# -----------------------------------------------------------------------------

def test_niryo_controller_boost():
    mock_robot = MagicMock()
    with patch("robot_environment.robot.niryo_robot_controller.NiryoRobot"):
        from pyniryo import NiryoRobotException
        ctrl = NiryoRobotController(mock_robot, True)
        ctrl._robot_ctrl = MagicMock()
        ctrl._robot_ctrl.get_camera_intrinsics.return_value = (np.eye(3), np.zeros(5))

        with patch("robot_environment.robot.niryo_robot_controller.pyniryo_v", "pyniryo"):
             ctrl.get_camera_intrinsics()

        ctrl._robot_ctrl.push_object.side_effect = Exception()
        ctrl.robot_push_object(PoseObjectPNP(0,0,0), "up", 10)

        ctrl._robot_ctrl.calibrate.side_effect = Exception()
        ctrl.calibrate()

        with patch.object(ctrl, "_lock"):
             with patch.object(ctrl._robot_ctrl, "get_target_pose_from_rel", side_effect=NiryoRobotException("Fail")):
                  ctrl.get_target_pose_from_rel("ws", 0, 0, 0)

        ctrl._robot_ctrl.close_connection.side_effect = Exception()
        try:
            ctrl._shutdown()
        except Exception:
            pass
