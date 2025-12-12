"""
Unit tests for Environment class - FIXED VERSION

Key fixes:
1. Removed VisualCortex patching (not imported in environment.py anymore)
2. Fixed ObjectMemoryManager usage
3. Fixed Redis-based object detection
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, create_autospec
from robot_environment.environment import Environment
from robot_workspace import Objects, PoseObjectPNP
from tests.conftest import create_mock_workspace


@pytest.fixture
def mock_dependencies():
    """Mock all Environment dependencies - FIXED"""
    from robot_environment.robot.niryo_robot_controller import NiryoRobotController

    with patch("robot_environment.environment.Robot") as mock_robot, patch(
        "robot_environment.environment.NiryoFrameGrabber"
    ) as mock_fg, patch("robot_environment.environment.NiryoWorkspaces") as mock_ws, patch(
        "robot_environment.environment.Text2Speech"
    ) as mock_tts, patch(
        "robot_environment.environment.RedisMessageBroker"
    ) as mock_broker, patch(
        "robot_environment.environment.RedisLabelManager"
    ) as mock_labels:

        # Setup robot controller with proper isinstance support
        mock_robot_ctrl = create_autospec(NiryoRobotController, instance=True)
        mock_robot_instance = Mock()
        mock_robot_instance.get_robot_controller.return_value = mock_robot_ctrl
        mock_robot_instance.robot.return_value = mock_robot_ctrl
        mock_robot_instance.robot_in_motion.return_value = False
        mock_robot_instance.get_pose.return_value = PoseObjectPNP(0.2, 0.0, 0.3)
        mock_robot_instance.move2observation_pose = Mock()
        mock_robot.return_value = mock_robot_instance

        # Setup framegrabber
        mock_fg_instance = Mock()
        mock_fg_instance.get_current_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_fg_instance.get_current_frame_width_height.return_value = (480, 640)
        mock_fg.return_value = mock_fg_instance

        # Setup workspaces with complete interface
        ws1 = create_mock_workspace("niryo_ws_left")
        ws2 = create_mock_workspace("niryo_ws_right")

        mock_ws_instance = Mock()
        mock_ws_instance.__iter__ = Mock(return_value=iter([ws1, ws2]))

        mock_workspace = Mock()
        mock_workspace.id.return_value = "test_ws"
        mock_workspace.img_shape.return_value = (640, 480, 3)
        mock_workspace.set_img_shape = Mock()

        # Helper to create proper PoseObjectPNP-like mocks
        def create_mock_pose(x, y, z=0.0):
            mock_pose = Mock()
            mock_pose.xy_coordinate.return_value = [x, y]
            mock_pose.x = x
            mock_pose.y = y
            mock_pose.z = z
            return mock_pose

        mock_workspace.xy_ul_wc.return_value = create_mock_pose(0.4, 0.15, 0.0)
        mock_workspace.xy_ur_wc.return_value = create_mock_pose(0.4, -0.15, 0.0)
        mock_workspace.xy_ll_wc.return_value = create_mock_pose(0.1, 0.15, 0.0)
        mock_workspace.xy_lr_wc.return_value = create_mock_pose(0.1, -0.15, 0.0)
        mock_workspace.xy_center_wc.return_value = create_mock_pose(0.25, 0.0, 0.0)

        def mock_transform(ws_id, u_rel, v_rel, yaw=0.0):
            x = 0.4 - u_rel * 0.3
            y = 0.15 - v_rel * 0.3
            return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

        mock_workspace.transform_camera2world_coords = mock_transform

        mock_ws_instance.get_workspace = Mock(return_value=mock_workspace)
        mock_ws_instance.get_workspace_by_id = Mock(return_value=mock_workspace)
        mock_ws_instance.get_workspace_home_id.return_value = "test_ws"
        mock_ws_instance.get_workspace_id = Mock(return_value="test_ws")
        mock_ws_instance.get_observation_pose.return_value = PoseObjectPNP(0.2, 0.0, 0.3)
        mock_ws_instance.get_visible_workspace.return_value = mock_workspace
        mock_ws_instance.get_home_workspace.return_value = mock_workspace
        mock_ws.return_value = mock_ws_instance

        # Setup TTS
        mock_tts_instance = Mock()
        mock_tts_instance.call_text2speech_async.return_value = Mock()
        mock_tts.return_value = mock_tts_instance

        # Setup Redis broker for object detection
        mock_broker_instance = Mock()
        mock_broker_instance.get_latest_objects.return_value = []
        mock_broker_instance.test_connection.return_value = True
        mock_broker.return_value = mock_broker_instance

        # Setup Redis label manager
        mock_labels_instance = Mock()
        mock_labels_instance.get_latest_labels.return_value = ["pencil", "pen", "eraser"]
        mock_labels_instance.add_label.return_value = True
        mock_labels.return_value = mock_labels_instance

        yield {
            "robot": mock_robot,
            "framegrabber": mock_fg,
            "workspaces": mock_ws,
            "tts": mock_tts,
            "broker": mock_broker,
            "labels": mock_labels,
        }


class TestEnvironment:
    """Test suite for Environment class"""

    def test_initialization_basic(self, mock_dependencies):
        """Test basic initialization"""
        env = Environment(
            el_api_key="test_key", use_simulation=False, robot_id="niryo", verbose=False, start_camera_thread=False
        )

        assert env.use_simulation() is False
        assert env.verbose() is False

    def test_initialization_with_simulation(self, mock_dependencies):
        """Test initialization with simulation"""
        env = Environment(el_api_key="test_key", use_simulation=True, robot_id="niryo", start_camera_thread=False)

        assert env.use_simulation() is True

    def test_initialization_creates_all_components(self, mock_dependencies):
        """Test that all components are created"""
        env = Environment(el_api_key="test_key", use_simulation=False, robot_id="niryo", start_camera_thread=False)

        assert env.robot() is not None
        assert env.framegrabber() is not None
        assert env.workspaces() is not None

    def test_initialization_with_camera_thread(self, mock_dependencies):
        """Test initialization with camera thread enabled"""
        with patch.object(Environment, "start_camera_updates") as mock_start:
            Environment(el_api_key="test_key", use_simulation=False, robot_id="niryo", start_camera_thread=True)

            mock_start.assert_called_once_with(visualize=False)

    def test_initialization_without_camera_thread(self, mock_dependencies):
        """Test initialization without camera thread"""
        with patch.object(Environment, "start_camera_updates") as mock_start:
            Environment(el_api_key="test_key", use_simulation=False, robot_id="niryo", start_camera_thread=False)

            mock_start.assert_not_called()

    def test_get_current_frame(self, mock_dependencies):
        """Test getting current frame"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        frame = env.get_current_frame()

        assert frame is not None
        assert isinstance(frame, np.ndarray)

    def test_get_workspace(self, mock_dependencies):
        """Test getting workspace by index"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        workspace = env.get_workspace(0)

        assert workspace is not None

    def test_get_workspace_by_id(self, mock_dependencies):
        """Test getting workspace by ID"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        workspace = env.get_workspace_by_id("test_ws")

        assert workspace is not None

    def test_get_workspace_home_id(self, mock_dependencies):
        """Test getting home workspace ID"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        ws_id = env.get_workspace_home_id()

        assert ws_id == "test_ws"

    def test_get_object_labels_as_string(self, mock_dependencies):
        """Test getting object labels as string"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        labels_str = env.get_object_labels_as_string()

        assert "pencil" in labels_str
        assert "recognize" in labels_str.lower()

    def test_add_object_name2object_labels(self, mock_dependencies):
        """Test adding object label"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        result = env.add_object_name2object_labels("cup")

        assert "cup" in result
        assert "Added" in result

    def test_oralcom_call_text2speech_async(self, mock_dependencies):
        """Test asynchronous text-to-speech"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        mock_thread = Mock()
        mock_dependencies["tts"].return_value.call_text2speech_async.return_value = mock_thread

        thread = env.oralcom_call_text2speech_async("Test message")

        assert thread is not None

    def test_get_robot_controller(self, mock_dependencies):
        """Test getting robot controller"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        controller = env.get_robot_controller()

        assert controller is not None

    def test_get_robot_pose(self, mock_dependencies):
        """Test getting robot pose"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        pose = env.get_robot_pose()

        assert pose is not None
        assert isinstance(pose, PoseObjectPNP)

    def test_robot_move2observation_pose(self, mock_dependencies):
        """Test moving robot to observation pose"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.robot_move2observation_pose("test_ws")

        env.robot().move2observation_pose.assert_called_once_with("test_ws")

    def test_get_detected_objects_from_redis(self, mock_dependencies):
        """Test getting detected objects from Redis"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Mock Redis returning object data
        mock_dependencies["broker"].return_value.get_latest_objects.return_value = [
            {
                "label": "pencil",
                "x_com": 0.25,
                "y_com": 0.05,
                "width_m": 0.02,
                "height_m": 0.15,
                "bbox": {"x_min": 100, "y_min": 100, "x_max": 150, "y_max": 280},
                "has_mask": False,
            }
        ]

        objects = env.get_detected_objects()

        assert isinstance(objects, Objects)

    def test_clear_memory(self, mock_dependencies):
        """Test clearing object memory"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.clear_memory()

        # Should clear memory manager
        memory = env.get_detected_objects_from_memory()
        assert len(memory) == 0

    def test_get_detected_objects_from_memory(self, mock_dependencies):
        """Test getting objects from memory"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        memory = env.get_detected_objects_from_memory()

        assert isinstance(memory, Objects)

    def test_cleanup(self, mock_dependencies):
        """Test cleanup method"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.cleanup()

        assert env._stop_event.is_set()

    def test_stop_camera_updates(self, mock_dependencies):
        """Test stopping camera updates"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.stop_camera_updates()

        assert env._stop_event.is_set()


class TestEnvironmentMemoryManagement:
    """Test memory management with ObjectMemoryManager"""

    def test_remove_object_from_memory(self, mock_dependencies):
        """Test removing object from memory"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Set current workspace
        env._current_workspace_id = "test_ws"

        # This should not crash even if object doesn't exist
        env.remove_object_from_memory("pencil", [0.25, 0.05])

        assert True

    def test_update_object_in_memory(self, mock_dependencies):
        """Test updating object position in memory"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env._current_workspace_id = "test_ws"
        new_pose = PoseObjectPNP(0.30, 0.10, 0.01)

        # Should not crash
        env.update_object_in_memory("pencil", [0.25, 0.05], new_pose)

        assert True

    def test_get_current_workspace_id(self, mock_dependencies):
        """Test getting current workspace ID"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Should be set during initialization
        ws_id = env.get_current_workspace_id()

        assert ws_id is not None

    def test_set_current_workspace(self, mock_dependencies):
        """Test setting current workspace"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.set_current_workspace("new_ws")

        assert env.get_current_workspace_id() == "new_ws"


class TestEnvironmentWorkspaceOperations:
    """Test workspace-related operations"""

    def test_get_workspace_coordinate_from_point_upper_left(self, mock_dependencies):
        """Test getting upper left corner coordinate"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        coord = env.get_workspace_coordinate_from_point("test_ws", "upper left corner")

        assert coord is not None
        assert len(coord) == 2

    def test_get_workspace_coordinate_from_point_center(self, mock_dependencies):
        """Test getting center point coordinate"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        coord = env.get_workspace_coordinate_from_point("test_ws", "center point")

        assert coord is not None
        assert len(coord) == 2

    def test_get_workspace_coordinate_from_point_invalid(self, mock_dependencies):
        """Test getting invalid point returns None"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        coord = env.get_workspace_coordinate_from_point("test_ws", "invalid point")

        assert coord is None

    def test_is_any_workspace_visible(self, mock_dependencies):
        """Test checking workspace visibility"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        result = env.is_any_workspace_visible()

        assert isinstance(result, bool)


class TestEnvironmentProperties:
    """Test Environment properties"""

    def test_workspaces_property(self, mock_dependencies):
        """Test workspaces property"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        assert env.workspaces() is not None

    def test_framegrabber_property(self, mock_dependencies):
        """Test framegrabber property"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        assert env.framegrabber() is not None

    def test_robot_property(self, mock_dependencies):
        """Test robot property"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        assert env.robot() is not None

    def test_use_simulation_property(self, mock_dependencies):
        """Test use_simulation property"""
        env = Environment("key", True, "niryo", start_camera_thread=False)

        assert env.use_simulation() is True

    def test_verbose_property(self, mock_dependencies):
        """Test verbose property"""
        env = Environment("key", False, "niryo", verbose=True, start_camera_thread=False)

        assert env.verbose is True
