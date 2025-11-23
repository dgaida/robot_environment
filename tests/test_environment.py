"""
Unit tests for Environment class
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from robot_environment.environment import Environment
from robot_environment.objects.objects import Objects
from robot_environment.objects.object import Object


@pytest.fixture
def mock_dependencies():
    """Mock all Environment dependencies"""
    with patch("robot_environment.environment.Robot") as mock_robot, patch(
        "robot_environment.environment.NiryoFrameGrabber"
    ) as mock_fg, patch("robot_environment.environment.NiryoWorkspaces") as mock_ws, patch(
        "robot_environment.environment.Text2Speech"
    ) as mock_tts, patch(
        "robot_environment.environment.VisualCortex"
    ) as mock_vc, patch(
        "robot_environment.environment.get_default_config"
    ) as mock_config:

        # Setup robot mock
        mock_robot_instance = Mock()
        mock_robot_instance.robot.return_value = Mock()
        mock_robot_instance.robot_in_motion.return_value = False
        mock_robot_instance.get_pose.return_value = Mock()
        mock_robot.return_value = mock_robot_instance

        # Setup framegrabber mock
        mock_fg_instance = Mock()
        mock_fg_instance.get_current_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_fg.return_value = mock_fg_instance

        # Setup workspaces mock
        mock_ws_instance = Mock()
        mock_workspace = Mock()
        mock_workspace.id.return_value = "test_ws"
        mock_ws_instance.get_workspace.return_value = mock_workspace
        mock_ws_instance.get_workspace_by_id.return_value = mock_workspace
        mock_ws_instance.get_workspace_home_id.return_value = "test_ws"
        mock_ws.return_value = mock_ws_instance

        # Setup TTS mock
        mock_tts_instance = Mock()
        mock_tts.return_value = mock_tts_instance

        # Setup VisualCortex mock
        mock_vc_instance = Mock()
        mock_vc_instance.get_detected_objects.return_value = []
        mock_vc_instance.get_object_labels.return_value = [["pencil", "pen", "eraser"]]
        mock_vc_instance.add_object_name2object_labels = Mock()
        mock_vc_instance.get_annotated_image.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_vc.return_value = mock_vc_instance

        mock_config.return_value = {}

        yield {
            "robot": mock_robot,
            "framegrabber": mock_fg,
            "workspaces": mock_ws,
            "tts": mock_tts,
            "visual_cortex": mock_vc,
            "config": mock_config,
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

            mock_start.assert_called_once_with(visualize=True)

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

        mock_thread = Mock()
        mock_dependencies["tts"].return_value.call_text2speech_async.return_value = mock_thread

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

    def test_get_robot_in_motion(self, mock_dependencies):
        """Test checking if robot is in motion"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        in_motion = env.get_robot_in_motion()

        assert isinstance(in_motion, bool)

    def test_robot_move2observation_pose(self, mock_dependencies):
        """Test moving robot to observation pose"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.robot_move2observation_pose("test_ws")

        env.robot().move2observation_pose.assert_called_once_with("test_ws")

    def test_get_detected_objects(self, mock_dependencies):
        """Test getting detected objects"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Mock detected objects
        Mock()  # mock_workspace =
        mock_dependencies["visual_cortex"].return_value.get_detected_objects.return_value = [
            {"label": "pencil", "bbox": {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200}, "has_mask": False}
        ]

        objects = env.get_detected_objects()

        assert isinstance(objects, Objects)

    def test_get_workspace_coordinate_from_point_upper_left(self, mock_dependencies):
        """Test getting upper left corner coordinate"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Mock workspace corners
        from robot_environment.objects.pose_object import PoseObjectPNP

        mock_ws = env.get_workspace_by_id("test_ws")
        mock_ws.xy_ul_wc.return_value = PoseObjectPNP(0.3, 0.15, 0.0)

        coord = env.get_workspace_coordinate_from_point("test_ws", "upper left corner")

        assert coord is not None
        assert len(coord) == 2

    def test_get_workspace_coordinate_from_point_center(self, mock_dependencies):
        """Test getting center point coordinate"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        from robot_environment.objects.pose_object import PoseObjectPNP

        mock_ws = env.get_workspace_by_id("test_ws")
        mock_ws.xy_center_wc.return_value = PoseObjectPNP(0.2, 0.0, 0.0)

        coord = env.get_workspace_coordinate_from_point("test_ws", "center point")

        assert coord is not None

    def test_get_workspace_coordinate_from_point_invalid(self, mock_dependencies):
        """Test getting invalid point returns None"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        coord = env.get_workspace_coordinate_from_point("test_ws", "invalid point")

        assert coord is None

    def test_is_any_workspace_visible_true(self, mock_dependencies):
        """Test when workspace is visible"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        mock_dependencies["workspaces"].return_value.get_visible_workspace.return_value = Mock()

        result = env.is_any_workspace_visible()

        assert result is True

    def test_is_any_workspace_visible_false(self, mock_dependencies):
        """Test when no workspace is visible"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        mock_dependencies["workspaces"].return_value.get_visible_workspace.return_value = None

        result = env.is_any_workspace_visible()

        assert result is False

    def test_get_observation_pose(self, mock_dependencies):
        """Test getting observation pose"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        from robot_environment.objects.pose_object import PoseObjectPNP

        mock_pose = PoseObjectPNP(0.2, 0.0, 0.3)
        mock_dependencies["workspaces"].return_value.get_observation_pose.return_value = mock_pose

        pose = env.get_observation_pose("test_ws")

        assert pose == mock_pose

    def test_stop_camera_updates(self, mock_dependencies):
        """Test stopping camera updates"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.stop_camera_updates()

        assert env._stop_event.is_set()

    def test_cleanup(self, mock_dependencies):
        """Test cleanup method"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.cleanup()

        assert env._stop_event.is_set()

    def test_destructor(self, mock_dependencies):
        """Test destructor sets stop event"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.__del__()

        assert env._stop_event.is_set()

    def test_get_current_frame_width_height(self, mock_dependencies):
        """Test getting frame dimensions"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        mock_dependencies["framegrabber"].return_value.get_current_frame_width_height.return_value = (480, 640)

        width, height = env.get_current_frame_width_height()

        assert width == 480
        assert height == 640

    def test_get_robot_target_pose_from_rel(self, mock_dependencies):
        """Test getting target pose from relative coordinates"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        from robot_environment.objects.pose_object import PoseObjectPNP

        mock_pose = PoseObjectPNP(0.25, 0.05, 0.01)
        mock_dependencies["robot"].return_value.get_target_pose_from_rel.return_value = mock_pose

        pose = env.get_robot_target_pose_from_rel("test_ws", 0.5, 0.5, 0.0)

        assert pose == mock_pose

    def test_get_visible_workspace(self, mock_dependencies):
        """Test getting visible workspace"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        from robot_environment.objects.pose_object import PoseObjectPNP

        camera_pose = PoseObjectPNP(0.2, 0.0, 0.3)
        mock_ws = Mock()
        mock_dependencies["workspaces"].return_value.get_visible_workspace.return_value = mock_ws

        workspace = env.get_visible_workspace(camera_pose)

        assert workspace == mock_ws


class TestEnvironmentCameraUpdates:
    """Test camera update functionality"""

    def test_check_new_detections_adds_new_object(self, mock_dependencies):
        """Test that new objects are added to memory"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Create mock object
        mock_obj = Mock(spec=Object)
        mock_obj.label.return_value = "pencil"
        mock_obj.x_com.return_value = 0.25
        mock_obj.y_com.return_value = 0.05
        # Make xy_com() return a tuple
        mock_obj.xy_com.return_value = (0.25, 0.05)

        detected = Objects([mock_obj])

        env._check_new_detections(detected)

        assert len(env._obj_position_memory) == 1

    def test_check_new_detections_ignores_duplicate(self, mock_dependencies):
        """Test that duplicate objects are not added"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        Mock()  # mock_workspace =
        mock_obj = Mock(spec=Object)
        mock_obj.label.return_value = "pencil"
        mock_obj.x_com.return_value = 0.25
        mock_obj.y_com.return_value = 0.05
        mock_obj.xy_com.return_value = (0.25, 0.05)

        detected = Objects([mock_obj])

        # Add twice
        env._check_new_detections(detected)
        env._check_new_detections(detected)

        # Should only have one
        assert len(env._obj_position_memory) == 1

    def test_get_detected_objects_from_memory(self, mock_dependencies):
        """Test getting objects from memory"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        mock_obj = Mock(spec=Object)
        env._obj_position_memory.append(mock_obj)

        objects = env.get_detected_objects_from_memory()

        assert len(objects) == 1


class TestEnvironmentLargestFreeSpace:
    """Test largest free space calculation"""

    @patch("robot_environment.environment.cv2")
    def test_get_largest_free_space_with_center(self, mock_cv2, mock_dependencies):
        """Test calculating largest free space"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Setup workspace
        from robot_environment.objects.pose_object import PoseObjectPNP

        mock_ws = Mock()
        mock_ws.xy_ul_wc.return_value = PoseObjectPNP(0.4, 0.15, 0.0)
        mock_ws.xy_lr_wc.return_value = PoseObjectPNP(0.1, -0.15, 0.0)

        # Access the workspaces properly through the mock
        mock_dependencies["workspaces"].return_value.get_workspace.return_value = mock_ws

        # Mock detected objects
        env._obj_position_memory = Objects()

        area, cx, cy = env.get_largest_free_space_with_center()

        assert area >= 0
        assert isinstance(cx, float)
        assert isinstance(cy, float)

    @patch("robot_environment.environment.cv2")
    def test_get_largest_free_space_with_objects(self, mock_cv2, mock_dependencies):
        """Test free space calculation with objects present"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Setup workspace
        from robot_environment.objects.pose_object import PoseObjectPNP

        mock_ws = Mock()
        mock_ws.xy_ul_wc.return_value = PoseObjectPNP(0.4, 0.15, 0.0)
        mock_ws.xy_lr_wc.return_value = PoseObjectPNP(0.1, -0.15, 0.0)

        # Access the workspaces properly through the mock
        mock_dependencies["workspaces"].return_value.get_workspace.return_value = mock_ws

        # Add mock object
        mock_obj = Mock(spec=Object)
        mock_obj.x_com.return_value = 0.25
        mock_obj.y_com.return_value = 0.0
        mock_obj.width_m.return_value = 0.05
        mock_obj.height_m.return_value = 0.05
        env._obj_position_memory = Objects([mock_obj])

        area, cx, cy = env.get_largest_free_space_with_center()

        assert area >= 0


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

        assert env.verbose() is True
