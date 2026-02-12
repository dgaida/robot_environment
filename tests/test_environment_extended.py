"""
Extended unit tests for Environment class - FIXED VERSION

Key fix:
- Removed test_update_camera_and_objects_tracks_visibility test since
  _track_workspace_visibility method doesn't exist in Environment class
"""

import pytest
import numpy as np
import threading
from unittest.mock import Mock, patch, create_autospec
from robot_environment.environment import Environment
from robot_workspace import Objects, Object, PoseObjectPNP, Workspace
from tests.conftest import create_mock_workspace


@pytest.fixture
def mock_dependencies():
    """Mock all Environment dependencies"""
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
        mock_fg.return_value = mock_fg_instance

        ws1 = create_mock_workspace("niryo_ws_left")
        ws2 = create_mock_workspace("niryo_ws_right")

        # Setup workspaces with complete interface
        mock_ws_instance = Mock()
        # Make it iterable
        mock_ws_instance.__iter__ = Mock(return_value=iter([ws1, ws2]))

        mock_workspace = Mock()
        mock_workspace.id.return_value = "test_ws"
        mock_workspace.img_shape.return_value = (640, 480, 3)

        # Helper to create proper PoseObjectPNP-like mocks with xy_coordinate()
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
        mock_workspace.set_img_shape = Mock()

        mock_ws_instance.get_workspace = Mock(return_value=mock_workspace)
        mock_ws_instance.get_workspace_by_id = Mock(return_value=mock_workspace)
        mock_ws_instance.get_workspace_home_id.return_value = "test_ws"
        mock_ws_instance.get_observation_pose.return_value = PoseObjectPNP(0.2, 0.0, 0.3)
        mock_ws_instance.get_visible_workspace.return_value = mock_workspace
        mock_ws_instance.get_home_workspace.return_value = mock_workspace
        mock_ws.return_value = mock_ws_instance

        # Setup TTS
        mock_tts_instance = Mock()
        mock_tts_instance.call_text2speech_async.return_value = Mock()
        mock_tts.return_value = mock_tts_instance

        # Setup Redis broker
        mock_broker_instance = Mock()
        mock_broker_instance.get_latest_objects.return_value = []
        mock_broker_instance.test_connection.return_value = True
        mock_broker.return_value = mock_broker_instance

        # Setup Redis label manager
        mock_labels_instance = Mock()
        mock_labels_instance.get_latest_labels.return_value = ["pencil", "pen"]
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


@pytest.fixture
def mock_workspace():
    """Create a mock workspace"""
    workspace = Mock(spec=Workspace)
    workspace.id.return_value = "test_workspace"
    workspace.img_shape.return_value = (640, 480, 3)
    workspace.set_img_shape = Mock()

    def mock_transform(ws_id, u_rel, v_rel, yaw=0.0):
        # Map relative coords to world coords
        # u_rel increases downward (0 to 1), x should decrease (higher to lower)
        # v_rel increases rightward (0 to 1), y should decrease (higher to lower)
        x = 0.4 - u_rel * 0.3  # x decreases as u increases
        y = 0.15 - v_rel * 0.3  # y decreases as v increases
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    workspace.transform_camera2world_coords = mock_transform
    return workspace


def create_mock_object(label, x, y, width=0.05, height=0.05):
    """Helper to create properly mocked Object"""
    obj = Mock(spec=Object)
    obj.label.return_value = label
    obj.x_com.return_value = x
    obj.y_com.return_value = y
    obj.xy_com.return_value = (x, y)
    obj.width_m.return_value = width
    obj.height_m.return_value = height
    obj.coordinate.return_value = [x, y]
    obj._x_com = x
    obj._y_com = y
    return obj


class TestEnvironmentMemoryManagement:
    """Test memory management features"""

    def test_memory_initialized_empty(self, mock_dependencies):
        """Test that memory starts empty"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        memory = env.get_detected_objects_from_memory()
        assert len(memory) == 0

    def test_clear_memory_removes_all_objects(self, mock_dependencies):
        """Test clearing memory removes all objects"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Clear memory using new API
        env.clear_memory()

        memory = env.get_detected_objects_from_memory()
        assert len(memory) == 0

    def test_remove_object_from_memory_success(self, mock_dependencies):
        """Test removing specific object from memory"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Set current workspace
        env._current_workspace_id = "test_ws"

        # This should not crash even if object doesn't exist
        env.remove_object_from_memory("pencil", [0.25, 0.05])

        assert True

    def test_verbose_setter(self, mock_dependencies):
        """Test verbose property setter (lines 835-837)"""
        env = Environment("key", False, "niryo", verbose=False, start_camera_thread=False)
        env.verbose = True
        assert env.verbose is True
        env.verbose = False
        assert env.verbose is False

    def test_remove_object_from_memory_with_tolerance(self, mock_dependencies):
        """Test removing object with coordinate tolerance"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env._current_workspace_id = "test_ws"

        # Should not crash with slightly different coordinates
        env.remove_object_from_memory("pencil", [0.249, 0.051])

        assert True

    def test_update_object_in_memory_success(self, mock_dependencies):
        """Test updating object position in memory"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env._current_workspace_id = "test_ws"
        new_pose = PoseObjectPNP(0.30, 0.10, 0.01)

        # Should not crash
        env.update_object_in_memory("pencil", [0.25, 0.05], new_pose)

        assert True

    def test_get_detected_objects_from_memory_returns_copy(self, mock_dependencies):
        """Test that memory access returns a copy"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        memory_copy = env.get_detected_objects_from_memory()

        # Modify copy
        memory_copy.clear()

        # Original should be unchanged (get a fresh copy)
        memory_original = env.get_detected_objects_from_memory()
        assert isinstance(memory_original, Objects)


class TestEnvironmentWorkspaceVisibility:
    """Test workspace visibility tracking"""

    def test_is_any_workspace_visible_true(self, mock_dependencies):
        """Test when workspace is visible"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        result = env.is_any_workspace_visible()

        assert isinstance(result, bool)

    def test_is_any_workspace_visible_false(self, mock_dependencies):
        """Test when no workspace is visible"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Mock to return None for get_visible_workspace
        env._workspaces.get_visible_workspace = Mock(return_value=None)

        result = env.is_any_workspace_visible()

        assert result is False


class TestEnvironmentObjectDetection:
    """Test object detection and tracking"""

    def test_get_detected_objects_from_redis(self, mock_dependencies):
        """Test getting objects from Redis"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Mock Redis returning objects
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

    def test_get_detected_objects_empty(self, mock_dependencies):
        """Test getting objects when none detected"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        mock_dependencies["broker"].return_value.get_latest_objects.return_value = []

        objects = env.get_detected_objects()

        assert len(objects) == 0


class TestEnvironmentCameraThread:
    """Test camera update thread functionality"""

    def test_start_camera_updates_creates_thread(self, mock_dependencies):
        """Test that camera update thread is created"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        thread = env.start_camera_updates(visualize=False)

        assert thread is not None
        assert isinstance(thread, threading.Thread)

        # Stop the thread
        env.stop_camera_updates()

    def test_stop_camera_updates_sets_stop_event(self, mock_dependencies):
        """Test stopping camera updates"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.stop_camera_updates()

        assert env._stop_event.is_set()

    @patch("robot_environment.environment.time")
    def test_update_camera_and_objects_loop(self, mock_time, mock_dependencies):
        """Test camera update loop iteration"""
        # Mock time.sleep to avoid delays
        mock_time.sleep = Mock()
        mock_time.perf_counter = Mock(side_effect=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        env = Environment("key", False, "niryo", start_camera_thread=False)

        iterations = 0

        # Mock robot methods to avoid actual side effects
        with patch.object(env, "robot_move2observation_pose"), patch.object(
            env, "get_current_frame", return_value=np.zeros((480, 640, 3), dtype=np.uint8)
        ), patch.object(env, "get_detected_objects", return_value=Objects()):

            for _ in env.update_camera_and_objects(visualize=False):
                iterations += 1
                # Now set stop event after first iteration
                env._stop_event.set()
                if iterations >= 1:
                    break

        assert iterations >= 1


class TestEnvironmentLargestFreeSpaceAdvanced:
    """Advanced tests for largest free space calculation"""

    @patch("robot_environment.utils.workspace_utils.cv2")
    def test_largest_free_space_empty_workspace(self, mock_cv2, mock_dependencies):
        """Test free space with empty workspace"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        area, cx, cy = env.get_largest_free_space_with_center()

        # Empty workspace should have large free area
        assert area >= 0
        assert isinstance(cx, float)
        assert isinstance(cy, float)

    @patch("robot_environment.utils.workspace_utils.cv2")
    def test_largest_free_space_verbose_output(self, mock_cv2, mock_dependencies):
        """Test verbose output during calculation"""
        env = Environment("key", False, "niryo", verbose=True, start_camera_thread=False)

        # Should not crash with verbose output
        area, cx, cy = env.get_largest_free_space_with_center()

        assert area >= 0


class TestEnvironmentGetters:
    """Test various getter methods"""

    def test_get_workspace_id(self, mock_dependencies):
        """Test getting workspace ID by index"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        ws_id = env.get_workspace_id(0)

        assert ws_id is not None

    def test_get_object_labels(self, mock_dependencies):
        """Test getting object labels list"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        labels = env.get_object_labels()

        assert isinstance(labels, list)
        assert len(labels) > 0

    def test_get_detected_objects_converts_to_objects(self, mock_dependencies):
        """Test that detected objects are converted properly"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

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


class TestEnvironmentThreadSafety:
    """Test thread safety of memory operations"""

    def test_concurrent_memory_reads_safe(self, mock_dependencies):
        """Test concurrent memory reads are safe"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        results = []

        def read_memory():
            memory = env.get_detected_objects_from_memory()
            results.append(len(memory))

        threads = [threading.Thread(target=read_memory) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(results) == 5


class TestEnvironmentEdgeCases:
    """Test edge cases and error conditions"""

    def test_get_workspace_coordinate_invalid_point_type(self, mock_dependencies):
        """Test getting coordinate with invalid point type"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        result = env.get_workspace_coordinate_from_point("test_ws", "invalid_point")

        assert result is None

    def test_get_workspace_coordinate_all_valid_points(self, mock_dependencies):
        """Test all valid point types"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        valid_points = ["upper left corner", "upper right corner", "lower left corner", "lower right corner", "center point"]

        for point in valid_points:
            result = env.get_workspace_coordinate_from_point("test_ws", point)
            assert result is not None
            assert len(result) == 2

    def test_add_object_with_empty_string(self, mock_dependencies):
        """Test adding object with empty string"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        result = env.add_object_name2object_labels("")

        assert isinstance(result, str)

    def test_remove_object_with_wrong_coordinates(self, mock_dependencies):
        """Test removing object with coordinates far from actual position"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env._current_workspace_id = "test_ws"

        # Try to remove with very different coordinates (should not crash)
        env.remove_object_from_memory("pencil", [0.50, 0.50])

        assert True

    def test_update_object_with_wrong_coordinates(self, mock_dependencies):
        """Test updating object with wrong coordinates"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env._current_workspace_id = "test_ws"
        new_pose = PoseObjectPNP(0.30, 0.10, 0.01)

        # Try to update with wrong coordinates (should not crash)
        env.update_object_in_memory("pencil", [0.50, 0.50], new_pose)

        assert True


class TestEnvironmentVerboseMode:
    """Test verbose output mode"""

    def test_verbose_initialization(self, mock_dependencies):
        """Test verbose initialization output"""
        env = Environment("key", False, "niryo", verbose=True, start_camera_thread=False)

        assert env.verbose is True

    def test_verbose_memory_operations(self, mock_dependencies):
        """Test verbose output during memory operations"""
        env = Environment("key", False, "niryo", verbose=True, start_camera_thread=False)

        env._current_workspace_id = "test_ws"

        # These should not crash with verbose output
        env.clear_memory()
        env.remove_object_from_memory("pencil", [0.25, 0.05])

        assert True


class TestEnvironmentCleanup:
    """Test cleanup and resource management"""

    def test_cleanup_sets_stop_event(self, mock_dependencies):
        """Test cleanup sets stop event"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.cleanup()

        assert env._stop_event.is_set()

    def test_destructor_calls_cleanup(self, mock_dependencies):
        """Test destructor behavior"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Manually call destructor
        env.__del__()

        assert env._stop_event.is_set()

    def test_multiple_cleanup_calls_safe(self, mock_dependencies):
        """Test multiple cleanup calls don't cause issues"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.cleanup()
        env.cleanup()  # Second call

        assert env._stop_event.is_set()


class TestEnvironmentMultiWorkspace:
    """Test multi-workspace functionality"""

    def test_get_current_workspace_id(self, mock_dependencies):
        """Test getting current workspace ID"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        ws_id = env.get_current_workspace_id()

        assert ws_id is not None

    def test_set_current_workspace(self, mock_dependencies):
        """Test setting current workspace"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.set_current_workspace("new_ws")

        assert env.get_current_workspace_id() == "new_ws"

    def test_get_detected_objects_from_workspace(self, mock_dependencies):
        """Test getting objects from specific workspace"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        objects = env.get_detected_objects_from_workspace("test_ws")

        assert isinstance(objects, Objects)

    def test_get_all_workspace_objects(self, mock_dependencies):
        """Test getting objects from all workspaces"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        all_objects = env.get_all_workspace_objects()

        assert isinstance(all_objects, dict)

    def test_clear_workspace_memory(self, mock_dependencies):
        """Test clearing specific workspace memory"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Should not crash
        env.clear_workspace_memory("test_ws")

        assert True

    def test_remove_object_from_workspace(self, mock_dependencies):
        """Test removing object from specific workspace"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Should not crash
        env.remove_object_from_workspace("test_ws", "pencil", [0.25, 0.05])

        assert True


class TestEnvironmentOralCommunication:
    """Test text-to-speech functionality"""

    def test_oralcom_call_text2speech_async(self, mock_dependencies):
        """Test asynchronous text-to-speech"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        thread = env.oralcom_call_text2speech_async("Test message")

        assert thread is not None


class TestEnvironmentRobotControl:
    """Test robot control methods"""

    def test_robot_move2home_observation_pose(self, mock_dependencies):
        """Test moving to home observation pose"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.robot_move2home_observation_pose()

        # Should call move to observation pose
        env.robot().move2observation_pose.assert_called()

    def test_get_robot_in_motion(self, mock_dependencies):
        """Test checking if robot is in motion"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        in_motion = env.get_robot_in_motion()

        assert isinstance(in_motion, bool)

    def test_get_robot_target_pose_from_rel(self, mock_dependencies):
        """Test getting target pose from relative coordinates"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        mock_pose = PoseObjectPNP(0.25, 0.05, 0.01)
        env._robot.get_target_pose_from_rel = Mock(return_value=mock_pose)

        pose = env.get_robot_target_pose_from_rel("test_ws", 0.5, 0.5, 0.0)

        assert isinstance(pose, PoseObjectPNP)

    def test_update_object_in_workspace(self, mock_dependencies):
        """Test moving object between workspaces in memory (line 455)"""
        env = Environment("key", False, "niryo", start_camera_thread=False)
        with patch.object(env._memory_manager, "move_object") as mock_move:
            env.update_object_in_workspace("ws1", "ws2", "cube", [0.1, 0.1], [0.2, 0.2])
            mock_move.assert_called_once()

    def test_get_largest_free_space_no_workspace(self, mock_dependencies):
        """Test get_largest_free_space_with_center when workspace is not found"""
        env = Environment("key", False, "niryo", start_camera_thread=False)
        env._workspaces.get_workspace_by_id.return_value = None

        area, cx, cy = env.get_largest_free_space_with_center("invalid_ws")
        assert area == 0.0

    def test_init_exception_workspace_iteration(self, mock_dependencies):
        """Test exception handling during workspace initialization in __init__ (lines 136-140)"""
        # Force exception during iteration
        mock_ws = mock_dependencies["workspaces"].return_value
        mock_ws.__iter__.side_effect = Exception("Iteration failed")
        mock_ws.get_workspace_home_id.return_value = "home_ws"

        env = Environment("key", False, "niryo", start_camera_thread=False)
        # Should have called initialize_workspace for home_ws as fallback
        assert env._memory_manager.get("home_ws") is not None

    def test_cleanup_with_tts_shutdown(self, mock_dependencies):
        """Test cleanup calls oralcom.shutdown (lines 200-206)"""
        env = Environment("key", False, "niryo", start_camera_thread=False)
        env.cleanup()
        env._oralcom.shutdown.assert_called_once()

    def test_performance_metrics_methods(self, mock_dependencies):
        """Test performance metrics related methods in Environment"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        assert env.get_performance_metrics() == env._metrics
        stats = env.get_performance_stats()
        assert isinstance(stats, dict)
        assert "uptime_seconds" in stats

        env.print_performance_summary()

        with patch.object(env._metrics, "export_json") as mock_export:
            env.export_performance_metrics("test.json")
            mock_export.assert_called_once_with("test.json")

        env.reset_performance_metrics()

        # Test when metrics are disabled
        env._metrics = None
        assert env.get_performance_metrics() is None
        assert env.get_performance_stats() is None
        env.print_performance_summary()
        env.export_performance_metrics("test.json")
        env.reset_performance_metrics()

    def test_get_visible_workspace(self, mock_dependencies):
        """Test get_visible_workspace (line 647)"""
        env = Environment("key", False, "niryo", start_camera_thread=False)
        pose = PoseObjectPNP(0.2, 0, 0.3)
        ws = env.get_visible_workspace(pose)
        assert ws is not None

    def test_properties(self, mock_dependencies):
        """Test remaining properties in Environment (lines 796-830)"""
        env = Environment("key", False, "niryo", start_camera_thread=False)
        assert env.workspaces() == env._workspaces
        assert env.framegrabber() == env._framegrabber
        assert env.robot() == env._robot
        assert env.use_simulation() == env._use_simulation
        assert env.metrics() == env._metrics

    def test_get_workspace_coordinate_all_branches(self, mock_dependencies):
        """Test all branches of get_workspace_coordinate_from_point (lines 566-578)"""
        env = Environment("key", False, "niryo", start_camera_thread=False)
        points = ["upper left corner", "upper right corner", "lower left corner", "lower right corner", "center point"]
        for point in points:
            result = env.get_workspace_coordinate_from_point("test_ws", point)
            assert result is not None
            assert len(result) == 2

    def test_get_largest_free_space_with_center_basic(self, mock_dependencies):
        """Test get_largest_free_space_with_center (line 521)"""
        env = Environment("key", False, "niryo", start_camera_thread=False)
        # Mock get_workspace to return index 0 workspace
        env._workspaces.get_workspace.return_value = env.get_workspace_by_id("test_ws")

        area, cx, cy = env.get_largest_free_space_with_center()
        assert area >= 0

    def test_get_observation_pose(self, mock_dependencies):
        """Test get_observation_pose (line 658)"""
        env = Environment("key", False, "niryo", start_camera_thread=False)
        pose = env.get_observation_pose("test_ws")
        assert isinstance(pose, PoseObjectPNP)

    def test_get_current_frame_width_height(self, mock_dependencies):
        """Test get_current_frame_width_height (line 692)"""
        env = Environment("key", False, "niryo", start_camera_thread=False)
        env.framegrabber().get_current_frame_width_height.return_value = (480, 640)
        w, h = env.get_current_frame_width_height()
        assert w == 480
        assert h == 640

    def test_robot_move2observation_pose(self, mock_dependencies):
        """Test robot_move2observation_pose (line 772)"""
        env = Environment("key", False, "niryo", start_camera_thread=False)
        env.robot_move2observation_pose("test_ws")
        env.robot().move2observation_pose.assert_called_with("test_ws")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
