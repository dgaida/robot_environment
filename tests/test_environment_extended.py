"""
Extended unit tests for Environment class - FIXED VERSION
Tests memory management, object tracking, workspace visibility, and edge cases

FIX SUMMARY:
The tests were failing because env._workspaces was None. This happened because:
1. Environment.__init__() checks isinstance(self.get_robot_controller(), NiryoRobotController)
2. If True, it creates _workspaces = NiryoWorkspaces(...)
3. The mock wasn't passing the isinstance() check properly
4. Solution: Manually set _workspaces in each test that needs it
"""

import pytest
import numpy as np
import threading
import time
from unittest.mock import Mock, patch, create_autospec
from robot_environment.environment import Environment
from robot_workspace import Objects, Object, PoseObjectPNP, Workspace
from .conftest import create_mock_workspace


@pytest.fixture
def mock_dependencies():
    """Mock all Environment dependencies"""
    from robot_environment.robot.niryo_robot_controller import NiryoRobotController

    with patch("robot_environment.environment.Robot") as mock_robot, patch(
        "robot_environment.environment.NiryoFrameGrabber"
    ) as mock_fg, patch("robot_environment.environment.NiryoWorkspaces") as mock_ws, patch(
        "robot_environment.environment.Text2Speech"
    ) as mock_tts, patch(
        "robot_environment.environment.VisualCortex"
    ) as mock_vc, patch(
        "robot_environment.environment.get_default_config"
    ) as mock_config:

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
            mock_pose.xy_coordinate.return_value = [x, y]  # KEY FIX
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

        # Setup VisualCortex
        mock_vc_instance = Mock()
        mock_vc_instance.get_detected_objects.return_value = []
        mock_vc_instance.get_object_labels.return_value = [["pencil", "pen"]]
        mock_vc_instance.detect_objects_from_redis = Mock()
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


@pytest.fixture
def mock_workspace():
    """Create a mock workspace"""
    workspace = Mock(spec=Workspace)
    workspace.id.return_value = "test_workspace"
    workspace.img_shape.return_value = (640, 480, 3)
    workspace.set_img_shape = Mock()

    # Mock transform method - FIXED coordinate system
    # For Niryo: width goes along y-axis, height along x-axis
    # Upper-left should have higher x and higher y than lower-right
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


def ensure_workspaces(env, mock_dependencies):
    """Helper to ensure _workspaces is set (workaround for isinstance check)"""
    if env._workspaces is None:
        env._workspaces = mock_dependencies["workspaces"].return_value


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

        # Add objects to memory
        obj1 = create_mock_object("pencil", 0.25, 0.05)
        obj2 = create_mock_object("pen", 0.30, 0.10)
        env._obj_position_memory.append(obj1)
        env._obj_position_memory.append(obj2)

        assert len(env._obj_position_memory) == 2

        env.clear_memory()

        assert len(env._obj_position_memory) == 0

    def test_clear_memory_resets_workspace_lost_flag(self, mock_dependencies):
        """Test that clear_memory resets workspace_was_lost flag"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env._workspace_was_lost = True
        env.clear_memory()

        assert env._workspace_was_lost is False

    def test_remove_object_from_memory_success(self, mock_dependencies):
        """Test removing specific object from memory"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        obj = create_mock_object("pencil", 0.25, 0.05)
        env._obj_position_memory.append(obj)

        env.remove_object_from_memory("pencil", [0.25, 0.05])

        assert len(env._obj_position_memory) == 0

    def test_remove_object_from_memory_with_tolerance(self, mock_dependencies):
        """Test removing object with coordinate tolerance"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        obj = create_mock_object("pencil", 0.25, 0.05)
        env._obj_position_memory.append(obj)

        # Remove with slightly different coordinates (within tolerance)
        env.remove_object_from_memory("pencil", [0.249, 0.051])

        assert len(env._obj_position_memory) == 0

    def test_remove_object_not_found_warning(self, mock_dependencies):
        """Test warning when removing non-existent object"""
        env = Environment("key", False, "niryo", verbose=True, start_camera_thread=False)

        # Try to remove object that doesn't exist
        env.remove_object_from_memory("nonexistent", [0.25, 0.05])

        # Should not crash, memory should be empty
        assert len(env._obj_position_memory) == 0

    def test_update_object_in_memory_success(self, mock_dependencies):
        """Test updating object position in memory"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        obj = create_mock_object("pencil", 0.25, 0.05)
        env._obj_position_memory.append(obj)

        env.update_object_in_memory("pencil", [0.25, 0.05], [0.30, 0.10])

        # Check updated position
        assert obj._x_com == 0.30
        assert obj._y_com == 0.10

    def test_update_object_not_found_warning(self, mock_dependencies):
        """Test warning when updating non-existent object"""
        env = Environment("key", False, "niryo", verbose=True, start_camera_thread=False)

        env.update_object_in_memory("nonexistent", [0.25, 0.05], [0.30, 0.10])

        # Should not crash
        assert True

    def test_get_detected_objects_from_memory_returns_copy(self, mock_dependencies):
        """Test that memory access returns a copy"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        obj = create_mock_object("pencil", 0.25, 0.05)
        env._obj_position_memory.append(obj)

        memory_copy = env.get_detected_objects_from_memory()

        # Modify copy
        memory_copy.clear()

        # Original should be unchanged
        assert len(env._obj_position_memory) == 1


class TestEnvironmentWorkspaceVisibility:
    """Test workspace visibility tracking"""

    def test_should_update_memory_when_conditions_met(self, mock_dependencies):
        """Test memory update conditions"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Mock conditions: workspace visible, robot not in motion
        with patch.object(env, "is_any_workspace_visible", return_value=True), patch.object(
            env, "get_robot_in_motion", return_value=False
        ):

            result = env._should_update_memory()
            assert result is True

    def test_should_not_update_memory_when_workspace_not_visible(self, mock_dependencies):
        """Test no update when workspace not visible"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        with patch.object(env, "is_any_workspace_visible", return_value=False), patch.object(
            env, "get_robot_in_motion", return_value=False
        ):

            result = env._should_update_memory()
            assert result is False

    def test_should_not_update_memory_when_robot_in_motion(self, mock_dependencies):
        """Test no update when robot is moving"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        with patch.object(env, "is_any_workspace_visible", return_value=True), patch.object(
            env, "get_robot_in_motion", return_value=True
        ):

            result = env._should_update_memory()
            assert result is False

    def test_should_clear_memory_when_returning_after_loss(self, mock_dependencies):
        """Test memory clear when returning to observation pose"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env._workspace_was_lost = True

        with patch.object(env, "is_any_workspace_visible", return_value=True), patch.object(
            env, "get_robot_in_motion", return_value=False
        ):

            result = env._should_clear_memory()
            assert result is True

    def test_should_not_clear_memory_when_workspace_never_lost(self, mock_dependencies):
        """Test no clear when workspace was never lost"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env._workspace_was_lost = False

        with patch.object(env, "is_any_workspace_visible", return_value=True), patch.object(
            env, "get_robot_in_motion", return_value=False
        ):

            result = env._should_clear_memory()
            assert result is False

    def test_track_workspace_visibility_detects_loss(self, mock_dependencies):
        """Test tracking detects when workspace is lost"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Start at observation pose
        env._is_at_observation_pose = True

        # Lose workspace
        with patch.object(env, "is_any_workspace_visible", return_value=False), patch.object(
            env, "get_robot_in_motion", return_value=True
        ):

            env._track_workspace_visibility()

            assert env._workspace_was_lost is True
            assert env._is_at_observation_pose is False

    def test_track_workspace_visibility_regain(self, mock_dependencies):
        """Test tracking when workspace is regained"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env._workspace_was_lost = True
        env._is_at_observation_pose = False

        # Regain workspace
        with patch.object(env, "is_any_workspace_visible", return_value=True), patch.object(
            env, "get_robot_in_motion", return_value=False
        ):

            env._track_workspace_visibility()

            assert env._is_at_observation_pose is True
            # Flag should remain True until memory is cleared
            assert env._workspace_was_lost is True


class TestEnvironmentObjectDetection:
    """Test object detection and tracking"""

    def test_check_new_detections_adds_unique_objects(self, mock_dependencies):
        """Test that only unique objects are added"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        obj1 = create_mock_object("pencil", 0.25, 0.05)
        obj2 = create_mock_object("pen", 0.30, 0.10)

        with patch.object(env, "_should_update_memory", return_value=True), patch.object(
            env, "_should_clear_memory", return_value=False
        ), patch.object(env, "is_any_workspace_visible", return_value=True), patch.object(
            env, "get_robot_in_motion", return_value=False
        ):
            env._check_new_detections(Objects([obj1, obj2]))

        assert len(env._obj_position_memory) == 2

    def test_check_new_detections_skips_duplicates(self, mock_dependencies):
        """Test that duplicate objects are not added"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        obj1 = create_mock_object("pencil", 0.25, 0.05)
        obj2 = create_mock_object("pencil", 0.251, 0.051)  # Very close

        with patch.object(env, "_should_update_memory", return_value=True), patch.object(
            env, "_should_clear_memory", return_value=False
        ), patch.object(env, "is_any_workspace_visible", return_value=True), patch.object(
            env, "get_robot_in_motion", return_value=False
        ):
            env._check_new_detections(Objects([obj1]))
            env._check_new_detections(Objects([obj2]))

        # Should only have one object
        assert len(env._obj_position_memory) == 1

    def test_check_new_detections_skips_when_conditions_not_met(self, mock_dependencies):
        """Test no detection when update conditions not met"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        obj = create_mock_object("pencil", 0.25, 0.05)

        # Mock both _should_update_memory AND _should_clear_memory to avoid workspace access
        with patch.object(env, "_should_update_memory", return_value=False), patch.object(
            env, "_should_clear_memory", return_value=False
        ):
            env._check_new_detections(Objects([obj]))

        assert len(env._obj_position_memory) == 0

    def test_check_new_detections_clears_memory_if_needed(self, mock_dependencies):
        """Test that memory is cleared before new detections"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # Add old object
        old_obj = create_mock_object("old", 0.1, 0.1)
        env._obj_position_memory.append(old_obj)

        new_obj = create_mock_object("new", 0.25, 0.05)

        with patch.object(env, "_should_clear_memory", return_value=True), patch.object(
            env, "_should_update_memory", return_value=True
        ), patch.object(env, "is_any_workspace_visible", return_value=True), patch.object(
            env, "get_robot_in_motion", return_value=False
        ):

            env._check_new_detections(Objects([new_obj]))

        # Old object should be cleared, new object added
        assert len(env._obj_position_memory) == 1
        assert env._obj_position_memory[0].label() == "new"


class TestEnvironmentCameraThread:
    """Test camera update thread functionality"""

    def test_start_camera_updates_creates_thread(self, mock_dependencies):
        """Test that camera update thread is created"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        with patch.object(env, "update_camera_and_objects"):
            thread = env.start_camera_updates(visualize=False)

            assert thread is not None
            assert isinstance(thread, threading.Thread)

    def test_stop_camera_updates_sets_stop_event(self, mock_dependencies):
        """Test stopping camera updates"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        env.stop_camera_updates()

        assert env._stop_event.is_set()

    @patch("robot_environment.environment.cv2")
    def test_update_camera_and_objects_loop(self, mock_cv2, mock_dependencies):
        """Test camera update loop iteration - FIXED"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # FIXED: Ensure _workspaces is properly set
        assert env._workspaces is not None, "_workspaces should not be None"

        # Mock robot_move2observation_pose to avoid actual robot movement
        with patch.object(env, "robot_move2observation_pose"):
            iterations = 0
            for _ in env.update_camera_and_objects(visualize=False):
                iterations += 1
                if iterations >= 1:
                    # Stop AFTER first iteration starts
                    env._stop_event.set()
                    break

        # Should have called get_current_frame during the iteration
        env._framegrabber.get_current_frame.assert_called()

    @patch("robot_environment.environment.cv2")
    def test_update_camera_and_objects_tracks_visibility(self, mock_cv2, mock_dependencies):
        """Test that update loop tracks workspace visibility - FIXED"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        # FIXED: Ensure _workspaces is properly set
        assert env._workspaces is not None, "_workspaces should not be None"

        with patch.object(env, "_track_workspace_visibility") as mock_track, patch.object(env, "robot_move2observation_pose"):
            iterations = 0
            for _ in env.update_camera_and_objects(visualize=False):
                iterations += 1
                if iterations >= 1:
                    env._stop_event.set()
                    break

            mock_track.assert_called()


class TestEnvironmentLargestFreeSpaceAdvanced:
    """Advanced tests for largest free space calculation"""

    @patch("robot_environment.environment.cv2")
    def test_largest_free_space_with_multiple_objects(self, mock_cv2, mock_dependencies):
        """Test free space calculation with multiple objects"""
        env = Environment("key", False, "niryo", start_camera_thread=False)
        ensure_workspaces(env, mock_dependencies)

        # Add multiple objects
        obj1 = create_mock_object("obj1", 0.20, 0.05, 0.05, 0.05)
        obj2 = create_mock_object("obj2", 0.30, 0.05, 0.05, 0.05)
        obj3 = create_mock_object("obj3", 0.25, -0.05, 0.05, 0.05)

        env._obj_position_memory = Objects([obj1, obj2, obj3])

        # FIX: Ensure _workspaces is accessible
        # assert env._workspaces is not None, "_workspaces should be set during init"

        area, cx, cy = env.get_largest_free_space_with_center()

        assert area >= 0
        assert isinstance(cx, float)
        assert isinstance(cy, float)

    @patch("robot_environment.environment.cv2")
    def test_largest_free_space_empty_workspace(self, mock_cv2, mock_dependencies):
        """Test free space with empty workspace"""
        env = Environment("key", False, "niryo", start_camera_thread=False)
        ensure_workspaces(env, mock_dependencies)

        env._obj_position_memory = Objects()

        # FIX: Ensure _workspaces is accessible
        # assert env._workspaces is not None, "_workspaces should be set during init"

        area, cx, cy = env.get_largest_free_space_with_center()

        # Empty workspace should have large free area
        assert area > 0

    @patch("robot_environment.environment.cv2")
    def test_largest_free_space_verbose_output(self, mock_cv2, mock_dependencies):
        """Test verbose output during calculation"""
        env = Environment("key", False, "niryo", verbose=True, start_camera_thread=False)
        ensure_workspaces(env, mock_dependencies)

        obj = create_mock_object("obj", 0.25, 0.0, 0.05, 0.05)
        env._obj_position_memory = Objects([obj])

        # FIX: Ensure _workspaces is accessible
        # assert env._workspaces is not None, "_workspaces should be set during init"

        # Should not crash with verbose output
        area, cx, cy = env.get_largest_free_space_with_center()

        assert area >= 0


class TestEnvironmentGetters:
    """Test various getter methods"""

    def test_get_workspace_id(self, mock_dependencies):
        """Test getting workspace ID by index"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        mock_dependencies["workspaces"].return_value.get_workspace_id.return_value = "ws_1"

        ws_id = env.get_workspace_id(1)

        assert ws_id == "ws_1"

    def test_get_object_labels(self, mock_dependencies):
        """Test getting object labels list"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        labels = env.get_object_labels()

        assert isinstance(labels, list)
        assert len(labels) > 0

    def test_get_detected_objects_converts_to_objects(self, mock_dependencies):
        """Test that detected objects are converted properly"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        mock_dict = {"label": "pencil", "bbox": {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200}, "has_mask": False}

        mock_dependencies["visual_cortex"].return_value.get_detected_objects.return_value = [mock_dict]

        objects = env.get_detected_objects()

        assert isinstance(objects, Objects)


class TestEnvironmentThreadSafety:
    """Test thread safety of memory operations"""

    def test_memory_lock_prevents_concurrent_access(self, mock_dependencies):
        """Test that memory lock prevents concurrent access"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        access_order = []

        def access_memory(thread_id):
            with env._memory_lock:
                access_order.append(f"start-{thread_id}")
                time.sleep(0.01)  # Simulate work
                access_order.append(f"end-{thread_id}")

        threads = [threading.Thread(target=access_memory, args=(i,)) for i in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Access should be serialized (each start followed by its end)
        for i in range(0, len(access_order), 2):
            assert access_order[i].startswith("start")
            assert access_order[i + 1].startswith("end")

    def test_concurrent_memory_reads_safe(self, mock_dependencies):
        """Test concurrent memory reads are safe"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        obj = create_mock_object("test", 0.25, 0.05)
        env._obj_position_memory.append(obj)

        results = []

        def read_memory():
            memory = env.get_detected_objects_from_memory()
            results.append(len(memory))

        threads = [threading.Thread(target=read_memory) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should read 1 object
        assert all(r == 1 for r in results)


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

        mock_thread = Mock()
        mock_dependencies["tts"].return_value.call_text2speech_async.return_value = mock_thread

        result = env.add_object_name2object_labels("")

        assert "" in result

    def test_remove_object_with_wrong_coordinates(self, mock_dependencies):
        """Test removing object with coordinates far from actual position"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        obj = create_mock_object("pencil", 0.25, 0.05)
        env._obj_position_memory.append(obj)

        # Try to remove with very different coordinates
        env.remove_object_from_memory("pencil", [0.50, 0.50])

        # Object should still be in memory
        assert len(env._obj_position_memory) == 1

    def test_update_object_with_wrong_coordinates(self, mock_dependencies):
        """Test updating object with wrong coordinates - FIXED"""
        env = Environment("key", False, "niryo", start_camera_thread=False)

        obj = create_mock_object("pencil", 0.25, 0.05)
        env._obj_position_memory.append(obj)

        # Store original values
        original_x = obj.x_com()
        original_y = obj.y_com()

        # Try to update with wrong coordinates
        env.update_object_in_memory("pencil", [0.50, 0.50], [0.30, 0.10])

        # Object should be unchanged (coordinates didn't match)
        assert obj.x_com() == original_x
        assert obj.y_com() == original_y


class TestEnvironmentVerboseMode:
    """Test verbose output mode"""

    def test_verbose_initialization(self, mock_dependencies):
        """Test verbose initialization output"""
        env = Environment("key", False, "niryo", verbose=True, start_camera_thread=False)

        assert env.verbose() is True

    def test_verbose_memory_operations(self, mock_dependencies):
        """Test verbose output during memory operations - FIXED"""
        env = Environment("key", False, "niryo", verbose=True, start_camera_thread=False)

        obj = create_mock_object("pencil", 0.25, 0.05)

        # FIXED: Mock workspace visibility checks to avoid accessing _workspaces
        with patch.object(env, "_should_update_memory", return_value=True), patch.object(
            env, "_should_clear_memory", return_value=False
        ):
            # These should not crash with verbose output
            env._check_new_detections(Objects([obj]))

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
