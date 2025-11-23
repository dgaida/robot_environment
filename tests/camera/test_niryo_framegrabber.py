"""
Unit tests for NiryoFrameGrabber class - FIXED VERSION
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from robot_environment.camera.niryo_framegrabber import NiryoFrameGrabber
from robot_environment.robot.niryo_robot_controller import NiryoRobotController
from robot_workspace import PoseObjectPNP


@pytest.fixture
def mock_environment():
    """Create mock environment with NiryoRobotController"""
    env = Mock()
    env.verbose.return_value = False

    # Create mock robot controller
    robot_controller = Mock(spec=NiryoRobotController)

    # FIX: Properly mock the lock context manager
    mock_lock = Mock()
    mock_lock.__enter__ = Mock(return_value=None)
    mock_lock.__exit__ = Mock(return_value=None)
    robot_controller.lock.return_value = mock_lock

    # Mock the underlying robot
    mock_robot = Mock()
    mock_robot.get_camera_intrinsics.return_value = (
        np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]]),  # Camera matrix
        np.array([0.1, 0.01, 0, 0, 0]),  # Distortion coefficients
    )
    mock_robot.get_img_compressed.return_value = b"fake_compressed_image"

    robot_controller.robot_ctrl.return_value = mock_robot
    env.get_robot_controller.return_value = robot_controller

    return env


@pytest.fixture
def mock_redis_streamer():
    """Create mock Redis image streamer"""
    with patch("robot_environment.camera.niryo_framegrabber.RedisImageStreamer") as mock_streamer:
        mock_instance = Mock()
        mock_instance.publish_image.return_value = "stream_id_123"
        mock_streamer.return_value = mock_instance
        yield mock_streamer


class TestNiryoFrameGrabber:
    """Test suite for NiryoFrameGrabber class"""

    def test_initialization(self, mock_environment, mock_redis_streamer):
        """Test framegrabber initialization"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        assert framegrabber.environment() == mock_environment
        assert framegrabber.frame_counter == 0
        assert framegrabber.camera_matrix() is not None
        assert framegrabber.camera_dist_coeff() is not None

    def test_initialization_with_custom_stream_name(self, mock_environment, mock_redis_streamer):
        """Test initialization with custom stream name"""
        NiryoFrameGrabber(mock_environment, stream_name="custom_stream")

        mock_redis_streamer.assert_called_once_with(stream_name="custom_stream")

    def test_initialization_wrong_robot_type(self, mock_environment):
        """Test that initialization fails with wrong robot type"""
        mock_environment.get_robot_controller.return_value = Mock()  # Not NiryoRobotController

        with pytest.raises(TypeError, match="robot must be an instance of NiryoRobotController"):
            NiryoFrameGrabber(mock_environment)

    def test_camera_intrinsics_retrieved(self, mock_environment, mock_redis_streamer):
        """Test that camera intrinsics are retrieved"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        mtx = framegrabber.camera_matrix()
        dist = framegrabber.camera_dist_coeff()

        assert isinstance(mtx, np.ndarray)
        assert isinstance(dist, np.ndarray)
        assert mtx.shape == (3, 3)

    @patch("robot_environment.camera.niryo_framegrabber.uncompress_image")
    @patch("robot_environment.camera.niryo_framegrabber.undistort_image")
    @patch("robot_environment.camera.niryo_framegrabber.extract_img_workspace")
    @patch("robot_environment.camera.niryo_framegrabber.cv2")
    def test_get_current_frame_success(
        self, mock_cv2, mock_extract, mock_undistort, mock_uncompress, mock_environment, mock_redis_streamer
    ):
        """Test successful frame capture"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        # Setup mocks
        mock_uncompress.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_undistort.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_extract.return_value = np.zeros((400, 400, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((400, 400, 3), dtype=np.uint8)

        # Mock environment methods
        mock_environment.get_robot_pose.return_value = PoseObjectPNP(0.2, 0.0, 0.3)
        mock_workspace = Mock()
        mock_workspace.set_img_shape = Mock()
        mock_environment.get_visible_workspace.return_value = mock_workspace

        frame = framegrabber.get_current_frame()

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        mock_uncompress.assert_called_once()
        mock_undistort.assert_called_once()
        mock_extract.assert_called_once()

    @patch("robot_environment.camera.niryo_framegrabber.uncompress_image")
    @patch("robot_environment.camera.niryo_framegrabber.undistort_image")
    @patch("robot_environment.camera.niryo_framegrabber.extract_img_workspace")
    @patch("robot_environment.camera.niryo_framegrabber.cv2")
    def test_get_current_frame_no_workspace(
        self, mock_cv2, mock_extract, mock_undistort, mock_uncompress, mock_environment, mock_redis_streamer
    ):
        """Test frame capture when no workspace is extracted"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        mock_uncompress.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_undistort.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_extract.return_value = None  # No workspace found
        mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_environment.get_robot_pose.return_value = PoseObjectPNP(0.2, 0.0, 0.3)
        mock_environment.get_visible_workspace.return_value = None

        frame = framegrabber.get_current_frame()

        assert frame is not None
        # Should return undistorted image instead of workspace
        mock_cv2.cvtColor.assert_called_once()

    @patch("robot_environment.camera.niryo_framegrabber.uncompress_image")
    def test_get_current_frame_unicode_error(self, mock_uncompress, mock_environment, mock_redis_streamer):
        """Test handling of UnicodeDecodeError"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        # Setup initial frame
        framegrabber._current_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock robot to raise UnicodeDecodeError
        robot_ctrl = mock_environment.get_robot_controller()
        robot_ctrl.robot_ctrl().get_img_compressed.side_effect = UnicodeDecodeError("utf-8", b"\x00\x00", 0, 1, "invalid")

        # Need to also mock lock context manager
        robot_ctrl.lock.return_value.__enter__ = Mock(return_value=None)
        robot_ctrl.lock.return_value.__exit__ = Mock(return_value=None)

        frame = framegrabber.get_current_frame()

        # Should return previous frame
        assert frame is not None
        assert np.array_equal(frame, framegrabber._current_frame)

    def test_publish_workspace_image(self, mock_environment, mock_redis_streamer):
        """Test publishing workspace image to Redis"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        test_image = np.zeros((400, 400, 3), dtype=np.uint8)
        robot_pose = {"x": 0.2, "y": 0.0, "z": 0.3}

        stream_id = framegrabber.publish_workspace_image(test_image, "test_workspace", robot_pose)

        assert stream_id == "stream_id_123"
        assert framegrabber.frame_counter == 1

        # Check publish_image was called with correct parameters
        streamer_instance = framegrabber.streamer
        streamer_instance.publish_image.assert_called_once()

        call_args = streamer_instance.publish_image.call_args
        assert np.array_equal(call_args[0][0], test_image)
        assert call_args[1]["metadata"]["workspace_id"] == "test_workspace"
        assert call_args[1]["compress_jpeg"] is True

    def test_publish_workspace_image_increments_counter(self, mock_environment, mock_redis_streamer):
        """Test that frame counter increments"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        test_image = np.zeros((400, 400, 3), dtype=np.uint8)

        assert framegrabber.frame_counter == 0
        framegrabber.publish_workspace_image(test_image, "ws1")
        assert framegrabber.frame_counter == 1
        framegrabber.publish_workspace_image(test_image, "ws2")
        assert framegrabber.frame_counter == 2

    def test_is_point_visible_basic(self, mock_environment, mock_redis_streamer):
        """Test basic point visibility check"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        # Mock robot pose
        mock_pose = PoseObjectPNP(0.2, 0.0, 0.3, 0.0, 1.57, 0.0)
        robot_ctrl = mock_environment.get_robot_controller()
        robot_ctrl.robot_ctrl().get_pose.return_value = mock_pose

        point = np.array([0.25, 0.0, 0.01])

        # This is a complex calculation, just test it doesn't crash
        # Actual visibility depends on camera pose and intrinsics
        result = framegrabber.is_point_visible(point)

        assert isinstance(result, bool)

    def test_is_point_visible_behind_camera(self, mock_environment, mock_redis_streamer):
        """Test that point behind camera is not visible"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        mock_pose = PoseObjectPNP(0.2, 0.0, 0.3, 0.0, 1.57, 0.0)
        robot_ctrl = mock_environment.get_robot_controller()
        robot_ctrl.robot_ctrl().get_pose.return_value = mock_pose

        # Point far behind camera
        point = np.array([-10.0, 0.0, 0.0])

        result = framegrabber.is_point_visible(point)

        # FIX: Use == instead of is for NumPy boolean compatibility
        assert not result
        # Alternative: assert not result

    def test_camera_matrix_property(self, mock_environment, mock_redis_streamer):
        """Test camera_matrix property"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        mtx = framegrabber.camera_matrix()

        assert isinstance(mtx, np.ndarray)
        assert mtx.shape == (3, 3)

    def test_camera_dist_coeff_property(self, mock_environment, mock_redis_streamer):
        """Test camera_dist_coeff property"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        dist = framegrabber.camera_dist_coeff()

        assert isinstance(dist, np.ndarray)
        assert len(dist) == 5

    def test_verbose_property(self, mock_environment, mock_redis_streamer):
        """Test verbose property"""
        framegrabber = NiryoFrameGrabber(mock_environment, verbose=True)

        assert framegrabber.verbose() is True

    def test_environment_property(self, mock_environment, mock_redis_streamer):
        """Test environment property"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        assert framegrabber.environment() == mock_environment

    @patch("robot_environment.camera.niryo_framegrabber.cv2")
    def test_is_point_visible_with_custom_transform(self, mock_cv2, mock_environment, mock_redis_streamer):
        """Test point visibility with custom camera-to-gripper transform"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        mock_pose = PoseObjectPNP(0.2, 0.0, 0.3, 0.0, 1.57, 0.0)
        robot_ctrl = mock_environment.get_robot_controller()
        robot_ctrl.robot_ctrl().get_pose.return_value = mock_pose

        point = np.array([0.25, 0.0, 0.01])
        custom_transform = np.eye(4)
        custom_transform[0, 3] = 0.05  # 5cm offset

        # Mock undistortPoints
        mock_cv2.undistortPoints.return_value = np.array([[[320, 240]]], dtype=np.float32)

        result = framegrabber.is_point_visible(point, custom_transform)

        assert isinstance(result, bool)

    def test_get_current_frame_shape(self, mock_environment, mock_redis_streamer):
        """Test getting current frame shape"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        # Set a frame
        framegrabber._current_frame = np.zeros((400, 400, 3), dtype=np.uint8)

        shape = framegrabber.get_current_frame_shape()

        assert shape == (400, 400, 3)

    def test_get_current_frame_width_height(self, mock_environment, mock_redis_streamer):
        """Test getting frame width and height"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        framegrabber._current_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        width, height = framegrabber.get_current_frame_width_height()

        assert width == 480
        assert height == 640


class TestNiryoFrameGrabberIntegration:
    """Integration tests for NiryoFrameGrabber"""

    @patch("robot_environment.camera.niryo_framegrabber.uncompress_image")
    @patch("robot_environment.camera.niryo_framegrabber.undistort_image")
    @patch("robot_environment.camera.niryo_framegrabber.extract_img_workspace")
    @patch("robot_environment.camera.niryo_framegrabber.cv2")
    def test_complete_frame_acquisition_pipeline(
        self, mock_cv2, mock_extract, mock_undistort, mock_uncompress, mock_environment, mock_redis_streamer
    ):
        """Test complete frame acquisition pipeline"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        # Setup complete pipeline
        raw_img = np.zeros((480, 640, 3), dtype=np.uint8)
        undistorted_img = np.zeros((480, 640, 3), dtype=np.uint8)
        workspace_img = np.zeros((400, 400, 3), dtype=np.uint8)
        rgb_img = np.zeros((400, 400, 3), dtype=np.uint8)

        mock_uncompress.return_value = raw_img
        mock_undistort.return_value = undistorted_img
        mock_extract.return_value = workspace_img
        mock_cv2.cvtColor.return_value = rgb_img

        mock_environment.get_robot_pose.return_value = PoseObjectPNP(0.2, 0.0, 0.3)
        mock_workspace = Mock()
        mock_workspace.set_img_shape = Mock()
        mock_environment.get_visible_workspace.return_value = mock_workspace

        frame = framegrabber.get_current_frame()

        # Verify pipeline executed
        assert frame is not None
        mock_uncompress.assert_called_once()
        mock_undistort.assert_called_once()
        mock_extract.assert_called_once()
        mock_cv2.cvtColor.assert_called_once()
        mock_workspace.set_img_shape.assert_called_once()

    def test_multiple_frame_captures(self, mock_environment, mock_redis_streamer):
        """Test capturing multiple frames"""
        framegrabber = NiryoFrameGrabber(mock_environment)

        test_image = np.zeros((400, 400, 3), dtype=np.uint8)

        # Capture multiple frames
        for i in range(5):
            framegrabber.publish_workspace_image(test_image, f"ws_{i}")

        assert framegrabber.frame_counter == 5
