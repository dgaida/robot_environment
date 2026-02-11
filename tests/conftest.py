"""
Pytest configuration and shared fixtures for robot_environment tests
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Mock text2speech to avoid import errors if not properly installed
mock_t2s = MagicMock()
sys.modules["text2speech"] = mock_t2s
sys.modules["text2speech.text2speech"] = mock_t2s
sys.modules["text2speech.engines"] = mock_t2s

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_mock_workspace(workspace_id="test_ws"):
    """Helper to create a properly mocked workspace"""
    workspace = Mock()
    workspace.id.return_value = workspace_id
    workspace.img_shape.return_value = (640, 480, 3)
    workspace.set_img_shape = Mock()

    # Mock workspace corners
    # from robot_workspace import PoseObjectPNP

    def create_mock_pose(x, y, z=0.0):
        mock_pose = Mock()
        mock_pose.xy_coordinate.return_value = [x, y]
        mock_pose.x = x
        mock_pose.y = y
        mock_pose.z = z
        return mock_pose

    workspace.xy_ul_wc.return_value = create_mock_pose(0.4, 0.15, 0.0)
    workspace.xy_ur_wc.return_value = create_mock_pose(0.4, -0.15, 0.0)
    workspace.xy_ll_wc.return_value = create_mock_pose(0.1, 0.15, 0.0)
    workspace.xy_lr_wc.return_value = create_mock_pose(0.1, -0.15, 0.0)
    workspace.xy_center_wc.return_value = create_mock_pose(0.25, 0.0, 0.0)

    return workspace


@pytest.fixture
def mock_workspaces():
    """Create a properly mocked iterable NiryoWorkspaces"""
    from robot_workspace import PoseObjectPNP

    # Create mock workspaces
    ws1 = create_mock_workspace("niryo_ws_left")
    ws2 = create_mock_workspace("niryo_ws_right")

    mock_ws_collection = Mock()

    # CRITICAL FIX: Make the mock iterable
    mock_ws_collection.__iter__ = Mock(return_value=iter([ws1, ws2]))

    # Set up all the methods
    mock_ws_collection.get_workspace = Mock(return_value=ws1)
    mock_ws_collection.get_workspace_by_id = Mock(side_effect=lambda id: ws1 if "left" in id else ws2)
    mock_ws_collection.get_workspace_home_id.return_value = "niryo_ws_left"
    mock_ws_collection.get_workspace_left_id.return_value = "niryo_ws_left"
    mock_ws_collection.get_workspace_right_id.return_value = "niryo_ws_right"
    mock_ws_collection.get_workspace_ids.return_value = ["niryo_ws_left", "niryo_ws_right"]
    mock_ws_collection.get_observation_pose.return_value = PoseObjectPNP(0.2, 0.0, 0.3)
    mock_ws_collection.get_visible_workspace.return_value = ws1
    mock_ws_collection.get_home_workspace.return_value = ws1

    return mock_ws_collection


@pytest.fixture
def mock_dependencies():
    """Mock all Environment dependencies with iterable workspaces"""
    from robot_environment.robot.niryo_robot_controller import NiryoRobotController
    from robot_workspace import PoseObjectPNP
    from unittest.mock import create_autospec, patch
    import numpy as np

    with patch("robot_environment.environment.Robot") as mock_robot, patch(
        "robot_environment.environment.NiryoFrameGrabber"
    ) as mock_fg, patch("robot_environment.environment.NiryoWorkspaces") as mock_ws, patch(
        "robot_environment.environment.Text2Speech"
    ) as mock_tts, patch(
        "robot_environment.environment.VisualCortex"
    ) as mock_vc, patch(
        "robot_environment.environment.get_default_config"
    ) as mock_config:

        # Setup robot controller
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

        # CRITICAL FIX: Create iterable workspaces
        ws1 = create_mock_workspace("niryo_ws_left")
        ws2 = create_mock_workspace("niryo_ws_right")

        mock_ws_instance = Mock()
        # Make it iterable
        mock_ws_instance.__iter__ = Mock(return_value=iter([ws1, ws2]))

        mock_ws_instance.get_workspace = Mock(return_value=ws1)
        mock_ws_instance.get_workspace_by_id = Mock(side_effect=lambda id: ws1 if "left" in id else ws2)
        mock_ws_instance.get_workspace_home_id.return_value = "niryo_ws_left"
        mock_ws_instance.get_workspace_id = Mock(return_value="niryo_ws_left")
        mock_ws_instance.get_observation_pose.return_value = PoseObjectPNP(0.2, 0.0, 0.3)
        mock_ws_instance.get_visible_workspace.return_value = ws1
        mock_ws_instance.get_home_workspace.return_value = ws1
        mock_ws.return_value = mock_ws_instance

        # Setup TTS
        mock_tts_instance = Mock()
        mock_tts_instance.call_text2speech_async.return_value = Mock()
        mock_tts.return_value = mock_tts_instance

        # Setup VisualCortex
        mock_vc_instance = Mock()
        mock_vc_instance.get_detected_objects.return_value = []
        mock_vc_instance.get_object_labels.return_value = [["pencil", "pen", "eraser"]]
        mock_vc_instance.add_object_name2object_labels = Mock()
        mock_vc_instance.get_annotated_image.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_vc_instance.detect_objects_from_redis = Mock()
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


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory"""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_image():
    """Provide a sample test image"""
    import numpy as np

    # Create a simple test image (640x480, RGB)
    return np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Provide a sample segmentation mask"""
    import numpy as np

    # Create a simple binary mask
    mask = np.zeros((640, 480), dtype=np.uint8)
    mask[200:400, 200:400] = 255
    return mask


# Mark configuration
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_robot: marks tests that require real robot hardware")
    config.addinivalue_line("markers", "requires_redis: marks tests that require Redis server")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers"""
    skip_slow = pytest.mark.skip(reason="slow test, use -m slow to run")
    skip_integration = pytest.mark.skip(reason="integration test")
    skip_robot = pytest.mark.skip(reason="requires robot hardware")
    skip_redis = pytest.mark.skip(reason="requires Redis server")

    for item in items:
        if "slow" in item.keywords and not config.getoption("-m") == "slow":
            item.add_marker(skip_slow)
        if "integration" in item.keywords:
            # Skip integration tests by default unless explicitly requested
            if not config.getoption("-m") or "integration" not in config.getoption("-m"):
                item.add_marker(skip_integration)
        if "requires_robot" in item.keywords:
            item.add_marker(skip_robot)
        if "requires_redis" in item.keywords:
            item.add_marker(skip_redis)


@pytest.fixture
def mock_redis():
    """Provide a mock Redis client"""
    redis_mock = Mock()
    redis_mock.ping.return_value = True
    return redis_mock


@pytest.fixture
def clean_environment():
    """Ensure clean test environment"""
    import os
    import tempfile
    import shutil

    # Create temporary directory for test outputs
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()

    yield temp_dir

    # Cleanup
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir, ignore_errors=True)
