"""
Pytest configuration and shared fixtures for robot_environment tests
"""
import pytest
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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


# Mark configuration for slow tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_robot: marks tests that require real robot hardware"
    )
    config.addinivalue_line(
        "markers", "requires_redis: marks tests that require Redis server"
    )


# Skip tests based on markers
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
    from unittest.mock import Mock
    redis_mock = Mock()
    redis_mock.ping.return_value = True
    return redis_mock


@pytest.fixture
def clean_environment():
    """Ensure clean test environment"""
    import os
    import tempfile
    
    # Create temporary directory for test outputs
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    
    yield temp_dir
    
    # Cleanup
    os.chdir(original_cwd)
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
