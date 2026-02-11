"""
Unit tests for workspace_utils.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from robot_environment.utils.workspace_utils import calculate_largest_free_space
from robot_workspace import Objects, Object, PoseObjectPNP


@pytest.fixture
def mock_workspace():
    """Create a mock workspace"""
    workspace = Mock()

    def create_mock_pose(x, y):
        pose = Mock()
        pose.x = x
        pose.y = y
        return pose

    workspace.xy_ul_wc.return_value = create_mock_pose(0.4, 0.15)
    workspace.xy_lr_wc.return_value = create_mock_pose(0.1, -0.15)
    return workspace


def create_mock_object(x, y, width=0.05, height=0.05):
    """Helper to create a properly mocked object"""
    obj = Mock(spec=Object)
    obj.x_com.return_value = x
    obj.y_com.return_value = y
    obj.width_m.return_value = width
    obj.height_m.return_value = height
    return obj


class TestWorkspaceUtils:
    """Test suite for workspace utilities"""

    def test_calculate_largest_free_space_empty(self, mock_workspace):
        """Test with no objects"""
        detected_objects = Objects()

        area, cx, cy = calculate_largest_free_space(mock_workspace, detected_objects)

        assert area > 0
        # Center of 0.1-0.4 (x) is 0.25, -0.15-0.15 (y) is 0.0
        assert cx == pytest.approx(0.25, abs=0.01)
        assert cy == pytest.approx(0.0, abs=0.01)

    def test_calculate_largest_free_space_with_objects(self, mock_workspace):
        """Test with multiple objects to exercise lines 65-82"""
        # Place objects in corners
        obj1 = create_mock_object(0.35, 0.1)  # Top left-ish
        obj2 = create_mock_object(0.15, -0.1) # Bottom right-ish

        detected_objects = Objects([obj1, obj2])

        area, cx, cy = calculate_largest_free_space(mock_workspace, detected_objects)

        assert area > 0
        assert area < (0.3 * 0.3) # Total area is 0.3 * 0.3

    @patch("robot_environment.utils.workspace_utils.cv2")
    def test_calculate_largest_free_space_visualize(self, mock_cv2, mock_workspace):
        """Test visualization branch"""
        detected_objects = Objects()

        area, cx, cy = calculate_largest_free_space(mock_workspace, detected_objects, visualize=True)

        assert mock_cv2.imshow.called
        assert mock_cv2.waitKey.called

    def test_calculate_largest_free_space_with_logger(self, mock_workspace):
        """Test with explicit logger"""
        logger = Mock()
        detected_objects = Objects([create_mock_object(0.25, 0.0)])

        calculate_largest_free_space(mock_workspace, detected_objects, logger=logger)

        assert logger.debug.called
        assert logger.info.called

    @patch("robot_environment.utils.workspace_utils.cv2")
    def test_calculate_largest_free_space_visualize_error(self, mock_cv2, mock_workspace):
        """Test visualization error handling (lines 104-105)"""
        mock_cv2.imshow.side_effect = Exception("Display error")
        logger = Mock()
        detected_objects = Objects()

        calculate_largest_free_space(mock_workspace, detected_objects, visualize=True, logger=logger)

        assert logger.warning.called
        assert "Could not visualize" in logger.warning.call_args[0][0]
