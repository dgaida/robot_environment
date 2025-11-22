"""
Unit tests for Workspace classes
"""

import pytest
from unittest.mock import Mock
from robot_environment.workspaces.niryo_workspace import NiryoWorkspace
from robot_environment.workspaces.workspaces import Workspaces
from robot_environment.workspaces.niryo_workspaces import NiryoWorkspaces
from robot_environment.objects.pose_object import PoseObjectPNP


@pytest.fixture
def mock_environment():
    """Create a mock environment"""
    env = Mock()
    env.use_simulation.return_value = False
    env.verbose.return_value = False

    # Mock robot target pose method
    def mock_get_target_pose(ws_id, u_rel, v_rel, yaw):
        x = 0.1 + u_rel * 0.3
        y = -0.15 + v_rel * 0.3
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    env.get_robot_target_pose_from_rel = mock_get_target_pose

    return env


class TestNiryoWorkspace:
    """Test suite for NiryoWorkspace class"""

    def test_initialization(self, mock_environment):
        """Test workspace initialization"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment)

        assert workspace.id() == "niryo_ws"
        assert workspace.environment() == mock_environment

    def test_observation_pose_niryo_ws(self, mock_environment):
        """Test observation pose for niryo_ws"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment)
        pose = workspace.observation_pose()

        assert isinstance(pose, PoseObjectPNP)
        assert pose.x is not None

    def test_observation_pose_gazebo(self, mock_environment):
        """Test observation pose for gazebo_1"""
        workspace = NiryoWorkspace("gazebo_1", mock_environment)
        pose = workspace.observation_pose()

        assert isinstance(pose, PoseObjectPNP)
        assert pose.x is not None

    def test_observation_pose_unknown(self, mock_environment):
        """Test observation pose for unknown workspace"""
        workspace = NiryoWorkspace("unknown_ws", mock_environment)
        pose = workspace.observation_pose()

        assert pose is None

    def test_transform_camera2world_coords(self, mock_environment):
        """Test camera to world coordinate transformation"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment)

        pose = workspace.transform_camera2world_coords("niryo_ws", 0.5, 0.5, 0.0)

        assert isinstance(pose, PoseObjectPNP)
        assert pose.x is not None
        assert pose.y is not None

    def test_corners_of_workspace(self, mock_environment):
        """Test that all four corners are set"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment)

        assert workspace.xy_ul_wc() is not None
        assert workspace.xy_ll_wc() is not None
        assert workspace.xy_ur_wc() is not None
        assert workspace.xy_lr_wc() is not None

    def test_width_height(self, mock_environment):
        """Test width and height calculation"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment)

        width = workspace.width_m()
        height = workspace.height_m()

        assert width > 0
        assert height > 0

    def test_center_of_workspace(self, mock_environment):
        """Test center calculation"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment)
        center = workspace.xy_center_wc()

        assert isinstance(center, PoseObjectPNP)

    def test_is_visible(self, mock_environment):
        """Test visibility check"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment)

        # Get observation pose and check if visible from there
        obs_pose = workspace.observation_pose()
        is_visible = workspace.is_visible(obs_pose)

        # Should be visible from observation pose
        assert is_visible is True

    def test_is_not_visible(self, mock_environment):
        """Test visibility check with different pose"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment)

        # Random pose far from observation pose
        random_pose = PoseObjectPNP(10.0, 10.0, 10.0, 0.0, 0.0, 0.0)
        is_visible = workspace.is_visible(random_pose)

        assert is_visible is False

    def test_set_img_shape(self, mock_environment):
        """Test setting image shape"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment)

        workspace.set_img_shape((640, 480, 3))
        shape = workspace.img_shape()

        assert shape == (640, 480, 3)

    def test_str_representation(self, mock_environment):
        """Test string representation"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment)
        str_repr = str(workspace)

        assert "niryo_ws" in str_repr
        assert "Workspace" in str_repr

    def test_repr_equals_str(self, mock_environment):
        """Test that repr equals str"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment)
        assert repr(workspace) == str(workspace)

    def test_verbose_property(self, mock_environment):
        """Test verbose property"""
        workspace = NiryoWorkspace("niryo_ws", mock_environment, verbose=True)
        assert workspace.verbose() is True


class TestWorkspaces:
    """Test suite for Workspaces collection class"""

    def test_initialization(self, mock_environment):
        """Test workspaces collection initialization"""
        workspaces = Workspaces(mock_environment)

        assert len(workspaces) == 0
        assert workspaces.environment() == mock_environment

    def test_append_workspace(self, mock_environment):
        """Test appending workspace"""
        workspaces = Workspaces(mock_environment)
        ws = NiryoWorkspace("test_ws", mock_environment)

        workspaces.append_workspace(ws)

        assert len(workspaces) == 1
        assert workspaces[0] == ws

    def test_get_workspace(self, mock_environment):
        """Test getting workspace by index"""
        workspaces = Workspaces(mock_environment)
        ws1 = NiryoWorkspace("ws1", mock_environment)
        ws2 = NiryoWorkspace("ws2", mock_environment)

        workspaces.append_workspace(ws1)
        workspaces.append_workspace(ws2)

        assert workspaces.get_workspace(0) == ws1
        assert workspaces.get_workspace(1) == ws2

    def test_get_workspace_by_id(self, mock_environment):
        """Test getting workspace by ID"""
        workspaces = Workspaces(mock_environment)
        ws1 = NiryoWorkspace("ws1", mock_environment)
        ws2 = NiryoWorkspace("ws2", mock_environment)

        workspaces.append_workspace(ws1)
        workspaces.append_workspace(ws2)

        assert workspaces.get_workspace_by_id("ws1") == ws1
        assert workspaces.get_workspace_by_id("ws2") == ws2
        assert workspaces.get_workspace_by_id("nonexistent") is None

    def test_get_workspace_ids(self, mock_environment):
        """Test getting all workspace IDs"""
        workspaces = Workspaces(mock_environment)
        ws1 = NiryoWorkspace("ws1", mock_environment)
        ws2 = NiryoWorkspace("ws2", mock_environment)

        workspaces.append_workspace(ws1)
        workspaces.append_workspace(ws2)

        ids = workspaces.get_workspace_ids()

        assert len(ids) == 2
        assert "ws1" in ids
        assert "ws2" in ids

    def test_get_workspace_id(self, mock_environment):
        """Test getting workspace ID by index"""
        workspaces = Workspaces(mock_environment)
        ws = NiryoWorkspace("test_ws", mock_environment)

        workspaces.append_workspace(ws)

        assert workspaces.get_workspace_id(0) == "test_ws"

    def test_get_workspace_home_id(self, mock_environment):
        """Test getting home workspace ID"""
        workspaces = Workspaces(mock_environment)
        ws = NiryoWorkspace("home_ws", mock_environment)

        workspaces.append_workspace(ws)

        assert workspaces.get_workspace_home_id() == "home_ws"

    def test_get_home_workspace(self, mock_environment):
        """Test getting home workspace"""
        workspaces = Workspaces(mock_environment)
        ws = NiryoWorkspace("home_ws", mock_environment)

        workspaces.append_workspace(ws)

        assert workspaces.get_home_workspace() == ws

    def test_get_observation_pose(self, mock_environment):
        """Test getting observation pose by workspace ID"""
        workspaces = Workspaces(mock_environment)
        ws = NiryoWorkspace("niryo_ws", mock_environment)

        workspaces.append_workspace(ws)

        pose = workspaces.get_observation_pose("niryo_ws")

        assert isinstance(pose, PoseObjectPNP)

    def test_get_width_height_m(self, mock_environment):
        """Test getting workspace dimensions"""
        workspaces = Workspaces(mock_environment)
        ws = NiryoWorkspace("niryo_ws", mock_environment)

        workspaces.append_workspace(ws)

        width, height = workspaces.get_width_height_m("niryo_ws")

        assert width > 0
        assert height > 0

    def test_get_img_shape(self, mock_environment):
        """Test getting image shape"""
        workspaces = Workspaces(mock_environment)
        ws = NiryoWorkspace("niryo_ws", mock_environment)
        ws.set_img_shape((640, 480, 3))

        workspaces.append_workspace(ws)

        shape = workspaces.get_img_shape("niryo_ws")

        assert shape == (640, 480, 3)

    def test_get_visible_workspace(self, mock_environment):
        """Test getting visible workspace"""
        workspaces = Workspaces(mock_environment)
        ws1 = NiryoWorkspace("niryo_ws", mock_environment)
        ws2 = NiryoWorkspace("gazebo_1", mock_environment)

        workspaces.append_workspace(ws1)
        workspaces.append_workspace(ws2)

        # Use observation pose of first workspace
        obs_pose = ws1.observation_pose()
        visible = workspaces.get_visible_workspace(obs_pose)

        assert visible == ws1

    def test_get_visible_workspace_none(self, mock_environment):
        """Test when no workspace is visible"""
        workspaces = Workspaces(mock_environment)
        ws = NiryoWorkspace("niryo_ws", mock_environment)

        workspaces.append_workspace(ws)

        # Use a pose that's not the observation pose
        random_pose = PoseObjectPNP(10.0, 10.0, 10.0, 0.0, 0.0, 0.0)
        visible = workspaces.get_visible_workspace(random_pose)

        assert visible is None

    def test_verbose_property(self, mock_environment):
        """Test verbose property"""
        workspaces = Workspaces(mock_environment, verbose=True)
        assert workspaces.verbose() is True


class TestNiryoWorkspaces:
    """Test suite for NiryoWorkspaces class"""

    def test_initialization_real_robot(self, mock_environment):
        """Test initialization with real robot"""
        mock_environment.use_simulation.return_value = False

        workspaces = NiryoWorkspaces(mock_environment)

        assert len(workspaces) == 1
        assert workspaces[0].id() == "niryo_ws2"

    def test_initialization_simulation(self, mock_environment):
        """Test initialization with simulation"""
        mock_environment.use_simulation.return_value = True

        workspaces = NiryoWorkspaces(mock_environment)

        assert len(workspaces) == 1
        assert workspaces[0].id() == "gazebo_1"

    def test_inherits_from_workspaces(self, mock_environment):
        """Test that NiryoWorkspaces inherits from Workspaces"""
        workspaces = NiryoWorkspaces(mock_environment)

        assert isinstance(workspaces, Workspaces)

    def test_can_add_more_workspaces(self, mock_environment):
        """Test that additional workspaces can be added"""
        workspaces = NiryoWorkspaces(mock_environment)

        # Add another workspace
        ws = NiryoWorkspace("custom_ws", mock_environment)
        workspaces.append_workspace(ws)

        assert len(workspaces) == 2
