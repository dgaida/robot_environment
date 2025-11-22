"""
Unit tests for PoseObjectPNP class
"""

import pytest
import numpy as np
import math
from robot_environment.objects.pose_object import PoseObjectPNP


class TestPoseObjectPNP:
    """Test suite for PoseObjectPNP class"""

    def test_initialization(self):
        """Test basic initialization"""
        pose = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.roll == 0.1
        assert pose.pitch == 0.2
        assert pose.yaw == 0.3

    def test_default_initialization(self):
        """Test initialization with default values"""
        pose = PoseObjectPNP()
        assert pose.x == 0.0
        assert pose.y == 0.0
        assert pose.z == 0.0
        assert pose.roll == 0.0
        assert pose.pitch == 0.0
        assert pose.yaw == 0.0

    def test_addition(self):
        """Test pose addition"""
        pose1 = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        pose2 = PoseObjectPNP(0.5, 0.5, 0.5, 0.05, 0.05, 0.05)
        result = pose1 + pose2

        assert result.x == 1.5
        assert result.y == 2.5
        assert result.z == 3.5
        assert result.roll == 0.15
        assert result.pitch == 0.25
        assert result.yaw == 0.35

    def test_subtraction(self):
        """Test pose subtraction"""
        pose1 = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        pose2 = PoseObjectPNP(0.5, 0.5, 0.5, 0.05, 0.05, 0.05)
        result = pose1 - pose2

        assert result.x == 0.5
        assert result.y == 1.5
        assert result.z == 2.5
        assert result.roll == pytest.approx(0.05)
        assert result.pitch == pytest.approx(0.15)
        assert result.yaw == pytest.approx(0.25)

    def test_equality(self):
        """Test pose equality"""
        pose1 = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        pose2 = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        pose3 = PoseObjectPNP(1.1, 2.0, 3.0, 0.1, 0.2, 0.3)

        assert pose1 == pose2
        assert pose1 != pose3

    def test_approx_eq(self):
        """Test approximate equality"""
        pose1 = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        pose2 = PoseObjectPNP(1.05, 2.05, 3.05, 0.12, 0.22, 0.32)

        # Should be equal with tolerance 0.1
        assert pose1.approx_eq(pose2, eps_position=0.1, eps_orientation=0.1)

        # Should not be equal with tight tolerance
        assert not pose1.approx_eq(pose2, eps_position=0.01, eps_orientation=0.01)

    def test_approx_eq_xyz(self):
        """Test approximate xyz equality"""
        pose1 = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        pose2 = PoseObjectPNP(1.05, 2.05, 3.05, 0.5, 0.6, 0.7)

        # Should be equal in position only
        assert pose1.approx_eq_xyz(pose2, eps=0.1)
        assert not pose1.approx_eq_xyz(pose2, eps=0.01)

    def test_copy_with_offsets(self):
        """Test copying with offsets"""
        pose = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        new_pose = pose.copy_with_offsets(x_offset=0.5, y_offset=0.5, z_offset=0.5)

        assert new_pose.x == 1.5
        assert new_pose.y == 2.5
        assert new_pose.z == 3.5
        assert new_pose.roll == 0.1
        assert new_pose.pitch == 0.2
        assert new_pose.yaw == 0.3

        # Original should be unchanged
        assert pose.x == 1.0

    def test_to_list(self):
        """Test conversion to list"""
        pose = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        pose_list = pose.to_list()

        assert len(pose_list) == 6
        assert pose_list == [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]

    def test_to_transformation_matrix(self):
        """Test conversion to transformation matrix"""
        pose = PoseObjectPNP(1.0, 2.0, 3.0, 0.0, 0.0, math.pi / 2)
        matrix = pose.to_transformation_matrix()

        # Check matrix shape
        assert matrix.shape == (4, 4)

        # Check translation part
        assert matrix[0, 3] == 1.0
        assert matrix[1, 3] == 2.0
        assert matrix[2, 3] == 3.0

        # Check homogeneous row
        assert np.allclose(matrix[3, :], [0, 0, 0, 1])

        # Check rotation part for 90° yaw rotation
        assert pytest.approx(matrix[0, 0], abs=1e-5) == 0.0
        assert pytest.approx(matrix[1, 1], abs=1e-5) == 0.0

    def test_quaternion_property(self):
        """Test quaternion conversion"""
        pose = PoseObjectPNP(0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2)
        quat = pose.quaternion

        assert len(quat) == 4
        # For 90° yaw rotation, qw should be cos(pi/4) and qz should be sin(pi/4)
        assert pytest.approx(quat[3], abs=1e-5) == math.cos(math.pi / 4)
        assert pytest.approx(quat[2], abs=1e-5) == math.sin(math.pi / 4)

    def test_quaternion_pose_property(self):
        """Test quaternion pose conversion"""
        pose = PoseObjectPNP(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
        quat_pose = pose.quaternion_pose

        assert len(quat_pose) == 7
        assert quat_pose[0] == 1.0
        assert quat_pose[1] == 2.0
        assert quat_pose[2] == 3.0

    def test_euler_to_quaternion(self):
        """Test Euler to quaternion conversion"""
        quat = PoseObjectPNP.euler_to_quaternion(0.0, 0.0, 0.0)

        # Identity rotation
        assert len(quat) == 4
        assert pytest.approx(quat[0]) == 0.0
        assert pytest.approx(quat[1]) == 0.0
        assert pytest.approx(quat[2]) == 0.0
        assert pytest.approx(quat[3]) == 1.0

    def test_quaternion_to_euler_angle(self):
        """Test quaternion to Euler conversion"""
        # Identity quaternion
        roll, pitch, yaw = PoseObjectPNP.quaternion_to_euler_angle(0, 0, 0, 1)

        assert pytest.approx(roll) == 0.0
        assert pytest.approx(pitch) == 0.0
        assert pytest.approx(yaw) == 0.0

    def test_quaternion_roundtrip(self):
        """Test Euler -> Quaternion -> Euler conversion"""
        original_roll = 0.1
        original_pitch = 0.2
        original_yaw = 0.3

        # Convert to quaternion
        quat = PoseObjectPNP.euler_to_quaternion(original_roll, original_pitch, original_yaw)

        # Convert back to Euler
        roll, pitch, yaw = PoseObjectPNP.quaternion_to_euler_angle(*quat)

        assert pytest.approx(roll) == original_roll
        assert pytest.approx(pitch) == original_pitch
        assert pytest.approx(yaw) == original_yaw

    def test_xy_coordinate(self):
        """Test xy_coordinate property"""
        pose = PoseObjectPNP(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
        xy = pose.xy_coordinate()

        assert len(xy) == 2
        assert xy[0] == 1.0
        assert xy[1] == 2.0

    def test_str_representation(self):
        """Test string representation"""
        pose = PoseObjectPNP(1.234, 2.345, 3.456, 0.1, 0.2, 0.3)
        str_repr = str(pose)

        assert "1.23" in str_repr
        assert "2.35" in str_repr
        assert "3.46" in str_repr

    def test_repr_equals_str(self):
        """Test that repr equals str"""
        pose = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        assert repr(pose) == str(pose)
