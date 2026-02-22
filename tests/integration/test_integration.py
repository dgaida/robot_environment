"""
Integration tests for robot_environment package
"""

import pytest
import numpy as np
from unittest.mock import Mock
from robot_workspace import Object
from robot_workspace import Objects
from robot_workspace import PoseObjectPNP
from robot_workspace import NiryoWorkspace
from robot_workspace import Location
from robot_workspace.config import WorkspaceConfig, PoseConfig


@pytest.fixture
def integrated_workspace():
    """Create an integrated workspace with mock environment"""
    mock_env = Mock()
    mock_env.use_simulation.return_value = True
    mock_env.verbose.return_value = False

    def mock_get_target_pose(ws_id, u_rel, v_rel, yaw):
        # Realistic Niryo coordinate mapping:
        # u_rel is horizontal (y-axis in world, left is positive)
        # v_rel is vertical (x-axis in world, away is positive)
        # Top-left of image (0,0) -> Furthest Left (max X, max Y)
        # Bottom-right (1,1) -> Closest Right (min X, min Y)
        x = 0.4 - v_rel * 0.3  # Range [0.1, 0.4]
        y = 0.15 - u_rel * 0.3  # Range [-0.15, 0.15]
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)

    mock_env.get_robot_target_pose_from_rel = mock_get_target_pose

    # Create configuration for the workspace
    pose = PoseConfig(x=0.2, y=0.0, z=0.3, roll=0.0, pitch=1.57, yaw=0.0)
    config = WorkspaceConfig(id="test_ws", observation_pose=pose)

    workspace = NiryoWorkspace.from_config(config, mock_env)
    workspace.set_img_shape((640, 480, 3))

    return workspace


@pytest.fixture
def sample_scene(integrated_workspace):
    """Create a sample scene with multiple objects"""
    objects = Objects()

    # Add various objects at different positions
    obj1 = Object("pencil", 100, 100, 150, 180, None, integrated_workspace)
    obj2 = Object("pen", 300, 200, 350, 280, None, integrated_workspace)
    obj3 = Object("eraser", 200, 400, 250, 450, None, integrated_workspace)
    obj4 = Object("ruler", 400, 100, 500, 120, None, integrated_workspace)

    objects.append(obj1)
    objects.append(obj2)
    objects.append(obj3)
    objects.append(obj4)

    return objects, integrated_workspace


@pytest.mark.integration
class TestObjectWorkspaceIntegration:
    """Integration tests for Object and Workspace interaction"""

    def test_object_position_in_workspace(self, integrated_workspace):
        """Test that object positions are correctly mapped to workspace"""
        obj = Object("test", 320, 240, 400, 300, None, integrated_workspace)

        # Object should have valid world coordinates
        x, y = obj.xy_com()
        assert 0.0 < x < 1.0
        assert -1.0 < y < 1.0

    def test_multiple_objects_different_positions(self, sample_scene):
        """Test multiple objects have different positions"""
        objects, workspace = sample_scene

        positions = [obj.xy_com() for obj in objects]

        # All positions should be unique
        assert len(positions) == len(set(positions))

    def test_object_dimensions_consistency(self, integrated_workspace):
        """Test that object dimensions are consistent"""
        obj = Object("test", 100, 100, 200, 200, None, integrated_workspace)

        width, height = obj.shape_m()
        size = obj.size_m2()

        # Size should approximately equal width * height
        assert abs(size - (width * height)) < 0.001


@pytest.mark.integration
class TestObjectsCollectionOperations:
    """Integration tests for Objects collection operations"""

    def test_spatial_queries(self, sample_scene):
        """Test spatial query operations on object collection"""
        objects, workspace = sample_scene

        # Get first object's position
        ref_obj = objects[0]
        ref_x, ref_y = ref_obj.xy_com()

        # Find objects left of reference
        left_objs = objects.get_detected_objects(location=Location.LEFT_NEXT_TO, coordinate=[ref_x, ref_y])

        # All returned objects should be left (higher y)
        for obj in left_objs:
            assert obj.y_com() > ref_y

    def test_size_based_queries(self, sample_scene):
        """Test size-based queries"""
        objects, workspace = sample_scene

        # Get largest and smallest
        largest, max_size = objects.get_largest_detected_object()
        smallest, min_size = objects.get_smallest_detected_object()

        assert max_size >= min_size
        assert largest.size_m2() == max_size
        assert smallest.size_m2() == min_size

    def test_sorting_operations(self, sample_scene):
        """Test sorting operations"""
        objects, workspace = sample_scene

        # Sort ascending
        sorted_asc = objects.get_detected_objects_sorted(ascending=True)
        sizes_asc = [obj.size_m2() for obj in sorted_asc]
        assert sizes_asc == sorted(sizes_asc)

        # Sort descending
        sorted_desc = objects.get_detected_objects_sorted(ascending=False)
        sizes_desc = [obj.size_m2() for obj in sorted_desc]
        assert sizes_desc == sorted(sizes_desc, reverse=True)

    def test_nearest_object_query(self, sample_scene):
        """Test finding nearest object"""
        objects, workspace = sample_scene

        # Pick a point
        target = [0.2, 0.0]

        nearest, distance = objects.get_nearest_detected_object(target)

        # Check that this is indeed the nearest
        for obj in objects:
            if obj != nearest:
                obj_distance = np.sqrt((obj.x_com() - target[0]) ** 2 + (obj.y_com() - target[1]) ** 2)
                assert obj_distance >= distance

    def test_label_filtering(self, sample_scene):
        """Test filtering by label"""
        objects, workspace = sample_scene

        # Filter for "pen" (should match "pen" and "pencil")
        pen_objs = objects.get_detected_objects(label="pen")

        assert len(pen_objs) >= 2  # pen and pencil
        assert all("pen" in obj.label() for obj in pen_objs)


@pytest.mark.integration
class TestSerializationIntegration:
    """Integration tests for serialization"""

    def test_object_roundtrip(self, integrated_workspace):
        """Test complete object serialization roundtrip"""
        original = Object("test", 100, 100, 200, 200, None, integrated_workspace)

        # To dict
        original.to_dict()

        # To JSON
        json_str = original.to_json()

        # From JSON back to object
        reconstructed = Object.from_json(json_str, integrated_workspace)

        assert reconstructed.label() == original.label()
        assert abs(reconstructed.x_com() - original.x_com()) < 0.01
        assert abs(reconstructed.y_com() - original.y_com()) < 0.01

    def test_objects_collection_serialization(self, sample_scene):
        """Test serializing entire collection"""
        objects, workspace = sample_scene

        # Convert to dict list
        dict_list = Objects.objects_to_dict_list(objects)

        # Reconstruct
        reconstructed = Objects.dict_list_to_objects(dict_list, workspace)

        assert len(reconstructed) == len(objects)

        # Check labels match
        original_labels = sorted([obj.label() for obj in objects])
        reconstructed_labels = sorted([obj.label() for obj in reconstructed])
        assert original_labels == reconstructed_labels


@pytest.mark.integration
class TestPoseTransformations:
    """Integration tests for pose transformations"""

    def test_pose_arithmetic(self):
        """Test pose arithmetic operations"""
        pose1 = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        pose2 = PoseObjectPNP(0.5, 0.5, 0.5, 0.05, 0.05, 0.05)

        # Addition
        sum_pose = pose1 + pose2
        assert sum_pose.x == 1.5
        assert sum_pose.y == 2.5

        # Subtraction
        diff_pose = pose1 - pose2
        assert diff_pose.x == 0.5
        assert diff_pose.y == 1.5

    def test_pose_transformation_matrix(self):
        """Test transformation matrix operations"""
        pose = PoseObjectPNP(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
        matrix = pose.to_transformation_matrix()

        # Test point transformation
        point = np.array([1, 0, 0, 1])  # Homogeneous coordinates
        transformed = matrix @ point

        # Should translate by (1, 2, 3)
        assert transformed[0] == 2.0  # 1 + 1
        assert transformed[1] == 2.0  # 0 + 2
        assert transformed[2] == 3.0  # 0 + 3

    def test_quaternion_conversion_roundtrip(self):
        """Test Euler <-> Quaternion conversion roundtrip"""
        original_pose = PoseObjectPNP(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)

        # Get quaternion
        quat = original_pose.quaternion

        # Convert back
        roll, pitch, yaw = PoseObjectPNP.quaternion_to_euler_angle(*quat)

        assert abs(roll - 0.1) < 0.0001
        assert abs(pitch - 0.2) < 0.0001
        assert abs(yaw - 0.3) < 0.0001


@pytest.mark.integration
class TestWorkspaceOperations:
    """Integration tests for workspace operations"""

    def test_workspace_coordinate_system(self, integrated_workspace):
        """Test workspace coordinate system consistency"""
        # Get corners
        ul = integrated_workspace.xy_ul_wc()
        ur = integrated_workspace.xy_ur_wc()
        ll = integrated_workspace.xy_ll_wc()
        lr = integrated_workspace.xy_lr_wc()

        # Check spatial relationships
        # Upper left should be "above" lower left (higher x)
        assert ul.x > ll.x
        # Upper right should be "right of" upper left (lower y for Niryo)
        assert ur.y < ul.y
        # Lower right should be "right of" lower left (lower y for Niryo)
        assert lr.y < ll.y

    def test_workspace_dimensions(self, integrated_workspace):
        """Test workspace dimension calculations"""
        width = integrated_workspace.width_m()
        height = integrated_workspace.height_m()

        assert width > 0
        assert height > 0

        # Reasonable workspace size (between 10cm and 1m)
        assert 0.1 < width < 1.0
        assert 0.1 < height < 1.0

    def test_workspace_center(self, integrated_workspace):
        """Test workspace center calculation"""
        center = integrated_workspace.xy_center_wc()
        ul = integrated_workspace.xy_ul_wc()
        lr = integrated_workspace.xy_lr_wc()

        # Center should be between corners
        assert ul.x > center.x > lr.x
        assert ul.y > center.y > lr.y


@pytest.mark.integration
@pytest.mark.slow
class TestComplexScenarios:
    """Complex integration test scenarios"""

    def test_pick_and_place_scenario(self, sample_scene):
        """Test a complete pick and place workflow"""
        objects, workspace = sample_scene

        # Find largest object
        largest, _ = objects.get_largest_detected_object()
        pick_coord = largest.coordinate()

        # Find placement location (center of workspace)
        center = workspace.xy_center_wc()
        place_coord = [center.x, center.y]

        # Verify we have valid coordinates
        assert len(pick_coord) == 2
        assert len(place_coord) == 2

        # Simulate pick and place distance calculation
        distance = np.sqrt((pick_coord[0] - place_coord[0]) ** 2 + (pick_coord[1] - place_coord[1]) ** 2)

        # Should be a reasonable distance
        assert 0.0 < distance < 1.0

    def test_multiple_object_manipulation(self, sample_scene):
        """Test manipulating multiple objects"""
        objects, workspace = sample_scene

        # Sort by size
        sorted_objects = objects.get_detected_objects_sorted()

        # Plan to stack objects from largest to smallest
        stack_location = [0.25, 0.0]
        z_offset = 0.0

        stack_plan = []
        for obj in sorted_objects:
            pick_coord = obj.coordinate()
            place_coord = [stack_location[0], stack_location[1], z_offset]
            stack_plan.append((pick_coord, place_coord))
            z_offset += 0.02  # Stack height

        assert len(stack_plan) == len(objects)

    def test_workspace_scanning(self, integrated_workspace):
        """Test scanning workspace systematically"""
        # Generate grid of sample points
        grid_points = []
        for u in np.linspace(0.1, 0.9, 5):
            for v in np.linspace(0.1, 0.9, 5):
                pose = integrated_workspace.transform_camera2world_coords("test_ws", u, v, 0.0)
                grid_points.append((pose.x, pose.y))

        assert len(grid_points) == 25

        # Check that points cover the workspace
        x_coords = [p[0] for p in grid_points]
        y_coords = [p[1] for p in grid_points]

        assert max(x_coords) > min(x_coords)
        assert max(y_coords) > min(y_coords)
