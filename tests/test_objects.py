"""
Unit tests for Objects class (collection of Object instances)
"""
import pytest
from unittest.mock import Mock
from robot_environment.objects.objects import Objects
from robot_environment.objects.object import Object
from robot_environment.objects.pose_object import PoseObjectPNP
from robot_environment.robot.robot_api import Location


@pytest.fixture
def mock_workspace():
    """Create a mock workspace"""
    workspace = Mock()
    workspace.id.return_value = "test_workspace"
    workspace.img_shape.return_value = (640, 480, 3)
    
    def mock_transform(ws_id, u_rel, v_rel, yaw=0.0):
        x = 0.1 + u_rel * 0.3
        y = -0.15 + v_rel * 0.3
        return PoseObjectPNP(x, y, 0.05, 0.0, 1.57, yaw)
    
    workspace.transform_camera2world_coords = mock_transform
    return workspace


@pytest.fixture
def sample_objects(mock_workspace):
    """Create sample objects for testing"""
    obj1 = Object("pencil", 100, 100, 150, 150, None, mock_workspace)
    obj2 = Object("pen", 200, 200, 250, 250, None, mock_workspace)
    obj3 = Object("eraser", 300, 300, 350, 350, None, mock_workspace)
    return [obj1, obj2, obj3]


class TestObjects:
    """Test suite for Objects collection class"""

    def test_initialization_empty(self):
        """Test empty initialization"""
        objects = Objects()
        assert len(objects) == 0

    def test_initialization_with_iterable(self, sample_objects):
        """Test initialization with iterable"""
        objects = Objects(sample_objects)
        assert len(objects) == 3

    def test_append(self, mock_workspace):
        """Test appending objects"""
        objects = Objects()
        obj = Object("test", 100, 100, 150, 150, None, mock_workspace)
        objects.append(obj)
        
        assert len(objects) == 1
        assert objects[0].label() == "test"

    def test_get_detected_object_by_label(self, sample_objects):
        """Test getting object by label and coordinate"""
        objects = Objects(sample_objects)
        
        # Get object close to first object's position
        obj = objects.get_detected_object([0.2, 0.0], label="pencil")
        
        assert obj is not None
        assert obj.label() == "pencil"

    def test_get_detected_object_not_found(self, sample_objects):
        """Test getting non-existent object"""
        objects = Objects(sample_objects)
        
        # Try to get object that doesn't exist
        obj = objects.get_detected_object([10.0, 10.0], label="nonexistent")
        
        assert obj is None

    def test_get_detected_objects_no_filter(self, sample_objects):
        """Test getting all objects without filter"""
        objects = Objects(sample_objects)
        result = objects.get_detected_objects()
        
        assert len(result) == 3

    def test_get_detected_objects_by_label(self, sample_objects):
        """Test filtering by label"""
        objects = Objects(sample_objects)
        result = objects.get_detected_objects(label="pen")
        
        assert len(result) == 2  # "pen" and "pencil" both contain "pen"

    def test_get_detected_objects_left_next_to(self, sample_objects):
        """Test filtering by LEFT_NEXT_TO location"""
        objects = Objects(sample_objects)
        
        # Objects left of y=0.0
        result = objects.get_detected_objects(
            location=Location.LEFT_NEXT_TO,
            coordinate=[0.2, 0.0]
        )
        
        # Should return objects with y > 0.0
        assert all(obj.y_com() > 0.0 for obj in result)

    def test_get_detected_objects_right_next_to(self, sample_objects):
        """Test filtering by RIGHT_NEXT_TO location"""
        objects = Objects(sample_objects)
        
        result = objects.get_detected_objects(
            location=Location.RIGHT_NEXT_TO,
            coordinate=[0.2, 0.1]
        )
        
        # Should return objects with y < 0.1
        assert all(obj.y_com() < 0.1 for obj in result)

    def test_get_detected_objects_above(self, sample_objects):
        """Test filtering by ABOVE location"""
        objects = Objects(sample_objects)
        
        result = objects.get_detected_objects(
            location=Location.ABOVE,
            coordinate=[0.2, 0.0]
        )
        
        # Should return objects with x > 0.2
        assert all(obj.x_com() > 0.2 for obj in result)

    def test_get_detected_objects_below(self, sample_objects):
        """Test filtering by BELOW location"""
        objects = Objects(sample_objects)
        
        result = objects.get_detected_objects(
            location=Location.BELOW,
            coordinate=[0.3, 0.0]
        )
        
        # Should return objects with x < 0.3
        assert all(obj.x_com() < 0.3 for obj in result)

    def test_get_detected_objects_close_to(self, sample_objects):
        """Test filtering by CLOSE_TO location"""
        objects = Objects(sample_objects)
        obj1 = sample_objects[0]
        
        # Get objects close to first object (within 2cm)
        result = objects.get_detected_objects(
            location=Location.CLOSE_TO,
            coordinate=[obj1.x_com(), obj1.y_com()]
        )
        
        assert len(result) >= 1  # At least the object itself

    def test_get_detected_objects_with_string_location(self, sample_objects):
        """Test filtering with string location"""
        objects = Objects(sample_objects)
        
        result = objects.get_detected_objects(
            location="left next to",
            coordinate=[0.2, 0.0]
        )
        
        assert isinstance(result, Objects)

    def test_get_nearest_detected_object(self, sample_objects):
        """Test finding nearest object"""
        objects = Objects(sample_objects)
        
        nearest, distance = objects.get_nearest_detected_object([0.2, 0.0])
        
        assert nearest is not None
        assert distance >= 0
        assert isinstance(distance, float)

    def test_get_nearest_detected_object_with_label(self, sample_objects):
        """Test finding nearest object with specific label"""
        objects = Objects(sample_objects)
        
        nearest, distance = objects.get_nearest_detected_object([0.2, 0.0], label="pen")
        
        assert nearest is not None
        assert nearest.label() == "pen"

    def test_get_nearest_no_objects(self):
        """Test nearest object with empty collection"""
        objects = Objects()
        
        nearest, distance = objects.get_nearest_detected_object([0.0, 0.0])
        
        assert nearest is None
        assert distance == float('inf')

    def test_get_largest_detected_object(self, sample_objects):
        """Test getting largest object"""
        objects = Objects(sample_objects)
        
        largest, size = objects.get_largest_detected_object()
        
        assert largest is not None
        assert size > 0
        assert isinstance(size, float)

    def test_get_smallest_detected_object(self, sample_objects):
        """Test getting smallest object"""
        objects = Objects(sample_objects)
        
        smallest, size = objects.get_smallest_detected_object()
        
        assert smallest is not None
        assert size > 0
        assert isinstance(size, float)

    def test_get_detected_objects_sorted_ascending(self, sample_objects):
        """Test sorting objects ascending"""
        objects = Objects(sample_objects)
        
        sorted_objs = objects.get_detected_objects_sorted(ascending=True)
        
        assert len(sorted_objs) == 3
        # Check that sizes are in ascending order
        sizes = [obj.size_m2() for obj in sorted_objs]
        assert sizes == sorted(sizes)

    def test_get_detected_objects_sorted_descending(self, sample_objects):
        """Test sorting objects descending"""
        objects = Objects(sample_objects)
        
        sorted_objs = objects.get_detected_objects_sorted(ascending=False)
        
        assert len(sorted_objs) == 3
        # Check that sizes are in descending order
        sizes = [obj.size_m2() for obj in sorted_objs]
        assert sizes == sorted(sizes, reverse=True)

    def test_get_detected_objects_as_comma_separated_string(self, sample_objects):
        """Test comma-separated string representation"""
        objects = Objects(sample_objects)
        
        result = objects.get_detected_objects_as_comma_separated_string()
        
        assert "pencil" in result
        assert "pen" in result
        assert "eraser" in result
        assert "," in result

    def test_objects_to_dict_list(self, sample_objects):
        """Test conversion to dictionary list"""
        objects = Objects(sample_objects)
        
        dict_list = Objects.objects_to_dict_list(objects)
        
        assert isinstance(dict_list, list)
        assert len(dict_list) == 3
        assert all(isinstance(d, dict) for d in dict_list)
        assert all('label' in d for d in dict_list)

    def test_dict_list_to_objects(self, sample_objects, mock_workspace):
        """Test reconstruction from dictionary list"""
        objects = Objects(sample_objects)
        
        # Convert to dict list
        dict_list = Objects.objects_to_dict_list(objects)
        
        # Convert back
        reconstructed = Objects.dict_list_to_objects(dict_list, mock_workspace)
        
        assert len(reconstructed) == len(objects)
        assert all(isinstance(obj, Object) for obj in reconstructed)

    def test_serializable_methods(self, sample_objects):
        """Test serializable parameter in various methods"""
        objects = Objects(sample_objects)
        
        # Test get_detected_object with serializable=True
        obj_dict = objects.get_detected_object([0.2, 0.0], label="pencil", serializable=True)
        assert isinstance(obj_dict, dict)
        
        # Test get_largest_detected_object with serializable=True
        largest_dict, size = objects.get_largest_detected_object(serializable=True)
        assert isinstance(largest_dict, dict)
        
        # Test get_smallest_detected_object with serializable=True
        smallest_dict, size = objects.get_smallest_detected_object(serializable=True)
        assert isinstance(smallest_dict, dict)
        
        # Test get_detected_objects_sorted with serializable=True
        sorted_dicts = objects.get_detected_objects_sorted(serializable=True)
        assert isinstance(sorted_dicts, list)
        assert all(isinstance(d, dict) for d in sorted_dicts)

    def test_get_detected_objects_serializable(self, sample_objects):
        """Test get_detected_objects_serializable method"""
        objects = Objects(sample_objects)
        
        result = objects.get_detected_objects_serializable(label="pen")
        
        assert isinstance(result, list)
        assert all(isinstance(d, dict) for d in result)

    def test_iteration(self, sample_objects):
        """Test that Objects is iterable"""
        objects = Objects(sample_objects)
        
        count = 0
        for obj in objects:
            assert isinstance(obj, Object)
            count += 1
        
        assert count == 3

    def test_indexing(self, sample_objects):
        """Test indexing"""
        objects = Objects(sample_objects)
        
        assert objects[0].label() == "pencil"
        assert objects[1].label() == "pen"
        assert objects[2].label() == "eraser"

    def test_length(self, sample_objects):
        """Test length"""
        objects = Objects(sample_objects)
        assert len(objects) == 3

    def test_verbose_property(self):
        """Test verbose property"""
        objects = Objects(verbose=True)
        assert objects.verbose() is True
        
        objects = Objects(verbose=False)
        assert objects.verbose() is False

    def test_location_enum_conversion(self):
        """Test Location enum to string conversion"""
        assert Location.convert_str2location("left next to") == Location.LEFT_NEXT_TO
        assert Location.convert_str2location(Location.RIGHT_NEXT_TO) == Location.RIGHT_NEXT_TO
        assert Location.convert_str2location(None) == Location.NONE

    def test_location_enum_invalid_string(self):
        """Test invalid location string raises error"""
        with pytest.raises(ValueError):
            Location.convert_str2location("invalid_location")
