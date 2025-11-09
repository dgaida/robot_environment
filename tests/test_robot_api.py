"""
Unit tests for Robot API classes
"""
import pytest
from robot_environment.robot.robot_api import Location, RobotAPI


class TestLocation:
    """Test suite for Location enum"""

    def test_location_values(self):
        """Test that all location values are correct"""
        assert Location.LEFT_NEXT_TO.value == "left next to"
        assert Location.RIGHT_NEXT_TO.value == "right next to"
        assert Location.ABOVE.value == "above"
        assert Location.BELOW.value == "below"
        assert Location.ON_TOP_OF.value == "on top of"
        assert Location.INSIDE.value == "inside"
        assert Location.CLOSE_TO.value == "close to"
        assert Location.NONE.value is None

    def test_convert_str2location_with_string(self):
        """Test string to Location conversion"""
        assert Location.convert_str2location("left next to") == Location.LEFT_NEXT_TO
        assert Location.convert_str2location("right next to") == Location.RIGHT_NEXT_TO
        assert Location.convert_str2location("above") == Location.ABOVE
        assert Location.convert_str2location("below") == Location.BELOW
        assert Location.convert_str2location("on top of") == Location.ON_TOP_OF
        assert Location.convert_str2location("inside") == Location.INSIDE
        assert Location.convert_str2location("close to") == Location.CLOSE_TO

    def test_convert_str2location_with_enum(self):
        """Test Location to Location conversion (identity)"""
        assert Location.convert_str2location(Location.LEFT_NEXT_TO) == Location.LEFT_NEXT_TO
        assert Location.convert_str2location(Location.RIGHT_NEXT_TO) == Location.RIGHT_NEXT_TO

    def test_convert_str2location_with_none(self):
        """Test None to Location.NONE conversion"""
        assert Location.convert_str2location(None) == Location.NONE

    def test_convert_str2location_invalid_string(self):
        """Test invalid string raises ValueError"""
        with pytest.raises(ValueError, match="Invalid location string"):
            Location.convert_str2location("invalid location")

    def test_convert_str2location_invalid_type(self):
        """Test invalid type raises TypeError"""
        with pytest.raises(TypeError, match="Location must be either a string or a Location enum"):
            Location.convert_str2location(123)

    def test_all_locations_have_unique_values(self):
        """Test that all location values are unique"""
        values = [loc.value for loc in Location if loc != Location.NONE]
        assert len(values) == len(set(values))

    def test_location_enumeration(self):
        """Test iterating over Location enum"""
        locations = list(Location)
        assert len(locations) == 8  # All defined locations
        assert Location.LEFT_NEXT_TO in locations
        assert Location.NONE in locations


class TestRobotAPI:
    """Test suite for RobotAPI abstract class"""

    def test_is_abstract(self):
        """Test that RobotAPI cannot be instantiated directly"""
        with pytest.raises(TypeError):
            RobotAPI()

    def test_has_required_methods(self):
        """Test that RobotAPI defines required abstract methods"""
        required_methods = [
            'pick_place_object',
            'pick_object',
            'place_object',
            'push_object',
            'move2observation_pose'
        ]
        
        for method in required_methods:
            assert hasattr(RobotAPI, method)

    def test_pick_place_object_signature(self):
        """Test pick_place_object method signature"""
        import inspect
        sig = inspect.signature(RobotAPI.pick_place_object)
        params = list(sig.parameters.keys())
        
        assert 'object_name' in params
        assert 'pick_coordinate' in params
        assert 'place_coordinate' in params
        assert 'location' in params

    def test_pick_object_signature(self):
        """Test pick_object method signature"""
        import inspect
        sig = inspect.signature(RobotAPI.pick_object)
        params = list(sig.parameters.keys())
        
        assert 'object_name' in params
        assert 'pick_coordinate' in params

    def test_place_object_signature(self):
        """Test place_object method signature"""
        import inspect
        sig = inspect.signature(RobotAPI.place_object)
        params = list(sig.parameters.keys())
        
        assert 'place_coordinate' in params
        assert 'location' in params

    def test_push_object_signature(self):
        """Test push_object method signature"""
        import inspect
        sig = inspect.signature(RobotAPI.push_object)
        params = list(sig.parameters.keys())
        
        assert 'object_name' in params
        assert 'push_coordinate' in params
        assert 'direction' in params
        assert 'distance' in params

    def test_move2observation_pose_signature(self):
        """Test move2observation_pose method signature"""
        import inspect
        sig = inspect.signature(RobotAPI.move2observation_pose)
        params = list(sig.parameters.keys())
        
        assert 'workspace_id' in params


class TestLocationIntegration:
    """Integration tests for Location usage"""

    def test_location_in_dict(self):
        """Test that Location can be used as dict key"""
        loc_dict = {
            Location.LEFT_NEXT_TO: "left",
            Location.RIGHT_NEXT_TO: "right"
        }
        
        assert loc_dict[Location.LEFT_NEXT_TO] == "left"
        assert loc_dict[Location.RIGHT_NEXT_TO] == "right"

    def test_location_comparison(self):
        """Test Location comparison"""
        loc1 = Location.LEFT_NEXT_TO
        loc2 = Location.LEFT_NEXT_TO
        loc3 = Location.RIGHT_NEXT_TO
        
        assert loc1 == loc2
        assert loc1 != loc3
        assert loc1 is loc2  # Same enum instance

    def test_location_string_representation(self):
        """Test string representation of Location"""
        loc = Location.LEFT_NEXT_TO
        
        # The name property
        assert loc.name == "LEFT_NEXT_TO"
        # The value property
        assert loc.value == "left next to"

    def test_location_in_set(self):
        """Test that Location can be used in sets"""
        locations = {Location.LEFT_NEXT_TO, Location.RIGHT_NEXT_TO, Location.LEFT_NEXT_TO}
        
        # Should deduplicate
        assert len(locations) == 2
        assert Location.LEFT_NEXT_TO in locations

    def test_location_switch_pattern(self):
        """Test switch-like pattern with Location"""
        def get_offset(location: Location):
            if location == Location.LEFT_NEXT_TO:
                return (0, 1)
            elif location == Location.RIGHT_NEXT_TO:
                return (0, -1)
            elif location == Location.ABOVE:
                return (1, 0)
            elif location == Location.BELOW:
                return (-1, 0)
            else:
                return (0, 0)
        
        assert get_offset(Location.LEFT_NEXT_TO) == (0, 1)
        assert get_offset(Location.RIGHT_NEXT_TO) == (0, -1)
        assert get_offset(Location.ABOVE) == (1, 0)
        assert get_offset(Location.BELOW) == (-1, 0)
        assert get_offset(Location.NONE) == (0, 0)

    def test_location_with_match_statement(self):
        """Test Location with match statement (Python 3.10+)"""
        import sys
        if sys.version_info >= (3, 10):
            def describe_location(loc: Location) -> str:
                match loc:
                    case Location.LEFT_NEXT_TO:
                        return "to the left"
                    case Location.RIGHT_NEXT_TO:
                        return "to the right"
                    case Location.ABOVE:
                        return "above"
                    case Location.BELOW:
                        return "below"
                    case _:
                        return "somewhere"
            
            assert describe_location(Location.LEFT_NEXT_TO) == "to the left"
            assert describe_location(Location.NONE) == "somewhere"
