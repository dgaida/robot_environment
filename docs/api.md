# API Reference

## Environment Class

- `get_current_frame()` - Capture current camera image
- `get_detected_objects()` - Get list of detected objects
- `robot_move2observation_pose(workspace_id)` - Move robot to observation position
- `get_workspace(index)` - Get workspace by index
- `get_robot_pose()` - Get current gripper pose

## Object Class

- `label()` - Object label/name
- `xy_com()` - Center of mass coordinates
- `shape_m()` - Width and height in meters
- `gripper_rotation()` - Optimal gripper orientation
- `to_dict()` - Serialize to dictionary
- `from_dict(data, workspace)` - Deserialize from dictionary

## Objects Class

- `get_detected_object(coordinate, label)` - Find object at location
- `get_nearest_detected_object(coordinate, label)` - Find nearest object
- `get_largest_detected_object()` - Get largest object
- `get_detected_objects_sorted(ascending)` - Sort by size
