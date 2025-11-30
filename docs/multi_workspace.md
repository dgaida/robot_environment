# Multi-Workspace Operations Guide

## Overview

The `robot_environment` package supports multi-workspace operations, allowing the robot to:
- Detect objects in multiple workspaces simultaneously
- Pick objects from one workspace and place them in another
- Maintain separate object memory for each workspace
- Coordinate movements between workspaces

## Architecture

### Workspace Memory Management

Each workspace has its own object memory that tracks detected objects:

```python
# Per-workspace memory structure
_workspace_memories: Dict[str, Objects] = {
    "niryo_ws_left": Objects([...]),
    "niryo_ws_right": Objects([...])
}
```

### Workspace Configuration

Define multiple workspaces in `NiryoWorkspaces`:

```python
from robot_workspace.workspaces.niryo_workspaces import NiryoWorkspaces

# Initialize with multiple workspaces
workspaces = NiryoWorkspaces(environment, verbose=True)

# Access individual workspaces
left_ws = workspaces.get_workspace_left()
right_ws = workspaces.get_workspace_right()
```

## Basic Usage

### 1. Initialize Environment

```python
from robot_environment.environment import Environment

env = Environment(
    el_api_key="your_key",
    use_simulation=False,
    robot_id="niryo",
    verbose=True
)

# Get workspace IDs
left_ws_id = env.workspaces().get_workspace_left_id()
right_ws_id = env.workspaces().get_workspace_right_id()
```

### 2. Observe Multiple Workspaces

```python
# Observe left workspace
env.robot_move2observation_pose(left_ws_id)
env.set_current_workspace(left_ws_id)
time.sleep(2)  # Wait for detection

# Observe right workspace
env.robot_move2observation_pose(right_ws_id)
env.set_current_workspace(right_ws_id)
time.sleep(2)  # Wait for detection
```

### 3. Query Objects by Workspace

```python
# Get objects from specific workspace
left_objects = env.get_detected_objects_from_workspace(left_ws_id)
right_objects = env.get_detected_objects_from_workspace(right_ws_id)

# Get all objects from all workspaces
all_objects = env.get_all_workspace_objects()
for ws_id, objects in all_objects.items():
    print(f"{ws_id}: {len(objects)} objects")
```

### 4. Transfer Objects Between Workspaces

```python
robot = env.robot()

# Simple transfer
robot.pick_place_object_across_workspaces(
    object_name='cube',
    pick_workspace_id=left_ws_id,
    pick_coordinate=[0.2, 0.05],
    place_workspace_id=right_ws_id,
    place_coordinate=[0.25, -0.05],
    location=Location.NONE
)
```

## Advanced Operations

### Organized Placement

Place objects in a grid pattern:

```python
placement_grid = [
    [0.20, -0.08], [0.25, -0.08], [0.30, -0.08],
    [0.20, -0.04], [0.25, -0.04], [0.30, -0.04]
]

for i, obj in enumerate(objects):
    robot.pick_place_object_across_workspaces(
        object_name=obj.label(),
        pick_workspace_id=source_ws_id,
        pick_coordinate=[obj.x_com(), obj.y_com()],
        place_workspace_id=target_ws_id,
        place_coordinate=placement_grid[i],
        location=Location.NONE
    )
```

### Object Sorting

Sort objects by type into different workspaces:

```python
for obj in all_detected_objects:
    if 'cube' in obj.label().lower():
        target_ws = left_ws_id
    elif 'cylinder' in obj.label().lower():
        target_ws = right_ws_id
    else:
        continue

    robot.pick_place_object_across_workspaces(
        object_name=obj.label(),
        pick_workspace_id=obj.workspace().id(),
        pick_coordinate=[obj.x_com(), obj.y_com()],
        place_workspace_id=target_ws,
        place_coordinate=[0.25, 0.0],
        location=Location.CLOSE_TO
    )
```

### Find Free Space

Find the largest free space in a workspace:

```python
# Get largest free space in target workspace
env.robot_move2observation_pose(target_ws_id)
env.set_current_workspace(target_ws_id)

largest_area, center_x, center_y = env.get_largest_free_space_with_center()
print(f"Free space: {largest_area*10000:.2f} cm² at [{center_x:.2f}, {center_y:.2f}]")

# Place object at center of free space
robot.pick_place_object_across_workspaces(
    object_name='box',
    pick_workspace_id=source_ws_id,
    pick_coordinate=[0.2, 0.0],
    place_workspace_id=target_ws_id,
    place_coordinate=[center_x, center_y],
    location=Location.NONE
)
```

## Memory Management

### Per-Workspace Memory

Each workspace maintains its own object memory:

```python
# Clear specific workspace memory
env.clear_workspace_memory(workspace_id)

# Remove object from workspace
env.remove_object_from_workspace(
    workspace_id='niryo_ws_left',
    object_label='cube',
    coordinate=[0.2, 0.05]
)

# Update object position across workspaces
env.update_object_in_workspace(
    source_workspace_id='niryo_ws_left',
    target_workspace_id='niryo_ws_right',
    object_label='cube',
    old_coordinate=[0.2, 0.05],
    new_coordinate=[0.25, -0.05]
)
```

### Memory Synchronization

Memory is automatically updated during:
- Object detection at observation pose
- Pick operations (removes from source workspace)
- Place operations (adds to target workspace)
- Cross-workspace transfers (moves between workspace memories)

## Workspace Coordinate Systems

Each workspace has its own coordinate system:

```
Left Workspace (niryo_ws_left):
  - Origin: Upper-left corner
  - X-axis: Forward (0.163 to 0.337 m)
  - Y-axis: Right to Left (0.087 to -0.087 m)
  - Observation pose: x=0.173, y=0.10, z=0.277

Right Workspace (niryo_ws_right):
  - Origin: Upper-left corner
  - X-axis: Forward (0.163 to 0.337 m)
  - Y-axis: Right to Left (0.087 to -0.087 m)
  - Observation pose: x=0.173, y=-0.10, z=0.277
```

## Error Handling

```python
try:
    # Attempt transfer
    success = robot.pick_place_object_across_workspaces(
        object_name='cube',
        pick_workspace_id=left_ws_id,
        pick_coordinate=[0.2, 0.05],
        place_workspace_id=right_ws_id,
        place_coordinate=[0.25, -0.05],
        location=Location.NONE
    )

    if not success:
        print("Transfer failed")
        # Handle failure

except Exception as e:
    print(f"Error during transfer: {e}")
    # Cleanup or recovery
```

## Best Practices

1. **Always observe workspaces before operations**:
   ```python
   env.robot_move2observation_pose(workspace_id)
   env.set_current_workspace(workspace_id)
   time.sleep(2)  # Wait for detection
   ```

2. **Check for objects before picking**:
   ```python
   objects = env.get_detected_objects_from_workspace(workspace_id)
   if len(objects) == 0:
       print("No objects found")
       return
   ```

3. **Verify workspace IDs**:
   ```python
   workspace_ids = env.workspaces().get_workspace_ids()
   if target_ws_id not in workspace_ids:
       print(f"Invalid workspace: {target_ws_id}")
       return
   ```

4. **Use appropriate wait times**:
   ```python
   # After moving to observation pose
   time.sleep(2)  # Allow detection to complete

   # Between transfers
   time.sleep(0.5)  # Brief pause
   ```

## Examples

See `examples/multi_workspace_example.py` for complete examples:

```bash
cd robot_environment/examples
python multi_workspace_example.py
```

Available examples:
1. **Simple Transfer**: Move one object between workspaces
2. **Organized Transfer**: Arrange multiple objects in a grid
3. **Object Sorting**: Sort objects by type
4. **Workspace Cleanup**: Clear one workspace into another

## Troubleshooting

### Objects not detected

```python
# Ensure workspace is visible
is_visible = env.is_any_workspace_visible()
if not is_visible:
    env.robot_move2observation_pose(workspace_id)

# Clear and refresh memory
env.clear_workspace_memory(workspace_id)
env.set_current_workspace(workspace_id)
time.sleep(2)
```

### Transfer failures

```python
# Check object still exists
objects = env.get_detected_objects_from_workspace(source_ws_id)
obj = objects.get_detected_object(coordinate, label)
if obj is None:
    print("Object no longer detected")
```

### Memory inconsistencies

```python
# Force memory refresh
env.clear_memory()  # Clear all workspaces
env.robot_move2observation_pose(workspace_id)
time.sleep(2)  # Re-detect all objects
```

## Performance Tips

1. **Minimize workspace switches**: Group operations by workspace
2. **Use memory queries**: Avoid unnecessary observations
3. **Batch transfers**: Transfer multiple objects in sequence
4. **Pre-calculate positions**: Determine all target positions before starting

## Future Enhancements

- Automatic workspace detection
- Collision avoidance between workspaces
- Parallel processing for multiple robots
- Dynamic workspace reconfiguration
