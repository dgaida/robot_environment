# Robot Environment

A comprehensive Python framework for robotic pick-and-place operations with vision-based object detection and manipulation capabilities.

## Badges

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Quality](https://github.com/dgaida/robot_environment/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/robot_environment/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/robot_environment/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/robot_environment/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/robot_environment/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/robot_environment/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Overview

This project provides a complete software stack for controlling robotic arms (currently supporting Niryo Ned2 and WidowX) with integrated computer vision for object detection, workspace management, and intelligent manipulation. The system combines real-time camera processing, Redis-based communication, and natural language interaction capabilities.

## Key Features

- **Multi-Robot Support**: Modular architecture supporting Niryo Ned2 and WidowX robotic arms
- **Vision-Based Object Detection**: Integration with OwlV2 for real-time object detection and segmentation
- **Workspace Management**: Flexible workspace definition and camera-to-world coordinate transformation
- **Redis Communication**: Efficient image streaming and object data sharing via Redis
- **Text-to-Speech**: Natural language feedback using Kokoro TTS
- **Thread-Safe Operations**: Concurrent camera updates and robot control
- **Simulation Support**: Compatible with both real robots and Gazebo simulation

## Architecture

### Core Components

```
robot_environment/
├── camera/              # Frame grabbing and image acquisition
│   ├── framegrabber.py
│   ├── niryo_framegrabber.py
│   └── widowx_framegrabber.py
├── objects/             # Object detection and representation
│   ├── object.py
│   ├── objects.py
│   └── pose_object.py
├── robot/               # Robot control abstractions
│   ├── robot.py
│   ├── robot_controller.py
│   ├── niryo_robot_controller.py
│   └── widowx_robot_controller.py
├── workspaces/          # Workspace definitions and management
│   ├── workspace.py
│   ├── workspaces.py
│   └── niryo_workspace.py
├── text2speech/         # Audio output capabilities
│   └── text2speech.py
└── environment.py       # Main environment orchestration
```

## Installation

### Prerequisites

- Python 3.8+
- Redis server
- Robot-specific drivers (pyniryo for Niryo, interbotix for WidowX)

### Dependencies

```bash
pip install numpy opencv-python redis torch torchaudio
pip install vision-detect-segment redis-robot-comm
pip install elevenlabs kokoro  # For text-to-speech
pip install pyniryo  # For Niryo robots
```

## Quick Start

### Basic Usage

```bash
docker run -p 6379:6379 redis:alpine
```

```python
from robot_environment.environment import Environment
import threading
import time

# Initialize environment
env = Environment(
    el_api_key="your_elevenlabs_key",  # For TTS
    use_simulation=False,               # True for Gazebo
    robot_id="niryo",                   # or "widowx"
    verbose=True
)

# Start camera updates in background
def start_camera_updates(environment, visualize=False):
    def loop():
        for img in environment.update_camera_and_objects(visualize=visualize):
            pass
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t

camera_thread = start_camera_updates(env, visualize=True)

# Move to observation pose
env.robot_move2observation_pose(env.get_workspace_home_id())

# Wait for objects to be detected
time.sleep(2)

# Get detected objects
detected_objects = env.get_detected_objects()
print(f"Detected {len(detected_objects)} objects")

# Pick and place an object
robot = env.robot()
robot.pick_place_object(
    object_label="pencil",
    pick_location=[-0.1, 0.01],
    place_location=[0.1, 0.11],
    location="right next to"
)
```

### Advanced Features

#### Object Filtering

```python
from robot_environment.robot.robot_api import Location

# Get objects in specific locations
objects_left = detected_objects.get_detected_objects(
    location=Location.LEFT_NEXT_TO,
    coordinate=[0.2, 0.0],
    label="cube"
)

# Get nearest object
nearest, distance = detected_objects.get_nearest_detected_object(
    coordinate=[0.25, 0.05],
    label="pencil"
)

# Get largest/smallest objects
largest, size = detected_objects.get_largest_detected_object()
smallest, size = detected_objects.get_smallest_detected_object()
```

#### Workspace Management

```python
# Get workspace information
workspace = env.get_workspace(0)
print(f"Workspace size: {workspace.width_m()}m x {workspace.height_m()}m")

# Transform coordinates
pose = workspace.transform_camera2world_coords(
    workspace_id="niryo_ws",
    u_rel=0.5,  # Center of image
    v_rel=0.5,
    yaw=0.0
)
```

#### Redis Integration

The system automatically publishes camera frames and detected objects to Redis:

```python
# Objects are automatically published as JSON-serializable dictionaries
# Images are streamed with metadata including:
# - workspace_id
# - frame_id
# - robot_pose
# - timestamp
```

## Object Detection

Objects are detected using the VisualCortex module with OwlV2:

```python
# Object properties
obj = detected_objects[0]
print(f"Label: {obj.label()}")
print(f"Position: {obj.x_com():.3f}, {obj.y_com():.3f}")
print(f"Size: {obj.width_m():.3f}m x {obj.height_m():.3f}m")
print(f"Rotation: {obj.gripper_rotation():.2f} rad")
print(f"Area: {obj.size_m2() * 10000:.2f} cm²")

# Serialize for transmission
obj_dict = obj.to_dict()
obj_json = obj.to_json()
```

## Robot Control

### Pick and Place

```python
# Simple pick and place
robot.pick_place_object(
    object_label="red cube",
    pick_location=[0.235, 0.3],
    place_location=[0.54, 0.43],
    location="right next to"
)
```

### Direct Manipulation

```python
# Get object
obj = detected_objects.get_detected_object(
    coordinate=[0.2, 0.0],
    label="cube"
)

# Pick object
success = robot_controller.robot_pick_object(obj)

# Place at specific pose
place_pose = workspace.transform_camera2world_coords(
    workspace_id="niryo_ws",
    u_rel=0.8,
    v_rel=0.5,
    yaw=0.0
)
success = robot_controller.robot_place_object(place_pose)
```

## Text-to-Speech

```python
# Asynchronous speech output
thread = env.oralcom_call_text2speech_async(
    "I have detected a red cube at position 0.25, 0.05"
)
thread.join()  # Wait for speech to complete
```

## Configuration

### Adding New Workspaces

For Niryo robots, edit `niryo_workspace.py`:

```python
def _set_observation_pose(self) -> None:
    if self._id == "my_new_workspace":
        self._observation_pose = PoseObjectPNP(
            x=0.18, y=0.0, z=0.36,
            roll=0.0, pitch=math.pi/2, yaw=0.0
        )
```

### Camera Calibration

Camera intrinsics are automatically retrieved from the robot. For custom cameras:

```python
# In framegrabber implementation
self._mtx, self._dist = get_camera_intrinsics()
```

## Thread Safety

All robot operations are thread-safe through the use of locks:

```python
with robot_controller.lock():
    # Thread-safe robot operations
    pose = robot.get_pose()
```

## Error Handling

```python
try:
    success = robot.pick_place_object("cube", [0.2, 0.0], [0.3, 0.0])
    if not success:
        print("Pick and place operation failed")
except Exception as e:
    print(f"Error: {e}")
```

## Simulation Mode

```python
# Use Gazebo simulation
env = Environment(
    el_api_key="key",
    use_simulation=True,
    robot_id="niryo",
    verbose=True
)
```

## API Reference

### Environment Class

- `get_current_frame()` - Capture current camera image
- `get_detected_objects()` - Get list of detected objects
- `robot_move2observation_pose(workspace_id)` - Move robot to observation position
- `get_workspace(index)` - Get workspace by index
- `get_robot_pose()` - Get current gripper pose

### Object Class

- `label()` - Object label/name
- `xy_com()` - Center of mass coordinates
- `shape_m()` - Width and height in meters
- `gripper_rotation()` - Optimal gripper orientation
- `to_dict()` - Serialize to dictionary
- `from_dict(data, workspace)` - Deserialize from dictionary

### Objects Class

- `get_detected_object(coordinate, label)` - Find object at location
- `get_nearest_detected_object(coordinate, label)` - Find nearest object
- `get_largest_detected_object()` - Get largest object
- `get_detected_objects_sorted(ascending)` - Sort by size

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the existing architecture patterns
2. Thread safety is maintained
3. Documentation is updated
4. Type hints are included

## License

MIT License

## Acknowledgments

- OwlV2 for object detection
- Kokoro for text-to-speech
- Redis for inter-process communication
- Niryo and Trossen Robotics for robot hardware

## Support

For issues and questions, please open an issue on GitHub.
