# Robot Environment Architecture Documentation

This document describes the architectural design of the `robot_environment` package, detailing component interactions, data flows, and integration with external packages.

## Overview

The `robot_environment` package provides a comprehensive framework for robotic pick-and-place operations with vision-based object detection. The architecture follows a layered design with clear separation of concerns:

1. **Environment Layer** - Central orchestrator
2. **Robot Control Layer** - Hardware abstraction and motion control
3. **Vision Layer** - Object detection and tracking
4. **Workspace Layer** - Coordinate transformation and workspace management
5. **Communication Layer** - Redis-based data streaming
6. **Interaction Layer** - Text-to-speech feedback

## System Architecture

![System Architecture](architecture_package.png)

## Core Components

### 1. Environment (`environment.py`)

The `Environment` class serves as the central orchestrator that coordinates all subsystems.

**Responsibilities:**
- Initialize and manage all subsystems
- Coordinate camera updates and object detection
- Provide unified API for robot operations
- Manage object memory and detection history

**Key Dependencies:**
- `Robot` - Robot control interface
- `FrameGrabber` - Camera interface
- `Workspaces` - Workspace management
- `VisualCortex` - Object detection (from `vision_detect_segment`)
- `Text2Speech` - Audio feedback
- `RedisMessageBroker` - Object data publishing

**Workflow:**
```python
env = Environment(el_api_key, use_simulation, robot_id, verbose)
# 1. Creates Robot (which creates RobotController)
# 2. Creates FrameGrabber for camera
# 3. Creates Workspaces collection
# 4. Creates VisualCortex for detection
# 5. Starts background camera thread
```

### 2. Robot Control Layer

#### Robot (`robot/robot.py`)

High-level robot API implementing `RobotAPI` interface.

**Responsibilities:**
- Implement pick-and-place operations
- Object position verification
- Command parsing and execution
- Safety checks

**Key Methods:**
- `pick_place_object()` - Complete pick and place operation
- `pick_object()` - Pick operation only
- `place_object()` - Place operation with location logic
- `push_object()` - Push operation for non-grippable objects

#### RobotController (`robot/robot_controller.py`)

Abstract base class for robot hardware control.

**Implementations:**
- `NiryoRobotController` - Niryo Ned2 robot
- `WidowXRobotController` - WidowX robot (stub)

**Responsibilities:**
- Hardware communication
- Low-level motion commands
- Thread-safe operation (via locks)
- Pose transformations

**Critical Feature - Thread Safety:**
```python
with self._lock:  # Ensures thread-safe robot access
    pose = self._robot_ctrl.get_pose()
```

### 3. Vision Layer

#### VisualCortex (from `vision_detect_segment`)

External package providing object detection capabilities.

**Features:**
- Multi-model support (OWL-V2, YOLO-World, Grounding-DINO)
- Object tracking
- Instance segmentation (SAM2, FastSAM)
- Redis integration

**Integration:**
```python
self._visual_cortex = VisualCortex(
    objdetect_model_id="owlv2",
    device="auto",
    config=config
)
```

**Data Flow:**
1. FrameGrabber captures image → Redis
2. VisualCortex reads from Redis
3. Detects objects with bounding boxes
4. Optional tracking and segmentation
5. Publishes results to Redis

#### FrameGrabber (`camera/framegrabber.py`)

Abstract base class for camera interfaces.

**Implementations:**
- `NiryoFrameGrabber` - Niryo robot's mounted camera
- `WidowXFrameGrabber` - Intel RealSense (stub)

**Responsibilities:**
- Image acquisition
- Image undistortion
- Workspace extraction
- Redis streaming via `RedisImageStreamer`

**Key Features:**
```python
# Variable-size image streaming
streamer.publish_image(
    image,
    metadata={'workspace_id': ws_id, 'robot_pose': pose},
    compress_jpeg=True
)
```

### 4. Object Representation Layer

#### Object (`objects/object.py`)

Represents a detected object with full spatial information.

**Properties:**
- Label and workspace reference
- Pixel coordinates (bounding box)
- World coordinates (pose)
- Physical dimensions (meters)
- Segmentation mask (optional)
- Gripper rotation (for optimal pickup)

**Coordinate Systems:**
- Image coordinates: `(u, v)` pixels
- Relative coordinates: `(u_rel, v_rel)` normalized [0,1]
- World coordinates: `(x, y, z)` meters

**Serialization:**
```python
# Convert to JSON for Redis
obj_dict = obj.to_dict()
json_str = obj.to_json()

# Reconstruct from JSON
reconstructed = Object.from_dict(obj_dict, workspace)
```

#### Objects (`objects/objects.py`)

Collection class extending Python's `List`.

**Query Methods:**
- `get_detected_object(coordinate, label)` - Find by location
- `get_nearest_detected_object(coordinate)` - Nearest search
- `get_largest_detected_object()` - Size-based queries
- `get_detected_objects(location, coordinate, label)` - Spatial filtering

**Spatial Filters:**
```python
Location.LEFT_NEXT_TO   # y > coordinate[1]
Location.RIGHT_NEXT_TO  # y < coordinate[1]
Location.ABOVE          # x > coordinate[0]
Location.BELOW          # x < coordinate[0]
Location.CLOSE_TO       # distance <= 2cm
```

#### PoseObjectPNP (`objects/pose_object.py`)

File source: [Niryo Robotics](https://niryorobotics.github.io/pyniryo/v1.2.1/api/api.html#pyniryo.api.objects.PoseObject)

6-DOF pose representation.

**Components:**
- Position: `(x, y, z)` in meters
- Orientation: `(roll, pitch, yaw)` in radians

**Features:**
- Arithmetic operations (`+`, `-`)
- Transformation matrices (4×4 homogeneous)
- Quaternion conversion
- Approximate equality checks

### 5. Workspace Layer

#### Workspace (`workspaces/workspace.py`)

Abstract base class defining a robot workspace.

**Responsibilities:**
- Camera-to-world coordinate transformation
- Workspace boundary definition
- Visibility detection
- Observation pose management

**Key Concept - Coordinate Transformation:**
```
Image Coordinates (pixels) 
    ↓ normalize
Relative Coordinates [0,1]
    ↓ transform_camera2world_coords()
World Coordinates (meters)
```

#### NiryoWorkspace (`workspaces/niryo_workspace.py`)

Niryo-specific workspace implementation.

**Features:**
- Uses Niryo's built-in vision system
- Supports multiple workspace definitions
- Automatic corner detection
- Predefined observation poses

**Workspace Corners:**
- `xy_ul_wc()` - Upper left
- `xy_ur_wc()` - Upper right
- `xy_ll_wc()` - Lower left
- `xy_lr_wc()` - Lower right
- `xy_center_wc()` - Center point

#### Workspaces (`workspaces/workspaces.py`)

Collection managing multiple workspaces.

**Features:**
- Workspace lookup by ID
- Visible workspace detection
- Home workspace management

### 6. Communication Layer

#### RedisImageStreamer (from `redis_robot_comm`)

Handles variable-size image streaming.

**Features:**
- JPEG compression
- Metadata attachment
- Variable image sizes
- Automatic base64 encoding

**Usage:**
```python
streamer = RedisImageStreamer(stream_name='robot_camera')
stream_id = streamer.publish_image(
    image,
    metadata={'workspace_id': 'ws1', 'frame_id': 42},
    compress_jpeg=True,
    quality=85
)
```

#### RedisMessageBroker (from `redis_robot_comm`)

Publishes object detection results.

**Data Format:**
```json
{
  "id": "obj_abc123",
  "label": "pencil",
  "position": {"x": 0.25, "y": 0.05, "z": 0.01},
  "dimensions": {"width_m": 0.02, "height_m": 0.15},
  "workspace_id": "niryo_ws",
  "timestamp": 1699876543.21
}
```

### 7. Interaction Layer

#### Text2Speech (from `text2speech` package)

Provides natural language feedback.

**Features:**
- Asynchronous speech generation
- ElevenLabs or Kokoro TTS
- Thread-safe operation

**Usage:**
```python
thread = env.oralcom_call_text2speech_async(
    "I have picked up the pencil"
)
thread.join()  # Wait for completion
```

## Data Flow Architecture

### Complete Pick-and-Place Workflow

```
1. INITIALIZATION
   Environment
      ├─→ Robot → RobotController (connects to hardware)
      ├─→ FrameGrabber (initializes camera)
      ├─→ Workspaces (loads workspace definitions)
      └─→ VisualCortex (loads detection models)

2. OBSERVATION
   Robot.move2observation_pose(workspace_id)
      └─→ RobotController.move2observation_pose()
          └─→ Gripper moves to hover above workspace

3. IMAGE CAPTURE & DETECTION
   FrameGrabber.get_current_frame()
      ├─→ Capture image from camera
      ├─→ Undistort image
      ├─→ Extract workspace region
      └─→ RedisImageStreamer.publish_image()
              └─→ Redis: 'robot_camera' stream

   VisualCortex.detect_objects_from_redis()
      ├─→ Read from Redis stream
      ├─→ Run detection model
      ├─→ Optional: Track objects
      ├─→ Optional: Segment objects
      └─→ RedisMessageBroker.publish_objects()
              └─→ Redis: 'detected_objects' stream

4. OBJECT PROCESSING
   Environment.get_detected_objects()
      ├─→ Read from Redis
      ├─→ Deserialize JSON
      └─→ Convert to Object instances
              ├─→ Calculate world coordinates
              ├─→ Determine object orientation (needed for gripper rotation)
              └─→ Store in Objects collection

5. PICK OPERATION
   Robot.pick_object(label, coordinate)
      ├─→ Find nearest object with label
      ├─→ Text2Speech: "Going to pick {label}"
      └─→ RobotController.robot_pick_object(obj)
              ├─→ Lock acquired
              ├─→ Move to pre-grasp pose
              ├─→ Move to grasp pose (z_offset=0.001m)
              ├─→ Close gripper
              ├─→ Lift object
              └─→ Lock released

6. PLACE OPERATION
   Robot.place_object(coordinate, location)
      ├─→ Calculate placement pose
      │   └─→ Apply location offset (LEFT_NEXT_TO, ABOVE, etc.)
      ├─→ Text2Speech: "Going to place at {coordinate}"
      └─→ RobotController.robot_place_object(pose)
              ├─→ Lock acquired
              ├─→ Move to pre-place pose
              ├─→ Move to place pose (z_offset=0.005m)
              ├─→ Open gripper
              ├─→ Retract
              └─→ Lock released
```

### Continuous Camera Update Loop

```
Background Thread (daemon):
    Loop:
        1. Move to observation pose (if not in motion)
        2. Capture frame → Redis
        3. Wait 0.5s
        4. Trigger detection
        5. Wait 0.5s
        6. Get detected objects
        7. Update memory
        8. Display annotated image (if visualize=True)
        9. Sleep (0.25s or 0.5s depending on robot motion)
```

## Thread Safety

### Critical Sections

1. **Robot Operations**
   - All `RobotController` methods use `self._lock`
   - Prevents concurrent hardware commands
   - Essential for Niryo robot (not thread-safe API)

2. **Camera Updates**
   - Background thread for continuous updates
   - Daemon thread (terminates with main program)
   - Uses `_stop_event` for clean shutdown

3. **Text-to-Speech**
   - Asynchronous threads for non-blocking speech
   - Thread returned for synchronization if needed

### Example - Thread-Safe Robot Access

```python
def get_target_pose_from_rel(self, ws_id, u_rel, v_rel, yaw):
    with self._lock:  # Thread-safe section
        try:
            obj_coords = self._robot_ctrl.get_target_pose_from_rel(
                ws_id, 0.0, u_rel, v_rel, yaw
            )
        except (NiryoRobotException, UnicodeDecodeError) as e:
            print(f"Error: {e}")
            obj_coords = PoseObject(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    return PoseObjectPNP.convert_niryo_pose_object2pose_object(obj_coords)
```

## External Package Integration

### 1. vision_detect_segment

**Purpose:** Object detection and segmentation

**Integration Points:**
- `VisualCortex` instantiated in `Environment.__init__()`
- Reads images from Redis (via `detect_objects_from_redis()`)
- Returns detections as list of dictionaries
- Provides annotated images for visualization

**Configuration:**
```python
config = get_default_config("owlv2")
self._visual_cortex = VisualCortex(
    objdetect_model_id="owlv2",  # or "yoloworld", "groundingdino"
    device="auto",                # or "cuda", "cpu"
    verbose=verbose,
    config=config
)
```

**Models Supported:**
- **OWL-V2**: Open-vocabulary detection
- **YOLO-World**: Fast real-time detection
- **Grounding-DINO**: Text-guided detection

### 2. redis_robot_comm

**Purpose:** Redis-based communication

**Components Used:**
- `RedisImageStreamer` - Variable-size image streaming
- `RedisMessageBroker` - Object data publishing

**Integration:**
```python
# In FrameGrabber
self.streamer = RedisImageStreamer(stream_name='robot_camera')

# In Environment (for object publishing)
self._broker = RedisMessageBroker()
self._broker.publish_objects(objects_dict_list)
```

**Data Streams:**
- `robot_camera` - Compressed images with metadata
- `detected_objects` - Object detection results

### 3. text2speech

**Purpose:** Natural language feedback

**Integration:**
```python
self._oralcom = Text2Speech(el_api_key, verbose=verbose)

# Asynchronous usage
thread = self._oralcom.call_text2speech_async(
    "I have detected a pencil at position 0.25, 0.05"
)
thread.join()  # Optional: wait for completion
```

**TTS Engines:**
- **ElevenLabs API** (primary)
- **Kokoro TTS** (local alternative)

### 4. pyniryo / pyniryo2

**Purpose:** Niryo robot hardware control

**Integration:**
```python
# In NiryoRobotController
self._robot_ctrl = NiryoRobot(robot_ip_address)
self._robot_ctrl.calibrate_auto()
self._robot_ctrl.update_tool()

# Pick operation
self._robot_ctrl.pick_from_pose(pick_pose)

# Place operation
self._robot_ctrl.place_from_pose(place_pose)
```

## Coordinate Systems

### Three Coordinate Systems

1. **Image Coordinates (Pixels)**
   - Origin: Top-left corner
   - Units: Pixels
   - Range: `u ∈ [0, width]`, `v ∈ [0, height]`

2. **Relative Coordinates**
   - Origin: Top-left corner
   - Units: Normalized [0, 1]
   - Range: `u_rel, v_rel ∈ [0, 1]`
   - Used for workspace-independent calculations

3. **World Coordinates (Robot Base Frame)**
   - Origin: Robot base
   - Units: Meters
   - Niryo axes:
     - `x`: Forward (away from base)
     - `y`: Right (when facing robot)
     - `z`: Up

### Transformation Chain

```
Image (u, v)
    ↓ divide by image dimensions
Relative (u_rel, v_rel)
    ↓ Workspace.transform_camera2world_coords()
    ↓ (uses Niryo's get_target_pose_from_rel())
World (x, y, z) + orientation (roll, pitch, yaw)
```

### Example Transformation

```python
# Object at pixel (320, 240) in 640x480 image
u, v = 320, 240
u_rel = 320 / 640 = 0.5  # Center horizontally
v_rel = 240 / 480 = 0.5  # Center vertically

# Transform to world coordinates
pose = workspace.transform_camera2world_coords(
    "niryo_ws", u_rel=0.5, v_rel=0.5, yaw=0.0
)
# Result: pose.x ≈ 0.25, pose.y ≈ 0.0, pose.z ≈ 0.01
```

## Configuration

### Robot Selection

```python
# Niryo Ned2 (real robot)
env = Environment(
    el_api_key="key",
    use_simulation=False,
    robot_id="niryo"
)

# Niryo in Gazebo simulation
env = Environment(
    el_api_key="key",
    use_simulation=True,
    robot_id="niryo"
)
```

### Workspace Configuration

Workspaces are defined in `niryo_workspace.py`:

```python
def _set_observation_pose(self):
    if self._id == "niryo_ws":
        self._observation_pose = PoseObjectPNP(
            x=0.173, y=-0.002, z=0.277,
            roll=-3.042, pitch=1.327, yaw=-3.027
        )
    elif self._id == "gazebo_1":
        self._observation_pose = PoseObjectPNP(
            x=0.18, y=0, z=0.36,
            roll=2.4, pitch=π/2, yaw=2.4
        )
```

## Error Handling

### Robot Connection Issues

```python
try:
    success = robot.pick_place_object("cube", [0.2, 0.0], [0.3, 0.0])
    if not success:
        print("Pick and place operation failed")
except (NiryoRobotException, UnicodeDecodeError) as e:
    print(f"Error: {e}")
    # Optionally: reset connection
    robot_controller.reset_connection()
```

### Object Not Found

```python
obj = objects.get_detected_object([0.2, 0.0], label="nonexistent")
if obj is None:
    print("Object not found")
    # Handle missing object
```

### Thread Cleanup

```python
# Proper cleanup
env = Environment(...)
try:
    # Use environment
    pass
finally:
    env.cleanup()  # Stops threads, closes connections
```

## Performance Considerations

### Detection Speed

| Model | Detection | Segmentation | Total FPS |
|-------|-----------|--------------|-----------|
| YOLO-World | 20-50ms | 50-100ms (FastSAM) | 10-25 FPS |
| OWL-V2 | 100-200ms | 200-500ms (SAM2) | 1-3 FPS |
| Grounding-DINO | 200-400ms | 200-500ms (SAM2) | 1-2 FPS |

### Recommendations

- **Real-time**: Use YOLO-World + FastSAM
- **Accuracy**: Use OWL-V2 or Grounding-DINO + SAM2
- **Camera rate**: 5 FPS sufficient for pick-and-place

### Memory Management

- Object memory stores detection history
- Background thread continuously updates
- Old detections persist until new scan

## Extension Points

### Adding New Robot

1. Create `MyRobotController(RobotController)`
2. Implement abstract methods
3. Create `MyRobotWorkspace(Workspace)`
4. Add to `Robot.__init__()` selection

### Adding New Workspace

1. Add ID to `NiryoWorkspace._set_observation_pose()`
2. Define observation pose
3. No code changes needed elsewhere

### Custom Object Queries

```python
class MyObjects(Objects):
    def get_objects_in_region(self, x_min, x_max, y_min, y_max):
        return Objects(
            obj for obj in self
            if x_min <= obj.x_com() <= x_max
            and y_min <= obj.y_com() <= y_max
        )
```

## Summary

The `robot_environment` architecture provides:

✅ **Modular Design** - Clear separation of concerns  
✅ **Hardware Abstraction** - Easy to add new robots  
✅ **Thread Safety** - Concurrent camera and control  
✅ **External Integration** - Clean package boundaries  
✅ **Redis Communication** - Decoupled data flow  
✅ **Flexible Workspaces** - Multiple workspace support  
✅ **Rich Object Representation** - Full spatial information  
✅ **Natural Interaction** - Text-to-speech feedback

This architecture enables robust pick-and-place operations with vision-based object detection while maintaining extensibility and clean code organization.