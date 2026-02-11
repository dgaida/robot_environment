# Robot Environment - Troubleshooting

Common issues and solutions for the Robot Environment system.

## Table of Contents

- [Object Detection Problems](#object-detection-problems)
- [Robot Movement Issues](#robot-movement-issues)
- [Hardware Problems](#hardware-problems)
- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
- [Getting Help](#getting-help)

---

## Frequently Asked Questions (FAQ)

### 1. Why do I get a `ModuleNotFoundError: No module named 'text2speech'`?
This usually means the `text2speech` package was not installed correctly or is not in your Python path.
Ensure you ran:
```bash
pip install git+https://github.com/dgaida/text2speech.git
```
If you are running in a virtual environment, make sure it is activated.

### 2. The robot moves to the wrong place or misses the object. What should I do?
First, check if the workspace is correctly calibrated. Use `env.get_workspace_by_id("your_ws_id").get_bounds()` to see the world coordinates the system is using.
Second, ensure you are using **fresh** detections. Always move to an observation pose and wait a second before calling `get_detected_objects()`.

### 3. How do I switch between simulation and real robot?
When initializing the `Environment` class, set `use_simulation=True` for Gazebo and `use_simulation=False` for the real hardware.
Note that the real Niryo robot requires a specific IP address (default is `192.168.0.140`).

### 4. Can I use this without a GPU?
Yes! While object detection is faster on a GPU, models like `yolo-world` or `yoloe-11s` can run on a standard CPU with reasonable performance for pick-and-place tasks.

### 5. Why is the camera feed delayed?
The camera feed is streamed via Redis. If you notice a lag, it might be due to network congestion (if using a real robot over Wi-Fi) or high CPU usage by the vision models. Try a lighter model or reduce the camera update frequency in the configuration.

---

## Object Detection Problems

### No Objects Detected

**Symptoms:**
- `get_detected_objects()` returns empty list
- Camera shows black screen
- "No objects detected" messages

**Solutions:**

1. **Verify camera is working:**
```python
from redis_robot_comm import RedisImageStreamer
streamer = RedisImageStreamer(stream_name="robot_camera")
img, metadata = streamer.get_latest_image()
print(f"Image shape: {img.shape}")  # Should be (480, 640, 3)
```

2. **Check Redis is running:**
```bash
docker ps | grep redis

# If not running:
docker run -p 6379:6379 redis:alpine
```

3. **Verify camera thread is started:**
```python
# In server initialization
env = Environment(
    ...
    start_camera_thread=True  # Must be True!
)
```

4. **Check lighting conditions:**
- Ensure workspace is well-lit
- Avoid shadows and glare
- Use consistent lighting

5. **Verify object labels:**
```python
labels = env.get_object_labels_as_string()
print(f"Recognizable objects: {labels}")

# Add custom labels if needed
env.add_object_name2object_labels("your_object")
```

---

### Objects Detected at Wrong Positions

**Symptoms:**
- Robot misses objects when picking
- Coordinates don't match visual position
- Objects appear shifted in camera view

**Solutions:**

1. **Recalibrate camera transformation:**
```python
# Check workspace calibration
workspace = env.get_workspace_by_id("niryo_ws")
print(f"Workspace bounds: {workspace.get_bounds()}")

# Verify transformation parameters
# May need to recalibrate camera-to-world transform
```

2. **Check workspace is level:**
- Ensure robot base is stable
- Workspace surface should be flat
- Check for tilting or movement

3. **Update detection immediately before pick:**
```python
# ✅ Good: Fresh detection
objects = get_detected_objects()
obj = objects[0]
pick_object(obj['label'], [obj['x'], obj['y']])

# ❌ Bad: Stale coordinates
pick_object("pencil", [0.15, -0.05])  # May have moved!
```

4. **Verify coordinate system understanding:**
```
Niryo workspace (top view):
    Y-axis →
    ┌─────────┐
    │         │
X ↓ │ Center  │
    │  (0,0)  │
    │         │
    └─────────┘
```

---

### Detection is Too Slow

**Symptoms:**
- Long delays before robot responds
- Camera updates lag behind
- Low FPS (< 1 frame/second)

**Solutions:**

1. **Use faster detection model:**
```python
# In Environment initialization
visual_cortex = VisualCortex(
    objdetect_model_id="yoloworld",  # Faster than owlv2
    device="cuda"  # Use GPU if available
)
```

2. **Reduce camera update rate:**
```python
# In camera thread
time.sleep(0.5)  # Update every 0.5s instead of 0.1s
```

3. **Check GPU availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# If no GPU:
# - Use CPU with yoloworld model
# - Or add GPU to system
```

4. **Optimize detection parameters:**
```python
config = {
    'confidence_threshold': 0.20,  # Higher = fewer false positives
    'iou_threshold': 0.5,
    'max_detections': 50  # Lower = faster
}
```

---

## Robot Movement Issues

### Robot Won't Move

**Symptoms:**
- Commands accepted but no movement
- Robot stays in same position
- "Movement failed" errors

**Solutions:**

1. **Check robot connection:**
```python
# For Niryo
robot = env.robot()
status = robot.robot_ctrl().get_hardware_status()
print(f"Robot connected: {status}")
```

2. **Verify simulation vs. real mode:**
```bash
# TODO

# --no-simulation flag for real robot
# Without flag = simulation mode
```

3. **Check robot power and calibration:**
- Ensure robot is powered on
- Run calibration routine if needed
- Check for error LEDs on robot

4. **Verify coordinates are reachable:**
```python
# Check workspace bounds
upper_left = get_workspace_coordinate_from_point("niryo_ws", "upper left corner")
lower_right = get_workspace_coordinate_from_point("niryo_ws", "lower right corner")

print(f"Valid X range: [{lower_right[0]}, {upper_left[0]}]")
print(f"Valid Y range: [{lower_right[1]}, {upper_left[1]}]")

# Niryo: X=[0.163, 0.337], Y=[-0.087, 0.087]
```

---

### Collision Detection Triggered

**Symptoms:**
- Robot stops suddenly
- "Collision detected" messages
- Robot needs reset before continuing

**Solutions:**

1. **Clear collision flag:**
```python
clear_collision_detected()
```

2. **Check workspace for obstacles:**
- Remove objects outside workspace
- Ensure cables aren't blocking movement
- Check gripper clearance

3. **Adjust movement parameters:**
```python
# In robot controller (if accessible)
robot_ctrl.set_collision_threshold(higher_value)
```

4. **Move to safe observation pose:**
```python
move2observation_pose("niryo_ws")
clear_collision_detected()
```

---

### Gripper Problems

**Symptoms:**
- Objects slip out of gripper
- Gripper doesn't close/open
- "Failed to grasp" errors

**Solutions:**

1. **Check object size:**
```python
obj = get_detected_object([x, y])
if obj['width_m'] > 0.05:
    print("Object too large for gripper!")
    # Use push_object() instead
```

2. **Verify gripper calibration:**
```python
# Test gripper
robot.robot_ctrl().open_gripper()
time.sleep(2)
robot.robot_ctrl().close_gripper()
```

3. **Check object graspability:**
- Objects should have flat surfaces
- Avoid round or irregular shapes
- Ensure objects aren't too heavy (< 500g)

4. **Adjust grasp approach angle:**
```python
# Object rotation affects grasp success
obj = get_detected_object([x, y])
print(f"Object rotation: {obj['rotation_rad']} rad")

# Robot adjusts approach automatically
```

---

## Hardware Problems

### Niryo Robot Specific

**Issue: Robot not responding**
```bash
# Check Niryo connection
ping <robot_ip>

# Default: 192.168.1.xxx
```

**Issue: Calibration needed**
```python
# Run calibration
robot.robot_ctrl().calibrate()
```

**Issue: Learning mode activated**
- Manually disable learning mode on robot
- Robot will be stiff when learning mode is off

---

### WidowX Robot Specific

**Issue: Joint limits**
```python
# WidowX has different workspace
# Adjust coordinates accordingly
```

**Issue: Power supply**
- Ensure adequate power (12V)
- Check for voltage drops during operation

---

### Camera Issues

**Issue: Poor image quality**
```python
# Adjust camera settings
camera.set(cv2.CAP_PROP_EXPOSURE, -7)
camera.set(cv2.CAP_PROP_BRIGHTNESS, 130)
```

**Issue: Wrong camera selected**
```python
# List available cameras
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
    cap.release()
```

---

## Getting Help

### Resources

- **GitHub Issues:** https://github.com/dgaida/robot_environment/issues
- **Documentation:** [README.md](../README.md)

---

## Quick Diagnostic Checklist

Before opening an issue, check:

- [ ] Redis is running
- [ ] Robot is powered on (if using real robot)
- [ ] Camera is working (check Redis stream)
- [ ] Object detection is running (check for detections)
- [ ] Coordinates are within workspace bounds
- [ ] Object names match detected labels exactly
- [ ] All dependencies are installed
- [ ] Log files checked for errors

If all checked and still having issues, please open a GitHub issue with the information above!
