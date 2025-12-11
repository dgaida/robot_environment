from robot_environment.config import ConfigManager, WorkspaceConfig
from pathlib import Path
from robot_environment import Environment


# Load and customize
manager = ConfigManager()
config = manager.load(robot_type="niryo")

# Add custom workspace
config.workspaces["custom"] = WorkspaceConfig(
    id="custom_ws",
    observation_pose={"x": 0.25, "y": 0.05, "z": 0.30, "roll": 0.0, "pitch": 1.57, "yaw": 0.0},
    upper_left={"x": 0.35, "y": 0.15},
    lower_right={"x": 0.15, "y": -0.05},
)

# Adjust vision settings
config.vision.confidence_threshold = 0.3
config.vision.object_labels.append("screwdriver")

# Save customized config
manager.save(Path("my_custom_config.yaml"))

# Use it
env = Environment(config=config)
