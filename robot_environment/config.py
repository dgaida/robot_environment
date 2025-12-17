"""
Configuration management system for robot_environment package.

This module provides a centralized configuration system that replaces
scattered hardcoded values throughout the codebase.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import json
from enum import Enum


class RobotType(str, Enum):
    """Supported robot types."""

    NIRYO = "niryo"
    WIDOWX = "widowx"


# Custom YAML representer for RobotType enum
def represent_robot_type(dumper, data):
    """Custom YAML representer for RobotType enum"""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data.value)


yaml.add_representer(RobotType, represent_robot_type)


class DetectionModel(str, Enum):
    """Supported object detection models."""

    YOLOE_11L = "yoloe-11l"
    YOLOE_11S = "yoloe-11s"
    YOLO_WORLD = "yolo-world"
    OWL_V2 = "owlv2"
    GROUNDING_DINO = "grounding_dino"


# Custom YAML representer for DetectionModel enum
def represent_detection_model(dumper, data):
    """Custom YAML representer for DetectionModel enum"""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data.value)


yaml.add_representer(DetectionModel, represent_detection_model)


@dataclass
class WorkspaceConfig:
    """Configuration for a single workspace."""

    id: str
    observation_pose: Dict[str, float]  # x, y, z, roll, pitch, yaw
    upper_left: Dict[str, float]  # x, y world coordinates
    lower_right: Dict[str, float]  # x, y world coordinates
    height_m: float = 0.0  # workspace height above table

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CameraConfig:
    """Camera configuration."""

    resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    # Camera intrinsics (can be None, will use robot-specific defaults)
    camera_matrix: Optional[List[List[float]]] = None
    distortion_coeffs: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraConfig":
        """Create from dictionary."""
        # Convert resolution tuple if it's a list
        if "resolution" in data and isinstance(data["resolution"], list):
            data["resolution"] = tuple(data["resolution"])
        return cls(**data)


@dataclass
class VisionConfig:
    """Vision system configuration."""

    detection_model: str = DetectionModel.YOLOE_11L
    confidence_threshold: float = 0.2
    enable_segmentation: bool = True
    enable_tracking: bool = True
    object_labels: List[str] = field(default_factory=lambda: ["pen", "pencil", "cube", "cylinder", "box", "bottle"])
    redis_stream_name: str = "robot_camera"
    redis_objects_channel: str = "robot_objects"
    redis_labels_channel: str = "robot_labels"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RobotControlConfig:
    """Robot control parameters."""

    # Pick and place offsets
    pick_z_offset_m: float = 0.001
    place_z_offset_m: float = 0.005
    approach_height_m: float = 0.05

    # Placement offsets relative to reference objects
    placement_offset_x_m: float = 0.02
    placement_offset_y_m: float = 0.02
    stack_offset_z_m: float = 0.02

    # Push operation parameters
    push_speed_mm_s: float = 50.0

    # Safety parameters
    max_velocity: float = 0.1  # m/s
    max_acceleration: float = 0.5  # m/s²

    # Timeouts
    move_timeout_s: float = 30.0
    pick_timeout_s: float = 15.0
    place_timeout_s: float = 15.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotControlConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MemoryConfig:
    """Object memory management configuration."""

    position_tolerance_m: float = 0.05  # Consider objects at same position if within this
    manual_update_timeout_s: float = 5.0  # Keep manual updates for this long
    enable_multi_workspace: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RedisConfig:
    """Redis connection configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedisConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TTSConfig:
    """Text-to-speech configuration."""

    provider: str = "elevenlabs"  # "elevenlabs" or "kokoro"
    api_key: Optional[str] = None
    voice_id: Optional[str] = None
    enable_speech: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TTSConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RobotConfig:
    """Complete robot system configuration."""

    # Robot identification
    robot_id: str = RobotType.NIRYO
    robot_type: str = RobotType.NIRYO

    # Connection
    ip_address: str = "192.168.0.140"
    simulation_ip_address: str = "192.168.247.128"
    use_simulation: bool = False

    # Workspaces
    workspaces: Dict[str, WorkspaceConfig] = field(default_factory=dict)

    # Subsystem configurations
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    control: RobotControlConfig = field(default_factory=RobotControlConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)

    # System settings
    verbose: bool = False
    start_camera_thread: bool = True
    camera_update_rate_hz: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert workspace configs
        data["workspaces"] = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in self.workspaces.items()}

        if isinstance(data.get("robot_type"), RobotType):
            data["robot_type"] = data["robot_type"].value

        return data

    def to_yaml(self, filepath: Path) -> None:
        """Save configuration to YAML file."""
        with open(filepath, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, filepath: Path) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotConfig":
        """Create from dictionary."""
        # Handle nested dataclasses
        if "workspaces" in data:
            data["workspaces"] = {
                k: WorkspaceConfig.from_dict(v) if isinstance(v, dict) else v for k, v in data["workspaces"].items()
            }

        if "camera" in data and isinstance(data["camera"], dict):
            data["camera"] = CameraConfig.from_dict(data["camera"])

        if "vision" in data and isinstance(data["vision"], dict):
            data["vision"] = VisionConfig.from_dict(data["vision"])

        if "control" in data and isinstance(data["control"], dict):
            data["control"] = RobotControlConfig.from_dict(data["control"])

        if "memory" in data and isinstance(data["memory"], dict):
            data["memory"] = MemoryConfig.from_dict(data["memory"])

        if "redis" in data and isinstance(data["redis"], dict):
            data["redis"] = RedisConfig.from_dict(data["redis"])

        if "tts" in data and isinstance(data["tts"], dict):
            data["tts"] = TTSConfig.from_dict(data["tts"])

        return cls(**data)

    @classmethod
    def from_yaml(cls, filepath: Path) -> "RobotConfig":
        """Load configuration from YAML file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, filepath: Path) -> "RobotConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def get_default_niryo(cls) -> "RobotConfig":
        """Get default configuration for Niryo Ned2."""
        config = cls(
            robot_id="niryo", robot_type=RobotType.NIRYO, ip_address="192.168.0.140", simulation_ip_address="192.168.247.128"
        )

        # Add default Niryo workspaces
        config.workspaces = {
            "niryo_ws": WorkspaceConfig(
                id="niryo_ws",
                observation_pose={"x": 0.20, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.57, "yaw": 0.0},
                upper_left={"x": 0.35, "y": 0.15},
                lower_right={"x": 0.15, "y": -0.15},
            ),
            "niryo_ws_left": WorkspaceConfig(
                id="niryo_ws_left",
                observation_pose={"x": 0.15, "y": 0.20, "z": 0.35, "roll": 0.0, "pitch": 1.57, "yaw": 0.0},
                upper_left={"x": 0.28, "y": 0.30},
                lower_right={"x": 0.10, "y": 0.10},
            ),
            "niryo_ws_right": WorkspaceConfig(
                id="niryo_ws_right",
                observation_pose={"x": 0.15, "y": -0.20, "z": 0.35, "roll": 0.0, "pitch": 1.57, "yaw": 0.0},
                upper_left={"x": 0.28, "y": -0.10},
                lower_right={"x": 0.10, "y": -0.30},
            ),
        }

        return config

    @classmethod
    def get_default_widowx(cls) -> "RobotConfig":
        """Get default configuration for WidowX robot."""
        config = cls(robot_id="widowx", robot_type=RobotType.WIDOWX, ip_address="")  # WidowX uses ROS, no direct IP

        # Add default WidowX workspace
        config.workspaces = {
            "widowx_ws": WorkspaceConfig(
                id="widowx_ws",
                observation_pose={"x": 0.30, "y": 0.0, "z": 0.20, "roll": 0.0, "pitch": 1.57, "yaw": 0.0},
                upper_left={"x": 0.40, "y": 0.20},
                lower_right={"x": 0.20, "y": -0.20},
            )
        }

        # WidowX uses RealSense camera with different intrinsics
        config.camera = CameraConfig(
            resolution=(640, 480),
            camera_matrix=[[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]],
            distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
        )

        return config


# Configuration manager singleton
class ConfigManager:
    """Manages configuration loading and access."""

    _instance: Optional["ConfigManager"] = None
    _config: Optional[RobotConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, source: Optional[Path] = None, robot_type: Optional[str] = None) -> RobotConfig:
        """
        Load configuration from file or defaults.

        Args:
            source: Path to config file (YAML or JSON)
            robot_type: Robot type for default config ("niryo" or "widowx")

        Returns:
            RobotConfig instance
        """
        if source is not None:
            source = Path(source)
            if source.suffix in [".yaml", ".yml"]:
                self._config = RobotConfig.from_yaml(source)
            elif source.suffix == ".json":
                self._config = RobotConfig.from_json(source)
            else:
                raise ValueError(f"Unsupported config format: {source.suffix}")
        elif robot_type is not None:
            if robot_type.lower() == "niryo":
                self._config = RobotConfig.get_default_niryo()
            elif robot_type.lower() == "widowx":
                self._config = RobotConfig.get_default_widowx()
            else:
                raise ValueError(f"Unknown robot type: {robot_type}")
        else:
            # Default to Niryo
            self._config = RobotConfig.get_default_niryo()

        return self._config

    def get(self) -> RobotConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = RobotConfig.get_default_niryo()
        return self._config

    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        if self._config is None:
            self._config = RobotConfig.get_default_niryo()

        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def save(self, filepath: Path, format: str = "yaml") -> None:
        """Save current configuration to file."""
        if self._config is None:
            raise ValueError("No configuration loaded")

        filepath = Path(filepath)
        if format.lower() == "yaml":
            self._config.to_yaml(filepath)
        elif format.lower() == "json":
            self._config.to_json(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience function
def get_config() -> RobotConfig:
    """Get current configuration from manager."""
    return ConfigManager().get()


# Example usage and documentation
if __name__ == "__main__":
    # Example 1: Use default configuration
    config = RobotConfig.get_default_niryo()
    print("Default Niryo config:")
    print(f"  Robot: {config.robot_id}")
    print(f"  IP: {config.ip_address}")
    print(f"  Workspaces: {list(config.workspaces.keys())}")
    print(f"  Vision model: {config.vision.detection_model}")

    # Example 2: Save configuration to file
    config.to_yaml(Path("robot_config.yaml"))
    print("\nSaved to robot_config.yaml")

    # Example 3: Load from file
    loaded_config = RobotConfig.from_yaml(Path("robot_config.yaml"))
    print(f"\nLoaded config robot: {loaded_config.robot_id}")

    # Example 4: Use ConfigManager
    manager = ConfigManager()
    manager.load(robot_type="niryo")
    current_config = manager.get()

    # Update parameters
    manager.update(verbose=True, use_simulation=True)

    # Example 5: Create custom configuration
    custom_config = RobotConfig(robot_id="niryo_lab", ip_address="192.168.1.100", use_simulation=False, verbose=True)

    # Add custom workspace
    custom_config.workspaces["custom_ws"] = WorkspaceConfig(
        id="custom_ws",
        observation_pose={"x": 0.25, "y": 0.0, "z": 0.30, "roll": 0.0, "pitch": 1.57, "yaw": 0.0},
        upper_left={"x": 0.35, "y": 0.20},
        lower_right={"x": 0.15, "y": -0.20},
    )

    # Customize vision settings
    custom_config.vision.detection_model = DetectionModel.YOLO_WORLD
    custom_config.vision.confidence_threshold = 0.3
    custom_config.vision.object_labels = ["pen", "cube", "bottle"]

    print("\nCustom config created with:")
    print(f"  {len(custom_config.workspaces)} workspaces")
    print(f"  Model: {custom_config.vision.detection_model}")
    print(f"  Labels: {custom_config.vision.object_labels}")
