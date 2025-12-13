# test_config.py
# import pytest
from robot_environment.config import RobotConfig, ConfigManager

# from pathlib import Path


def test_default_configs():
    """Test default configurations."""
    niryo_config = RobotConfig.get_default_niryo()
    assert niryo_config.robot_type == "niryo"
    assert len(niryo_config.workspaces) >= 1

    widowx_config = RobotConfig.get_default_widowx()
    assert widowx_config.robot_type == "widowx"


def test_config_serialization(tmp_path):
    """Test saving and loading configs."""
    config = RobotConfig.get_default_niryo()
    config.verbose = True

    # Save and load YAML
    yaml_path = tmp_path / "test.yaml"
    config.to_yaml(yaml_path)
    loaded = RobotConfig.from_yaml(yaml_path)
    assert loaded.verbose
    assert loaded.robot_id == config.robot_id


def test_config_manager():
    """Test configuration manager."""
    manager = ConfigManager()
    config = manager.load(robot_type="niryo")
    assert config.robot_type == "niryo"

    # Update
    manager.update(verbose=True)
    assert manager.get().verbose


def test_workspace_config():
    """Test workspace configuration."""
    config = RobotConfig.get_default_niryo()
    assert "niryo_ws" in config.workspaces

    ws = config.workspaces["niryo_ws"]
    assert ws.id == "niryo_ws"
    assert "x" in ws.observation_pose
