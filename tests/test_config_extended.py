"""
Extended unit tests for config.py
"""

import pytest
import yaml
import json
from robot_environment.config import (
    RobotConfig,
    ConfigManager,
    RobotType,
    DetectionModel,
    WorkspaceConfig,
    CameraConfig,
    VisionConfig,
    RobotControlConfig,
    MemoryConfig,
    RedisConfig,
    TTSConfig,
)


class TestConfigExtended:
    """Test suite for config system coverage"""

    def test_yaml_representers(self):
        """Test custom YAML representers for enums (lines 26, 45)"""
        data = {"robot": RobotType.NIRYO, "model": DetectionModel.YOLOE_11L}
        yaml_str = yaml.dump(data)
        assert "niryo" in yaml_str
        assert "yoloe-11l" in yaml_str

    def test_workspace_config_from_dict(self):
        """Test WorkspaceConfig.from_dict (line 83)"""
        data = {
            "id": "test",
            "observation_pose": {"x": 1, "y": 2, "z": 3, "roll": 0, "pitch": 0, "yaw": 0},
            "upper_left": {"x": 10, "y": 10},
            "lower_right": {"x": 0, "y": 0},
        }
        config = WorkspaceConfig.from_dict(data)
        assert config.id == "test"

    def test_camera_config_from_dict(self):
        """Test CameraConfig.from_dict (line 109)"""
        data = {"resolution": [1280, 720], "fps": 60}
        config = CameraConfig.from_dict(data)
        assert config.resolution == (1280, 720)
        assert config.fps == 60

    def test_vision_config_from_dict(self):
        """Test VisionConfig.from_dict (line 145)"""
        data = {"detection_model": "owlv2"}
        config = VisionConfig.from_dict(data)
        assert config.detection_model == "owlv2"

    def test_robot_control_config_from_dict(self):
        """Test RobotControlConfig.from_dict (line 163)"""
        data = {"pick_z_offset_m": 0.1}
        config = RobotControlConfig.from_dict(data)
        assert config.pick_z_offset_m == 0.1

    def test_memory_config_from_dict(self):
        """Test MemoryConfig.from_dict (line 183)"""
        data = {"position_tolerance_m": 0.1}
        config = MemoryConfig.from_dict(data)
        assert config.position_tolerance_m == 0.1

    def test_redis_config_from_dict(self):
        """Test RedisConfig.from_dict (line 202)"""
        data = {"host": "test_host"}
        config = RedisConfig.from_dict(data)
        assert config.host == "test_host"

    def test_tts_config_from_dict(self):
        """Test TTSConfig.from_dict"""
        data = {"provider": "kokoro"}
        config = TTSConfig.from_dict(data)
        assert config.provider == "kokoro"

    def test_robot_config_to_json(self, tmp_path):
        """Test RobotConfig.to_json (lines 302-304)"""
        config = RobotConfig.get_default_niryo()
        json_file = tmp_path / "test.json"
        config.to_json(json_file)
        assert json_file.exists()
        with open(json_file, "r") as f:
            data = json.load(f)
            assert data["robot_id"] == "niryo"

    def test_robot_config_to_dict_branches(self):
        """Test branches in to_dict (lines 260-261)"""
        config = RobotConfig.get_default_niryo()
        config.robot_type = RobotType.NIRYO
        config.vision.detection_model = DetectionModel.YOLOE_11L
        data = config.to_dict()
        assert data["robot_type"] == "niryo"
        assert data["vision"]["detection_model"] == "yoloe-11l"

    def test_robot_config_from_yaml_json(self, tmp_path):
        """Test from_yaml and from_json (lines 386-392, 396-402)"""
        config = RobotConfig.get_default_niryo()
        yaml_file = tmp_path / "test.yaml"
        json_file = tmp_path / "test.json"
        config.to_yaml(yaml_file)
        config.to_json(json_file)

        c1 = RobotConfig.from_yaml(yaml_file)
        c2 = RobotConfig.from_json(json_file)
        assert c1.robot_id == "niryo"
        assert c2.robot_id == "niryo"

    def test_config_manager_load_branches(self, tmp_path):
        """Test ConfigManager.load branches (lines 409-438)"""
        manager = ConfigManager()

        # Test loading by robot_type
        c1 = manager.load(robot_type="niryo")
        assert c1.robot_type == "niryo"
        c2 = manager.load(robot_type="widowx")
        assert c2.robot_type == "widowx"

        with pytest.raises(ValueError, match="Unknown robot type"):
            manager.load(robot_type="invalid")

        # Test loading by source
        yaml_file = tmp_path / "config.yaml"
        RobotConfig.get_default_niryo().to_yaml(yaml_file)
        c3 = manager.load(source=yaml_file)
        assert c3.robot_type == "niryo"

        json_file = tmp_path / "config.json"
        RobotConfig.get_default_niryo().to_json(json_file)
        c4 = manager.load(source=json_file)
        assert c4.robot_type == "niryo"

        # Test invalid extension
        invalid_file = tmp_path / "config.txt"
        invalid_file.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported config format"):
            manager.load(source=invalid_file)

    def test_config_manager_save_unsupported(self, tmp_path):
        """Test ConfigManager.save with unsupported format (line 468 - wait, report said 468 for environment, let me check config)"""
        manager = ConfigManager()
        manager.load(robot_type="niryo")
        with pytest.raises(ValueError, match="Unsupported format"):
            manager.save(tmp_path / "test.txt", format="txt")

    def test_get_default_widowx(self):
        """Test get_default_widowx coverage"""
        config = RobotConfig.get_default_widowx()
        assert config.robot_type == "widowx"
        assert config.camera.camera_matrix is not None
