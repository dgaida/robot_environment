from robot_environment import Environment
from robot_environment.config import RobotConfig

# Option 1: Use defaults
env = Environment()

# Option 2: Load from file
env = Environment(config_file="config/my_robot.yaml")

# Option 3: Create custom config
config = RobotConfig.get_default_niryo()
config.verbose = True
config.vision.detection_model = "yolo-world"
env = Environment(config=config)

# Option 4: Mix file and overrides
env = Environment(config_file="config/niryo.yaml", verbose=True, el_api_key="your_key")  # Override
