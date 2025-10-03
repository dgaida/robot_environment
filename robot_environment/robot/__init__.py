"""
Robot control module.

Provides abstract and concrete implementations for different robotic arms
including Niryo Ned2 and WidowX.
"""

# from .robot import Robot
from .robot_controller import RobotController
from .robot_api import RobotAPI, Location

try:
    from .niryo_robot_controller import NiryoRobotController
except ImportError:
    NiryoRobotController = None

try:
    from .widowx_robot_controller import WidowXRobotController
except ImportError:
    WidowXRobotController = None

__all__ = [
    "Robot",
    "RobotController",
    "RobotAPI",
    "Location",
    "NiryoRobotController",
    "WidowXRobotController",
]
