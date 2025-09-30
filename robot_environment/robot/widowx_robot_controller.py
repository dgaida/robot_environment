# robot class around WidowX robot for smart pick and place
# TODO: has to be implemented

from .robot_controller import RobotController

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..objects.pose_object import PoseObjectPNP
    from ..objects.object import Object
    from .robot import Robot


class WidowXRobotController(RobotController):
    # *** CONSTRUCTORS ***
    def __init__(self, robot: "Robot", use_simulation: bool, verbose: bool = False):
        super().__init__(robot, use_simulation, verbose)

    # Deleting (Calling destructor)
    def __del__(self):
        pass

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    def get_pose(self) -> "PoseObjectPNP":
        """
        Get current pose of gripper of robot.

        Returns:
            current pose of gripper of robot.
        """
        pass

    # *** PUBLIC methods ***

    def robot_pick_object(self, obj2pick: "Object") -> bool:
        pass

    def robot_place_object(self, place_pose: "PoseObjectPNP") -> bool:
        pass

    def robot_push_object(self, push_pose: "PoseObjectPNP", direction: str, distance: float) -> bool:
        """
        Push given object (its Pose) into the given direction by the given distance.

        Args:
            push_pose: the Pose of the object that should be pushed.
            direction: "up", "down", "left", "right"
            distance: distance in millimeters

        Returns:
            True, if push was successful, else False
        """
        pass

    def get_target_pose_from_rel(self, workspace_id: str, x_rel: float, y_rel: float, yaw: float) -> "PoseObjectPNP":
        pass

    def move2observation_pose(self, workspace_id: str) -> None:
        pass

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    def _init_robot(self, use_simulation: bool) -> bool:
        return False

    # *** PUBLIC properties ***

    # *** PRIVATE variables ***
