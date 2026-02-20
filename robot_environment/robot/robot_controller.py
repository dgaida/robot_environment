"""
Robot controller base class for robot_environment.
"""

# abstract class RobotController for the pnp_robot_genai package
# should be final
# Documentation and type definitions are final (class documentation could be improved with chatgpt)

from abc import ABC, abstractmethod

import threading

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_workspace import PoseObjectPNP
    from .robot import Robot


class RobotController(ABC):
    """
    An abstract class for the pick-and-place robot that provides the primitive tasks of the robot like
    pick and place operations.

    """

    # *** CONSTRUCTORS ***
    def __init__(self, robot: "Robot", use_simulation: bool, verbose: bool = False):
        """
        Initializes the robot (connects to it and calibrates it).

        Args:
            robot: object of the Robot class.
            use_simulation: True, if working with a simulation model of the robot,
            else False if we work with a real robot.
            verbose:
        """
        super().__init__()

        # Initialize the asyncio lock
        self._lock = threading.Lock()  # asyncio.Lock()

        self._verbose = verbose
        self._robot = robot
        self._in_motion = False

        self._init_robot(use_simulation)

    # Deleting (Calling destructor)
    def __del__(self):
        """
        Destructor for the RobotController.
        """
        pass

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    @abstractmethod
    def get_pose(self) -> "PoseObjectPNP":
        """
        Get current pose of gripper of robot.

        Returns:
            current pose of gripper of robot.
        """
        pass

    # *** PUBLIC methods ***

    @abstractmethod
    def calibrate(self) -> bool:
        """
        Calibrates the Robot.

        Returns:
            True, if calibration was successful, else False
        """
        pass

    # TODO: also possible to only pass PoseObject of the object. The advantage of passing Object might be
    #  that an object has more then one pick position. then robot can try and pick at a few positions.
    # @abstractmethod
    # def robot_pick_object(self, obj2pick: "Object") -> bool:
    #     """
    #     Calls the pick command of the self._robot_ctrl to pick the given Object
    #
    #     Args:
    #         obj2pick: Object that shall be picked
    #
    #     Returns:
    #         True, if pick was successful, else False
    #     """
    #     return False
    @abstractmethod
    def robot_pick_object(self, pick_pose: "PoseObjectPNP") -> bool:
        """
        Calls the pick command of the self._robot_ctrl to pick an object at the given pose.

        Args:
            pick_pose: Pose where the object should be picked (includes z-offset if needed)

        Returns:
            True, if pick was successful, else False
        """
        return False

    @abstractmethod
    def robot_place_object(self, place_pose: "PoseObjectPNP") -> bool:
        """
        Places an already picked object at the given place_pose.

        Args:
            place_pose: Pose where to place the already picked object

        Returns:
            True, if place was successful, else False
        """
        return False

    @abstractmethod
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

    @abstractmethod
    def get_target_pose_from_rel(self, workspace_id: str, u_rel: float, v_rel: float, yaw: float) -> "PoseObjectPNP":
        """
        Given relative image coordinates [u_rel, v_rel] and optionally an orientation of the point (yaw),
        calculate the corresponding pose in world coordinates. The parameter yaw is useful, if we want to pick at the
        given coordinate an object that has the given orientation. For this method to work, it is important that
        only the workspace of the robot is visible in the image and nothing else. At least for the Niryo robot
        this is important. This means, (u_rel, v_rel) = (0, 0), is the upper left corner of the workspace.

        Args:
            workspace_id: id of the workspace
            u_rel: horizontal coordinate in image of workspace, normalized between 0 and 1
            v_rel: vertical coordinate in image of workspace, normalized between 0 and 1
            yaw: orientation of an object at the pixel coordinates [u_rel, v_rel].

        Returns:
            pose_object: Pose of the point in world coordinates of the robot.
        """
        pass

    @abstractmethod
    def move2observation_pose(self, workspace_id: str) -> None:
        """
        The robot should move to a pose where it can observe the workspace given by workspace_id.

        Args:
            workspace_id: id of the workspace
        """
        pass

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    @abstractmethod
    def _init_robot(self, use_simulation: bool) -> bool:
        """
        Code to initialize the robot. After calling this method the robot should be ready to receive commands.

        Args:
            use_simulation: True, if working with a simulation model of the robot,
            else False if we work with a real robot.

        Returns:
            bool: True, if initialization was successful, else False
        """
        pass

    def _set_in_motion(self, in_motion: bool):
        """Set the robot motion state."""
        self._in_motion = in_motion
        # if hasattr(self._robot, "_robot_in_motion"):
        #     self._robot._robot_in_motion = in_motion

    # *** PUBLIC properties ***

    def is_in_motion(self) -> bool:
        """Check if robot is currently in motion."""
        return self._in_motion

    def robot_ctrl(self):
        """

        Returns:
            the object of the underlying robot. For the Niryo Ned2 it is an object of the NiryoRobot class.
        """
        return self._robot_ctrl

    def robot(self) -> "Robot":
        """
        Returns the robot object.

        Returns:
            Robot: The robot instance.
        """
        return self._robot

    def lock(self) -> threading.Lock:
        """

        Returns:
            a lock, to only call the robot interface with one method and not many methods in parallel, because for Niryo
            the interface is not thread safe.
        """
        return self._lock

    def verbose(self) -> bool:
        """

        Returns: True, if verbose is on, else False

        """
        return self._verbose

    # *** PRIVATE variables ***

    # the object of the underlying robot. For the Niryo Ned2 it is an object of the NiryoRobot class.
    _robot_ctrl = None

    # object of the Robot class
    _robot = None

    # a lock, to only call the robot interface with one method and not many methods in parallel, because for Niryo
    # the interface is not thread safe.
    _lock = None

    _verbose = False
