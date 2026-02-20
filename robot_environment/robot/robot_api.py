"""
Robot API definitions for robot_environment.
"""
# abstract robot class around Niryo robot for smart pick and place - defines the functions that the LLM may call
# a few TODOs
# Documentation and type definitions are almost final (chatgpt can maybe improve it) - however, documentation in
# robot.py is more up-to-date.

from typing import List, Union
from abc import ABC, abstractmethod

from robot_workspace import Location


class RobotAPI(ABC):
    """
    Abstract class that defines the functions that the LLM may call
    """

    # *** CONSTRUCTORS ***
    def __init__(self):
        """
        Initialize the RobotAPI.
        """
        pass

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    @abstractmethod
    def pick_place_object(
        self,
        object_name: str,
        pick_coordinate: List,
        place_coordinate: List,
        location: Union["Location", str, None] = None,
        z_offset: float = 0.001,
    ) -> bool:
        """
        Instructs the pick-and-place robot arm to pick a specific object and place it using its gripper.
        The gripper will move to the specified 'pick_coordinate' and pick the named object. It will then move to the
        specified 'place_coordinate' and place the object there. If you need to pick-and-place an object, call this
        function and not robot_pick_object() followed by robot_place_object().

        Example call:

        robot.pick_place_object(
            object_name='chocolate bar',
            pick_coordinate=[-0.1, 0.01],
            place_coordinate=[0.1, 0.11],
            location=Location.RIGHT_NEXT_TO
        )
        --> Picks the chocolate bar that is located at world coordinates [-0.1, 0.01] and places it right next to an
        object that exists at world coordinate [0.1, 0.11].

        robot.pick_place_object(
            object_name='cube',
            pick_coordinate=[0.2, 0.05],
            place_coordinate=[0.3, 0.1],
            location=Location.ON_TOP_OF,
            z_offset=0.02
        )
        --> Picks the cube with a 2cm z-offset (useful if it's on top of another object).

        Args:
            object_name (str): The name of the object to be picked up. Ensure this name matches an object visible in
            the robot's workspace.
            pick_coordinate (List): The world coordinates [x, y] where the object should be picked up. Use these
            coordinates to identify the object's exact position.
            place_coordinate (List): The world coordinates [x, y] where the object should be placed at.
            location (Location): Specifies the relative placement position of the picked object with respect to an object
            being at the 'place_coordinate'. Possible values are defined in the `Location` Enum:
                - `Location.LEFT_NEXT_TO`: Left of the reference object.
                - `Location.RIGHT_NEXT_TO`: Right of the reference object.
                - `Location.ABOVE`: Above the reference object.
                - `Location.BELOW`: Below the reference object.
                - `Location.ON_TOP_OF`: On top of the reference object.
                - `Location.INSIDE`: Inside the reference object.
                - `Location.NONE`: No specific location relative to another object.
            z_offset (float): Additional height offset in meters to apply when picking (default: 0.001).
            Useful for picking objects that are stacked on top of other objects.

        Returns:
            bool: Always returns `True` after the pick-and-place operation.
        """
        return True

    @abstractmethod
    def pick_object(self, object_name: str, pick_coordinate: List, z_offset: float = 0.001) -> bool:
        """
        Command the pick-and-place robot arm to pick up a specific object using its gripper. The gripper will move to
        the specified 'pick_coordinate' and pick the named object.

        Example call:

        robot.pick_object("pen", [0.01, -0.15])
        --> Picks the pen that is located at world coordinates [0.01, -0.15].

        robot.pick_object("pen", [0.01, -0.15], z_offset=0.02)
        --> Picks the pen with a 2cm offset above its detected position (useful for stacked objects).

        Args:
            object_name (str): The name of the object to be picked up. Ensure this name matches an object visible in
            the robot's workspace.
            pick_coordinate (List): The world coordinates [x, y] where the object should be picked up. Use these
            coordinates to identify the object's exact position.
            z_offset (float): Additional height offset in meters to apply when picking (default: 0.001).
            Useful for picking objects that are stacked on top of other objects.
        Returns:
            bool: True
        """
        return False

    @abstractmethod
    def place_object(self, place_coordinate: List, location: Union["Location", str, None] = None) -> bool:
        """
        Instruct the pick-and-place robot arm to place a picked object at the specified 'place_coordinate'. The
        function moves the gripper to the specified 'place_coordinate' and calculates the exact placement position from
        the given 'location'. Before calling this function you have to call robot_pick_object() to pick an object.

        Example call:

        robot.place_object([0.2, 0.0], "left next to")
        --> Places the already gripped object left next to the world coordinate [0.2, 0.0].

        Args:
            place_coordinate: The world coordinates [x, y] of the target object.
            location (str): Specifies the relative placement position of the picked object in relation to an object
            being at the 'place_coordinate'. Possible positions: 'left next to', 'right next to', 'above', 'below',
            'on top of', 'inside', or None. Set to None, if there is no location given in the task.
        Returns:
            bool: True
        """

        return True

    @abstractmethod
    def push_object(self, object_name: str, push_coordinate: List, direction: str, distance: float):
        """
        Direct the pick-and-place robot arm to push a specific object to a new position.
        This function should only be called if it is not possible to pick the object.
        An object cannot be picked if its shorter side is larger than the gripper.

        Args:
            object_name (str): The name of the object to be pushed.
            Ensure the name matches an object in the robot's environment.
            push_coordinate: The world coordinates [x, y] where the object to be pushed is located.
            These coordinates indicate the initial position of the object.
            direction (str): The direction in which the object should be pushed.
            Valid options are: "up", "down", "left", "right".
            distance: The distance (in millimeters) to push the object in the specified direction.
            Ensure the value is within the robot's operational range.
        Returns:
            bool: True
        """
        return False

    def move2observation_pose(self, workspace_id: str) -> None:
        """
        The robot will move to a pose where it can observe (the gripper hovers over) the workspace given by workspace_id.
        Before a robot can pick up or place an object in a workspace, it must first move to this observation pose of the corresponding workspace.

        Args:
            workspace_id: id of the workspace

        Returns:
            None
        """
        pass

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    # *** PRIVATE variables ***
