# robot class around Niryo robot for smart pick and place
# Updated with proper logging throughout

from ..common.logger import log_start_end_cls
from ..common.logger_config import get_package_logger

from .robot_api import RobotAPI, Location

from .niryo_robot_controller import NiryoRobotController

from robot_workspace import PoseObjectPNP
from robot_workspace import Object
from robot_workspace import Objects

from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from ..environment import Environment
    from .robot_controller import RobotController
    from robot_workspace import Object

import math
import re

# import json
import ast


class Robot(RobotAPI):
    # *** CONSTRUCTORS ***
    @log_start_end_cls()
    def __init__(
        self, environment: "Environment", use_simulation: bool = False, robot_id: str = "niryo", verbose: bool = False
    ):
        """
        Creates robot object. Creates these objects:
        - RobotController

        Args:
            environment:
            use_simulation: if True, then simulate the robot, else the real robot is used
            robot_id: string defining the robot. can be "niryo" or "widowx"
            verbose:
        """
        super().__init__()

        self._environment = environment
        self._verbose = verbose
        self._object_last_picked = None

        self._logger = get_package_logger(__name__, verbose)
        self._logger.info(f"Initializing robot: {robot_id}")

        if robot_id == "niryo":
            self._robot = NiryoRobotController(self, use_simulation, verbose)
        else:
            self._robot = None

    def handle_object_detection(self, objects_dict_list):
        """Process incoming object detections from Redis"""
        # Convert dictionaries back to Object instances
        objects = Objects.dict_list_to_objects(objects_dict_list, self.environment().get_workspace(0))

        # Now work with Object instances as before
        for obj in objects:
            self._logger.debug(f"Received object: {obj.label()} at {obj.xy_com()}")

    def get_pose(self) -> "PoseObjectPNP":
        """
        Get current pose of gripper of robot.

        Returns:
            current pose of gripper of robot.
        """
        return self._robot.get_pose()

    @log_start_end_cls()
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
        self._logger.debug(f"robot::get_target_pose_from_rel {workspace_id}, {u_rel}, {v_rel}, {yaw}")

        return self._robot.get_target_pose_from_rel(workspace_id, u_rel, v_rel, yaw)

    # *** PUBLIC methods ***

    def calibrate(self) -> bool:
        """
        Calibrates the Robot.

        Returns:
            True, if calibration was successful, else False
        """
        return self._robot.calibrate()

    @log_start_end_cls()
    def move2observation_pose(self, workspace_id: str) -> None:
        """
        The robot will move to a pose where it can observe (the gripper hovers over) the workspace given by workspace_id.
        Before a robot can pick up or place an object in a workspace, it must first move to this observation pose of the corresponding workspace.

        Args:
            workspace_id: id of the workspace

        Returns:
            None
        """
        self._robot.move2observation_pose(workspace_id)

    # TODO: the documentation of these pick methods is more upto date as teh one in robot_api

    @log_start_end_cls()
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
            bool: True if successful
        """
        success = self.pick_object(object_name, pick_coordinate, z_offset=z_offset)

        if success:
            place_success = self.place_object(place_coordinate, location)
            return place_success
        else:
            return False

    @log_start_end_cls()
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
        coords_str = "[" + ", ".join(f"{x:.2f}" for x in pick_coordinate) + "]"
        message = f"Going to pick {object_name} at coordinate {coords_str}."
        self._logger.info(message)

        self.environment().oralcom_call_text2speech_async(message, priority=8)

        obj_to_pick = self._get_nearest_object(object_name, pick_coordinate)

        if obj_to_pick:
            self._object_last_picked = obj_to_pick

            # Apply z_offset to the pick pose
            pick_pose = obj_to_pick.pose_com()
            pick_pose = pick_pose.copy_with_offsets(z_offset=z_offset)

            success = self._robot.robot_pick_object(pick_pose)
        else:
            success = False

        # thread_oral.join()

        return success

    @log_start_end_cls()
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
        location = Location.convert_str2location(location)

        if self._object_last_picked:
            old_coordinate = [self._object_last_picked.x_com(), self._object_last_picked.y_com()]
            message = (
                f"Going to place {self._object_last_picked.label()} {location} coordinate ["
                f"{place_coordinate[0]:.2f}, {place_coordinate[1]:.2f}]."
            )
        else:
            old_coordinate = None
            message = f"Going to place it {location} coordinate [{place_coordinate[0]:.2f}, {place_coordinate[1]:.2f}]."

        self._logger.info(message)

        self.environment().oralcom_call_text2speech_async(message, priority=8)
        obj_where_to_place = None

        if location is not None and location is not Location.NONE:
            obj_where_to_place = self._get_nearest_object(None, place_coordinate)
            if obj_where_to_place is None:
                place_pose = PoseObjectPNP(place_coordinate[0], place_coordinate[1], 0.09, 0.0, 1.57, 0.0)
            else:
                place_pose = obj_where_to_place.pose_center()
        else:
            place_pose = PoseObjectPNP(place_coordinate[0], place_coordinate[1], 0.09, 0.0, 1.57, 0.0)
            self._logger.debug(f"place_object: {place_pose}")

        x_off = 0.02
        y_off = 0.02

        if self._object_last_picked:
            x_off += self._object_last_picked.height_m() / 2
            y_off += self._object_last_picked.width_m() / 2

        if place_pose:
            # TODO: use height of object instead
            if location == Location.ON_TOP_OF:
                place_pose.z += 0.02
            elif location == Location.INSIDE:
                place_pose.z += 0.01
            elif location == Location.RIGHT_NEXT_TO:
                place_pose.y -= obj_where_to_place.width_m() / 2 + y_off
            elif location == Location.LEFT_NEXT_TO:
                place_pose.y += obj_where_to_place.width_m() / 2 + y_off
            elif location == Location.BELOW:
                # print(obj_where_to_place.height_m(), self._object_last_picked.width_m(), x_off)
                # TODO: nutze hier auch width, da width immer die größere größe ist und nicht eine koordinatenrichtugn hat
                #  ich muss anstatt width und height eine größe haben dim_x und dim_y, die a x und y koordinate gebunden sind
                #  ich habe das in object klasse repariert, width geht immer entlang y-achse jetzt. prüfen hier
                place_pose.x -= obj_where_to_place.height_m() / 2 + x_off
            elif location == Location.ABOVE:
                # TODO: nutze hier auch width, da width immer die größere größe ist und nicht eine koordinatenrichtugn hat
                #  ich habe das in object klasse repariert, width geht immer entlang y-achse jetzt. prüfen hier
                print(obj_where_to_place.height_m(), self._object_last_picked.width_m(), x_off)
                place_pose.x += obj_where_to_place.height_m() / 2 + x_off
                self._logger.debug(f"{place_pose}")
            elif location is Location.NONE or location is None:
                pass  # I do not have to do anything as the given location is where to place the object
            else:
                self._logger.error(f"Unknown location: {location} (type: {type(location)})")

            success = self._robot.robot_place_object(place_pose)

            # update position of placed object to the new position
            # Update memory after successful placement
            if success and self._object_last_picked and old_coordinate:
                final_coordinate = [place_pose.x, place_pose.y]
                self._logger.debug(f"final_coordinate: {final_coordinate}")

                self.environment().update_object_in_memory(
                    self._object_last_picked.label(), old_coordinate, new_pose=place_pose
                )

                # Give the memory system a moment to register the update
                import time

                time.sleep(0.1)
        else:
            success = False

        self._object_last_picked = None

        # thread_oral.join()

        return success

    @log_start_end_cls()
    def push_object(self, object_name: str, push_coordinate: List, direction: str, distance: float) -> bool:
        """
        Instruct the pick-and-place robot arm to push a specific object to a new position.
        This function should only be called if it is not possible to pick the object.
        An object cannot be picked if its shorter side is larger than the gripper.

        Args:
            object_name (str): The name of the object to be pushed.
            Ensure the name matches an object in the robot's environment.
            push_coordinate: The world coordinates [x, y] where the object to push is located.
            These coordinates indicate the initial position of the object.
            direction (str): The direction in which to push the object.
            Valid options are: "up", "down", "left", "right".
            distance: The distance (in millimeters) to push the object in the specified direction.
            Ensure the value is within the robot's operating range.

        Returns:
            bool: True
        """
        message = f"Calling push with {object_name} and {direction}"
        self._logger.info(message)

        self.environment().oralcom_call_text2speech_async(message, priority=8)

        obj_to_push = self._get_nearest_object(object_name, push_coordinate)

        push_pose = obj_to_push.pose_com()

        # it is certainly better when pushing up to move under the object with a closed gripper so we can
        #  actually push up. same for the other directions.
        if direction == "up":
            push_pose.x -= obj_to_push.height_m() / 2.0
            # gripper 90° rotated. TODO: I have to test these orientations
            push_pose.yaw = math.pi / 2.0
        elif direction == "down":
            push_pose.x += obj_to_push.height_m() / 2.0
            # gripper 90° rotated. TODO: I have to test these orientations
            push_pose.yaw = math.pi / 2.0
        elif direction == "left":
            push_pose.y += obj_to_push.width_m() / 2.0
            # gripper 0° rotated. TODO: I have to test these orientations
            push_pose.yaw = 0.0
        elif direction == "right":
            push_pose.y -= obj_to_push.width_m() / 2.0
            # gripper 0° rotated. TODO: I have to test these orientations
            push_pose.yaw = 0.0
        else:
            self._logger.error(f"Unknown direction: {direction}")

        if obj_to_push is not None:
            success = self._robot.robot_push_object(push_pose, direction, distance)
        else:
            success = False

        # thread_oral.join()

        return success

    def pick_place_object_across_workspaces(
        self,
        object_name: str,
        pick_workspace_id: str,
        pick_coordinate: List,
        place_workspace_id: str,
        place_coordinate: List,
        location: Union["Location", str, None] = None,
        z_offset: float = 0.001,
    ) -> bool:
        """
        Pick an object from one workspace and place it in another workspace.

        Args:
            object_name: Name of the object to pick
            pick_workspace_id: ID of the workspace to pick from
            pick_coordinate: [x, y] coordinate in pick workspace
            place_workspace_id: ID of the workspace to place in
            place_coordinate: [x, y] coordinate in place workspace
            location: Relative placement location (Location enum or string)
            z_offset: Additional height offset in meters when picking (default: 0.001)

        Returns:
            bool: True if successful, False otherwise

        Example:
            robot.pick_place_object_across_workspaces(
                object_name='cube',
                pick_workspace_id='niryo_ws_left',
                pick_coordinate=[0.2, 0.05],
                place_workspace_id='niryo_ws_right',
                place_coordinate=[0.25, -0.05],
                location=Location.RIGHT_NEXT_TO,
                z_offset=0.02
            )
        """
        self._logger.debug(f"Multi-workspace operation: {object_name}")
        self._logger.debug(f"  Pick from: {pick_workspace_id} at {pick_coordinate}")
        self._logger.debug(f"  Place in: {place_workspace_id} at {place_coordinate}")

        # Step 1: Move to pick workspace observation pose
        self.move2observation_pose(pick_workspace_id)
        self.environment()._current_workspace_id = pick_workspace_id

        # Step 2: Pick the object
        success = self.pick_object_from_workspace(object_name, pick_workspace_id, pick_coordinate, z_offset=z_offset)

        if not success:
            self._logger.error(f"Failed to pick {object_name} from {pick_workspace_id}")
            return False

        # Step 3: Move to place workspace observation pose
        self.move2observation_pose(place_workspace_id)
        self.environment()._current_workspace_id = place_workspace_id

        # Step 4: Place the object
        place_success = self.place_object_in_workspace(place_workspace_id, place_coordinate, location)

        if place_success:
            # Update memory: remove from source, add to target
            self.environment().update_object_in_workspace(
                source_workspace_id=pick_workspace_id,
                target_workspace_id=place_workspace_id,
                object_label=object_name,
                old_coordinate=pick_coordinate,
                new_coordinate=place_coordinate,
            )

            self._logger.info(f"Successfully moved {object_name} from {pick_workspace_id} to {place_workspace_id}")

        return place_success

    def pick_object_from_workspace(
        self, object_name: str, workspace_id: str, pick_coordinate: List, z_offset: float = 0.001
    ) -> bool:
        """
        Pick an object from a specific workspace.

        Args:
            object_name: Name of the object to pick
            workspace_id: ID of the workspace
            pick_coordinate: [x, y] coordinate in workspace
            z_offset: Additional height offset in meters (default: 0.001)

        Returns:
            bool: True if successful
        """
        coords_str = "[" + ", ".join(f"{x:.2f}" for x in pick_coordinate) + "]"
        message = f"Picking {object_name} from workspace {workspace_id} at {coords_str}."
        self._logger.info(message)

        self.environment().oralcom_call_text2speech_async(message, priority=8)

        # Get object from specific workspace memory
        obj_to_pick = self._get_nearest_object_in_workspace(object_name, workspace_id, pick_coordinate)

        if obj_to_pick:
            self._object_last_picked = obj_to_pick
            self._object_source_workspace = workspace_id

            # Apply z_offset to the pick pose
            pick_pose = obj_to_pick.pose_com()
            pick_pose = pick_pose.copy_with_offsets(z_offset=z_offset)

            success = self._robot.robot_pick_object(pick_pose)
        else:
            success = False

        # thread_oral.join()
        return success

    def place_object_in_workspace(
        self, workspace_id: str, place_coordinate: List, location: Union["Location", str, None] = None
    ) -> bool:
        """
        Place a picked object in a specific workspace.

        Args:
            workspace_id: ID of the target workspace
            place_coordinate: [x, y] coordinate in workspace
            location: Relative placement location

        Returns:
            bool: True if successful
        """
        location = Location.convert_str2location(location)

        if self._object_last_picked:
            message = (
                f"Placing {self._object_last_picked.label()} in workspace "
                f"{workspace_id} {location} coordinate "
                f"[{place_coordinate[0]:.2f}, {place_coordinate[1]:.2f}]."
            )
        else:
            message = (
                f"Placing object in workspace {workspace_id} {location} "
                f"coordinate [{place_coordinate[0]:.2f}, {place_coordinate[1]:.2f}]."
            )

        self._logger.info(message)
        self.environment().oralcom_call_text2speech_async(message, priority=8)

        # Get workspace for coordinate transformation
        workspace = self.environment().get_workspace_by_id(workspace_id)
        if workspace is None:
            self._logger.error(f"Workspace {workspace_id} not found")
            # thread_oral.join()
            return False

        # Find reference object in target workspace if location specified
        obj_where_to_place = None
        if location is not None and location is not Location.NONE:
            obj_where_to_place = self._get_nearest_object_in_workspace(None, workspace_id, place_coordinate)

            if obj_where_to_place is None:
                place_pose = PoseObjectPNP(place_coordinate[0], place_coordinate[1], 0.09, 0.0, 1.57, 0.0)
            else:
                place_pose = obj_where_to_place.pose_center()
        else:
            place_pose = PoseObjectPNP(place_coordinate[0], place_coordinate[1], 0.09, 0.0, 1.57, 0.0)

        # Calculate placement offset based on location
        if place_pose and obj_where_to_place:
            x_off = 0.02
            y_off = 0.02

            if self._object_last_picked:
                x_off += self._object_last_picked.height_m() / 2
                y_off += self._object_last_picked.width_m() / 2

            if location == Location.ON_TOP_OF:
                place_pose.z += 0.02
            elif location == Location.INSIDE:
                place_pose.z += 0.01
            elif location == Location.RIGHT_NEXT_TO:
                place_pose.y -= obj_where_to_place.width_m() / 2 + y_off
            elif location == Location.LEFT_NEXT_TO:
                place_pose.y += obj_where_to_place.width_m() / 2 + y_off
            elif location == Location.BELOW:
                place_pose.x -= obj_where_to_place.height_m() / 2 + x_off
            elif location == Location.ABOVE:
                place_pose.x += obj_where_to_place.height_m() / 2 + x_off

        success = self._robot.robot_place_object(place_pose)

        # Clear last picked object
        self._object_last_picked = None
        if hasattr(self, "_object_source_workspace"):
            del self._object_source_workspace

        # thread_oral.join()
        return success

    def _get_nearest_object_in_workspace(
        self, label: Union[str, None], workspace_id: str, target_coords: List
    ) -> Optional["Object"]:
        """
        Find the nearest object in a specific workspace.

        Args:
            label: Object label to search for (None for any object)
            workspace_id: ID of the workspace
            target_coords: Target coordinates [x, y]

        Returns:
            Object or None
        """
        # Get objects from specific workspace memory
        detected_objects = self.environment().get_detected_objects_from_workspace(workspace_id)

        self._logger.debug(f"Objects in workspace {workspace_id}: {detected_objects}")

        if len(target_coords) == 0:
            nearest_object = next((obj for obj in detected_objects if obj.label() == label), None)
            min_distance = 0
        else:
            nearest_object, min_distance = detected_objects.get_nearest_detected_object(target_coords, label)

        if nearest_object:
            self._logger.debug(f"Found {nearest_object.label()} at distance {min_distance:.3f}m")
        else:
            self._logger.warning(f"Object {label} not found in workspace {workspace_id}")
            self._logger.info(f"Available objects: " f"{detected_objects.get_detected_objects_as_comma_separated_string()}")

        return nearest_object

    @staticmethod
    def _parse_command(line: str) -> tuple[str, str, list, dict]:
        """
        Parse a single line of the input into the target object, method, positional arguments, and keyword arguments.

        Args:
            line (str): The command string
            (e.g., 'robot.pick_place_object(object_name="pencil", location="right next to")').

        Returns:
            tuple[str, str, list, dict]: The target object ('robot' or 'agent'), the method name, positional arguments,
            and keyword arguments as a dictionary.
        """
        try:
            # Match the object, method, and arguments
            match = re.match(r"(\w+)\.(\w+)\((.*)\)", line.strip())
            if not match:
                raise ValueError(f"Invalid command format: {line}")

            target_object, method, args_str = match.groups()

            # Use AST to safely parse the arguments
            positional_args = []
            keyword_args = {}

            if args_str:
                # Parse the argument string using AST
                args_list = ast.parse(f"func({args_str})").body[0].value.args
                keywords = ast.parse(f"func({args_str})").body[0].value.keywords

                # Convert AST nodes to Python objects
                positional_args = [ast.literal_eval(arg) for arg in args_list]
                keyword_args = {kw.arg: ast.literal_eval(kw.value) for kw in keywords}

            # Replace location string with corresponding Location enum
            if "location" in keyword_args:
                location_value = keyword_args["location"]
                if isinstance(location_value, str):  # Ensure it's a string
                    # Map the string to the corresponding Location enum
                    keyword_args["location"] = next(
                        (loc for loc in Location if loc.value == location_value),
                        Location.NONE,  # Default to Location.NONE if no match
                    )

            return target_object, method, positional_args, keyword_args
        except Exception as e:
            # Using a module logger since this is a static method
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error parsing command: {e}", exc_info=True)
            return None, None, [], {}

    def get_detected_objects(self):
        """Get latest detected objects from memory."""
        latest_objects = self._environment.get_detected_objects_from_memory()
        return latest_objects

    def _get_nearest_object(self, label: Union[str, None], target_coords: List) -> Optional["Object"]:
        """
        Find the nearest object with the specified label.

        Args:
            label:
            target_coords:

        Returns:
            object:
        """
        detected_objects = self.get_detected_objects()

        self._logger.debug(f"detected_objects: {detected_objects}")

        if len(target_coords) == 0:  # then no target coords are given, true for push method
            nearest_object = next((obj for obj in detected_objects if obj.label() == label), None)
            min_distance = 0
        else:
            nearest_object, min_distance = detected_objects.get_nearest_detected_object(target_coords, label)

        if nearest_object:
            self._logger.debug(f"Nearest object found: {nearest_object} with distance {min_distance}")
        else:
            self._logger.warning(
                f"Object {label} does not exist: " f"{detected_objects.get_detected_objects_as_comma_separated_string()}"
            )

            # add functionality that looks for the most similar object in self.get_detected_objects()
            # and ask user whether this object should be used instead. if answer of user is yes, then set
            # nearest_object to this new object
            # TODO: get_most_similar_object wieder nutzen
            #  nearest_object_name = self.environment().get_most_similar_object(label)
            nearest_object_name = None

            if nearest_object_name is not None:
                self._logger.info(
                    f"I have detected the object {nearest_object_name}. Do you want to handle this object instead?"
                )

                # TODO: auf antwort von user warten und diese prüfen.
                answer = "yes"

                if answer != "yes":
                    return None
                else:
                    nearest_object = next((obj for obj in detected_objects if obj.label() == nearest_object_name), None)

        return nearest_object

    # *** PUBLIC properties ***

    def environment(self) -> "Environment":
        return self._environment

    def robot_in_motion(self) -> bool:
        """
        :return: value of _robot_in_motion:
        False: robot is not in motion
        True: robot is in motion and therefore maybe cannot see the workspace markers
        """
        return self._robot.is_in_motion()

    def robot(self) -> "RobotController":
        """
        Returns:
            RobotController: object that controls the robot.
        """
        return self._robot

    def verbose(self) -> bool:
        """
        Returns: True, if verbose is on, else False
        """
        return self._verbose

    # *** PRIVATE variables ***

    _environment = None

    # RobotController object
    _robot = None

    # True, if robot is in motion and therefore cannot see the workspace markers
    # _robot_in_motion = False

    _object_source_workspace: Optional[str] = None  # Track source workspace

    _verbose = False
    _logger = None
