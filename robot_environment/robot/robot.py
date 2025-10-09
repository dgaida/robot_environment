# robot class around Niryo robot for smart pick and place
# a few TODOs
# Documentation and type definitions are almost final (chatgpt can maybe improve it)

from ..common.logger import log_start_end_cls

from .robot_api import RobotAPI, Location

from .niryo_robot_controller import NiryoRobotController
from redis_robot_comm import RedisMessageBroker
from ..objects.pose_object import PoseObjectPNP
from ..objects.object import Object
from ..objects.objects import Objects

from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from ..environment import Environment
    from .robot_controller import RobotController
    from ..objects.object import Object

import math
import re
import json
import ast
import traceback


class Robot(RobotAPI):
    # *** CONSTRUCTORS ***
    @log_start_end_cls()
    def __init__(self, environment: "Environment", use_simulation: bool = False, robot_id: str = "niryo",
                 verbose: bool = False):
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

        if robot_id == "niryo":
            self._robot = NiryoRobotController(self, use_simulation, verbose)
        else:
            self._robot = None

        self._broker = RedisMessageBroker()

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    def handle_object_detection(self, objects_dict_list):
        """Process incoming object detections from Redis"""
        # Convert dictionaries back to Object instances
        objects = Objects.dict_list_to_objects(objects_dict_list, self.environment().get_workspace(0))

        # Now work with Object instances as before
        for obj in objects:
            print(f"Received object: {obj.label()} at {obj.xy_com()}")

    # GET methods from RobotController

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
        if self.verbose():
            print("robot::get_target_pose_from_rel", workspace_id, u_rel, v_rel, yaw)

        return self._robot.get_target_pose_from_rel(workspace_id, u_rel, v_rel, yaw)

    # *** PUBLIC methods ***

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
    def pick_place_object(self, object_name: str, pick_coordinate: List, place_coordinate: List,
                          location: Union["Location", str, None] = None) -> bool:
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

        Returns:
            bool: Always returns `True` after the pick-and-place operation.
        """
        success = self.pick_object(object_name, pick_coordinate)

        if success:
            return self.place_object(place_coordinate, location)
        else:
            return False

    @log_start_end_cls()
    def pick_object(self, object_name: str, pick_coordinate: List) -> bool:
        """
        Command the pick-and-place robot arm to pick up a specific object using its gripper. The gripper will move to
        the specified 'pick_coordinate' and pick the named object.

        Example call:

        robot.pick_object("pen", [0.01, -0.15])
        --> Picks the pen that is located at world coordinates [0.01, -0.15].

        Args:
            object_name (str): The name of the object to be picked up. Ensure this name matches an object visible in
            the robot's workspace.
            pick_coordinate (List): The world coordinates [x, y] where the object should be picked up. Use these
            coordinates to identify the object's exact position.
        Returns:
            bool: True
        """
        coords_str = "[" + ", ".join(f"{x:.2f}" for x in pick_coordinate) + "]"
        message = f'Going to pick {object_name} at coordinate {coords_str}.'
        print(message)

        thread_oral = self.environment().oralcom_call_text2speech_async(message)

        obj_to_pick = self._get_nearest_object(object_name, pick_coordinate)

        if obj_to_pick:
            self._object_last_picked = obj_to_pick

            success = self._robot.robot_pick_object(obj_to_pick)
        else:
            success = False

        thread_oral.join()

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
            message = (f'Going to place {self._object_last_picked.label()} {location} coordinate ['
                       f'{place_coordinate[0]:.2f}, {place_coordinate[1]:.2f}].')
        else:
            message = f'Going to place it {location} coordinate [{place_coordinate[0]:.2f}, {place_coordinate[1]:.2f}].'
        print(message)

        thread_oral = self.environment().oralcom_call_text2speech_async(message)
        obj_where_to_place = None

        if location is not None and location is not Location.NONE:
            obj_where_to_place = self._get_nearest_object(None, place_coordinate)
            if obj_where_to_place is None:
                place_pose = PoseObjectPNP(place_coordinate[0], place_coordinate[1], 0.09,  # 0.068,
                                           0.0, 1.57, 0.0)
            else:
                place_pose = obj_where_to_place.pose_center()
        else:
            # gegeben eine xyz Koordinate, wie kann ich die benötigte greifer pose berechnen? das ist so korrekt und
            # funkt so in der Simulation
            # TODO: die Zahl 0.068 kommt von Niryo Robot greifer. bei der Höhe des Greifers, schwebt der Greifer gerade
            # über den Workspace
            place_pose = PoseObjectPNP(place_coordinate[0], place_coordinate[1], 0.09,  # 0.068,
                                       0.0, 1.57, 0.0)
            if self.verbose():
                print("place_object:", place_pose)

        x_off = 0.02  # 2 cm # 10
        y_off = 0.02  # 2 cm # 10

        if self._object_last_picked:
            x_off += self._object_last_picked.height_m() / 2
            y_off += self._object_last_picked.width_m() / 2

        # ich muss das in x, y Koordinaten machen und nicht in Pixelkoordinaten
        if place_pose:
            # TODO: use height of object instead
            if location == Location.ON_TOP_OF:
                place_pose.z += 0.02
            elif location == Location.INSIDE:
                place_pose.z += 0.01
            elif location == Location.RIGHT_NEXT_TO:
                place_pose.y -= obj_where_to_place.width_m() / 2 - y_off
            elif location == Location.LEFT_NEXT_TO:
                place_pose.y += obj_where_to_place.width_m() / 2 + y_off
            elif location == Location.BELOW:
                # TODO: nutze hier auch width, da width immer die größere größe ist und nicht eine koordinatenrichtugn hat
                #  ich muss anstatt width und height eine größe haben dim_x und dim_y, die a x und y koordinate gebunden sind
                #  ich habe das in object klasse repariert, width geht immer entlang y-achse jetzt. prüfen hier
                place_pose.x -= obj_where_to_place.height_m() / 2 - x_off
                # place_pose.x -= obj_where_to_place.width_m() / 2 - x_off
            elif location == Location.ABOVE:
                # TODO: nutze hier auch width, da width immer die größere größe ist und nicht eine koordinatenrichtugn hat
                #  ich habe das in object klasse repariert, width geht immer entlang y-achse jetzt. prüfen hier
                print(obj_where_to_place.height_m(), self._object_last_picked.width_m(), x_off)
                place_pose.x += obj_where_to_place.height_m() / 2 + x_off
                # place_pose.x += obj_where_to_place.width_m() / 2 + x_off
                print(place_pose)
            elif location is Location.NONE or location is None:
                pass  # I do not have to do anything as the given location is where to place the object
            else:
                print('unknown location!!!!!!!!!!!', location, type(location))

            success = self._robot.robot_place_object(place_pose)

            # update position of placed object to the new position
            if self._object_last_picked:
                self._object_last_picked.set_position(place_pose)
            # TODO: have to get access to objects in environment because _object_last_picked is deleted in 3 lines
            # TODO: das Problem an meiner Implementierung ist, dass sobald das LLM aufgerufen wird, wird es
            # mit einer statischen Liste von Objekten mit deren Positionen aufgerufen. wenn während der Ausführung des
            # LLms oder des roboters danach etwas an der Position der Objekte ändert, bekommt der roboter das nicht
            # mit, da die objekte quasi nur im visualcortex leben und keine realen objekte sind. eigentlich
            # bräuchte man noch ein objekt tracker, der objekte eindeutig trackt. dann kann roboter auch neue koordinaten
            # des objekts erhalten, falls es sich wegbewegt hat.
        else:
            success = False

        self._object_last_picked = None

        thread_oral.join()

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
        message = f'Calling push with {object_name} and {direction}'
        print(message)

        thread_oral = self.environment().oralcom_call_text2speech_async(message)

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
            print("Unknown direction!", direction)

        if obj_to_push is not None:
            success = self._robot.robot_push_object(push_pose, direction, distance)
        else:
            success = False

        thread_oral.join()

        return success

    @log_start_end_cls()
    def execute_python_code_safe(self, python_code: str) -> tuple[None, bool]:
        """
        Safely execute Python code using structured commands.

        Args:
            python_code (str): The commands to execute.

        Returns:
            tuple[None, bool]: (None, success)
        """
        self._robot_in_motion = True
        message = f"Executing python code:\n{python_code}"
        self.environment().append_assistant_message2chat_history(message)

        if self.verbose():
            print(message)

        success = True

        # Process each line as a command
        for line in python_code.splitlines():
            target_object, method, pos_args, kw_args = self._parse_command(line)
            if target_object and method:
                success = success and self._execute_command(target_object, method, pos_args, kw_args)
            else:
                self.environment().append_assistant_message2chat_history(f"Invalid command: {line}")
                success = False
                continue

        self._robot_in_motion = False

        return None, success

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
                        Location.NONE  # Default to Location.NONE if no match
                    )

            return target_object, method, positional_args, keyword_args
        except Exception as e:
            print(f"Error parsing command: {e}")
            return None, None, [], {}

    @log_start_end_cls()
    def _execute_command(self, target_object: str, method: str, positional_args: list, keyword_args: dict) -> bool:
        """
        Execute a parsed command.

        Args:
            target_object (str): The object to invoke the method on ('robot' or 'agent').
            method (str): The method name to call.
            positional_args (list): Positional arguments for the method.
            keyword_args (dict): Keyword arguments for the method.

        Returns:
            bool: True if the command was executed successfully, False otherwise.
        """
        if self.verbose():
            print(target_object, method)
            print(positional_args, keyword_args)

        try:
            # Route commands to the appropriate target object
            if target_object == "robot":
                # with self.environment().lock():
                if method == "pick_place_object":
                    return self.pick_place_object(*positional_args, **keyword_args)
                elif method == "pick_object":
                    return self.pick_object(*positional_args, **keyword_args)
                elif method == "place_object":
                    return self.place_object(*positional_args, **keyword_args)
                elif method == "push_object":
                    return self.push_object(*positional_args, **keyword_args)
                else:
                    raise ValueError(f"Unknown method for robot: {method}")

            elif target_object == "agent":
                if method == "get_object_labels_and_print2chat":
                    self.environment().agent().get_object_labels_and_print2chat()
                    return True
                elif method == "add_object_name2object_labels":
                    self.environment().agent().add_object_name2object_labels(*positional_args, **keyword_args)
                    return True
                else:
                    raise ValueError(f"Unknown method for agent: {method}")

            else:
                raise ValueError(f"Unknown target object: {target_object}")

        except Exception as e:
            self.environment().append_assistant_message2chat_history(f"Error executing command: {e}")
            return False

    @log_start_end_cls()
    def execute_python_code_not_safe(self, python_code: str) -> tuple[dict, bool]:
        """
        Execute the provided Python code within the context of the Robot instance.

        Args:
            python_code (str): The Python code to execute.

        Returns:
            tuple:
            - dict: A dictionary of local variables after code execution.
            - bool: success
        """
        self._robot_in_motion = True

        message = f"Executing python code:\n{python_code}"

        self.environment().append_assistant_message2chat_history(message)

        # Execute the Python code in the context of this instance
        local_vars = {"robot": self}  # Pass 'self' as 'robot' to make methods accessible in the code

        if self.verbose():
            print(message, local_vars)

        try:
            exec(python_code, globals(), local_vars)
            success = True
        except (Exception, RuntimeError, UnicodeDecodeError) as e:
            self.environment().append_assistant_message2chat_history(
                f"Error executing generated Python code: {e}")
            # Log or re-raise with additional context
            # raise RuntimeError(f"Error executing generated Python code: {e}")
            success = False

        self._robot_in_motion = False

        return local_vars, success  # Return all local variables for further analysis if needed

    @log_start_end_cls()
    def execute_python_code(self, python_code: str) -> tuple[None, bool]:
        """
        Safely execute Python code using an isolated function.

        Args:
            python_code (str): The Python code to execute.

        Returns:
            tuple[None, bool]: (None, success)
        """
        self._robot_in_motion = True

        # Indent the code properly for function encapsulation
        indented_code = "\n".join(f"    {line}" for line in python_code.splitlines())

        # Wrap the indented code in a function definition
        wrapped_code = f"def generated_function():\n{indented_code}"

        print(wrapped_code)

        message = f"Executing python code:\n{wrapped_code}"
        self.environment().append_assistant_message2chat_history(message)

        if self.verbose():
            print(message)

        try:
            # with self.environment().lock():
            # Create an isolated function for the code
            exec_globals = {"robot": self, "agent": self.environment().agent()}
            exec_locals = {}
            exec(wrapped_code, exec_globals, exec_locals)

            # Call the generated function
            exec_locals["generated_function"]()
            success = True
        except Exception as e:
            self.environment().append_assistant_message2chat_history(
                f"Error executing generated Python code: {e}")
            # Get the full stack trace as a string
            self.environment().append_assistant_message2chat_history(traceback.format_exc())
            success = False
        finally:
            self._robot_in_motion = False

        return None, success

    # this is old, not used anymore
    # Function to parse and execute
    def execute_tasks_from_string(self, llm_output):
        """

        Args:
            llm_output:
        """
        # Regex to extract function calls and JSON payloads
        function_pattern = re.compile(r"<function=(\w+)>\{\{(.+?)\}\}</function>")

        # Match all function calls in the string
        matches = function_pattern.findall(llm_output)

        for func_name, json_payload in matches:
            # Parse JSON arguments
            args = json.loads(f"{{{json_payload}}}")

            message = f"Executing task: {func_name} for {args['object_name']}"
            new_message = {"role": "assistant", "content": message}

            thread_oral = self._oralcom.call_text2speech_async(message)

            self._chat_history.append(new_message)

            self._robot_in_motion = True

            # Dynamically get the method by name
            method = getattr(self, func_name, None)
            if method:
                # Call the method with the extracted arguments
                method(**args)
            else:
                print(f"Method '{func_name}' not found in Robot.")

            # wait for thread to finish
            thread_oral.join()

        self._robot_in_motion = False

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    def get_detected_objects(self):
        # TODO: 12 means 12 seconds, change to 2 in real environment
        # objects_dict_list = self._broker.get_latest_objects(10)

        latest_objects = self._environment.get_detected_objects_from_memory()

        # print([obj['label'] for obj in objects_dict_list])

        # Convert dictionaries back to Object instances
        # latest_objects = Objects.dict_list_to_objects(objects_dict_list, self.environment().get_workspace(0))

        return latest_objects

    def _get_nearest_object(self, label: str | None, target_coords: List) -> Optional["Object"]:
        """
        Find the nearest object with the specified label.

        Args:
            label:
            target_coords:

        Returns:
            object:
        """
        # TODO: hier muss ich eine FUnktion aufrufen, die dtected objects von einem buffer holt, denn wenn roboter
        # einmal in bewegung, erkennt er ja keine objekte mehr. bzw. wenn robote rin motion, dann sollte keine neuen objekte
        # detektiert werden.
        detected_objects = self.get_detected_objects()

        print("detected_objects", detected_objects)

        if len(target_coords) == 0:  # then no target coords are given, true for push method
            nearest_object = next((obj for obj in detected_objects if obj.label() == label),
                                  None)
            min_distance = 0
        else:
            nearest_object, min_distance = (detected_objects.get_nearest_detected_object(target_coords, label))

        if nearest_object:
            print(f"Nearest object found: {nearest_object} with {min_distance}")
        else:
            # TODO: append_assistant_message2chat_history not used anymore
            print(f"Object {label} does not exist: "
                  f"{detected_objects.get_detected_objects_as_comma_separated_string()}")
            # self.environment().append_assistant_message2chat_history(
            #    f"Object {label} does not exist: "
            #    f"{detected_objects.get_detected_objects_as_comma_separated_string()}")

            # add functionality that looks for the most similar object in self.get_detected_objects()
            # and ask user whether this object should be used instead. if answer of user is yes, then set
            # nearest_object to this new object
            # TODO: get_most_similar_object wieder nutzen
            #  nearest_object_name = self.environment().get_most_similar_object(label)
            nearest_object_name = None

            if nearest_object_name is not None:
                print(f"I have detected the object {nearest_object_name}. Do you want to handle this object instead?")
                # TODO: append_assistant_message2chat_history not used anymore
                # self.environment().append_assistant_message2chat_history(
                #     f"I have detected the object {nearest_object_name}. Do you want to handle this object instead?")

                # TODO: auf antwort von user warten und diese prüfen.
                answer = "yes"

                if answer != "yes":
                    return None
                else:
                    nearest_object = next(
                        (obj for obj in detected_objects if obj.label() == nearest_object_name), None)

        # print("nearest_object", nearest_object)

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
        return self._robot_in_motion

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
    _robot_in_motion = False

    _verbose = False
