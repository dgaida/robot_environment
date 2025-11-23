# environment class in which a robot for smart pick and place exists
# one TODO. but more or less finished.
# Documentation and type definitions are almost final (chatgpt can maybe improve it)

import threading

from .common.logger import log_start_end_cls

import numpy as np
import time
import cv2

from robot_workspaces import Workspaces
from robot_workspaces import NiryoWorkspaces
from .camera.framegrabber import FrameGrabber
from .camera.niryo_framegrabber import NiryoFrameGrabber
from .camera.widowx_framegrabber import WidowXFrameGrabber
from .robot.robot import Robot
from .robot.niryo_robot_controller import NiryoRobotController
from .robot.widowx_robot_controller import WidowXRobotController

from text2speech import Text2Speech
from robot_workspaces import Object
from robot_workspaces import Objects
from redis_robot_comm import RedisMessageBroker

from vision_detect_segment import VisualCortex
from vision_detect_segment import get_default_config

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from robot_workspaces import Workspace
    from robot_workspacess import Workspaces
    from .camera.framegrabber import FrameGrabber
    from .robot.robot import Robot
    from .robot.robot_controller import RobotController
    from robot_workspaces import PoseObjectPNP


class Environment:
    # *** CONSTRUCTORS ***
    def __init__(
        self, el_api_key: str, use_simulation: bool, robot_id: str, verbose: bool = False, start_camera_thread: bool = True
    ):
        """
        Creates environment object. Creates these objects:
        - FrameGrabber
        - Robot
        - Agent

        Args:
            el_api_key (str): the ElevenLabs API Key as string
            use_simulation: if True, then simulate the robot, else the real robot is used
            robot_id: string defining the robot. can be "niryo" or "widowx"
            LLM create Python Code.
            verbose: enable verbose output
            start_camera_thread: if True, start camera update thread (default: True)
                                Set to False for MCP server!
        """
        self._use_simulation = use_simulation
        self._verbose = verbose

        # self._lock = threading.Lock()

        # important that Robot comes before framegrabber and before workspace
        self._robot = Robot(self, use_simulation, robot_id, verbose)

        if isinstance(self.get_robot_controller(), NiryoRobotController):
            self._framegrabber = NiryoFrameGrabber(self, verbose=verbose)
            self._workspaces = NiryoWorkspaces(self, verbose)

            if verbose:
                print(self._workspaces.get_home_workspace())
        elif isinstance(self.get_robot_controller(), WidowXRobotController):
            self._framegrabber = WidowXFrameGrabber(self, verbose=verbose)
            # TODO: WidowXWorkspaces erstellen
            self._workspaces = Workspaces(self, verbose)
        else:
            if verbose:
                print("error:", self.get_robot_controller())

        self._oralcom = Text2Speech(el_api_key, verbose=verbose)

        # Initialize speech recognition
        # self._speech2text = Speech2Text(
        #     el_api_key=el_api_key,
        #     use_whisper_mic=True,
        #     verbose=verbose
        # )

        self._stop_event = threading.Event()

        # self._streamer = RedisImageStreamer()

        self._obj_position_memory = Objects()

        config = get_default_config("owlv2")

        # owlv2
        self._visual_cortex = VisualCortex(objdetect_model_id="yoloe-11s", device="auto", verbose=verbose, config=config)

        if start_camera_thread:
            if verbose:
                print("Starting camera update thread...")
            # Start background camera updates
            self.start_camera_updates(visualize=True)  # set visualize=False if you don't want OpenCV windows
        else:
            if verbose:
                print("Camera thread disabled (manual control)")

    def __del__(self):
        """ """
        if hasattr(self, "_stop_event"):
            if self.verbose():
                print("Shutting down environment in del...")
            self._stop_event.set()

    def cleanup(self):
        """
        Explicit cleanup method - call this when you're done with the object.
        This is more reliable than relying on __del__.
        """
        if hasattr(self, "_stop_event"):
            if self.verbose():
                print("Shutting down environment...")
            self._stop_event.set()

    # PUBLIC methods

    def start_camera_updates(self, visualize=False):
        def loop():
            for img in self.update_camera_and_objects(visualize=visualize):
                # In CLI, we might not use img, but you could save or log info here
                pass  # or print("Camera updated")

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        return t

    def _check_new_detections(self, detected_objects: "Objects") -> None:
        """
        Check for newly detected objects and update the memory with their positions.

        Args:
            detected_objects (Objects): List of objects detected in the current frame.
        """
        for obj in detected_objects:
            x_center, y_center = obj.xy_com()
            # obj_position = (obj.label(), x_center, y_center)
            if not any(
                memory.label() == obj.label()
                and abs(memory.x_com() - x_center) <= 0.05
                and abs(memory.y_com() - y_center) <= 0.05
                for memory in self._obj_position_memory
            ):
                self._obj_position_memory.append(obj)
                # message = obj.as_string_for_chat_window()

    def get_detected_objects_from_memory(self) -> "Objects":
        return self._obj_position_memory

    def update_camera_and_objects(self, visualize: bool = False):
        """
        Continuously updates the camera and detected objects.

        Args:
            visualize (bool): If True, displays the updated camera feed in a window.
        """

        self.robot_move2observation_pose(self._workspaces.get_workspace_home_id())

        while not self._stop_event.is_set():
            self.get_current_frame()  # img =

            time.sleep(0.5)

            self._visual_cortex.detect_objects_from_redis()

            time.sleep(0.5)

            detected_objects = self.get_detected_objects()

            self._check_new_detections(detected_objects)

            annotated_image = self._visual_cortex.get_annotated_image()

            # Display the image if visualize is True
            if visualize:
                # TODO: nicht der richtige Ort hier, nur temporär
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                cv2.imshow("Camera View", annotated_image)
                # Break the loop if ESC key is pressed
                if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the ESC key
                    if self.verbose():
                        print("Exiting camera update loop.")
                    break

            yield annotated_image  # img

            if self.get_robot_in_motion():
                # TODO: change back to 0.5 and 0.25
                time.sleep(0.5)  # Wait before the next camera update 0.25
            else:
                time.sleep(0.25)  # Wait before the next camera update 0.25

        # Close the OpenCV window when exiting
        if visualize:
            cv2.destroyAllWindows()

    # *** PUBLIC SET methods ***

    @log_start_end_cls()
    def add_object_name2object_labels(self, object_name: str) -> str:
        """
        Call this method if the user wants to add another object to the list of recognizable objects. Adds the
        object called object_name to the list of recognizable objects.

        Args:
            object_name (str): The name of the object that should also be recognizable by the robot.

        Returns:
            str: Message saying that the given object_name was added to the list of recognizable objects.
        """
        self._visual_cortex.add_object_name2object_labels(object_name)
        mymessage = f"Added {object_name} to the list of recognizable objects."
        thread_oral = self._oralcom.call_text2speech_async(mymessage)
        thread_oral.join()
        return mymessage

    def stop_camera_updates(self):
        self._stop_event.set()

    # def speech2text_record_and_transcribe(self) -> str:
    #     """
    #     Record from microphone and transcribe using Whisper ASR model until silence is detected.
    #
    #     Returns:
    #         transcribed text
    #     """
    #     return self._speech2text.record_and_transcribe()

    def oralcom_call_text2speech_async(self, text: str) -> threading.Thread:
        """
        Asynchronously calls the text2speech ElevenLabs API with the given text

        Args:
            text: a message that should be passed to text-2-speech API of ElevenLabs

        Returns:
            the thread object is returned. Once the text is spoken, the thread is being closed.
        """
        return self._oralcom.call_text2speech_async(text)

    def create_dummy_environment(self):
        self._broker = RedisMessageBroker()

        self.robot_move2observation_pose("gazebo_1")

        self._img_work = self.get_current_frame()

        cv2.imshow("Camera View", self._img_work)
        # cv2.waitKey(0)
        # Break the loop if ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the ESC key
            print("Exiting camera update loop.")

        time.sleep(8)

        detected_objects = [
            Object(
                label="pencil",
                u_min=50 - 34 // 2,
                v_min=80 - 34 // 2,
                u_max=50 + 34 // 2,
                v_max=80 + 34 // 2,
                mask_8u=None,
                workspace=self.get_workspace(0),
            ),
            Object(
                label="yellow circle",
                u_min=150 - 34 // 2,
                v_min=90 - 34 // 2,
                u_max=150 + 34 // 2,
                v_max=90 + 34 // 2,
                mask_8u=None,
                workspace=self.get_workspace(0),
            ),
            Object(
                label="blue square",
                u_min=100 - 45 // 2,
                v_min=200 - 45 // 2,
                u_max=100 + 45 // 2,
                v_max=200 + 45 // 2,
                mask_8u=None,
                workspace=self.get_workspace(0),
            ),
        ]

        detected_objects = Objects(detected_objects)

        # Convert Object instances to dictionaries
        objects_dict_list = Objects.objects_to_dict_list(detected_objects)

        # Publish to Redis (now JSON serializable)
        self._broker.publish_objects(objects_dict_list)

    # *** PUBLIC GET methods ***

    def get_largest_free_space_with_center(self) -> tuple[float, float, float]:
        """
        Determines the largest free space in the workspace in square metres and its center coordinate in metres.
        This method can be used to determine at which location an object can be placed safely.

        Example call:
        To pick a 'chocolate bar' and place it at the center of the largest free space of the workspace, call:

        largest_free_area_m2, center_x, center_y = agent.get_largest_free_space_with_center()

        robot.pick_place_object(
            object_name='chocolate bar',
            pick_coordinate=[-0.1, 0.01],
            place_coordinate=[center_x, center_y],
            location=Location.RIGHT_NEXT_TO
        )

        Returns:
            tuple: (largest_free_area_m2, center_x, center_y) where:
                - largest_free_area_m2 (float): Largest free area in square meters.
                - center_x (float): X-coordinate of the center of the largest free area in meters.
                - center_y (float): Y-coordinate of the center of the largest free area in meters.
        """
        # grid_resolution (int): Resolution of the workspace grid (e.g., 100x100 cells).
        grid_resolution = 100

        detected_objects = self.get_detected_objects()
        # TODO: using workspace 0 here
        workspace_top_left = self.get_workspace(0).xy_ul_wc()
        workspace_bottom_right = self.get_workspace(0).xy_lr_wc()

        # Unpack workspace corners
        x_max, y_max = workspace_top_left.x, workspace_top_left.y
        x_min, y_min = workspace_bottom_right.x, workspace_bottom_right.y

        if self.verbose():
            print(x_min, y_min, x_max, y_max)

            print("\n".join(obj.as_string_for_llm_lbl() for obj in detected_objects))

        # Calculate workspace dimensions in meters
        workspace_width = abs(y_max - y_min)
        workspace_height = abs(x_max - x_min)

        # Create a grid to represent the workspace
        grid = np.zeros((grid_resolution, grid_resolution), dtype=int)

        # Map world coordinates to grid indices
        def to_grid_coords(x, y):
            v = int((x_max - x) / workspace_height * grid_resolution)
            u = int((y_max - y) / workspace_width * grid_resolution)
            return u, v

        # Map grid indices back to world coordinates
        def to_world_coords(u, v):
            x = x_max - (v + 0.5) * (workspace_height / grid_resolution)
            y = y_max - (u + 0.5) * (workspace_width / grid_resolution)
            return x, y

        # Mark the grid cells occupied by objects
        for obj in detected_objects:
            x_start = obj.x_com() - obj.height_m() / 2
            x_end = obj.x_com() + obj.height_m() / 2
            y_start = obj.y_com() - obj.width_m() / 2
            y_end = obj.y_com() + obj.width_m() / 2

            # Convert object bounds to grid indices
            u_end, v_end = to_grid_coords(x_start, y_start)
            u_start, v_start = to_grid_coords(x_end, y_end)

            if self.verbose():
                print(x_start, y_start, x_end, y_end)
                print(u_start, v_start, u_end, v_end)

            # Mark grid cells as occupied
            grid[v_start : v_end + 1, u_start : u_end + 1] = 1

        # Find the largest rectangle of zeros in the grid
        def max_rectangle_area(matrix):
            max_area = 0
            top_left = (0, 0)
            bottom_right = (0, 0)
            dp = [0] * len(matrix[0])  # DP array for heights

            for v, row in enumerate(matrix):  # Iterate over rows (v-axis)
                for u in range(len(row)):  # Iterate over columns (u-axis)
                    dp[u] = dp[u] + 1 if row[u] == 0 else 0  # Update heights

                # Compute the maximum area with the updated histogram
                stack = []
                for k in range(len(dp) + 1):
                    while stack and (k == len(dp) or dp[k] < dp[stack[-1]]):
                        h = dp[stack.pop()]
                        w = k if not stack else k - stack[-1] - 1
                        area = h * w
                        if area > max_area:
                            max_area = area
                            top_left = (v - h + 1, stack[-1] + 1 if stack else 0)
                            bottom_right = (v, k - 1)
                    stack.append(k)

            return max_area, top_left, bottom_right

        largest_area_cells, (v_start, u_start), (v_end, u_end) = max_rectangle_area(grid)
        largest_area_m2 = (largest_area_cells / (grid_resolution**2)) * (workspace_width * workspace_height)

        # Calculate the center of the largest rectangle in grid coordinates
        v_center = (v_start + v_end) // 2
        u_center = (u_start + u_end) // 2

        # Map the center to world coordinates
        center_x, center_y = to_world_coords(u_center, v_center)

        if self.verbose():
            grid[v_center : v_center + 1, u_center : u_center + 1] = 2

            # Normalize grid to 0–255 for visualization
            grid_visual = (grid * 255 // 2).astype(np.uint8)

            cv2.imshow("grid", grid_visual)
            cv2.waitKey(0)

        print(f"Largest free area: {largest_area_m2:.4f} square meters")
        print(f"Center of the largest free area: ({center_x:.4f}, {center_y:.4f}) meters")

        return largest_area_m2, center_x, center_y

    def get_workspace_coordinate_from_point(self, workspace_id: str, point: str) -> Optional[List[float]]:
        """
        Get the world coordinate of a special point of the given workspace.

        Args:
            workspace_id (str): ID of workspace.
            point (str): description of point. Possible values are:
            - 'upper left corner': Returns the world coordinate of the upper left corner of the workspace.
            - 'upper right corner': Returns the world coordinate of the upper right corner of the workspace.
            - 'lower left corner': Returns the world coordinate of the lower left corner of the workspace.
            - 'lower right corner': Returns the world coordinate of the lower right corner of the workspace.
            - 'center point': Returns the world coordinate of the center of the workspace.

        Returns:
            List[float]: (x,y) world coordinate of the point on the workspace that was specified by the argument point.
        """
        if point == "upper left corner":
            return self.get_workspace_by_id(workspace_id).xy_ul_wc().xy_coordinate()
        elif point == "upper right corner":
            return self.get_workspace_by_id(workspace_id).xy_ur_wc().xy_coordinate()
        elif point == "lower left corner":
            return self.get_workspace_by_id(workspace_id).xy_ll_wc().xy_coordinate()
        elif point == "lower right corner":
            return self.get_workspace_by_id(workspace_id).xy_lr_wc().xy_coordinate()
        elif point == "center point":
            return self.get_workspace_by_id(workspace_id).xy_center_wc().xy_coordinate()
        else:
            print("Error: get_workspace_coordinate_from_point:", point)
            return None

    # GET methods from Workspaces

    def get_workspace(self, index: int = 0) -> "Workspace":
        """
        Return the workspace at the given position index in the list of workspaces.

        Args:
            index: 0-based index in the list of workspaces.

        Returns:

        """
        return self._workspaces.get_workspace(index)

    def get_workspace_by_id(self, workspace_id: str) -> "Workspace":
        """
        Return the Workspace object with the given id, if existent, else None is returned.

        Args:
            id: workspace ID

        Returns:
            Workspace or None, if no workspace with the given id exists.
        """
        return self._workspaces.get_workspace_by_id(workspace_id)

    def get_workspace_home_id(self) -> str:
        """
        Returns the ID of the workspace at index 0.

        Returns:
            the ID of the workspace at index 0.
        """
        return self._workspaces.get_workspace_home_id()

    def get_workspace_id(self, index: int) -> str:
        """
        Return the id of the workspace at the given position index in the list of workspaces.

        Args:
            index: 0-based index in the list of workspaces.

        Returns:
            str: id of the workspace at the given position index in the list of workspaces.
        """
        return self._workspaces.get_workspace_id(index)

    @log_start_end_cls()
    def get_visible_workspace(self, camera_pose: "PoseObjectPNP") -> "Workspace":
        return self._workspaces.get_visible_workspace(camera_pose)

    def is_any_workspace_visible(self) -> bool:
        pose = self.get_robot_pose()
        if self.get_visible_workspace(pose) is None:
            return False
        else:
            return True

    def get_observation_pose(self, workspace_id: str) -> "PoseObjectPNP":
        """
        Return the observation pose of the given workspace id

        Args:
            workspace_id: id of the workspace

        Returns:
            PoseObjectPNP: observation pose of the gripper where it can observe the workspace given by workspace_id
        """
        return self._workspaces.get_observation_pose(workspace_id)

    # GET methods from FrameGrabber

    def get_current_frame(self) -> np.ndarray:
        """
        Captures an image of the robot's workspace, ensuring proper undistortion in RGB.

        Returns:
            numpy.ndarray: Raw image captured from the robot's camera.
        """
        return self._framegrabber.get_current_frame()

    def get_current_frame_width_height(self) -> tuple[int, int]:
        """
        Returns width and height of current frame in pixels.

        Returns:
            width and height of current frame in pixels.
        """
        return self._framegrabber.get_current_frame_width_height()

    # GET methods from Robot

    def get_robot_controller(self) -> "RobotController":
        """

        Returns:
            RobotController: object that controls the robot.
        """
        return self._robot.robot()

    @log_start_end_cls()
    def get_robot_in_motion(self) -> bool:
        """
        :return: value of _robot_in_motion:
        False: robot is not in motion
        True: robot is in motion and therefore maybe cannot see the workspace markers
        """
        return self._robot.robot_in_motion()

    def get_robot_pose(self) -> "PoseObjectPNP":
        """
        Get current pose of gripper of robot.

        Returns:
            current pose of gripper of robot.
        """
        return self._robot.get_pose()

    @log_start_end_cls()
    def get_robot_target_pose_from_rel(self, workspace_id: str, u_rel: float, v_rel: float, yaw: float) -> "PoseObjectPNP":
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
        return self._robot.get_target_pose_from_rel(workspace_id, u_rel, v_rel, yaw)

    # GET methods from VisualCortex

    def get_object_labels_as_string(self) -> str:
        """
        Returns all object labels that the object detection model is able to detect as a comma separated string.
        Call this method if the user wants to know which objects the robot can pick or is able to detect.

        Returns:
            str: "chocolate bar, blue box, cigarette, ..."
        """
        object_labels = self.get_object_labels()
        mymessage = f"I can recognize these objects: {', '.join(object_labels[0])}"

        return mymessage

    def get_detected_objects(self) -> "Objects":
        detected_obj_list_dict = self._visual_cortex.get_detected_objects()

        return Objects.dict_list_to_objects(detected_obj_list_dict, self.get_workspace(0))

    def get_object_labels(self) -> List[List[str]]:
        """

        Returns:
            list of a list of all detectable objects as strings
        """
        return self._visual_cortex.get_object_labels()

    # *** PUBLIC methods ***

    # methods from Robot

    def robot_move2observation_pose(self, workspace_id: str) -> None:
        """
        The robot is going to move to a pose where it can observe (the gripper hovers over) the workspace
        given by workspace_id.

        Args:
            workspace_id: id of the workspace
        """
        self._robot.move2observation_pose(workspace_id)

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    def workspaces(self) -> "Workspaces":
        return self._workspaces

    def framegrabber(self) -> "FrameGrabber":
        return self._framegrabber

    def robot(self) -> "Robot":
        return self._robot

    def use_simulation(self) -> bool:
        return self._use_simulation

    def verbose(self) -> bool:
        """

        Returns:
            True, if verbose is on, else False
        """
        return self._verbose

    # *** PRIVATE variables ***

    # Workspaces object
    _workspaces = None

    # FrameGraber object
    _framegrabber = None

    # Robot object
    _robot = None

    _use_simulation = False

    _verbose = False
