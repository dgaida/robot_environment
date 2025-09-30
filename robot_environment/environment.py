# environment class in which a robot for smart pick and place exists
# one TODO. but more or less finished.
# Documentation and type definitions are almost final (chatgpt can maybe improve it)

import threading

from .common.logger import log_start_end_cls

import numpy as np
import time
import cv2

from .workspaces.workspaces import Workspaces
from .workspaces.niryo_workspaces import NiryoWorkspaces
from .camera.framegrabber import FrameGrabber
from .camera.niryo_framegrabber import NiryoFrameGrabber
from .camera.widowx_framegrabber import WidowXFrameGrabber
from .robot.robot import Robot
from .robot.niryo_robot_controller import NiryoRobotController
from .robot.widowx_robot_controller import WidowXRobotController

from .text2speech.text2speech import Text2Speech
from .objects.object import Object
from .objects.objects import Objects
from redis_robot_comm import RedisMessageBroker

from vision_detect_segment.visualcortex import VisualCortex
from vision_detect_segment.config import get_default_config

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .workspaces.workspace import Workspace
    from .workspaces.workspaces import Workspaces
    from .camera.framegrabber import FrameGrabber
    from .robot.robot import Robot
    from .robot.robot_controller import RobotController
    from .objects.pose_object import PoseObjectPNP


class Environment:
    # *** CONSTRUCTORS ***
    def __init__(self, el_api_key: str, use_simulation: bool, robot_id: str, verbose: bool = False):
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
            verbose:
        """
        self._use_simulation = use_simulation
        self._verbose = verbose

        # self._lock = threading.Lock()

        # important that Robot comes before framegrabber and before workspace
        self._robot = Robot(self, use_simulation, robot_id, verbose)

        if isinstance(self.get_robot_controller(), NiryoRobotController):
            self._framegrabber = NiryoFrameGrabber(self, verbose=verbose)
            self._workspaces = NiryoWorkspaces(self, verbose)

            print(self._workspaces.get_home_workspace())
        elif isinstance(self.get_robot_controller(), WidowXRobotController):
            self._framegrabber = WidowXFrameGrabber(self, verbose=verbose)
            # TODO: WidowXWorkspaces erstellen
            self._workspaces = Workspaces(self, verbose)
        else:
            print("error:", self.get_robot_controller())

        self._oralcom = Text2Speech(el_api_key, verbose=verbose)

        self._stop_event = threading.Event()

        # self._streamer = RedisImageStreamer()

        config = get_default_config("owlv2")

        self._visual_cortex = VisualCortex(
            objdetect_model_id="owlv2",
            device="auto",
            verbose=True,
            config=config
        )

    def __del__(self):
        """

        """
        if hasattr(self, '_stop_event'):
            print("Shutting down environment in del...")
            self._stop_event.set()

    def cleanup(self):
        """
        Explicit cleanup method - call this when you're done with the object.
        This is more reliable than relying on __del__.
        """
        if hasattr(self, '_stop_event'):
            print("Shutting down environment...")
            self._stop_event.set()

    # PUBLIC methods

    def update_camera_and_objects(self, visualize: bool = False):
        """
        Continuously updates the camera and detected objects.

        Args:
            visualize (bool): If True, displays the updated camera feed in a window.
        """

        self.robot_move2observation_pose(self._workspaces.get_workspace_home_id())

        while not self._stop_event.is_set():
            img = self.get_current_frame()

            time.sleep(0.5)

            success = self._visual_cortex.detect_objects_from_redis()
            annotated_image = self._visual_cortex.get_annotated_image()

            # Display the image if visualize is True
            if visualize:
                cv2.imshow("Camera View", annotated_image)
                # Break the loop if ESC key is pressed
                if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the ESC key
                    print("Exiting camera update loop.")
                    break

            yield img

            if self.get_robot_in_motion():
                # TODO: change back to 0.5 and 0.25
                time.sleep(0.5)  # Wait before the next camera update 0.25
            else:
                time.sleep(0.25)  # Wait before the next camera update 0.25

        # Close the OpenCV window when exiting
        if visualize:
            cv2.destroyAllWindows()

    # *** PUBLIC SET methods ***

    def stop_camera_updates(self):
        self._stop_event.set()

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

    def get_detected_objects(self):
        return self._visual_cortex.get_detected_objects()

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
    def get_robot_target_pose_from_rel(self, workspace_id: str, u_rel: float, v_rel: float,
                                       yaw: float) -> "PoseObjectPNP":
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