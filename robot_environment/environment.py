# environment class in which a robot for smart pick and place exists
# Refactored to use ObjectMemoryManager for cleaner memory management

import threading
from .common.logger import log_start_end_cls
import numpy as np
import time
import cv2

from robot_workspace import Workspaces
from robot_workspace import NiryoWorkspaces
from robot_workspace import WidowXWorkspaces
from .camera.framegrabber import FrameGrabber
from .camera.niryo_framegrabber import NiryoFrameGrabber
from .camera.widowx_framegrabber import WidowXFrameGrabber
from .robot.robot import Robot
from .robot.niryo_robot_controller import NiryoRobotController
from .robot.widowx_robot_controller import WidowXRobotController

from text2speech import Text2Speech

from robot_workspace import Objects

from redis_robot_comm import RedisMessageBroker
from redis_robot_comm import RedisLabelManager

from .object_memory_manager import ObjectMemoryManager

from .common.logger_config import get_package_logger
import logging

from typing import TYPE_CHECKING, List, Optional, Dict

if TYPE_CHECKING:
    from robot_workspace import Workspace
    from robot_workspace import Workspaces
    from .camera.framegrabber import FrameGrabber
    from .robot.robot import Robot
    from .robot.robot_controller import RobotController
    from robot_workspace import PoseObjectPNP


class Environment:
    """
    Environment class for robotic pick-and-place operations.

    Coordinates between:
    - Robot control
    - Vision system
    - Workspace management
    - Object memory tracking
    """

    # *** CONSTRUCTORS ***
    def __init__(
        self, el_api_key: str, use_simulation: bool, robot_id: str, verbose: bool = False, start_camera_thread: bool = True
    ):
        """
        Creates environment object.

        Args:
            el_api_key: ElevenLabs API Key for text-to-speech
            use_simulation: If True, simulate the robot, else use real robot
            robot_id: Robot identifier ("niryo" or "widowx")
            verbose: Enable verbose logging
            start_camera_thread: If True, start camera update thread
                                Set to False for MCP server!
        """
        self._use_simulation = use_simulation
        self._verbose = verbose
        self._logger = get_package_logger(__name__, verbose)

        self._logger.info("Initializing Environment")
        self._logger.debug(
            f"Configuration: simulation={use_simulation}, " f"robot_id={robot_id}, camera_thread={start_camera_thread}"
        )

        # Initialize robot (must come before framegrabber and workspaces)
        self._robot = Robot(self, use_simulation, robot_id, verbose)

        # Initialize robot-specific components
        if isinstance(self.get_robot_controller(), NiryoRobotController):
            self._framegrabber = NiryoFrameGrabber(self, verbose=verbose)
            self._workspaces = NiryoWorkspaces(self, verbose)
            self._logger.debug(f"Home workspace: {self._workspaces.get_home_workspace()}")
        elif isinstance(self.get_robot_controller(), WidowXRobotController):
            self._framegrabber = WidowXFrameGrabber(self, verbose=verbose)
            self._workspaces = WidowXWorkspaces(self, verbose)
        else:
            self._logger.error(f"Unknown robot controller type: {self.get_robot_controller()}")

        # Initialize text-to-speech
        self._oralcom = Text2Speech(el_api_key, verbose=verbose)

        # Thread control
        self._stop_event = threading.Event()

        # Initialize ObjectMemoryManager
        self._memory_manager = ObjectMemoryManager(manual_update_timeout=5.0, position_tolerance=0.05, verbose=verbose)

        # Initialize workspaces in memory manager
        if hasattr(self._workspaces, "__iter__"):
            try:
                for workspace in self._workspaces:
                    workspace_id = workspace.id()
                    self._memory_manager.initialize_workspace(workspace_id)
                    self._logger.debug(f"Initialized memory for workspace: {workspace_id}")
            except Exception as e:
                self._logger.warning(f"Could not iterate workspaces: {e}")
                if hasattr(self._workspaces, "get_workspace_home_id"):
                    default_ws_id = self._workspaces.get_workspace_home_id()
                    self._memory_manager.initialize_workspace(default_ws_id)

        # Current workspace tracking
        # self._current_workspace_id: Optional[str] = None
        self._current_workspace_id = self._workspaces.get_workspace_home_id()
        self._logger.debug(f"Set initial workspace to: {self._current_workspace_id}")

        # Redis-based communication
        self._object_broker = RedisMessageBroker()
        self._label_manager = RedisLabelManager()

        # Start camera thread if requested
        if start_camera_thread:
            self._logger.info("Starting camera update thread...")
            self.start_camera_updates(visualize=False)
        else:
            self._logger.info("Camera thread disabled (manual control)")

    def __del__(self):
        """Destructor."""
        if hasattr(self, "_stop_event"):
            self._logger.debug("Shutting down environment in destructor...")
            self._stop_event.set()

    def cleanup(self):
        """
        Explicit cleanup method - call when done with the object.
        More reliable than relying on __del__.
        """
        if hasattr(self, "_stop_event"):
            self._logger.info("Shutting down environment...")
            self._stop_event.set()

    # *** PUBLIC METHODS ***

    def start_camera_updates(self, visualize=False):
        """Start camera update thread."""

        def loop():
            for img in self.update_camera_and_objects(visualize=visualize):
                pass

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        return t

    def update_camera_and_objects(self, visualize: bool = False):
        """
        Continuously updates the camera and detected objects.

        Args:
            visualize: If True, displays the updated camera feed
        """
        t1 = 0.0

        # FIX: Get home workspace ID and set it as current
        home_workspace_id = self._workspaces.get_workspace_home_id()

        if self._current_workspace_id is None:
            self._current_workspace_id = home_workspace_id  # Set before moving

        self.robot_move2observation_pose(home_workspace_id)

        while not self._stop_event.is_set():
            t0 = time.time()

            # Get current frame (publishes to Redis)
            img = self.get_current_frame()
            t1 = time.time()
            self._logger.debug(f"Frame capture: {(t1 - t0) * 1000:.1f}ms")

            time.sleep(0.1)

            # Get detected objects from Redis
            detected_objects = self.get_detected_objects()
            t3 = time.time()
            self._logger.debug(f"Get objects: {(t3 - t1) * 1000:.1f}ms")

            # Update memory using ObjectMemoryManager
            if self._current_workspace_id:  # This should now always be True
                at_observation = self.is_any_workspace_visible()
                robot_moving = self.get_robot_in_motion()

                objects_added, objects_updated = self._memory_manager.update(
                    workspace_id=self._current_workspace_id,
                    detected_objects=detected_objects,
                    at_observation_pose=at_observation,
                    robot_in_motion=robot_moving,
                )

                self._logger.debug(
                    f"Memory update for '{self._current_workspace_id}': " f"added={objects_added}, updated={objects_updated}"
                )
            else:
                # This should never happen now, but log it if it does
                self._logger.error("Current workspace ID is None - memory not updated!")

            t5 = time.time()
            self._logger.debug(f"Total loop: {(t5 - t0) * 1000:.1f}ms")

            # Log memory stats
            if self._verbose:
                stats = self._memory_manager.get_memory_stats()
                if self._current_workspace_id in stats:
                    ws_stats = stats[self._current_workspace_id]
                    self._logger.debug(
                        f"Memory: {ws_stats['object_count']} objects, "
                        f"manual_updates={ws_stats['manual_updates']}, "
                        f"visible={ws_stats['visible']}\n"
                    )

            yield img

            if self.get_robot_in_motion():
                time.sleep(0.25)
            else:
                time.sleep(0.05)

    def clear_memory(self) -> None:
        """
        Manually clear all objects from memory.
        Useful when workspace has changed significantly.
        """
        self._logger.warning("Clearing memory of all objects")
        self._memory_manager.clear()

    def get_detected_objects_from_memory(self) -> "Objects":
        """
        Get a copy of the object memory for current workspace.
        Thread-safe access to memory.

        Returns:
            Objects: Copy of objects currently in memory
        """
        if self._current_workspace_id:
            return self._memory_manager.get(self._current_workspace_id)

        # Fallback: return home workspace memory
        home_ws_id = self._workspaces.get_workspace_home_id()
        return self._memory_manager.get(home_ws_id)

    def remove_object_from_memory(self, object_label: str, coordinate: List[float]) -> None:
        """
        Remove an object from memory after manipulation.

        Args:
            object_label: Label of the object to remove
            coordinate: Last known coordinate [x, y]
        """
        if not self._current_workspace_id:
            self._logger.warning("No current workspace set")
            return

        self._memory_manager.remove_object(
            workspace_id=self._current_workspace_id, object_label=object_label, coordinate=coordinate
        )

    def update_object_in_memory(self, object_label: str, old_coordinate: List[float], new_pose: "PoseObjectPNP") -> None:
        """
        Update an object's position in memory after movement.

        Args:
            object_label: Label of the object
            old_coordinate: Previous coordinate [x, y]
            new_pose: New pose after movement
        """
        if not self._current_workspace_id:
            self._logger.warning("No current workspace set")
            return

        self._memory_manager.mark_manual_update(
            workspace_id=self._current_workspace_id,
            object_label=object_label,
            old_coordinate=old_coordinate,
            new_pose=new_pose,
        )

    # Multi-workspace memory methods

    def get_current_workspace_id(self) -> Optional[str]:
        """Get the ID of the currently observed workspace."""
        return self._current_workspace_id

    def set_current_workspace(self, workspace_id: str) -> None:
        """Set the current workspace being observed."""
        self._current_workspace_id = workspace_id
        self._logger.debug(f"Current workspace set to: {workspace_id}")

    def get_detected_objects_from_workspace(self, workspace_id: str) -> Objects:
        """
        Get objects from a specific workspace memory.

        Args:
            workspace_id: ID of the workspace

        Returns:
            Objects: Copy of objects in that workspace's memory
        """
        return self._memory_manager.get(workspace_id)

    def get_all_workspace_objects(self) -> Dict[str, Objects]:
        """
        Get objects from all workspaces.

        Returns:
            Dict mapping workspace_id to Objects collection
        """
        return self._memory_manager.get_all()

    def clear_workspace_memory(self, workspace_id: str) -> None:
        """Clear memory for a specific workspace."""
        self._memory_manager.clear(workspace_id)

    def remove_object_from_workspace(self, workspace_id: str, object_label: str, coordinate: list) -> None:
        """Remove an object from a specific workspace's memory."""
        self._memory_manager.remove_object(workspace_id=workspace_id, object_label=object_label, coordinate=coordinate)

    def update_object_in_workspace(
        self, source_workspace_id: str, target_workspace_id: str, object_label: str, old_coordinate: list, new_coordinate: list
    ) -> None:
        """
        Move an object from one workspace to another in memory.

        Args:
            source_workspace_id: ID of workspace where object currently is
            target_workspace_id: ID of workspace where object will be placed
            object_label: Label of the object
            old_coordinate: Current coordinate in source workspace
            new_coordinate: New coordinate in target workspace
        """
        self._memory_manager.move_object(
            source_workspace_id=source_workspace_id,
            target_workspace_id=target_workspace_id,
            object_label=object_label,
            old_coordinate=old_coordinate,
            new_coordinate=new_coordinate,
        )

    # *** PUBLIC SET METHODS ***

    def add_object_name2object_labels(self, object_name: str) -> str:
        """
        Add a new object to the list of recognizable objects via Redis.

        Args:
            object_name: Name of the object to add

        Returns:
            str: Status message
        """
        # Add label to Redis stream
        success = self._label_manager.add_label(object_name)

        if success:
            mymessage = f"Added {object_name} to the list of recognizable objects."
        else:
            mymessage = f"{object_name} is already in the list of recognizable objects."

        # Provide audio feedback
        thread_oral = self._oralcom.call_text2speech_async(mymessage)
        thread_oral.join()

        return mymessage

    def stop_camera_updates(self):
        """Stop camera update thread."""
        self._stop_event.set()

    def oralcom_call_text2speech_async(self, text: str) -> threading.Thread:
        """
        Asynchronously call text-to-speech API.

        Args:
            text: Message for text-to-speech

        Returns:
            Thread object
        """
        return self._oralcom.call_text2speech_async(text)

    # *** PUBLIC GET METHODS ***

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

        x_max, y_max = workspace_top_left.x, workspace_top_left.y
        x_min, y_min = workspace_bottom_right.x, workspace_bottom_right.y

        self._logger.debug(f"Workspace bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
        self._logger.debug(f"Detected objects:\n{chr(10).join(obj.as_string_for_llm_lbl() for obj in detected_objects)}")

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

            self._logger.debug(f"Object bounds: x=[{x_start}, {x_end}], y=[{y_start}, {y_end}]")
            self._logger.debug(f"Grid coords: u=[{u_start}, {u_end}], v=[{v_start}, {v_end}]")

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

        if self.verbose:
            grid[v_center : v_center + 1, u_center : u_center + 1] = 2

            # Normalize grid to 0–255 for visualization
            grid_visual = (grid * 255 // 2).astype(np.uint8)

            cv2.imshow("grid", grid_visual)
            cv2.waitKey(0)

        self._logger.info(f"Largest free area: {largest_area_m2:.4f} square meters")
        self._logger.info(f"Center: ({center_x:.4f}, {center_y:.4f}) meters")

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
            self._logger.error(f"Unknown point type: {point}")
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
        """Get visible workspace from camera pose."""
        return self._workspaces.get_visible_workspace(camera_pose)

    def is_any_workspace_visible(self) -> bool:
        """Check if any workspace is currently visible."""
        pose = self.get_robot_pose()
        return self.get_visible_workspace(pose) is not None

    def get_observation_pose(self, workspace_id: str) -> "PoseObjectPNP":
        """
        Return the observation pose of the given workspace id

        Args:
            workspace_id: id of the workspace

        Returns:
            PoseObjectPNP: observation pose of the gripper where it can observe the workspace given by workspace_id
        """
        return self._workspaces.get_observation_pose(workspace_id)

    # Camera-related GET methods

    def get_current_frame(self) -> np.ndarray:
        """
        Capture image from robot's camera.

        Returns:
            numpy.ndarray: Camera image
        """
        return self._framegrabber.get_current_frame()

    def get_current_frame_width_height(self) -> tuple[int, int]:
        """
        Returns width and height of current frame in pixels.

        Returns:
            width and height of current frame in pixels.
        """
        return self._framegrabber.get_current_frame_width_height()

    # Robot-related GET methods

    def get_robot_controller(self) -> "RobotController":
        """

        Returns:
            RobotController: object that controls the robot.
        """
        return self._robot.robot()

    @log_start_end_cls()
    def get_robot_in_motion(self) -> bool:
        """
        Check if robot is in motion.

        Returns:
            bool: True if robot is moving, False otherwise
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
            PoseObjectPNP: Pose in world coordinates
        """
        return self._robot.get_target_pose_from_rel(workspace_id, u_rel, v_rel, yaw)

    # Vision-related GET methods

    def get_object_labels_as_string(self) -> str:
        """
        Return detectable object labels as comma-separated string.

        Returns:
            str: Comma-separated list of objects
        """
        object_labels = self.get_object_labels()

        if not object_labels or not object_labels[0]:
            return "No detectable objects configured."

        return f"I can recognize these objects: {', '.join(object_labels[0])}"

    def get_detected_objects(self) -> "Objects":
        """
        Get detected objects from Redis stream.

        Returns:
            Objects: Collection of detected objects
        """
        # Get latest objects from Redis (published by vision_detect_segment)
        objects_dict_list = self._object_broker.get_latest_objects(max_age_seconds=2.0)

        if not objects_dict_list:
            if self.verbose:
                print("No fresh object detections from Redis")
            return Objects()

        # Convert dictionaries to Object instances
        return Objects.dict_list_to_objects(objects_dict_list, self.get_workspace(0))

    def get_object_labels(self) -> List[List[str]]:
        """
        Get list of detectable object labels from Redis.

        Returns:
            List of lists of detectable strings
        """
        # Get latest labels from Redis (published by vision_detect_segment)
        labels = self._label_manager.get_latest_labels(timeout_seconds=60.0)

        if labels is None:
            if self.verbose:
                print("No labels from Redis, using empty list")
            return [[]]

        return [labels]

    # Robot control methods

    def robot_move2home_observation_pose(self) -> None:
        """Move robot to home workspace observation pose."""
        workspace_id = self.get_workspace_home_id()
        self.robot_move2observation_pose(workspace_id)

    def robot_move2observation_pose(self, workspace_id: str) -> None:
        """
        Move robot to observation pose for given workspace.

        Args:
            workspace_id: ID of workspace
        """
        self._robot.move2observation_pose(workspace_id)

        self._current_workspace_id = workspace_id
        self._logger.debug(f"Set current workspace to: {workspace_id}")

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    def workspaces(self) -> "Workspaces":
        """Return workspaces object."""
        return self._workspaces

    def framegrabber(self) -> "FrameGrabber":
        """Return framegrabber object."""
        return self._framegrabber

    def robot(self) -> "Robot":
        """Return robot object."""
        return self._robot

    def use_simulation(self) -> bool:
        """Check if using simulation."""
        return self._use_simulation

    @property
    def verbose(self) -> bool:
        """Check if verbose logging enabled."""
        return self._logger.isEnabledFor(logging.DEBUG)

    @verbose.setter
    def verbose(self, value: bool):
        """Set verbose logging on/off."""
        from .common.logger_config import set_verbose

        set_verbose(self._logger, value)

    # *** PRIVATE VARIABLES ***

    # Workspaces object
    _workspaces = None

    # FrameGraber object
    _framegrabber = None

    # Robot object
    _robot = None
    _use_simulation = False
    _verbose = False
    _memory_manager: Optional[ObjectMemoryManager] = None
    _current_workspace_id: Optional[str] = None
