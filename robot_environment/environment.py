# environment class in which a robot for smart pick and place exists
# Updated with proper logging throughout

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
            verbose: enable verbose output
            start_camera_thread: if True, start camera update thread (default: True)
                                Set to False for MCP server!
        """
        self._use_simulation = use_simulation
        self._verbose = verbose
        self._logger = get_package_logger(__name__, verbose)

        self._logger.info("Initializing Environment")
        self._logger.debug(
            f"Configuration: simulation={use_simulation}, " f"robot_id={robot_id}, camera_thread={start_camera_thread}"
        )

        # important that Robot comes before framegrabber and before workspace
        self._robot = Robot(self, use_simulation, robot_id, verbose)

        if isinstance(self.get_robot_controller(), NiryoRobotController):
            self._framegrabber = NiryoFrameGrabber(self, verbose=verbose)
            self._workspaces = NiryoWorkspaces(self, verbose)

            self._logger.debug(f"Home workspace: {self._workspaces.get_home_workspace()}")
        elif isinstance(self.get_robot_controller(), WidowXRobotController):
            self._framegrabber = WidowXFrameGrabber(self, verbose=verbose)
            self._workspaces = WidowXWorkspaces(self, verbose)
        else:
            self._logger.error(f"Unknown robot controller type: {self.get_robot_controller()}")

        self._oralcom = Text2Speech(el_api_key, verbose=verbose)

        self._stop_event = threading.Event()

        # Object memory management
        self._obj_position_memory = Objects()
        self._memory_lock = threading.Lock()
        self._is_at_observation_pose = False
        self._workspace_was_lost = False  # Track if workspace was lost during movement

        self._manual_memory_updates = {}  # Track manually updated objects: {label: timestamp}
        self._manual_update_timeout = 5.0  # seconds to keep manual updates

        # Enhanced multi-workspace memory management
        self._workspace_memories: Dict[str, Objects] = {}  # workspace_id -> Objects
        self._current_workspace_id: Optional[str] = None
        self._workspace_visibility_state: Dict[str, bool] = {}

        # Initialize memory for each workspace
        # SAFETY: Check if _workspaces is iterable before iterating
        if hasattr(self._workspaces, "__iter__"):
            try:
                for workspace in self._workspaces:
                    workspace_id = workspace.id()
                    self._workspace_memories[workspace_id] = Objects()
                    self._workspace_visibility_state[workspace_id] = False
            except Exception as e:
                self._logger.warning(f"Could not iterate workspaces: {e}")
                if hasattr(self._workspaces, "get_workspace_home_id"):
                    default_ws_id = self._workspaces.get_workspace_home_id()
                    self._workspace_memories[default_ws_id] = Objects()
                    self._workspace_visibility_state[default_ws_id] = False

        # Redis-based communication
        self._object_broker = RedisMessageBroker()
        self._label_manager = RedisLabelManager()

        if start_camera_thread:
            self._logger.info("Starting camera update thread...")
            self.start_camera_updates(visualize=False)
        else:
            self._logger.info("Camera thread disabled (manual control)")

    def __del__(self):
        """ """
        if hasattr(self, "_stop_event"):
            self._logger.debug("Shutting down environment in destructor...")
            self._stop_event.set()

    def cleanup(self):
        """
        Explicit cleanup method - call this when you're done with the object.
        This is more reliable than relying on __del__.
        """
        if hasattr(self, "_stop_event"):
            self._logger.info("Shutting down environment...")
            self._stop_event.set()

        # Close Redis connections
        if hasattr(self, "_object_broker"):
            # RedisMessageBroker doesn't need explicit cleanup
            pass

        if hasattr(self, "_label_manager"):
            # RedisLabelManager doesn't need explicit cleanup
            pass

    # PUBLIC methods

    def start_camera_updates(self, visualize=False):
        def loop():
            for img in self.update_camera_and_objects(visualize=visualize):
                pass

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        return t

    def _should_update_memory(self) -> bool:
        """
        Determine if object memory should be updated.
        Only update when at observation pose with full workspace visibility.

        Returns:
            bool: True if memory should be updated, False otherwise
        """
        # Check if workspace is currently visible
        workspace_visible = self.is_any_workspace_visible()
        robot_in_motion = self.get_robot_in_motion()

        # Memory should only be updated when:
        # 1. Workspace is visible
        # 2. Robot is not in motion
        # 3. We're at or near observation pose
        should_update = workspace_visible and not robot_in_motion

        if should_update:
            self._logger.debug("Memory update conditions met: workspace visible, robot stationary")

        return should_update

    def _should_clear_memory(self) -> bool:
        """
        Determine if object memory should be cleared.
        Clear when returning to observation pose after workspace was lost.

        Returns:
            bool: True if memory should be cleared, False otherwise
        """
        workspace_visible = self.is_any_workspace_visible()
        robot_in_motion = self.get_robot_in_motion()

        # If workspace is now visible but was previously lost, clear memory
        if workspace_visible and not robot_in_motion and self._workspace_was_lost:
            self._logger.debug("Clearing memory: returned to observation pose after workspace loss")
            return True

        return False

    def _track_workspace_visibility(self) -> None:
        """
        Track workspace visibility state to detect when workspace is lost/regained.
        """
        workspace_visible = self.is_any_workspace_visible()
        robot_in_motion = self.get_robot_in_motion()

        # Update tracking flags
        prev_at_observation = self._is_at_observation_pose
        self._is_at_observation_pose = workspace_visible and not robot_in_motion

        # Detect when workspace is lost (moving away from observation pose)
        if prev_at_observation and not self._is_at_observation_pose:
            self._workspace_was_lost = True
            self._logger.debug("Workspace lost - robot moved from observation pose")

    def _check_new_detections(self, detected_objects: "Objects") -> None:
        """
        Check for newly detected objects and update the memory with their positions.
        Only updates memory when conditions are appropriate (at observation pose).
        Respects manual updates from pick/place operations.

        Args:
            detected_objects (Objects): List of objects detected in the current frame.
        """
        import time

        with self._memory_lock:
            # Clear memory if we just returned to observation pose
            if self._should_clear_memory():
                self._logger.debug(f"Clearing memory of {len(self._obj_position_memory)} objects")
                self._obj_position_memory.clear()
                self._manual_memory_updates.clear()  # Also clear manual update tracking
                self._workspace_was_lost = False

            # Only update memory when at observation pose
            if not self._should_update_memory():
                self._logger.debug("Skipping memory update - conditions not met")
                return

            current_time = time.time()

            # Clean up expired manual updates
            expired_labels = [
                label
                for label, timestamp in self._manual_memory_updates.items()
                if current_time - timestamp > self._manual_update_timeout
            ]
            for label in expired_labels:
                del self._manual_memory_updates[label]
                self._logger.debug(f"Manual update timeout expired for {label}")

            # Update memory with new detections
            objects_added = 0
            objects_updated = 0

            for obj in detected_objects:
                x_center, y_center = obj.xy_com()
                label = obj.label()

                # Check if this object has a recent manual update
                if label in self._manual_memory_updates:
                    # Find the manually updated object in memory
                    found_manual = False
                    for memory_obj in self._obj_position_memory:
                        if memory_obj.label() == label:
                            # Check if detected position is close to manual update
                            manual_dist = ((memory_obj.x_com() - x_center) ** 2 + (memory_obj.y_com() - y_center) ** 2) ** 0.5

                            if manual_dist > 0.05:
                                self._logger.debug(
                                    f"Keeping manual update for {label} despite detection at different position"
                                )
                                found_manual = True
                                break
                            else:
                                # Detection confirms manual update, refresh it
                                memory_obj._x_com = x_center
                                memory_obj._y_com = y_center
                                objects_updated += 1
                                found_manual = True
                                self._logger.debug(f"Detection confirms manual update for {label}")
                                break

                    if found_manual:
                        continue

                # Check if object already exists in memory (within tolerance)
                is_duplicate = False
                for memory_obj in self._obj_position_memory:
                    if (
                        memory_obj.label() == label
                        and abs(memory_obj.x_com() - x_center) <= 0.05
                        and abs(memory_obj.y_com() - y_center) <= 0.05
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    self._obj_position_memory.append(obj)
                    objects_added += 1

            if objects_added > 0 or objects_updated > 0:
                self._logger.debug(
                    f"Memory update: added {objects_added}, updated {objects_updated} "
                    f"(total: {len(self._obj_position_memory)})"
                )
                self._logger.debug(f"Active manual updates: {list(self._manual_memory_updates.keys())}")

    def clear_memory(self) -> None:
        """
        Manually clear all objects from memory.
        Useful when you know the workspace has changed significantly.
        """
        self._logger.warning("Clearing memory of all objects")
        with self._memory_lock:
            self._logger.debug(f"Manually clearing memory of {len(self._obj_position_memory)} objects")
            self._obj_position_memory.clear()
            self._workspace_was_lost = False

    def remove_object_from_memory(self, object_label: str, coordinate: List[float]) -> None:
        """
        Remove an object from memory after it has been successfully manipulated.
        This prevents stale position data from being used.

        Args:
            object_label: Label of the object to remove
            coordinate: Last known coordinate [x, y] of the object
        """
        with self._memory_lock:
            removed = False
            for i, obj in enumerate(self._obj_position_memory):
                if (
                    obj.label() == object_label
                    and abs(obj.x_com() - coordinate[0]) <= 0.05
                    and abs(obj.y_com() - coordinate[1]) <= 0.05
                ):
                    del self._obj_position_memory[i]

                    # Clear manual update tracking for this object
                    if object_label in self._manual_memory_updates:
                        del self._manual_memory_updates[object_label]

                    removed = True
                    self._logger.info(f"Removed {object_label} from memory at {coordinate}")
                    break

            if not removed:
                self._logger.warning(f"Could not find {object_label} in memory to remove")
                self._logger.debug(
                    f"Memory contents: {[(obj.label(), [obj.x_com(), obj.y_com()]) for obj in self._obj_position_memory]}"
                )

    def update_object_in_memory(self, object_label: str, old_coordinate: List[float], new_pose: "PoseObjectPNP") -> None:
        """
        Update an object's position in memory after it has been moved.

        Args:
            object_label: Label of the object
            old_coordinate: Previous coordinate [x, y]
            new_pose: New pose after movement
        """
        import time

        with self._memory_lock:
            updated = False
            for obj in self._obj_position_memory:
                if (
                    obj.label() == object_label
                    and abs(obj.x_com() - old_coordinate[0]) <= 0.025
                    and abs(obj.y_com() - old_coordinate[1]) <= 0.025
                ):
                    # Update position
                    obj.set_pose_com(new_pose)
                    self._manual_memory_updates[object_label] = time.time()

                    updated = True
                    self._logger.info(f"Updated {object_label} position in memory: {old_coordinate} -> {new_pose}")
                    break

            if not updated:
                self._logger.warning(f"Could not find {object_label} in memory to update")
                self._logger.debug(
                    f"Memory contents: {[(obj.label(), [obj.x_com(), obj.y_com()]) for obj in self._obj_position_memory]}"
                )

    def get_detected_objects_from_memory(self) -> "Objects":
        """
        Get a copy of the object memory.
        Thread-safe access to memory.

        Returns:
            Objects: Copy of objects currently in memory
        """
        with self._memory_lock:
            # Return a copy to avoid external modifications
            return Objects(list(self._obj_position_memory))

    def update_camera_and_objects(self, visualize: bool = False):
        """
        Continuously updates the camera and detected objects.

        Args:
            visualize (bool): If True, displays the updated camera feed in a window.
        """
        t1 = 0.0

        self.robot_move2observation_pose(self._workspaces.get_workspace_home_id())

        while not self._stop_event.is_set():
            t0 = time.time()

            # Track workspace visibility state
            self._track_workspace_visibility()

            # this method gets a new frame from camera and publishes it to redis streamer
            img = self.get_current_frame()
            t1 = time.time()
            self._logger.debug(f"Frame capture: {(t1 - t0) * 1000:.1f}ms")

            time.sleep(0.1)

            detected_objects = self.get_detected_objects()
            t3 = time.time()
            self._logger.debug(f"Get objects: {(t3 - t1) * 1000:.1f}ms")

            self._check_new_detections(detected_objects)

            t5 = time.time()
            self._logger.debug(f"Total loop: {(t5 - t0) * 1000:.1f}ms")
            self._logger.debug(
                f"Memory status: {len(self._obj_position_memory)} objects, "
                f"at_observation={self._is_at_observation_pose}, "
                f"workspace_lost={self._workspace_was_lost}\n"
            )

            yield img

            if self.get_robot_in_motion():
                time.sleep(0.25)
            else:
                time.sleep(0.05)

    # *** PUBLIC SET methods ***

    def add_object_name2object_labels(self, object_name: str) -> str:
        """
        Add a new object to the list of recognizable objects via Redis.
        The vision_detect_segment system will pick up this change.

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

        # self._logger.debug(f"Largest free area: {largest_area_m2:.4f} m²")
        # self._logger.debug(f"Center: ({center_x:.4f}, {center_y:.4f})")

        self._logger.info(f"Largest free area: {largest_area_m2:.4f} square meters")
        self._logger.info(f"Center of the largest free area: ({center_x:.4f}, {center_y:.4f}) meters")

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
        return self._workspaces.get_visible_workspace(camera_pose)

    def is_any_workspace_visible(self) -> bool:
        pose = self.get_robot_pose()
        if self.get_visible_workspace(pose) is None:
            return False
        else:
            return True

    def get_current_workspace_id(self) -> Optional[str]:
        """Get the ID of the currently observed workspace."""
        return self._current_workspace_id

    def set_current_workspace(self, workspace_id: str) -> None:
        """Set the current workspace being observed."""
        if workspace_id in self._workspace_memories:
            self._current_workspace_id = workspace_id
            self._logger.debug(f"Current workspace set to: {workspace_id}")
        else:
            self._logger.warning(f"Workspace '{workspace_id}' not found")

    def get_detected_objects_from_workspace(self, workspace_id: str) -> Objects:
        """
        Get objects from a specific workspace memory.

        Args:
            workspace_id: ID of the workspace

        Returns:
            Objects: Copy of objects in that workspace's memory
        """
        with self._memory_lock:
            if workspace_id in self._workspace_memories:
                return Objects(list(self._workspace_memories[workspace_id]))
            return Objects()

    def get_all_workspace_objects(self) -> Dict[str, Objects]:
        """
        Get objects from all workspaces.

        Returns:
            Dict mapping workspace_id to Objects collection
        """
        with self._memory_lock:
            return {ws_id: Objects(list(objects)) for ws_id, objects in self._workspace_memories.items()}

    def clear_workspace_memory(self, workspace_id: str) -> None:
        """Clear memory for a specific workspace."""
        with self._memory_lock:
            if workspace_id in self._workspace_memories:
                self._logger.debug(f"Clearing memory for workspace: {workspace_id}")
                self._workspace_memories[workspace_id].clear()

    def remove_object_from_workspace(self, workspace_id: str, object_label: str, coordinate: list) -> None:
        """Remove an object from a specific workspace's memory."""
        with self._memory_lock:
            if workspace_id not in self._workspace_memories:
                return

            workspace_objects = self._workspace_memories[workspace_id]
            for i, obj in enumerate(workspace_objects):
                if (
                    obj.label() == object_label
                    and abs(obj.x_com() - coordinate[0]) <= 0.05
                    and abs(obj.y_com() - coordinate[1]) <= 0.05
                ):
                    del workspace_objects[i]
                    self._logger.debug(f"Removed {object_label} from workspace {workspace_id}")
                    break

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
        import time

        with self._memory_lock:
            # Remove from source workspace
            if source_workspace_id in self._workspace_memories:
                source_objects = self._workspace_memories[source_workspace_id]
                obj_to_move = None

                for i, obj in enumerate(source_objects):
                    if (
                        obj.label() == object_label
                        and abs(obj.x_com() - old_coordinate[0]) <= 0.05
                        and abs(obj.y_com() - old_coordinate[1]) <= 0.05
                    ):
                        obj_to_move = obj
                        del source_objects[i]
                        break

                # Add to target workspace with updated position
                if obj_to_move and target_workspace_id in self._workspace_memories:
                    # Update object's position
                    obj_to_move._x_com = new_coordinate[0]
                    obj_to_move._y_com = new_coordinate[1]

                    # Update workspace reference
                    target_workspace = self.get_workspace_by_id(target_workspace_id)
                    if target_workspace:
                        obj_to_move._workspace = target_workspace

                    self._workspace_memories[target_workspace_id].append(obj_to_move)

                    # Track manual update
                    if not hasattr(self, "_manual_memory_updates"):
                        self._manual_memory_updates = {}
                    self._manual_memory_updates[object_label] = time.time()

                    self._logger.debug(f"Moved {object_label} from {source_workspace_id} to {target_workspace_id}")

    def _check_new_detections_multi_workspace(self, detected_objects: Objects) -> None:
        """
        Check for newly detected objects and update the appropriate workspace memory.
        Enhanced version for multi-workspace support.
        """
        import time

        with self._memory_lock:
            current_ws_id = self._current_workspace_id
            if current_ws_id is None:
                return

            # Clear workspace memory if returning to observation pose
            if self._should_clear_memory():
                if self.verbose():
                    print(f"Clearing memory for workspace: {current_ws_id}")
                self._workspace_memories[current_ws_id].clear()
                if hasattr(self, "_manual_memory_updates"):
                    self._manual_memory_updates.clear()
                self._workspace_was_lost = False

            # Only update memory when at observation pose
            if not self._should_update_memory():
                return

            current_time = time.time()
            workspace_memory = self._workspace_memories[current_ws_id]

            # Clean up expired manual updates
            if hasattr(self, "_manual_memory_updates"):
                expired_labels = [
                    label
                    for label, timestamp in self._manual_memory_updates.items()
                    if current_time - timestamp > self._manual_update_timeout
                ]
                for label in expired_labels:
                    del self._manual_memory_updates[label]

            # Update memory with new detections
            objects_added = 0
            objects_updated = 0

            for obj in detected_objects:
                x_center, y_center = obj.xy_com()
                label = obj.label()

                # Check for manual updates
                if hasattr(self, "_manual_memory_updates") and label in self._manual_memory_updates:
                    found_manual = False
                    for memory_obj in workspace_memory:
                        if memory_obj.label() == label:
                            manual_dist = ((memory_obj.x_com() - x_center) ** 2 + (memory_obj.y_com() - y_center) ** 2) ** 0.5

                            if manual_dist > 0.05:
                                if self.verbose():
                                    print(f"Keeping manual update for {label}")
                                found_manual = True
                                break
                            else:
                                memory_obj._x_com = x_center
                                memory_obj._y_com = y_center
                                objects_updated += 1
                                found_manual = True
                                break

                    if found_manual:
                        continue

                # Check if object already exists in memory
                is_duplicate = False
                for memory_obj in workspace_memory:
                    if (
                        memory_obj.label() == label
                        and abs(memory_obj.x_com() - x_center) <= 0.05
                        and abs(memory_obj.y_com() - y_center) <= 0.05
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    workspace_memory.append(obj)
                    objects_added += 1

            if self.verbose() and (objects_added > 0 or objects_updated > 0):
                print(
                    f"Workspace '{current_ws_id}' memory update: "
                    f"added {objects_added}, updated {objects_updated} "
                    f"(total: {len(workspace_memory)})"
                )

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
        Returns all object labels that the object detection model is able to detect
        as a comma separated string.

        Returns:
            str: Comma-separated list of detectable objects
        """
        object_labels = self.get_object_labels()

        if not object_labels or not object_labels[0]:
            return "No detectable objects configured."

        return f"I can recognize these objects: {', '.join(object_labels[0])}"

    def get_detected_objects(self) -> "Objects":
        """
        Get detected objects directly from Redis stream.

        Returns:
            Objects: Collection of detected objects
        """
        # Get latest objects from Redis (published by vision_detect_segment)
        objects_dict_list = self._object_broker.get_latest_objects(max_age_seconds=2.0)

        if not objects_dict_list:
            if self.verbose():
                print("No fresh object detections available from Redis")
            return Objects()

        # Convert dictionaries to Object instances
        return Objects.dict_list_to_objects(objects_dict_list, self.get_workspace(0))

    def get_object_labels(self) -> List[List[str]]:
        """
        Get list of detectable object labels from Redis.

        Returns:
            List of lists of detectable object strings
        """
        # Get latest labels from Redis (published by vision_detect_segment)
        labels = self._label_manager.get_latest_labels(timeout_seconds=60.0)

        if labels is None:
            if self.verbose():
                print("No labels available from Redis, using empty list")
            return [[]]

        # Return in the expected nested list format
        return [labels]

    # *** PUBLIC methods ***

    # methods from Robot

    def robot_move2home_observation_pose(self) -> None:
        """
        The robot is going to move to a pose where it can observe (the gripper hovers over) the home workspace.
        """
        workspace_id = self.get_workspace_home_id()
        self.robot_move2observation_pose(workspace_id)

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

    @property
    def verbose(self) -> bool:
        """Check if verbose (DEBUG) logging is enabled."""
        return self._logger.isEnabledFor(logging.DEBUG)

    @verbose.setter
    def verbose(self, value: bool):
        """Set verbose logging on/off."""
        from .common.logger_config import set_verbose

        set_verbose(self._logger, value)

    # *** PRIVATE variables ***

    # Workspaces object
    _workspaces = None

    # FrameGraber object
    _framegrabber = None

    # Robot object
    _robot = None

    _use_simulation = False

    _verbose = False

    # Memory management
    _obj_position_memory = None
    _memory_lock = None
    _is_at_observation_pose = False
    _workspace_was_lost = False
