# environment class in which a robot for smart pick and place exists
# Refactored to use ObjectMemoryManager for cleaner memory management
from __future__ import annotations

import threading
from .common.logger import log_start_end_cls
import numpy as np
import time

try:
    from robot_workspace import Workspaces
except ImportError:
    from robot_workspace.workspaces.workspaces import Workspaces
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
from .performance_metrics import PerformanceMetrics, PerformanceMonitor
from .utils.workspace_utils import calculate_largest_free_space

from .common.logger_config import get_package_logger
import logging

from typing import TYPE_CHECKING, List, Optional, Dict, Tuple

if TYPE_CHECKING:
    from robot_workspace import Workspace
    try:
        from robot_workspace import Workspaces
    except ImportError:
        from robot_workspace.workspaces.workspaces import Workspaces
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
        self,
        el_api_key: str,
        use_simulation: bool,
        robot_id: str,
        verbose: bool = False,
        start_camera_thread: bool = True,
        enable_performance_monitoring: bool = True,
        performance_log_interval: float = 60.0,
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
            enable_performance_monitoring: Enable performance metrics tracking
            performance_log_interval: Interval in seconds for performance logging
        """
        self._use_simulation = use_simulation
        self._verbose = verbose
        self._logger = get_package_logger(__name__, verbose)

        self._logger.info("Initializing Environment")
        self._logger.debug(
            f"Configuration: simulation={use_simulation}, "
            f"robot_id={robot_id}, camera_thread={start_camera_thread}, "
            f"metrics={enable_performance_monitoring}"
        )

        # Initialize performance metrics
        self._metrics = PerformanceMetrics(history_size=100, verbose=verbose) if enable_performance_monitoring else None

        self._performance_monitor: Optional[PerformanceMonitor] = None
        if enable_performance_monitoring:
            self._performance_monitor = PerformanceMonitor(
                self._metrics, interval_seconds=performance_log_interval, verbose=verbose
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
        self._oralcom = Text2Speech(
            el_api_key,
            verbose=verbose,
            enable_queue=True,  # Enable built-in audio queue
            max_queue_size=50,
            duplicate_timeout=2.0,
        )

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

        # Start performance monitor if enabled
        if self._performance_monitor:
            self._performance_monitor.start()
            self._logger.info("Performance monitoring started")

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

        if hasattr(self, "_performance_monitor") and self._performance_monitor:
            self._performance_monitor.stop()

    def cleanup(self):
        """
        Explicit cleanup method - call when done with the object.
        More reliable than relying on __del__.
        """
        if hasattr(self, "_stop_event"):
            self._logger.info("Shutting down environment...")
            self._stop_event.set()

        if hasattr(self, "_performance_monitor") and self._performance_monitor:
            self._performance_monitor.stop()

        if hasattr(self, "_oralcom"):
            self._oralcom.shutdown(timeout=5.0)

    # *** PUBLIC METHODS ***

    def start_camera_updates(self, visualize: bool = False) -> threading.Thread:
        """
        Start the background camera update thread.

        Args:
            visualize: If True, show the camera feed (requires GUI).

        Returns:
            The started threading.Thread object.
        """

        def loop():
            for _ in self.update_camera_and_objects(visualize=visualize):
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
        # FIX: Get home workspace ID and set it as current
        home_workspace_id = self._workspaces.get_workspace_home_id()

        if self._current_workspace_id is None:
            self._current_workspace_id = home_workspace_id  # Set before moving

        self.robot_move2observation_pose(home_workspace_id)

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            # Get current frame with timing
            if self._metrics:
                with self._metrics.timer("frame_capture"):
                    img = self.get_current_frame()
            else:
                img = self.get_current_frame()

            time.sleep(0.1)

            # Get detected objects from Redis
            # Get detected objects from Redis with timing
            if self._metrics:
                with self._metrics.timer("object_fetch_redis"):
                    detected_objects = self.get_detected_objects()

                # Record object count
                self._metrics.increment_counter("objects_detected", len(detected_objects))
            else:
                detected_objects = self.get_detected_objects()

            # Update memory using ObjectMemoryManager
            if self._current_workspace_id:  # This should now always be True
                at_observation = self.is_any_workspace_visible()
                robot_moving = self.get_robot_in_motion()

                if self._metrics:
                    mem_start = time.perf_counter()

                objects_added, objects_updated = self._memory_manager.update(
                    workspace_id=self._current_workspace_id,
                    detected_objects=detected_objects,
                    at_observation_pose=at_observation,
                    robot_in_motion=robot_moving,
                )

                if self._metrics:
                    mem_duration = (time.perf_counter() - mem_start) * 1000
                    self._metrics.record_memory_update(mem_duration, objects_added, objects_updated)

                self._logger.debug(
                    f"Memory update for '{self._current_workspace_id}': " f"added={objects_added}, updated={objects_updated}"
                )
            else:
                # This should never happen now, but log it if it does
                self._logger.error("Current workspace ID is None - memory not updated!")

            # Record loop iteration time
            if self._metrics:
                loop_duration = (time.perf_counter() - loop_start) * 1000
                self._metrics.record_timing("camera_loop_iteration", loop_duration)

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
                time.sleep(0.05)
            else:
                time.sleep(0.05)

    def clear_memory(self) -> None:
        """
        Manually clear all objects from memory.
        Useful when workspace has changed significantly.
        """
        self._logger.warning("Clearing memory of all objects")

        if self._metrics:
            with self._metrics.timer("memory_clear"):
                self._memory_manager.clear()
            self._metrics.increment_counter("memory_clears")
        else:
            self._memory_manager.clear()

    def get_detected_objects_from_memory(self) -> Objects:
        """
        Get a copy of the object memory for current workspace.

        Returns:
            Objects: Copy of objects currently in memory
        """
        if self._metrics:
            with self._metrics.timer("memory_get"):
                result = self._get_memory_internal()
            return result
        else:
            return self._get_memory_internal()

    def _get_memory_internal(self) -> Objects:
        """Internal method for getting memory."""
        if self._current_workspace_id:
            return self._memory_manager.get(self._current_workspace_id)

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

    # Performance metrics methods

    def get_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """
        Get the performance metrics tracker.

        Returns:
            PerformanceMetrics instance or None if disabled
        """
        return self._metrics

    def get_performance_stats(self) -> Optional[Dict]:
        """
        Get current performance statistics.

        Returns:
            Dictionary with performance stats or None if disabled
        """
        if self._metrics:
            return self._metrics.get_stats()
        return None

    def print_performance_summary(self) -> None:
        """Print a human-readable performance summary."""
        if self._metrics:
            print(self._metrics.get_summary())
        else:
            print("Performance monitoring is disabled")

    def export_performance_metrics(self, filepath: str) -> None:
        """
        Export performance metrics to JSON file.

        Args:
            filepath: Path to output file
        """
        if self._metrics:
            self._metrics.export_json(filepath)
            self._logger.info(f"Performance metrics exported to {filepath}")
        else:
            self._logger.warning("Performance monitoring is disabled")

    def reset_performance_metrics(self) -> None:
        """Reset all performance metrics."""
        if self._metrics:
            self._metrics.reset()
            self._logger.info("Performance metrics reset")

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

    def stop_camera_updates(self) -> None:
        """Stop camera update thread."""
        self._stop_event.set()

    def oralcom_call_text2speech_async(self, text: str, priority: int = 0) -> bool:
        """
        Asynchronously call text-to-speech API.

        Args:
            text: Message for text-to-speech
            priority: Priority (0-10, higher = more urgent)

        Returns:
            True if queued successfully (or dummy thread for compatibility)
        """
        return self._oralcom.speak(text, priority=priority, blocking=False)

    # *** PUBLIC GET METHODS ***

    def get_largest_free_space_with_center(self, workspace_id: Optional[str] = None) -> Tuple[float, float, float]:
        """
        Determines the largest free space in the workspace in square metres and its center coordinate in metres.
        This method can be used to determine at which location an object can be placed safely.

        Args:
            workspace_id: Optional ID of the workspace to analyze. If None, the home workspace is used.

        Returns:
            tuple: (largest_free_area_m2, center_x, center_y) where:
                - largest_area_m2 (float): Largest free area in square meters.
                - center_x (float): X-coordinate of the center of the largest free area in meters.
                - center_y (float): Y-coordinate of the center of the largest free area in meters.
        """
        if workspace_id:
            workspace = self.get_workspace_by_id(workspace_id)
        else:
            workspace = self.get_workspace(0)

        if workspace is None:
            self._logger.error(f"Workspace not found: {workspace_id}")
            return 0.0, 0.0, 0.0

        detected_objects = self.get_detected_objects()

        return calculate_largest_free_space(
            workspace=workspace, detected_objects=detected_objects, visualize=self.verbose, logger=self._logger
        )

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

    def get_workspace(self, index: int = 0) -> Workspace:
        """
        Return the workspace at the given position index in the list of workspaces.

        Args:
            index: 0-based index in the list of workspaces.

        Returns:

        """
        return self._workspaces.get_workspace(index)

    def get_workspace_by_id(self, workspace_id: str) -> Optional[Workspace]:
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
    def get_visible_workspace(self, camera_pose: PoseObjectPNP) -> Workspace:
        """Get visible workspace from camera pose."""
        return self._workspaces.get_visible_workspace(camera_pose)

    def is_any_workspace_visible(self) -> bool:
        """Check if any workspace is currently visible."""
        pose = self.get_robot_pose()
        return self.get_visible_workspace(pose) is not None

    def get_observation_pose(self, workspace_id: str) -> PoseObjectPNP:
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
        frame = self._framegrabber.get_current_frame()

        if self._metrics and frame is not None:
            self._metrics.increment_counter("frames_captured")

        return frame

    def get_current_frame_width_height(self) -> tuple[int, int]:
        """
        Returns width and height of current frame in pixels.

        Returns:
            width and height of current frame in pixels.
        """
        return self._framegrabber.get_current_frame_width_height()

    # Robot-related GET methods

    def get_robot_controller(self) -> RobotController:
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

    def get_robot_pose(self) -> PoseObjectPNP:
        """
        Get current pose of gripper of robot.

        Returns:
            current pose of gripper of robot.
        """
        if self._metrics:
            with self._metrics.timer("robot_get_pose"):
                return self._robot.get_pose()
        else:
            return self._robot.get_pose()

    @log_start_end_cls()
    def get_robot_target_pose_from_rel(self, workspace_id: str, u_rel: float, v_rel: float, yaw: float) -> PoseObjectPNP:
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

    def get_detected_objects(self) -> Objects:
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
        if self._metrics:
            with self._metrics.timer("robot_move_observation"):
                self._robot.move2observation_pose(workspace_id)
        else:
            self._robot.move2observation_pose(workspace_id)

        self._current_workspace_id = workspace_id
        self._logger.debug(f"Set current workspace to: {workspace_id}")

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    def workspaces(self) -> Workspaces:
        """Return workspaces object."""
        return self._workspaces

    def framegrabber(self) -> FrameGrabber:
        """Return framegrabber object."""
        return self._framegrabber

    def robot(self) -> Robot:
        """Return robot object."""
        return self._robot

    def use_simulation(self) -> bool:
        """Check if using simulation."""
        return self._use_simulation

    def metrics(self) -> Optional[PerformanceMetrics]:
        """Get performance metrics tracker."""
        return self._metrics

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
