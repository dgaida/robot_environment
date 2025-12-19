"""
Object Memory Manager for robot_environment package.

Manages object position memory with workspace tracking and thread-safe operations.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple
from robot_workspace import Objects, Object, PoseObjectPNP

import logging


class ObjectMemoryManager:
    """
    Manages object position memory with workspace tracking.

    Features:
    - Multi-workspace memory management
    - Manual update tracking (for pick/place operations)
    - Thread-safe operations
    - Workspace visibility state tracking
    - Intelligent memory clearing based on robot state

    Example:
        manager = ObjectMemoryManager(verbose=True)

        # Initialize workspaces
        manager.initialize_workspace("niryo_ws")
        manager.initialize_workspace("niryo_ws_left")

        # Update memory with detected objects
        manager.update("niryo_ws", detected_objects, at_observation=True)

        # Get objects from memory
        objects = manager.get("niryo_ws")

        # Manual update after placing object
        manager.mark_manual_update("niryo_ws", "cube",
                                   old_coord=[0.2, 0.0],
                                   new_pose=new_pose)
    """

    def __init__(self, manual_update_timeout: float = 5.0, position_tolerance: float = 0.05, verbose: bool = False):
        """
        Initialize the Object Memory Manager.

        Args:
            manual_update_timeout: Seconds to keep manual updates (default: 5.0)
            position_tolerance: Distance threshold for duplicate detection in meters (default: 0.05)
            verbose: Enable verbose logging (default: False)
        """
        # Workspace memories
        self._memories: Dict[str, Objects] = {}

        # Thread safety
        self._lock = threading.Lock()

        # Manual updates tracking: {workspace_id: {object_label: timestamp}}
        self._manual_updates: Dict[str, Dict[str, float]] = {}

        # Configuration
        self._manual_update_timeout = manual_update_timeout
        self._position_tolerance = position_tolerance

        # State tracking
        self._workspace_visibility: Dict[str, bool] = {}
        self._workspace_was_lost: Dict[str, bool] = {}

        # Logging
        self._verbose = verbose
        self._logger = logging.getLogger(__name__)
        if verbose:
            self._logger.setLevel(logging.DEBUG)

    def initialize_workspace(self, workspace_id: str) -> None:
        """
        Initialize memory for a new workspace.

        Args:
            workspace_id: ID of the workspace to initialize
        """
        with self._lock:
            if workspace_id not in self._memories:
                self._memories[workspace_id] = Objects()
                self._manual_updates[workspace_id] = {}
                self._workspace_visibility[workspace_id] = False
                self._workspace_was_lost[workspace_id] = False

                self._logger.debug(f"Initialized memory for workspace: {workspace_id}")

    def update(
        self, workspace_id: str, detected_objects: Objects, at_observation_pose: bool, robot_in_motion: bool
    ) -> Tuple[int, int]:
        """
        Update memory with newly detected objects.

        Only updates when conditions are appropriate (at observation pose, not moving).
        Respects manual updates from pick/place operations.

        Args:
            workspace_id: ID of the workspace
            detected_objects: Objects detected in current frame
            at_observation_pose: Whether robot is at observation pose
            robot_in_motion: Whether robot is currently moving

        Returns:
            Tuple of (objects_added, objects_updated)
        """
        with self._lock:
            # Ensure workspace is initialized
            if workspace_id not in self._memories:
                self.initialize_workspace(workspace_id)

            # Check if we should clear memory first
            if self._should_clear_memory(workspace_id, at_observation_pose, robot_in_motion):
                self._clear_workspace_internal(workspace_id)

            # Only update when at observation pose
            if not self._should_update_memory(at_observation_pose, robot_in_motion):
                self._logger.debug(f"Skipping memory update for {workspace_id} - conditions not met")
                return 0, 0

            # Update visibility tracking
            self._update_visibility_state(workspace_id, at_observation_pose, robot_in_motion)

            # Clean up expired manual updates
            self._cleanup_expired_manual_updates(workspace_id)

            # Update memory with new detections
            objects_added, objects_updated = self._merge_detections(workspace_id, detected_objects)

            if objects_added > 0 or objects_updated > 0:
                self._logger.debug(
                    f"Memory update for '{workspace_id}': "
                    f"added={objects_added}, updated={objects_updated}, "
                    f"total={len(self._memories[workspace_id])}"
                )

            return objects_added, objects_updated

    def get(self, workspace_id: str) -> Objects:
        """
        Get a copy of objects from workspace memory.

        Args:
            workspace_id: ID of the workspace

        Returns:
            Copy of objects in memory for this workspace
        """
        with self._lock:
            if workspace_id not in self._memories:
                self._logger.warning(f"Workspace {workspace_id} not initialized")
                return Objects()

            # Return a copy to avoid external modifications
            return Objects(list(self._memories[workspace_id]))

    def get_all(self) -> Dict[str, Objects]:
        """
        Get objects from all workspaces.

        Returns:
            Dictionary mapping workspace_id to Objects collection
        """
        with self._lock:
            return {ws_id: Objects(list(objects)) for ws_id, objects in self._memories.items()}

    def clear(self, workspace_id: Optional[str] = None) -> None:
        """
        Clear memory for specific workspace or all workspaces.

        Args:
            workspace_id: ID of workspace to clear, or None to clear all
        """
        with self._lock:
            if workspace_id is None:
                # Clear all workspaces
                for ws_id in self._memories:
                    self._clear_workspace_internal(ws_id)
                self._logger.info("Cleared memory for all workspaces")
            else:
                if workspace_id in self._memories:
                    self._clear_workspace_internal(workspace_id)
                    self._logger.info(f"Cleared memory for workspace: {workspace_id}")
                else:
                    self._logger.warning(f"Workspace {workspace_id} not found")

    def remove_object(self, workspace_id: str, object_label: str, coordinate: List[float]) -> bool:
        """
        Remove an object from workspace memory.

        Args:
            workspace_id: ID of the workspace
            object_label: Label of the object to remove
            coordinate: Last known coordinate [x, y]

        Returns:
            True if object was found and removed, False otherwise
        """
        with self._lock:
            if workspace_id not in self._memories:
                self._logger.warning(f"Workspace {workspace_id} not found")
                return False

            workspace_objects = self._memories[workspace_id]

            for i, obj in enumerate(workspace_objects):
                if obj.label() == object_label and self._is_same_position(obj, coordinate):

                    del workspace_objects[i]

                    # Clear manual update tracking
                    if workspace_id in self._manual_updates:
                        self._manual_updates[workspace_id].pop(object_label, None)

                    self._logger.info(f"Removed {object_label} from {workspace_id} at {coordinate}")
                    return True

            self._logger.warning(f"Could not find {object_label} at {coordinate} in {workspace_id}")
            return False

    def mark_manual_update(
        self, workspace_id: str, object_label: str, old_coordinate: List[float], new_pose: PoseObjectPNP
    ) -> bool:
        """
        Update an object's position after manual manipulation.

        Args:
            workspace_id: ID of the workspace
            object_label: Label of the object
            old_coordinate: Previous coordinate [x, y]
            new_pose: New pose after movement

        Returns:
            True if object was found and updated, False otherwise
        """
        with self._lock:
            if workspace_id not in self._memories:
                self._logger.warning(f"Workspace {workspace_id} not found")
                return False

            workspace_objects = self._memories[workspace_id]

            for obj in workspace_objects:
                if obj.label() == object_label and self._is_same_position(obj, old_coordinate):

                    # Update position
                    obj.set_pose_com(new_pose)

                    # Track manual update
                    if workspace_id not in self._manual_updates:
                        self._manual_updates[workspace_id] = {}
                    self._manual_updates[workspace_id][object_label] = time.time()

                    self._logger.info(
                        f"Updated {object_label} in {workspace_id}: "
                        f"{old_coordinate} -> [{new_pose.x:.3f}, {new_pose.y:.3f}]"
                    )
                    return True

            self._logger.warning(f"Could not find {object_label} at {old_coordinate} in {workspace_id}")
            return False

    def move_object(
        self,
        source_workspace_id: str,
        target_workspace_id: str,
        object_label: str,
        old_coordinate: List[float],
        new_coordinate: List[float],
    ) -> bool:
        """
        Move an object from one workspace to another in memory.

        Args:
            source_workspace_id: ID of source workspace
            target_workspace_id: ID of target workspace
            object_label: Label of the object
            old_coordinate: Current coordinate in source workspace
            new_coordinate: New coordinate in target workspace

        Returns:
            True if object was found and moved, False otherwise
        """
        with self._lock:
            # Validate workspaces
            if source_workspace_id not in self._memories:
                self._logger.warning(f"Source workspace {source_workspace_id} not found")
                return False

            if target_workspace_id not in self._memories:
                self.initialize_workspace(target_workspace_id)

            # Find and remove from source
            source_objects = self._memories[source_workspace_id]
            obj_to_move = None

            for i, obj in enumerate(source_objects):
                if obj.label() == object_label and self._is_same_position(obj, old_coordinate):

                    obj_to_move = obj
                    del source_objects[i]
                    break

            if obj_to_move is None:
                self._logger.warning(f"Could not find {object_label} at {old_coordinate} " f"in {source_workspace_id}")
                return False

            # Update object's position
            obj_to_move._x_com = new_coordinate[0]
            obj_to_move._y_com = new_coordinate[1]

            # Add to target workspace
            self._memories[target_workspace_id].append(obj_to_move)

            # Track manual update in target workspace
            if target_workspace_id not in self._manual_updates:
                self._manual_updates[target_workspace_id] = {}
            self._manual_updates[target_workspace_id][object_label] = time.time()

            self._logger.info(f"Moved {object_label} from {source_workspace_id} " f"to {target_workspace_id}")
            return True

    def get_memory_stats(self) -> Dict[str, Dict]:
        """
        Get statistics about memory contents.

        Returns:
            Dictionary with stats for each workspace
        """
        with self._lock:
            stats = {}
            for ws_id, objects in self._memories.items():
                manual_updates = self._manual_updates.get(ws_id, {})
                stats[ws_id] = {
                    "object_count": len(objects),
                    "manual_updates": len(manual_updates),
                    "visible": self._workspace_visibility.get(ws_id, False),
                    "was_lost": self._workspace_was_lost.get(ws_id, False),
                    "objects": [
                        {
                            "label": obj.label(),
                            "position": [obj.x_com(), obj.y_com()],
                            "manually_updated": obj.label() in manual_updates,
                        }
                        for obj in objects
                    ],
                }
            return stats

    # Private helper methods

    def _should_update_memory(self, at_observation_pose: bool, robot_in_motion: bool) -> bool:
        """Determine if memory should be updated."""
        return at_observation_pose and not robot_in_motion

    def _should_clear_memory(self, workspace_id: str, at_observation_pose: bool, robot_in_motion: bool) -> bool:
        """Determine if memory should be cleared."""
        was_lost = self._workspace_was_lost.get(workspace_id, False)
        now_visible = at_observation_pose and not robot_in_motion

        # Clear memory when workspace becomes visible again after being lost
        should_clear = was_lost and now_visible

        if should_clear:
            self._workspace_was_lost[workspace_id] = False

        return should_clear

    def _update_visibility_state(self, workspace_id: str, at_observation_pose: bool, robot_in_motion: bool) -> None:
        """Update workspace visibility tracking."""
        was_visible = self._workspace_visibility.get(workspace_id, False)
        now_visible = at_observation_pose and not robot_in_motion

        self._workspace_visibility[workspace_id] = now_visible

        # Detect when workspace is lost
        if was_visible and not now_visible:
            self._workspace_was_lost[workspace_id] = True
            self._logger.debug(f"Workspace {workspace_id} lost - robot moved")

        # Clear lost flag when workspace becomes visible again
        if now_visible and self._workspace_was_lost.get(workspace_id, False):
            self._logger.debug(f"Workspace {workspace_id} visible again - will clear memory on next update")

    def _cleanup_expired_manual_updates(self, workspace_id: str) -> None:
        """Remove expired manual updates."""
        if workspace_id not in self._manual_updates:
            return

        current_time = time.time()
        manual_updates = self._manual_updates[workspace_id]

        expired_labels = [
            label for label, timestamp in manual_updates.items() if current_time - timestamp > self._manual_update_timeout
        ]

        for label in expired_labels:
            del manual_updates[label]
            self._logger.debug(f"Manual update expired for {label} in {workspace_id}")

    def _merge_detections(self, workspace_id: str, detected_objects: Objects) -> Tuple[int, int]:
        """
        Merge new detections into memory.

        Returns:
            Tuple of (objects_added, objects_updated)
        """
        workspace_memory = self._memories[workspace_id]
        manual_updates = self._manual_updates.get(workspace_id, {})

        objects_added = 0
        objects_updated = 0

        for obj in detected_objects:
            x_center, y_center = obj.xy_com()
            label = obj.label()

            # Check if this object has a recent manual update
            if label in manual_updates:
                # Find the manually updated object
                found_manual = False
                for memory_obj in workspace_memory:
                    if memory_obj.label() == label:
                        manual_dist = ((memory_obj.x_com() - x_center) ** 2 + (memory_obj.y_com() - y_center) ** 2) ** 0.5

                        if manual_dist > self._position_tolerance:
                            # Keep manual update, ignore detection
                            self._logger.debug(f"Keeping manual update for {label} " f"(distance: {manual_dist:.3f}m)")
                            found_manual = True
                            break
                        else:
                            # Detection confirms manual update
                            memory_obj._x_com = x_center
                            memory_obj._y_com = y_center
                            objects_updated += 1
                            found_manual = True
                            self._logger.debug(f"Detection confirms manual update for {label}")
                            break

                if found_manual:
                    continue

            # Check if object already exists in memory
            is_duplicate = False
            for memory_obj in workspace_memory:
                if memory_obj.label() == label and self._is_same_position(memory_obj, [x_center, y_center]):
                    is_duplicate = True
                    break

            if not is_duplicate:
                workspace_memory.append(obj)
                objects_added += 1

        return objects_added, objects_updated

    def _is_same_position(self, obj: Object, coordinate: List[float]) -> bool:
        """Check if object is at the same position within tolerance."""
        return (
            abs(obj.x_com() - coordinate[0]) <= self._position_tolerance
            and abs(obj.y_com() - coordinate[1]) <= self._position_tolerance
        )

    def _clear_workspace_internal(self, workspace_id: str) -> None:
        """Internal method to clear workspace memory (assumes lock is held)."""
        if workspace_id in self._memories:
            count = len(self._memories[workspace_id])
            self._memories[workspace_id].clear()

            if workspace_id in self._manual_updates:
                self._manual_updates[workspace_id].clear()

            self._workspace_was_lost[workspace_id] = False

            self._logger.debug(f"Cleared {count} objects from {workspace_id}")
