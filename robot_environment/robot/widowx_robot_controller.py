# robot class around WidowX robot for smart pick and place
# Implementation based on InterbotixManipulatorXS API

from ..common.logger import log_start_end_cls
from .robot_controller import RobotController
from robot_workspace import PoseObjectPNP

# import threading
import numpy as np

try:
    from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

    INTERBOTIX_AVAILABLE = True
except ImportError:
    INTERBOTIX_AVAILABLE = False
    print("Warning: interbotix_xs_modules not available. WidowX controller will not function.")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .robot import Robot


class WidowXRobotController(RobotController):
    """
    Class for the pick-and-place WidowX robot that provides the primitive tasks of the robot like
    pick and place operations using the InterbotixManipulatorXS interface.
    """

    # *** CONSTRUCTORS ***
    @log_start_end_cls()
    def __init__(self, robot: "Robot", use_simulation: bool, verbose: bool = False):
        """
        Initializes the robot (connects to it and sets up the interface).

        Args:
            robot: object of the Robot class.
            use_simulation: True, if working with a simulation model of the robot,
                else False if we work with a real robot.
            verbose: enable verbose output
        """
        if not INTERBOTIX_AVAILABLE:
            raise ImportError("interbotix_xs_modules is required for WidowX controller")

        super().__init__(robot, use_simulation, verbose)

    # Deleting (Calling destructor)
    def __del__(self):
        super().__del__()
        if hasattr(self, "_robot_ctrl") and self._robot_ctrl is not None:
            if self.verbose():
                print("Shutting down WidowX robot controller...")
            with self._lock:
                self._shutdown()

    # *** PUBLIC GET methods ***

    def get_pose(self) -> "PoseObjectPNP":
        """
        Get current pose of gripper of robot.

        Returns:
            current pose of gripper of robot.
        """
        with self._lock:
            try:
                # Get end-effector pose components
                # InterbotixManipulatorXS stores current pose internally
                # We need to query the current joint states and compute forward kinematics
                # For simplicity, we'll use the last commanded pose or read from robot state

                # This is a simplified version - in practice you'd use the arm's FK
                # or track the last commanded pose
                if hasattr(self, "_last_pose") and self._last_pose is not None:
                    return self._last_pose
                else:
                    # Return home pose as default
                    return PoseObjectPNP(0.3, 0.0, 0.2, 0.0, 1.57, 0.0)
            except Exception as e:
                if self.verbose():
                    print(f"Error getting pose: {e}")
                return PoseObjectPNP(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def get_camera_intrinsics(self):
        """
        Get camera intrinsics for the WidowX camera (if available).

        Returns:
            tuple: (camera_matrix, distortion_coefficients)
        """
        # WidowX typically uses external camera (e.g., RealSense)
        # These are placeholder values - should be calibrated for your setup

        # Default camera matrix for Intel RealSense D435 (640x480)
        mtx = np.array([[615.0, 0.0, 320.0], [0.0, 615.0, 240.0], [0.0, 0.0, 1.0]])

        # Distortion coefficients (k1, k2, p1, p2, k3)
        dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        return mtx, dist

    # *** PUBLIC methods ***

    def calibrate(self) -> bool:
        """
        Calibrates the WidowX robot.

        Returns:
            True, if calibration was successful, else False
        """
        # TODO: implement calibration

        return True

    def reset_connection(self) -> None:
        """
        Reset the connection to the robot by safely disconnecting and reconnecting.
        """
        if self.verbose():
            print("Resetting WidowX connection...")
        try:
            if self._robot_ctrl is not None:
                with self._lock:
                    self._shutdown()
        except Exception as e:
            print(f"Error while closing connection: {e}")

        # Reinitialize the connection
        try:
            self._create_robot()
            if self.verbose():
                print("Connection successfully reset.")
        except Exception as e:
            print(f"Failed to reconnect to the robot: {e}")
            self._robot_ctrl = None

    @log_start_end_cls()
    def robot_pick_object(self, pick_pose: "PoseObjectPNP") -> bool:
        """
        Calls the pick command to pick an object at the given pose.

        Args:
            pick_pose: Pose where to pick the object (z-offset already applied if needed)

        Returns:
            True, if pick was successful, else False
        """
        try:
            # Add small z-offset for approach (additional to any z-offset already in pick_pose)
            pick_pose_approach = pick_pose.copy_with_offsets(z_offset=0.05)
            pick_pose_grasp = pick_pose  # Use the pose as-is (z-offset already applied)

            with self._lock:
                # Open gripper
                self._robot_ctrl.gripper.release()

                # Move to approach pose (above object)
                self._move_to_pose(pick_pose_approach)

                # Move down to grasp pose
                self._move_to_pose(pick_pose_grasp)

                # Close gripper to grasp
                self._robot_ctrl.gripper.grasp()

                # Lift object
                lift_pose = pick_pose.copy_with_offsets(z_offset=0.05)
                self._move_to_pose(lift_pose)

            if self.verbose():
                print("Pick operation completed successfully")

            return True

        except Exception as e:
            print(f"Error during pick operation: {e}")
            return False

    @log_start_end_cls()
    def robot_place_object(self, place_pose: "PoseObjectPNP") -> bool:
        """
        Places an already picked object at the given place_pose.

        Args:
            place_pose: Pose where to place the already picked object

        Returns:
            True, if place was successful, else False
        """
        try:
            # Add z-offset for approach
            place_pose_approach = place_pose.copy_with_offsets(z_offset=0.05)
            place_pose_final = place_pose.copy_with_offsets(z_offset=0.005)

            with self._lock:
                # Move to approach pose
                self._move_to_pose(place_pose_approach)

                # Move down to place pose
                self._move_to_pose(place_pose_final)

                # Release gripper
                self._robot_ctrl.gripper.release()

                # Retract
                self._move_to_pose(place_pose_approach)

            if self.verbose():
                print("Place operation completed successfully")

            return True

        except Exception as e:
            print(f"Error during place operation: {e}")
            return False

    @log_start_end_cls()
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
        try:
            with self._lock:
                # Close gripper first
                self._robot_ctrl.gripper.release()

                # Move to push starting position
                self._move_to_pose(push_pose)

                # Calculate push distance in meters
                push_dist_m = distance / 1000.0

                # Perform push based on direction
                if direction == "up":
                    # Push along positive X axis
                    self._robot_ctrl.arm.set_ee_cartesian_trajectory(x=push_dist_m)
                elif direction == "down":
                    # Push along negative X axis
                    self._robot_ctrl.arm.set_ee_cartesian_trajectory(x=-push_dist_m)
                elif direction == "left":
                    # Push along positive Y axis
                    self._robot_ctrl.arm.set_ee_cartesian_trajectory(y=push_dist_m)
                elif direction == "right":
                    # Push along negative Y axis
                    self._robot_ctrl.arm.set_ee_cartesian_trajectory(y=-push_dist_m)
                else:
                    print(f"Unknown direction: {direction}")
                    return False

            if self.verbose():
                print(f"Push operation completed: {direction}, {distance}mm")

            return True

        except Exception as e:
            print(f"Error during push operation: {e}")
            return False

    def get_target_pose_from_rel(self, workspace_id: str, u_rel: float, v_rel: float, yaw: float) -> "PoseObjectPNP":
        """
        Given relative image coordinates [u_rel, v_rel] and optionally an orientation of the point (yaw),
        calculate the corresponding pose in world coordinates.

        Args:
            workspace_id: id of the workspace
            u_rel: horizontal coordinate in image of workspace, normalized between 0 and 1
            v_rel: vertical coordinate in image of workspace, normalized between 0 and 1
            yaw: orientation of an object at the pixel coordinates [u_rel, v_rel].

        Returns:
            pose_object: Pose of the point in world coordinates of the robot.
        """
        with self._lock:
            try:
                # Clamp coordinates to [0, 1]
                u_rel = max(0.0, min(u_rel, 1.0))
                v_rel = max(0.0, min(v_rel, 1.0))

                # Get workspace from environment
                workspace = self._robot.environment().get_workspace_by_id(workspace_id)

                if workspace is None:
                    if self.verbose():
                        print(f"Workspace {workspace_id} not found")
                    return PoseObjectPNP(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                # Use workspace transformation
                pose = workspace.transform_camera2world_coords(workspace_id, u_rel, v_rel, yaw)

                return pose

            except Exception as e:
                if self.verbose():
                    print(f"Error in get_target_pose_from_rel: {e}")
                return PoseObjectPNP(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @log_start_end_cls()
    def move2observation_pose(self, workspace_id: str) -> None:
        """
        The robot moves to a pose where it can observe the workspace given by workspace_id.

        Args:
            workspace_id: id of the workspace
        """
        try:
            observation_pose = self._robot.environment().get_observation_pose(workspace_id)

            if observation_pose is None:
                if self.verbose():
                    print(f"No observation pose defined for workspace: {workspace_id}")
                return

            with self._lock:
                self._move_to_pose(observation_pose)

            if self.verbose():
                print(f"Moved to observation pose for workspace: {workspace_id}")

        except Exception as e:
            print(f"Error moving to observation pose: {e}")

    # *** PRIVATE methods ***

    def _shutdown(self) -> None:
        """
        Closes connection to InterbotixManipulatorXS.
        """
        if self._robot_ctrl is not None:
            try:
                # Move to sleep pose before shutdown
                self._robot_ctrl.arm.go_to_sleep_pose()
                # Shutdown the robot interface
                self._robot_ctrl.shutdown()
            except Exception as e:
                print(f"Error during shutdown: {e}")

    def _move_to_pose(self, pose: "PoseObjectPNP") -> None:
        """
        Move gripper of robot to given pose using set_ee_pose_components.

        Args:
            pose: pose of gripper (PoseObjectPNP)
        """
        try:
            # Use set_ee_pose_components method
            # Note: InterbotixManipulatorXS uses roll, pitch for orientation
            self._robot_ctrl.arm.set_ee_pose_components(x=pose.x, y=pose.y, z=pose.z, roll=pose.roll, pitch=pose.pitch)

            # Store last commanded pose
            self._last_pose = pose

        except Exception as e:
            if self.verbose():
                print(f"Error moving to pose: {e}")
            raise

    @log_start_end_cls()
    def _create_robot(self) -> None:
        """
        Creates the InterbotixManipulatorXS object and initializes the robot.
        """
        with self._lock:
            # Create robot interface
            self._robot_ctrl = InterbotixManipulatorXS(
                robot_model="wx250s",
                group_name="arm",
                gripper_name="gripper",
            )

            # Move to home pose on initialization
            self._robot_ctrl.arm.go_to_home_pose()

            # Initialize last pose
            self._last_pose = PoseObjectPNP(0.3, 0.0, 0.2, 0.0, 1.57, 0.0)

    @log_start_end_cls()
    def _init_robot(self, use_simulation: bool) -> bool:
        """
        Creates the InterbotixManipulatorXS object and connects to it.

        Args:
            use_simulation: Currently not used for WidowX (uses ROS parameter)

        Returns:
            bool: True if initialization was successful, else False
        """
        try:
            self._create_robot()

            if self.verbose():
                print("WidowX robot initialized successfully")

            return True

        except Exception as e:
            print(f"Failed to initialize WidowX robot: {e}")
            return False

    # *** PUBLIC properties ***

    # *** PRIVATE variables ***

    # Last commanded pose (for tracking)
    _last_pose = None
