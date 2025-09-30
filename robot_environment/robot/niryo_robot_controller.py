# robot class around Niryo robot for smart pick and place
# a few TODOs
# Documentation and type definitions are final (chatgpt can improve it)

from ..common.logger import log_start_end_cls, pyniryo_v

from .robot_controller import RobotController
from ..objects.pose_object import PoseObjectPNP

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

if pyniryo_v == "pyniryo2":
    from pyniryo2 import RobotAxis, NiryoRobot, PoseObject
    import pyniryo
else:
    from pyniryo import NiryoRobot
    from pyniryo.api.objects import PoseObject
    from pyniryo.api.enums_communication import RobotAxis
from pyniryo.api.exceptions import NiryoRobotException  # RobotCommandException
from pyniryo.api.exceptions import TcpCommandException  # RobotCommandException

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..objects.pose_object import PoseObjectPNP
    from ..objects.object import Object
    from .robot import Robot


class NiryoRobotController(RobotController):
    """
    Class for the pick-and-place niryo ned2 robot that provides the primitive tasks of the robot like
    pick and place operations.

    """

    # *** CONSTRUCTORS ***
    @log_start_end_cls()
    def __init__(self, robot: "Robot", use_simulation: bool, verbose: bool = False):
        """
        Initializes the robot (connects to it and calibrates it).

        Args:
            robot: object of the Robot class.
            use_simulation: True, if working with a simulation model of the robot,
            else False if we work with a real robot.
            verbose:
        """
        self._executor = ThreadPoolExecutor(max_workers=1)  # Dedicated thread pool for long-running tasks

        # print(pyniryo.__version__)
        self._shutdown_v = False

        super().__init__(robot, use_simulation, verbose)

    # Deleting (Calling destructor)
    def __del__(self):
        super().__del__()

        if hasattr(self, '_executor') and self._executor:
            print("Shutting down ThreadPoolExecutor in destructor...")
            self._executor.shutdown(wait=True)

        print('Destructor called, Robot deleted.')
        with self._lock:
            print('Destructor called, Robot deleted.2')
            self._shutdown()

    # ============================================
    # METHOD 2: Explicit cleanup method (RECOMMENDED)
    # ============================================
    def cleanup(self):
        """
        Explicit cleanup method - call this when you're done with the object.
        This is more reliable than relying on __del__.
        """
        if hasattr(self, '_executor') and self._executor:
            print("Shutting down ThreadPoolExecutor...")
            self._shutdown_v = True
            self._executor.shutdown(wait=True)
            self._executor = None

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    def get_pose(self) -> "PoseObjectPNP":
        """
        Get current pose of gripper of robot.

        Returns:
            current pose of gripper of robot.
        """
        with self._lock:
            if pyniryo_v == "pyniryo2":
                pose = self._robot_ctrl.arm.get_pose()
            else:
                pose = self._robot_ctrl.get_pose()

        return PoseObjectPNP.convert_niryo_pose_object2pose_object(pose)

    # *** PUBLIC methods ***

    def reset_connection(self) -> None:
        """
        Reset the connection to the robot by safely disconnecting and reconnecting.
        """
        print("Resetting the robot connection...")
        try:
            # Attempt to close the connection safely
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
            self._robot = None

    @log_start_end_cls()
    def robot_pick_object(self, obj2pick: "Object") -> bool:
        """
        Calls the pick command of the self._robot_ctrl to pick the given Object

        Args:
            obj2pick: Object that shall be picked

        Returns:
            True, if pick was successful, else False
        """
        pick_pose = obj2pick.pose_com()
        pick_pose = pick_pose.copy_with_offsets(z_offset=0.001)
        pick_pose = PoseObjectPNP.convert_pose_object2niryo_pose_object(pick_pose)

        with self._lock:
            if pyniryo_v == "pyniryo2":
                self._robot_ctrl.pick_place.pick_from_pose(pick_pose)
            else:
                self._robot_ctrl.pick_from_pose(pick_pose)

        print("finished pick_from_pose")
        return True
        # TODO: in newest version available
        # return not self._robot_ctrl.collision_detected

    @log_start_end_cls()
    def robot_place_object(self, place_pose: "PoseObjectPNP") -> bool:
        """
        Places an already picked object at the given place_pose.

        Args:
            place_pose: Pose where to place the already picked object

        Returns:
            True, if place was successful, else False
        """
        place_pose = PoseObjectPNP.convert_pose_object2niryo_pose_object(place_pose)
        place_pose = place_pose.copy_with_offsets(z_offset=0.005)

        if self.verbose():
            print(place_pose)

        with self._lock:
            if pyniryo_v == "pyniryo2":
                self._robot_ctrl.pick_place.place_from_pose(place_pose)
            else:
                self._robot_ctrl.place_from_pose(place_pose)

        return True
        # TODO: in newest version available
        # return not self._robot_ctrl.collision_detected

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
        push_pose = PoseObjectPNP.convert_pose_object2niryo_pose_object(push_pose)

        if self.verbose():
            print(push_pose)

        with self._lock:
            if pyniryo_v == "pyniryo2":
                self._robot_ctrl.tool.close_gripper()
            else:
                self._robot_ctrl.close_gripper()

            self._move_pose(push_pose)

            if direction == "up":
                self._shift_pose(RobotAxis.X, distance)
            elif direction == "down":
                self._shift_pose(RobotAxis.X, -distance)
            elif direction == "left":
                self._shift_pose(RobotAxis.Y, distance)
            elif direction == "right":
                self._shift_pose(RobotAxis.Y, -distance)
            else:
                print("Unknown direction!", direction)

        return True
        # TODO: in newest version available
        # return not self._robot_ctrl.collision_detected

    # @log_start_end_cls()
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
            print(f"Thread {threading.current_thread().name}: {workspace_id}, {u_rel}, {v_rel}, {yaw}")

        # Use the asyncio lock for thread-safe access
        with self._lock:
            if self.verbose():
                print(f"Thread {threading.current_thread().name} acquired lock")

            try:
                x_rel = max(0.0, min(u_rel, 1.0))
                y_rel = max(0.0, min(v_rel, 1.0))

                if pyniryo_v == "pyniryo2":
                    obj_coords = self._robot_ctrl.vision.get_target_pose_from_rel(workspace_id, 0.0,
                                                                                  x_rel, y_rel, yaw)
                else:
                    obj_coords = self._robot_ctrl.get_target_pose_from_rel(workspace_id, 0.0,
                                                                           x_rel, y_rel, yaw)
            except (NiryoRobotException, UnicodeDecodeError, SyntaxError, TcpCommandException) as e:
                print(f"Thread {threading.current_thread().name} Error: {e}")
                obj_coords = PoseObject(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            finally:
                if self.verbose():
                    print(f"Thread {threading.current_thread().name} releasing lock")

        if self.verbose():
            print(f"Thread {threading.current_thread().name} exiting: {obj_coords}")
        obj_coords = PoseObjectPNP.convert_niryo_pose_object2pose_object(obj_coords)

        return obj_coords

    def get_target_pose_from_rel_timeout(self, workspace_id: str, x_rel: float, y_rel: float, yaw: float,
                                 timeout: float = 0.75) -> "PoseObjectPNP":
        print(f"Thread {threading.current_thread().name} entering: {workspace_id}, {x_rel}, {y_rel}, {yaw}")

        if not self._lock.acquire(timeout=timeout):
            print(f"Thread {threading.current_thread().name} failed to acquire lock within timeout")
            return PoseObject(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        try:
            print(f"Thread {threading.current_thread().name} acquired lock")
            future = self._executor.submit(self._robot_ctrl.get_target_pose_from_rel,
                                           workspace_id, 0.0, x_rel, y_rel, yaw)

            try:
                obj_coords = future.result(timeout=timeout)
            except FuturesTimeoutError:
                print(f"Thread {threading.current_thread().name} timeout waiting for robot response")
                # TODO: Ich kann nicht einfach die Verbindung resetten, da ja auch an anderen Orten auf den Roboter
                #  in threads zugegriffen wird.
                # self.reset_connection()
                obj_coords = PoseObject(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                future.cancel()  # Attempt to cancel the task if it is still running

        except (NiryoRobotException, UnicodeDecodeError, SyntaxError, TcpCommandException) as e:
            print(f"Thread {threading.current_thread().name} Error: {e}")
            # self.reset_connection()
            obj_coords = PoseObject(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        finally:
            print(f"Thread {threading.current_thread().name} releasing lock")
            self._lock.release()

        print(f"Thread {threading.current_thread().name} exiting: {obj_coords}")
        obj_coords = PoseObjectPNP.convert_niryo_pose_object2pose_object(obj_coords)

        return obj_coords

    @log_start_end_cls()
    def move2observation_pose(self, workspace_id: str) -> None:
        """
        The robot moves to a pose where it can observe the workspace given by workspace_id.

        Args:
            workspace_id: id of the workspace
        """
        observation_pose = self._robot.environment().get_observation_pose(workspace_id)
        observation_pose = PoseObjectPNP.convert_pose_object2niryo_pose_object(observation_pose)

        if observation_pose is not None:
            try:
                with self._lock:
                    self._move_pose(observation_pose)
            except UnicodeDecodeError as e:
                print("move2observation_pose:", e)
                print("move2observation_pose:", observation_pose)
        else:
            print("observation_pose is:", observation_pose)

        if self.verbose():
            print("move_pose finished", observation_pose)

        if self.verbose():
            # with self._lock:
            print(self.get_pose(), observation_pose)

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    def _shutdown(self) -> None:
        """
        Closes connection to NiryoRobot.
        """
        if pyniryo_v == "pyniryo2":
            # End Robot Connection
            self._robot_ctrl.end()
        else:
            self._robot_ctrl.close_connection()

    def _shift_pose(self, axis: RobotAxis, distance: float) -> None:
        """
        Shifts the gripper along the given axis for the specified distance.

        Args:
            axis (RobotAxis): axis of the robot to shift the gripper along
            distance: distance in meters to shift the gripper along
        """
        if pyniryo_v == "pyniryo2":
            self._robot_ctrl.arm.shift_pose(axis, distance / 1000)
        else:
            self._robot_ctrl.shift_pose(axis, distance / 1000)

    def _move_pose(self, pose: PoseObject) -> None:
        """
        Move gripper of robot to given pose.

        Args:
            pose (PoseObject): pose of gripper
        """
        if pyniryo_v == "pyniryo2":
            self._robot_ctrl.arm.move_pose(pose)
        else:
            self._robot_ctrl.move_pose(pose)

    @log_start_end_cls()
    def _create_robot(self) -> None:
        """
        Creates the NiryoRobot object and calibrates the robot.
        """
        with self._lock:
            self._robot_ctrl = NiryoRobot(self._robot_ip_address)
            if pyniryo_v == "pyniryo2":
                self._robot_ctrl.tool.update_tool()
                self._robot_ctrl.arm.calibrate_auto()
            else:
                self._robot_ctrl.update_tool()
                self._robot_ctrl.calibrate_auto()

    @log_start_end_cls()
    def _init_robot(self, use_simulation: bool) -> bool:
        """
        Creates the NiryoRobot object and connects to it.

        Args:
            use_simulation:

        Returns:

        """
        # Connect to Niryo Robot
        if not use_simulation:
            robot_ip_address = "192.168.0.140"
        else:
            robot_ip_address = "192.168.247.128"

        self._robot_ip_address = robot_ip_address

        self._create_robot()

        return True

    # *** PUBLIC properties ***

    # *** PRIVATE variables ***

    # ip address of the robot
    _robot_ip_address = ""
