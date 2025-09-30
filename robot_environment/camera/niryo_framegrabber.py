# class implementing a framegrabber for Niryo Ned2 robot arm
# TODO: is_point_visible prüfen
# Documentation and type definitions are almost final (chatgpt might be able to improve it).

from ..common.logger import log_start_end_cls, pyniryo_v

import cv2
# from pyniryo2 import NiryoRobot
from pyniryo import uncompress_image, undistort_image, extract_img_workspace

from ..robot.niryo_robot_controller import NiryoRobotController
from ..objects.pose_object import PoseObjectPNP

from redis_robot_comm import RedisImageStreamer

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..environment import Environment

from ..camera.framegrabber import FrameGrabber
import numpy as np


class NiryoFrameGrabber(FrameGrabber):
    """
    A class implementing a framegrabber for Niryo Ned2 robot arm
    """

    # *** CONSTRUCTORS ***
    @log_start_end_cls()
    def __init__(self, environment: "Environment", stream_name='robot_camera', verbose: bool = False):
        """

        Args:
            environment: Environment object this FrameGrabber is installed in
            verbose:

        Returns:
            object:

        Raises:
            TypeError: robot must be an instance of NiryoRobotController.
        """
        super().__init__(environment, verbose)

        robot = environment.get_robot_controller()

        if not isinstance(robot, NiryoRobotController):
            raise TypeError("robot must be an instance of NiryoRobotController.")

        # get NiryoRobot from NiryoRobotController
        self._robot = robot.robot_ctrl()

        # all calls of methods of the _robot (NiryoRobot) object are locked, because they are not safe thread
        with robot.lock():
            if pyniryo_v == "pyniryo2":
                self._mtx, self._dist = self._robot.vision.get_camera_intrinsics()
            else:
                self._mtx, self._dist = self._robot.get_camera_intrinsics()

        self.streamer = RedisImageStreamer(stream_name=stream_name)
        self.frame_counter = 0

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    # @log_start_end_cls()
    def get_current_frame(self) -> np.ndarray:
        """
        Captures an image of the robot's workspace, ensuring proper undistortion in RGB.

        Returns:
            numpy.ndarray: Raw image captured from the robot's camera.
        """
        try:
            with self.environment().get_robot_controller().lock():
                if pyniryo_v == "pyniryo2":
                    img_compressed = self._robot.vision.get_img_compressed()
                else:
                    img_compressed = self._robot.get_img_compressed()
        except UnicodeDecodeError as e:
            print("get_current_frame:", e)
            return self._current_frame

        img_raw = uncompress_image(img_compressed)
        img = undistort_image(img_raw, self._mtx, self._dist)

        img_work = extract_img_workspace(img, workspace_ratio=1)

        if img_work is not None:
            gripper_pose = self.environment().get_robot_pose()

            # TODO: try to get transformation between camera and gripper
            camera_pose = gripper_pose

            if self.verbose():
                print("camera_pose:", camera_pose)

            current_frame = img_work
            myworkspace = self._environment.get_visible_workspace(camera_pose)

            if myworkspace is not None:
                myworkspace.set_img_shape(img_work.shape)
            else:
                print("DEBUG", myworkspace)
        else:
            current_frame = img

        self._current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        # TODO: benötige ich die Methode? für was? falls ja, wo muss die aufgerufen werden? funktioniert hat die noch
        #  nicht. sollte in Methode get_visible_workspace() auf den Mittelpunkt des workspaces angewandt werden
        # is_visible = self.is_point_visible(np.array([0.24, 0.01, 0.001]))
        # print(is_visible)

        # cv2.imshow("Camera View", self._current_frame)
        # Break the loop if ESC key is pressed
        # cv2.waitKey(0)

        self.publish_workspace_image(self._current_frame, "id")

        return self._current_frame

    def publish_workspace_image(self, image: np.ndarray, workspace_id: str,
                                robot_pose: Dict[str, float] = None):
        """
        Publish workspace image with robot context
        Image size can vary based on workspace and robot position
        """
        metadata = {
            'workspace_id': workspace_id,
            'frame_id': self.frame_counter,
            'robot_pose': robot_pose or {},
            'image_source': 'robot_mounted_camera'
        }

        # Redis automatically handles the variable image size
        stream_id = self.streamer.publish_image(
            image,
            metadata=metadata,
            compress_jpeg=True,
            quality=85  # Good balance of quality/size for robotics
        )

        self.frame_counter += 1
        return stream_id

    # def run_adaptive_capture(self, get_current_image_func, target_fps: float = 5.0):
    #     """
    #     Run image capture that adapts to changing image sizes
    #
    #     Args:
    #         get_current_image_func: Function that returns (image, workspace_id, robot_pose)
    #     """
    #     frame_interval = 1.0 / target_fps
    #
    #     print(f"Starting adaptive image capture at {target_fps} FPS")
    #
    #     try:
    #         while True:
    #             start_time = time.time()
    #
    #             # Get current image (size may vary)
    #             try:
    #                 image, workspace_id, robot_pose = get_current_image_func()
    #
    #                 if image is not None:
    #                     self.publish_workspace_image(image, workspace_id, robot_pose)
    #
    #                     if self.frame_counter % 25 == 0:
    #                         height, width = image.shape[:2]
    #                         print(f"Frame {self.frame_counter}: {width}x{height}, workspace: {workspace_id}")
    #
    #             except Exception as e:
    #                 print(f"Error capturing frame: {e}")
    #
    #             # Maintain target FPS
    #             elapsed = time.time() - start_time
    #             sleep_time = frame_interval - elapsed
    #             if sleep_time > 0:
    #                 time.sleep(sleep_time)
    #
    #     except KeyboardInterrupt:
    #         print("Vision publisher stopped")

    def is_point_visible(self, world_point: np.array, camera_to_gripper_transform=np.eye(4)) -> bool:
        """
        Determines whether a given world point is visible in the camera's field of view.

        Args:
            world_point (np.array): A 3D point in world coordinates as a NumPy array of shape (3,).
            camera_to_gripper_transform (np.array): A 4x4 transformation matrix that defines
                the relationship between the camera and the gripper frame. Defaults to the identity matrix.

        Returns:
            bool: True if the point is visible in the camera's view, False otherwise.

        Explanation:
            1. Transforms the `world_point` from world coordinates to the gripper frame using the
               gripper's pose transformation matrix.
            2. Transforms the point to the camera frame using the provided camera-to-gripper transformation.
            3. Projects the point onto the image plane using the camera's intrinsic matrix.
            4. Checks if the projected point lies within the camera's image bounds (640x480 pixels).
            5. Optionally undistorts the projected point to account for lens distortion.

        Notes:
            - This method assumes the camera intrinsic matrix (`self._mtx`) and distortion coefficients (`self._dist`)
              are precomputed and valid for a resolution of 640x480 pixels.
            - Points behind the camera (with z <= 0 in the camera frame) are not visible.
        """
        # print(self._mtx, self._dist)
        # Unpack camera intrinsics
        K = np.array(self._mtx)
        distortion = np.array(self._dist)

        # img_width, img_height = super().get_current_frame_width_height()
        # ich muss hier mit 640 x 480 pixel arbeiten, da camera intrinsics nur dafür gilt: uo, vo
        img_width, img_height = 640, 480

        gripper_pose = self._robot.get_pose()
        gripper_pose = PoseObjectPNP.convert_niryo_pose_object2pose_object(gripper_pose)

        # Transform the world point to the gripper frame
        world_to_gripper_transform = gripper_pose.to_transformation_matrix()
        # Convert `world_point` to homogeneous coordinates by appending a 1
        world_point_homogeneous = np.append(world_point, 1)

        # print(world_to_gripper_transform)
        # print(gripper_pose)

        # Transform `world_point` to the gripper frame
        # gripper_frame_point = np.linalg.inv(world_to_gripper_transform) @ world_point_homogeneous
        gripper_frame_point = np.dot(world_to_gripper_transform, world_point_homogeneous)

        # Transform the point to the camera frame
        camera_point = np.dot(camera_to_gripper_transform, gripper_frame_point)

        # print(camera_point)

        # Check if the point is in front of the camera
        if camera_point[2] <= 0:
            return False

        # Project the point to the image plane
        pixel_coords = np.dot(K, camera_point[:3] / camera_point[2])

        # Normalize homogeneous coordinates
        u, v = pixel_coords[:2] / pixel_coords[2]

        # print(u, v)

        # Check if the point is within the image bounds
        if 0 <= u < img_width and 0 <= v < img_height:
            # Optional: Account for lens distortion
            undistorted_points = cv2.undistortPoints(np.array([[u, v]], dtype=np.float32), K, distortion)
            u, v = undistorted_points[0][0]
            return 0 <= u < img_width and 0 <= v < img_height

        return False

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    # *** PRIVATE STATIC/CLASS methods ***

    # *** PUBLIC properties ***

    def camera_matrix(self):
        return self._mtx

    def camera_dist_coeff(self):
        return self._dist

    # *** PRIVATE variables ***

    # NiryoRobot object
    _robot = None

    # camera intrinsic transformation matrix
    _mtx = None

    # camera distortion coefficients
    _dist = None
