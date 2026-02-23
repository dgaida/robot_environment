"""
Niryo frame grabber implementation for robot_environment.
"""

# class implementing a framegrabber for Niryo Ned2 robot arm
# Updated with proper logging

from ..common.logger import log_start_end_cls
from ..common.logger_config import get_package_logger

import cv2

from pyniryo import uncompress_image, undistort_image, extract_img_workspace

from ..robot.niryo_robot_controller import NiryoRobotController
from robot_workspace import PoseObjectPNP

from redis_robot_comm import RedisImageStreamer

from typing import TYPE_CHECKING, Dict, Optional

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
    def __init__(self, environment: "Environment", stream_name: str = "robot_camera", verbose: bool = False):
        """
        Initialize the Niryo framegrabber.

        Args:
            environment: Environment object this FrameGrabber is installed in.
            stream_name: Name of the Redis stream for image publishing.
            verbose: Enable verbose logging.

        Raises:
            TypeError: If the robot controller is not an instance of NiryoRobotController.
        """
        super().__init__(environment, verbose)

        self._logger = get_package_logger(__name__, verbose)

        robot = environment.get_robot_controller()

        if not isinstance(robot, NiryoRobotController):
            raise TypeError("robot must be an instance of NiryoRobotController.")

        self._robot = robot

        self._mtx, self._dist = self._robot.get_camera_intrinsics()

        self.streamer = RedisImageStreamer(stream_name=stream_name)
        self.frame_counter = 0

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    # @log_start_end_cls()
    def get_current_frame(self) -> np.ndarray:
        """
        Captures an image of the robot's workspace, ensuring proper undistortion in BGR.

        Returns:
            numpy.ndarray: Raw image captured from the robot's camera.
        """
        try:
            img_compressed = self._robot.get_img_compressed()
        except UnicodeDecodeError as e:
            self._logger.error(f"Error getting compressed image: {e}", exc_info=True)
            return self._current_frame

        img_raw = uncompress_image(img_compressed)
        img = undistort_image(img_raw, self._mtx, self._dist)

        img_work = extract_img_workspace(img, workspace_ratio=1)

        if img_work is not None:
            gripper_pose = self._robot.get_pose()

            # TODO: try to get transformation between camera and gripper
            camera_pose = gripper_pose

            self._logger.debug(f"camera_pose: {camera_pose}")

            current_frame = img_work
            myworkspace = self._environment.get_visible_workspace(camera_pose)

            if myworkspace is not None:
                myworkspace.set_img_shape(img_work.shape)
                workspace_id = myworkspace.id()
            else:
                self._logger.debug(f"No visible workspace: {myworkspace}")
                workspace_id = "unknown"
        else:
            current_frame = img
            workspace_id = "none"

        self._current_frame = current_frame

        # TODO: benötige ich die Methode? für was? falls ja, wo muss die aufgerufen werden? funktioniert hat die noch
        #  nicht. sollte in Methode get_visible_workspace() auf den Mittelpunkt des workspaces angewandt werden
        # is_visible = self.is_point_visible(np.array([0.24, 0.01, 0.001]))
        # print(is_visible)

        # cv2.imshow("Camera View", self._current_frame)
        # Break the loop if ESC key is pressed
        # cv2.waitKey(0)

        self.publish_workspace_image(self._current_frame, workspace_id)

        return self._current_frame

    def publish_workspace_image(
        self, image: np.ndarray, workspace_id: str, robot_pose: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Publish workspace image with robot context via Redis.

        Args:
            image: The image to publish.
            workspace_id: ID of the workspace shown in the image.
            robot_pose: Optional robot pose metadata.

        Returns:
            str: The Redis stream ID of the published image.
        """
        metadata = {
            "workspace_id": workspace_id,
            "frame_id": self.frame_counter,
            "robot_pose": robot_pose or {},
            "image_source": "robot_mounted_camera",
        }

        stream_id = self.streamer.publish_image(image, metadata=metadata, compress_jpeg=True, quality=85)

        self.frame_counter += 1
        return stream_id

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

    def camera_matrix(self) -> np.ndarray:
        """
        Returns the camera intrinsic matrix.

        Returns:
            np.ndarray: 3x3 intrinsic matrix.
        """
        return self._mtx

    def camera_dist_coeff(self) -> np.ndarray:
        """
        Returns the camera distortion coefficients.

        Returns:
            np.ndarray: Distortion coefficients.
        """
        return self._dist

    # *** PRIVATE variables ***

    # NiryoRobotController object
    _robot = None

    # camera intrinsic transformation matrix
    _mtx = None

    # camera distortion coefficients
    _dist = None
    _logger = None
