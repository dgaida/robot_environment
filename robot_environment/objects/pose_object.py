# source: https://archive-docs.niryo.com/dev/pyniryo2/v1.0.0/en/_modules/pyniryo2/objects.html#PoseObject
# should be final
# Documentation and type definitions are final (maybe chatgpt can improve it).

from ..common.logger import log_start_end_cls, pyniryo_v

import numpy as np
import math

if pyniryo_v == "pyniryo2":
    from pyniryo2 import PoseObject
else:
    from pyniryo.api.objects import PoseObject

from typing import TYPE_CHECKING, List

# if TYPE_CHECKING:
    # from pyniryo.api.objects import PoseObject
    # from pyniryo2 import PoseObject


class PoseObjectPNP:
    """
    Pose object which stores x, y, z, roll, pitch & yaw parameters

    :ivar x: X (meter)
    :vartype x: float
    :ivar y: Y (meter)
    :vartype y: float
    :ivar z: Z (meter)
    :vartype z: float
    :ivar roll: Roll (radian)
    :vartype roll: float
    :ivar pitch: Pitch (radian)
    :vartype pitch: float
    :ivar yaw: Yaw (radian)
    :vartype yaw: float

    """

    # *** CONSTRUCTORS ***
    def __init__(self, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        # X (meter)
        self.x = float(x)
        # Y (meter)
        self.y = float(y)
        # Z (meter)
        self.z = float(z)
        # Roll (radian)
        self.roll = float(roll)
        # Pitch (radian)
        self.pitch = float(pitch)
        # Yaw (radian)
        self.yaw = float(yaw)

    def __str__(self):
        position = "x = {:.4f}, y = {:.4f}, z = {:.4f}".format(self.x, self.y, self.z)
        orientation = "roll = {:.3f}, pitch = {:.3f}, yaw = {:.3f}".format(self.roll, self.pitch, self.yaw)
        return position + "\n" + orientation

    def __repr__(self):
        return self.__str__()

    def __add__(self, other: "PoseObjectPNP"):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        roll = self.roll + other.roll
        pitch = self.pitch + other.pitch
        yaw = self.yaw + other.yaw
        return PoseObjectPNP(x, y, z, roll, pitch, yaw)

    def __sub__(self, other: "PoseObjectPNP"):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        roll = self.roll - other.roll
        pitch = self.pitch - other.pitch
        yaw = self.yaw - other.yaw
        return PoseObjectPNP(x, y, z, roll, pitch, yaw)

    def __eq__(self, other: "PoseObjectPNP") -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z \
            and self.roll == other.roll and self.pitch == other.pitch and self.yaw == other.yaw

    @log_start_end_cls()
    def approx_eq(self, other: "PoseObjectPNP", eps_position: float = 0.1, eps_orientation: float = 0.1) -> bool:
        """
        Determines if two poses are approximately the same, accounting for angle periodicity.

        Args:
            other: other pose object to compare with
            eps_position (float): Tolerance for position differences (in meters).
            eps_orientation (float): Tolerance for orientation differences (in radians).

        Returns:
            bool: True if poses are approximately the same, False otherwise.
        """
        # Calculate position differences
        delta_position = math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

        # Calculate orientation differences
        delta_roll = abs(PoseObjectPNP._angular_difference(self.roll, other.roll))
        delta_pitch = abs(PoseObjectPNP._angular_difference(self.pitch, other.pitch))
        delta_yaw = abs(PoseObjectPNP._angular_difference(self.yaw, other.yaw))

        # Check if both position and orientation differences are within tolerances
        return (
                delta_position < eps_position and
                delta_roll < eps_orientation and
                delta_pitch < eps_orientation and
                delta_yaw < eps_orientation
        )

    def approx_eq_xyz(self, other: "PoseObjectPNP", eps: float = 0.1) -> bool:
        """
        Compares the (x, y, z) coordinates of this pose object with another pose object
        to determine if they are approximately equal within a given tolerance.

        Args:
            other (PoseObjectPNP): The pose object to compare with.
            eps (float): The allowable deviation for equality in each coordinate (default: 0.1).

        Returns:
            bool: True if the difference in x, y, and z coordinates between the two poses
            is less than the specified tolerance (eps), otherwise False.
        """
        return (np.abs(self.x - other.x) < eps and
                np.abs(self.y - other.y) < eps and
                np.abs(self.z - other.z) < eps)

    def copy_with_offsets(self, x_offset=0., y_offset=0., z_offset=0.,
                          roll_offset=0., pitch_offset=0., yaw_offset=0.) -> "PoseObjectPNP":
        """
        Creates a new pose object by copying the current pose and applying specified offsets
        to its position (x, y, z) and orientation (roll, pitch, yaw).

        Args:
            x_offset (float): Offset to apply to the x-coordinate (default: 0.0).
            y_offset (float): Offset to apply to the y-coordinate (default: 0.0).
            z_offset (float): Offset to apply to the z-coordinate (default: 0.0).
            roll_offset (float): Offset to apply to the roll orientation (default: 0.0).
            pitch_offset (float): Offset to apply to the pitch orientation (default: 0.0).
            yaw_offset (float): Offset to apply to the yaw orientation (default: 0.0).

        Returns:
            PoseObjectPNP: A new pose object with the offsets applied.
        """
        return PoseObjectPNP(self.x + x_offset, self.y + y_offset, self.z + z_offset,
                             self.roll + roll_offset, self.pitch + pitch_offset, self.yaw + yaw_offset)

    def to_list(self) -> list[float, float, float, float, float, float]:
        """
        Return a list [x, y, z, roll, pitch, yaw] corresponding to the pose's parameters

        :rtype: list[float]
        """
        list_pos = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]
        return list(map(float, list_pos))

    def to_transformation_matrix(self) -> np.ndarray:
        """
        Converts the pose into a 4x4 transformation matrix.

        The transformation matrix represents the pose of an object in a
        3D coordinate system. It combines translation and rotation into
        a single homogeneous transformation.

        Returns:
            np.ndarray: A 4x4 transformation matrix of type float, where:
                - The upper-left 3x3 submatrix represents the rotation.
                - The last column represents the translation.
                - The bottom row is [0, 0, 0, 1] for homogeneity.
        """
        # Compute the rotation matrix from roll, pitch, yaw
        rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.roll), -np.sin(self.roll)],
            [0, np.sin(self.roll), np.cos(self.roll)]
        ])

        ry = np.array([
            [np.cos(self.pitch), 0, np.sin(self.pitch)],
            [0, 1, 0],
            [-np.sin(self.pitch), 0, np.cos(self.pitch)]
        ])

        rz = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw), 0],
            [np.sin(self.yaw), np.cos(self.yaw), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix (Rz * Ry * Rx)
        rotation_matrix = rz @ ry @ rx

        # Translation vector
        translation_vector = np.array([self.x, self.y, self.z])

        # Construct the 4x4 transformation matrix
        transformation_matrix = np.eye(4)  # Start with an identity matrix
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector

        return transformation_matrix

    @property
    def quaternion(self) -> list[float, float, float, float]:
        """
        Return the quaternion in a list [qx, qy, qz, qw]

        :return: quaternion [qx, qy, qz, qw]
        :rtype: list[float, float, float, float]
        """
        return self.euler_to_quaternion(self.roll, self.pitch, self.yaw)

    @property
    def quaternion_pose(self) -> list[float, float, float, float, float, float, float]:
        """
        Return the position and the quaternion in a list [x, y, z, qx, qy, qz, qw]

        :return: position [x, y, z] + quaternion [qx, qy, qz, qw]
        :rtype: list[float, float, float, float, float, float, float]

        """
        return [self.x, self.y, self.z] + list(self.euler_to_quaternion(self.roll, self.pitch, self.yaw))

    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> list[float, float, float, float]:
        """
        Convert euler angles to quaternion

        :param roll: roll in radians
        :type roll: float
        :param pitch: pitch in radians
        :type pitch: float
        :param yaw: yaw in radians
        :type yaw: float
        :return: quaternion in a list [qx, qy, qz, qw]
        :rtype: list[float, float, float, float]
        """
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)

        return [qx, qy, qz, qw]

    @staticmethod
    def quaternion_to_euler_angle(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
        """
        Convert euler angles to quaternion

        :param qx:
        :type qx: float
        :param qy:
        :type qy: float
        :param qz:
        :type qz: float
        :param qw:
        :type qw: float
        :return: euler angles in a list [roll, pitch, yaw]
        :rtype: list[float, float, float]
        """
        ysqr = qy * qy

        t0 = +2.0 * (qw * qx + qy * qz)
        t1 = +1.0 - 2.0 * (qx * qx + ysqr)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (qw * qy - qz * qx)

        t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
        pitch = np.arcsin(t2)

        t3 = +2.0 * (qw * qz + qx * qy)
        t4 = +1.0 - 2.0 * (ysqr + qz * qz)
        yaw = np.arctan2(t3, t4)

        return roll, pitch, yaw

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    # *** PUBLIC STATIC/CLASS GET methods ***

    @staticmethod
    def convert_niryo_pose_object2pose_object(pose_object: "PoseObject") -> "PoseObjectPNP":
        """
        Converts a PoseObject from Niryo class to a pnp_robot_genai.pose_object.PoseObject object. The latter is
        a copy of the first.

        Args:
            pose_object: PoseObject

        Returns:
            pnp_robot_genai.pose_object.PoseObject
        """
        print("pose_object", pose_object)
        pose = PoseObjectPNP(pose_object.x, pose_object.y, pose_object.z,
                             pose_object.roll, pose_object.pitch, pose_object.yaw)
        print("pose", pose)
        return pose

    @staticmethod
    def convert_pose_object2niryo_pose_object(pose_object: "PoseObject") -> "PoseObjectPNP":
        """
        Convert a pnp_robot_genai.pose_object.PoseObject to a PoseObject from Niryo Robot

        Args:
            pose_object: pnp_robot_genai.pose_object.PoseObject

        Returns:
            PoseObject: Pose object that the Niryo robot defines
        """
        pose = PoseObject(pose_object.x, pose_object.y, pose_object.z,
                          pose_object.roll, pose_object.pitch, pose_object.yaw)
        return pose

    # *** PRIVATE methods ***

    # *** PRIVATE STATIC/CLASS methods ***

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """
        Normalize an angle to the range [-π, π].

        Args:
            angle: some angle in radians
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _angular_difference(angle1: float, angle2: float) -> float:
        """
        Calculate the smallest difference between two angles (angle1 - angle2), accounting for periodicity.

        Args:
            angle1: angle 1 in radians
            angle2: angle 2 in radians
        """
        return PoseObjectPNP._normalize_angle(angle1 - angle2)

    # *** PUBLIC properties ***

    def xy_coordinate(self) -> List[float]:
        return [self.x, self.y]

    # *** PRIVATE variables ***
