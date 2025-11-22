# abstract class Workspace for the pnp_robot_genai package
# Not yet final
# TODO in is_visible
# Documentation and type definitions are almost final (chatgpt might be able to improve it).

from ..common.logger import log_start_end_cls

from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
from ..objects.pose_object import PoseObjectPNP
from ..objects.object import Object

if TYPE_CHECKING:
    from ..environment import Environment
    from ..objects.pose_object import PoseObjectPNP


class Workspace(ABC):
    """
    A workspace where the pick-and-place robot can pick or place objects from. this is an abstract class.
    For a new robot derive a workspace for the robot from this class.

    Important methods:
    - is_visible: Checks whether this workspace is visible with the current pose of the camera
    - transform_camera2world_coords: transforms camera to world coordinates
    """

    # *** CONSTRUCTORS ***
    @log_start_end_cls()
    def __init__(self, workspace_id: str, environment: "Environment", verbose: bool = False):
        """
        Initializes the workspace completely. the derivatives do not have to do anything else in their constructors
        except calling this constructor.

        Args:
            workspace_id: ID of the workspace
            environment: Environment this workspace belongs to
            verbose:
        """
        self._id = workspace_id
        self._environment = environment
        self._verbose = verbose

        self._set_observation_pose()

        self._set_4corners_of_workspace()

        self._calc_width_height()

        self._calc_center_of_workspace()

    def __str__(self):
        size = "width = {:.2f}, height = {:.2f}".format(self._width_m, self._height_m)
        return "Workspace " + self.id() + "\n" + size + "\n" + str(self._xy_center_wc)

    def __repr__(self):
        return self.__str__()

    # *** PUBLIC GET methods ***

    @log_start_end_cls()
    def is_visible(self, camera_pose: "PoseObjectPNP") -> bool:
        """

        Args:
            camera_pose:

        Returns:

        """
        # TODO: I should check if from current camera pose the center poseObject (self._xy_center_wc) of workspace
        #  is visible. in niryo_framegrabber started this

        if self.verbose():
            print("is_visible:", camera_pose, self._observation_pose)

        # TODO: hier sollte eigentlich approx_eq genutzt werden, aber die Winkel, die vom Roboter zurück gegeben werden
        #  stimmen nicht
        # temporary solution
        # TODO: for widowx this will not work, as there camera and observation pose are not the same
        if camera_pose.approx_eq_xyz(self._observation_pose):
            return True
        else:
            return False

    # *** PUBLIC methods ***

    def set_img_shape(self, img_shape: tuple[int, int, int]) -> None:
        self._img_shape = img_shape

    @abstractmethod
    def transform_camera2world_coords(
        self, workspace_id: str, u_rel: float, v_rel: float, yaw: float = 0.0
    ) -> "PoseObjectPNP":
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
            PoseObjectPNP: Pose of the point in world coordinates of the robot.
        """
        pass

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    @abstractmethod
    def _set_4corners_of_workspace(self) -> None:
        """
        sets the 4 corners of the workspace, being _xy_ul_wc, ...
        """
        pass

    @abstractmethod
    def _set_observation_pose(self) -> None:
        """
        Set the variable _observation_pose for the given workspace. An observation pose is a pose of the gripper
        where the gripper hovers over the workspace. For the niryo robot this is a gripper pose in which the
        gripper mounted camera can observe the complete workspace.
        """
        pass

    @log_start_end_cls()
    def _calc_width_height(self) -> None:
        """
        Calculates width and height of the workspace.

        """
        self._width_m, self._height_m = Object.calc_width_height(self._xy_ul_wc, self._xy_lr_wc)

    @log_start_end_cls()
    def _calc_center_of_workspace(self) -> None:
        """
        Calculate the variable _xy_center_wc that is a pose object in the middle of the workspace. we could use
        it to figure out whether the workspace can be seen in the camera. As the workspace of the niryo robot has
        4 markers and niryo provides the method "extract_img_workspace" we know that a workspace is completely
        visible. then with the center of the workspace we just have to check which workspace is visible. for other
        robots we might have to check all 4 corners of the workspace to figure out whether the workspace is
        completely visible. However, often it is not important to know whether the complete workspace is visible but
        if just the center of it is visible. Also see the TODOs in is_visible().
        """
        dx = self._xy_ll_wc.x - self._xy_ul_wc.x
        dy = self._xy_ll_wc.y - self._xy_lr_wc.y

        self._xy_center_wc = self._xy_lr_wc.copy_with_offsets(-dx / 2.0, dy / 2.0)

        if self.verbose():
            print("_calc_center_of_workspace:", self._xy_center_wc, self._xy_ll_wc.x, self._xy_ur_wc.x)

    # *** PRIVATE STATIC/CLASS methods ***

    # *** PUBLIC properties ***

    def id(self) -> str:
        return self._id

    def environment(self) -> "Environment":
        return self._environment

    def xy_ul_wc(self) -> "PoseObjectPNP":
        """

        Returns:
            (x, y, z) coordinate of upper left corner of workspace in world coordinates in meter
        """
        return self._xy_ul_wc

    def xy_ll_wc(self) -> "PoseObjectPNP":
        """

        Returns:
            (x, y, z) coordinate of lower left corner of workspace in world coordinates in meter
        """
        return self._xy_ll_wc

    def xy_ur_wc(self) -> "PoseObjectPNP":
        """

        Returns:
            (x, y, z) coordinate of upper right corner of workspace in world coordinates in meter
        """
        return self._xy_ur_wc

    def xy_lr_wc(self) -> "PoseObjectPNP":
        """

        Returns:
            (x, y, z) coordinate of lower right corner of workspace in world coordinates in meter
        """
        return self._xy_lr_wc

    def xy_center_wc(self) -> "PoseObjectPNP":
        """

        Returns:
            center pose of the workspace
        """
        return self._xy_center_wc

    def width_m(self) -> float:
        """

        Returns:
            width of workspace in meter
        """
        return self._width_m

    def height_m(self) -> float:
        """

        Returns:
            height of workspace in meter
        """
        return self._height_m

    def img_shape(self) -> tuple[int, int, int]:
        """

        Returns:
            shape of image of the workspace. width, height, num of channels (3, because of RGB image)
        """
        return self._img_shape

    def observation_pose(self) -> "PoseObjectPNP":
        """

        Returns:
            pose of robot arm when it hovers over the workspace.
        """
        return self._observation_pose

    def verbose(self) -> bool:
        """

        Returns: True, if verbose is on, else False

        """
        return self._verbose

    # *** PRIVATE variables ***

    # id of workspace
    _id = ""

    # environment this workspace belongs to
    _environment = None

    # (x, y, z) coordinate of upper left corner of workspace in world coordinates in meter
    _xy_ul_wc = None
    # (x, y, z) coordinate of lower left corner of workspace in world coordinates in meter
    _xy_ll_wc = None
    # (x, y, z) coordinate of upper right corner of workspace in world coordinates in meter
    _xy_ur_wc = None
    # (x, y, z) coordinate of lower right corner of workspace in world coordinates in meter
    _xy_lr_wc = None

    # center pose of the workspace
    _xy_center_wc = None

    # width and height of workspace in meter
    _width_m = 0.0
    _height_m = 0.0

    # size of robot's workspace in the camera image in pixels
    _img_shape = None

    # pose of robot arm when it hovers over the workspace.
    _observation_pose = None

    _verbose = False
