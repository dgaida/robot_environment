"""
Frame grabber base class for robot_environment.
"""

# abstract class FrameGrabber for the robot_environment package
# should be Final
# Documentation and type definitions are almost final (chatgpt might be able to improve it).

from abc import ABC, abstractmethod
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..environment import Environment


class FrameGrabber(ABC):
    """
    An abstract class that provides the abstract method get_current_frame() to be implemented by classes
    inheriting from this one. The FrameGrabber grabs frames from the camera and provides them.
    """

    # *** CONSTRUCTORS ***
    def __init__(self, environment: "Environment", verbose: bool = False):
        """
        Initialize the FrameGrabber.

        Args:
            environment: Environment object this FrameGrabber is installed in.
            verbose: Enable verbose logging.
        """
        self._current_frame = None
        self._environment = environment
        self._verbose = verbose

    # *** PUBLIC GET methods ***

    def get_current_frame_shape(self) -> tuple[int, ...]:
        """
        Return the shape of the current frame.

        Returns:
            tuple[int, ...]: Shape of the frame.
        """
        return self._current_frame.shape

    def get_current_frame_width_height(self) -> tuple[int, int]:
        """
        Returns width and height of current frame in pixels.

        Returns:
            width and height of current frame in pixels.
        """
        return self._current_frame.shape[0], self._current_frame.shape[1]

    # *** PUBLIC methods ***

    # *** PUBLIC STATIC/CLASS GET methods ***

    @abstractmethod
    def get_current_frame(self) -> np.ndarray:
        """
        Captures an image of the robot's workspace, ensuring proper undistortion in RGB.

        Returns:
            numpy.ndarray: Raw image captured from the robot's camera.
        """
        return self._current_frame

    # *** PRIVATE methods ***

    # *** PRIVATE STATIC/CLASS methods ***

    # *** PUBLIC properties ***

    def current_frame(self) -> np.ndarray:
        """
        Returns current frame.

        Returns:
            numpy.ndarray: Current frame.
        """
        return self._current_frame

    def environment(self) -> "Environment":
        """
        Returns the environment object.

        Returns:
            Environment: The environment instance.
        """
        return self._environment

    def verbose(self) -> bool:
        """
        Returns the verbosity status.

        Returns:
            bool: True if verbose is on, else False.
        """
        return self._verbose

    # *** PRIVATE variables ***

    # current frame.
    _current_frame = None

    _environment = None

    _verbose = False
