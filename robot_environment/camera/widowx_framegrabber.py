"""
WidowX frame grabber implementation for robot_environment.
"""
# framegrabber for WidowX robot arm - here Intel RealSense camera as third person camera
# TODO: has to be implemented

from .framegrabber import FrameGrabber
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..environment import Environment


class WidowXFrameGrabber(FrameGrabber):
    """
    A class implementing a framegrabber for WidowX robot arm - here Intel RealSense camera as third person camera
    """

    # *** CONSTRUCTORS ***
    def __init__(self, environment: "Environment", verbose: bool = False):
        """
        Initialize the WidowX framegrabber.

        Args:
            environment: Environment object this FrameGrabber is installed in.
            verbose: Enable verbose logging.
        """
        super().__init__(environment, verbose)

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    def get_current_frame(self) -> np.ndarray:
        """
        Captures an image of the robot's workspace, ensuring proper undistortion in BGR.

        Args:

        Returns:
            numpy.ndarray: Raw image captured from the robot's camera.
        """
        # TODO: capture a frame from the camera and set _current_frame

        return self._current_frame

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    # *** PRIVATE STATIC/CLASS methods ***

    # *** PUBLIC properties ***

    # *** PRIVATE variables ***
