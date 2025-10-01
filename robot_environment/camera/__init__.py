"""
Camera module for frame grabbing and image acquisition.

Provides abstract and concrete implementations for different robot cameras.
"""

from .framegrabber import FrameGrabber
from .niryo_framegrabber import NiryoFrameGrabber
from .widowx_framegrabber import WidowXFrameGrabber

__all__ = [
    "FrameGrabber",
    "NiryoFrameGrabber",
    "WidowXFrameGrabber",
]
