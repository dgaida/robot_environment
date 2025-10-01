"""
Object detection and representation module.

Provides classes for representing detected objects with their physical
properties, positions, and segmentation information.
"""

from .object import Object
from .objects import Objects
from .object_api import ObjectAPI
from .pose_object import PoseObjectPNP

__all__ = [
    "Object",
    "Objects",
    "ObjectAPI",
    "PoseObjectPNP",
]
