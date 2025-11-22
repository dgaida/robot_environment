"""
Workspaces package for the robot_environment system.

This package provides workspace management functionality for pick-and-place robots,
including abstract workspace definitions and concrete implementations for specific robots.

Classes:
    Workspace: Abstract base class for robot workspaces
    Workspaces: Collection class for managing multiple workspaces
    NiryoWorkspace: Concrete workspace implementation for Niryo Ned2 robot
    NiryoWorkspaces: Collection of NiryoWorkspace instances

Example:
    from robot_environment.workspaces import NiryoWorkspaces, NiryoWorkspace

    # Create a collection of workspaces for Niryo robot
    workspaces = NiryoWorkspaces(environment, verbose=True)

    # Get a specific workspace by ID
    workspace = workspaces.get_workspace_by_id("niryo_ws")
"""

from .workspace import Workspace
from .workspaces import Workspaces
from .niryo_workspace import NiryoWorkspace
from .niryo_workspaces import NiryoWorkspaces

__all__ = ["Workspace", "Workspaces", "NiryoWorkspace", "NiryoWorkspaces"]

__version__ = "0.1.0"
__author__ = "Daniel Gaida"
