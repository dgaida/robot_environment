# class defining a list of NiryoWorkspace class
# final, apart from that more workspaces can be added
# Documentation and type definitions are almost final (chatgpt might be able to improve it).

from .workspaces import Workspaces
from .niryo_workspace import NiryoWorkspace

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..environment import Environment


class NiryoWorkspaces(Workspaces):
    """
    class defining a list of NiryoWorkspace class
    """

    # *** CONSTRUCTORS ***
    def __init__(self, environment: "Environment", verbose: bool = False):
        """
        Adds list of NiryoWorkspace to the list of Workspaces

        Args:
            environment:
            verbose:
        """
        super().__init__(environment, verbose)

        if not environment.use_simulation():
            # Define Workspace
            workspace_id = "niryo_ws2"  # "niryo_ws"
        else:
            workspace_id = "gazebo_1"

        # TODO: add more workspaces
        # important to do this after the _robot object was created
        super().append_workspace(NiryoWorkspace(workspace_id, environment, verbose))

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    # *** PRIVATE variables ***
