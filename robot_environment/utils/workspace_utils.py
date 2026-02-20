"""
Workspace utility functions for free space calculation and coordinate transformations.
"""

from __future__ import annotations
import numpy as np
import cv2
import logging
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from robot_workspace import Workspace, Objects


def calculate_largest_free_space(
    workspace: Workspace,
    detected_objects: Objects,
    grid_resolution: int = 100,
    visualize: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float, float]:
    """
    Determines the largest free space in the workspace in square metres and its center coordinate in metres.

    Args:
        workspace: The workspace object to analyze.
        detected_objects: Collection of objects detected in the workspace.
        grid_resolution: Resolution of the workspace grid (default: 100x100).
        visualize: If True, displays the grid visualization (requires GUI).
        logger: Optional logger for debug information.

    Returns:
        tuple: (largest_area_m2, center_x, center_y)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    workspace_top_left = workspace.xy_ul_wc()
    workspace_bottom_right = workspace.xy_lr_wc()

    x_max, y_max = workspace_top_left.x, workspace_top_left.y
    x_min, y_min = workspace_bottom_right.x, workspace_bottom_right.y

    logger.debug(f"Workspace bounds: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")

    workspace_width = abs(y_max - y_min)
    workspace_height = abs(x_max - x_min)

    # Create a grid to represent the workspace
    grid = np.zeros((grid_resolution, grid_resolution), dtype=int)

    # Map world coordinates to grid indices
    def to_grid_coords(x: float, y: float) -> Tuple[int, int]:
        """
        Map world coordinates to grid indices.

        Args:
            x: World x-coordinate.
            y: World y-coordinate.

        Returns:
            Tuple[int, int]: (u, v) grid indices.
        """
        v = int((x_max - x) / workspace_height * grid_resolution)
        u = int((y_max - y) / workspace_width * grid_resolution)
        # Clip to ensure indices are within grid bounds
        v = max(0, min(v, grid_resolution - 1))
        u = max(0, min(u, grid_resolution - 1))
        return u, v

    # Map grid indices back to world coordinates
    def to_world_coords(u: int, v: int) -> Tuple[float, float]:
        """
        Map grid indices back to world coordinates.

        Args:
            u: Grid u-index.
            v: Grid v-index.

        Returns:
            Tuple[float, float]: (x, y) world coordinates.
        """
        x = x_max - (v + 0.5) * (workspace_height / grid_resolution)
        y = y_max - (u + 0.5) * (workspace_width / grid_resolution)
        return x, y

    # Mark the grid cells occupied by objects
    for obj in detected_objects:
        x_start = obj.x_com() - obj.height_m() / 2
        x_end = obj.x_com() + obj.height_m() / 2
        y_start = obj.y_com() - obj.width_m() / 2
        y_end = obj.y_com() + obj.width_m() / 2

        # Convert object bounds to grid indices
        u_end, v_end = to_grid_coords(x_start, y_start)
        u_start, v_start = to_grid_coords(x_end, y_end)

        logger.debug(f"Object bounds: x=[{x_start}, {x_end}], y=[{y_start}, {y_end}]")
        logger.debug(f"Grid coords: u=[{u_start}, {u_end}], v=[{v_start}, {v_end}]")

        # Mark grid cells as occupied (ensuring correct order for slicing)
        v_min_idx = min(v_start, v_end)
        v_max_idx = max(v_start, v_end)
        u_min_idx = min(u_start, u_end)
        u_max_idx = max(u_start, u_end)
        grid[v_min_idx : v_max_idx + 1, u_min_idx : u_max_idx + 1] = 1

    # Find the largest rectangle of zeros in the grid
    largest_area_cells, (v_start_rect, u_start_rect), (v_end_rect, u_end_rect) = _max_rectangle_area(grid)
    largest_area_m2 = (largest_area_cells / (grid_resolution**2)) * (workspace_width * workspace_height)

    # Calculate the center of the largest rectangle in grid coordinates
    v_center = (v_start_rect + v_end_rect) // 2
    u_center = (u_start_rect + u_end_rect) // 2

    # Map the center to world coordinates
    center_x, center_y = to_world_coords(u_center, v_center)

    if visualize:
        try:
            # Mark center in the grid for visualization
            grid_vis = grid.copy()
            grid_vis[v_center : v_center + 1, u_center : u_center + 1] = 2
            # Normalize grid to 0–255 for visualization
            grid_visual = (grid_vis * 255 // 2).astype(np.uint8)
            cv2.imshow("Largest Free Space Grid", grid_visual)
            cv2.waitKey(1)  # Use 1 instead of 0 to avoid blocking in non-interactive mode
        except Exception as e:
            logger.warning(f"Could not visualize free space grid: {e}")

    logger.info(f"Largest free area: {largest_area_m2:.4f} square meters")
    logger.info(f"Center: ({center_x:.4f}, {center_y:.4f}) meters")

    return largest_area_m2, center_x, center_y


def _max_rectangle_area(matrix: np.ndarray) -> Tuple[int, Tuple[int, int], Tuple[int, int]]:
    """
    Find the largest rectangle of zeros in a binary matrix.
    Uses a dynamic programming approach with a stack-based histogram calculation.
    """
    max_area = 0
    top_left = (0, 0)
    bottom_right = (0, 0)
    dp = [0] * len(matrix[0])  # DP array for heights

    for v, row in enumerate(matrix):  # Iterate over rows (v-axis)
        for u in range(len(row)):  # Iterate over columns (u-axis)
            dp[u] = dp[u] + 1 if row[u] == 0 else 0  # Update heights

        # Compute the maximum area with the updated histogram
        stack = []
        for k in range(len(dp) + 1):
            while stack and (k == len(dp) or dp[k] < dp[stack[-1]]):
                h = dp[stack.pop()]
                w = k if not stack else k - stack[-1] - 1
                area = h * w
                if area > max_area:
                    max_area = area
                    top_left = (v - h + 1, stack[-1] + 1 if stack else 0)
                    bottom_right = (v, k - 1)
            stack.append(k)

    return max_area, top_left, bottom_right
