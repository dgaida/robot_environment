"""
Multi-Workspace Example for Niryo Ned2 Robot
===========================================

This example demonstrates how to use two workspaces simultaneously:
- Pick objects from the left workspace
- Place them in the right workspace

Create this file at: robot_environment/examples/multi_workspace_example.py
"""

import sys
import os
import threading
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from robot_environment.environment import Environment
from robot_workspace import Location


def start_camera_updates(environment, visualize=False):
    """Start camera updates in background thread."""

    def loop():
        for img in environment.update_camera_and_objects(visualize=visualize):
            pass

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def example_1_simple_transfer():
    """
    Example 1: Simple Object Transfer
    Pick an object from left workspace and place in right workspace.
    """
    print("\n" + "=" * 70)
    print("Example 1: Simple Object Transfer Between Workspaces")
    print("=" * 70)

    # Initialize environment
    env = Environment(
        el_api_key="your_elevenlabs_key",
        use_simulation=False,  # Set to True for Gazebo
        robot_id="niryo",
        verbose=True,
        start_camera_thread=False,  # We'll start it manually
    )

    # Start camera updates
    start_camera_updates(env, visualize=True)

    # Get workspaces
    left_workspace_id = env.workspaces().get_workspace_left_id()
    right_workspace_id = env.workspaces().get_workspace_right_id()

    print("\nConfigured workspaces:")
    print(f"  Left workspace: {left_workspace_id}")
    print(f"  Right workspace: {right_workspace_id}")

    # Move to left workspace and observe
    print("\nObserving left workspace...")
    env.robot_move2observation_pose(left_workspace_id)
    env.set_current_workspace(left_workspace_id)
    time.sleep(2)  # Wait for detection

    # Check what objects are in left workspace
    left_objects = env.get_detected_objects_from_workspace(left_workspace_id)
    print(f"\nObjects in left workspace: {len(left_objects)}")
    for obj in left_objects:
        print(f"  - {obj.label()} at [{obj.x_com():.2f}, {obj.y_com():.2f}]")

    # Move to right workspace and observe
    print("\nObserving right workspace...")
    env.robot_move2observation_pose(right_workspace_id)
    env.set_current_workspace(right_workspace_id)
    time.sleep(2)  # Wait for detection

    # Check what objects are in right workspace
    right_objects = env.get_detected_objects_from_workspace(right_workspace_id)
    print(f"\nObjects in right workspace: {len(right_objects)}")
    for obj in right_objects:
        print(f"  - {obj.label()} at [{obj.x_com():.2f}, {obj.y_com():.2f}]")

    # Pick from left and place in right
    if len(left_objects) > 0:
        obj_to_move = left_objects[0]
        print(f"\nMoving '{obj_to_move.label()}' from left to right workspace...")

        robot = env.robot()
        success = robot.pick_place_object_across_workspaces(
            object_name=obj_to_move.label(),
            pick_workspace_id=left_workspace_id,
            pick_coordinate=[obj_to_move.x_com(), obj_to_move.y_com()],
            place_workspace_id=right_workspace_id,
            place_coordinate=[0.25, -0.05],  # Specific location in right workspace
            location=Location.NONE,
        )

        if success:
            print("\n✓ Object successfully transferred!")
        else:
            print("\n✗ Transfer failed")
    else:
        print("\nNo objects found in left workspace to transfer")

    # Cleanup
    env.stop_camera_updates()
    print("\nExample 1 complete")


def example_2_organized_transfer():
    """
    Example 2: Organized Transfer
    Pick multiple objects from left workspace and arrange them in right workspace.
    """
    print("\n" + "=" * 70)
    print("Example 2: Organized Multi-Object Transfer")
    print("=" * 70)

    # Initialize environment
    env = Environment(
        el_api_key="your_elevenlabs_key", use_simulation=False, robot_id="niryo", verbose=True, start_camera_thread=False
    )

    start_camera_updates(env, visualize=True)

    left_workspace_id = env.workspaces().get_workspace_left_id()
    right_workspace_id = env.workspaces().get_workspace_right_id()

    robot = env.robot()

    # Observe left workspace
    print("\nObserving left workspace...")
    env.robot_move2observation_pose(left_workspace_id)
    env.set_current_workspace(left_workspace_id)
    time.sleep(2)

    left_objects = env.get_detected_objects_from_workspace(left_workspace_id)
    print(f"\nFound {len(left_objects)} objects in left workspace")

    if len(left_objects) == 0:
        print("No objects to transfer")
        env.stop_camera_updates()
        return

    # Observe right workspace
    print("\nObserving right workspace...")
    env.robot_move2observation_pose(right_workspace_id)
    env.set_current_workspace(right_workspace_id)
    time.sleep(2)

    # Define placement grid in right workspace
    # Objects will be placed in a neat row
    placement_positions = [
        [0.20, -0.08],  # Position 1
        [0.25, -0.08],  # Position 2
        [0.30, -0.08],  # Position 3
        [0.20, -0.04],  # Position 4
        [0.25, -0.04],  # Position 5
        [0.30, -0.04],  # Position 6
    ]

    # Transfer each object to a specific position
    for i, obj in enumerate(left_objects):
        if i >= len(placement_positions):
            print("\nNo more placement positions available")
            break

        target_pos = placement_positions[i]

        print(f"\n--- Transferring object {i+1}/{len(left_objects)} ---")
        print(f"Object: {obj.label()}")
        print(f"From: left workspace [{obj.x_com():.2f}, {obj.y_com():.2f}]")
        print(f"To: right workspace {target_pos}")

        success = robot.pick_place_object_across_workspaces(
            object_name=obj.label(),
            pick_workspace_id=left_workspace_id,
            pick_coordinate=[obj.x_com(), obj.y_com()],
            place_workspace_id=right_workspace_id,
            place_coordinate=target_pos,
            location=Location.NONE,
        )

        if success:
            print(f"✓ Successfully placed {obj.label()}")
        else:
            print(f"✗ Failed to place {obj.label()}")

        time.sleep(0.5)  # Brief pause between transfers

    # Final observation of result
    print("\n--- Final Result ---")
    env.robot_move2observation_pose(right_workspace_id)
    env.set_current_workspace(right_workspace_id)
    time.sleep(2)

    final_objects = env.get_detected_objects_from_workspace(right_workspace_id)
    print(f"\nObjects now in right workspace: {len(final_objects)}")
    for obj in final_objects:
        print(f"  - {obj.label()} at [{obj.x_com():.2f}, {obj.y_com():.2f}]")

    env.stop_camera_updates()
    print("\nExample 2 complete")


def example_3_sorting():
    """
    Example 3: Object Sorting
    Sort objects by type: cubes to left workspace, cylinders to right workspace.
    """
    print("\n" + "=" * 70)
    print("Example 3: Object Sorting Between Workspaces")
    print("=" * 70)

    # Initialize environment
    env = Environment(
        el_api_key="your_elevenlabs_key", use_simulation=False, robot_id="niryo", verbose=True, start_camera_thread=False
    )

    start_camera_updates(env, visualize=True)

    left_workspace_id = env.workspaces().get_workspace_left_id()
    right_workspace_id = env.workspaces().get_workspace_right_id()

    robot = env.robot()

    # Start by observing both workspaces to get all objects
    print("\nScanning all workspaces...")

    # Scan left workspace
    env.robot_move2observation_pose(left_workspace_id)
    env.set_current_workspace(left_workspace_id)
    time.sleep(2)

    # Scan right workspace
    env.robot_move2observation_pose(right_workspace_id)
    env.set_current_workspace(right_workspace_id)
    time.sleep(2)

    # Get all objects from both workspaces
    all_objects = env.get_all_workspace_objects()

    total_objects = sum(len(objs) for objs in all_objects.values())
    print(f"\nTotal objects detected: {total_objects}")
    for ws_id, objects in all_objects.items():
        print(f"  {ws_id}: {len(objects)} objects")
        for obj in objects:
            print(f"    - {obj.label()}")

    # Sorting rules
    left_workspace_labels = ["cube", "block", "box"]  # Square objects
    right_workspace_labels = ["cylinder", "can", "bottle"]  # Round objects

    print("\n--- Starting sorting operation ---")
    print("Square objects → Left workspace")
    print("Round objects → Right workspace")

    # Process each workspace
    for source_ws_id, objects in all_objects.items():
        for obj in objects:
            obj_label = obj.label().lower()

            # Determine target workspace
            if any(label in obj_label for label in left_workspace_labels):
                target_ws_id = left_workspace_id
                target_pos = [0.25, 0.08]  # Collection area in left
            elif any(label in obj_label for label in right_workspace_labels):
                target_ws_id = right_workspace_id
                target_pos = [0.25, -0.08]  # Collection area in right
            else:
                print(f"Skipping '{obj.label()}' - unknown type")
                continue

            # Skip if already in correct workspace
            if source_ws_id == target_ws_id:
                print(f"'{obj.label()}' already in correct workspace")
                continue

            # Move object to correct workspace
            print(f"\nMoving '{obj.label()}' from {source_ws_id} to {target_ws_id}")

            success = robot.pick_place_object_across_workspaces(
                object_name=obj.label(),
                pick_workspace_id=source_ws_id,
                pick_coordinate=[obj.x_com(), obj.y_com()],
                place_workspace_id=target_ws_id,
                place_coordinate=target_pos,
                location=Location.CLOSE_TO,  # Place near the collection area
            )

            if success:
                print(f"✓ Sorted {obj.label()}")
            else:
                print(f"✗ Failed to sort {obj.label()}")

    print("\n--- Sorting complete ---")

    # Show final distribution
    print("\nFinal object distribution:")
    final_objects = env.get_all_workspace_objects()
    for ws_id, objects in final_objects.items():
        print(f"\n{ws_id}: {len(objects)} objects")
        for obj in objects:
            print(f"  - {obj.label()} at [{obj.x_com():.2f}, {obj.y_com():.2f}]")

    env.stop_camera_updates()
    print("\nExample 3 complete")


def example_4_workspace_cleanup():
    """
    Example 4: Workspace Cleanup
    Clear one workspace by moving all objects to the other workspace.
    """
    print("\n" + "=" * 70)
    print("Example 4: Workspace Cleanup")
    print("=" * 70)

    # Initialize environment
    env = Environment(
        el_api_key="your_elevenlabs_key", use_simulation=False, robot_id="niryo", verbose=True, start_camera_thread=False
    )

    start_camera_updates(env, visualize=True)

    left_workspace_id = env.workspaces().get_workspace_left_id()
    right_workspace_id = env.workspaces().get_workspace_right_id()

    robot = env.robot()

    # Workspace to clear
    source_ws_id = left_workspace_id
    target_ws_id = right_workspace_id

    print(f"\nClearing workspace: {source_ws_id}")
    print(f"Moving all objects to: {target_ws_id}")

    # Observe source workspace
    env.robot_move2observation_pose(source_ws_id)
    env.set_current_workspace(source_ws_id)
    time.sleep(2)

    objects_to_move = env.get_detected_objects_from_workspace(source_ws_id)

    if len(objects_to_move) == 0:
        print(f"\nWorkspace {source_ws_id} is already empty")
        env.stop_camera_updates()
        return

    print(f"\nFound {len(objects_to_move)} objects to move")

    # Find largest free space in target workspace
    env.robot_move2observation_pose(target_ws_id)
    env.set_current_workspace(target_ws_id)
    time.sleep(2)

    largest_area, center_x, center_y = env.get_largest_free_space_with_center()
    print("\nLargest free space in target workspace:")
    print(f"  Area: {largest_area*10000:.2f} cm²")
    print(f"  Center: [{center_x:.2f}, {center_y:.2f}]")

    # Move all objects
    for i, obj in enumerate(objects_to_move):
        print(f"\n--- Moving object {i+1}/{len(objects_to_move)} ---")
        print(f"Object: {obj.label()}")

        # Place objects in a grid pattern around the free space center
        offset_x = 0.04 * (i // 3)  # 4cm spacing
        offset_y = 0.04 * (i % 3 - 1)  # -4cm, 0, +4cm

        target_pos = [center_x + offset_x, center_y + offset_y]

        success = robot.pick_place_object_across_workspaces(
            object_name=obj.label(),
            pick_workspace_id=source_ws_id,
            pick_coordinate=[obj.x_com(), obj.y_com()],
            place_workspace_id=target_ws_id,
            place_coordinate=target_pos,
            location=Location.NONE,
        )

        if success:
            print(f"✓ Moved {obj.label()}")
        else:
            print(f"✗ Failed to move {obj.label()}")

    # Verify source workspace is empty
    print("\n--- Verifying cleanup ---")
    env.robot_move2observation_pose(source_ws_id)
    env.set_current_workspace(source_ws_id)
    time.sleep(2)

    remaining_objects = env.get_detected_objects_from_workspace(source_ws_id)

    if len(remaining_objects) == 0:
        print(f"✓ Workspace {source_ws_id} is now empty")
    else:
        print(f"✗ {len(remaining_objects)} objects still in {source_ws_id}")

    env.stop_camera_updates()
    print("\nExample 4 complete")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Multi-Workspace Examples for Niryo Ned2")
    print("=" * 70)
    print("\nAvailable examples:")
    print("  1 - Simple Transfer: Move one object between workspaces")
    print("  2 - Organized Transfer: Move multiple objects to organized positions")
    print("  3 - Object Sorting: Sort objects by type into different workspaces")
    print("  4 - Workspace Cleanup: Clear one workspace into another")
    print()

    try:
        choice = input("Select example (1-4) or 'all' to run all: ").strip()

        if choice == "1":
            example_1_simple_transfer()
        elif choice == "2":
            example_2_organized_transfer()
        elif choice == "3":
            example_3_sorting()
        elif choice == "4":
            example_4_workspace_cleanup()
        elif choice.lower() == "all":
            example_1_simple_transfer()
            time.sleep(2)
            example_2_organized_transfer()
            time.sleep(2)
            example_3_sorting()
            time.sleep(2)
            example_4_workspace_cleanup()
        else:
            print("Invalid choice")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Examples complete")
    print("=" * 70)
