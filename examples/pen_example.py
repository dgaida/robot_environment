"""
Example: Searching for a pen and performing pick-and-place.
"""

import argparse
import cv2
import time
import os
from robot_environment import Environment
from robot_workspace import Location


def draw_detections(image, objects):
    if image is None:
        return image
    canvas = image.copy()
    for obj in objects:
        try:
            contour = obj.largest_contour()
            if contour is not None and len(contour) > 0:
                overlay = canvas.copy()
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
                cv2.drawContours(canvas, [contour], -1, (0, 255, 0), 2)
        except Exception:
            pass
        try:
            contour = obj.largest_contour()
            if contour is not None and len(contour) > 0:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(canvas, obj.label(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        except Exception:
            pass
    return canvas


def example_search_pick_place_pen(env, show_gui=True):
    print("Example 1: Search, Pick and Place Pen")
    env.robot_move2home_observation_pose()
    time.sleep(1)
    img = env.get_current_frame()
    if img is not None:
        objs = env.get_detected_objects_from_memory()
        vis = draw_detections(img, objs)
        if show_gui:
            cv2.imshow("View", vis)
            cv2.waitKey(100)

    pen = next((obj for obj in env.get_detected_objects_from_memory() if obj.label().lower() == "pen"), None)
    if pen:
        target = env.get_workspace_coordinate_from_point(env.get_workspace_home_id(), "lower right corner")
        if target:
            env.robot().pick_place_object(
                object_name="pen", pick_coordinate=pen.xy_com(), place_coordinate=target, location=Location.NONE
            )


def example_print_all_objects(env):
    print("Example 2: List All Detected Objects")
    objs = env.get_detected_objects_from_memory()
    for i, obj in enumerate(objs, 1):
        print(f" {i}. {obj.label()} at {obj.xy_com()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", default=True)
    parser.add_argument("--no-sim", action="store_false", dest="sim")
    parser.add_argument("--robot", type=str, default="niryo")
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    env = Environment(el_api_key="key", use_simulation=args.sim, robot_id=args.robot, verbose=args.verbose)
    try:
        example_search_pick_place_pen(env, show_gui="DISPLAY" in os.environ)
        example_print_all_objects(env)
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()
