#

import threading
import time

from robot_environment.environment import Environment


def start_camera_updates(environment, visualize=False):
    def loop():
        for img in environment.update_camera_and_objects(visualize=visualize):
            # In CLI, we might not use img, but you could save or log info here
            pass  # or print("Camera updated")
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    env = Environment("el", True, "niryo", True)

    # env.create_dummy_environment()

    # Start background camera updates
    start_camera_updates(env, visualize=True)  # set visualize=False if you don't want OpenCV windows

    time.sleep(15)

    robot = env.robot()

    detected_objects = env.get_detected_objects()

    # print(detected_objects)

#     robot.pick_place_object("red square", [0.235, 0.3], [0.54, 0.43], location="right next to")
    robot.pick_place_object("pencil", [-0.1, 0.01], [0.1, 0.11], location="right next to")

#     robot.pick_place_object("red cube", [0.235, 0.3], [0.54, 0.43], location="right next to")

    env.robot_move2observation_pose(env.get_workspace_home_id())

    # print("End of program")

    # del env

