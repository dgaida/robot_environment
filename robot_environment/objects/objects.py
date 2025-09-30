# Objects class that is a List of Object's
# Documentation and type definitions are final

from ..common.logger import log_start_end_cls

from ..robot.robot_api import Location

import numpy as np
import math

from .object import Object

from typing import TYPE_CHECKING, List, Optional, Union, Dict, Any

if TYPE_CHECKING:
    from ..robot.robot_api import Location
    from ..workspaces.workspace import Workspace


class Objects(List):
    """
    A class representing a list of Object instances. Objects are stored in VisualCortex class and therefore are not
    real things, but just seen from a camera.
    """

    # *** CONSTRUCTORS ***
    def __init__(self, iterable=None, verbose=False):
        """
        Initializes the Objects instance.

        Args:
            iterable (iterable, optional): An iterable of Object instances. Defaults to an empty list.
            verbose:
        """
        if iterable is None:
            iterable = []
        super().__init__(iterable)

        self._verbose = verbose

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    @log_start_end_cls()
    def get_detected_object(self, coordinate: List[float], label: Optional[str] = None) -> Optional["Object"]:
        """
        Retrieves a detected object at or near a specified world coordinate, optionally filtering by label.

        This method checks for objects detected by the camera that are close to the specified coordinate (within
        2 centimeters). If multiple objects meet the criteria, the first object in the list is returned.

        Args:
            coordinate (List[float]): A 2D coordinate in the world coordinate system [x, y].
                Only objects within a 2-centimeter radius of this coordinate are considered.
            label (Optional[str]): An optional filter for the object's label. If specified, only an object
                with the matching label is returned.

        Returns:
            Optional[Object]: The first object detected near the given coordinate (and matching the label, if provided).
            Returns `None` if no such object is found.
        """
        detected_objects = self.get_detected_objects(Location.CLOSE_TO, coordinate, label)

        if detected_objects:
            return detected_objects[0]
        else:
            return None

    def get_detected_objects(self, location: Union["Location", str] = Location.NONE, coordinate: List[float] = None,
                             label: Optional[str] = None) -> Optional["Objects"]:
        """
        Get list of objects detected by the camera in the workspace.

        Args:
            location (Location, optional): acts as filter. can have the values:
            - "left next to": Only objects left of the given coordinate are returned,
            - "right next to": Only objects right of the given coordinate are returned,
            - "above": Only objects above the given coordinate are returned,
            - "below": Only objects below the given coordinate are returned,
            - "close to": Only objects close to the given coordinate are returned (within 2 centimeters),
            - None: no filter, all objects are returned (default).
            coordinate (List[float], optional): some (x,y) coordinate in the world coordinate system.
            Together with 'location' it acts as a filter. Required if 'location' is specified.
            label (str, optional): Only objects with the given label are returned.

        Returns:
            Optional["Objects"]: list of objects detected by the camera in the workspace.
        """
        detected_objects = self

        if label is not None:
            # TODO: ich muss hier obj.label in teile splitten und prüfen ob label == eine der teile ist. weil pen auch in pencil ist
            # detected_objects = Objects(obj for obj in self if obj.label() == label)
            detected_objects = Objects(obj for obj in self if label in obj.label())

        location = Location.convert_str2location(location)

        if location is Location.NONE:
            return detected_objects
        elif location == Location.LEFT_NEXT_TO:
            return Objects(obj for obj in detected_objects if obj.y_com() > coordinate[1])
        elif location == Location.RIGHT_NEXT_TO:
            return Objects(obj for obj in detected_objects if obj.y_com() < coordinate[1])
        elif location == Location.ABOVE:
            return Objects(obj for obj in detected_objects if obj.x_com() > coordinate[0])
        elif location == Location.BELOW:
            return Objects(obj for obj in detected_objects if obj.x_com() < coordinate[0])
        elif location == Location.CLOSE_TO:
            return Objects(obj for obj in detected_objects if np.sqrt((obj.x_com() - coordinate[0])**2 +
                                                                      (obj.y_com() - coordinate[1])**2) <= 0.02)
        else:
            print("Error in get_detected_objects: Unknown Location:", location)
            return None

    def get_nearest_detected_object(self, coordinate: List[float],
                                    label: Optional[str] = None) -> tuple[Optional["Object"], float]:
        """
        Retrieves a detected object nearest to a specified world coordinate, optionally filtering by label.

        This method goes through all objects detected by the camera and returns the one
        (optionally with the given label) nearest to the given coordinate.

        Args:
            coordinate (List[float]): A 2D coordinate in the world coordinate system [x, y].
            The object nearest to this coordinate is returned.
            label (Optional[str]): An optional filter for the object's label. If specified, only an object
                with the matching label is returned.

        Returns:
            tuple:
            - Optional[Object]: The object nearest to the given coordinate (and matching the label, if provided).
              Returns `None` if no such object is found.
            - float: distance (RMSE) of the returned object to the given coordinate in meters.
        """
        nearest_object = None
        min_distance = float('inf')

        for obj in self:
            if label is None or obj.label() == label:
                # Calculate Euclidean distance
                distance = math.sqrt(
                    (obj.x_com() - coordinate[0]) ** 2 + (obj.y_com() - coordinate[1]) ** 2
                )
                # print(distance, min_distance)
                if distance < min_distance:
                    min_distance = distance
                    nearest_object = obj

        return nearest_object, min_distance

    def get_detected_objects_as_comma_separated_string(self) -> str:
        """
        Returns detected objects as "," separated string.

        Returns:
            detected objects as "," separated string.
        """
        return f"""{', '.join(f"'{item.label()}'" for item in self)}"""

    def get_largest_detected_object(self) -> tuple[Optional["Object"], float]:
        """
        Returns the largest detected object based on its size in square meters.

        Returns:
            tuple: (largest_object, largest_size_m2) where:
                - largest_object (Object): The largest detected object.
                - largest_size_m2 (float): The size of the largest object in square meters.
        """
        largest_object = max(self, key=lambda obj: obj.size_m2())
        return largest_object, largest_object.size_m2()

    def get_smallest_detected_object(self) -> tuple[Optional["Object"], float]:
        """
        Returns the smallest detected object based on its size in square meters.

        Returns:
            tuple: (smallest_object, smallest_size_m2) where:
                - smallest_object (Object): The smallest detected object.
                - smallest_size_m2 (float): The size of the smallest object in square meters.
        """
        smallest_object = min(self, key=lambda obj: obj.size_m2())
        return smallest_object, smallest_object.size_m2()

    def get_detected_objects_sorted(self, ascending: bool = True) -> "Objects":
        """
        Returns the detected objects sorted by size in square meters.

        Args:
            ascending (bool): If True, sorts the objects in ascending order.
                              If False, sorts in descending order.

        Returns:
            Objects: The list of detected objects sorted by size.
        """
        return Objects(sorted(self, key=lambda obj: obj.size_m2(), reverse=not ascending))

    # *** PUBLIC methods ***

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** HELPER METHODE FÜR REDIS PUBLISHER ***

    @staticmethod
    def objects_to_dict_list(objects: "Objects") -> List[Dict[str, Any]]:
        """
        Converts a list of Object instances to a list of dictionaries.

        Args:
            objects: List of Object instances

        Returns:
            List[Dict[str, Any]]: List of dictionary representations
        """
        return [obj.to_dict() for obj in objects]

    @staticmethod
    def dict_list_to_objects(dict_list: List[Dict[str, Any]], workspace: "Workspace") -> "Objects":
        """
        Converts a list of dictionaries back to Object instances.

        Args:
            dict_list: List of dictionary representations
            workspace: Workspace instance

        Returns:
            List[Object]: List of reconstructed Object instances
        """
        objects = Objects()
        for obj_dict in dict_list:
            obj = Object.from_dict(obj_dict, workspace)
            if obj is not None:
                objects.append(obj)
        return objects

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    def verbose(self) -> bool:
        """

        Returns: True, if verbose is on, else False

        """
        return self._verbose

    # *** PRIVATE variables ***

    _verbose = False
