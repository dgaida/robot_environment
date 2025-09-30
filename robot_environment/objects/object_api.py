# abstract Class defining an object. An object has pixel coordinates as well as world coordinates.
# Documentation and type definitions are final.

from abc import ABC, abstractmethod

from typing import List


class ObjectAPI(ABC):
    """
    Abstract Class defining an object. An object has pixel coordinates as well as world coordinates.
    This class provides those methods of teh class Object that are provided to the LLM.

    """

    # *** CONSTRUCTORS ***
    def __init__(self):
        pass

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    @abstractmethod
    def label(self) -> str:
        """
        Returns the label of the object (e.g., "chocolate bar").

        Returns:
            str: the label of the object.
        """
        pass

    @abstractmethod
    def coordinate(self) -> List[float]:
        """
        Returns (x,y) world coordinates of the center of mass of the object, measured in meters.
        At these coordinates you can pick the object. The List contains two float values.

        Returns:
            List[float]: x,y world coordinates of the object. At these coordinates you can pick the object.
        """
        pass

    # *** PRIVATE variables ***
