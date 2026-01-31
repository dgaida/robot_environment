from __future__ import annotations
import re
import ast
import logging
from typing import Tuple, List, Dict, Any, Optional

from .robot_api import Location

logger = logging.getLogger(__name__)


def parse_robot_command(line: str) -> Tuple[Optional[str], Optional[str], List[Any], Dict[str, Any]]:
    """
    Parse a single line of input into target object, method name, positional arguments, and keyword arguments.

    Example:
        'robot.pick_place_object(object_name="pencil", location="right next to")'
        returns: ('robot', 'pick_place_object', [], {'object_name': 'pencil', 'location': <Location.RIGHT_NEXT_TO>})

    Args:
        line: The command string to parse.

    Returns:
        Tuple containing:
            - target_object (str or None): 'robot', 'agent', etc.
            - method_name (str or None): name of the method to call.
            - positional_args (list): list of positional arguments.
            - keyword_args (dict): dictionary of keyword arguments.
    """
    try:
        # Match the object, method, and arguments
        # Pattern: object.method(args)
        match = re.match(r"(\w+)\.(\w+)\((.*)\)", line.strip())
        if not match:
            logger.warning(f"Invalid command format: {line}")
            return None, None, [], {}

        target_object, method, args_str = match.groups()

        # Use AST to safely parse the arguments
        positional_args = []
        keyword_args = {}

        if args_str:
            # Parse the argument string using AST
            # We wrap it in a function call so AST can parse it as arguments
            parsed_args = ast.parse(f"func({args_str})").body[0].value

            # Convert AST nodes to Python objects
            positional_args = [ast.literal_eval(arg) for arg in parsed_args.args]
            keyword_args = {kw.arg: ast.literal_eval(kw.value) for kw in parsed_args.keywords if kw.arg is not None}

        # Post-processing: Convert location string to Location enum if present
        if "location" in keyword_args:
            location_value = keyword_args["location"]
            if isinstance(location_value, str):
                keyword_args["location"] = Location.convert_str2location(location_value)

        return target_object, method, positional_args, keyword_args

    except Exception as e:
        logger.error(f"Error parsing command '{line}': {e}", exc_info=True)
        return None, None, [], {}
