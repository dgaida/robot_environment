"""
Unit tests for command_processor.py
"""

import pytest
from robot_environment.robot.command_processor import parse_robot_command
from robot_environment.robot.robot_api import Location


def test_parse_robot_command_success():
    line = 'robot.pick_place_object(object_name="pencil", location="right next to")'
    target, method, pos, kw = parse_robot_command(line)
    assert target == "robot"
    assert method == "pick_place_object"
    assert kw["object_name"] == "pencil"
    assert kw["location"] == Location.RIGHT_NEXT_TO


def test_parse_robot_command_invalid_format():
    line = "invalid command"
    target, method, pos, kw = parse_robot_command(line)
    assert target is None


def test_parse_robot_command_exception():
    # Provide something that re.match accepts but AST fails on
    # e.g. unclosed parenthesis inside the arguments string
    line = "robot.method(arg="
    target, method, pos, kw = parse_robot_command(line)
    assert target is None

def test_parse_robot_command_empty_args():
    line = "robot.home()"
    target, method, pos, kw = parse_robot_command(line)
    assert target == "robot"
    assert method == "home"
    assert pos == []
    assert kw == {}
