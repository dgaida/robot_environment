"""
Centralized logging configuration for robot_environment package.
"""

import logging
import sys
from typing import Optional

# Default format with timestamp, level, module, and message
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"


def setup_logger(
    name: str, level: int = logging.INFO, format_string: Optional[str] = None, log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup a logger with console and optional file handlers.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Use simple format for console by default
    formatter = logging.Formatter(format_string or SIMPLE_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = logging.Formatter(DEFAULT_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def set_verbose(logger: logging.Logger, verbose: bool):
    """
    Set logger level based on verbose flag.

    Args:
        logger: Logger instance to configure
        verbose: If True, set to DEBUG; if False, set to INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def get_package_logger(module_name: str, verbose: bool = False) -> logging.Logger:
    """
    Get a logger for a specific module in the robot_environment package.

    Args:
        module_name: Name of the module (use __name__)
        verbose: Enable verbose (DEBUG level) logging

    Returns:
        Configured logger
    """
    logger = setup_logger(module_name)
    set_verbose(logger, verbose)
    return logger
