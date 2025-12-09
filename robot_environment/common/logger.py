"""
Logging decorators for robot_environment package.
"""

import logging
import inspect
from functools import wraps

# TODO: in future move definition elsewhere
# variable that defines whether pyniryo or pyniryo2 should be used
pyniryo_v = "pyniryo"  # "pyniryo2"


def log_start_end(logger: logging.Logger):
    """
    Decorator to log function entry and exit.

    Args:
        logger: Logger instance to use

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"START {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"END {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"ERROR in {func.__name__}: {e}", exc_info=True)
                raise

        return wrapper

    return decorator


def log_start_end_cls():
    """
    Decorator to log class method entry and exit.
    Uses instance's logger if available, or creates a module logger.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Try to get logger from instance
            if hasattr(self, "_logger"):
                logger = self._logger
            else:
                # Fallback to module logger
                logger = logging.getLogger(self.__class__.__module__)

            class_name = ""

            # Only log if verbose or DEBUG level
            should_log = (hasattr(self, "_verbose") and self._verbose) or logger.isEnabledFor(logging.DEBUG)

            if should_log:
                class_name = self.__class__.__name__
                func_line = inspect.getsourcelines(func)[1]
                logger.debug(f"START {func.__name__} " f"(Class: {class_name}, Line: {func_line})")

            try:
                result = func(self, *args, **kwargs)

                if should_log:
                    logger.debug(f"END {func.__name__}")

                return result

            except Exception as e:
                logger.error(f"ERROR in {class_name}.{func.__name__}: {e}", exc_info=True)
                raise

        return wrapper

    return decorator
