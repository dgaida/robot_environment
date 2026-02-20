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
        """
        Decorator that wraps the function with logging.

        Args:
            func: Function to wrap.

        Returns:
            Wrapped function.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper function that logs start, end, and errors.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.

            Returns:
                Result of the wrapped function.
            """
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
        """
        Decorator that wraps the class method with logging.

        Args:
            func: Method to wrap.

        Returns:
            Wrapped method.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            """
            Wrapper method that logs start, end, and errors using the instance logger.

            Args:
                self: Instance of the class.
                *args: Positional arguments.
                **kwargs: Keyword arguments.

            Returns:
                Result of the wrapped method.
            """
            # Try to get logger from instance
            if hasattr(self, "_logger") and self._logger is not None:
                logger = self._logger
            else:
                # Fallback to module logger - ensure it's properly initialized
                logger = logging.getLogger(self.__class__.__module__)
                # Ensure logger has at least one handler to avoid issues
                if not logger.handlers and not logger.parent.handlers:
                    # Add a basic handler if none exists
                    handler = logging.StreamHandler()
                    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
                    logger.addHandler(handler)
                    logger.setLevel(logging.INFO)

            class_name = ""

            # Only log if verbose or DEBUG level - check logger is valid first
            should_log = False
            try:
                should_log = (hasattr(self, "_verbose") and self._verbose) or (
                    logger is not None and logger.isEnabledFor(logging.DEBUG)
                )
            except Exception:
                # If logger check fails, just skip logging but continue execution
                should_log = False

            if should_log:
                class_name = self.__class__.__name__
                try:
                    func_line = inspect.getsourcelines(func)[1]
                    logger.debug(f"START {func.__name__} (Class: {class_name}, Line: {func_line})")
                except Exception:
                    # If we can't get source lines, just log without line number
                    logger.debug(f"START {func.__name__} (Class: {class_name})")

            try:
                result = func(self, *args, **kwargs)

                if should_log:
                    logger.debug(f"END {func.__name__}")

                return result

            except Exception as e:
                if logger is not None:
                    logger.error(f"ERROR in {class_name}.{func.__name__}: {e}", exc_info=True)
                raise

        return wrapper

    return decorator
