# class defining a decorator to log functions entry and exit
# final

import inspect

# TODO: in future move definition elsewhere
# variable that defines whether pyniryo or pyniryo2 should be used
pyniryo_v = "pyniryo"  # "pyniryo2"


def log_start_end(verbose: bool = True):
    """
    Returns a decorator to log the start and end of a function.

    Args:
        verbose (bool): If True, logs will be printed.

    Returns:
        function: Decorator for logging start and end of functions.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if verbose:
                print(f"START {func.__name__}")
            result = func(*args, **kwargs)
            if verbose:
                print(f"END {func.__name__}")
            return result

        return wrapper

    return decorator


def log_start_end_cls():
    """
    Decorator to log the start and end of a method, including class name and line number.
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Get verbose attribute from the instance or class
            verbose = getattr(self, "_verbose", False)
            class_name = None
            func_line_number = None

            if verbose:
                # Retrieve class name
                class_name = self.__class__.__name__
                # Retrieve line number where the function is defined
                func_line_number = inspect.getsourcelines(func)[1]
                # Log start message
                print(f"START {func.__name__} (Class: {class_name}, Line: {func_line_number})")
            result = func(self, *args, **kwargs)
            if verbose:
                # Log end message
                print(f"END {func.__name__} (Class: {class_name}, Line: {func_line_number})")
            return result

        return wrapper

    return decorator
