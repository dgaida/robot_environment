# Docstring Style Guide

We follow the **Google Python Style Guide** for all docstrings in this project.

## Format

```python
def function(parameter: int) -> bool:
    """
    Short summary of the function.

    Detailed description of the function and its logic,
    if necessary.

    Args:
        parameter: Description of the parameter.

    Returns:
        bool: Description of the return value.

    Raises:
        ValueError: If the parameter is invalid.
    """
```

## Requirements
- All public classes, methods, and functions must be documented.
- Module docstrings at the beginning of each file are required.
- We enforce an API documentation coverage of at least 95% using `interrogate`.
