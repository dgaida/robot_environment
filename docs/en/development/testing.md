# Testing

We use `pytest` for unit and integration testing. Our goal is to maintain >95% code coverage to ensure system reliability across various robot configurations.

## Running Tests

### Standard Execution
```bash
# Run all unit tests (skipping integration and slow tests by default)
python3 -m pytest

# Run with coverage report
python3 -m pytest --cov=robot_environment --cov-report=term-missing

### Advanced Options
```bash
# Run specific test file
python3 -m pytest tests/test_environment.py

# Run integration tests (these require specific setup)
python3 -m pytest -m integration

# Run everything EXCEPT slow tests
python3 -m pytest -m "not slow"

# Run tests that require a real robot
python3 -m pytest -m requires_robot
```

## Test Markers

We use markers to categorize tests:
- `integration`: Tests that verify interaction between multiple components.
- `slow`: Tests that take a long time to run (e.g. complex simulations).
- `requires_robot`: Tests that can only run when connected to actual hardware.
- `requires_redis`: Tests that require a running Redis server.

## Test Organization

```
tests/
├── conftest.py              # Fixtures and configuration
├── test_environment.py      # Environment tests
├── test_environment_extended.py  # Extended environment tests
├── camera/
│   └── test_niryo_framegrabber.py
├── robot/
│   ├── test_robot.py
│   ├── test_robot_api.py
│   ├── test_niryo_robot_controller.py
│   └── test_widowx_robot_controller.py
└── test_integration.py      # Integration tests
```

For detailed testing information, see **[../tests/README.md](../tests/README.md)**
