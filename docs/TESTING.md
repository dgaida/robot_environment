# Testing

## Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=robot_environment --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_environment.py

# Run integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

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
