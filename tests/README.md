# Robot Environment Tests

Comprehensive test suite for the robot_environment package.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and shared fixtures
├── README.md               # This file
├── test_pose_object.py     # Unit tests for PoseObjectPNP class
├── test_object.py          # Unit tests for Object class
├── test_objects.py         # Unit tests for Objects collection
├── test_workspace.py       # Unit tests for Workspace classes
├── test_robot_api.py       # Unit tests for Robot API
└── test_integration.py     # Integration tests
```

## Running Tests

### Run All Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=robot_environment --cov-report=html --cov-report=term
```

### Run Specific Test Files
```bash
pytest tests/test_pose_object.py
pytest tests/test_object.py
pytest tests/test_objects.py
```

### Run Tests by Marker
```bash
# Run only integration tests
pytest -m integration

# Run only slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"

# Skip integration tests
pytest -m "not integration"
```

### Verbose Output
```bash
pytest -v
pytest -vv  # Extra verbose
```

### Run Specific Test
```bash
pytest tests/test_pose_object.py::TestPoseObjectPNP::test_initialization
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.integration` - Integration tests that test multiple components together
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.requires_robot` - Tests that require actual robot hardware (skipped by default)
- `@pytest.mark.requires_redis` - Tests that require Redis server (skipped by default)

## Test Coverage

The test suite covers:

### Unit Tests
- **PoseObjectPNP** (test_pose_object.py)
  - Initialization and properties
  - Arithmetic operations (addition, subtraction)
  - Equality and approximate equality
  - Coordinate transformations
  - Quaternion conversions
  - Transformation matrices

- **Object** (test_object.py)
  - Object initialization with/without masks
  - Position and dimension calculations
  - Serialization (to_dict, to_json, from_dict, from_json)
  - String representations for LLM and chat
  - Center of mass calculations
  - Workspace integration

- **Objects Collection** (test_objects.py)
  - Collection operations (append, get, filter)
  - Spatial queries (left/right/above/below)
  - Size-based queries (largest, smallest, sorted)
  - Label filtering
  - Nearest object search
  - Serialization of collections

- **Workspace** (test_workspace.py)
  - NiryoWorkspace initialization
  - Coordinate transformations (camera to world)
  - Workspace dimensions and corners
  - Visibility checks
  - Observation poses

- **Robot API** (test_robot_api.py)
  - Location enum operations
  - String to Location conversions
  - RobotAPI abstract interface

### Integration Tests (test_integration.py)
- Object-Workspace integration
- Spatial query operations on collections
- Serialization roundtrips
- Pose transformation chains
- Complete pick-and-place scenarios
- Workspace scanning patterns

## Fixtures

Common fixtures are defined in `conftest.py`:

- `mock_workspace` - Mock workspace for testing objects
- `sample_objects` - Collection of sample objects
- `sample_image` - Test image (640x480)
- `sample_mask` - Test segmentation mask
- `mock_redis` - Mock Redis client
- `clean_environment` - Temporary test directory

## Writing New Tests

### Basic Test Structure
```python
import pytest
from robot_environment.objects.object import Object

def test_my_feature():
    """Test description"""
    # Arrange
    obj = Object(...)
    
    # Act
    result = obj.some_method()
    
    # Assert
    assert result == expected_value
```

### Using Fixtures
```python
def test_with_fixture(mock_workspace):
    """Test using a fixture"""
    obj = Object("test", 100, 100, 200, 200, None, mock_workspace)
    assert obj.label() == "test"
```

### Parametrized Tests
```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_with_params(input, expected):
    assert input * 2 == expected
```

### Testing Exceptions
```python
def test_invalid_input():
    with pytest.raises(ValueError, match="Invalid"):
        some_function_that_raises()
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Clear Names**: Use descriptive test names that explain what is being tested
3. **AAA Pattern**: Arrange, Act, Assert
4. **One Assertion**: Focus on testing one thing per test (when possible)
5. **Use Fixtures**: Reuse common setup code
6. **Mock External Dependencies**: Don't rely on real robot hardware or Redis
7. **Test Edge Cases**: Include tests for boundary conditions
8. **Document Complex Tests**: Add docstrings explaining non-obvious test logic

## Continuous Integration

Tests run automatically on GitHub Actions for:
- Push to master branch
- Pull requests to master
- Multiple Python versions (3.8, 3.9, 3.10, 3.11)
- Multiple operating systems (Ubuntu, Windows, macOS)

See `.github/workflows/tests.yml` for CI configuration.

## Coverage Goals

Target coverage: **>90%**

Current coverage can be checked with:
```bash
pytest --cov=robot_environment --cov-report=term-missing
```

## Troubleshooting

### Import Errors
If you see import errors, ensure the package is installed:
```bash
pip install -e .
```

### Missing Dependencies
Install test dependencies:
```bash
pip install pytest pytest-cov
```

### Slow Tests
Skip slow tests during development:
```bash
pytest -m "not slow"
```

### Test Discovery Issues
Ensure `__init__.py` exists in test directories and test files start with `test_`.

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
