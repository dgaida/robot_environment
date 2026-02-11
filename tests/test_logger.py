"""
Unit tests for logger decorators
"""

import pytest
import logging
from unittest.mock import Mock
from robot_environment.common.logger import log_start_end, log_start_end_cls
from robot_environment.common.logger_config import setup_logger, set_verbose, get_package_logger


class TestLogger:
    """Test suite for logging decorators and config"""

    def test_log_start_end(self):
        """Test log_start_end decorator"""
        mock_logger = Mock(spec=logging.Logger)

        @log_start_end(mock_logger)
        def test_func(a, b):
            return a + b

        result = test_func(1, 2)
        assert result == 3
        assert mock_logger.debug.call_count == 2
        assert "START test_func" in mock_logger.debug.call_args_list[0][0][0]
        assert "END test_func" in mock_logger.debug.call_args_list[1][0][0]

    def test_log_start_end_error(self):
        """Test log_start_end decorator with exception"""
        mock_logger = Mock(spec=logging.Logger)

        @log_start_end(mock_logger)
        def error_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            error_func()

        assert mock_logger.error.called
        assert "ERROR in error_func" in mock_logger.error.call_args[0][0]

    def test_log_start_end_cls(self):
        """Test log_start_end_cls decorator"""

        class TestCls:
            def __init__(self):
                self._logger = Mock(spec=logging.Logger)
                self._verbose = True

            @log_start_end_cls()
            def method(self, x):
                return x * 2

        obj = TestCls()
        result = obj.method(5)
        assert result == 10
        assert obj._logger.debug.call_count == 2

    def test_log_start_end_cls_fallback(self):
        """Test log_start_end_cls fallback to module logger (lines 60-63)"""

        class FallbackCls:
            # No _logger attribute
            @log_start_end_cls()
            def method(self):
                return True

        obj = FallbackCls()
        # Should not crash, will use module logger
        assert obj.method() is True

    def test_log_start_end_cls_error(self):
        """Test log_start_end_cls with exception"""

        class ErrorCls:
            def __init__(self):
                self._logger = Mock(spec=logging.Logger)
                self._verbose = True

            @log_start_end_cls()
            def error_method(self):
                raise RuntimeError("Cls error")

        obj = ErrorCls()
        with pytest.raises(RuntimeError):
            obj.error_method()

        assert obj._logger.error.called

    def test_setup_logger(self):
        """Test setup_logger utility"""
        logger = setup_logger("test_setup", level=logging.DEBUG)
        assert logger.name == "test_setup"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) >= 1

    def test_set_verbose(self):
        """Test set_verbose utility"""
        logger = logging.getLogger("test_verbose")
        set_verbose(logger, True)
        assert logger.level == logging.DEBUG
        set_verbose(logger, False)
        assert logger.level == logging.INFO

    def test_get_package_logger(self):
        """Test get_package_logger utility"""
        logger = get_package_logger("test_pkg", verbose=True)
        assert logger.name == "test_pkg"
        assert logger.level == logging.DEBUG
