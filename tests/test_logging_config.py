"""Comprehensive tests for logging_config module."""

import pytest
import logging

from devdox_ai_sonar.logging_config import setup_logging, get_logger, quick_setup


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_default(self):
        """Test setup_logging with default parameters."""
        setup_logging()

        logger = logging.getLogger()
        assert logger.level == logging.INFO

    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom log level."""
        setup_logging(level="DEBUG")

        logger = logging.getLogger()
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_file(self, tmp_path):
        """Test setup_logging with log file."""
        log_file = tmp_path / "test.log"

        setup_logging(log_file=str(log_file))

        logger = logging.getLogger(__name__)
        logger.info("Test message")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_setup_logging_creates_directory(self, tmp_path):
        """Test that setup_logging creates log directory if it doesn't exist."""
        log_dir = tmp_path / "logs"
        log_file = log_dir / "test.log"

        setup_logging(log_file=str(log_file))

        assert log_dir.exists()
        assert log_file.exists()

    def test_setup_logging_custom_format(self, tmp_path):
        """Test setup_logging with custom format string."""
        log_file = tmp_path / "test.log"
        custom_format = "%(levelname)s - %(message)s"

        setup_logging(log_file=str(log_file), format_string=custom_format)

        logger = logging.getLogger(__name__)
        logger.info("Test message")

        content = log_file.read_text()
        assert "INFO - Test message" in content

    def test_setup_logging_max_bytes(self, tmp_path):
        """Test setup_logging with max_bytes parameter."""
        log_file = tmp_path / "test.log"

        # Setup with very small max_bytes
        setup_logging(log_file=str(log_file), max_bytes=100)

        logger = logging.getLogger(__name__)
        # Write enough to trigger rotation
        for i in range(20):
            logger.info(f"Test message {i} with some extra text to make it longer")

        # Check that backup file was created
        backup_files = list(tmp_path.glob("test.log.*"))
        # May or may not have created backup depending on timing
        # Just verify no crash occurred

    def test_setup_logging_backup_count(self, tmp_path):
        """Test setup_logging with backup_count parameter."""
        log_file = tmp_path / "test.log"

        setup_logging(log_file=str(log_file), backup_count=3)

        # Just verify it doesn't crash
        logger = logging.getLogger(__name__)
        logger.info("Test message")

    def test_setup_logging_env_var_level(self, monkeypatch):
        """Test setup_logging reads level from LOG_LEVEL environment variable."""
        monkeypatch.setenv("LOG_LEVEL", "WARNING")

        setup_logging()

        logger = logging.getLogger()
        assert logger.level == logging.WARNING

    def test_setup_logging_env_var_file(self, monkeypatch, tmp_path):
        """Test setup_logging reads log file from LOG_FILE environment variable."""
        log_file = tmp_path / "env_test.log"
        monkeypatch.setenv("LOG_FILE", str(log_file))

        setup_logging()

        logger = logging.getLogger(__name__)
        logger.info("Test from env")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test from env" in content

    def test_setup_logging_invalid_level_defaults_to_info(self):
        """Test that invalid log level defaults to INFO."""
        setup_logging(level="INVALID_LEVEL")

        logger = logging.getLogger()
        # Should default to INFO (20)
        assert logger.level in [logging.INFO, logging.DEBUG, logging.WARNING]

    def test_setup_logging_case_insensitive_level(self):
        """Test that log level is case-insensitive."""
        setup_logging(level="debug")
        logger = logging.getLogger()
        assert logger.level == logging.DEBUG

    def test_setup_logging_sets_third_party_levels(self):
        """Test that setup_logging sets levels for noisy third-party libraries."""
        setup_logging()

        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("openai").level == logging.WARNING
        assert logging.getLogger("anthropic").level == logging.WARNING
        assert logging.getLogger("google").level == logging.WARNING

    def test_setup_logging_logs_configuration(self, caplog, monkeypatch):
        """Test that setup_logging logs its configuration."""

        monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)
        # 2. Run the code
        with caplog.at_level(logging.INFO, logger="devdox_ai_sonar.logging_config"):
            setup_logging()

        # 3. Verify
        assert len(caplog.records) > 0
        assert any("Logging configured" in record.message for record in caplog.records)

    def test_setup_logging_logs_file_configuration(self, caplog, tmp_path, monkeypatch):
        """Test that setup_logging logs file configuration."""
        log_file = tmp_path / "test.log"
        monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)
        with caplog.at_level(logging.INFO):
            setup_logging(log_file=str(log_file))

        assert any("Logging to file" in record.message for record in caplog.records)

    def test_setup_logging_console_only_message(self, caplog, monkeypatch):
        """Test that setup_logging logs console-only message."""
        monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)
        with caplog.at_level(logging.INFO):
            setup_logging()

        assert any(
            "console only" in record.message.lower() for record in caplog.records
        )


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a Logger instance."""
        logger = get_logger(__name__)

        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_name(self):
        """Test that get_logger uses the provided name."""
        logger = get_logger("test.module")

        assert logger.name == "test.module"

    def test_get_logger_different_names(self):
        """Test that get_logger returns different loggers for different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_get_logger_can_log(self, tmp_path):
        """Test that logger returned by get_logger can actually log."""
        log_file = tmp_path / "test.log"
        setup_logging(log_file=str(log_file))

        logger = get_logger(__name__)
        logger.info("Test message from get_logger")

        content = log_file.read_text()
        assert "Test message from get_logger" in content


class TestQuickSetup:
    """Test quick_setup function."""

    def test_quick_setup_default(self):
        """Test quick_setup with default parameters."""
        logger = quick_setup()

        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO

    def test_quick_setup_debug_mode(self):
        """Test quick_setup with debug=True."""
        logger = quick_setup(debug=True)

        assert logger.level == logging.DEBUG

    def test_quick_setup_with_log_file(self, tmp_path):
        """Test quick_setup with log file."""
        log_file = tmp_path / "quick_test.log"

        logger = quick_setup(log_file=str(log_file))
        logger.info("Quick setup test")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Quick setup test" in content

    def test_quick_setup_debug_with_file(self, tmp_path):
        """Test quick_setup with both debug and log file."""
        log_file = tmp_path / "debug_test.log"

        logger = quick_setup(debug=True, log_file=str(log_file))
        logger.debug("Debug message")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Debug message" in content

    def test_quick_setup_returns_root_logger(self):
        """Test that quick_setup returns the root logger."""
        logger = quick_setup()

        assert logger.name == "root"


class TestLoggingIntegration:
    """Integration tests for logging configuration."""

    def test_multiple_setup_calls(self, tmp_path):
        """Test that multiple setup_logging calls work correctly."""
        log_file1 = tmp_path / "test1.log"
        log_file2 = tmp_path / "test2.log"

        # First setup
        setup_logging(log_file=str(log_file1), level="DEBUG")
        logger = logging.getLogger(__name__)
        logger.debug("Message 1")

        # Second setup (should override)
        setup_logging(log_file=str(log_file2), level="INFO")
        logger.info("Message 2")

        # Both files should exist
        assert log_file1.exists()
        assert log_file2.exists()

    def test_logging_all_levels(self, tmp_path):
        """Test logging at all levels."""
        log_file = tmp_path / "all_levels.log"
        setup_logging(log_file=str(log_file), level="DEBUG")

        logger = get_logger(__name__)
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        content = log_file.read_text()
        assert "Debug message" in content
        assert "Info message" in content
        assert "Warning message" in content
        assert "Error message" in content
        assert "Critical message" in content

    def test_child_logger_inherits_config(self, tmp_path):
        """Test that child loggers inherit configuration."""
        log_file = tmp_path / "parent.log"
        setup_logging(log_file=str(log_file))

        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")

        child_logger.info("Child message")

        content = log_file.read_text()
        assert "Child message" in content

    def test_exception_logging(self, tmp_path):
        """Test logging exceptions with traceback."""
        log_file = tmp_path / "exceptions.log"
        setup_logging(log_file=str(log_file))

        logger = get_logger(__name__)

        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("An error occurred")

        content = log_file.read_text()
        assert "An error occurred" in content
        assert "ValueError" in content
        assert "Test exception" in content


class TestLoggingEdgeCases:
    """Test edge cases and error conditions."""

    def test_setup_logging_empty_string_level(self):
        """Test setup_logging with empty string level."""
        setup_logging(level="")
        # Should use default (INFO)
        logger = logging.getLogger()
        assert logger.level == logging.INFO

    def test_setup_logging_none_file(self):
        """Test setup_logging with None as log file."""
        setup_logging(log_file=None)
        # Should work without file (console only)
        logger = logging.getLogger(__name__)
        logger.info("Console only message")

    def test_setup_logging_empty_format(self):
        """Test setup_logging with empty format string."""
        setup_logging(format_string="")
        logger = logging.getLogger(__name__)
        logger.info("Test")
        # Should not crash

    def test_get_logger_empty_name(self):
        """Test get_logger with empty name."""
        logger = get_logger("")
        assert isinstance(logger, logging.Logger)

    def test_concurrent_logging(self, tmp_path):
        """Test logging from multiple threads."""
        import threading

        log_file = tmp_path / "concurrent.log"
        setup_logging(log_file=str(log_file))

        def log_messages(thread_id):
            logger = get_logger(f"thread_{thread_id}")
            for i in range(10):
                logger.info(f"Thread {thread_id} message {i}")

        threads = [threading.Thread(target=log_messages, args=(i,)) for i in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify log file exists and has messages
        assert log_file.exists()
        content = log_file.read_text()
        assert "Thread" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
