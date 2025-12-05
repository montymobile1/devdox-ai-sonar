"""
Centralized logging configuration for DevDox AI Sonar.

Usage:
    from devdox_ai_sonar.logging_config import setup_logging

    # Basic setup
    setup_logging()

    # With custom log file
    setup_logging(log_file='my_app.log')

    # With DEBUG level
    setup_logging(level='DEBUG')

    # Environment variables:
    # export LOG_LEVEL=DEBUG
    # export LOG_FILE=devdox_sonar.log
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, List
from logging.handlers import RotatingFileHandler


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10_485_760,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               If None, reads from LOG_LEVEL environment variable (default: INFO)
        log_file: Path to log file. If None, reads from LOG_FILE environment variable.
                  If not set, logs only to console.
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        format_string: Custom format string for log messages

    Examples:
        # Console only
        setup_logging()

        # Console + file
        setup_logging(log_file='app.log')

        # Debug level with file
        setup_logging(level='DEBUG', log_file='debug.log')

        # Using environment variables
        # export LOG_LEVEL=DEBUG
        # export LOG_FILE=sonar.log
        setup_logging()
    """
    # Get level from parameter or environment variable
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    else:
        level = level.upper()

    # Validate level
    numeric_level = getattr(logging, level, logging.INFO)

    # Get log file from parameter or environment variable
    if log_file is None:
        log_file = os.getenv("LOG_FILE")

    # Default format string
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Create handlers list
    handlers: List[logging.Handler] = []

    # Console handler (always included)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    handlers.append(console_handler)

    # File handler (if log file specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use RotatingFileHandler to prevent huge log files
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Set levels for noisy third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {level}")
    if log_file:
        logger.info(
            f"Logging to file: {log_file} (max size: {max_bytes / 1024 / 1024:.1f}MB, {backup_count} backups)"
        )
    else:
        logger.info("Logging to console only")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Hello world")
    """
    return logging.getLogger(name)


# Convenience function for quick setup in scripts
def quick_setup(debug: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """
    Quick logging setup for scripts and testing.

    Args:
        debug: If True, set level to DEBUG, otherwise INFO
        log_file: Optional log file path

    Returns:
        Root logger instance

    Example:
        logger = quick_setup(debug=True, log_file='test.log')
        logger.debug("Debug message")
    """
    level = "DEBUG" if debug else "INFO"
    setup_logging(level=level, log_file=log_file)
    return logging.getLogger()


if __name__ == "__main__":
    # Demo usage
    print("Demo: Logging Configuration\n")

    # Example 1: Console only
    print("Example 1: Console only")
    setup_logging(level="DEBUG")
    logger = get_logger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    print("\n" + "=" * 60 + "\n")

    # Example 2: Console + File
    print("Example 2: Console + File (demo.log)")
    setup_logging(level="INFO", log_file="demo.log")
    logger = get_logger(__name__)
    logger.info("This message goes to both console and file")
    logger.warning("Check demo.log to see this message")

    print("\n✅ Check demo.log file")
