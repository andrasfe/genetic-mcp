"""
Logging configuration for the Genetic MCP server.
This module provides centralized logging configuration with:
- Structured logging with context
- Different log levels for different components
- Optional file logging
- Colored console output for better readability
"""

import logging
import os
import sys
from pathlib import Path

# ANSI color codes for console output
COLORS = {
    'DEBUG': '\033[36m',    # Cyan
    'INFO': '\033[32m',     # Green
    'WARNING': '\033[33m',  # Yellow
    'ERROR': '\033[31m',    # Red
    'CRITICAL': '\033[35m', # Magenta
    'RESET': '\033[0m'      # Reset
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output."""

    def format(self, record: logging.LogRecord) -> str:
        # Add color to the level name
        levelname = record.levelname
        if levelname in COLORS and sys.stderr.isatty():
            record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"

        # Format the message
        result = super().format(record)

        # Reset levelname for other handlers
        record.levelname = levelname
        return result


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    component: str | None = None
) -> logging.Logger:
    """
    Set up logging configuration for a component.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        component: Component name for the logger

    Returns:
        Configured logger instance
    """
    # Get logger
    logger_name = f"genetic_mcp.{component}" if component else "genetic_mcp"
    logger = logging.getLogger(logger_name)

    # Don't add handlers if already configured
    if logger.handlers:
        return logger

    # Set level from environment or parameter
    env_level = os.getenv("GENETIC_MCP_LOG_LEVEL", level).upper()
    logger.setLevel(getattr(logging, env_level))

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stderr)
    console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_formatter = ColoredFormatter(console_format, datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file or os.getenv("GENETIC_MCP_LOG_FILE"):
        file_path = log_file or os.getenv("GENETIC_MCP_LOG_FILE")
        if file_path:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(file_path)
            file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            file_formatter = logging.Formatter(file_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


def get_logger(component: str) -> logging.Logger:
    """
    Get a logger for a specific component.

    Args:
        component: Component name

    Returns:
        Logger instance
    """
    return logging.getLogger(f"genetic_mcp.{component}")


# Structured logging helpers
def log_operation(logger: logging.Logger, operation: str, **kwargs: object) -> None:
    """Log an operation with structured context."""
    context = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"[{operation}] {context}")


def log_error(logger: logging.Logger, operation: str, error: Exception, **kwargs: object) -> None:
    """Log an error with structured context."""
    context = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.error(f"[{operation}] {context} error={type(error).__name__}: {str(error)}")


def log_performance(logger: logging.Logger, operation: str, duration: float, **kwargs: object) -> None:
    """Log performance metrics."""
    context = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"[{operation}] duration={duration:.3f}s {context}")


# Initialize root logger
root_logger = setup_logging(
    level=os.getenv("GENETIC_MCP_LOG_LEVEL", "INFO"),
    log_file=os.getenv("GENETIC_MCP_LOG_FILE")
)
