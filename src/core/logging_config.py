"""
Production-ready logging configuration with rotation and monitoring.

Features:
- Environment-based log levels
- File rotation (size and time-based)
- JSON formatting for production
- Structured logging with context
- Performance metrics logging
"""

import logging
import logging.handlers
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging in production.

    Outputs logs in JSON format for easy parsing by log aggregators
    (Elasticsearch, Splunk, CloudWatch, etc.)
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'execution_time'):
            log_data['execution_time'] = record.execution_time

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for development console output.
    """

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    environment: str = 'development',
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging based on environment.

    Args:
        environment: 'development', 'production', or 'testing'
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (default: logs/app.log)

    Returns:
        Configured root logger

    Features:
        Development:
        - Colored console output
        - DEBUG level by default
        - Human-readable format

        Production:
        - JSON formatted logs
        - INFO level by default
        - Rotating file handler (100MB max, 10 backups)
        - Daily rotation for audit logs

        Testing:
        - Minimal logging (WARNING+)
        - Memory handler for assertions
    """
    # Get log level from environment or parameter
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'DEBUG' if environment == 'development' else 'INFO')

    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Get or create root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Remove existing handlers
    logger.handlers.clear()

    # ==================== DEVELOPMENT ====================
    if environment == 'development':
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)

        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler for debugging
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=3
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    # ==================== PRODUCTION ====================
    elif environment == 'production':
        # Rotating file handler with JSON formatting
        log_file = log_file or 'logs/app.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=100*1024*1024,  # 100MB
            backupCount=10
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

        # Daily rotating handler for audit logs
        audit_handler = logging.handlers.TimedRotatingFileHandler(
            'logs/audit.log',
            when='midnight',
            interval=1,
            backupCount=30  # Keep 30 days
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(JSONFormatter())
        logger.addHandler(audit_handler)

        # Console handler (JSON for container logs)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only warnings+ to console
        console_handler.setFormatter(JSONFormatter())
        logger.addHandler(console_handler)

    # ==================== TESTING ====================
    elif environment == 'testing':
        # Minimal logging for tests
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    logger.info(f"Logging configured for {environment} environment (level: {log_level})")
    return logger


def log_performance(
    logger: logging.Logger,
    operation: str,
    execution_time: float,
    success: bool,
    **kwargs
):
    """
    Log performance metrics in a structured way.

    Args:
        logger: Logger instance
        operation: Operation name (e.g., "artwork_analysis")
        execution_time: Time taken in seconds
        success: Whether operation succeeded
        **kwargs: Additional context (user_id, image_size, etc.)

    Example:
        log_performance(
            logger,
            "artwork_analysis",
            execution_time=6.8,
            success=True,
            image_size=1024,
            from_cache=False
        )
    """
    log_data = {
        'operation': operation,
        'execution_time': round(execution_time, 3),
        'success': success,
        **kwargs
    }

    # Add structured data as extra
    extra = {'execution_time': execution_time}

    if success:
        logger.info(f"Performance: {operation}", extra=extra)
    else:
        logger.warning(f"Performance (failed): {operation}", extra=extra)

    # Log to metrics file for monitoring
    metrics_file = Path('logs/metrics.jsonl')
    try:
        with open(metrics_file, 'a') as f:
            log_data['timestamp'] = datetime.utcnow().isoformat() + 'Z'
            f.write(json.dumps(log_data) + '\n')
    except Exception as e:
        logger.error(f"Failed to write metrics: {e}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance

    Usage:
        from src.core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Starting analysis...")
    """
    return logging.getLogger(name)


# ==================== MONITORING HELPERS ====================

class PerformanceLogger:
    """
    Context manager for automatic performance logging.

    Usage:
        with PerformanceLogger(logger, "artwork_analysis") as perf:
            result = analyze_artwork(image)
            perf.add_context(image_size=len(image))
    """

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
        self.context = {}
        self.success = True

    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        execution_time = time.time() - self.start_time

        if exc_type is not None:
            self.success = False
            self.context['error'] = str(exc_val)

        log_performance(
            self.logger,
            self.operation,
            execution_time,
            self.success,
            **self.context
        )

        return False  # Don't suppress exceptions

    def add_context(self, **kwargs):
        """Add additional context to performance log."""
        self.context.update(kwargs)
