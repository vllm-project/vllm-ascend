# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""
vLLM-Ascend Logging Configuration - Lightweight Extension Approach

This module provides lightweight extensions to vLLM's logging system:
1. AscendFormatter: Adds [vllm-ascend] prefix and module identification
2. AscendColoredFormatter: Colored version with ANSI codes
3. configure_ascend_logging(): Apply Ascend formatters to vLLM logging

Usage:
    # In your module
    from vllm.logger import init_logger  # Use vLLM's init_logger directly

    logger = init_logger(__name__)

    # At application startup (e.g., in platform.py or __init__.py)
    from vllm_ascend.logger import configure_ascend_logging
    configure_ascend_logging()
"""

import logging
import sys
from functools import lru_cache
from typing import Literal

from vllm import envs

# Re-export vLLM's init_logger - no need to wrap!
from vllm.logger import init_logger
from vllm.logging_utils.formatter import ColoredFormatter, NewLineFormatter

LogScope = Literal["process", "global", "local"]

# Default format - we'll add [vllm-ascend] prefix in the formatter
_FORMAT = f"{envs.VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


def _use_color() -> bool:
    """Determine if colored output should be used."""
    if envs.NO_COLOR or envs.VLLM_LOGGING_COLOR == "0":
        return False
    if envs.VLLM_LOGGING_COLOR == "1":
        return True
    if envs.VLLM_LOGGING_STREAM == "ext://sys.stdout":  # stdout
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    elif envs.VLLM_LOGGING_STREAM == "ext://sys.stderr":  # stderr
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    return False


def _infer_module_name(logger_name: str) -> str:
    """
    Infer module category from logger name.

    Args:
        logger_name: Usually __name__ from the calling module

    Returns:
        Module category name for display in logs

    Examples:
        >>> _infer_module_name("vllm_ascend.attention.mla_v1")
        'attention'
        >>> _infer_module_name("vllm_ascend.worker")
        'worker'
        >>> _infer_module_name("vllm.attention")
        'core'  # Non-vllm_ascend modules
    """
    if not logger_name:
        return "core"

    # For vllm_ascend modules, extract the first component after vllm_ascend
    if logger_name.startswith("vllm_ascend."):
        parts = logger_name.split(".")
        if len(parts) >= 2:
            module = parts[1]
            # Handle submodules like vllm_ascend.attention.mla_v1 -> attention
            if module in (
                "ops",
                "distributed",
                "compilation",
                "quantization",
                "model_loader",
                "eplb",
                "worker",
                "core",
            ):
                return module
            # For __init__ modules, use parent
            if module == "__init__":
                return parts[0] if len(parts) > 0 else "core"
            return module
        return "core"

    # For non-vllm_ascend modules (e.g., vllm.*, torch.*)
    return "core"


class AscendFormatter(NewLineFormatter):
    """
    Extends vLLM's NewLineFormatter with [vllm-ascend] prefix and module identification.

    Log format:
        (VLLM) INFO 06-09 10:30:15 [vllm-ascend] [attention.py:123] Message

    The [vllm-ascend] prefix helps distinguish Ascend-specific logs from vLLM core logs.
    Module identification helps identify which vLLM-Ascend component generated the log.
    """

    def __init__(self, fmt=None, datefmt=None, style="%"):
        if fmt is None:
            fmt = _FORMAT
        if datefmt is None:
            datefmt = _DATE_FORMAT
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        # Infer module name and add [vllm-ascend] prefix to message
        module_name = _infer_module_name(record.name)
        original_msg = record.getMessage()
        record.msg = f"[vllm-ascend] [{module_name}] - {original_msg}"
        record.args = ()

        # Call parent formatter to handle the actual formatting
        return super().format(record)


class AscendColoredFormatter(ColoredFormatter):
    """
    Colored version of AscendFormatter with ANSI color codes.

    Inherits from ColoredFormatter which already handles:
    - Color injection for timestamp and file info
    - Dynamic color selection for log levels
    - Multi-line message alignment

    We just need to ensure module identification works correctly.
    """

    def __init__(self, fmt=None, datefmt=None, style="%"):
        if fmt is None:
            fmt = _FORMAT
        if datefmt is None:
            datefmt = _DATE_FORMAT
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        # Infer module name and add [vllm-ascend] prefix to message
        module_name = _infer_module_name(record.name)
        original_msg = record.getMessage()
        record.msg = f"[vllm-ascend] [{module_name}] - {original_msg}"
        record.args = ()

        # Call parent formatter (ColoredFormatter) to handle colors
        return super().format(record)


def configure_ascend_logging() -> None:
    """
    Configure vLLM logging to use Ascend formatters.

    This function:
    1. Configures vLLM's default logging if not already configured
    2. Adds AscendFormatter/AscendColoredFormatter to vllm_ascend logger

    Should be called once at application startup, before any logging occurs.

    Example:
        from vllm_ascend.logger import configure_ascend_logging
        configure_ascend_logging()

        # Now all vLLM and vllm-ascend logs will have [vllm-ascend] prefix
        logger = init_logger(__name__)
        logger.info("This will show as: (VLLM) INFO [...] [vllm-ascend] [module] message")
    """
    # Import vllm.logger to ensure vLLM's logging is properly initialized.
    # This triggers vLLM's _configure_vllm_root_logger() which configures
    # the vLLM logging system. This is essential for loggers like elastic.logger
    # to work correctly.
    import logging.config

    import vllm.logger  # noqa: F401

    # Configure vllm_ascend logger directly without affecting other loggers.
    # vLLM's logging is already configured during import, so we don't need
    # to call logging.config.dictConfig() here. This avoids resetting other
    # loggers (e.g., elastic.logger) which could break their functionality.
    ascend_logger = logging.getLogger("vllm_ascend")

    # Only add handler if not already configured
    if not ascend_logger.handlers:
        # Parse stream parameter
        if envs.VLLM_LOGGING_STREAM == "ext://sys.stdout":
            stream = sys.stdout
        elif envs.VLLM_LOGGING_STREAM == "ext://sys.stderr":
            stream = sys.stderr
        else:
            stream = sys.stderr  # Default value

        # Create handler with AscendFormatter
        handler = logging.StreamHandler(stream)
        handler.setLevel(envs.VLLM_LOGGING_LEVEL)

        # Use colored formatter if color is enabled
        if _use_color():
            handler.setFormatter(AscendColoredFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))
        else:
            handler.setFormatter(AscendFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))

        ascend_logger.addHandler(handler)
        ascend_logger.setLevel(envs.VLLM_LOGGING_LEVEL)
        ascend_logger.propagate = False


# Convenience function for module initialization
def setup_module_logger(name: str) -> logging.Logger:
    """
    Setup logger for a module with Ascend formatting.

    This is a convenience wrapper that:
    1. Calls configure_ascend_logging() once (cached)
    2. Returns a logger using vLLM's init_logger

    Usage:
        # At module level
        logger = setup_module_logger(__name__)

        # Or simply:
        from vllm.logger import init_logger
        logger = init_logger(__name__)
        # And ensure configure_ascend_logging() is called at startup

    Args:
        name: Module name, typically __name__

    Returns:
        Configured logger instance
    """
    # Configure logging on first call (cached)
    _configure_once()

    # Use vLLM's init_logger directly
    return init_logger(name)


@lru_cache(maxsize=1)
def _configure_once() -> None:
    """Call configure_ascend_logging() only once."""
    configure_ascend_logging()


# Note: Logging configuration is now deferred to NPUPlatform.__init__()
# to avoid affecting other modules during import. This prevents issues
# where configure_ascend_logging() might interfere with logger initialization
# in other modules (e.g., elastic.logger).

# Transformers uses httpx to access the Hugging Face Hub. httpx is quite verbose,
# so we set its logging level to WARNING when vLLM's logging level is INFO.
if envs.VLLM_LOGGING_LEVEL == "INFO":
    logging.getLogger("httpx").setLevel(logging.WARNING)

# Optional: Add debug/info/warning once methods if needed
# But vLLM already provides these via init_logger, so we don't need to duplicate
