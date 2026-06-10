# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logging configuration for vLLM-Ascend.

Approach B: Thin wrapper — provide init_logger that wraps vLLM's init_logger,
registers a "vllm_ascend" namespace handler, and replaces formatters to add
[vllm-ascend] [module] prefix. Compatible with vLLM log system evolution.
"""

import logging

from vllm.logger import init_logger as _vllm_init_logger
from vllm.logging_utils import ColoredFormatter, NewLineFormatter

_FORMAT = (
    "%(levelname)s %(asctime)s "
    "[%(fileinfo)s:%(lineno)d] %(message)s"
)
_DATE_FORMAT = "%m-%d %H:%M:%S"

_ascend_handler_registered = False


def _infer_module_name(logger_name: str) -> str:
    """Infer module name from the logger name."""
    if not logger_name.startswith("vllm_ascend."):
        return "core"
    parts = logger_name.split(".")
    if len(parts) < 2:
        return "core"
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if len(parts) >= 2:
        return parts[1]
    return "core"


class AscendFormatter(NewLineFormatter):
    """Extends NewLineFormatter with [vllm-ascend] prefix and module name."""

    def format(self, record):
        module = _infer_module_name(record.name)
        original = record.getMessage()
        record.msg = f"[vllm-ascend] [{module}] - {original}"
        record.args = ()
        return super().format(record)


class AscendColoredFormatter(ColoredFormatter):
    """Extends ColoredFormatter with [vllm-ascend] prefix and module name."""

    def format(self, record):
        module = _infer_module_name(record.name)
        original = record.getMessage()
        record.msg = f"[vllm-ascend] [{module}] - {original}"
        record.args = ()
        return super().format(record)


def _configure_ascend_handler() -> None:
    """Register handler for 'vllm_ascend' namespace with ascend formatters.

    This mirrors vLLM's approach: a dedicated handler for the ascend namespace
    so that loggers created via init_logger(__name__) under 'vllm_ascend.*'
    have a valid handler and don't silently drop messages.
    """
    import vllm.envs as envs

    ascend_logger = logging.getLogger("vllm_ascend")
    if ascend_logger.handlers:
        return

    use_color = _use_color()
    handler = logging.StreamHandler(envs.VLLM_LOGGING_STREAM)
    handler.setLevel(envs.VLLM_LOGGING_LEVEL)
    if use_color:
        handler.setFormatter(AscendColoredFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))
    else:
        handler.setFormatter(AscendFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))
    ascend_logger.addHandler(handler)
    ascend_logger.setLevel(envs.VLLM_LOGGING_LEVEL)
    ascend_logger.propagate = False


def _use_color() -> bool:
    import sys

    import vllm.envs as envs

    if envs.NO_COLOR or envs.VLLM_LOGGING_COLOR == "0":
        return False
    if envs.VLLM_LOGGING_COLOR == "1":
        return True
    if envs.VLLM_LOGGING_STREAM == "ext://sys.stdout":
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    elif envs.VLLM_LOGGING_STREAM == "ext://sys.stderr":
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    return False


def init_logger(name: str) -> logging.Logger:
    """Initialize a vLLM-Ascend logger.

    Wraps vLLM's init_logger to get _once methods, then registers the
    ascend handler and replaces formatters to add the [vllm-ascend] prefix
    and module identification.

    Args:
        name: Logger name, usually __name__

    Returns:
        Configured logger with [vllm-ascend] [module] prefix
    """
    global _ascend_handler_registered
    if not _ascend_handler_registered:
        _configure_ascend_handler()
        _ascend_handler_registered = True

    logger = _vllm_init_logger(name)

    for handler in logger.handlers:
        if isinstance(handler.formatter, ColoredFormatter):
            handler.formatter = AscendColoredFormatter(
                fmt=_FORMAT, datefmt=_DATE_FORMAT
            )
        elif isinstance(handler.formatter, NewLineFormatter):
            handler.formatter = AscendFormatter(
                fmt=_FORMAT, datefmt=_DATE_FORMAT
            )

    return logger