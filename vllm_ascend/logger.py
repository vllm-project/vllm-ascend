# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logging configuration for vLLM-Ascend.

Approach A: Minimal — replace vLLM handler formatters to add
[vllm-ascend] [module] prefix. No module code changes required.
Module identification is inferred from record.pathname.
"""

import logging

from vllm.logging_utils import ColoredFormatter, NewLineFormatter

_FORMAT = "%(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


def _is_ascend_module(pathname: str) -> bool:
    if not pathname:
        return False
    return "vllm_ascend" in pathname.replace("\\", "/")


def _infer_module_name(pathname: str) -> str:
    """Infer module name from the file path of the log caller."""
    if not pathname:
        return "core"
    parts = pathname.replace("\\", "/").split("/")
    try:
        idx = parts.index("vllm_ascend")
        if idx + 1 >= len(parts):
            return "core"
        item = parts[idx + 1]
        if idx + 2 >= len(parts):
            return item[:-3] if item.endswith(".py") else item
        return item
    except ValueError:
        return "core"


def _format_with_ascend_prefix(self, record, super_format):
    if not _is_ascend_module(record.pathname):
        return super_format(record)
    module = _infer_module_name(record.pathname)
    if record.filename == module + ".py":
        prefix = "[vllm-ascend]"
    else:
        prefix = f"[vllm-ascend] [{module}]"
    orig_msg = record.msg
    orig_args = record.args
    try:
        record.msg = f"{prefix} - {record.getMessage()}"
        record.args = ()
        return super_format(record)
    finally:
        record.msg = orig_msg
        record.args = orig_args


class AscendFormatter(NewLineFormatter):
    """Extends NewLineFormatter with [vllm-ascend] prefix and module name."""

    def format(self, record):
        return _format_with_ascend_prefix(self, record, super().format)


class AscendColoredFormatter(ColoredFormatter):
    """Extends ColoredFormatter with [vllm-ascend] prefix and module name."""

    def format(self, record):
        return _format_with_ascend_prefix(self, record, super().format)


def _patch_handler(handler: logging.Handler) -> None:
    if isinstance(handler.formatter, ColoredFormatter):
        handler.formatter = AscendColoredFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)
    elif isinstance(handler.formatter, NewLineFormatter):
        handler.formatter = AscendFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)


def _patch_vllm_formatter() -> None:
    """Replace vLLM handler formatters with ascend-aware versions.

    Handlers added after this call are also patched via addHandler monkey-patch,
    making the patch robust against import order.
    """
    vllm_logger = logging.getLogger("vllm")

    for handler in vllm_logger.handlers:
        _patch_handler(handler)

    _original_add_handler = vllm_logger.addHandler

    def _patched_add_handler(handler: logging.Handler) -> None:
        _patch_handler(handler)
        _original_add_handler(handler)

    vllm_logger.addHandler = _patched_add_handler  # type: ignore[method-assign,assignment]


_patch_vllm_formatter()
