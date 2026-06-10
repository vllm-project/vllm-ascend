# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logging configuration for vLLM-Ascend.

Approach A: Minimal — replace vLLM handler formatters to add
[vllm-ascend] [module] prefix. No module code changes required.
Module identification is inferred from record.pathname.
"""

import logging

from vllm.logging_utils import ColoredFormatter, NewLineFormatter

_FORMAT = (
    "%(levelname)s %(asctime)s "
    "[%(fileinfo)s:%(lineno)d] %(message)s"
)
_DATE_FORMAT = "%m-%d %H:%M:%S"


def _infer_module_name(pathname: str) -> str:
    """Infer module name from the file path of the log caller."""
    parts = pathname.replace("\\", "/").split("/")
    try:
        idx = parts.index("vllm_ascend")
        if idx + 1 >= len(parts):
            return "core"
        if idx + 2 >= len(parts):
            return "core"
        return parts[idx + 1]
    except ValueError:
        return "core"


class AscendFormatter(NewLineFormatter):
    """Extends NewLineFormatter with [vllm-ascend] prefix and module name."""

    def format(self, record):
        module = _infer_module_name(record.pathname)
        original = record.getMessage()
        record.msg = f"[vllm-ascend] [{module}] - {original}"
        record.args = ()
        return super().format(record)


class AscendColoredFormatter(ColoredFormatter):
    """Extends ColoredFormatter with [vllm-ascend] prefix and module name."""

    def format(self, record):
        module = _infer_module_name(record.pathname)
        original = record.getMessage()
        record.msg = f"[vllm-ascend] [{module}] - {original}"
        record.args = ()
        return super().format(record)


def _patch_vllm_formatter() -> None:
    """Replace vLLM handler formatters with ascend-aware versions."""
    vllm_logger = logging.getLogger("vllm")
    for handler in vllm_logger.handlers:
        if isinstance(handler.formatter, ColoredFormatter):
            handler.formatter = AscendColoredFormatter(
                fmt=_FORMAT, datefmt=_DATE_FORMAT
            )
        elif isinstance(handler.formatter, NewLineFormatter):
            handler.formatter = AscendFormatter(
                fmt=_FORMAT, datefmt=_DATE_FORMAT
            )


_patch_vllm_formatter()