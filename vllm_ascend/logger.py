# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logging configuration for vLLM-Ascend.

Provides two logging mechanisms:
1. Console: A dedicated handler on the vllm_ascend logger with
   [vllm-ascend] [module] prefix. No modification to vLLM's global
   logging state — safe for upstream tests and multiprocessing.
2. File: A rotating file handler on both vllm and vllm_ascend loggers,
   capturing all logs with Ascend formatting.
"""

import logging
import os
import sys
from datetime import datetime
from types import MethodType

from vllm import envs
from vllm.logger import _METHODS_TO_PATCH
from vllm.logging_utils import ColoredFormatter, NewLineFormatter

_FORMAT = "%(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

_LOG_DIR = os.path.join(os.path.expanduser("~"), "ascend", "log", "vllm_ascend")
_LOG_MAX_BYTES = 20 * 1024 * 1024


def init_logger_ascend(name: str) -> logging.Logger:
    """Return a stdlib logger for Ascend / runtime_guard modules.

    Uses ``logging.getLogger`` directly (plus vLLM's once-method patching)
    instead of delegating to ``vllm.logger.init_logger``. The Ascend logging
    framework (UCM) monkey-patches ``vllm.logger.init_logger`` to return a UCM
    logger whose level/format live in a C extension and bypass Python
    ``setLevel``/``Formatter``. Bypassing that patch keeps Ascend loggers on
    the stdlib tree, so they stay controllable via ``setLevel`` and
    ``VLLM_LOGGING_LEVEL``.
    """
    logger = logging.getLogger(name)
    for method_name, method in _METHODS_TO_PATCH.items():
        setattr(logger, method_name, MethodType(method, logger))
    return logger


def _resolve_log_level(level: str) -> int:
    return getattr(logging, str(level).upper(), logging.INFO)


def _normalize_module_logger(entry: str) -> str:
    name = str(entry).strip().strip(".")
    if not name:
        return ""
    if name == "vllm_ascend" or name.startswith("vllm_ascend.") or name.startswith("vllm."):
        return name
    return f"vllm_ascend.{name}"


def _iter_logger_names(prefix: str | None = None) -> list[str]:
    names: list[str] = []
    for name in list(logging.Logger.manager.loggerDict):
        if not isinstance(name, str):
            continue
        if prefix is None:
            names.append(name)
            continue
        if name == prefix or name.startswith(prefix + "."):
            names.append(name)
    return names


def _set_logger_tree_level(prefix: str, level: int) -> None:
    logging.getLogger(prefix).setLevel(level)
    dotted = prefix + "."
    for name in _iter_logger_names():
        if name == prefix or name.startswith(dotted):
            logging.getLogger(name).setLevel(level)


def _merge_module_level_overrides(
    debug_modules: list[str] | None,
    module_levels: dict[str, str] | None,
) -> dict[str, str]:
    merged: dict[str, str] = {}
    for entry in debug_modules or ():
        prefix = _normalize_module_logger(entry)
        if prefix:
            merged[prefix] = "DEBUG"
    for raw_name, raw_level in (module_levels or {}).items():
        prefix = _normalize_module_logger(str(raw_name))
        if prefix:
            merged[prefix] = str(raw_level).strip().upper()
    return merged


_last_applied_ascend_log: tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]] | None = None


def apply_ascend_log_level(
    level: str = "INFO",
    debug_modules: list[str] | None = None,
    module_levels: dict[str, str] | None = None,
) -> None:
    """Apply default level and optional per-module overrides for hot-reload / actions."""
    global _last_applied_ascend_log

    if logging.root.manager.disable:
        stuck_at = logging.getLevelName(logging.root.manager.disable)
        logging.disable(logging.NOTSET)
        logging.getLogger("vllm_ascend.logger").warning(
            "[ascend_log] cleared a stuck global logging.disable(%s)",
            stuck_at,
        )

    configure_ascend_logging()
    default_level = _resolve_log_level(level)
    debug_list = list(debug_modules or ())
    overrides = _merge_module_level_overrides(debug_list, module_levels)
    level_key = str(level).upper()
    debug_key = tuple(debug_list)
    modules_key = tuple(sorted(overrides.items()))
    needs_debug = default_level <= logging.DEBUG or any(
        _resolve_log_level(lvl) <= logging.DEBUG for lvl in overrides.values()
    )
    announce = _last_applied_ascend_log != (level_key, debug_key, modules_key)

    logging.root.setLevel(default_level)
    ascend = logging.getLogger("vllm_ascend")
    ascend.setLevel(default_level)
    ascend.propagate = False
    for h in ascend.handlers:
        if h.level > logging.DEBUG:
            h.setLevel(logging.DEBUG)

    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(default_level)
    for name in _iter_logger_names("vllm_ascend"):
        logging.getLogger(name).setLevel(default_level)
    for name in _iter_logger_names("vllm"):
        if name != "vllm":
            logging.getLogger(name).setLevel(default_level)

    for prefix, level_name in overrides.items():
        _set_logger_tree_level(prefix, _resolve_log_level(level_name))

    if needs_debug:
        for h in logging.root.handlers:
            if h.level > logging.DEBUG:
                h.setLevel(logging.DEBUG)
        for h in vllm_logger.handlers:
            if h.level > logging.DEBUG:
                h.setLevel(logging.DEBUG)

    _last_applied_ascend_log = (level_key, debug_key, modules_key)
    if announce:
        logging.getLogger("vllm_ascend.logger").info(
            "[ascend_log] applied level=%s debug=%s modules=%s",
            level_key,
            debug_list,
            dict(overrides),
        )


def _use_color() -> bool:
    """Determine if colored output should be used."""
    if envs.NO_COLOR or envs.VLLM_LOGGING_COLOR == "0":
        return False
    if envs.VLLM_LOGGING_COLOR == "1":
        return True
    if envs.VLLM_LOGGING_STREAM == "ext://sys.stdout":
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    elif envs.VLLM_LOGGING_STREAM == "ext://sys.stderr":
        return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    return False


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


class RotatingAscendFileHandler(logging.FileHandler):
    """FileHandler that rotates log files when they exceed a size limit.

    Naming convention:
        vllm_ascend_{timestamp}_{pid}.log          <- first file
        vllm_ascend_{timestamp}_{pid}_002.log       <- second file
        vllm_ascend_{timestamp}_{pid}_003.log       <- third file
    """

    def __init__(self, log_dir: str, max_bytes: int = _LOG_MAX_BYTES) -> None:
        self._log_dir = log_dir
        self._max_bytes = max_bytes
        self._sequence = 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._base_name = f"vllm_ascend_{timestamp}_{os.getpid()}"
        log_file = os.path.join(log_dir, f"{self._base_name}.log")
        super().__init__(log_file, encoding="utf-8")

    def emit(self, record) -> None:
        try:
            if self.stream is not None and os.path.isfile(self.baseFilename):
                if os.path.getsize(self.baseFilename) >= self._max_bytes:
                    self._rotate()
        except OSError:
            pass
        super().emit(record)

    def _rotate(self) -> None:
        self.stream.close()
        self.stream = None  # type: ignore[assignment]
        self._sequence += 1
        new_file = os.path.join(self._log_dir, f"{self._base_name}_{self._sequence:03d}.log")
        self.baseFilename = new_file
        self.stream = self._open()


_file_logging_configured = False
_file_handler: logging.Handler | None = None


def _setup_file_logging(log_dir: str | None = None) -> None:
    global _file_logging_configured, _file_handler
    if _file_logging_configured:
        return
    target_dir = log_dir or _LOG_DIR
    os.makedirs(target_dir, exist_ok=True)
    file_handler = RotatingAscendFileHandler(target_dir)
    vllm_logger = logging.getLogger("vllm")
    ascend_logger = logging.getLogger("vllm_ascend")
    log_level = logging.INFO
    if vllm_logger.handlers:
        log_level = vllm_logger.handlers[0].level
    file_handler.setLevel(log_level)
    file_handler.setFormatter(AscendFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))
    vllm_logger.addHandler(file_handler)
    ascend_logger.addHandler(file_handler)
    _file_handler = file_handler
    _file_logging_configured = True


def configure_ascend_file_logging() -> None:
    global _file_logging_configured, _file_handler
    log_dir = _LOG_DIR
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        ascend_config = get_ascend_config()
        log_dir = ascend_config.ascend_log_path
    except Exception:
        pass
    if log_dir != _LOG_DIR:
        vllm_logger = logging.getLogger("vllm")
        ascend_logger = logging.getLogger("vllm_ascend")
        if _file_handler is not None:
            vllm_logger.removeHandler(_file_handler)
            ascend_logger.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None
        _file_logging_configured = False
    _setup_file_logging(log_dir)


def configure_ascend_logging() -> None:
    """Configure vllm_ascend logger with Ascend formatters.

    Creates a dedicated handler for the vllm_ascend logger namespace,
    avoiding any modification to vLLM's global logging state.
    This approach is safe for upstream tests and multiprocessing.
    """
    ascend_logger = logging.getLogger("vllm_ascend")
    if ascend_logger.handlers:
        return

    # Parse stream parameter
    if envs.VLLM_LOGGING_STREAM == "ext://sys.stdout":
        stream = sys.stdout
    elif envs.VLLM_LOGGING_STREAM == "ext://sys.stderr":
        stream = sys.stderr
    else:
        stream = sys.stderr

    handler = logging.StreamHandler(stream)
    handler.setLevel(envs.VLLM_LOGGING_LEVEL)

    if _use_color():
        handler.setFormatter(AscendColoredFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))
    else:
        handler.setFormatter(AscendFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))

    ascend_logger.addHandler(handler)
    ascend_logger.setLevel(envs.VLLM_LOGGING_LEVEL)
    ascend_logger.propagate = False
