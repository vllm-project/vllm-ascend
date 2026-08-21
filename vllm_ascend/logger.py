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
import time
from datetime import datetime

from vllm import envs
from vllm.logger import init_logger
from vllm.logging_utils import ColoredFormatter, NewLineFormatter

_FORMAT = "%(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s"
# Second-level strftime pattern; milliseconds are appended in formatTime.
_DATE_FORMAT = "%m-%d %H:%M:%S"


def _format_time_ms(formatter: logging.Formatter, record: logging.LogRecord, datefmt: str | None = None) -> str:
    """Format record time with millisecond precision (strftime has no %f here)."""
    ct = formatter.converter(record.created)
    s = time.strftime(datefmt or _DATE_FORMAT, ct)
    return f"{s}.{int(record.msecs):03d}"


_LOG_DIR = os.path.join(os.path.expanduser("~"), "ascend", "log", "vllm_ascend")
_LOG_MAX_BYTES = 20 * 1024 * 1024


def init_logger_ascend(name: str) -> logging.Logger:
    """Logger under ``vllm_ascend.*`` (Ascend handler), not ``vllm.*``.

    Nesting under ``vllm.`` makes records hit the root ``vllm`` StreamHandler,
    whose level tracks ``VLLM_LOGGING_LEVEL`` (often INFO). That filters out
    DEBUG even when package / module logger levels are DEBUG.
    Ascend's own handler stays at DEBUG; levels are gated by
    :func:`apply_ascend_log_level`.
    """
    return init_logger(name)


def _resolve_log_level(level: str) -> int:
    return getattr(logging, str(level).upper(), logging.INFO)


def _normalize_debug_module(entry: str) -> str:
    """Map config entry to a ``vllm_ascend.*`` logger name."""
    name = str(entry).strip().strip(".")
    if not name:
        return ""
    if name == "vllm_ascend" or name.startswith("vllm_ascend."):
        return name
    return f"vllm_ascend.{name}"


def _iter_ascend_logger_names() -> list[str]:
    names: list[str] = []
    for name in list(logging.Logger.manager.loggerDict):
        if not isinstance(name, str):
            continue
        if name.startswith("vllm_ascend."):
            names.append(name)
    return names


def _set_logger_tree_level(prefix: str, level: int) -> None:
    """Set ``prefix`` and any already-created ``prefix.*`` loggers."""
    logging.getLogger(prefix).setLevel(level)
    dotted = prefix + "."
    for name in _iter_ascend_logger_names():
        if name == prefix or name.startswith(dotted):
            logging.getLogger(name).setLevel(level)


_last_applied_ascend_log: tuple[str, tuple[str, ...]] | None = None


def apply_ascend_log_level(
    level: str = "INFO",
    debug_modules: list[str] | None = None,
) -> None:
    """Apply package root level and per-module DEBUG whitelist.

    Args:
        level: Level for the ``vllm_ascend`` root logger (e.g. ``INFO``).
        debug_modules: Relative module paths forced to DEBUG, such as
            ``\"dfx\"`` or ``\"dfx.runtime_config\"`` (mapped to
            ``vllm_ascend.<entry>``). Full ``vllm_ascend.*`` names are also
            accepted.
    """
    global _last_applied_ascend_log

    # ``logging.disable(level)`` (e.g. vLLM's ``suppress_logging`` helper) is a
    # *global* kill-switch checked before any individual logger's level. If a
    # caller enters it and raises before restoring (missing try/finally), every
    # logger goes silent forever — console and file handlers alike — even
    # though ``ascend_log`` itself looks correctly configured. Self-heal on
    # every apply so a stuck global disable can never mask our hot-reloaded
    # level.
    if logging.root.manager.disable:
        stuck_at = logging.getLevelName(logging.root.manager.disable)
        logging.disable(logging.NOTSET)
        logging.getLogger("vllm_ascend.logger").warning(
            "[ascend_log] cleared a stuck global logging.disable(%s) — some "
            "code path left logging globally muted without restoring it",
            stuck_at,
        )

    configure_ascend_logging()
    root_level = _resolve_log_level(level)
    debug_list = list(debug_modules or ())
    level_key = str(level).upper()
    debug_key = tuple(debug_list)
    needs_debug = root_level <= logging.DEBUG or bool(debug_list)
    announce = _last_applied_ascend_log != (level_key, debug_key)

    ascend = logging.getLogger("vllm_ascend")
    ascend.setLevel(root_level)
    ascend.propagate = False
    # Keep handlers able to emit DEBUG; logger.level is the gate.
    for h in ascend.handlers:
        if h.level > logging.DEBUG:
            h.setLevel(logging.DEBUG)

    # Reset known children to the root level (clears a prior debug whitelist).
    for name in _iter_ascend_logger_names():
        logging.getLogger(name).setLevel(root_level)

    for entry in debug_list:
        prefix = _normalize_debug_module(entry)
        if not prefix or prefix == "vllm_ascend":
            # Root already set; ignore empty / redundant root entries.
            if prefix == "vllm_ascend":
                ascend.setLevel(logging.DEBUG)
            continue
        _set_logger_tree_level(prefix, logging.DEBUG)

    if needs_debug:
        # Outer collectors (e.g. UC) often attach INFO-level handlers on root /
        # ``vllm``. Lower those handlers so DEBUG records are not dropped after
        # our loggers allow them. Logger levels elsewhere stay unchanged.
        for h in logging.root.handlers:
            if h.level > logging.DEBUG:
                h.setLevel(logging.DEBUG)
        vllm_logger = logging.getLogger("vllm")
        for h in vllm_logger.handlers:
            if h.level > logging.DEBUG:
                h.setLevel(logging.DEBUG)
        for name in list(logging.Logger.manager.loggerDict):
            if not isinstance(name, str):
                continue
            # Huawei UC / slog-style module loggers seen in the wild.
            # Force DEBUG even when level is NOTSET (otherwise effective level
            # stays WARNING/ERROR via root and UC DEBUG never appears).
            if name.upper() == "UC" or name.endswith(".UC"):
                lg = logging.getLogger(name)
                if lg.level == logging.NOTSET or lg.level > logging.DEBUG:
                    lg.setLevel(logging.DEBUG)
                for h in lg.handlers:
                    if h.level > logging.DEBUG:
                        h.setLevel(logging.DEBUG)

    _last_applied_ascend_log = (level_key, debug_key)
    if announce:
        # INFO so operators can confirm apply even when DEBUG is still filtered.
        logging.getLogger("vllm_ascend.logger").info(
            "[ascend_log] applied level=%s debug=%s effective=%s",
            level_key,
            debug_list,
            logging.getLevelName(ascend.getEffectiveLevel()),
        )
        # TEMP DIAGNOSTIC (remove once the DEBUG-goes-silent issue is root
        # caused): written straight to the original stderr fd, bypassing the
        # ``logging`` module entirely, so it still shows up even if every
        # logger/handler in this process has gone silent for some other
        # reason (stuck ``logging.disable``, broken handler, closed stream).
        try:
            ascend_handlers = [(h.__class__.__name__, logging.getLevelName(h.level)) for h in ascend.handlers]
            vllm_handlers = [
                (h.__class__.__name__, logging.getLevelName(h.level)) for h in logging.getLogger("vllm").handlers
            ]
            sys.__stderr__.write(
                f"[ascend_log][diag] level={level_key} "
                f"effective={logging.getLevelName(ascend.getEffectiveLevel())} "
                f"disable={logging.getLevelName(logging.root.manager.disable)} "
                f"ascend.disabled={ascend.disabled} handlers={ascend_handlers} "
                f"vllm.handlers={vllm_handlers}\n"
            )
            sys.__stderr__.flush()
        except Exception:
            pass


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

    def formatTime(self, record, datefmt=None):
        return _format_time_ms(self, record, datefmt)

    def format(self, record):
        return _format_with_ascend_prefix(self, record, super().format)


class AscendColoredFormatter(ColoredFormatter):
    """Extends ColoredFormatter with [vllm-ascend] prefix and module name."""

    def formatTime(self, record, datefmt=None):
        return _format_time_ms(self, record, datefmt)

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
    file_handler.setLevel(logging.DEBUG)
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
    handler.setLevel(logging.DEBUG)

    if _use_color():
        handler.setFormatter(AscendColoredFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))
    else:
        handler.setFormatter(AscendFormatter(fmt=_FORMAT, datefmt=_DATE_FORMAT))

    ascend_logger.addHandler(handler)
    ascend_logger.setLevel(envs.VLLM_LOGGING_LEVEL)
    ascend_logger.propagate = False

    # Keep handlers able to emit DEBUG records (handler level remains DEBUG).
    # The actual effective package/module levels are controlled by
    # ``apply_ascend_log_level`` (driven by DFX ``ascend_log``).
