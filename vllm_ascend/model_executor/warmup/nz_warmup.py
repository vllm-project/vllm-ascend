# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Pay the first NZ format-cast lazy init while weight I/O is on the critical path.

The first ``npu_format_cast`` on a rank is ~15-19 s; later casts are free. This
is independent of the quantization scheme, so any model that casts weights to
NZ can overlap that init with weight I/O. ``NPUModelRunner.load_model`` joins
this thread before it returns, which is before memory profiling.

Enable with ``ascend_warmup_config.enable_early_nz_warmup``.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import maybe_trans_nz

_NZ_JOIN_TIMEOUT_S = 600.0

_NZ_WARMED = False
_NZ_THREAD_STARTED = False
_NZ_THREAD: Any = None


def warm_nz_format_cast(tag: str) -> None:
    """Pay the first NZ format-cast lazy init on a tiny scratch tensor."""
    del tag
    global _NZ_WARMED
    if _NZ_WARMED:
        return
    _NZ_WARMED = True
    try:
        scratch = maybe_trans_nz(torch.zeros((256, 256), dtype=torch.int8, device="npu"))
        torch.npu.synchronize()
        del scratch
    except Exception:
        logger.warning("NZ format-cast warmup skipped", exc_info=True)


def _nz_enabled() -> bool:
    try:
        return bool(get_ascend_config().ascend_warmup_config.enable_early_nz_warmup)
    except RuntimeError:
        return False


def start_nz_warm_thread(tag: str) -> None:
    """Run ``warm_nz_format_cast`` on a daemon thread. Idempotent."""
    global _NZ_THREAD, _NZ_THREAD_STARTED
    if not _nz_enabled():
        return
    if _NZ_THREAD_STARTED or _NZ_WARMED:
        return
    _NZ_THREAD_STARTED = True
    try:
        device = torch.npu.current_device()

        def _run() -> None:
            try:
                torch.npu.set_device(device)
                warm_nz_format_cast(tag)
            except Exception:
                logger.warning("NZ format-cast warmup thread failed", exc_info=True)

        _NZ_THREAD = threading.Thread(target=_run, name="coldstart-nz-warm", daemon=True)
        _NZ_THREAD.start()
    except Exception:
        logger.warning("NZ format-cast warmup thread not started", exc_info=True)


def join_nz_warm_thread() -> None:
    """Wait for ``start_nz_warm_thread`` to finish. No-op if it never started."""
    thread = _NZ_THREAD
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=_NZ_JOIN_TIMEOUT_S)
    if thread.is_alive():
        logger.warning("NZ format-cast warmup still running after %.0f s", _NZ_JOIN_TIMEOUT_S)
