# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compile Triton warmup kernels while weight I/O is already on the critical path.

Rejection-sampler / penalty / RMS JIT is host work. Starting it at the
beginning of ``NPUModelRunner.load_model`` overlaps the compile with
safetensors load. That method joins this thread before it returns, before
memory profiling, so the regular kernel_warmup hits the Triton cache.

Enable with ``ascend_warmup_config.enable_early_kernel_warmup``. Imports
that the thread needs are done on the main thread first so a concurrent import
cannot walk sys.modules mid-registration.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config

_JOIN_TIMEOUT_S = 600.0

_STATE: dict[str, Any] = {"started": False, "thread": None, "done": False}


def _enabled() -> bool:
    try:
        return bool(get_ascend_config().ascend_warmup_config.enable_early_kernel_warmup)
    except RuntimeError:
        return False


class _WarmupShim:
    """Config-only stand-in for NPUWorker during construct-time warmup."""

    def __init__(self, vllm_config: Any, device: torch.device) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.scheduler_config = vllm_config.scheduler_config
        self.model_config = vllm_config.model_config
        self.model_runner = None


def _preimport() -> None:
    import vllm.triton_utils  # noqa: F401

    import vllm_ascend.ops.triton.penalty  # noqa: F401
    import vllm_ascend.ops.triton.reject_sample  # noqa: F401
    import vllm_ascend.ops.triton.rms_norm  # noqa: F401
    import vllm_ascend.ops.triton.spec_decode.utils  # noqa: F401
    from vllm_ascend.model_executor.warmup.penalties_triton_warmup import (  # noqa: F401
        penalties_triton_warmup,
    )
    from vllm_ascend.model_executor.warmup.rejection_sampler_triton_warmup import (  # noqa: F401
        rejection_sampler_triton_warmup,
    )
    from vllm_ascend.model_executor.warmup.rms_triton_warmup import (  # noqa: F401
        triton_rms_warmup,
    )


def _run(shim: _WarmupShim, device_index: int) -> None:
    from vllm_ascend.model_executor.warmup.penalties_triton_warmup import (
        penalties_triton_warmup,
    )
    from vllm_ascend.model_executor.warmup.rejection_sampler_triton_warmup import (
        rejection_sampler_triton_warmup,
    )
    from vllm_ascend.model_executor.warmup.rms_triton_warmup import triton_rms_warmup

    try:
        torch.npu.set_device(device_index)
    except Exception:
        logger.warning("Early kernel warmup set_device failed; regular warmup still runs", exc_info=True)
        return

    items = (
        ("rejection_sampler", lambda: rejection_sampler_triton_warmup(shim)),
        ("penalties", lambda: penalties_triton_warmup(shim)),
        ("rms", lambda: triton_rms_warmup(shim, assume_used=True)),
    )
    for name, fn in items:
        try:
            fn()
        except Exception:
            logger.warning("Early kernel warmup item %s failed; regular warmup will compile it", name, exc_info=True)


def start_early_kernel_warmup() -> None:
    if not _enabled():
        return
    if _STATE["started"]:
        return

    try:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config is None:
            # Construct has not entered the config context yet. Leave the
            # state untouched so a later call can still start the thread.
            return
        _STATE["started"] = True

        device_index = torch.npu.current_device()
        shim = _WarmupShim(vllm_config, torch.device(f"npu:{device_index}"))
        _preimport()

        def _target() -> None:
            try:
                _run(shim, device_index)
            finally:
                _STATE["done"] = True

        thread = threading.Thread(
            target=_target,
            name="coldstart-early-kernel-warmup",
            daemon=True,
        )
        _STATE["thread"] = thread
        thread.start()
    except Exception:
        logger.warning("Early kernel warmup not started; regular warmup unaffected", exc_info=True)


def join_early_kernel_warmup(where: str) -> None:
    thread = _STATE.get("thread")
    if thread is None:
        return
    if thread is threading.current_thread():
        return
    if not thread.is_alive():
        return
    thread.join(timeout=_JOIN_TIMEOUT_S)
    if thread.is_alive():
        logger.warning("Early kernel warmup still running after %.0f s at %s", _JOIN_TIMEOUT_S, where)
