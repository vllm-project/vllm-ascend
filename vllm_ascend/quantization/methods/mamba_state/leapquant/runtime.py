# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime glue between the LeapQuant state ops and the GDN/KDA attention paths.

``state.py`` holds the device-agnostic numerics; this module owns the
integration concerns: reading the (optional) AscendConfig singleton, deriving
window-boundary rows from per-request sequence lengths, and applying the
flush to rows of a state cache pool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.quantization.methods.mamba_state.leapquant.state import leap_window_flush

if TYPE_CHECKING:
    from vllm_ascend.ascend_config import LeapQuantConfig

__all__ = [
    "leapquant_config_or_none",
    "leapquant_flush_state_rows",
]

_device_gate_checked = False


def _check_device_once() -> None:
    """Fail fast on non-A3 devices at the first flush (worker process).

    The check cannot run at config-validation time: resolving the Ascend
    device family initializes the NPU runtime, which poisons the forked
    EngineCore process under the default ``fork`` start method ("Cannot
    re-initialize NPU in forked subprocess"). At the first flush the
    worker's NPU context already exists, so resolving the family here is
    safe.
    """
    global _device_gate_checked
    if _device_gate_checked:
        return
    from vllm_ascend.device.device_config import AscendDeviceType, get_ascend_device_type

    try:
        device_type = get_ascend_device_type()
    except Exception:
        # No resolvable Ascend device (e.g. CPU-only CI running unit tests);
        # a real serve always resolves, so only fire the gate when it does.
        # The flag stays unset so a worker whose runtime comes up late
        # retries at the next flush instead of running ungated forever.
        return
    if device_type != AscendDeviceType.A3:
        # Keep the flag unset: if a caller catches this error, the next
        # flush raises again rather than silently skipping the gate.
        raise RuntimeError(
            "leapquant_config is validated on Atlas A3 only in this milestone; "
            "please disable leapquant_config on this device."
        )
    _device_gate_checked = True


def leapquant_config_or_none() -> LeapQuantConfig | None:
    """Return the enabled LeapQuantConfig, or None when disabled/uninitialized.

    Attention layers and metadata builders call this during construction; a
    missing AscendConfig singleton (CPU unit tests) degrades to "disabled".
    """
    from vllm_ascend.ascend_config import get_ascend_config

    try:
        config = get_ascend_config()
    except RuntimeError:
        return None
    leapquant = config.leapquant_config
    return leapquant if leapquant.enabled else None


def leapquant_flush_state_rows(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    config: LeapQuantConfig,
    state_copy: Any | None = None,
) -> None:
    """Re-quantize the state rows whose sequence just crossed a window boundary.

    The scheduler orders requests decodes-first, and ``seq_lens`` counts the
    tokens processed *after* this step's query, so the window-boundary
    condition is simply ``seq_len % window == 0`` for both decode and prefill
    rows.  The flush runs once per step, only on due rows, and leaves the
    pool layout intact (fp32 in, fp32 out — the storage-side checkpoint lands
    with the fused-kernel milestone).

    Prefill caveat: a chunked-prefill chunk that *crosses* a window boundary
    without ending on it is not flushed at that interior boundary — the
    chunked delta-rule kernel only materializes the end-of-chunk state.  In
    that window the state accumulates one (or a few) extra tokens before the
    next boundary-aligned flush, which can only *reduce* the injected
    quantization error per boundary; the exact per-boundary cadence is
    restored from the first decode boundary on.  Exact interior-boundary
    flushes are deferred to the fused-kernel milestone, whose chunk loop can
    emit per-window states.

    Args:
        state: state pool ``[slots, H, V, K]`` (may have a padded batch stride).
        state_indices: per-request state slot ids aligned with ``seq_lens``.
        seq_lens: CPU int tensor of per-request post-step sequence lengths.
        config: enabled LeapQuantConfig.
        state_copy: optional layer-owned gather/scatter plan (KDA); must
            expose ``gather(state, indices, flags)`` and
            ``scatter(state, source, indices)``. Falls back to plain indexing
            on CPU and to the GDN triton kernels on NPU.
    """
    total = state_indices.shape[0]
    if total == 0:
        return
    _check_device_once()
    due = (seq_lens[:total].to(torch.long) % config.window) == 0
    due_rows = torch.nonzero(due, as_tuple=False).flatten()
    if due_rows.numel() == 0:
        return

    rows = due_rows.to(state_indices.device)
    slot_ids = state_indices.index_select(0, rows)
    flushed = None
    if state_copy is not None and state.device.type != "cpu":
        present = torch.ones(slot_ids.shape[0], dtype=torch.bool, device=slot_ids.device)
        gathered = state_copy.gather(state, slot_ids, present)
        flushed = leap_window_flush(gathered, compensator_tokens=config.compensator_tokens)
        state_copy.scatter(state, flushed.to(state.dtype).contiguous(), slot_ids)
        return
    if state.device.type == "cpu":
        # Plain torch indexing covers CPU unit tests and any non-NPU device;
        # it handles arbitrary batch strides, padded pools included.
        gathered = state.index_select(0, slot_ids)
        flushed = leap_window_flush(gathered, compensator_tokens=config.compensator_tokens)
        state.index_copy_(0, slot_ids, flushed.to(state.dtype))
        return

    from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_

    present = torch.ones(slot_ids.shape[0], dtype=torch.bool, device=slot_ids.device)
    gathered = gather_ssm_states(state, slot_ids, present)
    flushed = leap_window_flush(gathered, compensator_tokens=config.compensator_tokens)
    scatter_ssm_states_(state, slot_ids, flushed.to(state.dtype).contiguous())
