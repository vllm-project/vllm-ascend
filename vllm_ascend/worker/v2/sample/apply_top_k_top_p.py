import os
from collections import Counter

import torch
import torch_npu
from vllm.logger import init_logger
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

# Read once, before worker execution / graph capture. Restart workers to change mode.
_SHORTLIST_MODE = os.environ.get("VLLM_ASCEND_TOPKP_SHORTLIST_MODE", "off").lower()
_SHADOW_MAX_ROWS = 9
if _SHORTLIST_MODE == "shadow":
    try:
        _SHADOW_MAX_ROWS = max(0, int(os.environ.get("VLLM_ASCEND_TOPKP_SHORTLIST_SHADOW_MAX_ROWS", "9")))
    except ValueError:
        _SHADOW_MAX_ROWS = 0  # Invalid shadow configuration fails closed.
_LOGGER = init_logger(__name__)
_LOG_PID = os.getpid()
_COUNTS: Counter[str] = Counter()
_CALLS = 0
_VOCAB_SIZE = 151936
_SHORTLIST_SIZE = 128
_SHADOW_FREE_MARGIN_BYTES = 2 * 1024 * 1024 * 1024


def _record(event: str, rows: int, *, hit: bool = False) -> None:
    """First outcome of each kind plus one aggregate per 1024 opt-in calls."""
    global _LOG_PID, _CALLS
    pid = os.getpid()
    if pid != _LOG_PID:
        _LOG_PID = pid
        _COUNTS.clear()
        _CALLS = 0
    _COUNTS[event] += 1
    _CALLS += 1
    if _COUNTS[event] == 1 or _CALLS % 1024 == 0:
        log = _LOGGER.error if event.startswith("shadow_mismatch") else _LOGGER.info
        log(
            "topkp_shortlist pid=%s mode=%s rows=%s hit=%s event=%s calls=%s counts=%s",
            pid,
            _SHORTLIST_MODE,
            rows,
            int(hit),
            event,
            _CALLS,
            dict(_COUNTS),
        )


if _SHORTLIST_MODE == "shadow":
    _SHADOW_RECORD_BASE = _record
    _SHADOW_ROW_EVENTS: set[tuple[str, int]] = set()

    def _record_shadow_rows(event: str, rows: int, *, hit: bool = False) -> None:
        if os.getpid() != _LOG_PID:
            _SHADOW_ROW_EVENTS.clear()
        coverage_key = (event, rows)
        first_shape = coverage_key not in _SHADOW_ROW_EVENTS
        _SHADOW_RECORD_BASE(event, rows, hit=hit)
        if first_shape and _COUNTS[event] != 1 and _CALLS % 1024 != 0:
            _LOGGER.info(
                "topkp_shortlist pid=%s mode=%s rows=%s hit=%s event=%s calls=%s counts=%s",
                os.getpid(),
                _SHORTLIST_MODE,
                rows,
                int(hit),
                event,
                _CALLS,
                dict(_COUNTS),
            )
        _SHADOW_ROW_EVENTS.add(coverage_key)

    _record = _record_shadow_rows


def _metadata_reason(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> str | None:
    if logits.dtype != torch.float32 or logits.layout != torch.strided:
        return "metadata_logits_dtype_or_layout"
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != _VOCAB_SIZE:
        return "metadata_logits_shape"
    if logits.device.type != "npu" or logits.stride(1) != 1:
        return "metadata_logits_device_or_stride"
    if not isinstance(k, torch.Tensor) or not isinstance(p, torch.Tensor):
        return "metadata_missing_k_or_p"
    if k.dtype != torch.int32 or p.dtype != torch.float32:
        return "metadata_k_or_p_dtype"
    if k.shape != (logits.shape[0],) or p.shape != (logits.shape[0],):
        return "metadata_k_or_p_shape"
    if k.device != logits.device or p.device != logits.device:
        return "metadata_k_or_p_device"
    if k.layout != torch.strided or p.layout != torch.strided:
        return "metadata_k_or_p_layout"
    return None


def _capture_reason(logits: torch.Tensor) -> str | None:
    npu = getattr(torch, "npu", None)
    probe = getattr(npu, "is_current_stream_capturing", None)
    current_device = getattr(npu, "current_device", None)
    if not callable(probe) or not callable(current_device):
        return "capture_unknown"
    try:
        if current_device() != logits.device.index:
            return "capture_device_mismatch"
        capturing = probe()
    except Exception:
        return "capture_probe_failed"
    if not isinstance(capturing, bool):
        return "capture_unknown"
    return "capture_active" if capturing else None


def _shadow_memory_reason(logits: torch.Tensor) -> str | None:
    npu = getattr(torch, "npu", None)
    mem_get_info = getattr(npu, "mem_get_info", None)
    if not callable(mem_get_info):
        return "shadow_uncovered_memory_unknown"
    try:
        free_bytes, _ = mem_get_info(logits.device)
    except Exception:
        return "shadow_uncovered_memory_probe_failed"
    full_bytes = logits.numel() * logits.element_size()
    estimated_extra = 8 * full_bytes + logits.shape[0] * 129 * 12 * 4 + 16 * 1024 * 1024
    if free_bytes < estimated_extra + _SHADOW_FREE_MARGIN_BYTES:
        return "shadow_uncovered_memory_margin"
    return None


def _guard_and_topk(logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor):
    values, token_ids = torch.topk(logits, _SHORTLIST_SIZE + 1, dim=-1, sorted=True)
    failed = (
        (~torch.isfinite(logits).all()).to(torch.int32)
        + 2 * (~(values[:, 39] > values[:, 128]).all()).to(torch.int32)
        + 4 * (~(k == 40).all()).to(torch.int32)
        + 8 * (~(p == 0.9).all()).to(torch.int32)
    )
    bits = int(failed.item())
    if bits:
        reasons = [name for bit, name in ((1, "nonfinite"), (2, "kth_margin"), (4, "k"), (8, "p")) if bits & bit]
        return values, token_ids, "value_guard_" + "+".join(reasons)
    return values, token_ids, None


def _shortlist_result(logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor, values, token_ids) -> torch.Tensor:
    selected_ids, original_order = token_ids[:, :_SHORTLIST_SIZE].to(dtype=torch.float32).sort(dim=-1)
    selected_ids = selected_ids.to(dtype=token_ids.dtype)
    short_logits = values[:, :_SHORTLIST_SIZE].gather(1, original_order)
    short_output = torch_npu.npu_top_k_top_p(short_logits, k=k, p=p)
    output = torch.full_like(logits, float("-inf"))
    output.scatter_(1, selected_ids, short_output)
    return output


def _apply_top_k_top_p_npu(  # type: ignore[misc]
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    has_topp_only_rows: bool = True,
) -> torch.Tensor:
    # ``has_topp_only_rows`` only selects the split top-p Triton pipeline upstream;
    # the fused operator handles top-p-only rows natively.
    #
    # NOTE: TopKtopPSampleV2 takes ``(logits, p, k)`` with ``p`` whose dtype matches ``logits``;
    # it is absent on 310P.
    if k is None and p is None:
        if _SHORTLIST_MODE != "off":
            _record("original_no_filter_identity", logits.shape[0] if logits.ndim else 0)
        return logits
    if _SHORTLIST_MODE == "off":
        return torch_npu.npu_top_k_top_p(logits, k=k, p=p)

    rows = logits.shape[0] if logits.ndim else 0
    if _SHORTLIST_MODE not in ("shadow", "on"):
        _record("invalid_mode_native", rows)
        return torch_npu.npu_top_k_top_p(logits, k=k, p=p)
    if _SHORTLIST_MODE == "on" and rows > 720:
        _record("on_uncovered_row_cap", rows)
        return torch_npu.npu_top_k_top_p(logits, k=k, p=p)

    reason = _metadata_reason(logits, k, p)
    if reason is None:
        reason = _capture_reason(logits)
    if reason is not None:
        _record(reason, rows)
        return torch_npu.npu_top_k_top_p(logits, k=k, p=p)

    if _SHORTLIST_MODE == "shadow":
        if rows > _SHADOW_MAX_ROWS:
            _record("shadow_uncovered_row_cap", rows)
            return torch_npu.npu_top_k_top_p(logits, k=k, p=p)
        reason = _shadow_memory_reason(logits)
        if reason is not None:
            _record(reason, rows)
            return torch_npu.npu_top_k_top_p(logits, k=k, p=p)

    assert k is not None and p is not None
    values, token_ids, reason = _guard_and_topk(logits, k, p)
    if reason is not None:
        _record(reason, rows)
        return torch_npu.npu_top_k_top_p(logits, k=k, p=p)
    candidate = _shortlist_result(logits, k, p, values, token_ids)
    if _SHORTLIST_MODE == "on":
        _record("on_hit", rows, hit=True)
        return candidate

    original = torch_npu.npu_top_k_top_p(logits, k=k, p=p)
    mismatch = (candidate.view(torch.int32) != original.view(torch.int32)).any().to(torch.int32) + 2 * (
        torch.isneginf(candidate) != torch.isneginf(original)
    ).any().to(torch.int32)
    mismatch_bits = int(mismatch.item())
    if mismatch_bits:
        _record(f"shadow_mismatch_bits_{mismatch_bits}", rows, hit=True)
        raise RuntimeError(
            f"topkp_shortlist shadow mismatch: pid={os.getpid()} rows={rows} "
            f"bits={mismatch_bits}; stop and restore original sampler"
        )
    _record("shadow_pass", rows, hit=True)
    return original


def _apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    has_topp_only_rows: bool = True,
) -> torch.Tensor:
    """Adapt the upstream reference implementation to the extra top-p flag."""
    return apply_top_k_top_p_pytorch(logits, k=k, p=p)


apply_top_k_top_p = (
    _apply_top_k_top_p_npu
    if get_current_hardware_profile().supports(HardwareCapability.NPU_TOP_K_TOP_P)
    else _apply_top_k_top_p_pytorch
)
