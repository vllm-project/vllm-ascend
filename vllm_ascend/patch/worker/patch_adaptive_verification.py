import numpy as np
import torch
import vllm.v1.worker.gpu.spec_decode.adaptive_verification as adaptive_verification

from vllm_ascend.utils import vllm_version_is


def _copy_to_npu_sync(
    value: torch.Tensor | np.ndarray,
    out: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Keep AV's dependent H2D copies ordered on the current NPU stream."""

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    assert value.is_cpu
    if out is None:
        assert device is not None
        out = torch.empty_like(value, device=device)
    return out.copy_(value, non_blocking=False)


class _IndexFillMode(TorchFunctionMode):
    """Temporarily route index_fill_ through the Ascend device adaptor.

    A5's current native index_fill_ path synchronizes while converting the
    device index tensor to a host vector. Keep the upstream budget allocator
    unchanged and override only this operation within its dynamic scope.
    Remove this mode once the native A5 index_fill_ operator is ready.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.Tensor.index_fill_:
            return _index_fill(*args, **kwargs)
        return func(*args, **kwargs)


# vLLM main (#63421) replaced the torch.compile'd budget assignment with a
# Triton kernel that no longer calls index_fill_, so the A5 workaround only
# applies to the v0.32.0 allocator.
if vllm_version_is("0.32.0"):
    _original_assign_draft_token_budget = adaptive_verification._assign_draft_token_budget

    def _assign_draft_token_budget_ascend(*args, **kwargs):
        with _IndexFillMode():
            return _original_assign_draft_token_budget(*args, **kwargs)

    adaptive_verification._assign_draft_token_budget_compiled = _assign_draft_token_budget_ascend

if hasattr(adaptive_verification, "async_copy_to_gpu"):
    adaptive_verification.async_copy_to_gpu = _copy_to_npu_sync
