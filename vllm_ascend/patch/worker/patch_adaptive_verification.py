import torch
from torch.overrides import TorchFunctionMode
from vllm.v1.worker.gpu.spec_decode import adaptive_verification

from vllm_ascend.utils import vllm_version_is


def _index_fill(tensor, dim, index, value):
    # Import lazily to avoid circular imports during plugin startup.
    from vllm_ascend.device.device_op import DeviceOperator

    return DeviceOperator.index_fill(tensor, dim, index, value)


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
