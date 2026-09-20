import torch
from torch.overrides import TorchFunctionMode

from vllm.v1.worker.gpu.spec_decode import adaptive_verification


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
            # Import lazily to avoid circular imports during plugin startup.
            from vllm_ascend.device.device_op import DeviceOperator

            tensor, dim, indices, value = args
            return DeviceOperator.index_fill(tensor, dim, indices, value)
        return func(*args, **kwargs)


_original_assign_draft_token_budget = adaptive_verification._assign_draft_token_budget


def _assign_draft_token_budget_ascend(*args, **kwargs):
    with _IndexFillMode():
        return _original_assign_draft_token_budget(*args, **kwargs)


adaptive_verification._assign_draft_token_budget_compiled = _assign_draft_token_budget_ascend
