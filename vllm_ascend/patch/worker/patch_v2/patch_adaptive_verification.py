import numpy as np
import torch
import vllm.v1.worker.gpu.spec_decode.adaptive_verification as adaptive_verification
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import _assign_draft_token_budget


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


adaptive_verification.async_copy_to_gpu = _copy_to_npu_sync
adaptive_verification._assign_draft_token_budget_compiled = _assign_draft_token_budget
