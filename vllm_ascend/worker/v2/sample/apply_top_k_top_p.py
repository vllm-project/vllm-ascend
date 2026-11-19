import torch
import torch_npu
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile


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
        return logits
    return torch_npu.npu_top_k_top_p(logits, k=k, p=p)


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
