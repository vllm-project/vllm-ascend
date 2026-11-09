import torch
import torch_npu
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch as _upstream_apply_top_k_top_p_pytorch

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.sample.apply_top_k_top_p import (
    apply_top_k_top_p as triton_apply_top_k_top_p,
)
from vllm_ascend.ops.triton.sample.apply_top_k_top_p import (
    is_triton_apply_topk_topp_supported,
    is_triton_topk_topp_profitable,
)


def _apply_top_k_top_p_npu(
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
    return _upstream_apply_top_k_top_p_pytorch(logits, k=k, p=p)


_HAS_CANN_TOP_K_TOP_P = get_current_hardware_profile().supports(HardwareCapability.NPU_TOP_K_TOP_P)


def _apply_top_k_top_p_triton(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor | None:
    """Triton fallback used when the native CANN op is unavailable.

    Returns None when the fallback is disabled, unsupported, or below the
    measured crossover, so the caller keeps the PyTorch path.
    """
    cfg = get_ascend_config()
    if not (cfg.enable_triton_apply_topk_topp and HAS_TRITON):
        return None
    if not is_triton_apply_topk_topp_supported():
        logger.warning_once(
            "[sample/v2] Triton apply_top_k_top_p is not supported on this device; falling back to the PyTorch path.",
        )
        return None
    B, V = logits.shape
    if not is_triton_topk_topp_profitable(B, V, disable_gate=cfg.triton_topk_topp_disable_gate):
        logger.warning_once(
            "[sample/v2] Triton apply_top_k_top_p skipped (B=%d V=%d below the "
            "measured performance crossover); using the PyTorch path.",
            B,
            V,
        )
        return None
    logger.info_once(
        "[sample/v2] Using Triton apply_top_k_top_p op (topk_renorm semantics) "
        "for top-k/top-p masking: logits shape %s.",
        tuple(logits.shape),
    )
    return triton_apply_top_k_top_p(
        logits,
        k,
        p,
        top_p_mode="topk_renorm",
        no_sort=not cfg.triton_topk_topp_disable_no_sort,
    )


# NOTE: During the warmup stage, if both k and p are None, the NPU op is skipped.
# As a result, its workspace is not reserved during warmup, which may lead to an OOM
# when the op is executed later.
# TODO: Fix the workspace reservation issue in the future.
def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    has_topp_only_rows: bool = True,
) -> torch.Tensor:
    if k is None and p is None:
        return logits
    # Native CANN op first; Triton only when it is unavailable; PyTorch last.
    if _HAS_CANN_TOP_K_TOP_P:
        return _apply_top_k_top_p_npu(logits, k, p, has_topp_only_rows)
    out = _apply_top_k_top_p_triton(logits, k, p)
    if out is not None:
        return out
    return _apply_top_k_top_p_pytorch(logits, k, p, has_topp_only_rows)
