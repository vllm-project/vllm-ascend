import torch
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.triton.apply_top_k_top_p import (
    apply_top_k_top_p as triton_apply_top_k_top_p,
)
from vllm_ascend.sample.sampler import _apply_top_k_top_p_pytorch
from vllm_ascend.utils import vllm_version_is

# NOTE: During the warmup stage, if both k and p are None, the NPU op is skipped.
# As a result, its workspace is not reserved during warmup, which may lead to an OOM
# when the op is executed later.
# TODO: Fix the workspace reservation issue in the future.
#
# Both variants route to the Triton apply_top_k_top_p op (masked-logits form;
# V2's gumbel_sample consumes logits, so the fused-softmax variants are not
# usable here) when enable_triton_apply_topk_topp is set, otherwise they fall
# back to the sort-based PyTorch reference.
if vllm_version_is("0.31.0"):

    def apply_top_k_top_p_npu(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
        if k is None and p is None:
            return logits
        if get_ascend_config().enable_triton_apply_topk_topp and HAS_TRITON:
            logger.info_once(
                "[sample/v2] Using Triton apply_top_k_top_p op for top-k/top-p masking: logits shape %s.",
                tuple(logits.shape),
            )
            return triton_apply_top_k_top_p(logits, k, p)
        # use pytorch ops
        return _apply_top_k_top_p_pytorch(logits, k, p)

else:
    # vLLM main added `has_topp_only_rows` to select the split top-p Triton
    # pipeline. The Triton op and the PyTorch reference both handle per-row
    # k/p (including top-p-only rows), so the value is accepted but unused.
    def apply_top_k_top_p_npu(  # type: ignore[misc]
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        has_topp_only_rows: bool = True,
    ) -> torch.Tensor:
        if k is None and p is None:
            return logits
        if get_ascend_config().enable_triton_apply_topk_topp and HAS_TRITON:
            logger.info_once(
                "[sample/v2] Using Triton apply_top_k_top_p op for top-k/top-p masking: logits shape %s.",
                tuple(logits.shape),
            )
            return triton_apply_top_k_top_p(logits, k, p)
        # use pytorch ops
        return _apply_top_k_top_p_pytorch(logits, k, p)
