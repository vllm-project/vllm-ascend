import torch

from vllm_ascend.sample.sampler import _apply_top_k_top_p_pytorch
from vllm_ascend.utils import vllm_version_is

# NOTE: During the warmup stage, if both k and p are None, the NPU op is skipped.
# As a result, its workspace is not reserved during warmup, which may lead to an OOM
# when the op is executed later.
# TODO: Fix the workspace reservation issue in the future.
if vllm_version_is("0.31.0"):

    def apply_top_k_top_p_npu(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
        if k is None and p is None:
            return logits
        # use pytorch ops
        return _apply_top_k_top_p_pytorch(logits, k, p)

else:
    # vLLM main added `has_topp_only_rows` to select the split top-p Triton
    # pipeline. This override delegates to the PyTorch reference, which handles
    # top-p-only rows either way, so the value is accepted but unused.
    def apply_top_k_top_p_npu(  # type: ignore[misc]
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        has_topp_only_rows: bool = True,
    ) -> torch.Tensor:
        if k is None and p is None:
            return logits
        # use pytorch ops
        return _apply_top_k_top_p_pytorch(logits, k, p)
