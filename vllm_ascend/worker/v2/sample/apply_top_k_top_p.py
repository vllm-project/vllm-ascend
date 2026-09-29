import torch
from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

if HAS_TRITON:
    from vllm_ascend.ops.triton.v2.sample.topk_topp import apply_top_k_top_p_triton


def apply_top_k_top_p_npu(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    """MRV2 top-k/top-p entry point patched over the upstream Triton hook.

    Prefers the Ascend Qrita Triton kernel and falls back to the
    sort+mask PyTorch chain when Triton is unavailable.
    """
    if HAS_TRITON:
        if k is None and p is None:
            # Defensive: a caller that reaches this hook with both filters
            # disabled (current upstream callers short-circuit earlier, but
            # warmup/dummy runs have reached here historically) still runs
            # the kernel once with k=V / p=1.0 no-op values, so the per-core
            # candidate buffer and lookup tables are allocated before the
            # first real sampling step.
            k_warmup = logits.new_full((logits.shape[0],), logits.shape[1], dtype=torch.int32)
            p_warmup = logits.new_ones((logits.shape[0],), dtype=torch.float32)
            return apply_top_k_top_p_triton(logits, k_warmup, p_warmup)
        return apply_top_k_top_p_triton(logits, k, p)

    if k is None and p is None:
        return logits
    # use pytorch ops
    return apply_top_k_top_p_pytorch(logits, k, p)
