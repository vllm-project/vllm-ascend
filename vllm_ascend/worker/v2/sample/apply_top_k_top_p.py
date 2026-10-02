import torch

from vllm_ascend.sample.sampler import _apply_top_k_top_p_pytorch


def apply_top_k_top_p_npu(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    # NOTE: During the warmup stage, if both k and p are None, the kernel is
    # skipped. To keep its workspace inside the memory profiling measurement,
    # NPUModelRunner._dummy_sampler_run temporarily stages non-default
    # top_k/top_p on the dummy batch (see vllm_ascend/worker/v2/model_runner.py),
    # so this early return is not taken during profile_run.
    if k is None and p is None:
        return logits
    # use pytorch ops
    return _apply_top_k_top_p_pytorch(logits, k, p)
