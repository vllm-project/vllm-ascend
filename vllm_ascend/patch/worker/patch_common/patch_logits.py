import torch
import vllm
from vllm._custom_ops import apply_repetition_penalties_torch


def apply_repetition_penalties(logits: torch.Tensor, prompt_mask: torch.Tensor,
                               output_mask: torch.Tensor,
                               repetition_penalties: torch.Tensor) -> None:
    """Apply repetition penalties to logits in-place.

    Args:
        logits: The logits tensor of shape [num_seqs, vocab_size].
        prompt_mask: A boolean tensor indicating which tokens appear in the prompt.
        output_mask: A boolean tensor indicating which tokens appear in the output.
        repetition_penalties: The repetition penalties of shape (num_seqs, ).
    """
    apply_repetition_penalties_torch(logits, prompt_mask, output_mask,
                                     repetition_penalties)


# NPU device type tensors have attributes is_cuda=True and is_npu=True, according to its implementation in
# https://github.com/Ascend/pytorch/blob/863b9071cbdf47023c12c246e3efa9c6e2285fc6/torch_npu/npu/_stream_check.py#L74
# This causes that vLLM's apply_repetition_penalties function will run into the branch of "if logits.is_cuda" and
# call the custom op implemented in CUDA, which is not compatible with NPU.
# Reference: https://github.com/vllm-project/vllm/blob/f66673a39d9f364194c249f28098cad8a5584ccb/vllm/_custom_ops.py#L314
vllm._custom_ops.apply_repetition_penalties = apply_repetition_penalties
