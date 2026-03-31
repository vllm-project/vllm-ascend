from typing import Dict, Any

import torch
import pytest
from vllm.v1.worker.gpu.input_batch import post_update as post_update_gpu
from vllm_ascend.worker.v2.input_batch import post_update as post_update_npu


def generate_test_data(num_reqs: int, max_num_reqs: int, vocab_size: int, num_speculative_steps: int, device: str) -> \
        Dict[str, Any]:
    """
    Generate random test data.
    Return a dictionary containing all input tensors and the additional field 'expected_query_lens' for validation.
    """
    num_cols = num_speculative_steps + 1

    if num_reqs > max_num_reqs:
        raise ValueError("num_reqs cannot be larger than max_num_reqs")

    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    num_computed_tokens = torch.randint(0, 100, (max_num_reqs,), dtype=torch.int32, device=device)
    last_sampled_tokens = torch.randint(0, vocab_size, (max_num_reqs,), dtype=torch.int32, device=device)
    output_bin_counts = torch.randint(0, 10, (max_num_reqs, vocab_size), dtype=torch.int32, device=device)
    sampled_tokens = torch.randint(0, vocab_size, (num_reqs, num_speculative_steps + 1), dtype=torch.int32,
                                   device=device)
    num_sampled = torch.randint(1, num_speculative_steps + 2, (num_reqs,), dtype=torch.int32, device=device)
    num_rejected = torch.randint(0, num_speculative_steps + 1, (num_reqs,), dtype=torch.int32, device=device)
    num_rejected = torch.min(num_rejected, num_sampled - 1)

    query_lengths = torch.randint(1, 20, (num_reqs,), dtype=torch.int32, device=device)
    query_start_loc = torch.cat([
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.cumsum(query_lengths, dim=0)
    ])
    total_len = torch.randint(50, 200, (max_num_reqs,), dtype=torch.int32, device=device)

    max_model_len = 3000  # 或者可以从total_len的最大值获取
    all_token_ids = torch.randint(0, vocab_size, (max_num_reqs, max_model_len), dtype=torch.int32, device=device)

    return {
        "idx_mapping": idx_mapping,
        "num_computed_tokens": num_computed_tokens,
        "last_sampled_tokens": last_sampled_tokens,
        "output_bin_counts": output_bin_counts,
        "sampled_tokens": sampled_tokens,
        "num_sampled": num_sampled,
        "num_rejected": num_rejected,
        "query_start_loc": query_start_loc,
        "all_token_ids": all_token_ids,
        "total_len": total_len
    }


@pytest.mark.parametrize("num_reqs,max_num_reqs,vocab_size,num_speculative_steps", [
    (36, 36, 200, 2),
    (48, 48, 32000, 5),
    (128, 128, 32000, 5),
])
def test_post_update(num_reqs: int, max_num_reqs: int, vocab_size: int, num_speculative_steps: int):
    """Test _topk_log_softmax_kernel for computing log probabilities
    Args:
        batch_size: Number of sequences in the batch
        vocab_size: Size of the vocabulary
        num_logprobs: Number of tokens to compute log probabilities for
    """
    torch.manual_seed(42)

    post_update_params = ["idx_mapping",
                          "num_computed_tokens",
                          "last_sampled_tokens",
                          "output_bin_counts",
                          "sampled_tokens",
                          "num_sampled",
                          "num_rejected",
                          "query_start_loc",
                          "all_token_ids",
                          "total_len"
                          ]

    data = generate_test_data(num_reqs, max_num_reqs, vocab_size, num_speculative_steps, device="npu")
    kernel_inputs_gpu = {k: data[k].clone() for k in post_update_params}
    kernel_inputs_npu = {k: data[k].clone() for k in post_update_params}

    # Invoke Triton kernel
    post_update_gpu(**kernel_inputs_gpu)
    torch.npu.synchronize()

    post_update_npu(**kernel_inputs_npu)
    torch.npu.synchronize()

    # ========== Verify results ==========
    assert torch.allclose(kernel_inputs_gpu["output_bin_counts"], kernel_inputs_npu["output_bin_counts"], rtol=1e-3,
                          atol=1e-3), \
        f"Triton output differs from PyTorch reference.\n" \
        f"Max diff: {torch.max(torch.abs(kernel_inputs_npu['output_bin_counts'] - kernel_inputs_npu['output_bin_counts']))}\n" \
        f"Mean diff: {torch.mean(torch.abs(kernel_inputs_npu['output_bin_counts'] - kernel_inputs_npu['output_bin_counts']))}"

