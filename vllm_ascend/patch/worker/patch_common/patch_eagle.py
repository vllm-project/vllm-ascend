# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.v1.spec_decode.eagle import EagleProposer


def prepare_inputs(
    # [batch_size + 1]
    cu_target_query_lens: torch.Tensor,
    # [batch_size]
    num_rejected_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # cu_target_query_lens: [0, a, a + b, a + b + c]
    # num_rejected_tokens: [n1, n2, n3]
    # num_tokens_per_req: [a - n1, b - n2, c - n3]
    # cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
    # token_indices: [0, 1, ..., a - n1 - 1,
    #                 a, a + 1, ..., a + b - n2 - 1,
    #                 a + b, a + b + 1, ..., a + b + c - n3 - 1]

    # [0, a, a + b, a + b + c] -> [a, b, c]
    query_len_per_req = (cu_target_query_lens[1:] - cu_target_query_lens[:-1])
    # [a, b, c] -> [a - n1, b - n2, c - n3]
    num_tokens_per_req = query_len_per_req - num_rejected_tokens

    cu_num_tokens = torch.empty_like(cu_target_query_lens)
    torch.cumsum(num_tokens_per_req, dim=0, out=cu_num_tokens[1:])
    cu_num_tokens[0] = 0

    # FIXME(woosuk): Avoid synchronization.
    num_tokens = cu_num_tokens[-1].item()
    token_indices = torch.empty(
        num_tokens,
        dtype=torch.int32,
        device=cu_num_tokens.device,
    )

    BLOCK_SIZE = 1024
    prepare_input_pytorch(
        token_indices,
        cu_target_query_lens,
        cu_num_tokens,
        block_size=BLOCK_SIZE,
    )
    return cu_num_tokens, token_indices


def prepare_input_pytorch(out_ptr: torch.Tensor, cu_query_lens: torch.Tensor,
                          cu_num_tokens: torch.Tensor, block_size: int):
    num_pids = cu_num_tokens.shape[0] - 1

    for pid in range(num_pids):
        start_pos = cu_num_tokens[pid].item()
        end_pos = cu_num_tokens[pid + 1].item()
        num_tokens = end_pos - start_pos

        index_start = cu_query_lens[pid].item()
        num_blocks = (num_tokens + block_size - 1)

        for i in range(num_blocks):
            offset = torch.arange(0,
                                  block_size,
                                  dtype=out_ptr.dtype,
                                  device=cu_query_lens.device)
            global_indices = start_pos + offset
            values = index_start + offset
            mask = offset < num_tokens
            out_ptr[global_indices[mask]] = values[mask]


EagleProposer.prepare_inputs = prepare_inputs
