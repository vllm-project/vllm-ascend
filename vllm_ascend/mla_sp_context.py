from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from vllm.attention import AttentionMetadata
from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.distributed.parallel_state import get_mla_sp_world_group


@dataclass
class SPContext:
    num_global_tokens: int
    num_tokens_per_dp: int
    num_tokens_per_rank: int
    start_token_of_dp: list[int]
    end_token_of_dp: list[int]
    global_tokens: torch.Tensor
    dp_sp_start_token: list[int]
    dp_sp_end_token: list[int]
    rank_sp_start_token: list[int]
    rank_sp_end_token: list[int]
    my_dp: int
    my_rank: int
    my_rank_sp_start_token_within_dp: int
    my_rank_sp_end_token_within_dp: int
    num_my_dp_sp_tokens: int
    input_split_sizes: list[int]
    output_split_sizes: list[int]


_sp_context: Optional[SPContext] = None


def get_sp_context() -> Optional[SPContext]:
    return _sp_context


def set_sp_context(
    input_ids: torch.Tensor,
    attn_metadata: Optional[Union["AttentionMetadata",
                                  dict[str, "AttentionMetadata"]]] = None,
):
    global _sp_context
    _sp_context = None
    sp_enabled = True
    dp_group = get_dp_group()
    tp_group = get_tp_group()
    sp_world_group = get_mla_sp_world_group()
    assert sp_world_group.world_size > 1

    forward_context = get_forward_context()
    if forward_context.in_profile_run:
        return

    num_input_tokens = input_ids.shape[0]
    max_num_tokens_across_dp = forward_context.max_tokens_across_dp

    if attn_metadata is None:
        attn_metadata = forward_context.attn_metadata
    if attn_metadata is not None:
        if isinstance(attn_metadata, dict):
            attn_metadata = next(iter(attn_metadata.values()))
        has_decode = attn_metadata.num_decode_tokens > 0
        has_prefill = attn_metadata.num_prefills > 0
        if has_decode or not has_prefill:
            sp_enabled = False
    else:
        assert num_input_tokens == 1, "Length of dummy run must be 1."

    sp_metadata = torch.cat([
        torch.tensor([sp_enabled, num_input_tokens],
                     device=input_ids.device,
                     dtype=torch.int32),
        nn.functional.pad(input_ids,
                          (0, max_num_tokens_across_dp - num_input_tokens)),
    ]).unsqueeze(0)
    sp_metadata_across_dp = dp_group.all_gather(sp_metadata, 0)
    for i in range(dp_group.world_size):
        row = sp_metadata_across_dp[i]
        sp_enabled = bool(row[0] > 0)
        if not sp_enabled:
            return

    num_global_tokens = 0
    start_token_of_dp = []
    end_token_of_dp = []
    for i in range(dp_group.world_size):
        row = sp_metadata_across_dp[i]
        num_tokens = int(row[1].item())
        start_token_of_dp.append(num_global_tokens)
        num_global_tokens += num_tokens
        end_token_of_dp.append(num_global_tokens)

    num_tokens_per_rank = calc_div_ceil(num_global_tokens,
                                        sp_world_group.world_size)
    num_tokens_per_dp = num_tokens_per_rank * tp_group.world_size
    global_tokens = torch.empty(num_global_tokens,
                                dtype=input_ids.dtype,
                                device=input_ids.device)
    for i in range(dp_group.world_size):
        row = sp_metadata_across_dp[i]
        num_tokens = row[1]
        global_tokens[start_token_of_dp[i]:end_token_of_dp[i]] = row[
            2:num_tokens + 2]

    dp_sp_start_token = []
    dp_sp_end_token = []
    rank_sp_start_token = []
    rank_sp_end_token = []
    for i in range(dp_group.world_size):
        start_token = i * num_tokens_per_dp
        dp_sp_start_token.append(start_token)
        dp_sp_end_token.append(
            min(start_token + num_tokens_per_dp, num_global_tokens))
    for i in range(sp_world_group.world_size):
        start_token = i * num_tokens_per_rank
        rank_sp_start_token.append(start_token)
        rank_sp_end_token.append(
            min(start_token + num_tokens_per_rank, num_global_tokens))

    my_dp = dp_group.rank_in_group
    my_rank = sp_world_group.rank_in_group
    my_rank_sp_start_token_within_dp = tp_group.rank_in_group * num_tokens_per_rank
    my_rank_sp_end_token_within_dp = min(
        my_rank_sp_start_token_within_dp + num_tokens_per_rank,
        max(0, dp_sp_end_token[my_dp] - dp_sp_start_token[my_dp]))
    num_my_dp_sp_tokens = max(
        0, dp_sp_end_token[my_dp] - dp_sp_start_token[my_dp])

    tp_size = tp_group.world_size
    input_split_sizes = []
    output_split_sizes = []
    if dp_group.world_size > 1:
        my_rank_start_token = rank_sp_start_token[my_rank]
        my_rank_end_token = rank_sp_end_token[my_rank]
        my_dp_start_token = start_token_of_dp[my_dp]
        my_dp_end_token = end_token_of_dp[my_dp]
        for i in range(sp_world_group.world_size):
            other_rank_start_token = rank_sp_start_token[i]
            other_rank_end_token = rank_sp_end_token[i]
            send_start = max(my_dp_start_token, other_rank_start_token)
            send_end = min(my_dp_end_token, other_rank_end_token)
            send_len = max(0, send_end - send_start)
            input_split_sizes.append(send_len)

            other_dp_start_token = start_token_of_dp[i // tp_size]
            other_dp_end_token = end_token_of_dp[i // tp_size]
            receive_start = max(other_dp_start_token, my_rank_start_token)
            receive_end = min(other_dp_end_token, my_rank_end_token)
            receive_len = max(0, receive_end - receive_start)
            output_split_sizes.append(receive_len)

    forward_context.with_prefill = True
    forward_context.max_tokens_across_dp = num_tokens_per_dp
    forward_context.padded_num_tokens = calc_div_ceil(num_tokens_per_dp,
                                                      tp_size) * tp_size
    from vllm_ascend.ascend_forward_context import FusedMoEState
    if forward_context.fused_moe_state == FusedMoEState.NaiveMulticast:
        forward_context.fused_moe_state = FusedMoEState.AllGather
    dp_metadata = forward_context.dp_metadata
    if dp_metadata is not None:
        dp_metadata.max_tokens_across_dp_cpu.fill_(num_tokens_per_dp)
        cu_moe_tokens = 0
        for i in range(dp_group.world_size):
            cu_moe_tokens += max(dp_sp_end_token[i] - dp_sp_start_token[i], 1)
            dp_metadata.cu_tokens_across_dp_cpu[i] = cu_moe_tokens

    _sp_context = SPContext(
        num_global_tokens=num_global_tokens,
        num_tokens_per_dp=num_tokens_per_dp,
        num_tokens_per_rank=num_tokens_per_rank,
        start_token_of_dp=start_token_of_dp,
        end_token_of_dp=end_token_of_dp,
        global_tokens=global_tokens,
        dp_sp_start_token=dp_sp_start_token,
        dp_sp_end_token=dp_sp_end_token,
        rank_sp_start_token=rank_sp_start_token,
        rank_sp_end_token=rank_sp_end_token,
        my_dp=my_dp,
        my_rank=my_rank,
        my_rank_sp_start_token_within_dp=my_rank_sp_start_token_within_dp,
        my_rank_sp_end_token_within_dp=my_rank_sp_end_token_within_dp,
        num_my_dp_sp_tokens=num_my_dp_sp_tokens,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
    )


def calc_div_ceil(up: int, down: int) -> int:
    return (up + down - 1) // down
