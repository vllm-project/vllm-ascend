from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.distributed.parallel_state import (
    get_ascend_config, get_mla_dp_rebalancing_world_group)


@dataclass
class RebalancingContext:
    num_padded_global_tokens: int
    num_tokens_per_dp: int
    num_tokens_per_device: int
    start_token_of_dp: list[int]  # no pad, original
    end_token_of_dp: list[int]  # no pad, original
    global_tokens: torch.Tensor
    dp_sp_start_token: list[int]  # i * num_tokens_per_dp
    dp_sp_end_token: list[int]  # (i + 1) * num_tokens_per_dp
    device_sp_start_token: list[int]  # i * num_tokens_per_device
    device_sp_end_token: list[int]  # (i + 1) * num_tokens_per_device
    local_dp: int
    local_device: int
    local_device_sp_start_token_within_dp: int  # tp_group.rank_in_group * num_tokens_per_device
    local_device_sp_end_token_within_dp: int  # (tp_group.rank_in_group + 1) * num_tokens_per_device
    local_device_total_receive_len: int
    input_split_sizes: list[int]
    output_split_sizes: list[int]
    num_output_tokens: int


_mla_dp_rebalancing_context: Optional[RebalancingContext] = None


def get_mla_dp_rebalancing_context() -> Optional[RebalancingContext]:
    return _mla_dp_rebalancing_context


def set_mla_dp_rebalancing_context(input_ids: torch.Tensor):
    global _mla_dp_rebalancing_context
    _mla_dp_rebalancing_context = None
    feature_enabled = get_ascend_config().enable_mla_prefill_dp_rebalancing
    dp_group = get_dp_group()
    tp_group = get_tp_group()
    sp_world_group = get_mla_dp_rebalancing_world_group()
    assert sp_world_group.world_size > 1

    forward_context = get_forward_context()
    if forward_context.in_profile_run:
        return

    num_output_tokens = input_ids.shape[0]
    max_num_tokens_across_dp = forward_context.max_tokens_across_dp

    attn_metadata = forward_context.attn_metadata
    if attn_metadata is not None:
        if isinstance(attn_metadata, dict):
            attn_metadata = next(iter(attn_metadata.values()))
        has_decode = attn_metadata.num_decode_tokens > 0
        has_prefill = attn_metadata.num_prefills > 0
        if has_decode or not has_prefill:
            feature_enabled = False
        num_input_tokens = attn_metadata.num_actual_tokens
    else:
        num_input_tokens = 1

    input_ids = input_ids[:num_input_tokens]

    rebalancing_metadata = torch.cat([
        torch.tensor([feature_enabled, num_input_tokens],
                     device=input_ids.device,
                     dtype=torch.int32),
        nn.functional.pad(input_ids,
                          (0, max_num_tokens_across_dp - num_input_tokens)),
    ]).unsqueeze(0)
    rebalancing_metadata_across_dp = dp_group.all_gather(
        rebalancing_metadata, 0)
    for i in range(dp_group.world_size):
        row = rebalancing_metadata_across_dp[i]
        feature_enabled = bool(row[0] > 0)
        if not feature_enabled:
            return

    num_global_tokens = 0
    start_token_of_dp = []
    end_token_of_dp = []
    for i in range(dp_group.world_size):
        row = rebalancing_metadata_across_dp[i]
        num_tokens = int(row[1].item())
        start_token_of_dp.append(num_global_tokens)
        num_global_tokens += num_tokens
        end_token_of_dp.append(num_global_tokens)

    num_tokens_per_device = calc_div_ceil(num_global_tokens,
                                          sp_world_group.world_size)
    num_tokens_per_dp = num_tokens_per_device * tp_group.world_size
    num_padded_global_tokens = num_tokens_per_dp * dp_group.world_size
    global_tokens = torch.empty(num_padded_global_tokens,
                                dtype=input_ids.dtype,
                                device=input_ids.device)
    for i in range(dp_group.world_size):
        row = rebalancing_metadata_across_dp[i]
        num_tokens = row[1]
        global_tokens[start_token_of_dp[i]:end_token_of_dp[i]] = row[
            2:num_tokens + 2]

    dp_sp_start_token = []
    dp_sp_end_token = []
    device_sp_start_token = []
    device_sp_end_token = []
    for i in range(dp_group.world_size):
        dp_sp_start_token.append(i * num_tokens_per_dp)
        dp_sp_end_token.append((i + 1) * num_tokens_per_dp)
    for i in range(sp_world_group.world_size):
        device_sp_start_token.append(i * num_tokens_per_device)
        device_sp_end_token.append((i + 1) * num_tokens_per_device)

    local_dp = dp_group.rank_in_group
    local_device = sp_world_group.rank_in_group
    local_device_sp_start_token_within_dp = tp_group.rank_in_group * num_tokens_per_device
    local_device_sp_end_token_within_dp = (tp_group.rank_in_group +
                                           1) * num_tokens_per_device

    tp_size = tp_group.world_size
    input_split_sizes = []
    output_split_sizes = []
    if dp_group.world_size > 1:
        local_device_start_token = device_sp_start_token[local_device]
        local_device_end_token = device_sp_end_token[local_device]
        local_dp_start_token = start_token_of_dp[local_dp]
        local_dp_end_token = end_token_of_dp[local_dp]
        local_device_total_receive_len = 0
        for i in range(sp_world_group.world_size):
            other_device_start_token = device_sp_start_token[i]
            other_device_end_token = device_sp_end_token[i]
            send_start = max(local_dp_start_token, other_device_start_token)
            send_end = min(local_dp_end_token, other_device_end_token)
            send_len = max(0, send_end - send_start)
            input_split_sizes.append(send_len)

            other_dp_start_token = start_token_of_dp[i // tp_size]
            other_dp_end_token = end_token_of_dp[i // tp_size]
            receive_start = max(other_dp_start_token, local_device_start_token)
            receive_end = min(other_dp_end_token, local_device_end_token)
            receive_len = max(0, receive_end - receive_start)
            output_split_sizes.append(receive_len)
            local_device_total_receive_len += receive_len
        local_device_total_receive_len //= tp_size
    else:
        local_device_total_receive_len = num_tokens_per_device

    forward_context.with_prefill = True
    forward_context.max_tokens_across_dp = num_tokens_per_dp
    forward_context.padded_num_tokens = num_tokens_per_dp
    dp_metadata = forward_context.dp_metadata
    if dp_metadata is not None:
        dp_metadata.max_tokens_across_dp_cpu.fill_(num_tokens_per_dp)
        for i in range(dp_group.world_size):
            dp_metadata.cu_tokens_across_dp_cpu[i] = (i +
                                                      1) * num_tokens_per_dp

    _mla_dp_rebalancing_context = RebalancingContext(
        num_padded_global_tokens=num_padded_global_tokens,
        num_tokens_per_dp=num_tokens_per_dp,
        num_tokens_per_device=num_tokens_per_device,
        start_token_of_dp=start_token_of_dp,
        end_token_of_dp=end_token_of_dp,
        global_tokens=global_tokens,
        dp_sp_start_token=dp_sp_start_token,
        dp_sp_end_token=dp_sp_end_token,
        device_sp_start_token=device_sp_start_token,
        device_sp_end_token=device_sp_end_token,
        local_dp=local_dp,
        local_device=local_device,
        local_device_sp_start_token_within_dp=
        local_device_sp_start_token_within_dp,
        local_device_sp_end_token_within_dp=local_device_sp_end_token_within_dp,
        local_device_total_receive_len=local_device_total_receive_len,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        num_output_tokens=num_output_tokens,
    )


def calc_div_ceil(up: int, down: int) -> int:
    return (up + down - 1) // down


def pre_forward_for_dp_rebalancing(input_ids: torch.Tensor) -> torch.Tensor:
    set_mla_dp_rebalancing_context(input_ids)
    context = get_mla_dp_rebalancing_context()
    if context is None:
        return input_ids
    local_dp = context.local_dp
    return context.global_tokens[context.dp_sp_start_token[local_dp]:context.
                                 dp_sp_end_token[local_dp]]


def recover_output(hidden_states: torch.Tensor) -> torch.Tensor:
    context = get_mla_dp_rebalancing_context()
    assert context is not None
    local_dp = context.local_dp
    local_dp_start_token = context.start_token_of_dp[local_dp]
    local_dp_end_token = context.end_token_of_dp[local_dp]
    local_dp_sp_start_token = context.dp_sp_start_token[local_dp]
    local_dp_sp_end_token = context.dp_sp_end_token[local_dp]
    send = hidden_states[:max(
        0,
        min(local_dp_sp_end_token, context.end_token_of_dp[-1]) -
        local_dp_sp_start_token)]
    dp_group = get_dp_group()
    if dp_group.world_size == 1:
        return send
    input_split_sizes = []
    output_split_sizes = []
    for i in range(dp_group.world_size):
        other_dp_start_token = context.start_token_of_dp[i]
        other_dp_end_token = context.end_token_of_dp[i]
        send_start = max(local_dp_sp_start_token, other_dp_start_token)
        send_end = min(local_dp_sp_end_token, other_dp_end_token)
        send_len = max(0, send_end - send_start)
        input_split_sizes.append(send_len)

        other_dp_sp_start_token = context.dp_sp_start_token[i]
        other_dp_sp_end_token = context.dp_sp_end_token[i]
        receive_start = max(other_dp_sp_start_token, local_dp_start_token)
        receive_end = min(other_dp_sp_end_token, local_dp_end_token)
        receive_len = max(0, receive_end - receive_start)
        output_split_sizes.append(receive_len)
    output = torch.empty(
        [local_dp_end_token - local_dp_start_token, hidden_states.shape[1]],
        dtype=send.dtype,
        device=send.device)
    dist.all_to_all_single(
        output=output,
        input=send,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=dp_group.device_group,
    )
    return output


def post_forward_for_dp_rebalancing(
        hidden_states: torch.Tensor) -> torch.Tensor:
    context = get_mla_dp_rebalancing_context()
    if context is None:
        return hidden_states
    output = recover_output(hidden_states)
    if output.shape[0] < context.num_output_tokens:
        output = nn.functional.pad(
            output, (0, 0, 0, context.num_output_tokens - output.shape[0]))
    return output
