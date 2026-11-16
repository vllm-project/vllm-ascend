# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise production request-history layouts across the int32 offset boundary."""

import pytest
import torch
from vllm.config import SchedulerConfig
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.sample.thinking_budget import (
    _load_effective_token_ascend,
    _update_committed_marker_cache_kernel_ascend,
)
from vllm_ascend.worker.v2.states import AscendRequestState

DEVICE = torch.device("npu")
SMALL_HISTORY_SHAPE = (3, 8)
LARGE_HISTORY_SHAPE = (8193, 262144)
HISTORY_MEMORY_HEADROOM = 1024**3
START = 90
END = 91


@triton.jit
def _load_history_token(history, stride, requests, output, inputs):
    slot = tl.load(requests)
    token = _load_effective_token_ascend(history, stride, inputs, 0, slot, 5, 1)
    tl.store(output, token)


@pytest.mark.parametrize("max_num_seqs,max_model_len", [SMALL_HISTORY_SHAPE, LARGE_HISTORY_SHAPE])
@pytest.mark.parametrize("operation", ["load", "cold_scan", "incremental_scan"])
def test_thinking_budget_history_addressing(max_num_seqs, max_model_len, operation):
    config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max(8, max_num_seqs),
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        enable_chunked_prefill=True,
    )
    required_bytes = max_num_seqs * max_model_len * torch.tensor([], dtype=torch.int32).element_size()
    free_bytes, _ = torch.npu.mem_get_info()
    if free_bytes < required_bytes + HISTORY_MEMORY_HEADROOM:
        pytest.skip("History-address regression needs the configured history allocation plus 1 GiB headroom")

    # Use the production allocator and request insertion, not an artificial
    # as_strided view. The first request takes the highest free slot, so a
    # large configured capacity can cross int32 with just one active request.
    state = AscendRequestState(
        max_num_reqs=config.max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=config.max_num_batched_tokens,
        num_speculative_steps=0,
        vocab_size=128,
        device=DEVICE,
    )
    state.add_request("req", 1, [1, START, 10, 11, 12], 5, 3)
    state.apply_staged_writes()
    slot = state.req_id_to_index["req"]
    history = state.all_token_ids.gpu
    assert state.num_reqs == 1
    assert slot == max_num_seqs - 1
    assert history.is_contiguous()
    assert history.shape == (max_num_seqs, max_model_len)
    assert history.stride() == (max_model_len, 1)
    if (max_num_seqs, max_model_len) == LARGE_HISTORY_SHAPE:
        assert slot * history.stride(0) == 2**31
    assert history[slot, :5].cpu().tolist() == [1, START, 10, 11, 12]

    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device=DEVICE)

    requests = tensor([slot])
    if operation == "load":
        output = tensor([0])
        _load_history_token[(1,)](history, history.stride(0), requests, output, tensor([12]))
        assert output.cpu().tolist() == [START]
        return

    incremental = operation == "incremental_scan"
    if incremental:
        state.all_token_ids.stage_write(slot, 5, [END])
        state.total_len.stage_write_elem(slot, 6)
        state.apply_staged_writes()

    starts = tensor([17, 18] + [17] * (slot - 2) + [1 if incremental else -1])
    ends = tensor([19, 20] + [19] * (slot - 2) + [-1])
    scans = tensor([21, 22] + [21] * (slot - 2) + [5 if incremental else 0])
    budgets = tensor([-1] * slot + [3])
    _update_committed_marker_cache_kernel_ascend[(1,)](
        requests,
        budgets,
        history,
        history.stride(0),
        state.total_len.gpu,
        starts,
        ends,
        scans,
        tensor([START]),
        tensor([END]),
        START_LEN=1,
        NATURAL_END_LEN=1,
        MAX_LEN=1,
        BLOCK=256,
    )
    assert starts.cpu().tolist() == [17, 18] + [17] * (slot - 2) + [1]
    assert ends.cpu().tolist() == [19, 20] + [19] * (slot - 2) + [5 if incremental else -1]
    assert scans.cpu().tolist() == [21, 22] + [21] * (slot - 2) + [6 if incremental else 5]
