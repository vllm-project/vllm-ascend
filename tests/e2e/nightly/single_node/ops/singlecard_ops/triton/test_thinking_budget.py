# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused Ascend history-load and committed-marker scan correctness tests."""

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
START = 90
END = 91


@triton.jit
def _load_history_token(history, stride, requests, output, inputs, total_len, pos):
    slot = tl.load(requests)
    token = _load_effective_token_ascend(history, stride, inputs, 0, slot, total_len, pos)
    tl.store(output, token)


@pytest.mark.parametrize("max_num_seqs,max_model_len", [(3, 8), (2048, 4096)])
@pytest.mark.parametrize(
    "operation", ["committed", "inflight", "cold_scan", "incremental_scan", "disabled", "no_markers"]
)
def test_thinking_budget_history_and_marker_scans(max_num_seqs, max_model_len, operation):
    config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max(8, max_num_seqs),
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        enable_chunked_prefill=True,
    )
    state = AscendRequestState(
        max_num_reqs=config.max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=config.max_num_batched_tokens,
        num_speculative_steps=0,
        vocab_size=128,
        device=DEVICE,
    )
    tokens = [1, 10, 11, 12, 13] if operation == "no_markers" else [1, START, 10, 11, 12]
    state.add_request("req", 1, tokens, len(tokens), 3)
    state.apply_staged_writes()
    slot = state.req_id_to_index["req"]
    history = state.all_token_ids.gpu
    assert state.num_reqs == 1
    assert slot == max_num_seqs - 1
    assert history.is_contiguous()
    assert history.shape == (max_num_seqs, max_model_len)
    assert history.stride() == (max_model_len, 1)
    assert history[slot, : len(tokens)].cpu().tolist() == tokens

    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device=DEVICE)

    requests = tensor([slot])
    if operation in ("committed", "inflight"):
        output = tensor([0])
        committed = operation == "committed"
        _load_history_token[(1,)](
            history, history.stride(0), requests, output, tensor([12, 14]), len(tokens), 1 if committed else len(tokens)
        )
        assert output.cpu().tolist() == [START if committed else 14]
        return

    incremental = operation == "incremental_scan"
    if incremental:
        state.all_token_ids.stage_write(slot, 5, [END])
        state.total_len.stage_write_elem(slot, 6)
        state.apply_staged_writes()

    expected_starts = [17] * slot + [1 if incremental else -1]
    expected_ends = [19] * slot + [-1]
    expected_scans = [21] * slot + [5 if incremental else 0]
    starts, ends, scans = map(tensor, (expected_starts, expected_ends, expected_scans))
    budgets = tensor([-1] * slot + [-1 if operation == "disabled" else 3])
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
    if operation != "disabled":
        expected_starts[slot] = -1 if operation == "no_markers" else 1
        expected_ends[slot] = 5 if incremental else -1
        expected_scans[slot] = 6 if incremental else 5
    assert starts.cpu().tolist() == expected_starts
    assert ends.cpu().tolist() == expected_ends
    assert scans.cpu().tolist() == expected_scans
