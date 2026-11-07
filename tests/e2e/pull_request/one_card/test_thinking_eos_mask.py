# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project.
"""Ascend coverage for the V2 thinking EOS mask kernels (PR #20811)."""

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsContext
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.patch.worker.patch_thinking_eos_mask import ThinkingEosMaskV2State

DEVICE = torch.device("npu")
START = 90
END = 91
END_A = 92
END_B = 93
IMPLICIT = 94
EOS_1 = 95
EOS_2 = 96
VOCAB_SIZE = 128
REQ_IDX = 3


class MockReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END]
    natural_reasoning_end_token_ids = [END]


class MockMultiTokenEndReasoningConfig:
    reasoning_start_token_ids = [START]
    reasoning_end_token_ids = [END_A, END_B]
    natural_reasoning_end_token_ids = [END_A, END_B]


def _make_req_states(tokens: list[int], prompt_len: int = 1) -> RequestState:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=max(64, len(tokens) + 1),
        max_num_batched_tokens=16,
        num_speculative_steps=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    req_states.add_request(
        req_id="req",
        prompt_len=prompt_len,
        all_token_ids=tokens,
        num_computed_tokens=len(tokens),
        max_tokens=32,
    )
    req_states.apply_staged_writes()
    return req_states


def _make_state(
    monkeypatch,
    req_states: RequestState,
    reasoning_config,
    sampling_params=None,
    eos_ids=f"{EOS_1},{EOS_2}",
    implicit_end_ids=f"{IMPLICIT}",
) -> ThinkingEosMaskV2State:
    if eos_ids is None:
        monkeypatch.delenv("VLLM_ASCEND_THINKING_MASK_EOS_TOKEN_IDS", raising=False)
    else:
        monkeypatch.setenv("VLLM_ASCEND_THINKING_MASK_EOS_TOKEN_IDS", eos_ids)
    if implicit_end_ids is None:
        monkeypatch.delenv("VLLM_ASCEND_THINKING_MASK_IMPLICIT_END_TOKEN_IDS", raising=False)
    else:
        monkeypatch.setenv("VLLM_ASCEND_THINKING_MASK_IMPLICIT_END_TOKEN_IDS", implicit_end_ids)
    state = ThinkingEosMaskV2State(req_states, reasoning_config)
    state.add_request(REQ_IDX, sampling_params or SamplingParams())
    state.apply_staged_writes()
    return state


def _apply(
    state: ThinkingEosMaskV2State,
    logits: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    idx_mapping = torch.tensor([REQ_IDX], dtype=torch.int32, device=DEVICE)
    expanded_idx_mapping = torch.tensor([REQ_IDX] * num_rows, dtype=torch.int32, device=DEVICE)
    ctx = LogitsContext(
        expanded_idx_mapping=expanded_idx_mapping,
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping.cpu().numpy(),
        expanded_local_pos=torch.zeros(num_rows, dtype=torch.int32, device=DEVICE),
        input_ids=torch.zeros(num_rows, dtype=torch.int32, device=DEVICE),
        pos=torch.zeros(num_rows, dtype=torch.int32, device=DEVICE),
        seq_lens_upper_bound_np=np.zeros(1, dtype=np.int32),
    )
    state.apply(logits, ctx)
    return logits.cpu()


def _assert_masked(out: torch.Tensor, row: int):
    assert out[row, EOS_1] == pytest.approx(-float("inf"))
    assert out[row, EOS_2] == pytest.approx(-float("inf"))
    others = torch.cat((out[row, :EOS_1], out[row, EOS_2 + 1 :]))
    assert torch.all(others == 0)


def test_v2_thinking_mask_covers_every_row_of_in_think_request(monkeypatch):
    req_states = _make_req_states([1, START, 10, 11, 12])
    state = _make_state(monkeypatch, req_states, MockReasoningConfig())

    logits = torch.zeros((3, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, num_rows=3)

    for row in range(3):
        _assert_masked(out, row)


def test_v2_thinking_mask_stops_after_natural_end(monkeypatch):
    req_states = _make_req_states([1, START, 10, END, 20, 21])
    state = _make_state(monkeypatch, req_states, MockReasoningConfig())

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, num_rows=1)

    assert torch.all(out == 0)


def test_v2_thinking_mask_stops_after_implicit_end(monkeypatch):
    req_states = _make_req_states([1, START, 10, IMPLICIT, 20, 21])
    state = _make_state(monkeypatch, req_states, MockReasoningConfig())

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, num_rows=1)

    assert torch.all(out == 0)


def test_v2_thinking_mask_ignores_implicit_end_before_start(monkeypatch):
    req_states = _make_req_states([1, IMPLICIT, START, 10, 11])
    state = _make_state(monkeypatch, req_states, MockReasoningConfig())

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, num_rows=1)

    _assert_masked(out, 0)


def test_v2_thinking_mask_requires_full_multi_token_end(monkeypatch):
    partial = _make_state(
        monkeypatch,
        _make_req_states([1, START, 10, END_A]),
        MockMultiTokenEndReasoningConfig(),
    )
    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(partial, logits, num_rows=1)
    _assert_masked(out, 0)

    ended = _make_state(
        monkeypatch,
        _make_req_states([1, START, 10, END_A, END_B, 20]),
        MockMultiTokenEndReasoningConfig(),
    )
    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(ended, logits, num_rows=1)
    assert torch.all(out == 0)


def test_v2_thinking_mask_reenters_think_after_end(monkeypatch):
    req_states = _make_req_states([1, START, 10, END, 20, START, 21])
    state = _make_state(monkeypatch, req_states, MockReasoningConfig())

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, num_rows=1)

    _assert_masked(out, 0)


def test_v2_thinking_mask_ignores_request_without_start(monkeypatch):
    req_states = _make_req_states([1, 10, 11, 12])
    state = _make_state(monkeypatch, req_states, MockReasoningConfig())

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, num_rows=1)

    assert torch.all(out == 0)


def test_v2_thinking_mask_rescans_recycled_request_slot(monkeypatch):
    req_states = _make_req_states([1, START, 10, 11, 12])
    state = _make_state(monkeypatch, req_states, MockReasoningConfig())

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, num_rows=1)
    _assert_masked(out, 0)

    # A marker-free request reusing the slot must start cold, not from the
    # previous occupant's cached in-think state.
    req_states.remove_request("req")
    req_states.add_request(
        req_id="req2",
        prompt_len=1,
        all_token_ids=[1, 20, 21, 22],
        num_computed_tokens=4,
        max_tokens=32,
    )
    req_states.apply_staged_writes()
    state.add_request(REQ_IDX, SamplingParams())
    state.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, num_rows=1)
    assert torch.all(out == 0)


def test_v2_thinking_mask_incrementally_scans_long_generation(monkeypatch):
    tokens = [1, START, *([10] * 16382)]
    req_states = _make_req_states(tokens)
    state = _make_state(monkeypatch, req_states, MockReasoningConfig())

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, num_rows=1)
    _assert_masked(out, 0)
    assert state._mask_cached_scan_pos[REQ_IDX].item() == len(tokens)

    # Commit an implicit end marker and one more token; the incremental scan
    # must pick them up without a cold rescan.
    req_states.all_token_ids.stage_write(REQ_IDX, len(tokens), [IMPLICIT, 10])
    req_states.total_len.stage_write_elem(REQ_IDX, len(tokens) + 2)
    req_states.apply_staged_writes()

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, num_rows=1)
    assert torch.all(out == 0)


def test_v2_thinking_mask_coexists_with_budget_forcing(monkeypatch):
    req_states = _make_req_states([1, START, 10, 11, 12])
    state = _make_state(
        monkeypatch,
        req_states,
        MockReasoningConfig(),
        sampling_params=SamplingParams(thinking_token_budget=3),
    )

    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, num_rows=1)

    # Budget exhaustion forces the end marker; the mask still applies to the
    # configured EOS ids on the same row.
    assert out[0, END] == pytest.approx(1.0e9)
    assert out[0, EOS_1] == pytest.approx(-float("inf"))
    assert out[0, EOS_2] == pytest.approx(-float("inf"))


def test_v2_thinking_mask_keeps_stock_behavior_when_unset(monkeypatch):
    req_states = _make_req_states([1, START, 10, 11, 12])
    state = _make_state(monkeypatch, req_states, MockReasoningConfig(), eos_ids=None)

    assert not state._mask_enabled
    logits = torch.zeros((1, VOCAB_SIZE), device=DEVICE)
    out = _apply(state, logits, num_rows=1)

    assert torch.all(out == 0)
