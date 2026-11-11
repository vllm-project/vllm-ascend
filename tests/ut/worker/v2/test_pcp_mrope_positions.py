# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""PCP-aware M-RoPE positions in AscendModelState."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from vllm_ascend.worker.v2.model_states.default import AscendModelState


NUM_DIMS = 3
MAX_NUM_REQS = 2
MAX_MODEL_LEN = 8
DEVICE = torch.device("npu")

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")


def _make_rope_state() -> SimpleNamespace:
    prefill_positions = torch.zeros(
        (NUM_DIMS * MAX_NUM_REQS, MAX_MODEL_LEN), dtype=torch.int32, device=DEVICE
    )
    for req_idx in range(MAX_NUM_REQS):
        for dim in range(NUM_DIMS):
            base = 100 * (req_idx + 1) + 10 * (dim + 1)
            prefill_positions[
                req_idx * NUM_DIMS + dim, :MAX_MODEL_LEN
            ] = torch.arange(base, base + MAX_MODEL_LEN, device=DEVICE)
    return SimpleNamespace(
        num_dims=NUM_DIMS,
        positions=torch.zeros(
            (NUM_DIMS, 2 * MAX_MODEL_LEN + 1), dtype=torch.int64, device=DEVICE
        ),
        prefill_positions=SimpleNamespace(gpu=prefill_positions),
        prefill_delta=SimpleNamespace(
            gpu=torch.tensor([0, 2], dtype=torch.int32, device=DEVICE)
        ),
    )


def _make_state(monkeypatch, pcp_manager: object | None) -> AscendModelState:
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda self, batch, reqs: {})
    state = AscendModelState.__new__(AscendModelState)
    state.model = SimpleNamespace()
    state.rope_state = _make_rope_state()
    state.pcp_manager = pcp_manager
    return state


def test_pcp_prefill_slices_noncontiguous_global_mrope_positions(monkeypatch):
    state = _make_state(monkeypatch, object())
    input_batch = SimpleNamespace(
        num_tokens_after_padding=5,
        num_tokens=4,
        num_reqs=2,
        idx_mapping=torch.tensor([0, 0], dtype=torch.int64, device=DEVICE),
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32, device=DEVICE),
        positions=torch.tensor([0, 1, 3, 4, 0], dtype=torch.int64, device=DEVICE),
    )
    req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(
            gpu=torch.tensor([MAX_MODEL_LEN, MAX_MODEL_LEN], dtype=torch.int32, device=DEVICE)
        )
    )

    result = state.prepare_inputs(input_batch, req_states)

    expected = torch.zeros((NUM_DIMS, 5), dtype=torch.int64, device=DEVICE)
    expected[:, :2] = torch.tensor(
        [[110, 111], [120, 121], [130, 131]], dtype=torch.int64, device=DEVICE
    )
    expected[:, 2:4] = torch.tensor(
        [[113, 114], [123, 124], [133, 134]], dtype=torch.int64, device=DEVICE
    )
    assert torch.equal(result["positions"], expected)


def test_pcp_decode_adds_mrope_delta(monkeypatch):
    state = _make_state(monkeypatch, object())
    input_batch = SimpleNamespace(
        num_tokens_after_padding=3,
        num_tokens=2,
        num_reqs=1,
        idx_mapping=torch.tensor([1], dtype=torch.int64, device=DEVICE),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32, device=DEVICE),
        positions=torch.tensor([4, 5, 0], dtype=torch.int64, device=DEVICE),
    )
    req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(
            gpu=torch.tensor([MAX_MODEL_LEN, 4], dtype=torch.int32, device=DEVICE)
        )
    )

    result = state.prepare_inputs(input_batch, req_states)

    expected = torch.zeros((NUM_DIMS, 3), dtype=torch.int64, device=DEVICE)
    expected[:, :2] = torch.tensor([[6, 7]] * NUM_DIMS, dtype=torch.int64, device=DEVICE)
    assert torch.equal(result["positions"], expected)


def test_non_pcp_uses_upstream_positions(monkeypatch):
    state = _make_state(monkeypatch, None)
    sentinel = object()
    monkeypatch.setattr(
        DefaultModelState, "prepare_inputs", lambda self, batch, reqs: {"positions": sentinel}
    )

    result = state.prepare_inputs(SimpleNamespace(), SimpleNamespace())

    assert result["positions"] is sentinel
