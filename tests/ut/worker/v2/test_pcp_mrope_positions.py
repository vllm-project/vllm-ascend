# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""PCP-aware M-RoPE positions in AscendModelState."""

from types import SimpleNamespace

import numpy as np
import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from vllm_ascend.worker.v2.model_states.default import AscendModelState


NUM_DIMS = 3
MAX_NUM_REQS = 2
MAX_MODEL_LEN = 8


def _make_rope_state() -> SimpleNamespace:
    prefill_positions = torch.zeros(
        (NUM_DIMS * MAX_NUM_REQS, MAX_MODEL_LEN), dtype=torch.int32
    )
    for req_idx in range(MAX_NUM_REQS):
        for dim in range(NUM_DIMS):
            base = 100 * (req_idx + 1) + 10 * (dim + 1)
            prefill_positions[
                req_idx * NUM_DIMS + dim, :MAX_MODEL_LEN
            ] = torch.arange(base, base + MAX_MODEL_LEN)
    return SimpleNamespace(
        num_dims=NUM_DIMS,
        prefill_positions=SimpleNamespace(gpu=prefill_positions),
        prefill_delta=SimpleNamespace(np=np.array([0, 2], dtype=np.int32)),
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
        num_reqs=2,
        num_scheduled_tokens=np.array([2, 2], dtype=np.int32),
        idx_mapping_np=np.array([0, 0], dtype=np.int64),
        is_prefilling_np=np.array([True, True], dtype=np.bool_),
        num_computed_prefill_tokens_np=np.array([0, 3], dtype=np.int32),
        positions=torch.tensor([0, 1, 3, 4, 0], dtype=torch.int64),
    )

    result = state.prepare_inputs(input_batch, SimpleNamespace())

    expected = torch.zeros((NUM_DIMS, 5), dtype=torch.int64)
    expected[:, :2] = torch.tensor(
        [[110, 111], [120, 121], [130, 131]], dtype=torch.int64
    )
    expected[:, 2:4] = torch.tensor(
        [[113, 114], [123, 124], [133, 134]], dtype=torch.int64
    )
    assert torch.equal(result["positions"], expected)


def test_pcp_decode_adds_mrope_delta(monkeypatch):
    state = _make_state(monkeypatch, object())
    input_batch = SimpleNamespace(
        num_tokens_after_padding=3,
        num_reqs=1,
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        idx_mapping_np=np.array([1], dtype=np.int64),
        is_prefilling_np=np.array([False], dtype=np.bool_),
        num_computed_prefill_tokens_np=np.array([MAX_MODEL_LEN], dtype=np.int32),
        positions=torch.tensor([4, 5, 0], dtype=torch.int64),
    )

    result = state.prepare_inputs(input_batch, SimpleNamespace())

    expected = torch.zeros((NUM_DIMS, 3), dtype=torch.int64)
    expected[:, :2] = torch.tensor([[6, 7]] * NUM_DIMS, dtype=torch.int64)
    assert torch.equal(result["positions"], expected)


def test_non_pcp_uses_upstream_positions(monkeypatch):
    state = _make_state(monkeypatch, None)
    sentinel = object()
    monkeypatch.setattr(
        DefaultModelState, "prepare_inputs", lambda self, batch, reqs: {"positions": sentinel}
    )

    result = state.prepare_inputs(SimpleNamespace(), SimpleNamespace())

    assert result["positions"] is sentinel
