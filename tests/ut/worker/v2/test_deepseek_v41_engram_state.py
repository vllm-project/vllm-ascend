# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 history preparation must not perform context-dependent lookups."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from vllm_ascend.models.deepseek_v41.engram import model_state as deepseek_v41
from vllm_ascend.ops.triton import engram_lookback
from vllm_ascend.worker.v2.model_states.default import AscendModelState


class EngramModel(torch.nn.Module):
    token_lookback_depth = 3

    def __init__(self):
        super().__init__()
        self.buffers_for_graph = {
            "engram_lookups": {0: torch.zeros(8, 9)},
            "engram_mask": torch.zeros(8, dtype=torch.bool),
        }
        self.prepare_engram_inputs = Mock(return_value=self.buffers_for_graph)

    def prime_engram_v2_graph_inputs(self, num_tokens):
        return self.buffers_for_graph

    def prepare_engram_graph_inputs(self, num_tokens):
        return self.buffers_for_graph

    def forward(self, **kwargs):
        return kwargs


@pytest.fixture
def state(monkeypatch, request):
    def init(self, config, model, encoder_cache, device):
        self.vllm_config = config
        self.model = model
        self.max_num_reqs = 4

    monkeypatch.setattr(AscendModelState, "__init__", init)
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda *args: {})
    monkeypatch.setattr(DefaultModelState, "prepare_dummy_inputs", lambda *args: {})
    monkeypatch.setattr(deepseek_v41, "ring_state_update_skipped", lambda: False)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=getattr(request, "param", 1))
    )
    return deepseek_v41.EngramModelState(config, EngramModel(), None, torch.device("cpu"))


def batch():
    return SimpleNamespace(
        num_tokens=3,
        is_dummy=False,
        num_tokens_after_padding=8,
        num_reqs=2,
        input_ids=torch.arange(8, dtype=torch.int32),
        positions=torch.arange(8),
        idx_mapping=torch.tensor([2, 0]),
        query_start_loc=torch.tensor([0, 1, 3, 8, 8], dtype=torch.int32),
    )


def test_history_preparation_defers_lookup_and_excludes_padding(state, monkeypatch):
    # CPU-only CI uses a Triton stub without numeric helpers.
    monkeypatch.setattr(deepseek_v41, "triton", SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()))
    launch = Mock()
    kernel = Mock()
    kernel.__getitem__ = Mock(return_value=launch)
    monkeypatch.setattr(engram_lookback, "_gather_lookback_kernel", kernel)
    reqs = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.arange(32).reshape(4, 8)),
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([5, 0, 2, 0])),
    )
    inputs = state.prepare_inputs(batch(), reqs)
    state.model.prepare_engram_inputs.assert_not_called()
    assert inputs["lookback_token_ids"] is state.lookback_token_ids
    assert state.lookback_token_ids.shape == (4, 3)
    assert state.lookback_token_ids.dtype == torch.int32
    args, kwargs = launch.call_args
    assert args[0] is state.lookback_token_ids
    assert args[2] is reqs.num_computed_tokens.gpu
    assert args[3] is reqs.all_token_ids.gpu
    assert args[5] == 2
    assert kwargs == {"DEPTH": 3, "BLOCK_DEPTH": 4}

    # Eager model entry is inside the runner's forward context.
    state.model(**inputs)
    call = state.model.prepare_engram_inputs.call_args
    assert call.kwargs["input_ids"].numel() == 3
    assert call.kwargs["positions"].numel() == 3
    assert call.kwargs["padded_tokens"] == 8
    torch.testing.assert_close(call.kwargs["query_start_loc"], torch.tensor([0, 1, 3], dtype=torch.int32))
    assert call.kwargs["lookback_token_ids"] is state.lookback_token_ids
    assert "block_table" not in call.kwargs
    # A FULL replay hook consumes the same pending preparation exactly once.
    state.prepare_engram()
    state.model.prepare_engram_inputs.assert_called_once()


def test_idle_rank_defers_collective_without_reading_request_history(state, monkeypatch):
    kernel = Mock()
    monkeypatch.setattr(engram_lookback, "_gather_lookback_kernel", kernel)
    state.kvpp_is_dummy_run = True
    state.lookback_token_ids.fill_(42)
    inputs = state.prepare_inputs(batch(), None)
    state.model.prepare_engram_inputs.assert_not_called()
    assert torch.all(state.lookback_token_ids == -1)
    state.prepare_engram()
    state.model.prepare_engram_inputs.assert_called_once()
    assert state.model.prepare_engram_inputs.call_args.kwargs["query_start_loc"] is None
    assert inputs["engram_mask"] is state.model.buffers_for_graph["engram_mask"]


def test_capture_discards_pending_lookup_and_resets_history(state):
    state.kvpp_is_dummy_run = True
    state.prepare_inputs(batch(), None)
    state.lookback_token_ids.fill_(42)
    result = state.prepare_dummy_inputs(2, 8)
    state.model(**result)
    state.model.prepare_engram_inputs.assert_not_called()
    assert torch.all(state.lookback_token_ids == -1)
    assert result["lookback_token_ids"] is state.lookback_token_ids


def test_model_selects_ascend_state():
    from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41LLMForCausalLM

    model = AscendDeepseekV41LLMForCausalLM.__new__(AscendDeepseekV41LLMForCausalLM)
    assert model.get_model_state_cls() is deepseek_v41.EngramModelState


def test_eager_hook_forwards_lookup_events(state):
    state.kvpp_is_dummy_run = True
    events = {0: object()}
    state.model.prepare_engram_inputs.return_value = dict(state.model.buffers_for_graph, engram_pending=events)
    inputs = state.prepare_inputs(batch(), None)
    result = state.model(**inputs)
    assert result["engram_pending"] is events
    state.prepare_engram()
    state.model.prepare_engram_inputs.assert_called_once()


def test_piecewise_capture_uses_stable_lookup_buffers_without_full_graph_producers(state):
    state._engram_graph_inputs = {
        "engram_query_start_loc": torch.zeros(5, dtype=torch.int32),
        "engram_valid_token_count": torch.zeros(1, dtype=torch.int32),
    }
    capture_inputs = {
        "input_ids": torch.zeros(8, dtype=torch.int32),
        "positions": torch.arange(8),
        "engram_pending": {0: object()},
        "engram_mask_ready_event": object(),
        "engram_graph_events": True,
        **state._engram_graph_inputs,
    }
    prepared = state.prepare_engram_piecewise_inputs(capture_inputs)
    assert prepared["engram_lookups"] is state.model.buffers_for_graph["engram_lookups"]
    assert prepared["engram_mask"] is state.model.buffers_for_graph["engram_mask"]
    assert "engram_query_start_loc" not in prepared
    assert "engram_valid_token_count" not in prepared
    assert "engram_pending" not in prepared
    assert "engram_mask_ready_event" not in prepared
    assert "engram_graph_events" not in prepared
    assert "engram_query_start_loc" in capture_inputs
    state.model.prepare_engram_inputs.assert_not_called()


def test_piecewise_replays_refresh_current_request_before_reading_stable_buffers(state):
    buffers = state.model.buffers_for_graph
    model_inputs = {"input_ids": torch.zeros(8, dtype=torch.int32), "positions": torch.arange(8)}
    readback = []

    def prepare(input_ids, **kwargs):
        assert kwargs["cg_mode"] == CUDAGraphMode.PIECEWISE
        buffers["engram_lookups"][0].fill_(int(input_ids[0]))
        return buffers

    state.model.prepare_engram_inputs.side_effect = prepare
    for token in [17, 29]:
        state._engram_inputs = {"input_ids": torch.tensor([token]), "cg_mode": CUDAGraphMode.PIECEWISE}
        prepared = state.prepare_engram_piecewise_inputs(model_inputs)
        assert prepared["engram_lookups"][0] is buffers["engram_lookups"][0]
        assert state._engram_inputs is None
        readback.append(prepared["engram_lookups"][0][0, 0].item())
        state.prepare_engram()
    assert readback == [17, 29]
    assert state.model.prepare_engram_inputs.call_count == 2


@pytest.mark.parametrize("failure", [None, "lookup", "replay"])
def test_piecewise_manager_reuses_graph_runner_after_model_preparation(monkeypatch, failure):
    from vllm_ascend.worker.v2 import aclgraph_utils as graph_mod

    calls = []
    model = object()
    original_inputs = {"input_ids": object()}
    prepared_inputs = {**original_inputs, "engram_mask": object()}
    output = object()

    def prepare(inputs):
        assert inputs is original_inputs
        calls.append("lookup")
        if failure == "lookup":
            raise RuntimeError("lookup")
        return prepared_inputs

    def replay(self, replay_model, inputs):
        assert replay_model is model
        assert inputs is prepared_inputs
        calls.append("replay")
        if failure == "replay":
            raise RuntimeError("replay")
        return output

    monkeypatch.setattr(graph_mod.ModelCudaGraphManager, "run_pw_graph", replay)
    metadata = Mock()
    manager = graph_mod.ModelAclGraphManager.__new__(graph_mod.ModelAclGraphManager)
    manager.model_runner = SimpleNamespace(
        model_state=SimpleNamespace(prepare_engram_piecewise_inputs=prepare, device_metadata=metadata)
    )
    if failure is None:
        assert manager.run_pw_graph(model, original_inputs) is output
    else:
        with pytest.raises(RuntimeError, match=failure):
            manager.run_pw_graph(model, original_inputs)
    assert calls == (["lookup"] if failure == "lookup" else ["lookup", "replay"])
    metadata.finish.assert_called_once_with()


@pytest.mark.parametrize("state", [1, 2, 4], indirect=True)
@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_replay_metadata_follows_local_segments_and_preserves_global_history(state, mode, monkeypatch):
    state._replay_start_np[:] = [11, 22, 33, 44]
    state._replay_groups = (128, torch.empty(0, dtype=torch.int32))
    pcp_enabled = state.vllm_config.parallel_config.prefill_context_parallel_size > 1
    local_capacity = 8 if pcp_enabled else 4
    # Reordered, duplicated global indices model PCP's local segments.
    mapping = np.array([3, 1, 3, 1, 0, 2, 0, 2] if pcp_enabled else [3, 1, 0])
    if pcp_enabled and mode == CUDAGraphMode.FULL:
        mapping = np.array([3, 1, 3, 1, 0, 0, 2])
    is_prefilling = np.ones(mapping.size, dtype=np.bool_)
    if not pcp_enabled or mode == CUDAGraphMode.FULL:
        is_prefilling[-1] = False
    batch = SimpleNamespace(
        num_reqs=mapping.size,
        num_reqs_after_padding=local_capacity,
        is_prefilling_np=is_prefilling,
        idx_mapping_np=mapping,
    )

    def prepare_attn(self, *args, model_specific_attn_metadata=None, **kwargs):
        return {"replay_start": model_specific_attn_metadata.replay_start}

    monkeypatch.setattr(AscendModelState, "prepare_attn", prepare_attn)
    # Stale metadata and FULL padding must be cleared on every preparation.
    state._replay_start.np.fill(-1)
    result = state.prepare_attn(batch, mode, (), torch.empty((0, 1), dtype=torch.int32), [], None)
    expected = np.where(is_prefilling, state._replay_start_np[mapping], 0)
    if mode == CUDAGraphMode.FULL:
        expected = np.pad(expected, (0, local_capacity - mapping.size))
    torch.testing.assert_close(result["replay_start"], torch.from_numpy(expected))
    assert state._replay_start_np.tolist() == [11, 22, 33, 44]
    assert state.lookback_token_ids.shape == (4, 3)


def test_full_graph_replay_metadata_is_padded_and_inactive_without_replay(monkeypatch):
    """FULL graph metadata must cover padded rows and skip inactive replay."""

    class Buffer:
        def __init__(self, size):
            self.np = np.zeros(size, dtype=np.int32)
            self.gpu = torch.zeros(size, dtype=torch.int32)

        def copy_to_gpu(self, count):
            self.gpu[:count].copy_(torch.from_numpy(self.np[:count]))

    captured = {}

    def fake_prepare_attn(self, *args, **kwargs):
        captured["model_specific_attn_metadata"] = kwargs.get("model_specific_attn_metadata")
        return {}

    monkeypatch.setattr(AscendModelState, "prepare_attn", fake_prepare_attn)
    state = object.__new__(deepseek_v41.EngramModelState)
    state.decoder_replay = None
    state._replay_groups = (128, torch.empty(0, dtype=torch.int32))
    state._replay_start_np = np.zeros(4, dtype=np.int32)
    state._replay_start = Buffer(8)
    state.decoder_replay_state = None

    batch = SimpleNamespace(
        num_reqs=1,
        num_reqs_after_padding=8,
        is_prefilling_np=np.array([True]),
        idx_mapping_np=np.array([0], dtype=np.int32),
    )
    state.prepare_attn(
        batch,
        CUDAGraphMode.FULL,
        (),
        torch.empty((0, 1), dtype=torch.int32),
        [],
        None,
    )

    assert state._replay_start.gpu.shape == (8,)
    assert torch.count_nonzero(state._replay_start.gpu) == 0
    assert captured["model_specific_attn_metadata"] is None
