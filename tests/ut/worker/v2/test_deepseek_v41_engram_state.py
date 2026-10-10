# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 history preparation must not perform context-dependent lookups."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
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
def state(monkeypatch):
    def init(self, config, model, encoder_cache, device):
        self.model = model
        self.max_num_reqs = 4

    monkeypatch.setattr(AscendModelState, "__init__", init)
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda *args: {})
    monkeypatch.setattr(DefaultModelState, "prepare_dummy_inputs", lambda *args: {})
    monkeypatch.setattr(deepseek_v41, "ring_state_update_skipped", lambda: False)
    return deepseek_v41.EngramModelState(None, EngramModel(), None, torch.device("cpu"))


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


def test_prepare_attn_pads_replay_slots_and_forwards_replay_metadata(state, monkeypatch):
    from vllm.config.compilation import CUDAGraphMode

    from vllm_ascend.core import kv_cache_interface
    from vllm_ascend.worker.v2.model_states import default as ascend_model_state

    state.device = torch.device("cpu")
    state._replay_start_np[1] = 2
    monkeypatch.setattr(kv_cache_interface, "is_prefix_cacheable", lambda spec: True)

    slot_mappings = torch.zeros((1, 3), dtype=torch.int64)
    positions = torch.tensor([2, 3, 4], dtype=torch.int64)
    input_batch = SimpleNamespace(
        num_reqs=2,
        is_prefilling_np=np.array([True, True]),
        idx_mapping_np=np.array([1, 2]),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        positions=positions,
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(prefix_replay_tokens=4))]
    )

    launch = Mock()
    kernel = Mock()
    kernel.__getitem__ = Mock(return_value=launch)
    monkeypatch.setattr(deepseek_v41, "_pad_v2_replayed_slots_kernel", kernel)

    delegated = {}

    def prepare_attn(_self, *args, **kwargs):
        delegated.update(kwargs)
        return {"prepared": True}

    monkeypatch.setattr(AscendModelState, "prepare_attn", prepare_attn)
    result = state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        (torch.zeros((2, 3), dtype=torch.int32),),
        slot_mappings,
        [],
        kv_cache_config,
    )

    assert result == {"prepared": True}
    args, kwargs = launch.call_args
    assert args[0] is slot_mappings
    assert args[4] is positions
    assert args[5].tolist() == [2, 0]
    assert args[6] == 4
    assert kwargs == {"NUM_GROUPS": 1, "BLOCK": 1024}

    metadata = delegated["model_specific_attn_metadata"]
    assert isinstance(metadata, ascend_model_state.ReplayAttnMetadata)
    assert metadata.get_extra_common_attn_kwargs(0, 2)["replay_start"].tolist() == [2, 0]
