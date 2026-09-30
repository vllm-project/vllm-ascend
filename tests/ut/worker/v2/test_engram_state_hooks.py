# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 hashing and lookup at the FULL replay boundary."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from vllm_ascend.models.deepseek_v41 import model as model_mod
from vllm_ascend.worker.v2.model_states.default import AscendModelState


@pytest.fixture
def engram_model():
    model = model_mod.DeepseekV41Model.__new__(model_mod.DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(engram_layer_ids=[0], image_token_id=100, engram_max_ngram_size=4, engram_n_heads=1)
    model.engram_hash = Mock(use_slot_cache=False, lookback_depth=3)
    model.engram_hash.ensure_cache.return_value = True
    model.engram_hash.return_value = torch.ones(2, 1, 3, dtype=torch.int32)
    table = Mock(dim=1, n_hash_cols=3)
    table.embed_gathered.return_value = torch.ones(2, 3, 1)
    model.layers = [SimpleNamespace(engram=SimpleNamespace(embed_tokens=table))]
    model.engram_dp_shared_memory = True
    model.engram_rotation = torch.eye(32)
    model._engram_input_buffers = None
    model._engram_max_tokens = 8
    return model


def test_mrv2_hashes_without_slot_metadata(engram_model, monkeypatch):
    monkeypatch.setattr(model_mod, "gather_engram_hashes", lambda hashes, **kwargs: hashes)
    history = torch.tensor([[7, 6, 5]], dtype=torch.int32)
    result = engram_model.prepare_engram_inputs(
        torch.tensor([8, 9]),
        torch.tensor([3, 4]),
        padded_tokens=8,
        query_start_loc=torch.tensor([0, 2]),
        lookback_token_ids=history,
    )
    engram_model.engram_hash.assert_called_once()
    assert engram_model.engram_hash.call_args.args[4] is history
    assert engram_model.engram_hash.call_args.args[6:] == (None, None)
    assert result["engram_mask"].tolist() == [True, True, False, False, False, False, False, False]
    assert torch.count_nonzero(result["engram_lookups"][0][2:]) == 0


def test_mrv2_rejects_missing_history(engram_model, monkeypatch):
    monkeypatch.setattr(model_mod, "gather_engram_hashes", lambda hashes, **kwargs: hashes)
    with pytest.raises(RuntimeError, match="lookback_token_ids"):
        engram_model.prepare_engram(
            torch.tensor([8, 9]),
            torch.tensor([3, 4]),
            query_start_loc=torch.tensor([0, 2]),
            block_table=torch.ones(1, 1),
        )


def test_v1_still_requires_slot_metadata(engram_model):
    engram_model.engram_hash.use_slot_cache = True
    engram_model.prepare_engram(torch.tensor([8, 9]), torch.tensor([3, 4]), query_start_loc=torch.tensor([0, 2]))
    engram_model.engram_hash.assert_not_called()


def test_generic_state_does_not_prepare_engram(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda *args: {"positions": None})
    state = AscendModelState.__new__(AscendModelState)
    state.model = Mock()
    assert state.prepare_inputs(None, None) == {"positions": None}
    state.model.prepare_engram_inputs.assert_not_called()


@pytest.mark.parametrize("updatable", [False, True])
def test_full_replay_refreshes_engram_inside_dp_context(monkeypatch, updatable):
    import vllm.forward_context as context_mod

    from vllm_ascend.worker.v2 import aclgraph_utils as graph_mod

    calls = []

    def prepare_engram():
        ctx = get_forward_context()
        assert ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL
        assert ctx.dp_metadata.num_tokens_across_dp_cpu.tolist() == [8, 8]
        calls.append("lookup")

    manager = graph_mod.ModelAclGraphManager.__new__(graph_mod.ModelAclGraphManager)
    manager.update_stream = Mock()
    manager.model_runner = SimpleNamespace(
        dp_size=2, attn_groups=[], model_state=SimpleNamespace(attn_metadata={}, prepare_engram=prepare_engram)
    )
    manager.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2, data_parallel_rank=0, is_moe_model=True),
        compilation_config=SimpleNamespace(fast_moe_cold_start=False, static_forward_context={}),
    )
    monkeypatch.setattr(graph_mod, "set_current_vllm_config", lambda *args: nullcontext())
    monkeypatch.setattr(graph_mod, "_get_graph_update_backend", lambda *args: object())
    monkeypatch.setattr(graph_mod, "use_updatable_graph", lambda *args: updatable)
    monkeypatch.setattr(context_mod.current_platform, "set_additional_forward_context", lambda **kwargs: {})
    manager._updatable_graph_replay = lambda *args: calls.append("replay")
    manager._graph_relay = lambda *args: calls.append("replay")
    assert not is_forward_context_available()
    manager.run_fullgraph(SimpleNamespace(num_tokens=8, cg_mode=CUDAGraphMode.FULL))
    assert calls == ["lookup", "replay"]
    assert not is_forward_context_available()
