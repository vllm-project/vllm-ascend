# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""AscendModelState engram injection hooks for MRV2 graph execution.

`prepare_inputs` must refresh the fixed-address engram buffers before every
replay/eager step, handing the step's history inputs (CPU boundaries, block
table pages, storage block size) to the eager routing explicitly because the
hook runs before set_forward_context. Dummy/profile batches route too: engram
routing joins a node-local collective spanning every DP group, so skipping it
on idle ranks deadlocks the busy ranks inside route_many's all_gather — their
history inputs resolve to None instead, which prepare_engram honors before
touching the n-gram store. `prepare_dummy_inputs` must bind those buffers
during FULL graph capture so the eager prepare_engram path (ContextVar.get()
inside) is never traced.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState


def _state(model, kvpp_is_dummy_run=False):
    from vllm_ascend.worker.v2.model_states import default

    state = default.AscendModelState.__new__(default.AscendModelState)
    state.model = model
    state.kvpp_is_dummy_run = kvpp_is_dummy_run
    return state


def _batch(num_tokens=8):
    return SimpleNamespace(
        input_ids=torch.arange(num_tokens, dtype=torch.int32),
        positions=torch.arange(num_tokens, dtype=torch.int64),
        num_tokens_after_padding=num_tokens,
    )


def _v41_model():
    model = SimpleNamespace()
    model.engram_cache_layer_name = "model.layers.0.attn"
    model.prepare_engram_inputs = Mock(return_value={"engram_lookups": {}, "engram_mask": torch.empty(0)})
    model.prepare_engram_graph_inputs = Mock(return_value={"engram_lookups": {}, "engram_mask": torch.empty(0)})
    return model


def test_prepare_inputs_skips_models_without_engram(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda self, batch, reqs: {"positions": None})
    state = _state(SimpleNamespace())  # Non-V4.1 model: no engram methods.
    assert state.prepare_inputs(_batch(), req_states=None) == {"positions": None}


def test_prepare_inputs_routes_real_steps_with_history_inputs(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda self, batch, reqs: {})
    model = _v41_model()
    # Bare state (prepare_attn never ran): the history lookup degrades to
    # None instead of raising, and the routing still happens.
    state = _state(model, kvpp_is_dummy_run=False)
    batch = _batch(num_tokens=8)

    result = state.prepare_inputs(batch, req_states=None)

    model.prepare_engram_inputs.assert_called_once()
    args, kwargs = model.prepare_engram_inputs.call_args
    assert args[2] == 8
    assert kwargs == {"history_inputs": None}
    # Compare field-by-field: dict == dict would bool() the empty mask tensor
    # and raise "Boolean value of Tensor with no values is ambiguous".
    assert set(result) == {"engram_lookups", "engram_mask"}
    assert result["engram_lookups"] == {}
    assert result["engram_mask"].shape == (0,)


def test_prepare_inputs_computes_history_inputs_from_cached_views(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda self, batch, reqs: {})
    model = _v41_model()
    state = _state(model, kvpp_is_dummy_run=False)
    state.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["model.layers.0.attn"],
                kv_cache_spec=SimpleNamespace(block_size=16),
            )
        ]
    )
    state.block_tables = (torch.tensor([[5, 6], [7, 8]]),)
    batch = _batch(num_tokens=8)
    batch.num_reqs = 2
    batch.query_start_loc_np = np.array([0, 4, 8], dtype=np.int32)

    state.prepare_inputs(batch, req_states=None)

    _, kwargs = model.prepare_engram_inputs.call_args
    boundaries, block_table, block_size = kwargs["history_inputs"]
    torch.testing.assert_close(boundaries, torch.tensor([0, 4, 8], dtype=torch.int32))
    torch.testing.assert_close(block_table, torch.tensor([[5, 6], [7, 8]]))
    assert block_size == 16


def test_prepare_inputs_dummy_runs_route_too(monkeypatch):
    """Idle DP ranks must join the engram routing collective or busy ranks hang."""
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda self, batch, reqs: {})
    model = _v41_model()
    state = _state(model, kvpp_is_dummy_run=True)
    batch = _batch(num_tokens=8)

    result = state.prepare_inputs(batch, req_states=None)

    model.prepare_engram_inputs.assert_called_once()
    args, kwargs = model.prepare_engram_inputs.call_args
    assert args[2] == 8
    # Dummy batches route with empty hashes: history_inputs=None keeps the
    # n-gram store clean while route_many still joins the collective.
    assert kwargs == {"history_inputs": None}
    model.prepare_engram_graph_inputs.assert_not_called()
    assert "engram_lookups" in result


def test_prepare_dummy_inputs_binds_capture_buffers(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "prepare_dummy_inputs", lambda self, num_reqs, num_tokens: {})
    model = _v41_model()
    state = _state(model)

    result = state.prepare_dummy_inputs(num_reqs=4, num_tokens=64)

    model.prepare_engram_graph_inputs.assert_called_once_with(64)
    assert "engram_lookups" in result


def test_prepare_dummy_inputs_skips_models_without_engram(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "prepare_dummy_inputs", lambda self, num_reqs, num_tokens: {})
    state = _state(SimpleNamespace())

    assert state.prepare_dummy_inputs(num_reqs=4, num_tokens=64) == {}
