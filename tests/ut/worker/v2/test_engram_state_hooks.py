# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""AscendModelState engram injection hooks for MRV2 graph execution.

`prepare_inputs` must refresh the fixed-address engram buffers before every
replay/eager step, handing the step's device request coordinates
(query_start_loc / slot_mapping / block_table) to the eager routing
explicitly because the hook runs before set_forward_context. The stubs here
carry the model's real signature, so a contract drift fails the call instead
of silently routing wrong metadata. Dummy/profile batches route too: engram
routing joins a node-local collective spanning every DP group, so skipping it
on idle ranks deadlocks the busy ranks inside route_many's all_gather — their
coordinates resolve to an empty dict, which prepare_engram honors with empty
hashes before touching the n-gram store. `prepare_dummy_inputs` must bind
those buffers during FULL graph capture so the eager prepare_engram path
(ContextVar.get() inside) is never traced.
"""

from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState


def _prepare_engram_inputs_stub(
    input_ids,
    positions,
    padded_tokens=None,
    lookback_token_ids=None,
    query_start_loc=None,
    slot_mapping=None,
    block_table=None,
):
    # Signature mirror of DeepseekV41Model.prepare_engram_inputs (Ascend main):
    # create_autospec rejects calls carrying removed kwargs such as the old
    # ``history_inputs``.
    return {"engram_lookups": {}, "engram_mask": torch.empty(0)}


def _state(model, kvpp_is_dummy_run=False):
    from vllm_ascend.worker.v2.model_states import default

    state = default.AscendModelState.__new__(default.AscendModelState)
    state.model = model
    state.kvpp_is_dummy_run = kvpp_is_dummy_run
    return state


def _batch(num_tokens=8, num_reqs=2):
    return SimpleNamespace(
        input_ids=torch.arange(num_tokens, dtype=torch.int32),
        positions=torch.arange(num_tokens, dtype=torch.int64),
        num_tokens_after_padding=num_tokens,
        num_reqs=num_reqs,
        query_start_loc=torch.tensor([0, 4, 8][: num_reqs + 1], dtype=torch.int32),
    )


def _v41_model():
    model = SimpleNamespace()
    model.engram_cache_layer_name = "model.layers.0.attn"
    model.prepare_engram_inputs = create_autospec(_prepare_engram_inputs_stub)
    model.prepare_engram_graph_inputs = Mock(return_value={"engram_lookups": {}, "engram_mask": torch.empty(0)})
    return model


def test_prepare_inputs_skips_models_without_engram(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda self, batch, reqs: {"positions": None})
    state = _state(SimpleNamespace())  # Non-V4.1 model: no engram methods.
    assert state.prepare_inputs(_batch(), req_states=None) == {"positions": None}


def test_prepare_inputs_routes_real_steps_with_device_inputs(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda self, batch, reqs: {})
    model = _v41_model()
    # Bare state (prepare_attn never ran): the coordinate lookup degrades to
    # no metadata instead of raising, and the routing still happens.
    state = _state(model, kvpp_is_dummy_run=False)
    batch = _batch(num_tokens=8)

    result = state.prepare_inputs(batch, req_states=None)

    model.prepare_engram_inputs.assert_called_once()
    args, kwargs = model.prepare_engram_inputs.call_args
    assert args[2] == 8
    assert kwargs == {}
    # Compare field-by-field: dict == dict would bool() the empty mask tensor
    # and raise "Boolean value of Tensor with no values is ambiguous".
    assert set(result) == {"engram_lookups", "engram_mask"}
    assert result["engram_lookups"] == {}
    assert result["engram_mask"].shape == (0,)


def test_prepare_inputs_skips_unknown_engram_group(monkeypatch):
    """A stale engram_cache_layer_name outside every group degrades to no metadata."""
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda self, batch, reqs: {})
    model = _v41_model()
    state = _state(model, kvpp_is_dummy_run=False)
    state.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["other.layer"])])
    state.block_tables = (torch.zeros(1, 1),)
    state.slot_mappings = torch.zeros(1, 4)

    state.prepare_inputs(_batch(), req_states=None)

    _, kwargs = model.prepare_engram_inputs.call_args
    assert kwargs == {}


def test_prepare_inputs_passes_device_coordinates_from_cached_views(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda self, batch, reqs: {})
    model = _v41_model()
    state = _state(model, kvpp_is_dummy_run=False)
    state.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["model.layers.0.attn"]),
            SimpleNamespace(layer_names=["model.layers.1.attn"]),
        ]
    )
    state.block_tables = (torch.tensor([[5, 6], [7, 8]]), torch.tensor([[9, 10]]))
    state.slot_mappings = torch.arange(16).reshape(2, 8)
    batch = _batch(num_tokens=8, num_reqs=2)

    state.prepare_inputs(batch, req_states=None)

    _, kwargs = model.prepare_engram_inputs.call_args
    # Full-request coordinates from the engram group's per-step device views;
    # the stub signature rejects removed kwargs such as ``history_inputs``.
    assert kwargs["query_start_loc"] is batch.query_start_loc
    torch.testing.assert_close(kwargs["slot_mapping"], torch.arange(8))
    torch.testing.assert_close(kwargs["block_table"], torch.tensor([[5, 6], [7, 8]]))


def test_prepare_inputs_uses_pcp_global_coordinates(monkeypatch):
    """PCP-local boundaries describe a token shard, not the request history."""
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda self, batch, reqs: {})
    model = _v41_model()
    state = _state(model, kvpp_is_dummy_run=False)
    state.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=["model.layers.0.attn"])])
    global_batch = _batch(num_tokens=16, num_reqs=2)
    global_batch.query_start_loc = torch.tensor([0, 7, 16], dtype=torch.int32)
    pcp_context = SimpleNamespace(
        global_batch=global_batch,
        global_block_tables=(torch.tensor([[1, 2], [3, 4]]),),
        global_slot_mappings=torch.arange(32).reshape(1, 32),
    )
    state.pcp_context = pcp_context
    batch = _batch(num_tokens=8, num_reqs=2)

    state.prepare_inputs(batch, req_states=None)

    _, kwargs = model.prepare_engram_inputs.call_args
    assert kwargs["query_start_loc"] is pcp_context.global_batch.query_start_loc
    torch.testing.assert_close(kwargs["slot_mapping"], pcp_context.global_slot_mappings[0])
    torch.testing.assert_close(kwargs["block_table"], pcp_context.global_block_tables[0])


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
    # Dummy batches route with empty hashes: no device coordinates keeps the
    # n-gram store clean while route_many still joins the collective.
    assert kwargs == {}
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
