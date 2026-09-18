# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""AscendModelState engram injection hooks for MRV2 graph execution.

`prepare_inputs` must refresh the fixed-address engram buffers before every
replay/eager step (building the Runner-side history inputs that the reworked
prepare_engram contract expects). Dummy/profile batches route too: engram
routing joins a node-local collective spanning every DP group, so skipping
it on idle ranks deadlocks the busy ranks inside route_many's all_gather.
`prepare_dummy_inputs` must bind those buffers during FULL graph capture so
the eager prepare_engram path is never traced.
"""

from types import SimpleNamespace
from unittest.mock import Mock

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
    state = _state(model, kvpp_is_dummy_run=False)
    batch = _batch(num_tokens=8)

    result = state.prepare_inputs(batch, req_states=None)

    # Tensor args cannot go through assert_called_once_with: tensor __eq__
    # returns a tensor, which mock comparison misreads as inequality.
    # SimpleNamespace carries no engram_cache_layer_name, so the history
    # mirror yields None and engram routing runs with lookups only.
    model.prepare_engram_inputs.assert_called_once()
    args, kwargs = model.prepare_engram_inputs.call_args
    torch.testing.assert_close(args[0], batch.input_ids[:8])
    torch.testing.assert_close(args[1], batch.positions[:8])
    assert args[2] == 8
    assert kwargs == {"history_inputs": None}
    model.prepare_engram_graph_inputs.assert_not_called()
    assert result["engram_lookups"] == {}
    assert result["engram_mask"] is model.prepare_engram_inputs.return_value["engram_mask"]


def test_get_engram_history_inputs_mirrors_device_block_table():
    model = SimpleNamespace(engram_cache_layer_name="model.layers.2.self_attn.swa_cache_layer")
    group = SimpleNamespace(
        layer_names=["model.layers.2.self_attn.swa_cache_layer"],
        kv_cache_spec=SimpleNamespace(block_size=128),
    )
    kv_cache_config = SimpleNamespace(kv_cache_groups=[group])
    device_table = torch.arange(3 * 4, dtype=torch.int32).reshape(3, 4)
    batch = SimpleNamespace(
        num_reqs=3,
        query_start_loc_np=torch.tensor([0, 2, 5, 8]).numpy(),
        block_table=SimpleNamespace(input_block_tables=[device_table]),
    )
    state = _state(model)
    state._kv_cache_config = kv_cache_config

    boundaries, block_table, block_size = state._get_engram_history_inputs(batch)

    assert torch.equal(boundaries, torch.tensor([0, 2, 5, 8]))
    assert block_table.device.type == "cpu"
    assert torch.equal(block_table, device_table.cpu())
    assert block_size == 128


def test_get_engram_history_inputs_skips_non_engram_models():
    state = _state(SimpleNamespace())  # No engram_cache_layer_name.
    assert state._get_engram_history_inputs(_batch()) is None


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
    assert kwargs == {"history_inputs": None}
    model.prepare_engram_graph_inputs.assert_not_called()
    assert "engram_lookups" in result


def test_prepare_inputs_profile_dummy_without_attn_metadata(monkeypatch):
    """Profile dummies (skip_attn) never build metadata; the hook must not AttributeError."""
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda self, batch, reqs: {})
    model = _v41_model()
    state = _state(model, kvpp_is_dummy_run=True)
    del state.attn_metadata  # prepare_attn never ran for skip_attn profile dummies.

    state.prepare_inputs(_batch(num_tokens=8), req_states=None)

    model.prepare_engram_inputs.assert_called_once()
    assert model.prepare_engram_inputs.call_args.kwargs == {"metadata": None}


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
