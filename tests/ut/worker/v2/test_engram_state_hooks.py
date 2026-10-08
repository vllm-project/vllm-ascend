# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""AscendModelState engram injection hooks for MRV2 graph execution.

`prepare_inputs` must refresh the fixed-address engram buffers before every
replay/eager step, handing the step's device request coordinates
(query_start_loc / slot_mapping / block_table) to the eager routing
explicitly because the hook runs before set_forward_context. The stubs here
carry the model's real signature, so a contract drift fails the call instead
of silently routing wrong metadata. Dummy/profile batches route too: engram
DP-sharded lookups join the new Engram DP collective, so skipping them
on idle ranks deadlocks busy ranks inside the hash/row gathers. Their
coordinates resolve to an empty dict, which prepare_engram honors with empty
hashes before touching the n-gram store. `prepare_dummy_inputs` must bind
those buffers during FULL graph capture so the eager prepare_engram path
(ContextVar.get() inside) is never traced.
"""

from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import numpy as np
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
    local_token_indices=None,
    pre_forward=False,
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
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        idx_mapping=torch.arange(num_reqs),
        is_dummy=False,
        query_start_loc=torch.tensor([0, 4, 8][: num_reqs + 1], dtype=torch.int32),
    )


def _v41_model():
    model = SimpleNamespace()
    model.engram_cache_layer_name = "model.layers.0.attn"
    model.token_lookback_depth = 0
    model.prepare_engram_inputs = create_autospec(
        _prepare_engram_inputs_stub,
        return_value={"engram_lookups": {}, "engram_mask": torch.empty(0)},
    )
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
    assert kwargs == {"pre_forward": True}
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
    assert kwargs == {"pre_forward": True}


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
    # n-gram store clean while the DP lookup still joins the collective.
    assert kwargs == {"pre_forward": True}
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


def test_pcp_keeps_other_models_on_their_existing_engram_hook(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda *args: {})

    # This other model cannot accept V4.1's pre_forward or PCP arguments.
    def generic_hook(input_ids, positions, padded_tokens=None):
        return {}

    model = SimpleNamespace(prepare_engram_inputs=create_autospec(generic_hook, return_value={}))
    state = _state(model)
    state.pcp_manager = object()
    state.prepare_inputs(_batch(), req_states=None)
    assert model.prepare_engram_inputs.call_args.kwargs == {}


def test_pcp_hashes_global_requests_and_selects_local_rows(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda *args: {})
    model = _v41_model()
    model.token_lookback_depth = 3
    state = _state(model)
    state.pcp_manager = SimpleNamespace(pcp_rank=1)
    state.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=[model.engram_cache_layer_name])]
    )
    global_batch = SimpleNamespace(
        num_tokens=5,
        num_reqs=2,
        input_ids=torch.tensor([13, 17, 19, 23, 59]),
        positions=torch.tensor([3, 4, 5, 6, 4]),
        query_start_loc=torch.tensor([0, 4, 5]),
        idx_mapping=torch.tensor([2, 0]),
    )
    state.pcp_context = SimpleNamespace(
        global_batch=global_batch,
        padded_gather_idx=torch.tensor([0, 3, 4, 1, 2, 0]),
        global_block_tables=(torch.tensor([[0], [1]]),),
        global_slot_mappings=torch.arange(5).reshape(1, 5),
    )
    batch = _batch(num_tokens=3)
    batch.num_tokens = 2  # The third local row is padding, not another token.
    batch.is_dummy = False
    token_table = torch.zeros(3, 8, dtype=torch.long)
    token_table[2] = torch.tensor([0, 5, 9, 13, 17, 19, 23, 29])
    token_table[0] = torch.tensor([41, 43, 47, 53, 59, 61, 67, 71])
    req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=token_table),
        total_len=SimpleNamespace(gpu=torch.tensor([5, 0, 7])),
    )
    state.prepare_inputs(batch, req_states)
    args, kwargs = model.prepare_engram_inputs.call_args
    torch.testing.assert_close(args[0], global_batch.input_ids)
    assert kwargs["lookback_token_ids"].tolist() == [[9, 5, 0], [53, 47, 43]]
    assert kwargs["local_token_indices"].tolist() == [1, 2]
    assert kwargs["slot_mapping"].tolist() == list(range(5))
    assert kwargs["block_table"].tolist() == [[0], [1]]
    # A rank can own no local queries but must still route an empty slice of
    # this real global batch while the other PCP partition is working.
    batch.num_tokens = 0
    state.prepare_inputs(batch, req_states)
    assert model.prepare_engram_inputs.call_args.kwargs["local_token_indices"].numel() == 0


def test_pcp_dummy_ignores_previous_global_batch(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda *args: {})
    model = _v41_model()
    model.token_lookback_depth = 3
    state = _state(model)
    state.pcp_manager = SimpleNamespace(pcp_rank=1)
    state.pcp_context = object()  # No fields may be read from this old context.
    batch = _batch(num_tokens=2)
    batch.is_dummy = True
    state.prepare_inputs(batch, object())
    model.prepare_engram_inputs.assert_called_once()
    assert model.prepare_engram_inputs.call_args.kwargs == {"pre_forward": True}


def test_pcp1_prefill_chunk_decode_have_complete_accepted_lookback(monkeypatch):
    """MRV2 hash state is stateless: even PCP1 needs request history."""
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda *args: {})
    model = _v41_model()
    model.token_lookback_depth = 3
    state = _state(model)
    state.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=[model.engram_cache_layer_name])]
    )
    state.block_tables = (torch.tensor([[0, 1, 2]]),)
    state.slot_mappings = torch.arange(4).reshape(1, 4)
    tokens = torch.tensor([0, 5, 9, 13, 17, 19, 23, 29])
    token_table = torch.full((1, 12), 97)
    req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=token_table),
        total_len=SimpleNamespace(gpu=torch.tensor([0])),
    )
    for start, stop, expected in ((0, 3, [-1, -1, -1]), (3, 6, [9, 5, 0]), (6, 7, [19, 17, 13]), (7, 8, [23, 19, 17])):
        token_table[0, :stop] = tokens[:stop]
        req_states.total_len.gpu[0] = stop
        batch = _batch(num_tokens=4, num_reqs=1)
        batch.num_tokens = stop - start
        batch.input_ids[: batch.num_tokens] = tokens[start:stop]
        batch.positions[: batch.num_tokens] = torch.arange(start, stop)
        batch.query_start_loc = torch.tensor([0, stop - start])
        state.prepare_inputs(batch, req_states)
        args, kwargs = model.prepare_engram_inputs.call_args
        assert args[0].tolist() == tokens[start:stop].tolist()
        assert args[2] == 4
        assert kwargs["lookback_token_ids"].tolist() == [expected]
        assert kwargs["lookback_token_ids"].device == batch.positions.device
        assert kwargs["slot_mapping"].numel() == stop - start


def test_pcp1_dummy_never_reads_request_history(monkeypatch):
    monkeypatch.setattr(DefaultModelState, "prepare_inputs", lambda *args: {})
    model = _v41_model()
    model.token_lookback_depth = 3
    state = _state(model)
    batch = _batch()
    batch.is_dummy = True
    state.prepare_inputs(batch, object())
    assert model.prepare_engram_inputs.call_args.kwargs == {"pre_forward": True}


def test_prepare_attn_accepts_truly_empty_pcp_rank(monkeypatch):
    from vllm_ascend.worker.v2.model_states import default

    builder = Mock(return_value={"empty": True})
    monkeypatch.setattr(default, "build_attn_metadata", builder)
    state = _state(_v41_model())
    state.max_model_len = 32
    state.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=2))
    state.pcp_manager = SimpleNamespace(build_attention_context=Mock(return_value=object()))
    batch = _batch(num_tokens=1, num_reqs=0)
    batch.num_tokens = 0
    batch.num_scheduled_tokens = np.empty(0, dtype=np.int32)
    batch.query_start_loc_np = np.array([0], dtype=np.int32)
    batch.is_prefilling_np = np.empty(0, dtype=np.bool_)
    batch.seq_lens = torch.empty(0, dtype=torch.int32)
    batch.seq_lens_np = np.empty(0, dtype=np.int32)
    batch.dcp_local_seq_lens = None
    batch.attn_state = None
    result = state.prepare_attn(batch, default.CUDAGraphMode.NONE, (), torch.empty((0, 1)), [], SimpleNamespace())
    assert result == {"empty": True}
    assert builder.call_args.kwargs["max_query_len"] == 0
    assert builder.call_args.kwargs["num_actual_tokens"] == 0
