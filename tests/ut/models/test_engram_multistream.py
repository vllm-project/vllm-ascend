# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.models.deepseek_v41 import model as model_mod
from vllm_ascend.models.deepseek_v41.engram import embedding, parallel


@pytest.fixture
def runtime(monkeypatch):
    calls: list[tuple[object, ...]] = []
    main = SimpleNamespace(
        name="main",
        wait_stream=lambda stream: calls.append(("join", stream.name)),
        wait_event=lambda event: calls.append(("wait_event", event)),
    )
    aux = SimpleNamespace(name="aux", wait_stream=lambda stream: calls.append(("reuse", stream.name)))
    current = [main]
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE, batch_descriptor="small")

    class Event:
        def record(self, stream):
            calls.append(("record", self, stream.name))

        def wait(self, stream):
            calls.append(("wait", self, stream.name))

        def reset(self, stream):
            calls.append(("reset", self, stream.name))

    @contextmanager
    def on_stream(stream):
        previous = current[0]
        current[0] = stream
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch.npu, "Event", Event)
    monkeypatch.setattr(torch.npu, "ExternalEvent", Event)
    monkeypatch.setattr(torch.npu, "Stream", lambda **kwargs: aux)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: current[0])
    monkeypatch.setattr(torch.npu, "stream", on_stream)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: calls.append(("retain", stream.name)))
    monkeypatch.setattr(model_mod, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(model_mod, "get_forward_context", lambda: context)
    return calls, context, aux


def make_model():
    model = object.__new__(model_mod.DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.has_engram, model.engram_dp_shared_memory = True, False
    model._engram_overlap_enabled = True
    model._engram_input_buffers, model._engram_prepare_stream = None, None
    model._engram_graph_events = {}
    model._engram_late_lookup_enabled, model._engram_late_lookup_stream = False, None
    model._engram_max_tokens = 8
    model.engram_rotation = torch.eye(32)
    model.config = SimpleNamespace(engram_layer_ids=(1, 2), image_token_id=999)
    model.layers = [
        SimpleNamespace(engram=SimpleNamespace(embed_tokens=SimpleNamespace(n_hash_cols=4, dim=2))) for _ in range(3)
    ]
    return model


def test_graph_events_are_reused_per_descriptor_and_primed_only_at_capture(runtime):
    calls, context, _ = runtime
    model = make_model()
    context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    first = model.prepare_engram_graph_inputs(4)
    assert len(calls) == 3
    replay = model.prepare_engram_graph_inputs(4, prime=False)
    assert len(calls) == 3
    context.batch_descriptor = "large"
    second = model.prepare_engram_graph_inputs(8)
    assert len(calls) == 6
    assert first["engram_pending"] is replay["engram_pending"]
    assert first["engram_mask_ready_event"] is replay["engram_mask_ready_event"]
    assert second["engram_mask_ready_event"] is not first["engram_mask_ready_event"]
    assert first["engram_lookups"] is second["engram_lookups"]
    for layer in (1, 2):
        assert first["engram_pending"][layer] is not second["engram_pending"][layer]
    event = first["engram_pending"][1]
    embedding.AscendParallelEngramEmbedding.wait_lookup(event, external=True)
    assert calls[-2:] == [("wait", event, "main"), ("reset", event, "main")]
    embedding.AscendParallelEngramEmbedding.wait_lookup(event)
    assert calls[-1] == ("wait_event", event)


@pytest.mark.parametrize("enabled,mode", [(True, "NONE"), (True, "FULL"), (False, "FULL"), (True, "PIECEWISE")])
def test_preparation_stream_and_buffer_reuse(runtime, enabled, mode):
    calls, context, aux = runtime
    context.cudagraph_runtime_mode = CUDAGraphMode[mode]
    model = make_model()
    model._engram_overlap_enabled = enabled
    expected_stream = "aux" if enabled and mode != "PIECEWISE" else "main"

    def prepare(*args, **kwargs):
        assert torch.npu.current_stream().name == expected_stream
        calls.append(("prepare",))
        assert kwargs["mask_output_buffer"] is model._engram_input_buffers[1]
        if expected_stream == "aux":
            assert kwargs["ready_events"] is not None
            assert kwargs["mask_ready_event"] is not None

    model.prepare_engram = prepare
    first = model.prepare_engram_inputs(torch.tensor([1]), torch.tensor([0]), 4)
    second = model.prepare_engram_inputs(torch.tensor([2]), torch.tensor([1]), 8)
    assert first["engram_lookups"] is second["engram_lookups"]
    if expected_stream == "aux":
        assert model._engram_prepare_stream is aux
        assert calls.count(("reuse", "main")) == 2
        assert calls.index(("reuse", "main")) < calls.index(("prepare",))
        assert not any(call[0] == "join" for call in calls)
        assert (first["engram_pending"] is second["engram_pending"]) == (mode == "FULL")
    else:
        assert model._engram_prepare_stream is None
        assert "engram_pending" not in first


def test_failed_producer_is_joined_and_capacity_checked_before_submission(runtime):
    calls, _, _ = runtime
    model = make_model()
    model.prepare_engram = Mock(side_effect=RuntimeError("lookup failed"))
    with pytest.raises(ValueError, match="capacity"):
        model.prepare_engram_inputs(torch.arange(9), torch.arange(9), 9)
    model.prepare_engram.assert_not_called()
    with pytest.raises(RuntimeError, match="lookup failed"):
        model.prepare_engram_inputs(torch.tensor([1]), torch.tensor([0]), 4)
    assert calls[-1] == ("join", "aux")
    model.retire_engram_lookups()
    assert calls[-1] == ("join", "aux")


def test_failed_graph_preparation_clears_unconsumed_events_after_join(runtime):
    calls, context, _ = runtime
    context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    model = make_model()
    binding = model.prepare_engram_graph_inputs(4)
    calls.clear()
    model.prepare_engram = Mock(side_effect=RuntimeError("lookup failed"))
    with pytest.raises(RuntimeError, match="lookup failed"):
        model.prepare_engram_inputs(torch.tensor([1]), torch.tensor([0]), 4)
    join = calls.index(("join", "aux"))
    assert calls[join + 1 :] == [
        ("reset", event, "main") for event in (binding["engram_mask_ready_event"], *binding["engram_pending"].values())
    ]


@pytest.mark.parametrize("participates", [False, True])
def test_shared_direct_lookup_writes_contiguous_ids_and_clears_padding(runtime, monkeypatch, participates):
    model = make_model()
    model.engram_dp_shared_memory = True
    hashes = torch.arange(2 * 2 * 4).reshape(2, 2, 4)
    model.engram_hash = Mock()
    model.engram_hash.ensure_cache.return_value = True
    model.engram_hash.return_value = hashes
    model.engram_hash.lookback_depth = 1
    monkeypatch.setattr(model_mod, "gather_engram_hashes", lambda ids, **kwargs: ids)
    for slot, layer in enumerate((1, 2)):
        table = model.layers[layer].engram.embed_tokens
        table.dp_size, table.tp_size = 1, 1

        def lookup(ids, output, slot=slot):
            assert ids.is_contiguous()
            torch.testing.assert_close(ids, hashes[:, slot])
            output.copy_(ids[:, :, None].expand(-1, -1, 2))

        table.lookup = Mock(side_effect=lookup)
    binding = model.prepare_engram_graph_inputs(4, prime=False)
    for tensor in binding["engram_lookups"].values():
        tensor.fill_(-99)
    result = model.prepare_engram_inputs(
        torch.tensor([1, 2]),
        torch.arange(2),
        4,
        query_start_loc=torch.tensor([0, 2] if participates else [0]),
        block_table=torch.zeros((1 if participates else 0, 1), dtype=torch.int32),
    )
    for slot, layer in enumerate((1, 2)):
        rows = result["engram_lookups"][layer]
        if participates:
            torch.testing.assert_close(rows[:2], hashes[:, slot, :, None].expand(-1, -1, 2).flatten(1).bfloat16())
        else:
            model.layers[layer].engram.embed_tokens.lookup.assert_not_called()
        assert not rows[2 if participates else 0 : 4].any()


@pytest.mark.parametrize("idle", [False, True])
def test_dp_alltoall_and_tp_gather_finish_before_each_table_is_published(runtime, monkeypatch, idle):
    calls, _, _ = runtime
    model = make_model()

    class HashState:
        lookback_depth = 2
        use_slot_cache = True

        def ensure_cache(self):
            return True

        def __call__(self, tokens, *args, **kwargs):
            assert torch.npu.current_stream().name == "aux"
            calls.append(("hash",))
            return tokens.new_zeros((tokens.shape[0], 2, 4))

        def dummy_hashes(self, tokens):
            calls.append(("dummy",))
            return tokens.new_full((tokens.shape[0], 2, 4), -1), tokens.new_zeros(tokens.shape[0], dtype=torch.bool)

    def gather(ids, dim):
        calls.append(("dp_hash",))
        assert torch.npu.current_stream().name == "aux" and dim == 0
        return torch.cat((ids, ids))

    dp = SimpleNamespace(world_size=2, rank_in_group=0, device_group="edp", all_gather=gather)
    monkeypatch.setattr(parallel, "get_engram_dp_group", lambda: dp)
    monkeypatch.setattr(parallel, "engram_gathered_num_tokens", lambda: 4)
    monkeypatch.setattr(model_mod, "get_engram_dp_size", lambda: 2)

    def exchange(recv, staged, group):
        assert group == "edp" and torch.npu.current_stream().name == "aux"
        calls.append(("dp_rows",))
        recv.copy_(staged)

    def tp_gather(rows, dim):
        assert torch.npu.current_stream().name == "aux" and dim == 1
        calls.append(("tp_heads",))
        return torch.cat((rows, rows), dim=dim)

    monkeypatch.setattr(parallel.dist, "all_to_all_single", exchange)
    monkeypatch.setattr(embedding, "tensor_model_parallel_all_gather", tp_gather)
    model.engram_hash = HashState()
    for layer in (1, 2):
        table = object.__new__(embedding.AscendParallelEngramEmbedding)
        torch.nn.Module.__init__(table)
        table.n_hash_cols, table.part_n_hash_cols, table.dim = 4, 1, 2
        table.dp_size, table.tp_size = 2, 2
        table.lookup = lambda ids, out, layer=layer: out.fill_(layer)
        model.layers[layer].engram.embed_tokens = table
    count = 1 if idle else 2
    result = model.prepare_engram_inputs(
        torch.ones(count, dtype=torch.int32),
        torch.arange(count),
        4,
        query_start_loc=torch.tensor([0] if idle else [0, count]),
        block_table=torch.zeros((0 if idle else 1, 1), dtype=torch.int32),
    )
    assert result["engram_mask"][:4].tolist() == [not idle] * count + [False] * (4 - count)
    work = [call[0] for call in calls if call[0] not in ("retain", "reuse")]
    prefix = ["dummy", "record"] if idle else ["record", "hash"]
    assert work == prefix + ["dp_hash", "dp_rows", "tp_heads", "record", "dp_rows", "tp_heads", "record"]
    for layer in (1, 2):
        rows = result["engram_lookups"][layer]
        assert torch.all(rows[:count] == layer)
        assert not rows[count:4].any()


def test_sp_row_copy_waits_for_its_table(runtime, monkeypatch):
    calls, _, _ = runtime
    model = make_model()
    model.use_sequence_parallel, model.engram_rotated = True, False
    model._mtp_hidden_buffer = None
    model.hc_mult, model.aux_hidden_state_layers = 1, ()
    model.shared_attention_state = SimpleNamespace(reset=lambda: None)
    model.norm = lambda values: values
    inputs = model.prepare_engram_graph_inputs(4, prime=False)
    events = {layer: torch.npu.Event() for layer in (1, 2)}

    def shard(tensor):
        for layer, rows in inputs["engram_lookups"].items():
            if tensor.data_ptr() == rows.data_ptr():
                assert ("wait_event", events[layer]) in calls
                calls.append(("sp", layer))
        return tensor[:2].clone()

    class Layer:
        def __init__(self, layer):
            self.layer_idx = layer
            self.engram = lambda hidden, *args: hidden

        def __call__(self, positions, hidden, pre_mix, *args, **kwargs):
            return hidden, pre_mix

        def hc_collapse(self, hidden, pre_mix):
            return hidden[:, 0]

    model.layers = [Layer(1), Layer(2)]
    monkeypatch.setattr(model_mod, "sp_shard", shard)
    monkeypatch.setattr(model_mod, "sp_all_gather", lambda values: torch.cat((values, values)))
    monkeypatch.setattr(model_mod.envs, "VLLM_MOE_SKIP_PADDING", False)
    model.forward(
        torch.arange(4),
        torch.arange(4),
        None,
        inputs_embeds=torch.ones(4, 8),
        engram_pending=events,
        **inputs,
    )
    assert ("sp", 1) in calls and ("sp", 2) in calls
