# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.models.deepseek_v41 import model as model_mod
from vllm_ascend.models.deepseek_v41.engram import model_state as state_mod
from vllm_ascend.ops.triton import engram_lookback


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
    # V2 prepares outside any forward context; forward-consumer tests flip
    # this to True to emulate the model running under set_forward_context.
    context_available = [False]
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
    monkeypatch.setattr(model_mod, "is_forward_context_available", lambda: context_available[0])
    monkeypatch.setattr(model_mod, "get_forward_context", lambda: context)
    return calls, context, aux, context_available


def make_model():
    model = object.__new__(model_mod.DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.has_engram, model.engram_dp_shared_memory = True, True
    model._engram_overlap_enabled = True
    model._engram_input_buffers, model._engram_prepare_stream = None, None
    model._engram_capture_stream, model._engram_capture_events = None, None
    model._engram_graph_events = {}
    model._engram_late_lookup_enabled, model._engram_late_lookup_stream = False, None
    model._engram_max_tokens = 8
    model._mtp_hidden_buffer = None
    model.engram_rotation = torch.eye(32)
    model.config = SimpleNamespace(engram_layer_ids=(1, 2), image_token_id=999)
    model.layers = [
        SimpleNamespace(engram=SimpleNamespace(embed_tokens=SimpleNamespace(n_hash_cols=4, dim=2))) for _ in range(3)
    ]
    return model


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.NONE])
def test_overlapped_modes_publish_events_on_the_aux_stream(runtime, mode):
    calls, _, aux, _ = runtime
    model = make_model()
    model.prepare_engram = Mock()

    def prepare(tokens, positions, lookback, query, slots, blocks, **kwargs):
        assert torch.npu.current_stream().name == "aux"
        assert kwargs["output_tokens"] == 4
        assert slots is None and blocks is None

    model.prepare_engram.side_effect = prepare
    first = model.prepare_engram_inputs(
        torch.ones(4, dtype=torch.int32),
        torch.arange(4),
        4,
        torch.full((2, 2), -1, dtype=torch.int32),
        torch.tensor([0, 4]),
        cg_mode=mode,
    )
    assert model._engram_prepare_stream is aux
    assert calls.count(("reuse", "main")) == 1
    assert first["engram_pending"] is not None and first["engram_mask_ready_event"] is not None
    external = mode == CUDAGraphMode.FULL
    assert first.get("engram_graph_events", False) is external

    second = model.prepare_engram_inputs(
        torch.ones(4, dtype=torch.int32),
        torch.arange(4),
        4,
        torch.full((2, 2), -1, dtype=torch.int32),
        torch.tensor([0, 4]),
        cg_mode=mode,
    )
    assert (first["engram_pending"] is second["engram_pending"]) == external
    assert (first["engram_mask_ready_event"] is second["engram_mask_ready_event"]) == external
    assert first["engram_lookups"] is second["engram_lookups"]


@pytest.mark.parametrize("mode", [CUDAGraphMode.PIECEWISE, None])
def test_non_overlappable_modes_stay_synchronous(runtime, mode):
    calls, _, _, _ = runtime
    model = make_model()
    model.prepare_engram = Mock(return_value=None)

    def prepare(tokens, positions, lookback, query, slots, blocks, **kwargs):
        assert torch.npu.current_stream().name == "main"
        assert kwargs["ready_events"] is None and kwargs["mask_ready_event"] is None

    model.prepare_engram.side_effect = prepare
    result = model.prepare_engram_inputs(
        torch.ones(4, dtype=torch.int32),
        torch.arange(4),
        4,
        torch.full((2, 2), -1, dtype=torch.int32),
        torch.tensor([0, 4]),
        cg_mode=mode,
    )
    assert model._engram_prepare_stream is None
    assert "engram_pending" not in result and "engram_mask_ready_event" not in result
    assert "engram_graph_events" not in result


def test_dummy_steps_zero_buffers_and_skip_hashing(runtime):
    calls, _, _, _ = runtime
    model = make_model()
    model.engram_hash = Mock()
    model.engram_hash.use_slot_cache = False
    model.engram_hash.ensure_cache.return_value = True
    binding = model.prime_engram_v2_graph_inputs(4)
    for tensor in (*binding["engram_lookups"].values(), binding["engram_mask"]):
        tensor.fill_(7)
    calls.clear()
    result = model.prepare_engram_inputs(
        torch.ones(4, dtype=torch.int32),
        torch.arange(4),
        4,
        torch.full((2, 2), -1, dtype=torch.int32),
        torch.tensor([0, 4]),
        cg_mode=CUDAGraphMode.FULL,
        force_dummy=True,
    )
    model.engram_hash.assert_not_called()
    for tensor in (*result["engram_lookups"].values(), result["engram_mask"]):
        assert not tensor[:4].any()
    # publish_mask records the mask first, then each layer's rows land.
    assert calls == [("reuse", "main")] + [("retain", "aux")] * 3 + [
        ("record", result["engram_mask_ready_event"], "aux"),
    ] + [("record", event, "aux") for event in result["engram_pending"].values()]


def test_capacity_is_checked_before_submission(runtime):
    calls, _, _, _ = runtime
    model = make_model()
    model.prepare_engram = Mock()
    with pytest.raises(ValueError, match="capacity"):
        model.prepare_engram_inputs(
            torch.arange(9),
            torch.arange(9),
            9,
            torch.full((2, 2), -1, dtype=torch.int32),
            torch.tensor([0, 9]),
            cg_mode=CUDAGraphMode.FULL,
        )
    model.prepare_engram.assert_not_called()


def test_failed_producer_resets_primed_bucket_events(runtime):
    calls, _, _, _ = runtime
    model = make_model()
    binding = model.prime_engram_v2_graph_inputs(4)
    calls.clear()
    model.prepare_engram = Mock(side_effect=RuntimeError("lookup failed"))
    with pytest.raises(RuntimeError, match="lookup failed"):
        model.prepare_engram_inputs(
            torch.ones(4, dtype=torch.int32),
            torch.arange(4),
            4,
            torch.full((2, 2), -1, dtype=torch.int32),
            torch.tensor([0, 4]),
            cg_mode=CUDAGraphMode.FULL,
        )
    join = calls.index(("join", "aux"))
    assert calls[join + 1 :] == [
        ("reset", event, "main") for event in (binding["engram_mask_ready_event"], *binding["engram_pending"].values())
    ]


def test_prime_registers_and_seeds_one_event_pair_per_bucket(runtime):
    calls, _, _, _ = runtime
    model = make_model()
    first = model.prime_engram_v2_graph_inputs(4)
    seeded = len(calls)
    replay = model.prime_engram_v2_graph_inputs(4)
    assert len(calls) == seeded * 2
    other = model.prime_engram_v2_graph_inputs(8)
    assert first["engram_pending"] is replay["engram_pending"]
    assert first["engram_mask_ready_event"] is replay["engram_mask_ready_event"]
    assert other["engram_mask_ready_event"] is not first["engram_mask_ready_event"]
    assert first["engram_lookups"] is other["engram_lookups"]


def test_retire_resets_all_captured_events(runtime):
    calls, context, aux, context_available = runtime
    model = make_model()
    model._engram_prepare_stream = aux
    context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    context_available[0] = True
    v1 = model.prepare_engram_graph_inputs(4)
    context_available[0] = False
    v2 = model.prime_engram_v2_graph_inputs(4)
    calls.clear()
    model.retire_engram_lookups(reset_events=True)
    expected = [
        event
        for binding in (v1, v2)
        for event in (binding["engram_mask_ready_event"], *binding["engram_pending"].values())
    ]
    assert calls == [("join", "aux")] + [("reset", event, "main") for event in expected]


@pytest.mark.parametrize("mode,waits", [(CUDAGraphMode.FULL, True), (CUDAGraphMode.NONE, False)])
def test_forward_only_waits_external_events_on_full_runs(runtime, monkeypatch, mode, waits):
    calls, context, _, context_available = runtime
    context.cudagraph_runtime_mode = mode
    context_available[0] = True
    model = make_model()
    model.use_sequence_parallel, model.engram_rotated = False, False
    model.hc_mult, model.aux_hidden_state_layers = 1, ()
    model.shared_attention_state = SimpleNamespace(reset=lambda: None)
    model.norm = lambda values: values
    binding = model.prime_engram_v2_graph_inputs(4)
    events = {layer: torch.npu.Event() for layer in (1, 2)}

    class Layer:
        def __init__(self, layer):
            self.layer_idx = layer
            self.engram = lambda hidden, *args: hidden

        def __call__(self, positions, hidden, pre_mix, *args, **kwargs):
            return hidden, pre_mix

        def hc_collapse(self, hidden, pre_mix):
            return hidden[:, 0]

    model.layers = [Layer(1), Layer(2)]
    model.forward(
        torch.arange(4),
        torch.arange(4),
        None,
        inputs_embeds=torch.ones(4, 8),
        engram_lookups=binding["engram_lookups"],
        engram_mask=binding["engram_mask"],
        engram_pending=events,
        engram_graph_events=True,
        engram_mask_ready_event=binding["engram_mask_ready_event"],
    )
    wait_names = [call for call in calls if call[0] in ("wait", "wait_event")]
    assert bool(wait_names) is waits


@pytest.mark.parametrize("tokens", [1, 4, 8])
def test_compressed_forward_preserves_capacity_scale_offset(runtime, tokens):
    from vllm_ascend.models.deepseek_v41.engram.npu import compressed_engram_views

    model = make_model()
    model.use_sequence_parallel, model.engram_rotated = False, False
    model.hc_mult, model.aux_hidden_state_layers = 1, ()
    model.shared_attention_state = SimpleNamespace(reset=lambda: None)
    model.norm = lambda values: values
    packed = torch.zeros((8, 4 * (64 + 2)), dtype=torch.uint8)
    codes, scales = compressed_engram_views(packed, 4, 64)
    codes.fill_(52)
    scales.fill_(127)
    observed = []

    class Gate:
        embed_tokens = SimpleNamespace(compressed_lookup=True, n_hash_cols=4, dim=64)

        def __call__(self, hidden, rows, mask, rotation, *, compressed_rows):
            actual_codes, actual_scales = compressed_rows
            observed.append(actual_codes.shape[0])
            assert actual_codes.shape == (tokens, 256)
            assert actual_scales.shape == (tokens, 4, 2)
            assert torch.equal(actual_codes.view(torch.uint8), codes[:tokens])
            assert torch.equal(actual_scales.view(torch.uint8).flatten(1), scales[:tokens])
            assert actual_scales.data_ptr() == scales.data_ptr()
            return hidden

    class Layer:
        layer_idx = 1
        engram = Gate()

        def __call__(self, positions, hidden, pre_mix, *args, **kwargs):
            return hidden, pre_mix

        def hc_collapse(self, hidden, pre_mix):
            return hidden[:, 0]

    model.layers = [Layer()]
    model.forward(
        torch.arange(tokens),
        torch.arange(tokens),
        None,
        inputs_embeds=torch.ones(tokens, 8),
        engram_lookups={1: packed},
        engram_mask=torch.ones(8, dtype=torch.bool),
    )
    assert observed == [tokens]


def make_state(model, monkeypatch):
    monkeypatch.setattr("vllm_ascend.ops.triton.engram_lookback._gather_lookback_kernel", MagicMock())
    monkeypatch.setattr(
        state_mod, "triton", SimpleNamespace(next_power_of_2=lambda value: 1 << (value - 1).bit_length())
    )
    state = object.__new__(state_mod.EngramModelState)
    state.vllm_config = SimpleNamespace()
    state.max_num_reqs, state.device = 4, torch.device("cpu")
    state.rope_state = None
    state.supports_mm_inputs = False
    state.prompt_embeds_state = None
    state.model = model
    state.lookback_token_ids = torch.full((4, 2), -1, dtype=torch.int32)
    state._cg_mode = CUDAGraphMode.FULL
    return state


def test_state_gathers_lookback_and_overlaps_engram(runtime, monkeypatch):
    calls, _, _, _ = runtime
    model = make_model()
    state = make_state(model, monkeypatch)
    window = state.lookback_token_ids
    window.fill_(-5)

    def gather(lookback, idx_mapping, num_computed, all_token_ids, stride, num_reqs, **kwargs):
        assert torch.npu.current_stream().name == "main"
        lookback.fill_(3)
        assert num_reqs == 2

    engram_lookback._gather_lookback_kernel.__getitem__.side_effect = lambda grid: gather
    input_batch = SimpleNamespace(
        input_ids=torch.ones(4, dtype=torch.int32),
        positions=torch.arange(4),
        num_tokens_after_padding=4,
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 4]),
        is_dummy=False,
    )
    req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.zeros(4, 8, dtype=torch.int32)),
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([2, 4, 0, 0], dtype=torch.int32)),
    )
    model.prepare_engram_inputs = Mock(return_value={"engram_lookups": {"l": "buf"}})
    result = state.prepare_inputs(input_batch, req_states)
    model.prepare_engram_inputs.assert_called_once()
    assert window[0].tolist() == [3, 3]
    assert result["lookback_token_ids"] is window
    assert result["engram_lookups"] == {"l": "buf"}
    _, kwargs = model.prepare_engram_inputs.call_args
    args = model.prepare_engram_inputs.call_args.args
    assert args[0] is input_batch.input_ids and args[2] == 4 and args[3] is window
    assert args[4] is input_batch.query_start_loc
    assert kwargs == {"cg_mode": CUDAGraphMode.FULL, "force_dummy": False}


def test_state_dummy_capture_primes_and_refills_window(runtime, monkeypatch):
    model = make_model()
    state = make_state(model, monkeypatch)
    model.prime_engram_v2_graph_inputs = Mock(return_value={"engram_lookups": {}, "engram_mask": "mask"})
    model.prepare_engram_graph_inputs = Mock(side_effect=AssertionError("duplicate parent preparation"))
    state.lookback_token_ids.fill_(3)
    result = state.prepare_dummy_inputs(2, 4)
    assert (state.lookback_token_ids == -1).all()
    assert result["lookback_token_ids"] is state.lookback_token_ids
    model.prime_engram_v2_graph_inputs.assert_called_once_with(4)
    assert result["engram_mask"] == "mask"


def test_state_without_engram_window_is_passthrough(runtime, monkeypatch):
    model = make_model()
    state = make_state(model, monkeypatch)
    state.lookback_token_ids = None
    assert state.prepare_inputs(SimpleNamespace(), SimpleNamespace()) == {}
    assert state.prepare_dummy_inputs(1, 1) == {}


@pytest.mark.parametrize("dummy", [False, True])
def test_graph_producer_stages_fixed_coordinates_without_external_submission(runtime, monkeypatch, dummy):
    model = make_model()
    state = make_state(model, monkeypatch)
    query = torch.full((5,), -99, dtype=torch.int32)
    valid = torch.tensor([-99], dtype=torch.int32)
    state._engram_graph_inputs = dict(engram_query_start_loc=query, engram_valid_token_count=valid)
    model.prepare_engram_inputs = Mock(side_effect=AssertionError("graph-external producer"))
    model.prime_engram_v2_graph_inputs = Mock(side_effect=AssertionError("external event"))
    batch = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
        idx_mapping=torch.tensor([2, 0]),
        num_reqs=2,
        num_tokens=3,
        num_tokens_after_padding=4,
        is_dummy=dummy,
        query_start_loc=torch.tensor([0, 2, 3, 4]),  # final entry is FIA padding
    )
    req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.zeros(4, 8)), num_computed_tokens=SimpleNamespace(gpu=torch.zeros(4))
    )
    result = state.prepare_engram_inputs(batch, req_states)
    assert result["engram_query_start_loc"] is query
    assert query.tolist() == ([0] * 5 if dummy else [0, 2, 3, 3, 3])
    assert valid.tolist() == ([0] if dummy else [3])
    captured = state.prepare_engram_dummy_inputs(4, 8)
    assert captured["engram_query_start_loc"] is query
    assert captured["engram_valid_token_count"] is valid
    assert query.tolist() == [0] * 5 and valid.tolist() == [0]
    assert (state.lookback_token_ids == -1).all()


def test_graph_capable_state_keeps_eager_preparation(runtime, monkeypatch):
    model = make_model()
    state = make_state(model, monkeypatch)
    state._engram_graph_inputs = {"unused": torch.zeros(1)}
    state._cg_mode = CUDAGraphMode.NONE
    model.prepare_engram_inputs = Mock(return_value={"legacy": True})
    batch = SimpleNamespace(
        input_ids=torch.arange(2),
        positions=torch.arange(2),
        idx_mapping=torch.tensor([0]),
        num_tokens_after_padding=2,
        query_start_loc=torch.tensor([0, 2]),
        is_dummy=False,
    )
    req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.zeros(4, 8)), num_computed_tokens=SimpleNamespace(gpu=torch.zeros(4))
    )
    result = state.prepare_engram_inputs(batch, req_states)
    assert result["legacy"] is True and "unused" not in result
    model.prepare_engram_inputs.assert_called_once()


@pytest.mark.parametrize("failure", [None, "producer", "consumer"])
def test_graph_producer_is_inside_forward_and_always_joins(runtime, failure):
    calls, _, aux, _ = runtime
    model = make_model()
    valid = torch.tensor([0], dtype=torch.int32)

    def produce(*args, **kwargs):
        assert torch.npu.current_stream() is aux
        assert kwargs["valid_token_count"] is valid
        assert not kwargs.get("force_dummy", False)
        calls.append(("producer",))
        if failure == "producer":
            raise RuntimeError("producer")

    model.prepare_engram = Mock(side_effect=produce)

    def run():
        with model.captured_engram_inputs(
            torch.arange(4), torch.arange(4), torch.full((4, 2), -1), torch.zeros(5), valid
        ) as inputs:
            assert torch.npu.current_stream().name == "main"
            assert "engram_graph_events" not in inputs  # ordinary in-graph events
            calls.append(("consumer",))
            if failure == "consumer":
                raise RuntimeError("consumer")

    if failure:
        with pytest.raises(RuntimeError, match=failure):
            run()
    else:
        run()
    assert calls[0] == ("reuse", "main") and calls[-1] == ("join", "aux")
    assert model._engram_graph_events == {}


@pytest.mark.parametrize(
    "valid_count,expected", [(0, [False] * 4), (2, [True, False, False, False]), (4, [True, False, True, True])]
)
def test_graph_producer_device_count_masks_padding_and_idle_ranks(runtime, monkeypatch, valid_count, expected):
    model = make_model()
    state = Mock(use_slot_cache=False, lookback_depth=2)
    state.ensure_cache.return_value = True
    state.return_value = torch.zeros(4, 2, 4, dtype=torch.int64)
    model.engram_hash = state
    for layer in model.layers:
        table = layer.engram.embed_tokens
        table.dp_size = table.tp_size = 1
        table.lookup = lambda indices, out: out.zero_()
    monkeypatch.setattr(model_mod, "gather_engram_hashes", lambda hashes, **kwargs: hashes)
    buffers, mask = model._get_engram_input_buffers()
    model.prepare_engram(
        torch.tensor([12, 999, 13, 14]),
        torch.arange(4),
        torch.full((4, 2), -1),
        torch.tensor([0, 2, 4, 4, 4]),
        output_buffers=buffers,
        mask_output_buffer=mask,
        output_tokens=4,
        valid_token_count=torch.tensor([valid_count]),
    )
    assert mask[:4].tolist() == expected
    assert state.call_args.args[3].tolist() == [not value for value in expected]


def test_graph_producer_capability_excludes_slot_cache_and_collectives(runtime, monkeypatch):
    model = make_model()
    model.use_sequence_parallel = False
    model.engram_hash = SimpleNamespace(use_slot_cache=False)
    group = SimpleNamespace(world_size=1)
    monkeypatch.setattr(model_mod, "get_pp_group", lambda: group)
    for layer in model.layers:
        layer.engram.embed_tokens.dp_size = layer.engram.embed_tokens.tp_size = 1
    assert model.can_capture_engram_producer()
    for owner, field, disabled_value in (
        (model, "has_engram", False),
        (model, "_engram_overlap_enabled", False),
        (model, "use_sequence_parallel", True),
        (model.engram_hash, "use_slot_cache", True),
        (group, "world_size", 2),
        (model.layers[1].engram.embed_tokens, "dp_size", 2),
        (model.layers[1].engram.embed_tokens, "tp_size", 2),
    ):
        old = getattr(owner, field)
        setattr(owner, field, disabled_value)
        assert not model.can_capture_engram_producer(), field
        setattr(owner, field, old)


def test_mask_wait_is_delayed_past_early_layer_compute(runtime):
    calls, _, _, _ = runtime
    model = make_model()
    model.use_sequence_parallel, model.engram_rotated = False, False
    model.hc_mult, model.aux_hidden_state_layers = 1, ()
    model.shared_attention_state = SimpleNamespace(reset=lambda: None)
    model.norm = lambda values: values
    buffers, mask = model._get_engram_input_buffers()
    mask_event = torch.npu.Event()

    class Layer:
        def __init__(self, index):
            self.layer_idx = index
            self.engram = (lambda hidden, *args: hidden) if index else None

        def __call__(self, positions, hidden, pre_mix, *args, **kwargs):
            calls.append(("layer", self.layer_idx))
            return hidden, pre_mix

        def hc_collapse(self, hidden, pre_mix):
            return hidden[:, 0]

    model.layers = [Layer(0), Layer(1), Layer(2)]
    model.forward(
        torch.arange(4),
        torch.arange(4),
        None,
        inputs_embeds=torch.ones(4, 8),
        engram_lookups=buffers,
        engram_mask=mask,
        engram_mask_ready_event=mask_event,
    )
    assert calls.index(("layer", 0)) < calls.index(("wait_event", mask_event)) < calls.index(("layer", 1))
    assert calls.count(("wait_event", mask_event)) == 1
