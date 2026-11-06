# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logical load snapshots and common connector boundaries; CPU contracts only."""

from types import SimpleNamespace as NS

import pytest
import torch

from . import complete_load, load_plan
from .test_kv_guard_runtime import bind, load, metadata
from .test_kv_guard_runtime import runtime_fixture as runtime_fixture
from .test_kv_guard_v2 import adapter_fixture as adapter_fixture
from .test_kv_guard_v2 import prepare


class Connector:
    def __init__(self):
        self.calls = []
        self.result = NS(finished_sending=set(), finished_recving=set(), failed_recving=set())
        self.invalid = set()

    def start_load_kv(self, *args, **kwargs):
        self.calls.append(("start", args, kwargs))
        return "started"

    def wait_for_layer_load(self, layer_name, *args, **kwargs):
        self.calls.append(("wait", layer_name, args, kwargs))
        return "ready"

    def get_transfer_results(self, *args, **kwargs):
        self.calls.append(("results", args, kwargs))
        return self.result

    def get_block_ids_with_load_errors(self):
        self.calls.append(("errors",))
        return self.invalid


@pytest.fixture
def connector(runtime):
    connector = Connector()
    runtime._connector = runtime.subject.ConnectorGuard(runtime, connector)
    return connector


def packet(meta=None, scheduled=None, requests=None, released=(), replay=None):
    scheduled = {} if scheduled is None else scheduled
    context = {
        "pool_id": "pool",
        "requests": requests or {},
        "scheduled_requests": scheduled,
        "released_request_ids": list(released),
        "loads": [meta] if meta is not None else [],
        "recompute_ranges": [replay] if replay is not None else [],
    }
    return NS(_ascend_kv_guard_context=context, total_num_scheduled_tokens=sum(n for _, n in scheduled.values()))


def poll(connector):
    result = connector.get_transfer_results(set())
    invalid = connector.get_block_ids_with_load_errors()
    return result, invalid


def reasons(runtime):
    return {event.get("reason") for event in runtime.events}


@pytest.mark.parametrize(
    "name, accepted", [("MooncakeConnector", True), ("MooncakeConnectorV1", True), ("AscendStoreConnector", False)]
)
def test_registered_ascend_mooncake_name_is_accepted(runtime, name, accepted):
    impl = type("AscendAttentionBackendImpl", (), {})()
    runtime.config = NS(
        parallel_config=NS(),
        model_config=NS(),
        speculative_config=None,
        kv_transfer_config=NS(kv_connector=name),
        compilation_config=NS(static_forward_context={"layer0": NS(impl=impl)}),
    )
    runner = NS(
        kv_cache_config=NS(kv_cache_groups=[NS(layer_names=["layer0"], kv_cache_spec=NS(block_size=128, head_size=64))])
    )
    if accepted:
        runtime.configure(runner)
    else:
        with pytest.raises(ValueError, match="unadapted execution"):
            runtime.configure(runner)


@pytest.mark.parametrize("local", [0, 19])
def test_async_completion_uses_loaded_end_not_transport_local_offset(runtime, connector, local):
    bind(runtime, computed=local)
    meta = load_plan(runtime.state, end=64, start=local)

    def submit(output):
        assert connector.start_load_kv("context", num_computed_tokens=local) == "started"
        assert runtime.state.blocks[0, 1].valid_end == local
        poll(connector)

    runtime.wrap_execute(submit)(packet(meta))
    connector.result.finished_recving = {"r"}
    result, invalid = runtime.wrap_execute(lambda output: poll(connector))(packet())
    assert result is connector.result and invalid is connector.invalid
    assert runtime.state.blocks[0, 1].valid_end == 64
    assert not runtime.state.loads and not runtime.events
    assert [call[0] for call in connector.calls] == ["start", "results", "errors", "results", "errors"]


@pytest.mark.parametrize("mode", ["NONE", "FULL"])
def test_sync_layer_waits_establish_group_readiness_only_after_every_layer(runtime, connector, mode):
    bind(runtime)
    meta = load_plan(runtime.state, end=64, is_async=False)
    item = metadata(torch.tensor([192]), torch.tensor([[1]]), [65])
    context = NS(cudagraph_runtime_mode=mode, attn_metadata=dict.fromkeys(runtime.layer_groups, item))
    runner = NS(input_batch=NS(req_ids=["r"]))

    def model(count):
        for index, layer in enumerate(runtime.layer_groups):
            assert connector.wait_for_layer_load(layer) == "ready"
            assert runtime.state.blocks[0, 1].valid_end == (0 if index == 0 else 64)
            runtime.write(layer, item.slot_mapping)
            runtime.read(layer, item.block_tables, [65])

    forward = runtime.wrap_forward(runner, model, lambda: context)

    def execute(output):
        connector.start_load_kv("context")
        forward(1)
        poll(connector)

    runtime.wrap_execute(execute)(packet(meta, {"r": (64, 1)}))
    assert not runtime.events and not runtime.state.loads
    assert runtime.state.blocks[0, 1].valid_end == 65


@pytest.mark.parametrize("wait_late", [False, True])
def test_missing_or_late_layer_wait_cannot_be_hidden_by_another_layer(runtime, connector, wait_late):
    bind(runtime)
    meta = load_plan(runtime.state, end=64, is_async=False)
    runner = NS(input_batch=NS(req_ids=["r"]))
    context = NS(cudagraph_runtime_mode="NONE")

    def model(count):
        runtime.write("layer0", torch.tensor([192]))
        runtime.read("layer0", torch.tensor([[1]]), [65])
        if wait_late:
            connector.wait_for_layer_load("layer0")
        connector.wait_for_layer_load("layer1")
        runtime.write("layer1", torch.tensor([192]))
        runtime.read("layer1", torch.tensor([[1]]), [65])

    forward = runtime.wrap_forward(runner, model, lambda: context)
    runtime.wrap_execute(lambda output: (connector.start_load_kv(None), forward(1)))(packet(meta, {"r": (64, 1)}))
    assert "load_layer_not_ready" in reasons(runtime)
    assert runtime.state.blocks[0, 1].valid_end == (64 if wait_late else 0)
    assert not any(e["event"] == "guard.violation" for e in runtime.events)


def test_full_replay_without_load_completion_evidence_is_incomplete(runtime, connector):
    bind(runtime)
    meta = load_plan(runtime.state, end=64, is_async=False)
    item = metadata(torch.tensor([192]), torch.tensor([[1]]), [65])
    context = NS(cudagraph_runtime_mode="FULL", attn_metadata=dict.fromkeys(runtime.layer_groups, item))
    forward = runtime.wrap_forward(NS(input_batch=NS(req_ids=["r"])), lambda n: None, lambda: context)
    runtime.wrap_execute(lambda output: (connector.start_load_kv(None), forward(1)))(packet(meta, {"r": (64, 1)}))
    assert "load_layer_not_ready" in reasons(runtime)
    assert runtime.state.blocks[0, 1].valid_end == 0


def test_receive_failure_preserves_local_prefix_and_allows_recompute(runtime, connector):
    state = runtime.state
    state.bind({"pool_id": "pool", "requests": {"r": [[(1, 1), (2, 1)]]}})
    complete_load(state, end=128)
    meta = load_plan(state, end=192, start=128)
    runtime.wrap_execute(lambda output: connector.start_load_kv(None))(packet(meta))
    connector.result.finished_recving = connector.result.failed_recving = {"r"}
    runtime.wrap_execute(lambda output: poll(connector))(packet())
    assert state.blocks[0, 1].valid_end == 128
    assert state.blocks[0, 2].valid_end == 0
    state.observe([{"group_id": 0, "slots": [256], "block_table": [[1, 2]], "seq_lens": [129]}], ["r"], "forward")
    assert not runtime.events


def test_invalid_block_report_only_invalidates_actual_loaded_destinations(runtime, connector):
    state = runtime.state
    state.bind({"pool_id": "pool", "requests": {"r": [[(1, 1), (2, 1), (3, 1)]]}})
    complete_load(state, end=128)
    meta = load_plan(state, end=384, start=128)
    runtime.wrap_execute(lambda output: connector.start_load_kv(None))(packet(meta))
    connector.result.finished_recving = {"r"}
    connector.invalid = {1, 2, 99}
    runtime.wrap_execute(lambda output: poll(connector))(packet())
    assert [state.blocks[0, n].valid_end for n in (1, 2, 3)] == [128, 0, 128]
    assert not runtime.events


@pytest.mark.parametrize("invalid, expected", [({2}, [128, 0, 128]), ({1, 2, 3}, [128, 0, 0])])
def test_early_invalid_blocks_survive_until_the_later_receive_completion(runtime, connector, invalid, expected):
    state = runtime.state
    state.bind({"pool_id": "pool", "requests": {"r": [[(1, 1), (2, 1), (3, 1)]]}})
    complete_load(state, end=128)
    meta = load_plan(state, end=384, start=128)
    runtime.wrap_execute(lambda output: connector.start_load_kv(None))(packet(meta))
    connector.invalid = invalid
    runtime.wrap_execute(lambda output: poll(connector))(packet())
    connector.invalid = set()
    connector.result.finished_recving = {"r"}
    runtime.wrap_execute(lambda output: poll(connector))(packet())
    assert [state.blocks[0, n].valid_end for n in (1, 2, 3)] == expected
    assert not state.loads and not runtime.events


def test_partial_block_failure_invalidates_the_whole_destination_block(runtime, connector):
    bind(runtime, computed=19)
    meta = load_plan(runtime.state, end=64, start=19)
    runtime.wrap_execute(lambda output: connector.start_load_kv(None))(packet(meta))
    connector.result.finished_recving = connector.result.failed_recving = {"r"}
    runtime.wrap_execute(lambda output: poll(connector))(packet())
    assert runtime.state.blocks[0, 1].valid_end == 0


def test_start_exception_preserves_original_error_and_local_kv(runtime):
    runtime.state.bind({"pool_id": "pool", "requests": {"r": [[(1, 1), (2, 1)]]}})
    complete_load(runtime.state, end=128)
    source = Connector()
    error = RuntimeError("backend load failed")

    def fail(*args):
        raise error

    source.start_load_kv = fail
    runtime._connector = runtime.subject.ConnectorGuard(runtime, source)
    meta = load_plan(runtime.state, end=192, start=128)
    with pytest.raises(RuntimeError) as raised:
        runtime.wrap_execute(lambda output: source.start_load_kv(None))(packet(meta))
    assert raised.value is error
    assert runtime.state.blocks[0, 1].valid_end == 128
    assert runtime.state.blocks[0, 2].valid_end == 0
    assert not runtime._connector.inflight and not runtime.state.loads


@pytest.mark.parametrize("epoch", [1, 2])
def test_old_completion_cannot_validate_reused_request_or_allocation(runtime, connector, epoch):
    bind(runtime)
    old = load_plan(runtime.state, end=64)
    runtime.wrap_execute(lambda output: connector.start_load_kv(None))(packet(old))
    runtime.wrap_execute(lambda output: None)(packet(requests={"r": [[(1, epoch)]]}, released=["r"]))
    new = load_plan(runtime.state, end=32)
    runtime.state.begin_load(new)
    connector.result.finished_recving = {"r"}
    runtime.wrap_execute(lambda output: poll(connector))(packet())
    assert runtime.state.blocks[0, 1].valid_end == 0
    assert runtime.state.loads["r"]["load_id"] == new["load_id"]
    assert not runtime.events  # A known old submission may legally drain after cancellation.


def test_overlapping_request_id_is_reported_and_never_promotes_the_new_load(runtime, connector):
    bind(runtime)
    old = load_plan(runtime.state, end=64)
    runtime.wrap_execute(lambda output: connector.start_load_kv(None))(packet(old))
    new = load_plan(runtime.state, end=32)
    runtime.wrap_execute(lambda output: connector.start_load_kv(None))(
        packet(new, released=["r"], requests={"r": [[(1, 2)]]})
    )
    connector.result.finished_recving = {"r"}
    runtime.wrap_execute(lambda output: poll(connector))(packet())
    assert "overlapping_load_request_id" in reasons(runtime)
    assert runtime.state.blocks[0, 1].valid_end == 0


def test_cancelled_request_can_drain_without_poisoning_later_requests(runtime, connector):
    bind(runtime)
    meta = load_plan(runtime.state, end=64)
    runtime.wrap_execute(lambda output: connector.start_load_kv(None))(packet(meta))
    runtime.wrap_execute(lambda output: None)(packet(released=["r"]))
    connector.result.finished_recving = {"r"}
    runtime.wrap_execute(lambda output: poll(connector))(packet())
    assert not runtime.events
    assert not runtime._connector.inflight
    runtime.state.bind({"requests": {"next": [[(1, 2)]]}})
    runtime.state.observe([{"group_id": 0, "slots": [128], "block_table": [[1]], "seq_lens": [1]}], ["next"], "forward")
    assert runtime.state.blocks[0, 1].valid_end == 1 and not runtime.events


def test_completion_metadata_cannot_expand_the_registered_range(runtime):
    bind(runtime)
    meta = load_plan(runtime.state, end=64)
    runtime.state.begin_load(meta)
    runtime.state.load({**meta, "token_end": 128})
    assert runtime.state.blocks[0, 1].valid_end == 0
    assert "stale_load_completion" in reasons(runtime)


def test_completion_cannot_erase_a_detected_corruption_in_the_same_allocation(runtime, connector):
    bind(runtime)
    meta = load_plan(runtime.state, end=64)
    runtime.wrap_execute(lambda output: connector.start_load_kv(None))(packet(meta))
    runtime.state.observe([{"group_id": 0, "slots": [142]}], [], "dummy")
    connector.result.finished_recving = {"r"}
    runtime.wrap_execute(lambda output: poll(connector))(packet())
    assert runtime.state.blocks[0, 1].valid_end == runtime.subject.INVALID_PREFIX
    assert "load_destination_tainted" in reasons(runtime)


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("mode", ["NONE", "FULL"])
def test_only_explicit_replay_allows_overwriting_a_loaded_tail(runtime, replay, mode):
    bind(runtime, computed=128)
    meta = load_plan(runtime.state, end=128, start=127)
    meta.pop("load_id")
    meta.pop("is_async")
    item = metadata(torch.tensor([255]), torch.tensor([[1]]), [128])
    context = NS(cudagraph_runtime_mode=mode, attn_metadata=dict.fromkeys(runtime.layer_groups, item))
    runner = NS(input_batch=NS(req_ids=["r"]))

    def model(count):
        for layer in runtime.layer_groups:
            runtime.write(layer, item.slot_mapping)
            runtime.read(layer, item.block_tables, [128])

    forward = runtime.wrap_forward(runner, model, lambda: context)
    runtime.wrap_execute(lambda output: forward(1))(packet(scheduled={"r": (127, 1)}, replay=meta if replay else None))
    if replay:
        assert runtime.state.blocks[0, 1].valid_end == 128 and not runtime.events
    else:
        assert "overwrite_valid_prefix" in reasons(runtime)


def test_replay_does_not_legalize_a_read_before_the_recomputed_write(runtime):
    bind(runtime, computed=128)
    replay = load_plan(runtime.state, end=128, start=127)
    runner = NS(input_batch=NS(req_ids=["r"]))

    def model(count):
        runtime.read("layer0", torch.tensor([[1]]), [128])
        runtime.write("layer0", torch.tensor([255]))

    forward = runtime.wrap_forward(runner, model, lambda: NS(cudagraph_runtime_mode="NONE"))
    runtime.wrap_execute(lambda output: forward(1))(packet(scheduled={"r": (127, 1)}, replay=replay))
    assert "read_unwritten_range" in reasons(runtime)


def test_post_forward_async_completion_does_not_legalize_an_earlier_read(runtime, connector):
    bind(runtime)
    meta = load_plan(runtime.state, end=64)
    runtime.wrap_execute(lambda output: connector.start_load_kv(None))(packet(meta))
    connector.result.finished_recving = {"r"}
    runner = NS(input_batch=NS(req_ids=["r"]))

    def model(count):
        runtime.read("layer0", torch.tensor([[1]]), [64])
        poll(connector)  # V2 polls inside its forward scope.
        assert runtime.state.blocks[0, 1].valid_end == 0

    forward = runtime.wrap_forward(runner, model, lambda: NS(cudagraph_runtime_mode="NONE"))
    runtime.wrap_execute(lambda output: forward(1))(packet())
    assert "read_unwritten_range" in reasons(runtime)
    assert runtime.state.blocks[0, 1].valid_end == 64


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("positional", [False, True])
def test_scheduler_records_loading_bindings_even_without_scheduled_tokens(runtime, is_async, positional):
    patch = load("guard_loading_scheduler_subject", "vllm_ascend/patch/platform/patch_kv_guard.py")

    class Pool:
        blocks = [NS(block_id=n) for n in range(4)]

        def get_new_blocks(self, count):
            return self.blocks[1 : count + 1]

    class Scheduler:
        def __init__(self):
            self.vllm_config = NS(additional_config={"kv_guard_config": {"enabled": True}})
            self.pool = Pool()
            self.request = NS(request_id="r", num_computed_tokens=0)
            self.kv_cache_manager = NS(
                block_pool=self.pool,
                allocate_slots=self.allocate,
                get_block_ids=lambda request: [[1, 2]],
                free=lambda request: None,
            )

        def allocate(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens=0,
            new_computed_blocks=None,
            num_lookahead_tokens=0,
            num_external_computed_tokens=0,
            delay_cache_blocks=False,
        ):
            self.pool.get_new_blocks(2)
            return object()

        def schedule(self):
            if positional:
                self.kv_cache_manager.allocate_slots(self.request, 0, 19, None, 0, 173, is_async)
            else:
                self.kv_cache_manager.allocate_slots(
                    self.request, 0, 19, num_external_computed_tokens=173, delay_cache_blocks=is_async
                )
            return NS(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=NS(req_ids=[], num_computed_tokens=[]),
                num_scheduled_tokens={},
                finished_req_ids=set(),
            )

    patch.install_scheduler_guard(Scheduler)
    scheduler = Scheduler()
    context = scheduler.schedule()._ascend_kv_guard_context
    assert context["requests"] == {"r": [[(1, 1), (2, 1)]]}
    meta = context["loads"][0]
    assert (meta["token_start"], meta["token_end"], meta["is_async"]) == (19, 192, is_async)
    assert meta["block_table"] == context["requests"]["r"]
    assert context["scheduled_requests"] == {}
    runtime.state.bind(context)
    assert runtime.state.blocks[0, 1].valid_end == 0
    assert runtime.state.loads["r"]["load_id"] == meta["load_id"]


@pytest.mark.parametrize("mode", ["NONE", "FULL"])
def test_v2_receive_only_completion_then_authorized_last_token_replay(adapter, connector, mode):
    runtime = adapter.runtime
    bind(runtime)
    meta = load_plan(runtime.state, end=128)
    adapter.wrap_execute(lambda output: connector.start_load_kv(None))(packet(meta))
    connector.result.finished_recving = {"r"}
    adapter.wrap_execute(lambda output: poll(connector))(packet())
    assert runtime.state.blocks[0, 1].valid_end == 128
    replay = {**meta, "token_start": 127}

    def execute(output):
        item = prepare(adapter, ["r"], [255], [[1]], [128], mode)["layer0"]
        runtime.write("layer0", item.slot_mapping)
        runtime.read("layer0", item.block_tables, [128])

    adapter.wrap_execute(execute)(packet(scheduled={"r": (127, 1)}, replay=replay))
    assert runtime.state.blocks[0, 1].valid_end == 128 and not runtime.events


def test_v2_async_submission_can_follow_execute_in_sample_tokens(adapter, connector):
    runtime = adapter.runtime
    bind(runtime, "running", 1)
    bind(runtime, "loading", 2)
    meta = load_plan(runtime.state, "loading", end=64)

    def execute(output):
        item = prepare(adapter, ["running"], [128], [[1]], [1])["layer0"]
        runtime.write("layer0", item.slot_mapping)
        runtime.read("layer0", item.block_tables, [1])
        return "awaiting sampling"

    assert adapter.wrap_execute(execute)(packet(meta, {"running": (0, 1)})) == "awaiting sampling"
    assert runtime._connector.staged == [meta] and not runtime.events
    # Actual MRV2 post_forward runs from sample_tokens, outside execute_model.
    connector.start_load_kv(None)
    connector.result.finished_recving = {"loading"}
    poll(connector)
    assert runtime.state.blocks[0, 2].valid_end == 64
    assert not runtime.state.loads and not runtime.events


def test_v2_unsubmitted_load_is_rejected_at_the_real_post_forward_boundary(adapter, connector):
    runtime = adapter.runtime
    bind(runtime, "running", 1)
    bind(runtime, "loading", 2)
    meta = load_plan(runtime.state, "loading", end=64)

    def execute(output):
        prepare(adapter, ["running"], [128], [[1]], [1])

    adapter.wrap_execute(execute)(packet(meta, {"running": (0, 1)}))
    assert "load_not_submitted" not in reasons(runtime)
    poll(connector)
    assert "load_not_submitted" in reasons(runtime)


def test_receive_only_binding_is_not_required_in_the_worker_execution_batch(runtime):
    waiting = packet(requests={"loading": [[(2, 1)]]})
    runner = NS(requests={})
    assert runtime.wrap_update(runner, lambda output: "unchanged")(waiting) == "unchanged"
    assert not runtime.events


def test_scheduler_records_explicit_async_tail_replay_without_changing_request(runtime):
    patch = load("guard_replay_scheduler_subject", "vllm_ascend/patch/platform/patch_kv_guard.py")

    class Scheduler:
        def __init__(self):
            self.vllm_config = NS(additional_config={"kv_guard_config": {"enabled": True}})
            self.request = NS(request_id="r", num_computed_tokens=128)
            pool = NS(blocks=[NS(block_id=0), NS(block_id=1)], get_new_blocks=lambda count: [NS(block_id=1)])
            self.kv_cache_manager = NS(block_pool=pool, get_block_ids=lambda request: [[1]], free=lambda request: None)

        def _update_waiting_for_remote_kv(self, request):
            request.num_computed_tokens -= 1
            return "unchanged"

        def schedule(self):
            self.kv_cache_manager.block_pool.get_new_blocks(1)
            assert self._update_waiting_for_remote_kv(self.request) == "unchanged"
            return NS(
                scheduled_new_reqs=[NS(req_id="r", num_computed_tokens=127)],
                scheduled_cached_reqs=NS(req_ids=[], num_computed_tokens=[]),
                num_scheduled_tokens={"r": 1},
                finished_req_ids=set(),
            )

    patch.install_scheduler_guard(Scheduler)
    scheduler = Scheduler()
    context = scheduler.schedule()._ascend_kv_guard_context
    assert scheduler.request.num_computed_tokens == 127
    assert context["scheduled_requests"] == {"r": (127, 1)}
    assert context["recompute_ranges"] == [
        {
            "pool_id": context["pool_id"],
            "request_id": "r",
            "block_table": [[(1, 1)]],
            "token_start": 127,
            "token_end": 128,
        }
    ]
