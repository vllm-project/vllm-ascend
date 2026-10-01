"""Business and lifecycle contracts that remain above the bound execution plan."""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1 import vllm_adapter
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.connector import (
    AscendStoreV1Connector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.availability import (
    ExternalPrefixPlan,
    LookupQuery,
    RemoteAvailability,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.planner import (
    TransferPlanner,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.progress import (
    AllocationLoadPublication,
    RequestSnapshot,
    ScheduledLoadPublication,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.spec import (
    TransferPlanningSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.planning.step import (
    ScheduledRequestKind,
    TransferPlanningStep,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.compiler import (
    ProgramCompilationError,
    compile_kv_pool_program,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.spec.compilation import (
    KVPoolCompilationSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.spec.schedule import (
    KVPoolSchedule,
    LoadScheduleKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.spec.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
    resolve_group_layers,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.stages.reachability import (
    HybridReachability,
    UnitaryReachability,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.evidence import (
    ChunkAvailability,
    GroupAvailability,
    ReachablePrefix,
    StoreCompletion,
    StoreEvidence,
    TransferEvidence,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.lookup import (
    LookupCodec,
    LookupRequest,
    LookupResult,
    TailKeyBoundary,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    CheckpointStoreCommand,
    KVTransferStep,
    LoadCommand,
    LoadCommandBatch,
    RangeStoreCommand,
    StateCheckpointSource,
    StoreCommandBatch,
)

from .v1.helpers import (
    FakeBackend,
    FakeEvent,
    make_backend_spec,
    make_runtime,
    make_topology,
    make_work,
)


class FakeAvailabilityProbe:
    def __init__(self, availability: RemoteAvailability | None) -> None:
        self.availability = availability
        self.queries: list[LookupQuery] = []

    def query(self, query: LookupQuery) -> RemoteAvailability | None:
        self.queries.append(query)
        return self.availability

    def close(self) -> None:
        pass


class FakeLoadStartGate:
    def __init__(self, opened: bool = False) -> None:
        self._opened = threading.Event()
        if opened:
            self._opened.set()

    def open(self) -> None:
        self._opened.set()

    def cancel(self) -> None:
        self._opened.set()

    def wait(self, timeout) -> bool:
        return self._opened.wait(timeout)


def begin_step(runtime, *, load=None, store=None) -> None:
    runtime.begin_step(KVTransferStep(load or LoadCommandBatch(), store or StoreCommandBatch()))


def make_planner(availability, *, async_load=False, save_decode=False, store_enabled=True):
    publication = AllocationLoadPublication() if async_load else ScheduledLoadPublication()
    return TransferPlanner(
        TransferPlanningSpec(4, 4, (0,), True),
        FakeAvailabilityProbe(availability),
        publication,
        store_enabled=store_enabled,
        save_decode_cache=save_decode,
    )


def build_planner_step(planner, scheduler_output, requests=None, *, resumed_request_ids=()):
    requests = requests or {}
    for request in requests.values():
        if not hasattr(request, "prompt_token_ids"):
            request.prompt_token_ids = [0] * getattr(request, "num_prompt_tokens", 0)
        if not hasattr(request, "num_prompt_tokens"):
            request.num_prompt_tokens = len(request.prompt_token_ids)
    cached = scheduler_output.scheduled_cached_reqs
    cached.resumed_req_ids = set(resumed_request_ids)
    scheduler_output.kv_connector_block_state = getattr(scheduler_output, "kv_connector_block_state", None)
    scheduler_output.num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {})
    scheduler_output.preempted_req_ids = getattr(scheduler_output, "preempted_req_ids", set())
    planning_step = vllm_adapter.adapt_scheduler_output(
        scheduler_output,
        requests,
        store_enabled=planner._store_enabled,
    )
    return planner.build_step(planning_step)


def test_small_value_and_codec_contracts_share_one_domain_smoke_test() -> None:
    invalid_ranges = ((-1, 0), (4, 3))
    for start, end in invalid_ranges:
        with pytest.raises(ValueError):
            TokenRange(start, end)

    layers = resolve_group_layers(
        ["model.layers.1.v", "mtp.layers.0.attn", "model.layers.1.k"],
        base_layer_count=4,
    )
    assert layers == (
        KVPoolLayerTopology(1, ("model.layers.1.k", "model.layers.1.v")),
        KVPoolLayerTopology(4, ("mtp.layers.0.attn",)),
    )

    codec = LookupCodec()
    request = LookupRequest(TokenRange(4, 12), (1, 3), (b"a", b"b"))
    result = LookupResult(8, (TailKeyBoundary(1, 12), TailKeyBoundary(3, 8)))
    assert codec.decode_request(codec.encode_request(request)) == request
    assert codec.decode_result(codec.encode_result(result)) == result


def test_compiler_rejects_each_unsupported_static_composition(monkeypatch) -> None:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program import compiler

    monkeypatch.setattr(compiler, "resolve_backend_spec", lambda _name: make_backend_spec())
    cases = (
        (
            make_topology(tp_mismatch=True, consumer_pipeline_partitions=(1, 1)),
            LoadScheduleKind.SYNC,
            "Consumer pipeline",
        ),
        (make_topology(tp_mismatch=True), LoadScheduleKind.LAYERWISE, "TP mismatch"),
        (make_topology(consumer_pipeline_partitions=(1, 1)), LoadScheduleKind.LAYERWISE, "consumer pipeline"),
    )
    for topology, load_kind, message in cases:
        schedule = KVPoolSchedule(load_kind, None, 2)
        with pytest.raises(ProgramCompilationError, match=message):
            compile_kv_pool_program(KVPoolCompilationSpec(topology, "fake", schedule, 64))


def test_reachability_matrix_preserves_contiguous_and_partial_tail_semantics() -> None:
    unitary = UnitaryReachability(0, max_model_len=64, cache_transfer_granularity=4)
    selection = unitary.select_for_lookup((b"a", b"b", b"c"), TokenRange(0, 12))
    availability = GroupAvailability(
        0,
        (
            ChunkAvailability(TokenRange(0, 4), b"a", True),
            ChunkAvailability(TokenRange(4, 8), b"b", False),
            ChunkAvailability(TokenRange(8, 12), b"c", True),
        ),
    )
    assert unitary.resolve_available_end(selection, (availability,)) == ReachablePrefix(4)

    groups = (
        KVPoolGroupTopology(
            0,
            FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=1, dtype=torch.float32),
            (KVPoolLayerTopology(0, ("layers.0",)),),
            make_topology().groups[0].key_metadata,
        ),
        KVPoolGroupTopology(
            1,
            MambaSpec(
                block_size=16,
                shapes=((1, 1),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
            ),
            (KVPoolLayerTopology(1, ("layers.1",)),),
            replace(make_topology().groups[0].key_metadata, kv_cache_group_id=1),
        ),
    )
    hybrid = HybridReachability(groups, 16, 4, 64)
    hybrid_selection = hybrid.select_for_lookup((b"a", b"b", b"c"), TokenRange(0, 12))
    hybrid_availability = tuple(
        GroupAvailability(
            group_id,
            (
                ChunkAvailability(TokenRange(0, 4), b"a", False),
                ChunkAvailability(TokenRange(0, 8), b"b", False),
                ChunkAvailability(TokenRange(0, 12), b"c", True),
            ),
        )
        for group_id in (0, 1)
    )
    assert hybrid.resolve_available_end(hybrid_selection, hybrid_availability) == ReachablePrefix(
        12,
        (TailKeyBoundary(0, 12), TailKeyBoundary(1, 12)),
    )


def test_store_completion_uses_transfer_source_evidence_without_reinterpreting_it() -> None:
    source = make_work(layerwise=False).sources[0]
    cases = (
        (StoreEvidence((TransferEvidence(source, 0),), True, True), None),
        (StoreEvidence((TransferEvidence(source, 0),), False, True), "success was not confirmed"),
        (StoreEvidence((TransferEvidence(source, -1),), False, False), "result codes"),
        (StoreEvidence((TransferEvidence(source, 0),), True, False), "source release is unknown"),
    )
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.program import BoundKVPoolProgram

    for evidence, message in cases:
        completion = StoreCompletion("request", evidence)
        if message is None:
            BoundKVPoolProgram.validate_store_completion(completion)
        else:
            with pytest.raises(RuntimeError, match=message):
                BoundKVPoolProgram.validate_store_completion(completion)

    cause = RuntimeError("publication failed")
    with pytest.raises(RuntimeError, match="Store failed") as failure:
        BoundKVPoolProgram.validate_store_completion(StoreCompletion("request", StoreEvidence((), False, True, cause)))
    assert failure.value.__cause__ is cause


def test_bulk_runtime_store_success_and_failure_preserve_source_safety() -> None:
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 7)

    success, resources, backend = make_runtime(source_ready_event_factory=FakeEvent)
    begin_step(success, store=StoreCommandBatch((command,)))
    success.finish_step()
    completions = success.fence_previous_store()
    assert completions[0].evidence.succeeded
    assert completions[0].evidence.source_release_confirmed
    assert success.take_released_store_job_ids() == {7}
    success.close()
    assert resources.closed and backend.closed is False

    for native_result in ([-1], RuntimeError("put failed")):
        failed_backend = FakeBackend()
        failed_backend.put_result = native_result
        failed, failed_resources, _ = make_runtime(failed_backend, source_ready_event_factory=FakeEvent)
        begin_step(failed, store=StoreCommandBatch((command,)))
        failed.finish_step()
        with pytest.raises(RuntimeError, match="Store failed|result codes"):
            failed.fence_previous_store()
        assert failed._pending_store_batch is not None
        assert not failed._pending_store_batch.completions[0].evidence.source_release_confirmed
        assert failed.take_released_store_job_ids() == set()
        with pytest.raises(RuntimeError, match="previous Store failure"):
            failed.close()
        assert not failed_resources.closed


def test_layerwise_load_happy_path_reuses_rows_and_closes_one_session() -> None:
    runtime, resources, backend = make_runtime(
        layerwise=True,
        store=False,
        start_gate_factory=lambda: FakeLoadStartGate(opened=True),
    )
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(runtime, load=LoadCommandBatch((command,)))

    runtime.start_load()
    runtime.wait_for_layer_load("layers.0.group.0")
    runtime.wait_for_layer_load("layers.1.group.0")
    runtime.close()

    session_calls = [call for call in backend.calls if call[0] in ("batch_get_start", "batch_get_end")]
    copy_calls = [call for call in backend.calls if call[0] == "batch_copy_get"]
    assert [call[0] for call in session_calls] == ["batch_get_start", "batch_get_end"]
    assert [call[2] for call in copy_calls] == [((1064,),), ((2064,),)]
    assert [call[4] for call in copy_calls] == [((0,),), ((32,),)]
    assert resources.closed


def test_layerwise_load_failure_cleans_sessions_and_preserves_block_identity() -> None:
    backend = FakeBackend()
    backend.session_copy_result = [-1]
    runtime, resources, _ = make_runtime(backend, layerwise=True, store=False, physical_layers=(0,))
    command = LoadCommand("request", TokenRange(0, 4), ((7,),), (b"a",))
    begin_step(runtime, load=LoadCommandBatch((command,)))

    runtime.start_load()
    runtime.wait_for_layer_load("layers.0.group.0")
    result = runtime.collect_load_result()
    assert result.failed_block_ids == frozenset({7})
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    runtime.close()
    assert resources.closed


def test_layerwise_load_keeps_a_bounded_prefetch_window() -> None:
    gates: list[FakeLoadStartGate] = []

    def make_gate():
        gate = FakeLoadStartGate()
        gates.append(gate)
        return gate

    runtime, _, backend = make_runtime(
        layerwise=True,
        store=False,
        physical_layers=(0, 1, 2),
        start_gate_factory=make_gate,
    )
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(runtime, load=LoadCommandBatch((command,)))

    runtime.start_load()
    assert not any(call[0] == "batch_copy_get" for call in backend.calls)
    runtime.wait_for_layer_load("layers.0.group.0")
    assert [call[0] for call in backend.calls].count("batch_copy_get") == 1
    gates[0].open()
    runtime.wait_for_layer_load("layers.1.group.0")
    assert [call[0] for call in backend.calls].count("batch_copy_get") == 2
    gates[1].open()
    runtime.wait_for_layer_load("layers.2.group.0")
    runtime.close()
    assert [call[0] for call in backend.calls].count("batch_copy_get") == 3


def test_layerwise_load_start_exception_closes_attempted_session_once(monkeypatch) -> None:
    backend = FakeBackend()

    def fail_start(keys):
        backend.calls.append(("batch_get_start", tuple(keys)))
        raise RuntimeError("session start failed")

    monkeypatch.setattr(backend, "batch_get_start", fail_start)
    runtime, resources, _ = make_runtime(backend, layerwise=True, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(runtime, load=LoadCommandBatch((command,)))

    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        runtime.start_load()
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    with pytest.raises(RuntimeError, match="Layerwise Load timeline terminated"):
        runtime.close()
    assert [call[0] for call in backend.calls].count("batch_get_end") == 1
    assert resources.closed


def test_layerwise_store_commits_only_after_all_layers_and_unknown_failure_retains_source() -> None:
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    runtime, resources, backend = make_runtime(layerwise=True)
    begin_step(runtime, store=StoreCommandBatch((command,)))

    runtime.save_layer("layers.0.group.0")
    runtime._timeline.store._executor._queue.join()
    assert [call[0] for call in backend.calls].count("batch_copy_put") == 1
    assert "batch_commit" not in [call[0] for call in backend.calls]

    runtime.save_layer("layers.1.group.0")
    runtime.finish_step()
    assert [call[0] for call in backend.calls].count("batch_commit") == 1
    assert runtime.take_released_store_job_ids() == {17}
    runtime.close()
    assert resources.closed

    failing_backend = FakeBackend()
    failing_backend.store_session_copy_result = RuntimeError("copy failed")
    failed, failed_resources, _ = make_runtime(failing_backend, layerwise=True)
    duplicate = RangeStoreCommand("duplicate", TokenRange(0, 4), ((3,),), (b"a",), 4, 18)
    begin_step(failed, store=StoreCommandBatch((command, duplicate)))
    failed.save_layer("layers.0.group.0")
    failed.save_layer("layers.1.group.0")
    with pytest.raises(RuntimeError, match="Store failed"):
        failed.finish_step()
    assert failed._pending_store_batch is not None
    assert not failed._pending_store_batch.completions[0].evidence.source_release_confirmed
    assert failed._pending_store_batch.completions[1].evidence.source_release_confirmed
    assert failed.take_released_store_job_ids() == {18}
    assert "batch_commit" not in [call[0] for call in failing_backend.calls]
    assert "batch_revoke" in [call[0] for call in failing_backend.calls]
    with pytest.raises(RuntimeError, match="previous Store failure"):
        failed.close()
    assert not failed_resources.closed


def test_runtime_close_reports_incomplete_layerwise_store_but_releases_safe_source() -> None:
    runtime, resources, backend = make_runtime(layerwise=True)
    command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
    begin_step(runtime, store=StoreCommandBatch((command,)))
    runtime.save_layer("layers.0.group.0")

    with pytest.raises(RuntimeError, match="Store failed"):
        runtime.close()

    assert runtime.take_released_store_job_ids() == {17}
    assert runtime._pending_store_batch is None
    assert resources.closed
    assert "batch_revoke" in [call[0] for call in backend.calls]
    assert "batch_commit" not in [call[0] for call in backend.calls]


def test_runtime_owns_active_step_and_async_completion_lifecycle() -> None:
    runtime, _, _ = make_runtime(async_load=True, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))

    with pytest.raises(RuntimeError, match="has not begun"):
        runtime.start_load()
    begin_step(runtime, load=LoadCommandBatch((command,)))
    with pytest.raises(RuntimeError, match="has not ended"):
        runtime.begin_step(KVTransferStep())
    runtime.start_load()
    runtime.end_step()
    begin_step(runtime)
    runtime._timeline.load._executor._queue.join()
    result = runtime.collect_load_result()
    assert result.completed_request_ids == {"request"}
    assert runtime.collect_load_result().completed_request_ids == set()
    runtime.end_step()
    runtime.close()


def test_async_load_failure_drains_pending_work_and_rejects_overlap(monkeypatch) -> None:
    backend = FakeBackend()

    def fail_get(*_args):
        raise RuntimeError("backend get failed")

    monkeypatch.setattr(backend, "get", fail_get)
    runtime, _, _ = make_runtime(backend, async_load=True, store=False)
    command = LoadCommand("request", TokenRange(0, 4), ((1,),), (b"a",))
    begin_step(runtime, load=LoadCommandBatch((command,)))
    runtime.start_load()
    runtime.end_step()

    begin_step(
        runtime,
        load=LoadCommandBatch((replace(command, request_id="other"), command)),
    )
    with pytest.raises(RuntimeError, match="already has a pending asynchronous Load"):
        runtime.start_load()
    assert runtime._pending_load_request_ids == {"request"}
    runtime._timeline.load._executor._queue.join()
    with pytest.raises(RuntimeError, match="asynchronous Load|backend get failed"):
        runtime.collect_load_result()
    with pytest.raises(RuntimeError, match="asynchronous Load|backend get failed"):
        runtime.close()


def test_planner_full_hit_and_deferred_load_publish_only_after_allocation() -> None:
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[0] * 12,
        num_tokens=12,
        block_hashes=[b"a", b"b", b"c"],
    )
    planner = make_planner(RemoteAvailability(TokenRange(0, 12), 11))
    assert planner.lookup(LookupQuery("request", 12, 12, request.block_hashes, 0)) == ExternalPrefixPlan(11, False)
    planner.confirm_allocation("request", ((1, 2, 3),), tuple(request.block_hashes), 12, 11)
    step = build_planner_step(
        planner,
        SimpleNamespace(
            finished_req_ids=set(),
            scheduled_new_reqs=[SimpleNamespace(req_id="request", num_computed_tokens=11, block_ids=([1, 2, 3],))],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
            num_scheduled_tokens={"request": 1},
        ),
        {"request": request},
    )
    assert step.load.commands[0].load_range == TokenRange(0, 12)
    assert step.store.commands == ()

    deferred = make_planner(RemoteAvailability(TokenRange(4, 8), 8), async_load=True)
    short_request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[0] * 8,
        num_tokens=8,
        block_hashes=[b"a", b"b"],
    )
    assert deferred.lookup(LookupQuery("request", 8, 9, short_request.block_hashes, 4)).load_is_deferred
    deferred.confirm_allocation("request", ((1, 2),), tuple(short_request.block_hashes), 8, 4)
    deferred_step = build_planner_step(
        deferred,
        SimpleNamespace(
            finished_req_ids=set(),
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        ),
        {"request": short_request},
    )
    assert deferred_step.load.commands[0].load_range == TokenRange(4, 8)


def test_planner_uses_scheduler_resumption_as_authoritative_request_kind() -> None:
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=[0] * 4,
        num_prompt_tokens=4,
        num_tokens=7,
        block_hashes=[b"a", b"b"],
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["request"],
            resumed_req_ids={"request"},
            new_block_ids=[([3, 4],)],
            num_computed_tokens=[4],
        ),
        num_scheduled_tokens={"request": 3},
        finished_req_ids=set(),
        preempted_req_ids=None,
        kv_connector_block_state=None,
    )
    step = vllm_adapter.adapt_scheduler_output(scheduler_output, {"request": request}, store_enabled=True)
    assert isinstance(step, TransferPlanningStep)
    assert step.scheduled_requests[0].kind is ScheduledRequestKind.RESUMED
    assert step.scheduled_requests[0].block_ids_by_group == ((3, 4),)


def test_checkpoint_planning_and_projection_keep_exact_source_identity() -> None:
    planner = TransferPlanner(
        TransferPlanningSpec(4, 4, (0, 1), True),
        FakeAvailabilityProbe(None),
        ScheduledLoadPublication(),
        store_enabled=True,
        save_decode_cache=False,
    )
    request = SimpleNamespace(block_hashes=[b"a", b"b", b"c"])
    planner.request_progress["request"] = RequestSnapshot(
        "request",
        12,
        ((1, 2), (10, 11)),
        (b"a", b"b", b"c"),
        8,
        published_store_end_token=4,
    )
    output = SimpleNamespace(
        finished_req_ids=set(),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[], new_block_ids=[]),
        kv_connector_block_state=SimpleNamespace(boundary_state_offloads={"request": [(1, 10, 8), (1, 11, 12)]}),
    )

    commands = build_planner_step(planner, output, {"request": request}).store.commands

    assert [command.sources for command in commands] == [
        (StateCheckpointSource(1, 10, 8),),
        (StateCheckpointSource(1, 11, 12),),
    ]
    assert all(isinstance(command, CheckpointStoreCommand) for command in commands)
    assert commands[0].block_ids_by_group == ((1, 2), (10, 11))


def test_connector_translates_lookup_allocation_and_step_lifecycle() -> None:
    queries = []
    confirmations = []
    planning_steps = []
    connector = AscendStoreV1Connector.__new__(AscendStoreV1Connector)
    connector._requests = {}
    connector._store_enabled = False
    connector._finished_checkpoint_stores = []
    connector._store_source_leases = SimpleNamespace(acquire=lambda command: command)
    connector.planner = SimpleNamespace(
        lookup=lambda query: queries.append(query) or ExternalPrefixPlan(4, True),
        confirm_allocation=lambda *args: confirmations.append(args),
        build_step=lambda step: planning_steps.append(step) or KVTransferStep(),
    )
    request = SimpleNamespace(
        request_id="request",
        prompt_token_ids=None,
        num_prompt_tokens=8,
        num_tokens=9,
        block_hashes=[b"a", b"b"],
    )

    assert connector.get_num_new_matched_tokens(request, 4) == (4, True)
    connector.update_state_after_alloc(request, SimpleNamespace(get_block_ids=lambda: ([1, 2],)), 4)
    connector.build_connector_meta(
        SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
            num_scheduled_tokens={},
            finished_req_ids={"request"},
            preempted_req_ids=None,
            kv_connector_block_state=None,
        )
    )

    assert queries == [LookupQuery("request", 8, 9, request.block_hashes, 4)]
    assert confirmations == [("request", ((1, 2),), (b"a", b"b"), 8, 4)]
    assert planning_steps[0].finished_request_ids == {"request"}
    assert connector._requests == {}
