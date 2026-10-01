"""Verify that v1 lowers static structure once and binds one request row axis."""

from __future__ import annotations

import numpy as np
import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.compiler import (
    compile_kv_pool_program,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.lowering import (
    bind_transfer_rows,
    enumerate_transfer_work,
    lower_transfer_plans,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.program import (
    BoundKVPoolProgram,
    KVPoolProgram,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.spec.compilation import (
    KVPoolCompilationSpec,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.representation import (
    KVBlockAssignment,
    KVBlockAssignmentBatch,
    KVChunk,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.selection import (
    TransferSpan,
    select_work_keys,
    selected_work_keys,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program.values.transfer import (
    ContiguousLayoutPlan,
    StridedLayoutPlan,
    TransferRows,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.backend.arguments import (
    materialize_gva,
    materialize_key_ranges,
    materialize_ranges,
)

from .helpers import (
    FakeBackend,
    make_backend_spec,
    make_memory_geometry,
    make_plan,
    make_rows,
    make_schedule,
    make_topology,
)


def test_compilation_and_memory_binding_form_distinct_immutable_states(monkeypatch) -> None:
    topology = make_topology()
    schedule = make_schedule(layerwise=True)
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.program import compiler

    monkeypatch.setattr(compiler, "resolve_backend_spec", lambda _name: make_backend_spec())
    unbound = compile_kv_pool_program(KVPoolCompilationSpec(topology, "fake", schedule, 64))
    geometry = make_memory_geometry(topology)

    bound = unbound.bind_memory(geometry, block_capacity=8)

    assert type(unbound) is KVPoolProgram
    assert type(bound) is BoundKVPoolProgram
    assert bound.topology is topology
    assert bound.schedule is schedule
    assert tuple(plan.group_id for plan in bound.plans.load) == topology.transfer_group_ids
    assert bound.plans.load[0] is bound.plans.store[0]
    assert bound.plans.load[0].block_capacity == 8
    assert not bound.plans.load[0].layouts[0].bases.flags.writeable


def test_lowering_matrix_fixes_backend_enumeration_before_requests() -> None:
    cases = []

    contiguous_topology = make_topology()
    contiguous = lower_transfer_plans(
        contiguous_topology,
        make_schedule(),
        make_memory_geometry(contiguous_topology),
        8,
    )
    cases.append(
        (
            "contiguous",
            contiguous.load[0],
            (None,),
            (ContiguousLayoutPlan,),
            ((0, 1),),
        )
    )

    layerwise_topology = make_topology()
    layerwise = lower_transfer_plans(
        layerwise_topology,
        make_schedule(layerwise=True),
        make_memory_geometry(layerwise_topology),
        8,
    )
    cases.append(
        (
            "layerwise",
            layerwise.load[0],
            (0, 1),
            (ContiguousLayoutPlan, ContiguousLayoutPlan),
            ((0,), (1,)),
        )
    )

    strided_topology = make_topology(tp_mismatch=True)
    strided = lower_transfer_plans(
        strided_topology,
        make_schedule(),
        make_memory_geometry(strided_topology),
        8,
    )
    cases.append(
        (
            "strided",
            strided.load[0],
            (None,),
            (StridedLayoutPlan, StridedLayoutPlan),
            ((0, 1), (0, 1)),
        )
    )

    pipeline_topology = make_topology(consumer_pipeline_partitions=(1, 1))
    pipeline = lower_transfer_plans(
        pipeline_topology,
        make_schedule(),
        make_memory_geometry(pipeline_topology),
        8,
    )
    cases.append(
        (
            "consumer-pipeline-store",
            pipeline.store[0],
            (None,),
            (ContiguousLayoutPlan, ContiguousLayoutPlan),
            ((0,), (1,)),
        )
    )

    for name, plan, submission_layers, layout_types, physical_layers in cases:
        assert tuple(item.physical_layer_id for item in plan.submissions) == submission_layers, name
        assert tuple(type(item) for item in plan.layouts) == layout_types, name
        assert tuple(item.physical_layer_ids for item in plan.layouts) == physical_layers, name
        assert tuple(item.order_index for item in plan.layouts) == tuple(range(len(plan.layouts))), name
        assert tuple(index for item in plan.submissions for index in item.layout_indices) == tuple(
            range(len(plan.layouts))
        ), name

    assert contiguous.load[0] is contiguous.store[0]
    assert layerwise.load[0] is layerwise.store[0]
    assert pipeline.load[0] is not pipeline.store[0]
    assert tuple(item.consumer_pp_slice for item in pipeline.store[0].coordinates) == (0, 1)
    assert tuple(item.effective_tp_rank for item in strided.load[0].coordinates) == (0, 1)


def test_request_rows_are_bound_once_and_reused_by_every_layer() -> None:
    rows = make_rows(make_plan(layerwise=True))
    work = enumerate_transfer_work((rows,))

    assert rows.row_count == 2
    assert rows.block_ids.tolist() == [1, 3]
    assert not rows.block_ids.flags.writeable
    assert not rows.memory_token_counts.flags.writeable
    assert all(
        item.chunk is chunk for item, chunk in zip(rows.remote_objects_by_coordinate[0], rows.chunks, strict=True)
    )
    assert [item.physical_layer_id for item in work] == [0, 1]
    assert all(span.rows is rows for item in work for span in item.spans)
    assert [span.row_indices for span in work[0].spans] == [range(0, 2)]
    assert [source.row_index for source in work[1].sources] == [0, 1]
    assert [source.block_id for source in work[1].sources] == [1, 3]

    selected = select_work_keys(work, {rows.keys_by_coordinate[0][0]})
    assert selected[0].spans[0].selection is selected[1].spans[0].selection
    assert [[source.row_index for source in item.sources] for item in selected] == [[0], [0]]
    assert selected_work_keys(selected) == (rows.keys_by_coordinate[0][0],)


def test_backend_materialization_uses_bound_geometry_without_reordering() -> None:
    rows = make_rows(make_plan(layerwise=True))
    first_layer, second_layer = enumerate_transfer_work((rows,))

    first = materialize_ranges(first_layer)
    second = materialize_key_ranges(second_layer)
    sessions = {key: (10_000 + index * 100, 64) for index, key in enumerate(first.keys)}
    gva = materialize_gva(first_layer, sessions)

    assert first.keys == list(rows.keys_by_coordinate[0])
    assert first.addresses == [[1064], [1192]]
    assert first.sizes == [[32], [32]]
    assert first.offsets == [[0], [0]]
    assert [source.row_index for source in first_layer.sources] == [0, 1]

    assert second.keys == list(rows.keys_by_coordinate[0])
    assert second.addresses == [[2064], [2192]]
    assert second.sizes == [[32], [32]]
    assert second.offsets == [[32], [32]]
    assert [source.row_index for source in second_layer.sources] == [0, 1]

    assert gva.remote_addresses.tolist() == [10_000, 10_100]
    assert gva.local_addresses.tolist() == [1064, 1192]
    assert gva.sizes.tolist() == [32, 32]


def test_dynamic_validation_is_confined_to_request_rows_and_work_selection() -> None:
    plan = make_plan(layerwise=True)
    valid = make_rows(plan, block_ids=(1,))
    chunk = valid.chunks[0]

    invalid_cases = (
        lambda: TransferRows(
            plan,
            (chunk,),
            np.asarray([1, 2], dtype=np.uint64),
            np.asarray([4], dtype=np.uint64),
            (("key",),),
            valid.remote_objects_by_coordinate,
        ),
        lambda: make_rows(plan, block_ids=(1,), token_counts=(5,)),
        lambda: make_rows(plan, block_ids=(8,), token_counts=(4,)),
        lambda: bind_transfer_rows(
            plan,
            KVBlockAssignmentBatch(
                1,
                (KVBlockAssignment(KVChunk(1, 0, chunk.token_range, b"x"), 1, 4),),
            ),
        ),
        lambda: TransferSpan(valid, (len(plan.layouts),), (0,)),
        lambda: TransferSpan(valid, (0,), (valid.row_count,)),
    )
    for build_invalid in invalid_cases:
        with pytest.raises(ValueError):
            build_invalid()

    duplicate_chunks = tuple(KVChunk(plan.group_id, index, chunk.token_range, b"same") for index in range(2))
    duplicate_rows = bind_transfer_rows(
        plan,
        KVBlockAssignmentBatch(
            plan.group_id,
            tuple(
                KVBlockAssignment(duplicate_chunk, block_id, 4)
                for duplicate_chunk, block_id in zip(duplicate_chunks, (1, 3), strict=True)
            ),
        ),
    )
    duplicate_work = enumerate_transfer_work((duplicate_rows,))
    duplicate_key = duplicate_rows.keys_by_coordinate[0][0]
    selected = select_work_keys(duplicate_work, {duplicate_key}, set())
    assert [[source.row_index for source in item.sources] for item in selected] == [[0], [0]]

    work = enumerate_transfer_work((valid,))[0]
    with pytest.raises(RuntimeError, match="GVA session"):
        materialize_gva(work, {})


def test_backend_evidence_retains_the_exact_transfer_source() -> None:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.backend import BackendIO

    work = enumerate_transfer_work((make_rows(make_plan(layerwise=False)),))[0]
    backend = FakeBackend()
    backend.get_result = [0, -7]
    backend_io = BackendIO(backend, make_backend_spec(layerwise_access=None))

    load = backend_io.load(work)

    assert [item.result_code for item in load] == [0, -7]
    assert [item.source.row_index for item in load] == [0, 1]
    assert [item.source.block_id for item in load] == [1, 3]
    assert [item.source.group_id for item in load] == [0, 0]

    backend.get_result = [0]
    malformed = backend_io.load(work)
    assert [item.result_code for item in malformed] == [None, None]

    backend.put_result = [0, 0]
    stored = backend_io.store(work)
    assert stored.succeeded and stored.source_release_confirmed
    assert [item.source.row_index for item in stored.transfer_evidence] == [0, 1]
    assert all(item.source_release_confirmed is True for item in stored.transfer_evidence)

    backend.put_result = RuntimeError("put failed")
    failed = backend_io.store(work)
    assert not failed.succeeded
    assert not failed.source_release_confirmed
    assert all(item.result_code is None for item in failed.transfer_evidence)
    assert all(item.source_release_confirmed is False for item in failed.transfer_evidence)
    assert isinstance(failed.error, RuntimeError)
