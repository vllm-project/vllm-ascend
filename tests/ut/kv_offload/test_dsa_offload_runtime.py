# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.worker.gpu_input_batch import InputBatch

from vllm_ascend.attention import dsa_offload_runtime
from vllm_ascend.attention.dsa_offload_runtime import (
    DsaGroupMetadata,
    DsaGroupSpec,
    DsaOffloadRuntime,
)
from vllm_ascend.attention.dsa_offload_state import (
    DsaLayerWorkspace,
    DsaResidentState,
)
from vllm_ascend.ops.dsa_offload import DsaOffloadConfig
from vllm_ascend.worker.npu_input_batch import NPUInputBatch


def _workspace(layer_id: int, batch_capacity: int = 2) -> DsaLayerWorkspace:
    selection_rows = batch_capacity * 4
    selection_blocks = selection_rows * 16
    return DsaLayerWorkspace(
        layer_id=layer_id,
        resident_kv_cache=torch.empty((batch_capacity, 16, 4)),
        resident_k_rope=torch.empty((batch_capacity, 16, 2)),
        selection_kv_cache=torch.empty((selection_blocks, 128, 4)),
        selection_k_rope=torch.empty((selection_blocks, 128, 2)),
        selection_block_table=torch.arange(
            selection_blocks, dtype=torch.int32
        ).view(selection_rows, 16),
    )


def _metadata(batch_capacity: int = 2) -> DsaGroupMetadata:
    return DsaGroupMetadata(
        pool_ids=torch.full((batch_capacity, 16), -1, dtype=torch.int32),
        id_to_slot=torch.full((batch_capacity, 64), -1, dtype=torch.int32),
        lru_counter=torch.zeros((batch_capacity, 1), dtype=torch.int32),
    )


def test_group_plan_serve_install_order_and_active_batch_slicing(monkeypatch):
    calls = []

    class Event:
        def wait(self):
            pass

    def fake_plan(topk, seq, pool_ids, id_to_slot, lru_counter, **kwargs):
        calls.append(
            (
                "plan",
                kwargs["group_id"],
                topk.shape,
                pool_ids.shape,
                id_to_slot.shape,
                lru_counter.shape,
            )
        )
        rows = topk.numel() // topk.shape[-1]
        return (
            torch.zeros((rows * 2048, 2), dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int8),
            torch.full((rows,), 2048, dtype=torch.int32),
        )

    def fake_serve(
        plan,
        full_kv_cache,
        full_k_rope,
        pool_kv_cache,
        pool_k_rope,
        selection_kv_cache,
        selection_k_rope,
        **kwargs,
    ):
        del plan, full_kv_cache, full_k_rope, pool_k_rope, selection_k_rope
        calls.append(
            (
                "serve",
                pool_kv_cache.shape,
                selection_kv_cache.shape,
                kwargs["full_kv_block_table"].shape,
            )
        )
        return Event()

    def fake_install(
        install_records,
        selection_kv_cache,
        selection_k_rope,
        selection_block_table,
        pool_kv_cache,
        pool_k_rope,
        pool_ids,
        id_to_slot,
        lru_counter,
        **kwargs,
    ):
        del (
            install_records,
            selection_kv_cache,
            selection_k_rope,
            selection_block_table,
            pool_kv_cache,
            pool_k_rope,
            id_to_slot,
            lru_counter,
        )
        calls.append(("install", kwargs["metadata_update"], pool_ids.shape))
        return Event()

    monkeypatch.setattr(dsa_offload_runtime, "dsa_plan", fake_plan)
    monkeypatch.setattr(dsa_offload_runtime, "dsa_serve", fake_serve)
    monkeypatch.setattr(dsa_offload_runtime, "dsa_install", fake_install)

    groups = (
        DsaGroupSpec(group_id=0, owner_layer=0, layers=(0, 1)),
        DsaGroupSpec(group_id=1, owner_layer=2, layers=(2,)),
    )
    runtime = DsaOffloadRuntime(
        groups,
        {0: _metadata(), 1: _metadata()},
        DsaResidentState([_workspace(0), _workspace(1), _workspace(2)]),
    )
    config = DsaOffloadConfig(raw_seq=1)
    topk = torch.zeros((1, 1, 2048), dtype=torch.int32)
    seq = torch.tensor([128], dtype=torch.int32)
    full_cache = torch.empty((4, 128, 4))
    full_rope = torch.empty((4, 128, 2))
    block_table = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32)

    runtime.plan_group(0, topk, seq, config=config)
    runtime.serve_layer(
        0, full_cache, full_rope, block_table, seq, config=config
    )
    runtime.install_after_layer(0, config=config)
    runtime.serve_layer(
        1, full_cache, full_rope, block_table, seq, config=config
    )
    runtime.install_after_layer(1, config=config)

    runtime.plan_group(1, topk, seq, config=config)
    runtime.serve_layer(
        2, full_cache, full_rope, block_table, seq, config=config
    )
    runtime.install_after_layer(2, config=config)

    assert [call[0] for call in calls] == [
        "plan",
        "serve",
        "serve",
        "install",
        "install",
        "plan",
        "serve",
        "install",
    ]
    assert calls[0][3:] == (
        torch.Size([1, 16]),
        torch.Size([1, 64]),
        torch.Size([1, 1]),
    )
    assert calls[1][1:] == (
        torch.Size([1, 16, 4]),
        torch.Size([16, 128, 4]),
        torch.Size([1, 4]),
    )
    assert [call[1] for call in calls if call[0] == "install"] == [0, 1, 1]


def test_row_condense_cold_invalidates_source_and_destination():
    metadata = _metadata(batch_capacity=2)
    metadata.pool_ids[1].fill_(7)
    metadata.id_to_slot[1].fill_(8)
    metadata.lru_counter[1].fill_(9)
    state = DsaResidentState([_workspace(0)])
    runtime = DsaOffloadRuntime(
        (DsaGroupSpec(group_id=0, owner_layer=0, layers=(0,)),),
        {0: metadata},
        state,
    )

    # remove row 0 followed by InputBatch.condense() moving row 1 to row 0.
    runtime.queue_row_changes(
        moves=[(1, 0)],
        invalidated=[0],
    )
    runtime.begin_step()

    assert torch.all(metadata.pool_ids[0] == -1)
    assert torch.all(metadata.id_to_slot[0] == -1)
    assert torch.all(metadata.lru_counter[0] == 0)
    assert torch.all(metadata.pool_ids[1] == -1)
    assert torch.all(metadata.id_to_slot[1] == -1)
    assert torch.all(metadata.lru_counter[1] == 0)


def test_chained_row_moves_cold_invalidate_every_touched_row():
    metadata = _metadata(batch_capacity=3)
    metadata.pool_ids.fill_(7)
    metadata.id_to_slot.fill_(8)
    metadata.lru_counter.fill_(9)
    runtime = DsaOffloadRuntime(
        (DsaGroupSpec(group_id=0, owner_layer=0, layers=(0,)),),
        {0: metadata},
        DsaResidentState([_workspace(0, batch_capacity=3)]),
    )

    runtime.queue_row_changes(
        moves=[(2, 0), (0, 1)],
        invalidated=[],
    )
    runtime.prepare_step()

    assert torch.all(metadata.pool_ids == -1)
    assert torch.all(metadata.id_to_slot == -1)
    assert torch.all(metadata.lru_counter == 0)


def test_row_replacement_without_move_clears_metadata():
    metadata = _metadata(batch_capacity=2)
    metadata.pool_ids[0].fill_(7)
    state = DsaResidentState([_workspace(0)])
    runtime = DsaOffloadRuntime(
        (DsaGroupSpec(group_id=0, owner_layer=0, layers=(0,)),),
        {0: metadata},
        state,
    )
    runtime.queue_row_changes(
        moves=[],
        invalidated=[0],
    )
    # This is also the full-graph replay path: row updates are submitted
    # outside the model graph without opening the Python Plan state machine.
    runtime.prepare_step()
    runtime.prepare_step()

    assert torch.all(metadata.pool_ids[0] == -1)


def test_new_request_invalidates_empty_row_after_graph_warmup(monkeypatch):
    notifications = []

    def record_changes(moves, invalidated):
        notifications.append((moves, invalidated))

    batch = object.__new__(NPUInputBatch)
    batch._req_ids = [None]
    batch._dsa_row_change_callback = record_changes
    monkeypatch.setattr(InputBatch, "add_request", lambda self, request: 0)

    request = SimpleNamespace(req_id="req-a")
    assert batch.add_request(request) == 0

    assert notifications == [([], [0])]


def test_group_builder_preserves_full_shared_ownership():
    assert dsa_offload_runtime.build_dsa_group_specs(
        ["full", "shared", "full", "shared"], 4
    ) == (
        DsaGroupSpec(group_id=0, owner_layer=0, layers=(0, 1)),
        DsaGroupSpec(group_id=1, owner_layer=2, layers=(2, 3)),
    )


def test_group_builder_rejects_unresolved_owner_mapping():
    with pytest.raises(TypeError, match="sequence"):
        dsa_offload_runtime.build_dsa_group_specs({0: 0, 1: 0}, 2)
