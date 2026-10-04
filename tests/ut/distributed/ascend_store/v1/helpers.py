"""Small fixtures expressed in the public v1 execution vocabulary."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import KeyMetadata
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.backend import (
    BackendSpec,
    LayerwiseAccessKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.rules import (
    KVPoolRuleSpec,
    compile_kv_pool_rules,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.runtime.runtime import KVPoolRuntime
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.timeline.schedule import (
    KVPoolSchedule,
    LoadScheduleKind,
    StoreScheduleKind,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.topology import (
    KVPoolGroupTopology,
    KVPoolLayerTopology,
    KVPoolTopology,
    TPPartitionSpec,
)


class FakeBackend:
    requires_exists_before_put = False

    def __init__(self) -> None:
        self.presence: list[int] | None = None
        self.get_result: object = None
        self.put_result: object = None
        self.session_start_result: list[int] | None = None
        self.session_copy_result: object = None
        self.session_end_result = 0
        self.store_session_start_result: list[int] | None = None
        self.store_session_copy_result: object = None
        self.store_session_commit_result: list[int] | None = None
        self.store_session_revoke_result: list[int] | None = None
        self.calls: list[tuple[Any, ...]] = []
        self.closed = False

    def exists(self, keys):
        self.calls.append(("exists", tuple(keys)))
        return self.presence if self.presence is not None else [1] * len(keys)

    def get(self, keys, addresses, sizes):
        self.calls.append(("get", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
        return self.get_result if self.get_result is not None else [0] * len(keys)

    def put(self, keys, addresses, sizes):
        self.calls.append(("put", tuple(keys), tuple(map(tuple, addresses)), tuple(map(tuple, sizes))))
        if isinstance(self.put_result, BaseException):
            raise self.put_result
        return self.put_result if self.put_result is not None else [0] * len(keys)

    def set_device(self) -> None:
        self.calls.append(("set_device",))

    def register_buffer(self, addresses, sizes) -> None:
        self.calls.append(("register_buffer", tuple(addresses), tuple(sizes)))

    def validate_layerwise_support(self) -> None:
        self.calls.append(("validate_layerwise_support",))

    def batch_get_start(self, keys):
        self.calls.append(("batch_get_start", tuple(keys)))
        return self.session_start_result if self.session_start_result is not None else [0] * len(keys)

    def batch_copy_get(self, keys, addresses, sizes, offsets):
        self.calls.append(
            (
                "batch_copy_get",
                tuple(keys),
                tuple(map(tuple, addresses)),
                tuple(map(tuple, sizes)),
                tuple(map(tuple, offsets)),
            )
        )
        if isinstance(self.session_copy_result, BaseException):
            raise self.session_copy_result
        return self.session_copy_result if self.session_copy_result is not None else [0] * len(keys)

    def batch_get_end(self, keys):
        self.calls.append(("batch_get_end", tuple(keys)))
        return self.session_end_result

    def batch_put_start(self, keys, object_sizes):
        self.calls.append(("batch_put_start", tuple(keys), tuple(object_sizes)))
        return self.store_session_start_result if self.store_session_start_result is not None else [0] * len(keys)

    def batch_copy_put(self, keys, addresses, sizes, offsets):
        self.calls.append(
            (
                "batch_copy_put",
                tuple(keys),
                tuple(map(tuple, addresses)),
                tuple(map(tuple, sizes)),
                tuple(map(tuple, offsets)),
            )
        )
        if isinstance(self.store_session_copy_result, BaseException):
            raise self.store_session_copy_result
        return self.store_session_copy_result if self.store_session_copy_result is not None else [0] * len(keys)

    def batch_commit(self, keys):
        self.calls.append(("batch_commit", tuple(keys)))
        return self.store_session_commit_result if self.store_session_commit_result is not None else [0] * len(keys)

    def batch_revoke(self, keys):
        self.calls.append(("batch_revoke", tuple(keys)))
        return self.store_session_revoke_result if self.store_session_revoke_result is not None else [0] * len(keys)

    def close(self) -> None:
        self.closed = True


def make_backend_spec(
    *,
    name: str = "fake",
    layerwise_access: LayerwiseAccessKind | None = LayerwiseAccessKind.KEY_RANGE,
    requires_exists_before_put: bool = False,
) -> BackendSpec:
    return BackendSpec(
        name,
        FakeBackend,
        SimpleNamespace(),
        layerwise_access,
        requires_exists_before_put,
    )


def make_topology(
    *,
    group_ids: tuple[int, ...] = (0,),
    physical_layers: tuple[int, ...] = (0, 1),
    tp_mismatch: bool = False,
    tp_rank: int = 0,
    consumer_pipeline_partitions: tuple[int, ...] | None = None,
) -> KVPoolTopology:
    groups = tuple(
        KVPoolGroupTopology(
            group_id,
            FullAttentionSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
            tuple(
                KVPoolLayerTopology(layer_id, (f"layers.{layer_id}.group.{group_id}",)) for layer_id in physical_layers
            ),
            KeyMetadata("model", 0, 0, 0, group_id),
        )
        for group_id in range(max(group_ids) + 1)
    )
    return KVPoolTopology(
        tp_rank=tp_rank,
        tp_size=1,
        pp_size=1,
        pcp_rank=0,
        pcp_size=1,
        dcp_size=1,
        put_step=1,
        cache_transfer_granularity=4,
        hash_block_size=4,
        tp_partition=TPPartitionSpec(
            tp_mismatch,
            2 if tp_mismatch else 1,
            2 if tp_mismatch else 1,
        ),
        groups=groups,
        transfer_group_ids=group_ids,
        consumer_pipeline_partitions=consumer_pipeline_partitions,
    )


def make_schedule(*, layerwise: bool = False, store: bool = True) -> KVPoolSchedule:
    return KVPoolSchedule(
        LoadScheduleKind.LAYERWISE if layerwise else LoadScheduleKind.SYNC,
        (StoreScheduleKind.LAYERWISE if layerwise else StoreScheduleKind.ASYNC) if store else None,
        2,
    )


class FakeEvent:
    def __init__(self) -> None:
        self.recorded = False
        self.synchronized = False

    def record(self) -> None:
        self.recorded = True

    def synchronize(self) -> None:
        self.synchronized = True


class FakeResources:
    def __init__(
        self,
        backend: FakeBackend,
        backend_spec: BackendSpec,
        topology: KVPoolTopology,
    ) -> None:
        self.backend = backend
        self.backend_spec = backend_spec
        self._topology = topology
        self.num_blocks = 8
        self.kv_caches = None
        self.closed = False

    def bind_kv_caches(self, kv_caches):
        self.kv_caches = kv_caches
        base_addresses = {}
        block_lengths = {}
        block_strides = {}
        layer_entry_offsets = {}
        object_sizes = {}
        object_offsets = {}
        for group in self._topology.transfer_groups:
            group_id = group.group_id
            base_addresses[group_id] = [
                1000 + group_id * 10000 + layer.physical_layer_id * 1000 for layer in group.layers
            ]
            block_lengths[group_id] = [32] * len(group.layers)
            block_strides[group_id] = [64] * len(group.layers)
            layer_entry_offsets[group_id] = list(range(len(group.layers) + 1))
            object_sizes[group_id] = 32 * len(group.layers)
            object_offsets[group_id] = 0
        return {
            "base_addresses": base_addresses,
            "block_lengths": block_lengths,
            "block_strides": block_strides,
            "layer_entry_offsets": layer_entry_offsets,
            "object_sizes": object_sizes,
            "object_offsets": object_offsets,
        }

    def close(self) -> None:
        self.closed = True
        self.kv_caches = None


def make_runtime(
    backend: FakeBackend | None = None,
    *,
    topology: KVPoolTopology | None = None,
    layerwise: bool = False,
    async_load: bool = False,
    store: bool = True,
    physical_layers: tuple[int, ...] = (0, 1),
    requires_exists_before_put: bool = False,
    source_ready_event_factory=None,
    start_gate_factory=None,
):
    backend = backend or FakeBackend()
    topology = topology or make_topology(physical_layers=physical_layers)
    schedule = KVPoolSchedule(
        LoadScheduleKind.LAYERWISE if layerwise else (LoadScheduleKind.ASYNC if async_load else LoadScheduleKind.SYNC),
        (StoreScheduleKind.LAYERWISE if layerwise else StoreScheduleKind.ASYNC) if store else None,
        2,
    )
    backend_spec = make_backend_spec(
        layerwise_access=LayerwiseAccessKind.KEY_RANGE if layerwise else None,
        requires_exists_before_put=requires_exists_before_put,
    )
    resources = FakeResources(backend, backend_spec, topology)
    rule_spec = KVPoolRuleSpec(topology, "fake", 64, use_layerwise=layerwise)
    full_key = lambda group, value, head, stage: f"g{group}:p{stage}:h{head}:{value}"
    with patch(
        "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.rules.compiler.resolve_backend_spec",
        return_value=backend_spec,
    ):
        rule_binder = compile_kv_pool_rules(
            rule_spec,
            layerwise_full_key=full_key if layerwise else None,
        )
    runtime = KVPoolRuntime(
        topology,
        schedule,
        rule_binder,
        resources,
        **({"start_gate_factory": start_gate_factory} if start_gate_factory is not None else {}),
        source_ready_event_factory=source_ready_event_factory or FakeEvent,
    )
    runtime.bind_kv_caches({"cache": object()})
    return runtime, resources, backend
