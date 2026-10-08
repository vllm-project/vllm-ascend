"""TP-mismatch key slices and real CPU tensor bytes roundtrip in each supported layout."""

from __future__ import annotations

import ctypes
from dataclasses import replace

import pytest
import torch

from tests.ut.distributed.ascend_store.v1.helpers import (
    FakeBackend,
    FakeEvent,
    begin_step,
    make_backend_spec,
    make_topology,
    make_worker,
    store_one,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.projection import (
    compile_bulk_projection_binder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import (
    LoadCommand,
    RangeStoreCommand,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk import SynchronousBulkWorker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.resources import KVPoolResources


@pytest.mark.parametrize(
    ("topology", "expected_ranks", "expected_addresses", "expected_sizes"),
    (
        (
            make_topology(
                physical_layers=(0,),
                tp_mismatch=True,
                tp_rank=1,
                tp_size=2,
                key_rank_count=4,
                key_slices_per_rank=2,
            ),
            (2, 3),
            ((1128, 1136, 1144), (1132, 1140, 1148)),
            ((4, 4, 4), (4, 4, 4)),
        ),
        (
            make_topology(
                physical_layers=(0,),
                tp_mismatch=True,
                tp_rank=2,
                tp_size=4,
                key_rank_count=4,
                key_slices_per_rank=1,
            ),
            (2,),
            ((1128,),),
            ((24,),),
        ),
    ),
)
def test_tp_mismatch_bulk_binds_effective_keys_to_exact_local_slices(
    topology,
    expected_ranks,
    expected_addresses,
    expected_sizes,
) -> None:
    backend = FakeBackend()
    worker, resources, _ = make_worker(backend, topology=topology)

    completion = store_one(
        worker,
        RangeStoreCommand("request", TokenRange(0, 3), ((2,),), (b"a",), 3, 17),
    )

    put_call = next(call for call in backend.calls if call[0] == "put")
    assert tuple(f"@head_or_tp_rank:{rank}@" in key for key, rank in zip(put_call[1], expected_ranks, strict=True)) == (
        True,
    ) * len(expected_ranks)
    assert put_call[2] == expected_addresses
    assert put_call[3] == expected_sizes
    assert [item.source.block_id for item in completion.evidence.transfer_evidence] == [2] * len(expected_ranks)

    worker.close()
    assert resources.closed


def test_tp_mismatch_bulk_preserves_axis_result_provenance() -> None:
    topology = make_topology(
        physical_layers=(0,),
        tp_mismatch=True,
        tp_rank=1,
        tp_size=2,
        key_rank_count=4,
        key_slices_per_rank=2,
    )
    backend = FakeBackend()
    backend.get_result = [0, -1]
    worker, resources, _ = make_worker(backend, topology=topology, store=False)
    command = LoadCommand("request", TokenRange(0, 3), ((2,),), (b"a",))

    batch = worker._build_load_batch((command,))
    (completion,) = worker._execute_bulk_load(batch)

    assert [item.result_code for item in completion.transfer_evidence] == [0, -1]
    assert [item.source.block_id for item in completion.transfer_evidence] == [2, 2]
    assert [
        "@head_or_tp_rank:2@" in completion.transfer_evidence[0].source.key,
        "@head_or_tp_rank:3@" in completion.transfer_evidence[1].source.key,
    ] == [True, True]

    worker.close()
    assert resources.closed


class TensorBytesBackend(FakeBackend):
    """Copy actual CPU tensor bytes through the public Bulk Backend contract."""

    def __init__(self) -> None:
        super().__init__()
        self.objects: dict[str, bytes] = {}
        self.registered_regions: list[tuple[int, int]] = []

    def register_buffer(self, addresses, sizes):
        self.registered_regions.extend(zip(addresses, sizes, strict=True))
        return super().register_buffer(addresses, sizes)

    def _assert_registered_ranges(self, addresses, sizes) -> None:
        for row_addresses, row_sizes in zip(addresses, sizes, strict=True):
            for address, size in zip(row_addresses, row_sizes, strict=True):
                assert any(
                    start <= address and address + size <= start + span for start, span in self.registered_regions
                )

    def store(self, keys, addresses, sizes):
        self._assert_registered_ranges(addresses, sizes)
        codes = self.put(keys, addresses, sizes)
        for key, row_addresses, row_sizes in zip(keys, addresses, sizes, strict=True):
            self.objects[key] = b"".join(
                ctypes.string_at(address, size) for address, size in zip(row_addresses, row_sizes, strict=True)
            )
        return codes

    def load(self, keys, addresses, sizes):
        self._assert_registered_ranges(addresses, sizes)
        codes = self.get(keys, addresses, sizes)
        for key, row_addresses, row_sizes in zip(keys, addresses, sizes, strict=True):
            payload = self.objects[key]
            assert sum(row_sizes) == len(payload)
            offset = 0
            for address, size in zip(row_addresses, row_sizes, strict=True):
                ctypes.memmove(address, payload[offset : offset + size], size)
                offset += size
        return codes


def _make_tensor_tp_worker(backend, layout, tp_size, tp_rank, *, total_heads, populate_cache, padded_kernel_blocks):
    cache_block_size = 4
    num_cache_blocks = 8
    local_heads = total_heads // tp_size
    topology = make_topology(
        physical_layers=(0, 1),
        tp_mismatch=True,
        tp_rank=tp_rank,
        tp_size=tp_size,
        key_rank_count=2,
        key_slices_per_rank=2 // tp_size,
    )
    spec = replace(topology.groups[0].kv_cache_spec, num_kv_heads=local_heads, head_size=2, head_size_v=3)
    topology = replace(topology, groups=(replace(topology.groups[0], kv_cache_spec=spec),))
    head_major = layout in ("HND", "LBHNC")
    kernel_block_size = 2 if padded_kernel_blocks else cache_block_size
    block_scale = cache_block_size // kernel_block_size
    caches = {}
    expected = {}
    for layer_index, name in enumerate(topology.groups[0].layer_names):
        entries = []
        expected_entries = []
        for entry_index, head_dim in enumerate((2, 3)):
            shape = (
                (num_cache_blocks * block_scale, local_heads, kernel_block_size, head_dim)
                if head_major
                else (num_cache_blocks * block_scale, kernel_block_size, local_heads, head_dim)
            )
            cache = torch.zeros(shape, dtype=torch.float32)
            if padded_kernel_blocks:
                kernel_elements = cache[0].numel()
                storage = torch.zeros(shape[0] * (kernel_elements + 1), dtype=cache.dtype)
                cache = torch.as_strided(storage, shape, (kernel_elements + 1, *cache.stride()[1:]))
            values = (
                layer_index * 10_000
                + entry_index * 1_000
                + torch.arange(cache_block_size, dtype=torch.float32)[:, None, None] * 100
                + (tp_rank * local_heads + torch.arange(local_heads, dtype=torch.float32))[None, :, None] * 10
                + torch.arange(head_dim, dtype=torch.float32)[None, None, :]
            )
            if populate_cache:
                for kernel_index in range(block_scale):
                    kernel_values = values[kernel_index * kernel_block_size : (kernel_index + 1) * kernel_block_size]
                    cache[7 * block_scale + kernel_index].copy_(
                        kernel_values.permute(1, 0, 2) if head_major else kernel_values
                    )
            entries.append(cache)
            expected_entries.append(values)
        caches[name] = tuple(entries)
        expected[name] = tuple(expected_entries)
    resources = KVPoolResources(
        backend, make_backend_spec(layerwise_access=None), num_cache_blocks, topology.transfer_groups
    )
    binder = compile_bulk_projection_binder(topology, 64, kv_cache_layout=layout)
    worker = SynchronousBulkWorker(topology, binder, resources, source_ready_event_factory=FakeEvent)
    worker.bind_kv_caches(caches)
    return worker, caches, expected


@pytest.mark.parametrize("layout", ("NHD", "HND", "LBHNC"))
@pytest.mark.parametrize("producer_tp_size", (1, 2))
@pytest.mark.parametrize("padded_kernel_blocks", (False, True))
@pytest.mark.parametrize("total_heads", (2, 4))
def test_tp_mismatch_roundtrip_preserves_tensor_values_in_each_layout(
    layout, producer_tp_size, padded_kernel_blocks, total_heads
) -> None:
    backend = TensorBytesBackend()
    workers = []
    expected_objects = {}
    try:
        for rank in range(producer_tp_size):
            worker, _caches, expected = _make_tensor_tp_worker(
                backend,
                layout,
                producer_tp_size,
                rank,
                total_heads=total_heads,
                populate_cache=True,
                padded_kernel_blocks=padded_kernel_blocks,
            )
            workers.append(worker)
            completion = store_one(worker, RangeStoreCommand("producer", TokenRange(0, 4), ((7,),), (b"a",), 4, 17))
            assert completion.evidence.succeeded
            slice_count = 2 // producer_tp_size
            kernel_tokens = 2 if padded_kernel_blocks else 4
            heads_per_key = total_heads // 2
            for slice_index in range(slice_count):
                payload_parts = []
                for entries in expected.values():
                    for values in entries:
                        # Each key must match the corresponding higher-TP rank's packed Bulk object.
                        head_slice = values[:, slice_index * heads_per_key : (slice_index + 1) * heads_per_key]
                        for start in range(0, len(head_slice), kernel_tokens):
                            kernel = head_slice[start : start + kernel_tokens]
                            if layout in ("HND", "LBHNC"):
                                kernel = kernel.permute(1, 0, 2)
                            payload_parts.append(kernel.contiguous().numpy().tobytes())
                expected_objects[rank * slice_count + slice_index] = b"".join(payload_parts)
        assert len(backend.objects) == 2
        assert [backend.objects[key] for key in sorted(backend.objects)] == [
            expected_objects[rank] for rank in range(2)
        ]

        consumer_tp_size = 3 - producer_tp_size
        for rank in range(consumer_tp_size):
            worker, caches, expected = _make_tensor_tp_worker(
                backend,
                layout,
                consumer_tp_size,
                rank,
                total_heads=total_heads,
                populate_cache=False,
                padded_kernel_blocks=padded_kernel_blocks,
            )
            workers.append(worker)
            begin_step(worker, load=(LoadCommand("consumer", TokenRange(0, 4), ((7,),), (b"a",)),))
            worker.start_load()
            assert not worker.collect_load_result().failed_block_ids
            worker.end_step()
            block_scale = 2 if padded_kernel_blocks else 1
            for name, entries in caches.items():
                for cache, values in zip(entries, expected[name], strict=True):
                    block = cache[7 * block_scale : 8 * block_scale]
                    if layout in ("HND", "LBHNC"):
                        block = block.permute(0, 2, 1, 3)
                    torch.testing.assert_close(block.reshape(values.shape), values)
    finally:
        for worker in workers:
            worker.close()
