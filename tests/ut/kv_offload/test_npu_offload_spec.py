# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""vLLM v0.27.1 contract tests for native NPU KV offloading."""

import ctypes
import inspect
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

import vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.cpu_npu as cpu_npu_mod
import vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu as npu_mod
import vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector as connector_mod
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector import (
    AscendOffloadingConnectorWorker,
)


def test_npu_worker_routes_store_and_load_by_direction() -> None:
    calls = []

    class FakeHandler:
        def __init__(self, direction):
            self.direction = direction

        def transfer_async(self, job_id, src_spec, dst_spec):
            calls.append((self.direction, job_id, src_spec, dst_spec))
            return True

    worker = npu_mod.NPUOffloadingWorker.__new__(npu_mod.NPUOffloadingWorker)
    worker._store_handler = FakeHandler("store")  # type: ignore[assignment]
    worker._load_handler = FakeHandler("load")  # type: ignore[assignment]

    assert worker.submit_store(1, "npu", "cpu")
    assert worker.submit_load(2, "cpu", "npu")
    assert calls == [
        ("store", 1, "npu", "cpu"),
        ("load", 2, "cpu", "npu"),
    ]


def test_npu_worker_releases_handlers_before_mmap() -> None:
    calls = []

    class FakeHandler:
        def __init__(self, name):
            self.name = name

        def shutdown(self):
            calls.append(self.name)

    class FakeRegion:
        def cleanup(self):
            calls.append("mmap")

    worker = npu_mod.NPUOffloadingWorker.__new__(npu_mod.NPUOffloadingWorker)
    worker._store_handler = FakeHandler("store")  # type: ignore[assignment]
    worker._load_handler = FakeHandler("load")  # type: ignore[assignment]
    worker._mmap_region = FakeRegion()

    worker.shutdown()

    assert calls == ["store", "load", "mmap"]
    assert worker._mmap_region is None


@pytest.mark.parametrize("view_method", ["create_next_worker_view", "create_next_view"])
def test_npu_worker_supports_shared_region_view_apis(
    monkeypatch: pytest.MonkeyPatch,
    view_method: str,
) -> None:
    created_page_sizes = []
    captured_cpu_tensors = []

    class FakeRegion:
        def cleanup(self):
            pass

    def create_view(self, page_size_bytes):
        created_page_sizes.append(page_size_bytes)
        return torch.empty((3, page_size_bytes), dtype=torch.int8)

    monkeypatch.setattr(FakeRegion, view_method, create_view, raising=False)

    class FakeHandler:
        def __init__(self, **kwargs):
            captured_cpu_tensors.append(kwargs["cpu_tensors"])

        def shutdown(self):
            pass

    monkeypatch.setattr(
        cpu_npu_mod,
        "SingleDirectionNPUOffloadingHandler",
        FakeHandler,
    )
    kv_caches = SimpleNamespace(
        tensors=[
            SimpleNamespace(
                page_size_bytes=4,
                tensor=torch.empty((6, 4), dtype=torch.int8),
            )
        ],
        group_data_refs=[],
    )

    worker = cpu_npu_mod.NPUOffloadingWorker(
        kv_caches=kv_caches,
        blocks_per_chunk=2,
        num_cpu_blocks=3,
        mmap_region=FakeRegion(),
    )

    assert created_page_sizes == [8]
    assert len(captured_cpu_tensors) == 2
    assert captured_cpu_tensors[0][0] is captured_cpu_tensors[1][0]
    assert captured_cpu_tensors[0][0].shape == (3, 8)
    worker.shutdown()


def test_npu_worker_cleans_shared_region_for_unsupported_view_api() -> None:
    class FakeRegion:
        cleaned = False

        def cleanup(self):
            self.cleaned = True

    region = FakeRegion()
    kv_caches = SimpleNamespace(tensors=[], group_data_refs=[])

    with pytest.raises(AttributeError, match="create_next_view"):
        cpu_npu_mod.NPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=2,
            num_cpu_blocks=3,
            mmap_region=region,
        )

    assert region.cleaned


def test_cpu_spec_uses_v027_blocks_per_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_worker(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(npu_mod, "NPUOffloadingWorker", fake_worker)
    spec = npu_mod.NPUOffloadingSpec.__new__(npu_mod.NPUOffloadingSpec)
    spec.blocks_per_chunk = 4
    spec.num_chunks = 17

    spec.create_worker(object())

    assert captured["blocks_per_chunk"] == 4
    assert captured["num_cpu_blocks"] == 17
    assert "mmap_region" not in captured


def test_only_tiering_spec_overrides_shared_region_capability() -> None:
    tiering_spec = npu_mod.NPUTieringOffloadingSpec.__new__(npu_mod.NPUTieringOffloadingSpec)

    assert "_uses_shared_region" not in npu_mod.NPUOffloadingSpec.__dict__
    assert tiering_spec._uses_shared_region()


@pytest.mark.parametrize(
    ("replicated_layout", "device", "world_size", "expected_rank"),
    [
        (False, 5, 4, 1),
        (True, 5, 4, 0),
    ],
)
def test_tiering_worker_matches_v027_shared_region_contract(
    monkeypatch: pytest.MonkeyPatch,
    replicated_layout: bool,
    device: int,
    world_size: int,
    expected_rank: int,
) -> None:
    captured: dict = {}

    class FakeRegion:
        def __init__(self, **kwargs):
            inspect.signature(SharedOffloadRegion.__init__).bind(self, **kwargs)
            captured.update(kwargs)
            captured["region"] = self

    sentinel_worker = object()

    def fake_worker(**kwargs):
        captured["worker_kwargs"] = kwargs
        return sentinel_worker

    monkeypatch.setattr(npu_mod, "SharedOffloadRegion", FakeRegion)
    monkeypatch.setattr(npu_mod, "NPUOffloadingWorker", fake_worker)
    monkeypatch.setattr(
        npu_mod.torch,
        "npu",
        SimpleNamespace(current_device=lambda: device),
        raising=False,
    )

    spec = npu_mod.NPUTieringOffloadingSpec.__new__(npu_mod.NPUTieringOffloadingSpec)
    spec.config = SimpleNamespace(
        parallel=SimpleNamespace(world_size=world_size),
    )
    spec._engine_id = "engine-dp0"
    spec.replicated_layout = replicated_layout
    spec.cpu_page_size_per_worker = 64
    spec.num_chunks = 10
    spec.blocks_per_chunk = 2
    spec.kv_bytes_per_chunk = 4096
    kv_caches = object()

    result = spec.create_worker(kv_caches=kv_caches)

    assert result is sentinel_worker
    assert captured["engine_id"] == "engine-dp0"
    assert captured["num_chunks"] == 10
    assert captured["rank"] == expected_rank
    assert captured["kv_bytes_per_chunk"] == 4096
    assert captured["cpu_page_size"] == 64
    assert captured["worker_kwargs"] == {
        "kv_caches": kv_caches,
        "blocks_per_chunk": 2,
        "num_cpu_blocks": 10,
        "mmap_region": captured["region"],
    }


def _make_layout_worker(kv_cache_config, monkeypatch: pytest.MonkeyPatch):
    worker = AscendOffloadingConnectorWorker.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = kv_cache_config
    worker.vllm_config = object()
    captured: list[Any] = []
    worker._init_worker = captured.append
    monkeypatch.setattr(connector_mod, "derive_canonical_mappings", lambda *args: {})
    return worker, captured


def test_hybrid_overlapping_page_reproduces_reported_worker_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    page_size, state_size = 25952256, 98304
    attention_spec = FullAttentionSpec(
        block_size=1, num_kv_heads=1, head_size=1, dtype=torch.int8, page_size_padded=page_size
    )
    mamba_spec = MambaSpec(block_size=1, shapes=((state_size,),), dtypes=(torch.int8,), page_size_padded=page_size)
    config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attention"], kv_cache_spec=attention_spec),
            KVCacheGroupSpec(layer_names=["mamba"], kv_cache_spec=mamba_spec),
        ],
    )
    worker, captured = _make_layout_worker(config, monkeypatch)
    backing = torch.empty((1, page_size), dtype=torch.int8)

    worker.register_kv_caches({"attention": backing, "mamba": [backing[:, :state_size]]})

    region = SharedOffloadRegion.__new__(SharedOffloadRegion)
    region.rank = 0
    region.num_chunks = 1
    region._base = torch.empty(page_size, dtype=torch.int8)
    region._row_stride = page_size
    region._worker_offset = 0
    region._worker_area_end = page_size
    region._views = []
    for tensor in captured[0].tensors:
        region.create_next_worker_view(tensor.page_size_bytes)
    assert region._worker_offset == page_size
    assert [sum(ref.page_size_bytes for ref in refs) for refs in captured[0].group_data_refs] == [2, state_size]


@pytest.mark.parametrize("blocks_per_chunk", [1, 3])
@pytest.mark.parametrize("group_sizes", [(2, 2), (2, 0), (0, 2)])
def test_overlapping_groups_round_trip_with_real_transfer_descriptors(
    monkeypatch: pytest.MonkeyPatch, blocks_per_chunk: int, group_sizes: tuple[int, int]
) -> None:
    attention_spec = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=3, dtype=torch.int8)
    mamba_spec = MambaSpec(block_size=2, shapes=((3,), (4,)), dtypes=(torch.int8, torch.int8), page_size_padded=12)
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attention"], kv_cache_spec=attention_spec),
            KVCacheGroupSpec(layer_names=["mamba"], kv_cache_spec=mamba_spec),
        ],
    )
    connector, captured = _make_layout_worker(config, monkeypatch)
    backing = torch.arange(48, dtype=torch.int8).view(4, 12)
    original = backing.clone()
    connector.register_kv_caches(
        {"attention": (backing[:, :6], backing[:, 6:]), "mamba": [backing[:, 2:5], backing[:, 5:9]]}
    )
    canonical = captured[0]
    npu_tensors = [cache.tensor for cache in canonical.tensors]
    row_bytes = 12 * blocks_per_chunk
    host = torch.full((4, row_bytes + 13), -1, dtype=torch.int8)
    cpu_tensors = []
    offset = 0
    for cache in canonical.tensors:
        width = cache.page_size_bytes * blocks_per_chunk
        cpu_tensors.append(host[:, offset : offset + width])
        offset += width
    assert offset == row_bytes
    fake_npu = MagicMock()
    fake_npu.stream.side_effect = lambda stream: nullcontext()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(cpu_npu_mod, "is_pin_memory_available", lambda: False)
    copied = []

    def memcpy(src, dst, sizes, direction):
        for source, destination, size in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            ctypes.memmove(destination, source, size)
        copied.append(int(sizes.sum()))

    monkeypatch.setattr(torch.ops._C_ascend, "swap_blocks_batch", memcpy, raising=False)
    block_ids = ([0, 2] if group_sizes[0] else []) + ([1, 3] if group_sizes[1] else [])
    npu_spec = GPULoadStoreSpec(block_ids, group_sizes, [0, 1])
    cpu_ids = []
    for count, block_index in zip(group_sizes, (0, 1)):
        if count:
            chunks = (count + block_index % blocks_per_chunk + blocks_per_chunk - 1) // blocks_per_chunk
            cpu_ids.extend(range(len(cpu_ids), len(cpu_ids) + chunks))
    cpu_spec = BlockIDsLoadStoreSpec(cpu_ids)

    def handler(store):
        obj = cpu_npu_mod.SingleDirectionNPUOffloadingHandler.__new__(cpu_npu_mod.SingleDirectionNPUOffloadingHandler)
        obj.src_tensors, obj.dst_tensors = (npu_tensors, cpu_tensors) if store else (cpu_tensors, npu_tensors)
        obj.npu_to_cpu = store
        obj.src_blocks_per_chunk, obj.dst_blocks_per_chunk = (1, blocks_per_chunk) if store else (blocks_per_chunk, 1)
        obj.direction = cpu_npu_mod.DIRECTION_D2H if store else cpu_npu_mod.DIRECTION_H2D
        obj.kv_cache_groups_data_refs = canonical.group_data_refs
        obj._buffer_pool, obj._stream_pool, obj._event_pool = [], [], []
        obj._transfers, obj._transfer_events = deque(), {}
        return obj

    assert handler(True).transfer_async(1, npu_spec, cpu_spec)
    if group_sizes[0]:
        backing[[0, 2]] = 0
    if group_sizes[1]:
        backing[1, 2:9] = 0
        backing[3, 2:9] = 0
    assert handler(False).transfer_async(2, cpu_spec, npu_spec)
    torch.testing.assert_close(backing, original)
    assert copied == [group_sizes[0] * 12 + group_sizes[1] * 7] * 2
    assert torch.all(host[:, row_bytes:] == -1)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("overlaid", [False, True])
@pytest.mark.parametrize("scratch_position", [0, 1, 2])
@pytest.mark.parametrize("blocks_per_chunk", [1, 3])
def test_hybrid_registration_excludes_scratch_from_shared_region(
    monkeypatch: pytest.MonkeyPatch,
    scratch_position: int,
    blocks_per_chunk: int,
    overlaid: bool,
    rank: int,
) -> None:
    """Capacity and group refs must match the scheduler with MTP enabled."""
    num_blocks, num_chunks = 4, 2
    attention_spec = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=3, dtype=torch.int8)
    mamba_spec = MambaSpec(block_size=2, shapes=((2,), (3,)), dtypes=(torch.int8, torch.int8))
    scratch_spec = CircularBufferSpec(block_size=2, num_kv_heads=1, head_size=3, dtype=torch.int8)
    groups = [
        KVCacheGroupSpec(layer_names=["attention"], kv_cache_spec=attention_spec),
        KVCacheGroupSpec(layer_names=["mamba"], kv_cache_spec=mamba_spec),
    ]
    groups.insert(scratch_position, KVCacheGroupSpec(layer_names=["scratch"], kv_cache_spec=scratch_spec))
    groups.append(KVCacheGroupSpec(layer_names=["local"], kv_cache_spec=attention_spec, enable_kv_transfer=False))
    config = KVCacheConfig(num_blocks=num_blocks, kv_cache_groups=groups, kv_cache_tensors=[])
    connector, captured = _make_layout_worker(config, monkeypatch)
    attention = torch.arange(num_blocks * 12, dtype=torch.int8).view(num_blocks, 12)
    caches = {
        "attention": (attention[:, :6], attention[:, 6:]),
        "mamba": [torch.ones((num_blocks, 2), dtype=torch.int8), torch.ones((num_blocks, 3), dtype=torch.int8)],
        "scratch": (torch.empty((num_blocks, 6), dtype=torch.int8),) * 2,
        "local": (torch.empty((num_blocks, 6), dtype=torch.int8),) * 2,
    }

    if overlaid:
        caches["mamba"] = [attention[:, :2], attention[:, 2:5]]

    connector.register_kv_caches(caches)

    canonical = captured[0]
    # Exercise upstream's real worker-slot bound, without allocating /dev/shm
    # or requiring NPU DMA. Only eligible groups fit in this slot.
    payload_size = attention_spec.page_size_bytes
    if not overlaid:
        payload_size += mamba_spec.page_size_bytes
    page_size = payload_size * blocks_per_chunk
    region = SharedOffloadRegion.__new__(SharedOffloadRegion)
    region.rank = rank
    region.num_chunks = num_chunks
    region._row_stride = 2 * page_size + 7
    region._worker_offset = rank * page_size
    region._worker_area_end = (rank + 1) * page_size
    region._base = torch.full((num_chunks * region._row_stride,), -1, dtype=torch.int8)
    region._views = []
    monkeypatch.setattr(region, "cleanup", lambda: None)
    monkeypatch.setattr(cpu_npu_mod, "is_pin_memory_available", lambda: False)
    handlers = []

    class FakeHandler:
        def __init__(self, **kwargs):
            self.cpu_tensors = kwargs["cpu_tensors"]
            handlers.append(self)

        def shutdown(self):
            pass

    monkeypatch.setattr(cpu_npu_mod, "SingleDirectionNPUOffloadingHandler", FakeHandler)
    worker = cpu_npu_mod.NPUOffloadingWorker(canonical, blocks_per_chunk, num_chunks, region)
    assert region._worker_offset == region._worker_area_end
    expected_attention_sizes = [2, 3, 1, 6] if overlaid else [6, 6]
    assert [[ref.page_size_bytes for ref in refs] for refs in canonical.group_data_refs] == [
        expected_attention_sizes,
        [2, 3],
    ]
    for npu_cache, cpu_tensor in zip(canonical.tensors, handlers[0].cpu_tensors):
        original = npu_cache.tensor[2].clone()
        size = npu_cache.page_size_bytes
        cpu_tensor[1, :size].copy_(original)
        npu_cache.tensor[2].zero_()
        npu_cache.tensor[2].copy_(cpu_tensor[1, :size])
        torch.testing.assert_close(npu_cache.tensor[2], original)
    for refs, layer_name in zip(canonical.group_data_refs, ("attention", "mamba")):
        payload = torch.cat([canonical.tensors[ref.tensor_idx].tensor[:, : ref.page_size_bytes] for ref in refs], dim=1)
        expected = torch.cat(caches[layer_name], dim=1)
        torch.testing.assert_close(payload, expected)
    # Other workers' slots and row padding must remain untouched.
    rows = region._base.view(num_chunks, -1)
    assert torch.all(rows[:, : rank * page_size] == -1)
    assert torch.all(rows[:, (rank + 1) * page_size :] == -1)
    worker.shutdown()


def test_excluded_split_cache_does_not_trigger_layout_adaptation(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=3, dtype=torch.int8)
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["attention"], kv_cache_spec=spec),
            KVCacheGroupSpec(layer_names=["local"], kv_cache_spec=spec, enable_kv_transfer=False),
        ],
    )
    worker, _ = _make_layout_worker(config, monkeypatch)
    calls = []
    monkeypatch.setattr(OffloadingConnectorWorker, "register_kv_caches", lambda self, caches: calls.append(caches))
    caches = {"attention": torch.empty((4, 12), dtype=torch.int8), "local": (torch.empty((4, 6)),) * 2}

    worker.register_kv_caches(caches)

    assert calls == [caches]


def test_upstream_compatible_layout_uses_upstream_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_name = "model.layers.0.self_attn"
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.bfloat16,
    )
    worker, _ = _make_layout_worker(
        KVCacheConfig(
            num_blocks=4,
            kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)],
            kv_cache_tensors=[],
            # The connector offloads the groups vLLM selects, not all of them.
            prefix_cacheable_group_ids=(0,),
            hisparse_host_num_blocks=None,
        ),
        monkeypatch,
    )
    calls = []

    def capture_upstream(self, kv_caches):
        calls.append(kv_caches)

    monkeypatch.setattr(
        OffloadingConnectorWorker,
        "register_kv_caches",
        capture_upstream,
    )
    caches = {layer_name: torch.empty((4, 2, 1, 6), dtype=torch.bfloat16)}

    worker.register_kv_caches(caches)

    assert calls == [caches]


def test_split_attention_cache_is_zero_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_name = "model.layers.0.self_attn"
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.bfloat16,
    )
    worker, captured = _make_layout_worker(
        KVCacheConfig(
            num_blocks=4,
            kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)],
            kv_cache_tensors=[],
            # The connector offloads the groups vLLM selects, not all of them.
            prefix_cacheable_group_ids=(0,),
            hisparse_host_num_blocks=None,
        ),
        monkeypatch,
    )
    key = torch.empty((4, 2, 1, 3), dtype=torch.bfloat16)
    value = torch.empty((4, 2, 1, 3), dtype=torch.bfloat16)

    worker.register_kv_caches({layer_name: (key, value)})

    canonical = captured[0]
    assert [tensor.tensor.shape for tensor in canonical.tensors] == [
        (4, 12),
        (4, 12),
    ]
    assert [ref.page_size_bytes for ref in canonical.group_data_refs[0]] == [
        12,
        12,
    ]
    assert all(ref.mapping is None for ref in canonical.group_data_refs[0])
    assert canonical.tensors[0].tensor.data_ptr() == key.data_ptr()
    assert canonical.tensors[1].tensor.data_ptr() == value.data_ptr()
