# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Owner-slab chunk arithmetic and DSV4.1 allocation wiring."""

from types import SimpleNamespace

import pytest
import torch
from vllm.config import set_current_vllm_config

from vllm_ascend.attention.context_parallel import pcp_owner_vmm as owner_vmm

GRANULARITY = owner_vmm.CANN_VMM_GRANULARITY_BYTES


@pytest.mark.parametrize("pcp_size", [2, 4])
@pytest.mark.parametrize("size_bytes", [1, GRANULARITY, GRANULARITY + 1, 5 * GRANULARITY - 1])
def test_slab_chunk_plan_covers_every_chunk_once(pcp_size: int, size_bytes: int) -> None:
    owners = []
    for rank in range(pcp_size):
        plan = owner_vmm.SlabChunkPlan(size_bytes=size_bytes, pcp_size=pcp_size, rank=rank)
        assert plan.chunk_count == (size_bytes + GRANULARITY - 1) // GRANULARITY
        assert plan.virtual_bytes == plan.chunk_count * GRANULARITY
        assert plan.virtual_bytes >= size_bytes
        assert all(plan.owner_of(chunk) == rank for chunk in plan.own_chunks)
        assert plan.own_bytes == len(plan.own_chunks) * GRANULARITY
        owners.extend(plan.own_chunks)
    assert sorted(owners) == list(range((size_bytes + GRANULARITY - 1) // GRANULARITY))


@pytest.mark.parametrize("placement", [owner_vmm.SLAB_PLACEMENT_CONTIGUOUS, owner_vmm.SLAB_PLACEMENT_MODULO])
@pytest.mark.parametrize("pcp_size", [2, 4])
def test_slab_chunk_plan_owner_offsets_are_compact(pcp_size: int, placement: str) -> None:
    size_bytes = 8 * GRANULARITY
    plan = owner_vmm.SlabChunkPlan(size_bytes=size_bytes, pcp_size=pcp_size, rank=0, placement=placement)
    for owner in range(pcp_size):
        owner_plan = owner_vmm.SlabChunkPlan(size_bytes=size_bytes, pcp_size=pcp_size, rank=owner, placement=placement)
        offsets = [owner_plan.offset_in_owner(chunk) for chunk in owner_plan.own_chunks]
        assert offsets == [index * GRANULARITY for index in range(len(offsets))]
        assert offsets == sorted(offsets)
    owner = plan.owner_of(pcp_size)
    if placement == owner_vmm.SLAB_PLACEMENT_MODULO:
        expected = ((pcp_size - owner) // pcp_size) * GRANULARITY
    else:
        expected = (pcp_size - owner * plan.chunks_per_rank) * GRANULARITY
    assert plan.offset_in_owner(pcp_size) == expected


def test_contiguous_slab_gives_each_owner_one_chunk_run() -> None:
    plan = owner_vmm.SlabChunkPlan(size_bytes=8 * GRANULARITY, pcp_size=4, rank=1)
    assert plan.own_chunks == (2, 3)
    assert plan.chunks_per_rank == 2
    assert plan.first_chunk == 2
    assert plan.owner_of(0) == 0 and plan.owner_of(1) == 0
    assert plan.owner_of(2) == 1 and plan.owner_of(3) == 1
    assert plan.owner_of(7) == 3


def test_slab_chunk_plan_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        owner_vmm.SlabChunkPlan(size_bytes=0, pcp_size=4, rank=0)
    with pytest.raises(ValueError, match="outside PCP size"):
        owner_vmm.SlabChunkPlan(size_bytes=GRANULARITY, pcp_size=4, rank=4)


def test_slab_plan_own_chunks_are_empty_when_pcp_exceeds_chunks() -> None:
    plan = owner_vmm.SlabChunkPlan(size_bytes=1, pcp_size=4, rank=3)
    assert plan.chunk_count == 1
    assert plan.own_chunks == ()
    assert plan.own_bytes == 0


def _fake_v41_config(num_blocks: int = 2):
    from tests.deepseek_v41_utils import make_cache_config

    return make_cache_config(num_blocks)


def _layer_specs(config):
    return {name: spec for group in config.kv_cache_groups for name, spec in group.kv_cache_spec.kv_cache_specs.items()}


@pytest.fixture
def _packed_pages(monkeypatch):
    """Opt the synthetic oracle config out of the page-contiguity guard.

    ``make_cache_config`` packs long KV and indexer planes into one slot but
    leaves them non-turboquant, while the production guard only expects that
    packing when the cache is turboquant. These allocation tests exercise slab
    wiring, not the contiguity check.
    """
    from vllm_ascend.worker.v2 import attn_utils

    monkeypatch.setattr(attn_utils, "uses_turboquant_groups", lambda groups: True)


def test_dsv41_owner_allocation_uses_one_slab_for_the_shared_backing(monkeypatch, _packed_pages) -> None:
    from vllm_ascend.worker.v2 import attn_utils

    config = _fake_v41_config()
    specs = _layer_specs(config)
    calls = []

    class FakeAllocator:
        def allocate_slab(self, layer_names, size_bytes):
            tensor = torch.zeros(size_bytes, dtype=torch.int8)
            calls.append((tuple(layer_names), size_bytes, tensor))
            return tensor

    def fail_replicated(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("replicated allocation used although the slab succeeded")

    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", fail_replicated)
    raw = attn_utils._allocate_deepseek_v41_shared_cache(
        config,
        specs,
        2 * 1024 * 1024,
        torch.device("cpu"),
        owner_allocator=FakeAllocator(),
    )
    # Without LOCAL_INDEX every layer tuple is a view into one shared backing.
    assert len(calls) == 1
    names, size, tensor = calls[0]
    assert set(names) == set(specs)
    assert size == config.kv_cache_tensors[0].size
    assert set(raw) == set(specs)
    for value in raw.values():
        assert value.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()


def test_dsv41_local_index_gets_its_own_replicated_backing(monkeypatch, _packed_pages) -> None:
    from vllm_ascend.worker.v2 import attn_utils

    monkeypatch.setenv("VLLM_ASCEND_PCP_LOCAL_INDEX", "1")
    config = _fake_v41_config()
    specs = _layer_specs(config)
    assert len({descriptor.size for descriptor in config.kv_cache_tensors}) == 2

    with set_current_vllm_config(SimpleNamespace(kv_transfer_config=None)):
        raw = attn_utils._allocate_deepseek_v41_shared_cache(
            config,
            specs,
            2 * 1024 * 1024,
            torch.device("cpu"),
            owner_allocator=None,
        )
    index_names = {name for name in raw if ".indexer.k_cache" in name}
    kv_names = set(raw) - index_names
    index_storage = {raw[name].untyped_storage().data_ptr() for name in index_names}
    kv_storage = {raw[name].untyped_storage().data_ptr() for name in kv_names}
    assert len(index_storage) == 1
    assert len(kv_storage) == 1
    assert index_storage.isdisjoint(kv_storage)


def test_dsv41_owner_allocation_falls_back_to_replicated(monkeypatch, _packed_pages) -> None:
    from vllm_ascend.worker.v2 import attn_utils

    config = _fake_v41_config()
    specs = _layer_specs(config)

    class FailingAllocator:
        @staticmethod
        def allocate_slab(layer_names, size_bytes):
            return None

    replicated_sizes = []

    def replicated(size, alignment, device):
        replicated_sizes.append(size)
        return torch.ones(size, dtype=torch.int8)

    monkeypatch.setattr(attn_utils, "_allocate_int8_cache_tensor", replicated)
    raw = attn_utils._allocate_deepseek_v41_shared_cache(
        config,
        specs,
        2 * 1024 * 1024,
        torch.device("cpu"),
        owner_allocator=FailingAllocator(),
    )
    assert replicated_sizes == [config.kv_cache_tensors[0].size]
    assert set(raw) == set(specs)


def test_capacity_mode_fails_closed_without_a_working_slab(monkeypatch) -> None:
    from vllm_ascend.worker.v2 import attn_utils

    config = _fake_v41_config()
    specs = _layer_specs(config)

    class FailingAllocator:
        @staticmethod
        def allocate_slab(layer_names, size_bytes):
            return None

    with pytest.raises(RuntimeError, match="OWNER_CAPACITY"):
        attn_utils._allocate_deepseek_v41_shared_cache(
            config,
            specs,
            2 * 1024 * 1024,
            torch.device("cpu"),
            owner_allocator=FailingAllocator(),
            require_owner_slab=True,
        )
    with pytest.raises(RuntimeError, match="OWNER_CAPACITY"):
        attn_utils._allocate_deepseek_v41_shared_cache(
            config,
            specs,
            2 * 1024 * 1024,
            torch.device("cpu"),
            owner_allocator=None,
            require_owner_slab=True,
        )


def test_dsv41_owner_allocator_is_disabled_by_default(monkeypatch) -> None:
    from vllm_ascend.worker.v2 import attn_utils

    monkeypatch.delenv("VLLM_ASCEND_PCP_KV_SHARING", raising=False)
    config = SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=4,
            decode_context_parallel_size=1,
            tensor_parallel_size=1,
        ),
        kv_transfer_config=None,
    )
    assert attn_utils._deepseek_v41_owner_allocator(config, {}, torch.device("cpu")) is None


class _FakeGroup:
    def __init__(self, world_size: int, rank: int) -> None:
        self.world_size = world_size
        self.rank_in_group = rank
        self.cpu_group = object()


class _FakeDist:
    """Collective double: every rank reports the local outcome of the caller."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size

    def all_gather_object(self, out, value, group=None) -> None:
        for rank in range(self.world_size):
            entry = dict(value)
            if rank != entry.get("rank", 0):
                entry["rank"] = rank
                entry["device"] = rank
                entry["tgid"] = rank + 1
                if entry.get("handle") is not None:
                    entry["handle"] = bytes(owner_vmm.ACL_V2_SHAREABLE_HANDLE_BYTES)
            out[rank] = entry

    def barrier(self, group=None) -> None:
        return None


class _FakeBackend:
    def __init__(self, *, fail_after_maps: int = 1) -> None:
        self.calls: list[tuple] = []
        self.fail_after_maps = fail_after_maps
        self.maps = 0

    def set_device(self, device_index: int) -> None:
        return None

    def bare_tgid(self, device_index: int) -> int:
        return 1

    def allocate_granularity(self, device_index: int) -> int:
        return owner_vmm.CANN_VMM_GRANULARITY_BYTES

    def enable_peer_access(self, device_index: int, peers) -> None:
        self.calls.append(("enable_peer_access", tuple(peers)))

    def reserve_address(self, *, size_bytes: int, device_index: int) -> int:
        self.calls.append(("reserve", size_bytes))
        return 0x10000000

    def malloc_physical(self, *, size_bytes: int, device_index: int):
        self.calls.append(("malloc", size_bytes))
        return owner_vmm.AclPhysicalHandle(0x2000, size_bytes, device_index)

    def export(self, *, handle, device_index: int):
        self.calls.append(("export", handle.value))
        return owner_vmm.AclShareableHandle(bytes(owner_vmm.ACL_V2_SHAREABLE_HANDLE_BYTES))

    def authorize(self, *, shareable_handle, bare_tgids, device_index: int) -> None:
        self.calls.append(("authorize", tuple(bare_tgids)))

    def import_handle(self, *, shareable_handle, device_index: int):
        self.calls.append(("import", device_index))
        return owner_vmm.AclPhysicalHandle(0x3000, 0, device_index)

    def map(self, *, base_address: int, size_bytes: int, offset_bytes: int, handle, device_index: int) -> None:
        self.calls.append(("map", base_address, size_bytes))
        self.maps += 1
        if self.maps > self.fail_after_maps:
            raise owner_vmm.PcpOwnerVmmError("aclrtMapMem", -1)

    def unmap(self, *, base_address: int, device_index: int) -> None:
        self.calls.append(("unmap", base_address))

    def free_physical(self, *, handle, device_index: int) -> None:
        self.calls.append(("free_physical", handle.value))

    def release_address(self, *, base_address: int, device_index: int) -> None:
        self.calls.append(("release", base_address))


def _fake_runtime(backend: _FakeBackend) -> owner_vmm.PcpOwnerSlabRuntime:
    # ``torch.device("npu:0")`` cannot be built without torch_npu; the
    # constructor only reads ``type`` and ``index``.
    return owner_vmm.PcpOwnerSlabRuntime(
        device=SimpleNamespace(type="npu", index=0),
        pcp_group=_FakeGroup(2, 0),
        backend=backend,
        dist_module=_FakeDist(2),
    )


def test_partial_map_failure_rolls_back_before_physical_release() -> None:
    granularity = owner_vmm.CANN_VMM_GRANULARITY_BYTES
    # Two chunks, one per owner: the second aclrtMapMem call fails.
    backend = _FakeBackend(fail_after_maps=1)
    runtime = _fake_runtime(backend)

    assert runtime.allocate_slab(["layers.0.long_kv_cache"], 2 * granularity) is None
    assert runtime.active is False

    calls = backend.calls
    unmaps = [index for index, call in enumerate(calls) if call[0] == "unmap"]
    frees = [index for index, call in enumerate(calls) if call[0] == "free_physical"]
    assert unmaps, "the successfully mapped owner range must be unmapped during rollback"
    assert calls[unmaps[0]] == ("unmap", 0x10000000)
    assert frees and unmaps[0] < frees[0], "peer mappings must be dropped before the owner frees HBM"
    assert any(call[0] == "release" for call in calls)


def test_close_unregisters_its_slabs() -> None:
    runtime = _fake_runtime(_FakeBackend())
    owner_vmm._register_slab(0xABC000, 0x1000, runtime)
    try:
        runtime.close()
        assert all(entry[2] is not runtime for entry in owner_vmm._slab_registry)
    finally:
        owner_vmm._slab_registry[:] = [entry for entry in owner_vmm._slab_registry if entry[2] is not runtime]
