# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
# ruff: noqa: E402

from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm.v1.core.kv_cache_config_builder", reason="the KV cache builder is main-only")
pytest.importorskip("vllm.v1.hisparse.coordinator")

import torch
from vllm.config.attention import HiSparseConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import HiSparseResidentManager, register_all_kvcache_specs
from vllm.v1.hisparse.coordinator import get_hisparse_coordinator
from vllm.v1.kv_cache_interface import (
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheTensor,
    MLAAttentionSpec,
    SparseCacheRole,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hisparse.layout import aligned_hisparse_allocation_config
from vllm_ascend.hisparse.binding import prepare_hisparse_cache_views

BLOCK_SIZE = 16


def _transfer_config(pool_connector):
    source = {
        "kv_connector": "HiSparseConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"host_pool_gib": 1},
    }
    if pool_connector is None:
        return KVTransferConfig(**source)
    pool = {"kv_connector": pool_connector, "kv_role": "kv_both"}
    if pool_connector == "AscendStoreConnector":
        pool["kv_connector_extra_config"] = {"backend": "memcache", "load_async": True}
    return KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"connectors": [source, pool]},
    )


def _cache_configs(
    *,
    pool_connector="AscendStoreConnector",
    host_blocks=None,
    device_blocks=14,
    max_model_blocks=8,
    in_flight_blocks=2,
    block_size=BLOCK_SIZE,
    hot_units=1,
):
    # Register Core types before the combined builder adds its import-only spec.
    register_all_kvcache_specs(None)
    hf_text_config = SimpleNamespace(model_type="glm_moe_dsa", index_topk=16)
    config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=HiSparseConfig(device_buffer_size=2 * block_size)),
        model_config=SimpleNamespace(
            hf_config=hf_text_config,
            hf_text_config=hf_text_config,
            max_model_len=max_model_blocks * block_size,
            supports_kv_cache_grouping=False,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, world_size=1),
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=device_blocks,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        kv_transfer_config=_transfer_config(pool_connector),
        max_in_flight_tokens=in_flight_blocks * block_size,
    )
    specs = {
        f"source{index}": MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            is_index_group_leader=True,
        )
        for index in range(hot_units)
    }
    specs["indexer"] = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        cache_role=SparseCacheRole.INDEXER,
    )
    builder = AscendKVCacheConfigBuilder()
    groups = builder.get_kv_cache_groups(config, specs)
    cache = builder.get_kv_cache_config_from_groups(config, groups, num_blocks=device_blocks)
    if host_blocks is not None:
        # Model the number of usable Host pages left after other requests.
        cache.hisparse_host_num_blocks = host_blocks
    return config, cache


def _cache_manager(config, cache):
    block_size = cache.kv_cache_groups[0].kv_cache_spec.block_size
    return KVCacheManager(
        generate_scheduler_kv_cache_config([cache]),
        max_model_len=config.model_config.max_model_len,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        max_in_flight_tokens=config.max_in_flight_tokens,
        enable_caching=False,
    )


@pytest.mark.parametrize(("window", "sizing_pages", "hot_units"), [(0, 1, 1), (2, 3, 2), (8, 8, 1), (12, 8, 1)])
def test_registry_shares_the_spec_window_and_sizes_for_the_import_tail(window, sizing_pages, hot_units):
    config, cache = _cache_configs(in_flight_blocks=window, hot_units=hot_units)
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hisparse.admission import (
        AscendStoreHiSparseResidentManager,
    )

    managers = get_hisparse_coordinator(_cache_manager(config, cache)).resident_managers
    assert len(managers) == hot_units
    for manager in managers:
        assert type(manager) is AscendStoreHiSparseResidentManager
        spec = manager.kv_cache_spec
        assert manager.max_admission_blocks_per_request == spec.max_admission_blocks_per_request(
            config.max_in_flight_tokens, config.model_config.max_model_len
        )
        assert manager.max_admission_blocks_per_request == min(window, 8)
        assert spec.max_memory_usage_bytes(config) == sizing_pages * spec.page_size_bytes
    assert not any(
        group.enable_kv_transfer
        for group in cache.kv_cache_groups
        if isinstance(group.kv_cache_spec, HiSparseResidentSpec)
    )


@pytest.mark.parametrize(("window", "expected"), [(0, 0), (2, 2)])
def test_resident_admission_counts_existing_pages_before_partial_import(window, expected):
    block_size = 128
    config, cache = _cache_configs(block_size=block_size, in_flight_blocks=window)
    (manager,) = get_hisparse_coordinator(_cache_manager(config, cache)).resident_managers
    request_id = "running-request"
    manager.allocate_new_blocks(request_id, 4 * block_size, 4 * block_size)

    # The import ends inside four already resident pages; its tail needs no allocation.
    assert (
        manager.get_num_blocks_to_allocate(
            request_id,
            8 * block_size,
            (),
            2 * block_size,
            0,
            8 * block_size,
            apply_admission_cap=True,
        )
        == expected
    )


@pytest.mark.parametrize("device_sizes", [(), (128, 256)], ids=["no-device-tensors", "unequal-device-sizes"])
def test_slab_reservation_requires_one_shared_device_size(device_sizes):
    host_tensor = KVCacheTensor(64, ["host"], 0, 64, host_resident=True)
    tensors = [host_tensor] + [
        KVCacheTensor(size, [f"cache{index}"], 0, 128) for index, size in enumerate(device_sizes)
    ]
    cache = KVCacheConfig(1, tensors, [])
    with pytest.raises(ValueError, match="requires one shared device slab size"):
        aligned_hisparse_allocation_config(cache)
    assert cache.kv_cache_tensors is tensors


def test_slab_reservation_preserves_runner_config_on_failed_binding():
    _, cache = _cache_configs()
    tensors = cache.kv_cache_tensors
    device_tensor = next(tensor for tensor in tensors if not tensor.host_resident)
    # The Core runner deep-copies its config and binds the views on that copy.
    core_cache = deepcopy(aligned_hisparse_allocation_config(cache))
    with pytest.raises(ValueError, match="missing its reserved alignment padding"):
        assert core_cache.kv_cache_tensors[1].size > device_tensor.size
        name = device_tensor.layers[0]
        prepare_hisparse_cache_views(core_cache, {name: torch.empty(device_tensor.size, dtype=torch.int8)}, {})
    assert cache.kv_cache_tensors is tensors
    with pytest.raises(RuntimeError, match="slab alignment reservation"):
        prepare_hisparse_cache_views(cache, {}, {})


@pytest.mark.parametrize(
    ("tokens", "computed", "local", "admission", "tail_resident", "expected"),
    [
        (129, 32, 0, True, False, 3),  # Two window pages and a writable import tail.
        (129, 32, 0, False, False, 8),  # The per-step allocation remains uncapped.
        (129, 32, 0, True, True, 2),
        (129, 31, 16, True, False, 3),  # Unaligned external hit following a local hit.
        (33, 32, 0, True, False, 2),  # Uncached suffix already fits.
        (32, 32, 0, True, False, 1),  # Full external hit.
        (128, 32, 32, True, False, 0),  # Local-only hit.
        (128, 0, 0, True, False, 2),  # Cold request.
    ],
)
def test_registered_resident_admission(tokens, computed, local, admission, tail_resident, expected):
    config, cache = _cache_configs()
    (manager,) = get_hisparse_coordinator(_cache_manager(config, cache)).resident_managers
    request_id = "request"
    if tail_resident:
        manager.allocate_external_computed_blocks(request_id, local, computed - local)
    assert (
        manager.get_num_blocks_to_allocate(
            request_id,
            tokens,
            (),
            computed,
            local,
            tokens,
            apply_admission_cap=admission,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("pool_connector", "host_blocks", "admitted"),
    [
        pytest.param(None, None, False, id="core-baseline"),
        pytest.param("MooncakeConnector", None, False, id="other-connector"),
        pytest.param("AscendStoreConnector", 8, False, id="host-capacity-limit"),
        pytest.param("AscendStoreConnector", None, True, id="registered-partial-hit"),
    ],
)
def test_scheduler_cache_manager_admits_partial_import(pool_connector, host_blocks, admitted):
    config, cache = _cache_configs(pool_connector=pool_connector, host_blocks=host_blocks)
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.hisparse.admission import (
        AscendStoreHiSparseResidentManager,
    )

    manager = _cache_manager(config, cache)
    (resident_manager,) = get_hisparse_coordinator(manager).resident_managers
    expected_manager = (
        AscendStoreHiSparseResidentManager if pool_connector == "AscendStoreConnector" else HiSparseResidentManager
    )
    assert type(resident_manager) is expected_manager
    request = Request(
        request_id="partial-hit",
        prompt_token_ids=list(range(8 * BLOCK_SIZE)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    result = manager.allocate_slots(
        request,
        num_new_tokens=0,
        num_external_computed_tokens=2 * BLOCK_SIZE,
        delay_cache_blocks=True,
        full_sequence_must_fit=True,
    )
    if admitted:
        assert result is not None
        source, indexer, resident, hot = manager.get_blocks(request.request_id).blocks
        assert len(source) == len(indexer) == len(resident) == len(hot) == 2
        assert resident[0].is_null and not resident[1].is_null
    else:
        assert result is None


def test_failed_resident_import_releases_reservation_and_can_retry_with_another_request():
    block = 128
    config, cache = _cache_configs(block_size=block, host_blocks=24, device_blocks=32)
    manager = _cache_manager(config, cache)
    coordinator = get_hisparse_coordinator(manager)
    (resident,) = coordinator.resident_managers
    host = coordinator.host_manager
    assert host is not None
    device_free = manager.block_pool.get_num_free_blocks()
    host_free = host.block_pool.get_num_free_blocks()
    requests = [Request(req_id, list(range(4 * block)), SamplingParams(max_tokens=1), None) for req_id in ("a", "b")]

    for request in requests:
        assert (
            manager.allocate_slots(
                request,
                num_new_tokens=0,
                num_external_computed_tokens=2 * block,
                delay_cache_blocks=True,
                full_sequence_must_fit=True,
            )
            is not None
        )
    assert set(coordinator._pending_imports) == {request.request_id for request in requests}
    # Older Core releases keep the import ledger without a reservation counter.
    reserved_pages = getattr(coordinator, "reserved_import_pages", None)
    if reserved_pages is not None:
        assert reserved_pages > 0
    first, second = requests
    coordinator.finish_host_import(first.request_id, failed=True)
    manager.free(first)
    assert first.request_id not in host.req_to_blocks
    assert first.request_id not in resident.req_to_blocks
    assert set(coordinator._pending_imports) == {second.request_id}
    reserved_pages = getattr(coordinator, "reserved_import_pages", None)
    if reserved_pages is not None:
        assert reserved_pages > 0
    assert second.request_id in host.req_to_blocks

    for computed in (0, block):
        first.num_computed_tokens = computed
        assert manager.allocate_slots(first, num_new_tokens=block, full_sequence_must_fit=True) is not None
        assert sum(not page.is_null for page in resident.get_residency_row(first.request_id)) <= (
            resident.max_admission_blocks_per_request
        )

    coordinator.finish_host_import(second.request_id, failed=True)
    for request in requests:
        manager.free(request)
        assert request.request_id not in coordinator.request_states
        assert request.request_id not in host.req_to_blocks
        assert request.request_id not in resident.req_to_blocks
    assert not coordinator._pending_imports
    reserved_pages = getattr(coordinator, "reserved_import_pages", None)
    if reserved_pages is not None:
        assert reserved_pages == 0
    assert host.block_pool.get_num_free_blocks() == host_free
    assert manager.block_pool.get_num_free_blocks() == device_free


def test_hisparse_slab_padding_is_idempotent():
    _, cache = _cache_configs()
    aligned = aligned_hisparse_allocation_config(cache)
    repeated = aligned_hisparse_allocation_config(aligned)
    assert [t.size for t in repeated.kv_cache_tensors] == [t.size for t in aligned.kv_cache_tensors]
    assert [t.size for t in cache.kv_cache_tensors] != [t.size for t in aligned.kv_cache_tensors]
