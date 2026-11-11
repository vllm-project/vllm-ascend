# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Native KV offloading must register exactly the scheduler's group set.

Hybrid models (e.g. DeepSeek-V4.1 Engram / sparse attention) expose KV cache
groups that are not eligible for offloading. The connector has to select the
same subset as vLLM's scheduler, worker and offloading config, otherwise
``GPULoadStoreSpec.group_sizes`` and the registered per-group data refs
diverge and ``NPUOffloadingWorker.transfer_async`` trips its length assertion
with misaligned block ids.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    get_offloading_group_ids,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
)
from vllm.v1.kv_offload.base import CanonicalKVCaches

import vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector as connector_mod
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector import (
    AscendOffloadingConnectorWorker,
)

NUM_BLOCKS = 4
BLOCK_SIZE = 2
HEAD_SIZE = 3
DTYPE = torch.bfloat16


def _attention_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=HEAD_SIZE,
        dtype=DTYPE,
    )


def _group(
    layer_name: str,
    *,
    enable_kv_transfer: bool = True,
    spec: object | None = None,
    role: KVCacheGroupRole = KVCacheGroupRole.DEFAULT,
) -> KVCacheGroupSpec:
    return KVCacheGroupSpec(
        layer_names=[layer_name],
        kv_cache_spec=_attention_spec() if spec is None else spec,
        enable_kv_transfer=enable_kv_transfer,
        role=role,
    )


def _config(
    groups: list[KVCacheGroupSpec],
    *,
    hisparse_host_num_blocks: int | None = None,
) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        hisparse_host_num_blocks=hisparse_host_num_blocks,
    )


def _split_attention_cache() -> tuple[torch.Tensor, torch.Tensor]:
    """Split K/V caches so the Ascend layout adapter is exercised."""
    shape = (NUM_BLOCKS, BLOCK_SIZE, 1, HEAD_SIZE)
    return (
        torch.empty(shape, dtype=DTYPE),
        torch.empty(shape, dtype=DTYPE),
    )


def _prepare_worker(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_config: KVCacheConfig,
) -> tuple[AscendOffloadingConnectorWorker, list[CanonicalKVCaches]]:
    worker = AscendOffloadingConnectorWorker.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = kv_cache_config
    worker.vllm_config = object()
    captured: list[CanonicalKVCaches] = []
    worker._init_worker = captured.append
    monkeypatch.setattr(connector_mod, "derive_canonical_mappings", lambda *args: {})
    return worker, captured


def test_only_prefix_cacheable_groups_are_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hybrid model with one cacheable group out of three.

    This mirrors DeepSeek-V4.1-Flash-w4a8 on TP8, which reports 12 KV cache
    groups but a single prefix-cacheable one. Registering all groups made
    ``len(group_sizes) != len(kv_cache_groups_data_refs)``.
    """
    groups = [
        _group("model.layers.0.self_attn"),
        _group("model.layers.0.engram", enable_kv_transfer=False),
        _group("model.layers.1.sparse_attn", enable_kv_transfer=False),
    ]
    kv_cache_config = _config(groups)

    assert get_offloading_group_ids(kv_cache_config) == (0,)

    worker, captured = _prepare_worker(monkeypatch, kv_cache_config)
    # Only the cacheable layer is present in the runner's kv_caches; touching
    # the other groups would raise KeyError.
    worker.register_kv_caches(
        {"model.layers.0.self_attn": _split_attention_cache()},
    )

    assert len(captured) == 1
    canonical = captured[0]
    group_ids = get_offloading_group_ids(kv_cache_config)
    assert len(canonical.group_data_refs) == len(group_ids) < len(kv_cache_config.kv_cache_groups)
    assert [ref.page_size_bytes for ref in canonical.group_data_refs[0]] == [12, 12]
    assert [tensor.tensor.shape for tensor in canonical.tensors] == [(NUM_BLOCKS, 12)] * 2


def test_registered_groups_follow_group_ids_not_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selected groups are picked by id, so a gap must not shift the refs."""
    layer_names = ["model.layers.0.self_attn", "model.layers.1.sparse_attn", "model.layers.2.self_attn"]
    groups = [
        _group(layer_names[0]),
        _group(layer_names[1], enable_kv_transfer=False),
        _group(layer_names[2]),
    ]
    kv_cache_config = _config(groups)
    assert get_offloading_group_ids(kv_cache_config) == (0, 2)

    worker, captured = _prepare_worker(monkeypatch, kv_cache_config)
    caches = {
        layer_names[0]: _split_attention_cache(),
        layer_names[2]: _split_attention_cache(),
    }
    worker.register_kv_caches(caches)

    canonical = captured[0]
    assert len(canonical.group_data_refs) == 2
    for refs, layer_name in zip(canonical.group_data_refs, (layer_names[0], layer_names[2])):
        assert [ref.page_size_bytes for ref in refs] == [12, 12]
        first_ref_tensor = canonical.tensors[refs[0].tensor_idx].tensor
        assert first_ref_tensor.data_ptr() == caches[layer_name][0].data_ptr()


def test_non_cacheable_group_spec_is_not_introspected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Excluded groups may carry spec types the connector cannot canonicalize."""
    groups = [
        _group("model.layers.0.self_attn"),
        _group(
            "model.layers.0.experimental_state",
            enable_kv_transfer=False,
            spec=SimpleNamespace(),
        ),
    ]
    kv_cache_config = _config(groups)

    worker, captured = _prepare_worker(monkeypatch, kv_cache_config)
    worker.register_kv_caches(
        {"model.layers.0.self_attn": _split_attention_cache()},
    )

    assert len(captured) == 1
    assert len(captured[0].group_data_refs) == len(get_offloading_group_ids(kv_cache_config)) == 1


def test_all_groups_registered_when_all_are_cacheable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense models keep registering every group, i.e. no over-filtering."""
    layer_names = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    kv_cache_config = _config([_group(name) for name in layer_names])
    assert get_offloading_group_ids(kv_cache_config) == (0, 1)

    worker, captured = _prepare_worker(monkeypatch, kv_cache_config)
    worker.register_kv_caches({name: _split_attention_cache() for name in layer_names})

    canonical = captured[0]
    assert len(canonical.group_data_refs) == 2
    assert len(canonical.tensors) == 4
    for refs in canonical.group_data_refs:
        assert [ref.page_size_bytes for ref in refs] == [12, 12]


def test_hisparse_host_pool_registers_only_indexer_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a HiSparse host pool the offload set is the indexer groups."""
    groups = [
        _group("model.layers.0.self_attn"),
        _group(
            "model.layers.0.hisparse_source",
            enable_kv_transfer=False,
            role=KVCacheGroupRole.HISPARSE_SOURCE,
        ),
        _group(
            "model.layers.0.hisparse_indexer",
            enable_kv_transfer=False,
            role=KVCacheGroupRole.HISPARSE_INDEXER,
        ),
    ]
    kv_cache_config = _config(groups, hisparse_host_num_blocks=8)
    assert get_offloading_group_ids(kv_cache_config) == (2,)

    worker, captured = _prepare_worker(monkeypatch, kv_cache_config)
    worker.register_kv_caches(
        {"model.layers.0.hisparse_indexer": _split_attention_cache()},
    )

    canonical = captured[0]
    assert len(canonical.group_data_refs) == 1
    assert [ref.page_size_bytes for ref in canonical.group_data_refs[0]] == [12, 12]
