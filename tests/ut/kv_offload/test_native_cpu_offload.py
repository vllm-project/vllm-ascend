# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.utils.math_utils import round_up
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheConfig,
    KVCacheGroupRole,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingWorker
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

import vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector as connector_mod
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.cpu_npu import (
    NPUOffloadingWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu import NPUOffloadingSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector import (
    AscendOffloadingConnector,
    AscendOffloadingConnectorWorker,
    _canonicalize_split_cache,
)


def _make_config(extra_config: dict[str, object]) -> OffloadingConfig:
    return OffloadingConfig(
        groups=(
            OffloadingGroupConfig(
                tokens_per_block=16,
                layer_names=("model.layers.0.self_attn",),
                # vLLM main added the originating KV cache group index.
                group_id=0,
            ),
        ),
        worker_kv_bytes_per_block=64,
        enable_kv_cache_events=False,
        extra_config=extra_config,
        engine_id="test-engine",
        model=OffloadingModelConfig(
            name="test-model",
            dtype="bfloat16",
        ),
        cache=OffloadingCacheConfig(
            tokens_per_hash=16,
            blocks_per_chunk=2,
        ),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=2,
            tp_size=2,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=True,
            data_parallel_size=1,
            data_parallel_rank_local=None,
        ),
    )


def test_npu_offloading_spec_uses_upstream_cpu_manager() -> None:
    bytes_per_chunk = 64 * 2 * 2
    aligned_bytes_per_chunk = round_up(
        bytes_per_chunk,
        NPUOffloadingSpec.BLOCK_SIZE_ALIGNMENT,
    )
    spec = NPUOffloadingSpec(_make_config({"cpu_bytes_to_use": 10 * aligned_bytes_per_chunk}))

    assert spec.num_chunks == 10
    assert isinstance(spec.get_manager(), CPUOffloadingManager)


def test_npu_offloading_spec_loads_through_vllm_factory() -> None:
    spec_cls = OffloadingSpecFactory.get_spec_cls(
        {
            "spec_name": "NPUOffloadingSpec",
            "spec_module_path": "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu",
        }
    )

    assert spec_cls is NPUOffloadingSpec


def test_npu_worker_reuses_upstream_worker_protocol() -> None:
    assert issubclass(NPUOffloadingWorker, OffloadingWorker)


def test_npu_spec_caches_worker_without_upstream_platform_gate(monkeypatch) -> None:
    spec = NPUOffloadingSpec(_make_config({"cpu_bytes_to_use": 1024}))
    worker = object()
    create_calls = 0

    def create_worker(kv_caches):
        nonlocal create_calls
        create_calls += 1
        return worker

    monkeypatch.setattr(spec, "create_worker", create_worker)
    kv_caches = object()

    assert spec.get_worker(kv_caches) is worker
    assert spec.get_worker(kv_caches) is worker
    assert create_calls == 1


def test_ascend_connector_replaces_worker_with_current_vllm_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_config = object()
    kv_cache_config = object()
    spec = SimpleNamespace(
        replicated_layout=False,
        config=SimpleNamespace(parallel=SimpleNamespace(rank=0)),
    )

    def fake_upstream_init(
        self,
        init_vllm_config,
        role,
        init_kv_cache_config,
    ) -> None:
        assert init_vllm_config is vllm_config
        assert role == KVConnectorRole.WORKER
        assert init_kv_cache_config is kv_cache_config
        self.connector_worker = SimpleNamespace(spec=spec)

    monkeypatch.setattr(OffloadingConnector, "__init__", fake_upstream_init)

    connector = AscendOffloadingConnector(
        vllm_config,
        KVConnectorRole.WORKER,
        kv_cache_config,
    )

    assert isinstance(connector.connector_worker, AscendOffloadingConnectorWorker)
    assert connector.connector_worker.spec is spec
    assert connector.connector_worker.vllm_config is vllm_config
    assert connector.connector_worker.kv_cache_config is kv_cache_config


def test_split_kv_cache_is_canonicalized_without_copy() -> None:
    key = torch.empty((4, 2, 3), dtype=torch.bfloat16)
    value = torch.empty((4, 2, 3), dtype=torch.bfloat16)

    views = _canonicalize_split_cache(
        (key, value),
        num_blocks=4,
        unpadded_page_size_bytes=24,
    )

    assert len(views) == 2
    assert [view.shape for view, _ in views] == [(4, 12), (4, 12)]
    assert [copy_size for _, copy_size in views] == [12, 12]
    assert views[0][0].data_ptr() == key.data_ptr()
    assert views[1][0].data_ptr() == value.data_ptr()


def test_split_kv_cache_coalesces_kernel_blocks() -> None:
    key = torch.empty((8, 2), dtype=torch.int8)
    value = torch.empty((8, 2), dtype=torch.int8)

    views = _canonicalize_split_cache(
        (key, value),
        num_blocks=4,
        unpadded_page_size_bytes=8,
    )

    assert [view.shape for view, _ in views] == [(4, 4), (4, 4)]
    assert [copy_size for _, copy_size in views] == [4, 4]


def test_extra_physical_blocks_do_not_hide_separate_value_cache() -> None:
    key = torch.empty((10, 2), dtype=torch.int8)
    value = torch.empty((10, 2), dtype=torch.int8)

    views = _canonicalize_split_cache(
        (key, value),
        num_blocks=4,
        unpadded_page_size_bytes=4,
    )

    assert len(views) == 2
    assert [view.shape for view, _ in views] == [(4, 2), (4, 2)]
    assert views[0][0].data_ptr() == key.data_ptr()
    assert views[1][0].data_ptr() == value.data_ptr()


def test_split_kv_cache_prefers_complete_overlapping_view() -> None:
    full = torch.empty((4, 8), dtype=torch.int8)
    key = full[:, :4]
    scale = full[:, 4:]

    views = _canonicalize_split_cache(
        (key, scale, full),
        num_blocks=4,
        unpadded_page_size_bytes=8,
    )

    assert len(views) == 1
    assert views[0][0].data_ptr() == full.data_ptr()
    assert views[0][0].shape == (4, 8)
    assert views[0][1] == 8


def test_split_kv_cache_rejects_noncontiguous_block_payload() -> None:
    key = torch.empty((4, 2, 3), dtype=torch.int8).transpose(1, 2)
    value = torch.empty((4, 3, 2), dtype=torch.int8)

    with pytest.raises(ValueError, match="block payload is non-contiguous"):
        _canonicalize_split_cache(
            (key, value),
            num_blocks=4,
            unpadded_page_size_bytes=12,
        )


def test_ascend_connector_worker_accepts_separate_kv_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_name = "model.layers.0.self_attn"
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.bfloat16,
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=4,
        hisparse_host_num_blocks=None,
        prefix_cacheable_group_ids=(0,),
        kv_cache_groups=[SimpleNamespace(layer_names=[layer_name], kv_cache_spec=spec, host_resident=False)],
        kv_cache_tensors=[],
    )
    worker = AscendOffloadingConnectorWorker.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = kv_cache_config
    worker.vllm_config = object()
    captured: list[CanonicalKVCaches] = []
    worker._init_worker = captured.append
    monkeypatch.setattr(connector_mod, "derive_canonical_mappings", lambda *args: {})

    worker.register_kv_caches(
        {
            layer_name: (
                torch.empty((4, 2, 1, 3), dtype=torch.bfloat16),
                torch.empty((4, 2, 1, 3), dtype=torch.bfloat16),
            )
        }
    )

    assert len(captured) == 1
    canonical = captured[0]
    assert len(canonical.tensors) == 2
    assert [tensor.tensor.shape for tensor in canonical.tensors] == [
        (4, 12),
        (4, 12),
    ]
    assert [ref.page_size_bytes for ref in canonical.group_data_refs[0]] == [
        12,
        12,
    ]


def test_ascend_connector_worker_accepts_aligned_mamba_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_name = "model.layers.0.mixer"
    spec = MambaSpec(
        block_size=1,
        shapes=((2,), (3,)),
        dtypes=(torch.int8, torch.int8),
        page_size_padded=8,
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=4,
        hisparse_host_num_blocks=None,
        prefix_cacheable_group_ids=(0,),
        kv_cache_groups=[SimpleNamespace(layer_names=[layer_name], kv_cache_spec=spec, host_resident=False)],
        kv_cache_tensors=[],
    )
    worker = AscendOffloadingConnectorWorker.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = kv_cache_config
    worker.vllm_config = object()
    captured: list[CanonicalKVCaches] = []
    worker._init_worker = captured.append
    monkeypatch.setattr(connector_mod, "derive_canonical_mappings", lambda *args: {})

    raw = torch.empty(1 + 4 * 2 + 4 * 3, dtype=torch.int8)
    first_state = raw[1:9].view(4, 2)
    second_state = raw[9:21].view(4, 3)
    worker.register_kv_caches({layer_name: [first_state, second_state]})

    assert len(captured) == 1
    canonical = captured[0]
    assert [tensor.tensor.shape for tensor in canonical.tensors] == [
        (4, 2),
        (4, 3),
    ]
    assert [ref.page_size_bytes for ref in canonical.group_data_refs[0]] == [
        2,
        3,
    ]
    assert canonical.tensors[0].tensor.data_ptr() == first_state.data_ptr()
    assert canonical.tensors[1].tensor.data_ptr() == second_state.data_ptr()


def test_sparse_prefix_registration_uses_durable_host_pair_not_debug_or_hot_cache(monkeypatch):
    main = AscendMLAAttentionSpec(block_size=2, num_kv_heads=1, head_size=3, dtype=torch.bfloat16, store_on_host=True)
    indexer = AscendSFAIndexerCacheSpec(block_size=2, num_kv_heads=1, head_size=2, dtype=torch.bfloat16)
    config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["main"], main), KVCacheGroupSpec(["indexer"], indexer)],
    )
    k = torch.zeros((4, 2, 1, 2), dtype=torch.bfloat16)
    v = torch.zeros((4, 2, 1, 1), dtype=torch.bfloat16)
    index = torch.zeros((4, 2, 1, 2), dtype=torch.bfloat16)
    manager = SimpleNamespace(k_caches_cpu=[k], v_caches_cpu=[v], _get_offload_layer_id=lambda name: 0)
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager.get_sparse_kv_offload_manager",
        lambda: manager,
    )
    monkeypatch.setattr(connector_mod, "derive_canonical_mappings", lambda *args: {})
    worker = AscendOffloadingConnectorWorker.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = config
    worker.vllm_config = object()
    worker.spec = SimpleNamespace(config=SimpleNamespace(extra_config={"sparse_prefix_cache": True}))
    captured = []
    worker._init_worker = captured.append
    # Non-TP0 tuples deliberately have no CPU views. The sparse manager owns
    # the actual TP-shared Host views, which must be used instead.
    worker.register_kv_caches({"main": (None, None, None, None, torch.zeros(1), torch.zeros(1)), "indexer": index})
    canonical = captured[0]
    assert len(canonical.tensors) == 3
    assert [tensor.tensor.data_ptr() for tensor in canonical.tensors] == [k.data_ptr(), v.data_ptr(), index.data_ptr()]
    assert [[ref.page_size_bytes for ref in refs] for refs in canonical.group_data_refs] == [[8, 4], [8]]


def test_sparse_prefix_budget_uses_selected_specs_instead_of_device_only_allocation(monkeypatch):
    config = KVCacheConfig(num_blocks=4, kv_cache_tensors=[object()], kv_cache_groups=[])
    transfer = SimpleNamespace(kv_connector_extra_config={"sparse_prefix_cache": True})
    seen = []

    def budget(vllm_config, kv_cache_config):
        seen.append(kv_cache_config)
        return SimpleNamespace(canonical_layout=False)

    monkeypatch.setattr(connector_mod, "build_offloading_config", budget)
    monkeypatch.setattr(OffloadingSpecFactory, "create_spec", lambda config: object())
    monkeypatch.setattr(
        "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.sparse_scheduler.SparsePrefixOffloadingScheduler.__init__",
        lambda *args: None,
    )
    connector = AscendOffloadingConnector(
        SimpleNamespace(kv_transfer_config=transfer), KVConnectorRole.SCHEDULER, config
    )
    assert seen[0] is not config
    assert seen[0].kv_cache_tensors == []
    assert len(config.kv_cache_tensors) == 1
    assert connector._kv_cache_config is config


def _hisparse_cache_config(indexer_spec) -> KVCacheConfig:
    source = MLAAttentionSpec(block_size=2, num_kv_heads=1, head_size=4, dtype=torch.bfloat16)
    return KVCacheConfig(
        num_blocks=4,
        hisparse_host_num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["main"], source, host_resident=True, role=KVCacheGroupRole.HISPARSE_SOURCE),
            KVCacheGroupSpec(["indexer.0"], indexer_spec, role=KVCacheGroupRole.HISPARSE_INDEXER),
            KVCacheGroupSpec(["resident"], HiSparseResidentSpec(block_size=2, page_size=source.page_size_bytes)),
            KVCacheGroupSpec(
                ["hot"], HiSparseHotSpec(block_size=2, page_size=source.page_size_bytes, blocks_per_request=1)
            ),
            KVCacheGroupSpec(["indexer.1"], indexer_spec, role=KVCacheGroupRole.HISPARSE_INDEXER),
        ],
    )


@pytest.mark.parametrize("split_scale", [False, True])
def test_hisparse_registers_only_indexers_in_upstream_group_order(monkeypatch, split_scale):
    """Host main KV and replicas need not even be exposed to the offloader."""
    spec = AscendSFAIndexerCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.int8,
        scale_dim=1 if split_scale else 0,
        scale_dtype=torch.float16,
    )
    config = _hisparse_cache_config(spec)
    caches = {}
    for layer in ("indexer.0", "indexer.1"):
        # Include storage offsets and gaps between pages. Saving/restoring a
        # logical page must neither include the gaps nor drop its scale.
        key = torch.arange(64, dtype=torch.int8)[5:].as_strided((4, 2, 1, 3), (11, 3, 3, 1))
        if split_scale:
            scale = torch.arange(32, dtype=torch.float16)[3:].as_strided((4, 2, 1), (5, 1, 1))
            caches[layer] = (key, scale)
        else:
            caches[layer] = (key,)
    worker = AscendOffloadingConnectorWorker.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = config
    worker.vllm_config = object()
    captured = []
    worker._init_worker = captured.append
    monkeypatch.setattr(connector_mod, "derive_canonical_mappings", lambda *args: {})

    worker.register_kv_caches(caches)

    (canonical,) = captured
    assert len(canonical.group_data_refs) == 2  # Original group IDs are 1 and 4.
    for refs, parts in zip(canonical.group_data_refs, caches.values()):
        assert sum(ref.page_size_bytes for ref in refs) == spec.page_size_bytes
        assert len(refs) == len(parts)
        for ref, part in zip(refs, parts):
            view = canonical.tensors[ref.tensor_idx].tensor
            assert view.data_ptr() == part.data_ptr()
            expected = part.clone()
            saved = view.clone()
            part.zero_()
            view.copy_(saved)
            torch.testing.assert_close(part, expected, rtol=0, atol=0)


def test_hisparse_tensor_indexer_delegates_without_touching_host_or_replicas(monkeypatch):
    spec = AscendSFAIndexerCacheSpec(block_size=2, num_kv_heads=1, head_size=3, dtype=torch.bfloat16)
    config = _hisparse_cache_config(spec)
    worker = AscendOffloadingConnectorWorker.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = config
    worker.vllm_config = object()
    caches = {name: torch.empty((4, 2, 1, 3), dtype=torch.bfloat16) for name in ("indexer.0", "indexer.1")}
    register = MagicMock()
    monkeypatch.setattr(connector_mod.OffloadingConnectorWorker, "register_kv_caches", register)

    worker.register_kv_caches(caches)

    register.assert_called_once_with(caches)


@pytest.mark.parametrize("host_hit,computed,expected", [(768, 256, 512), (256, 256, 0), (128, 256, 0), (0, 0, 0)])
def test_indexer_restore_uses_upstream_host_prefix_bound(host_hit, computed, expected):
    """A deeper indexer hit cannot skip an absent or shorter main-KV prefix."""
    spec = AscendSFAIndexerCacheSpec(block_size=2, num_kv_heads=1, head_size=3, dtype=torch.bfloat16)
    connector = object.__new__(AscendOffloadingConnector)
    connector._kv_cache_config = _hisparse_cache_config(spec)
    coordinator = MagicMock(spec=HybridKVCacheCoordinator)
    coordinator.find_longest_cache_hit_per_group.return_value = ((), {0: host_hit, 1: 128, 4: 128})
    connector._kv_cache_manager = SimpleNamespace(coordinator=coordinator)
    request = SimpleNamespace(block_hashes=[b"prefix"], num_tokens=1025)

    assert connector._max_loadable_tokens(request, computed) == expected
    assert connector.get_loaded_kv_cache_group_ids(request) == (1, 4)
    coordinator.find_longest_cache_hit_per_group.assert_called_once_with(request.block_hashes, 1024)


def test_hisparse_without_indexer_groups_fails_before_cache_access():
    spec = AscendSFAIndexerCacheSpec(block_size=2, num_kv_heads=1, head_size=3, dtype=torch.bfloat16)
    config = _hisparse_cache_config(spec)
    config.kv_cache_groups = [
        group for group in config.kv_cache_groups if group.role != KVCacheGroupRole.HISPARSE_INDEXER
    ]
    worker = object.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = config

    with pytest.raises(ValueError, match="No eligible"):
        worker.register_kv_caches({})


def test_offloading_connector_is_registered_with_ascend_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.distributed.kv_transfer import register_connector

    registrations: dict[str, tuple[str, str]] = {}

    def capture_registration(
        cls,
        name: str,
        module_path: str,
        class_name: str,
    ) -> None:
        registrations[name] = (module_path, class_name)

    monkeypatch.setattr(KVConnectorFactory, "_registry", {})
    monkeypatch.setattr(OffloadingSpecFactory, "_registry", {})
    monkeypatch.setattr(
        KVConnectorFactory,
        "register_connector",
        classmethod(capture_registration),
    )
    register_connector()

    assert registrations["OffloadingConnector"] == (
        "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector",
        "AscendOffloadingConnector",
    )
    assert set(OffloadingSpecFactory._registry) == {
        "CPUOffloadingSpec",
        "TieringOffloadingSpec",
    }
