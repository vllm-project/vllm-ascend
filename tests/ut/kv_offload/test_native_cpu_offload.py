# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheTensor, MambaSpec
from vllm.v1.kv_offload.base import CanonicalKVCaches, CanonicalKVCacheTensor, OffloadingWorker
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

import vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.cpu_npu as cpu_npu_mod
import vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector as connector_mod
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
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
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])
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
        kv_cache_groups=[SimpleNamespace(layer_names=[layer_name], kv_cache_spec=spec)],
        kv_cache_tensors=[],
        # The connector offloads the groups vLLM selects, not all of them.
        prefix_cacheable_group_ids=(0,),
        hisparse_host_num_blocks=None,
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
        kv_cache_groups=[SimpleNamespace(layer_names=[layer_name], kv_cache_spec=spec)],
        kv_cache_tensors=[],
        # The connector offloads the groups vLLM selects, not all of them.
        prefix_cacheable_group_ids=(0,),
        hisparse_host_num_blocks=None,
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


@pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
@pytest.mark.parametrize("bounded_replay", [False, True])
def test_v41_offload_capacity_reserves_every_physical_slot(
    monkeypatch: pytest.MonkeyPatch,
    role: KVConnectorRole,
    bounded_replay: bool,
) -> None:
    """A multi-slot V4.1 pool must not size every CPU mirror from slot zero."""

    num_blocks = 4
    slot_sizes = (131072, 131072, 131072, 147712)
    reserved_per_block = sum(slot_sizes)
    budget = 8 * round_up(reserved_per_block * 2 * 2, NPUOffloadingSpec.BLOCK_SIZE_ALIGNMENT)
    groups = [
        SimpleNamespace(
            layer_names=["main"],
            kv_cache_spec=AscendMLAAttentionSpec(
                block_size=128,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.bfloat16,
                model_version="deepseek_v41",
            ),
        ),
        SimpleNamespace(
            layer_names=["swa"],
            kv_cache_spec=AscendSlidingWindowMLASpec(
                block_size=128,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.bfloat16,
                sliding_window=128,
                bounded_replay=bounded_replay,
            ),
        ),
    ]
    cache_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=[SimpleNamespace(size=num_blocks * size) for size in slot_sizes],
    )
    vllm_config = object()

    def build_config(config, caches):
        assert config is vllm_config
        assert caches is cache_config
        return replace(
            _make_config({"cpu_bytes_to_use": budget}),
            worker_kv_bytes_per_block=caches.kv_cache_tensors[0].size // num_blocks,
        )

    def create_spec(config):
        return NPUOffloadingSpec(config)

    upstream_calls = []

    def upstream_init(self, config, connector_role, caches):
        upstream_calls.append(connector_role)
        spec = create_spec(build_config(config, caches))
        self.connector_scheduler = SimpleNamespace(spec=spec) if connector_role == KVConnectorRole.SCHEDULER else None
        self.connector_worker = SimpleNamespace(spec=spec) if connector_role == KVConnectorRole.WORKER else None

    monkeypatch.setattr(KVConnectorBase_V1, "__init__", lambda *args: None)
    monkeypatch.setattr(OffloadingConnector, "__init__", upstream_init)
    monkeypatch.setattr(connector_mod, "build_offloading_config", build_config, raising=False)
    monkeypatch.setattr(
        connector_mod, "OffloadingConnectorScheduler", lambda spec, *args: SimpleNamespace(spec=spec), raising=False
    )
    monkeypatch.setattr(OffloadingSpecFactory, "create_spec", create_spec)

    connector = AscendOffloadingConnector(vllm_config, role, cache_config)
    owner = connector.connector_scheduler if role == KVConnectorRole.SCHEDULER else connector.connector_worker
    spec = owner.spec
    if not bounded_replay:
        assert upstream_calls == [role]
        assert spec.config.worker_kv_bytes_per_block == slot_sizes[0]
        return
    assert upstream_calls == []
    assert spec.config.worker_kv_bytes_per_block == reserved_per_block
    assert spec.num_chunks == 8
    # Use the actual allocator with the production BF16 main/index geometry.
    cpu_mirrors = []

    def make_handler(**kwargs):
        cpu_mirrors.append(kwargs["cpu_tensors"])
        return SimpleNamespace(shutdown=lambda: None)

    monkeypatch.setattr(cpu_npu_mod, "SingleDirectionNPUOffloadingHandler", make_handler)
    payload_sizes = (65536, 8320) * 3 + (131072, 16640)
    canonical = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(torch.empty((1, size), dtype=torch.int8), size) for size in payload_sizes],
        group_data_refs=[[]],
    )
    spec.create_worker(canonical)
    actual_cpu_bytes = sum(tensor.numel() for tensor in cpu_mirrors[0])
    assert actual_cpu_bytes * spec.config.parallel.world_size <= budget
    assert [tensor.size for tensor in cache_config.kv_cache_tensors] == [num_blocks * size for size in slot_sizes]


def test_split_registration_excludes_swa_and_keeps_only_unpadded_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SWA-sized slot padding must not become part of the CPU mirror."""

    num_blocks = 4
    main_spec = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=3, dtype=torch.bfloat16, page_size_padded=64)
    split_spec = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=1, dtype=torch.bfloat16)
    swa_spec = AscendSlidingWindowMLASpec(
        block_size=2,
        num_kv_heads=1,
        head_size=3,
        dtype=torch.bfloat16,
        sliding_window=128,
        bounded_replay=True,
    )
    cache_config = SimpleNamespace(
        num_blocks=num_blocks,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["swa"], kv_cache_spec=swa_spec),
            SimpleNamespace(layer_names=["main"], kv_cache_spec=main_spec),
            SimpleNamespace(layer_names=["split"], kv_cache_spec=split_spec),
        ],
        kv_cache_tensors=[KVCacheTensor(size=num_blocks * 64, layers=["main", "swa"], layer_stride=0, block_stride=64)],
        prefix_cacheable_group_ids=(1, 2),
        hisparse_host_num_blocks=None,
    )
    backing = torch.arange(num_blocks * 64, dtype=torch.uint8)
    main = backing.view(num_blocks, 64)
    worker = AscendOffloadingConnectorWorker.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = cache_config
    worker.vllm_config = object()
    captured = []
    worker._init_worker = captured.append
    monkeypatch.setattr(connector_mod, "derive_canonical_mappings", lambda *args: {})
    # No SWA entry is needed: bounded replay rebuilds it on the NPU.
    worker.register_kv_caches(
        {
            "main": main,
            "split": (torch.zeros((num_blocks, 4), dtype=torch.int8), torch.zeros((num_blocks, 4), dtype=torch.int8)),
        }
    )
    canonical = captured[0]
    assert len(canonical.group_data_refs) == 2
    assert [tensor.page_size_bytes for tensor in canonical.tensors] == [24, 4, 4]
    assert canonical.tensors[0].tensor.stride(0) == 64
    assert torch.equal(canonical.tensors[0].tensor.view(torch.uint8), main[:, :24])
    assert sum(tensor.page_size_bytes for tensor in canonical.tensors) == 32
