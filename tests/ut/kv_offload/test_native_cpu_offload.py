# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.kv_offload.base import CanonicalKVCaches
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.cpu_npu import (
    NPUOffloadingWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu import NPUOffloadingSpec
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector import (
    AscendOffloadingConnector,
    AscendOffloadingConnectorWorker,
    _canonicalize_split_attention_cache,
)
from vllm_ascend.utils import vllm_version_is


def _make_config(extra_config: dict[str, object]) -> OffloadingConfig:
    return OffloadingConfig(
        groups=(
            OffloadingGroupConfig(
                tokens_per_block=16,
                layer_names=("model.layers.0.self_attn",),
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
            **(
                {}
                if vllm_version_is("0.27.1")
                else {
                    "data_parallel_size": 1,
                    "data_parallel_rank_local": None,
                }
            ),
        ),
    )


def test_npu_offloading_spec_uses_upstream_cpu_manager() -> None:
    bytes_per_chunk = 64 * 2 * 2
    aligned_bytes_per_chunk = round_up(
        bytes_per_chunk,
        NPUOffloadingSpec.BLOCK_SIZE_ALIGNMENT,
    )
    spec = NPUOffloadingSpec(_make_config({"cpu_bytes_to_use": 10 * aligned_bytes_per_chunk}))

    assert spec.num_blocks == 10
    assert isinstance(spec.get_manager(), CPUOffloadingManager)


def test_npu_offloading_spec_supports_legacy_num_cpu_blocks() -> None:
    extra_config: dict[str, object] = {"num_cpu_blocks": 10}
    spec = NPUOffloadingSpec(_make_config(extra_config))
    aligned_bytes_per_chunk = round_up(
        64 * 2 * 2,
        NPUOffloadingSpec.BLOCK_SIZE_ALIGNMENT,
    )

    assert spec.num_blocks == 10
    assert spec.extra_config["cpu_bytes_to_use"] == 10 * aligned_bytes_per_chunk
    assert "cpu_bytes_to_use" not in extra_config


def test_legacy_num_cpu_blocks_is_preserved_on_scheduler() -> None:
    config = _make_config({"num_cpu_blocks": 10})
    object.__setattr__(config, "worker_kv_bytes_per_block", 0)

    spec = NPUOffloadingSpec(config)

    assert spec.num_blocks == 10


def test_npu_offloading_spec_loads_through_vllm_factory() -> None:
    spec_cls = OffloadingSpecFactory.get_spec_cls(
        {
            "spec_name": "NPUOffloadingSpec",
            "spec_module_path": "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu",
        }
    )

    assert spec_cls is NPUOffloadingSpec


def test_npu_worker_reuses_upstream_worker_protocol() -> None:
    assert issubclass(NPUOffloadingWorker, CPUOffloadingWorker)


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

    views = _canonicalize_split_attention_cache(
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

    views = _canonicalize_split_attention_cache(
        (key, value),
        num_blocks=4,
        unpadded_page_size_bytes=8,
    )

    assert [view.shape for view, _ in views] == [(4, 4), (4, 4)]
    assert [copy_size for _, copy_size in views] == [4, 4]


def test_extra_physical_blocks_do_not_hide_separate_value_cache() -> None:
    key = torch.empty((10, 2), dtype=torch.int8)
    value = torch.empty((10, 2), dtype=torch.int8)

    views = _canonicalize_split_attention_cache(
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

    views = _canonicalize_split_attention_cache(
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
        _canonicalize_split_attention_cache(
            (key, value),
            num_blocks=4,
            unpadded_page_size_bytes=12,
        )


def test_ascend_connector_worker_accepts_separate_kv_tensors() -> None:
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
    )
    worker = AscendOffloadingConnectorWorker.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = kv_cache_config
    captured: list[CanonicalKVCaches] = []
    worker._init_worker = captured.append

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


def test_ascend_connector_worker_accepts_aligned_mamba_states() -> None:
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
    )
    worker = AscendOffloadingConnectorWorker.__new__(AscendOffloadingConnectorWorker)
    worker.kv_cache_config = kv_cache_config
    captured: list[CanonicalKVCaches] = []
    worker._init_worker = captured.append

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
