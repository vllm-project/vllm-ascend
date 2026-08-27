# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""vLLM v0.27.1 contract tests for native NPU KV offloading."""

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

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
    spec.num_blocks = 17

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
    spec.num_blocks = 10
    spec.blocks_per_chunk = 2
    spec.kv_bytes_per_chunk = 4096
    kv_caches = object()

    result = spec.create_worker(kv_caches=kv_caches)

    assert result is sentinel_worker
    assert captured["engine_id"] == "engine-dp0"
    assert captured["num_blocks"] == 10
    assert captured["rank"] == expected_rank
    assert captured["kv_bytes_per_block"] == 4096
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
        SimpleNamespace(
            num_blocks=4,
            kv_cache_groups=[SimpleNamespace(layer_names=[layer_name], kv_cache_spec=spec)],
            kv_cache_tensors=[],
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
        SimpleNamespace(
            num_blocks=4,
            kv_cache_groups=[SimpleNamespace(layer_names=[layer_name], kv_cache_spec=spec)],
            kv_cache_tensors=[],
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
