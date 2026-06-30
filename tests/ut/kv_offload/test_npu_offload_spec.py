# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Regression tests for the Ascend NPU KV offloading specs.

These guard against the worker-side specs drifting from the upstream
``CPUOffloadingSpec`` / ``TieringOffloadingSpec`` / ``SharedOffloadRegion``
APIs. Upstream reworked ``SharedOffloadRegion.__init__`` to size the mmap from
``num_blocks * kv_bytes_per_block`` and dropped the old ``total_size_bytes`` /
``num_workers`` kwargs; the multi-tier offloading adaptation must follow that
signature or it crashes at worker-side handler construction time.
"""

import inspect
from types import SimpleNamespace

from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

import vllm_ascend.kv_offload.npu as npu_mod


def _build_configs(num_cpu_blocks=1000, world_size=1, block_size=128, block_factor=None):
    extra: dict = {"num_cpu_blocks": num_cpu_blocks}
    if block_factor is not None:
        extra["block_size"] = block_size * block_factor
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            world_size=world_size,
        ),
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=10,
        # 2 tensors * 1280 bytes = 2560 total -> 256 bytes / gpu block
        kv_cache_tensors=[SimpleNamespace(size=1280), SimpleNamespace(size=1280)],
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=block_size))
        ],
    )
    return vllm_config, kv_cache_config, extra


def test_legacy_num_blocks_recovers_num_blocks():
    """cpu_bytes_to_use must be set so CPUOffloadingSpec recovers num_blocks."""
    vllm_config, kv_cache_config, extra = _build_configs(num_cpu_blocks=1000)
    npu_mod._set_cpu_bytes_from_legacy_num_blocks(vllm_config, kv_cache_config)

    kv_bytes_per_offloaded_block = (2560 // 10) * 1  # world_size=1, factor=1
    assert extra["cpu_bytes_to_use"] == 1000 * kv_bytes_per_offloaded_block
    # The upstream CPUOffloadingSpec recomputes num_blocks from this budget.
    assert extra["cpu_bytes_to_use"] // kv_bytes_per_offloaded_block == 1000


def test_legacy_num_blocks_honors_block_size_factor():
    vllm_config, kv_cache_config, extra = _build_configs(
        num_cpu_blocks=500, block_size=128, block_factor=4
    )
    npu_mod._set_cpu_bytes_from_legacy_num_blocks(vllm_config, kv_cache_config)
    kv_bytes_per_offloaded_block = (2560 // 10) * 1 * 4
    assert extra["cpu_bytes_to_use"] == 500 * kv_bytes_per_offloaded_block


def test_legacy_num_blocks_noop_when_bytes_already_set():
    vllm_config, kv_cache_config, extra = _build_configs()
    extra["cpu_bytes_to_use"] = 12345
    npu_mod._set_cpu_bytes_from_legacy_num_blocks(vllm_config, kv_cache_config)
    assert extra["cpu_bytes_to_use"] == 12345


def test_tiering_create_handlers_matches_shared_region_signature(monkeypatch):
    """create_handlers must call SharedOffloadRegion with a valid signature."""
    captured: dict = {}

    class _FakeRegion:
        def __init__(self, **kwargs):
            # Raises TypeError if kwargs don't bind to the real upstream
            # __init__ (this is exactly what the original bug violated).
            inspect.signature(SharedOffloadRegion.__init__).bind(self, **kwargs)
            captured.update(kwargs)
            captured["region"] = self

    sentinel_handlers = object()

    def _fake_handlers(**kwargs):
        captured["handler_kwargs"] = kwargs
        return sentinel_handlers

    monkeypatch.setattr(npu_mod, "SharedOffloadRegion", _FakeRegion)
    monkeypatch.setattr(npu_mod, "CpuNpuOffloadingHandlers", _fake_handlers)
    monkeypatch.setattr(
        npu_mod.torch,
        "npu",
        SimpleNamespace(current_device=lambda: 0),
        raising=False,
    )

    spec = npu_mod.NPUTieringOffloadingSpec.__new__(npu_mod.NPUTieringOffloadingSpec)
    spec.vllm_config = SimpleNamespace(
        instance_id="inst",
        parallel_config=SimpleNamespace(world_size=2),
    )
    spec.cpu_page_size_per_worker = 64
    spec.num_blocks = 10
    spec.block_size_factor = 1
    # Aligned per-block row stride exposed by CPUOffloadingSpec; the mmap region
    # derives its total size from num_blocks * this value.
    spec.kv_bytes_per_offloaded_block = 4096

    result = spec.create_handlers(kv_caches=object())

    assert result is sentinel_handlers
    # New upstream signature: size is derived from num_blocks * kv_bytes_per_block.
    assert captured["kv_bytes_per_block"] == 4096
    assert captured["num_blocks"] == 10
    assert captured["cpu_page_size"] == 64
    assert captured["rank"] == 0
    assert captured["instance_id"] == "inst"
    # The removed kwargs must never reappear.
    assert "total_size_bytes" not in captured
    assert "num_workers" not in captured
    # The worker handlers must receive the mmap region they will clean up.
    assert captured["handler_kwargs"]["mmap_region"] is captured["region"]
