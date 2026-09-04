# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.ascend_config import (
    get_dsv4_shared_compressor_workspace_fallback_reasons,
)
from vllm_ascend.core.kv_cache_interface import AscendCompressorTailSpec
from vllm_ascend.core.single_type_kv_cache_manager import (
    CompressorTailManager,
    get_manager_for_kv_cache_spec,
)

pytestmark = pytest.mark.cpu_test


class _FakeBlockPool:
    def __init__(self) -> None:
        self.null_block = object()
        self.next_block_id = 1
        self.freed_blocks = []

    def get_new_blocks(self, num_blocks: int):
        blocks = [
            SimpleNamespace(block_id=block_id)
            for block_id in range(
                self.next_block_id,
                self.next_block_id + num_blocks,
            )
        ]
        self.next_block_id += num_blocks
        return blocks

    def free_blocks(self, blocks) -> None:
        self.freed_blocks.extend(blocks)


def _tail_spec(*, ratio: int, block_size: int, state_dim: int):
    tail_tokens = ratio * (2 if ratio == 4 else 1)
    return AscendCompressorTailSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=state_dim,
        dtype=torch.float32,
        sliding_window=tail_tokens,
        compress_ratio=ratio,
        model_version="deepseek_v4",
        tail_tokens=tail_tokens,
        ring_blocks_per_request=(tail_tokens + block_size - 1) // block_size,
        state_dim=state_dim,
    )


@pytest.mark.parametrize(
    ("ratio", "block_size", "state_dim", "expected_blocks"),
    [
        pytest.param(4, 8, 2048, 1, id="c4"),
        pytest.param(128, 32, 1024, 4, id="c128"),
    ],
)
def test_tail_spec_has_constant_per_request_capacity(
    ratio: int,
    block_size: int,
    state_dim: int,
    expected_blocks: int,
) -> None:
    spec = _tail_spec(
        ratio=ratio,
        block_size=block_size,
        state_dim=state_dim,
    )

    assert spec.ring_blocks_per_request == expected_blocks
    assert spec.max_admission_blocks_per_request(16_384, 65_536) == expected_blocks
    assert spec.max_memory_usage_bytes(SimpleNamespace()) == (
        expected_blocks * spec.page_size_bytes
    )
    assert spec.max_num_blocks_per_req(SimpleNamespace(), 65_536) == expected_blocks


@pytest.mark.parametrize(
    ("ratio", "block_size", "state_dim", "expected_blocks"),
    [
        pytest.param(4, 8, 2048, 1, id="c4"),
        pytest.param(128, 32, 1024, 4, id="c128"),
    ],
)
def test_tail_manager_does_not_grow_with_chunk_length(
    ratio: int,
    block_size: int,
    state_dim: int,
    expected_blocks: int,
) -> None:
    pool = _FakeBlockPool()
    manager = CompressorTailManager(
        _tail_spec(
            ratio=ratio,
            block_size=block_size,
            state_dim=state_dim,
        ),
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=128,
    )

    for num_tokens in (1, ratio - 1, ratio, 4096, 16_384, 65_536):
        needed = manager.get_num_blocks_to_allocate(
            "request",
            num_tokens,
            (),
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=num_tokens,
        )
        if not manager.req_to_blocks["request"]:
            assert needed == expected_blocks
            manager.allocate_new_blocks("request", num_tokens, num_tokens)
        else:
            assert needed == 0
        assert len(manager.req_to_blocks["request"]) == expected_blocks

    manager.free("request")
    assert len(pool.freed_blocks) == expected_blocks
    assert "request" not in manager.req_to_blocks

    # A preempted request releases its ring and gets exactly the same fixed
    # reservation when scheduled again; sequence length never affects it.
    assert (
        manager.get_num_blocks_to_allocate(
            "request",
            65_536,
            (),
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=65_536,
        )
        == expected_blocks
    )


@pytest.mark.parametrize(
    ("tail_tokens", "block_size", "positions", "expected"),
    [
        pytest.param(8, 8, (0, 7, 8, 15), ((0, 0), (0, 7), (0, 0), (0, 7)), id="c4"),
        pytest.param(
            128,
            32,
            (0, 31, 32, 127, 128, 159),
            ((0, 0), (0, 31), (1, 0), (3, 31), (0, 0), (0, 31)),
            id="c128",
        ),
    ],
)
def test_absolute_position_maps_to_tail_ring(
    tail_tokens: int,
    block_size: int,
    positions: tuple[int, ...],
    expected: tuple[tuple[int, int], ...],
) -> None:
    actual = tuple(
        (
            (position % tail_tokens) // block_size,
            (position % tail_tokens) % block_size,
        )
        for position in positions
    )
    assert actual == expected


def test_all_compressor_call_sites_use_feature_gated_cache_mode() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    model_source = (repo_root / "vllm_ascend/models/deepseek_v4.py").read_text(
        encoding="utf-8"
    )
    assert "from vllm.utils.math_utils import cdiv" in model_source

    expected_call_counts = {
        repo_root / "vllm_ascend/attention/dsa_v1.py": 4,
        repo_root / "vllm_ascend/attention/context_parallel/dsa_cp.py": 2,
    }
    for source_path, expected_count in expected_call_counts.items():
        source = source_path.read_text(encoding="utf-8")
        assert source.count("cache_mode=_compressor_cache_mode()") == expected_count


def test_arch35_cycle_kernel_uses_multi_block_tail_ring() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    kernel_paths = (
        repo_root / "csrc/attention/compressor/op_kernel/arch35/compressor_block_vec.h",
        repo_root
        / "csrc/attention/compressor/op_kernel/arch35/compressor_block_vec_full_load.h",
    )
    for kernel_path in kernel_paths:
        source = kernel_path.read_text(encoding="utf-8")
        # Both read and write variants must index the request row, select a
        # physical ring block, and stop copies at the ring boundary.
        assert source.count("batchIdx * constInfo_.maxBlockNumPerBatch") == 4
        assert source.count("ringTokenIdx / constInfo_.blockSize") == 2
        assert source.count("ringTokenCount - ringTokenIdx") == 2

    tiling_source = (
        repo_root / "csrc/attention/compressor/op_host/arch35/compressor_tiling.cpp"
    ).read_text(encoding="utf-8")
    assert "blockTableShape.GetDimNum() == 1" in tiling_source
    assert "maxBlockNumPerBatch < ringBlocks" in tiling_source
    assert "blockNum < baseParams_->batchSize * ringBlocks" in tiling_source


def test_a3_arch32_cycle_kernel_uses_multi_block_tail_ring() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    kernel_source = (
        repo_root
        / "csrc/attention/compressor/op_kernel/arch32/compressor_block_vec_perf.h"
    ).read_text(encoding="utf-8")
    # PERF is the implementation selected on __CCE_AICORE__ == 220. Lock both
    # state read/write mappings and the fixed-tail SaveState branch.
    assert kernel_source.count("batchIdx * constInfo_.maxBlockNumPerBatch") == 2
    assert kernel_source.count("cacheSeqIdx %= ringTokenCount") == 2
    assert kernel_source.count("ringTokenCount - cacheSeqIdx") == 2
    assert "if constexpr (COMP::cacheMode == CACHE_MODE::CYCLE)" in kernel_source
    assert "writeSeqStartIdx" in kernel_source

    tiling_header = (
        repo_root / "csrc/attention/compressor/op_host/arch32/compressor_tiling.h"
    ).read_text(encoding="utf-8")
    assert "const std::vector<int> CACHE_MODE {1, 2};" in tiling_header

    tiling_source = (
        repo_root / "csrc/attention/compressor/op_host/arch32/compressor_tiling.cpp"
    ).read_text(encoding="utf-8")
    assert "blockTableShape.GetDimNum() == 1" in tiling_source
    assert "maxBlockNumPerBatch < ringBlocks" in tiling_source
    assert "blockNum < baseParams_->batchSize * ringBlocks" in tiling_source


def test_a3_build_route_passes_cache_mode_to_arch32_comp_type() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    entry_source = (
        repo_root / "csrc/attention/compressor/op_kernel/compressor.cpp"
    ).read_text(encoding="utf-8")
    assert '#include "arch32/compressor_kernel_perf.h"' in entry_source
    assert "__CCE_AICORE__ == 220" in entry_source
    assert (
        "CompressorKernelPerf, xLayout, xDtype, ropeDtype, coff, rotaryMode,\n"
        "                                          cacheMode"
    ) in entry_source

    comm_source = (
        repo_root / "csrc/attention/compressor/op_kernel/arch32/compressor_comm.h"
    ).read_text(encoding="utf-8")
    assert "ROTARY_MODE Rotary_Mode, CACHE_MODE Cache_Mode" in comm_source
    assert "static constexpr CACHE_MODE cacheMode = Cache_Mode;" in comm_source


def test_uniform_tail_group_selects_tail_manager() -> None:
    spec = _tail_spec(ratio=128, block_size=32, state_dim=1024)
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=spec.block_size,
        kv_cache_specs={"layer.0": spec, "layer.1": spec},
    )
    pool = _FakeBlockPool()

    with patch.object(
        KVCacheSpecRegistry,
        "get_manager_class",
        return_value=CompressorTailManager,
    ):
        manager = get_manager_for_kv_cache_spec(
            uniform_spec,
            max_in_flight_tokens=16_384,
            max_model_len=65_536,
            block_pool=pool,
            enable_caching=False,
            kv_cache_group_id=0,
            scheduler_block_size=128,
        )

    assert isinstance(manager, CompressorTailManager)
    assert manager.kv_cache_spec is spec
    assert manager.ring_blocks_per_request == 4


def _stage1_config(**overrides):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_v4"),
            enforce_eager=True,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        kv_transfer_config=None,
        speculative_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )
    for dotted_name, value in overrides.items():
        owner_name, field_name = dotted_name.split("__", maxsplit=1)
        setattr(getattr(config, owner_name), field_name, value)
    return config


def test_stage1_feature_gate_accepts_target_configuration() -> None:
    reasons = get_dsv4_shared_compressor_workspace_fallback_reasons(
        _stage1_config(),
        is_a3=True,
        multistream_dsv4_dsa_overlap=False,
    )
    assert reasons == []


@pytest.mark.parametrize(
    ("config", "is_a3", "multistream", "expected_reason"),
    [
        pytest.param(
            _stage1_config(cache_config__enable_prefix_caching=True),
            True,
            False,
            "prefix caching is enabled",
            id="prefix-cache",
        ),
        pytest.param(
            _stage1_config(),
            True,
            True,
            "multistream_dsv4_dsa_overlap is enabled",
            id="multistream",
        ),
        pytest.param(
            _stage1_config(),
            False,
            False,
            "device is not A3",
            id="device",
        ),
    ],
)
def test_stage1_feature_gate_falls_back(
    config,
    is_a3: bool,
    multistream: bool,
    expected_reason: str,
) -> None:
    reasons = get_dsv4_shared_compressor_workspace_fallback_reasons(
        config,
        is_a3=is_a3,
        multistream_dsv4_dsa_overlap=multistream,
    )
    assert expected_reason in reasons
