# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.patch.general import patch_get_request_block_hasher as hasher_patch
from vllm_ascend.patch.general.patch_hybrid_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
)
from vllm_ascend.patch.general.patch_mamba_manager import AscendMambaManager
from vllm_ascend.patch.worker import patch_mamba_utils
from vllm_ascend.worker.pcp_utils import PCPManager


def _set_hash_config_snapshot(
    *,
    cache_block_size: int,
    pcp_size: int = 1,
    dcp_size: int = 1,
) -> None:
    hasher_patch._VLLM_CONFIG_SNAPSHOT.clear()
    hasher_patch._VLLM_CONFIG_SNAPSHOT.update(
        {
            "decode_context_parallel_size": dcp_size,
            "prefill_context_parallel_size": pcp_size,
            "block_size": cache_block_size,
        }
    )


def _make_coordinator(dcp_world_size: int, pcp_world_size: int) -> AscendHybridKVCacheCoordinator:
    coord = object.__new__(AscendHybridKVCacheCoordinator)
    coord.dcp_world_size = dcp_world_size
    coord.pcp_world_size = pcp_world_size
    return coord


def _make_mamba_spec(block_size: int = 16) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )


def _make_full_attention_spec(block_size: int = 16) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        head_size_v=1,
        dtype=torch.float32,
    )


@pytest.mark.parametrize(
    ("pcp_size", "dcp_size", "input_block_size", "expected"),
    [
        (1, 1, 16, 16),
        (2, 1, 32, 32),
        (1, 2, 32, 16),
        (2, 2, 64, 32),
        (2, 2, 16, 16),
    ],
)
def test_get_hash_size_normalizes_virtual_block_size(
    pcp_size: int,
    dcp_size: int,
    input_block_size: int,
    expected: int,
) -> None:
    cache_block_size = 16
    _set_hash_config_snapshot(
        cache_block_size=cache_block_size,
        pcp_size=pcp_size,
        dcp_size=dcp_size,
    )
    assert hasher_patch.get_hash_size(input_block_size) == expected


@pytest.mark.parametrize(
    ("pcp_size", "dcp_size", "spec_factory", "expected"),
    [
        (1, 1, _make_full_attention_spec, 16),
        (2, 2, _make_full_attention_spec, 64),
        (2, 2, _make_mamba_spec, 32),
        (2, 1, _make_mamba_spec, 32),
        (1, 2, _make_mamba_spec, 16),
    ],
)
def test_ascend_hybrid_coordinator_effective_block_size(
    pcp_size: int,
    dcp_size: int,
    spec_factory,
    expected: int,
) -> None:
    coord = _make_coordinator(dcp_world_size=dcp_size, pcp_world_size=pcp_size)
    assert coord._get_effective_block_size(spec_factory()) == expected


def test_ascend_mamba_manager_divides_block_size_for_dcp() -> None:
    spec = _make_mamba_spec()
    block_pool = MagicMock()

    def _fake_mamba_init(self, kv_cache_spec, block_pool, **kwargs):
        self.block_size = kv_cache_spec.block_size * 4
        self.dcp_world_size = kwargs.get("dcp_world_size", 1)
        self.pcp_world_size = kwargs.get("pcp_world_size", 1)

    with patch(
        "vllm_ascend.patch.general.patch_mamba_manager.MambaManager.__init__",
        _fake_mamba_init,
    ):
        mgr = AscendMambaManager(
            spec,
            block_pool,
            dcp_world_size=2,
            pcp_world_size=2,
        )

    assert mgr.block_size == 32


@pytest.mark.parametrize(
    ("pcp_world_size", "expected"),
    [(1, 16), (2, 32), (4, 64)],
)
def test_mamba_utils_effective_block_size_scales_with_pcp_only(
    pcp_world_size: int,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        patch_mamba_utils,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=pcp_world_size),
    )
    assert patch_mamba_utils._get_effective_block_size(16) == expected


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("qwen3_next", True),
        ("qwen3_5_moe", True),
        ("qwen2", False),
    ],
)
def test_pcp_manager_hybrid_attn_model_types(
    model_type: str,
    expected: bool,
) -> None:
    vllm_config = MagicMock()
    vllm_config.model_config.hf_config.model_type = model_type
    vllm_config.parallel_config.cp_kv_cache_interleave_size = 64
    vllm_config.speculative_config.num_speculative_tokens = 0

    pcp_manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        dcp_world_size=1,
        dcp_rank=0,
        max_buffer_num_tokens=128,
        max_num_reqs=8,
        device="cpu",
        vllm_config=vllm_config,
        use_async_scheduling=False,
        pin_memory=False,
    )
    assert pcp_manager.pcp_use_hybrid_attn is expected
