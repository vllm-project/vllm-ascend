# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Capacity accounting for owner-sharded DeepSeek V4.1 KV tuples."""

from types import SimpleNamespace

import pytest

from vllm_ascend.attention.context_parallel import pcp_owner_stage
from vllm_ascend.models.deepseek_v41 import cache_config


def _config(pcp_size: int = 4, tp_size: int = 1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=1,
            tensor_parallel_size=tp_size,
        ),
        additional_config={"pcp_kv_sharing": "owner"},
        kv_transfer_config=None,
    )


def test_physical_page_sizes_leave_replicated_tuples_whole():
    page_sizes = [131072, 16640]
    layer_tuples = [
        ("layers.2.long_kv_cache", "layers.2.swa_cache"),
        ("layers.2.indexer.k_cache",),
    ]

    assert cache_config._physical_page_sizes(page_sizes, layer_tuples, 4) == [32768, 16640]


def test_physical_page_sizes_without_sharding_are_nominal():
    page_sizes = [582528, 16640]
    layer_tuples = [("layers.2.long_kv_cache",), ("layers.2.indexer.k_cache",)]

    assert cache_config._physical_page_sizes(page_sizes, layer_tuples, 1) == page_sizes


def test_physical_page_sizes_reject_unaligned_pages():
    with pytest.raises(ValueError, match="Cannot split"):
        cache_config._physical_page_sizes([131071], [("layers.2.long_kv_cache",)], 4)


def test_shard_size_requires_capacity_switch(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_PCP_OWNER_CAPACITY", raising=False)

    assert cache_config.owner_capacity_shard_size(_config()) == 1


def test_shard_size_requires_owner_mode(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_PCP_OWNER_CAPACITY", "1")
    config = _config()
    config.additional_config = {"pcp_kv_sharing": "replicated"}

    assert cache_config.owner_capacity_shard_size(config) == 1


def test_shard_size_requires_supported_config(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_PCP_OWNER_CAPACITY", "1")

    assert cache_config.owner_capacity_shard_size(_config(tp_size=2)) == 1


def test_shard_size_uses_pcp_when_owner_is_active(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_PCP_OWNER_CAPACITY", "1")

    assert cache_config.owner_capacity_shard_size(_config(pcp_size=4)) == 4


def test_stage_reserve_is_zero_when_staging_is_off(monkeypatch):
    for name in (
        "VLLM_ASCEND_PCP_OWNER_STAGE",
        "VLLM_ASCEND_PCP_OWNER_STAGE_SWA",
        "VLLM_ASCEND_PCP_OWNER_STAGE_SWA_CHUNK",
        "VLLM_ASCEND_PCP_OWNER_STAGE_INDEX",
    ):
        monkeypatch.delenv(name, raising=False)

    assert pcp_owner_stage.owner_stage_reserve_bytes() == 0


def test_stage_reserve_covers_enabled_budgets(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_PCP_OWNER_STAGE", "1")
    monkeypatch.setenv("VLLM_ASCEND_PCP_OWNER_STAGE_SWA_CHUNK", "1")
    monkeypatch.setenv("VLLM_ASCEND_PCP_OWNER_STAGE_INDEX", "1")

    reserve = pcp_owner_stage.owner_stage_reserve_bytes()

    expected = (
        pcp_owner_stage.DEFAULT_STAGE_BUDGET_BYTES
        + pcp_owner_stage.DEFAULT_SWA_CHUNK_STAGE_BUDGET_BYTES
        + pcp_owner_stage.DEFAULT_INDEX_STAGE_BUDGET_BYTES
    )
    assert reserve == expected
