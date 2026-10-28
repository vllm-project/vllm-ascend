# SPDX-License-Identifier: Apache-2.0
"""D2RH must transfer every logical DeepSeek-V4 compressed KV page."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_d2rh_connector import (
    BaseMooncakeConnectorScheduler,
    MooncakeConnectorScheduler,
)


def make_scheduler(compress_ratio):
    spec = AscendMLAAttentionSpec(
        block_size=32 * compress_ratio,
        num_kv_heads=1,
        head_size=640,
        dtype=torch.float8_e4m3fn,
        tokens_per_state=compress_ratio,
        model_version="deepseek_v4",
    )
    scheduler = object.__new__(MooncakeConnectorScheduler)
    scheduler.block_size = 32
    scheduler.pcp_size = scheduler.dcp_size = 1
    scheduler.group_transfer_info = [
        scheduler._get_group_transfer_info(SimpleNamespace(kv_cache_spec=spec, layer_names=["attention"]))
    ]
    scheduler.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(hash_block_size=32))
    return scheduler


@pytest.mark.parametrize("compress_ratio,expected_blocks", [(4, 1024), (128, 32)])
def test_128k_prompt_transfers_all_compressed_pages(compress_ratio, expected_blocks):
    scheduler = make_scheduler(compress_ratio)
    # The allocator also reserves a lookahead page. Transfer all prompt pages,
    # but exclude that page; a freshly zeroed destination can hide omissions.
    block_ids = (list(range(1, expected_blocks + 2)),)
    transferred = scheduler._get_transfer_block_ids(block_ids, 130966)
    assert transferred == (list(range(1, expected_blocks + 1)),)


@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_host_hash_changes_at_the_logical_page_boundary(compress_ratio):
    scheduler = make_scheduler(compress_ratio)
    tokens_per_page = 32 * compress_ratio
    tokens = [1] * (tokens_per_page * 3)
    original = SimpleNamespace(prompt_token_ids=tokens)
    changed_tokens = list(tokens)
    changed_tokens[tokens_per_page] = 2
    changed = SimpleNamespace(prompt_token_ids=changed_tokens)
    block_ids = ([10, 20, 30],)
    before = scheduler._d2rh_get_decode_block_hashes(original, block_ids)[0]
    after = scheduler._d2rh_get_decode_block_hashes(changed, block_ids)[0]
    assert all(value is not None for value in before + after)
    assert before[0] == after[0]
    assert before[1:] != after[1:]


def test_context_parallelism_counts_logical_tokens_once():
    scheduler = make_scheduler(4)
    scheduler.dcp_size = 2
    assert scheduler._get_transfer_block_ids((list(range(1, 10)),), 1024) == ([1, 2, 3, 4],)


def test_other_cache_specs_keep_base_transfer_policy():
    scheduler = object.__new__(MooncakeConnectorScheduler)
    group = SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=32), layer_names=["attention"])
    sentinel = object()
    with patch.object(BaseMooncakeConnectorScheduler, "_get_group_transfer_info", return_value=sentinel) as base:
        assert scheduler._get_group_transfer_info(group) is sentinel
    base.assert_called_once_with(group)
