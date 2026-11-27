# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.core.gdn_checkpoint_policy import GDNCheckpointPolicy


@pytest.mark.parametrize(
    "position,start,end,expected",
    [
        (1984, 0, 2003, 1152),
        (1984, 1152, 2003, 1984),
        (3456, 1152, 3503, 0),
        (6976, 5760, 7003, 6976),
        (8960, 6912, 9003, 0),
    ],
)
def test_internal_checkpoint_position(position, start, end, expected):
    policy = GDNCheckpointPolicy(8192, True, True)
    assert policy.internal_position(position, start, end, 1152, 64) == expected


@pytest.mark.parametrize(
    "position,start,end,expected",
    [
        (7424, 0, 7488, 6144),
        (7424, 6144, 7488, 7424),
        (8128, 0, 8192, 6144),
        (8192, 0, 8256, 0),
        (1984, 0, 2048, 1984),
    ],
)
def test_cold_prefill_preserves_last_reusable_physical_page(position, start, end, expected):
    policy = GDNCheckpointPolicy(8192, True, True)
    assert policy.internal_position(position, start, end, 2048, 64) == expected


@pytest.mark.parametrize(
    ("checkpoint", "expected_retention"),
    [
        ((6144, 2), None),
        ((4096, 1), 0),
        ((7424, 2), 0),
        (None, 0),
    ],
)
def test_cold_physical_checkpoint_survives_sparse_retention(checkpoint, expected_retention):
    from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager, MambaManager

    manager = object.__new__(AscendMambaManager)
    manager.gdn_checkpoint_policy = GDNCheckpointPolicy(8192, True, True)
    manager._checkpoints = {"prompt": checkpoint} if checkpoint is not None else {}
    manager.cache_hit_alignment_tokens = 64
    manager.block_size = 2048
    manager.kv_cache_spec = SimpleNamespace(prefill_checkpoint_alignment=64)
    manager.drop_eagle_checkpoint_block = False
    manager.block_pool = SimpleNamespace(num_hash_lookahead_tokens=0)
    request = SimpleNamespace(request_id="prompt", num_prompt_tokens=7488)
    with patch.object(MambaManager, "_cache_partial_tail_block", return_value=b"checkpoint") as cache:
        assert manager._cache_partial_tail_block(request, 7488, retention_interval=0) == b"checkpoint"
    cache.assert_called_once_with(request, 7488, expected_retention)
