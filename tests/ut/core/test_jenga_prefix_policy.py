# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[3] / "vllm_ascend" / "core" / "jenga_prefix_policy.py"
SPEC = importlib.util.spec_from_file_location("jenga_prefix_policy_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
jenga_prefix_policy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = jenga_prefix_policy
SPEC.loader.exec_module(jenga_prefix_policy)

CheckpointStatePrefixPolicy = jenga_prefix_policy.CheckpointStatePrefixPolicy
clip_prefill_to_state_checkpoint = jenga_prefix_policy.clip_prefill_to_state_checkpoint
FullAttentionPrefixPolicy = jenga_prefix_policy.FullAttentionPrefixPolicy
PrefixGroupState = jenga_prefix_policy.PrefixGroupState
SlidingWindowPrefixPolicy = jenga_prefix_policy.SlidingWindowPrefixPolicy
find_longest_common_prefix = jenga_prefix_policy.find_longest_common_prefix


def test_prefill_chunk_stops_at_each_real_state_checkpoint() -> None:
    assert clip_prefill_to_state_checkpoint(0, 700, 1024, 512) == 512
    assert clip_prefill_to_state_checkpoint(384, 316, 1024, 512) == 128
    assert clip_prefill_to_state_checkpoint(512, 512, 1024, 512) == 512
    assert clip_prefill_to_state_checkpoint(512, 188, 700, 512) == 188


def test_state_checkpoint_split_does_not_clip_decode() -> None:
    assert clip_prefill_to_state_checkpoint(1024, 4, 1024, 512) == 4


def test_full_attention_stops_at_first_hole_and_touches_entire_prefix() -> None:
    policy = FullAttentionPrefixPolicy(block_size_tokens=4)

    assert policy.possible_prefix_lengths({0, 1, 3, 4}, 20) == {0, 4, 8}
    assert policy.blocks_to_touch({0, 1, 3, 4}, 8) == (0, 1)

    with pytest.raises(ValueError, match="missing block"):
        policy.blocks_to_touch({0, 1, 3, 4}, 12)


def test_sliding_window_allows_holes_before_but_not_inside_tail() -> None:
    policy = SlidingWindowPrefixPolicy(
        block_size_tokens=4,
        window_size_tokens=8,
    )

    # At prefix 16 only blocks 2 and 3 intersect the trailing eight tokens.
    # Missing blocks 0 and 1 are outside the dependency window.
    assert policy.possible_prefix_lengths({2, 3}, 16) == {0, 16}
    assert policy.blocks_to_touch({2, 3}, 16) == (2, 3)

    with pytest.raises(ValueError, match="hole inside"):
        policy.blocks_to_touch({2}, 16)


def test_sliding_window_includes_every_partially_intersected_block() -> None:
    policy = SlidingWindowPrefixPolicy(
        block_size_tokens=4,
        window_size_tokens=5,
    )

    # [7, 12) intersects blocks 1 and 2, even though only one token from block
    # 1 is inside the five-token window.
    assert 12 not in policy.possible_prefix_lengths({2}, 12)
    assert 12 in policy.possible_prefix_lengths({1, 2}, 12)
    assert policy.blocks_to_touch({1, 2}, 12) == (1, 2)


def test_checkpoint_state_requires_a_cached_stride_boundary() -> None:
    policy = CheckpointStatePrefixPolicy(checkpoint_stride_tokens=8)

    assert policy.possible_prefix_lengths({0, 2, 4}, 40) == {0, 8, 24, 40}
    assert policy.blocks_to_touch({0, 2, 4}, 24) == (2,)

    with pytest.raises(ValueError, match="checkpoint aligned"):
        policy.blocks_to_touch({0, 2, 4}, 20)
    with pytest.raises(ValueError, match="not cached"):
        policy.blocks_to_touch({0, 2, 4}, 16)


def test_global_match_intersects_groups_and_uses_group_specific_touch_sets() -> None:
    groups = {
        "full": PrefixGroupState(
            FullAttentionPrefixPolicy(block_size_tokens=4),
            {0, 1, 2, 3, 4, 5},
        ),
        "sliding": PrefixGroupState(
            SlidingWindowPrefixPolicy(
                block_size_tokens=4,
                window_size_tokens=8,
            ),
            {2, 3, 4, 5},
        ),
        "state": PrefixGroupState(
            CheckpointStatePrefixPolicy(checkpoint_stride_tokens=8),
            {1, 2},
        ),
    }

    match = find_longest_common_prefix(groups, max_prefix_tokens=26)

    assert match.prefix_tokens == 24
    assert match.touched_blocks == {
        "full": (0, 1, 2, 3, 4, 5),
        "sliding": (4, 5),
        "state": (2,),
    }


def test_global_match_falls_back_to_zero_when_no_nonzero_prefix_is_common() -> None:
    groups = {
        0: PrefixGroupState(FullAttentionPrefixPolicy(4), {0}),
        1: PrefixGroupState(CheckpointStatePrefixPolicy(8), {1}),
    }

    match = find_longest_common_prefix(groups, max_prefix_tokens=16)

    assert match.prefix_tokens == 0
    assert match.touched_blocks == {0: (), 1: ()}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FullAttentionPrefixPolicy(0),
        lambda: SlidingWindowPrefixPolicy(4, 0),
        lambda: CheckpointStatePrefixPolicy(-1),
    ],
)
def test_policy_granularity_must_be_positive(factory) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        factory()


def test_invalid_cache_indices_and_empty_group_set_are_rejected() -> None:
    policy = FullAttentionPrefixPolicy(4)

    with pytest.raises(ValueError, match="non-negative integers"):
        policy.possible_prefix_lengths({-1, 0}, 8)
    with pytest.raises(ValueError, match="at least one"):
        find_longest_common_prefix({}, 8)
