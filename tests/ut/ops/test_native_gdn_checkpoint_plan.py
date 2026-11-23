# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.core.gdn_checkpoint_policy import GDNCheckpointPolicy
from vllm_ascend.ops.gdn_checkpoint import build_gdn_checkpoint_metadata
from vllm_ascend.ops.native_gdn_checkpoint import plan_native_checkpoint


def test_mixed_batch_restores_original_token_order():
    prefix, tail, tail_rows, restore, prefix_limits, tail_limits = plan_native_checkpoint([2, 0, 1], [4, 2, 3])
    packed = prefix + tail
    assert prefix == [0, 1, 4, 5, 6]
    assert tail == [2, 3, 7, 8]
    assert tail_rows == [0, 2]
    assert prefix_limits == [0, 2, 4, 5]
    assert tail_limits == [0, 2, 4]
    assert [packed[index] for index in restore] == list(range(9))


@pytest.mark.parametrize(("offsets", "lengths"), [([4], [4]), ([0], [4])])
def test_plan_requires_a_checkpoint_with_a_nonempty_tail(offsets, lengths):
    with pytest.raises(AssertionError):
        plan_native_checkpoint(offsets, lengths)


def test_host_block_ids_match_device_null_mask():
    cache = SimpleNamespace(hash_block_size=64, cache_hit_alignment_tokens=64)
    config = SimpleNamespace(cache_config=cache, num_hash_lookahead_tokens=0)
    spec = SimpleNamespace(block_size=1152, prefill_checkpoint_alignment=64)
    builder = SimpleNamespace(vllm_config=config, kv_cache_spec=spec, drop_eagle_block=False)
    query_starts = torch.tensor([0, 3400, 6800, 10200], dtype=torch.int32)
    blocks = torch.tensor([[0, 17, 19], [0, 0, 20], [0, 0, 0]], dtype=torch.int32)
    metadata = SimpleNamespace(
        seq_lens_cpu_upper_bound=torch.tensor([3400, 3400, 3400], dtype=torch.int32),
        query_start_loc_cpu=query_starts,
        query_start_loc=query_starts,
        block_table_tensor=blocks,
    )
    policy = GDNCheckpointPolicy(8192, True, True)
    rows = [2, 0, 1]
    fallback = build_gdn_checkpoint_metadata(builder, metadata, rows, policy)
    fallback.checkpoint_offsets.masked_fill_(fallback.state_indices == 0, 0)

    host_blocks = ([0, 17, 19], [0, 0, 20])
    optimized = build_gdn_checkpoint_metadata(builder, metadata, rows, policy, host_blocks=host_blocks)
    assert optimized.offsets == [0, 2304, 0] == fallback.checkpoint_offsets.tolist()
    assert optimized.state_indices.tolist() == fallback.state_indices.tolist() == [0, 17, 0]
    assert optimized.query_lens == fallback.query_lens
    assert build_gdn_checkpoint_metadata(builder, metadata, [1, 2], policy, host_blocks=host_blocks) is None
