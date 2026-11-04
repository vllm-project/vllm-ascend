"""Full and partial pages retain their destinations when allocation is batched."""

from unittest.mock import Mock

import numpy as np
import pytest

from tests.ut.distributed.ascend_store.test_reuse_snapshot_blocks import worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta


def allocation_case():
    instance = worker()
    instance.num_kv_cache_groups = 2
    instance.grouped_block_size = [128, 256]
    instance.group_uses_align_state = [False, False]
    instance.cacheable_group_ids = {0, 1}
    instance._global_group_alloc_size = lambda group: 4096 * (group + 1)
    instance._make_layerwise_full_key = lambda group, key: f"g{group}:{key}"
    blocks = [[1, 2, 3, 4, 5], [6, 7, 8]]
    request = ReqMeta(
        req_id="request",
        token_len_chunk=513,
        save_start_token=0,
        save_end_token=512,
        target_token_len=513,
        block_ids_by_group=blocks,
        block_ids_by_group_np=[np.asarray(group, dtype=np.int64) for group in blocks],
        block_hashes=["h0", "h1", "h2", "h3"],
        can_save=True,
    )
    return instance, request


def test_one_batch_preserves_group_sizes_and_partial_destinations():
    instance, request = allocation_case()
    instance.m_store.batch_alloc = Mock(wraps=instance.m_store.batch_alloc)
    assert instance._alloc_gvas_for_save_for_request([request]) is not False
    assert instance.m_store.batch_alloc.call_count == 1
    keys, sizes, _ttl = instance.m_store.batch_alloc.call_args.args
    assert len(set(keys)) == len(keys) == 8
    assert sizes == [4096] * 5 + [8192] * 3
    assert request.block_gvas_by_group_np[0].tolist() == [100, 101, 102, 103, 0]
    assert request.block_gvas_by_group_np[1].tolist() == [105, 106, 0]
    assert request.partial_save_gva_per_group == [104, 107]
    assert request.save_keys == keys


@pytest.mark.parametrize("result", [None, [], [0] * 8])
def test_incomplete_or_failed_batch_does_not_publish_targets(result):
    instance, request = allocation_case()
    instance.m_store.batch_alloc = Mock(return_value=result)
    assert instance._alloc_gvas_for_save_for_request([request]) is False
    assert not request.save_keys
