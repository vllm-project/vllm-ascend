"""Tests for worker.bulk.transfer."""

from dataclasses import replace

import numpy as np

from tests.ut.distributed.ascend_store.v1.worker.transfer.fixtures import make_group
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.bulk.transfer import _bulk_load_completions
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.transfer.batch import (
    KVTransferBatch,
)


def test_bulk_failure_uses_position_not_key_identity_and_keeps_empty_request_completion():
    first = replace(make_group(2, (0, 0, 1, 2), 1), key_axes=(("duplicate", "duplicate"),))
    second = make_group(7, (0, 0, 1, 2), 1)
    selection = np.array((False, True), dtype=np.bool_)
    selection.flags.writeable = False
    second = replace(second, selected_objects=selection)
    batch = KVTransferBatch(("empty", "first", "second"), (first, second))
    completions = _bulk_load_completions(batch, (-7, 0, -8))
    assert [completion.request_id for completion in completions] == ["empty", "first", "second"]
    assert completions[0].failures == ()
    assert [(item.result_code, item.source.group_id, item.source.block_id) for item in completions[1].failures] == [
        (-7, 2, 10)
    ]
    assert [(item.result_code, item.source.group_id, item.source.block_id) for item in completions[2].failures] == [
        (-8, 7, 11)
    ]
