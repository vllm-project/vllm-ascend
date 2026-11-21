# SPDX-License-Identifier: Apache-2.0

import pytest

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
