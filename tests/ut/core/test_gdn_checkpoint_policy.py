# SPDX-License-Identifier: Apache-2.0

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
