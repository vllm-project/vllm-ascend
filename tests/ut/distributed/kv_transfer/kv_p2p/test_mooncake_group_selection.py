# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import KVCacheRecvingThread


@pytest.mark.parametrize(
    "group_idx, layer_indices, expected",
    [
        (0, None, ["model.layers.0.attn", "model.layers.1.attn"]),
        (0, [1], ["model.layers.1.attn"]),
        (0, [], []),
        (1, None, ["model.layers.0.indexer"]),
        (1, [120], ["model.layers.0.indexer"]),
        (2, [60], ["eagle.layers.0.attn"]),
        (2, [61], ["mtp.layers.0.attn"]),
        (2, [0], []),
    ],
)
def test_reformat_selects_registered_transfer_group(group_idx, layer_indices, expected):
    # Attention, indexer and draft caches share model layer number zero but
    # have distinct registered transfer indices and must never be mixed.
    receiver = SimpleNamespace(
        kv_group2layeridx={
            0: ({"layer_names": ["model.layers.0.attn", "model.layers.1.attn"]}, [0, 1]),
            1: ({"layer_names": ["model.layers.0.indexer"]}, [120]),
            2: ({"layer_names": ["eagle.layers.0.attn", "mtp.layers.0.attn"]}, [60, 61]),
        },
        kv_caches={
            "model.layers.0.attn": (object(), object()),
            "model.layers.1.attn": (object(), object()),
            "model.layers.0.indexer": (object(),),
            "eagle.layers.0.attn": (object(), object()),
            "mtp.layers.0.attn": (object(), object()),
        },
    )
    result = KVCacheRecvingThread._get_group_kv_caches(receiver, group_idx, layer_indices)
    assert list(result) == expected
    for name in expected:
        assert result[name] is receiver.kv_caches[name]


def test_reformat_ignores_unregistered_local_cache():
    receiver = SimpleNamespace(
        kv_group2layeridx={0: ({"layer_names": ["registered", "absent"]}, [7, 8])},
        kv_caches={"registered": (object(), object())},
    )
    assert KVCacheRecvingThread._get_group_kv_caches(receiver, 0) == receiver.kv_caches
