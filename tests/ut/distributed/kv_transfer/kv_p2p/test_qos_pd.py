# SPDX-License-Identifier: Apache-2.0
"""Installed-stack unit tests for P/D lane routing."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import KVCacheRecvingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.qos import KvQosPolicy


@pytest.mark.parametrize("mapping,expected", [({"0": 0, "2": 7}, 7), ({"0": 7, "2": 0}, 0)])
def test_pd_selects_matching_local_and_remote_lane(mapping, expected):
    receiver = KVCacheRecvingThread.__new__(KVCacheRecvingThread)
    receiver.qos_policy = KvQosPolicy.from_config({"priority_to_qos": mapping})
    receiver.qos_pool = SimpleNamespace(read=Mock(return_value=0))
    receiver.remote_metadata_lock = threading.Lock()
    receiver.remote_qos_te_ports = {"P": {9000: {0: 9100, 7: 9107}}}
    meta = {"kv_priority": 2, "remote_engine_id": "P", "remote_handshake_port": 9000, "remote_host": "p-host"}
    assert receiver._submit_kv_read(meta, "unused:1", [100], [200], [8]) == 0
    receiver.qos_pool.read.assert_called_once_with(expected, f"p-host:{9100 + expected}", [100], [200], [8])
    receiver.remote_qos_te_ports = {}
    with pytest.raises(RuntimeError, match="invalid or mismatched"):
        receiver._submit_kv_read(meta, "unused:1", [100], [200], [8])
