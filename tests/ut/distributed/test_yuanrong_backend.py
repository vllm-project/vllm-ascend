# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.yuanrong_backend import (
    YuanrongBackend,
    YuanrongConfig,
)


def _make_backend():
    backend = YuanrongBackend.__new__(YuanrongBackend)
    backend._ds_set_param = object()
    backend.config = SimpleNamespace(get_sub_timeout_ms=1234)
    backend.store = MagicMock()
    backend.store.mget_h2d_from_multi_buffers.return_value = []
    backend.store.mset_d2h_from_multi_buffers.return_value = None
    backend.store.batch_is_exist.return_value = [1, 0]
    return backend


def test_get_and_put_use_multi_buffer_apis():
    backend = _make_backend()
    keys = ["Qwen2.5-7B@pcp0@dcp0@head_or_tp_rank:0@pp_rank:0@abcdef"]
    addrs = [[100, 200]]
    sizes = [[10, 20]]

    assert backend.get(keys, addrs, sizes) == [0]
    backend.put(keys, addrs, sizes)

    backend.store.mget_h2d_from_multi_buffers.assert_called_once_with(keys, addrs, sizes, 1234)
    backend.store.mset_d2h_from_multi_buffers.assert_called_once_with(keys, addrs, sizes, backend._ds_set_param)
    assert backend.store.mget_h2d_from_multi_buffers.call_args.args[0] is keys
    assert backend.store.mset_d2h_from_multi_buffers.call_args.args[0] is keys


def test_get_forwards_empty_keys_to_sdk():
    backend = _make_backend()
    backend.store.mget_h2d_from_multi_buffers.return_value = []
    assert backend.get([], [], []) == []
    backend.store.mget_h2d_from_multi_buffers.assert_called_once_with([], [], [], 1234)


def test_get_marks_failed_keys():
    backend = _make_backend()
    keys = ["k1", "k2", "k3"]
    addrs = [[100], [200], [300]]
    sizes = [[10], [20], [30]]
    backend.store.mget_h2d_from_multi_buffers.return_value = ["k2"]

    assert backend.get(keys, addrs, sizes) == [0, 1, 0]
    backend.store.mget_h2d_from_multi_buffers.assert_called_once_with(keys, addrs, sizes, 1234)


def test_get_returns_none_on_exception():
    backend = _make_backend()
    backend.store.mget_h2d_from_multi_buffers.side_effect = Exception("fail")
    assert backend.get(["k1"], [[100]], [[10]]) is None


def test_put_forwards_empty_keys_to_sdk():
    backend = _make_backend()
    backend.put([], [], [])
    backend.store.mset_d2h_from_multi_buffers.assert_called_once_with([], [], [], backend._ds_set_param)


def test_put_logs_on_exception():
    backend = _make_backend()
    backend.store.mset_d2h_from_multi_buffers.side_effect = Exception("fail")
    backend.put(["k1"], [[100]], [[10]])  # Should log but not raise


def test_exists_returns_native_int_list():
    backend = _make_backend()
    keys = ["Qwen2.5-key0", "Qwen2.5-key1"]
    result = backend.exists(keys)

    assert result == [1, 0]
    backend.store.batch_is_exist.assert_called_once_with(keys)
    assert backend.store.batch_is_exist.call_args.args[0] is keys


def test_exists_forwards_empty_keys_to_sdk():
    backend = _make_backend()
    backend.store.batch_is_exist.return_value = []
    assert backend.exists([]) == []
    backend.store.batch_is_exist.assert_called_once_with([])


def test_exists_exception_returns_zeros():
    backend = _make_backend()
    backend.store.batch_is_exist.side_effect = Exception("fail")
    assert backend.exists(["k1", "k2"]) == [0, 0]


def test_yuanrong_config_loads_from_file(tmp_path):
    cfg_path = tmp_path / "yuanrong.json"
    cfg_path.write_text(
        json.dumps(
            {
                "worker_addr": "127.0.0.1:31501",
                "enable_remote_h2d": False,
                "connect_timeout_ms": 12000,
                "request_timeout_ms": 8000,
                "get_sub_timeout_ms": 3000,
            }
        )
    )

    cfg = YuanrongConfig.from_file(str(cfg_path))

    assert cfg.worker_addr == "127.0.0.1:31501"
    assert cfg.enable_remote_h2d is False
    assert cfg.connect_timeout_ms == 12000
    assert cfg.request_timeout_ms == 8000
    assert cfg.get_sub_timeout_ms == 3000


def test_yuanrong_config_defaults_from_file(tmp_path):
    cfg_path = tmp_path / "yuanrong.json"
    cfg_path.write_text(json.dumps({"worker_addr": "h:1"}))

    cfg = YuanrongConfig.from_file(str(cfg_path))

    assert cfg.enable_remote_h2d is False
    assert cfg.connect_timeout_ms == 9000
    assert cfg.request_timeout_ms == 0
    assert cfg.get_sub_timeout_ms == 0


def test_yuanrong_config_load_from_env_requires_path(monkeypatch):
    monkeypatch.delenv("YR_CONFIG_PATH", raising=False)

    with pytest.raises(ValueError, match="YR_CONFIG_PATH"):
        YuanrongConfig.load_from_env()


def test_backend_forwards_configured_timeouts(tmp_path, monkeypatch):
    cfg_path = tmp_path / "yuanrong.json"
    cfg_path.write_text(
        json.dumps(
            {
                "worker_addr": "127.0.0.1:31501",
                "connect_timeout_ms": 12000,
                "request_timeout_ms": 8000,
                "get_sub_timeout_ms": 3000,
            }
        )
    )
    monkeypatch.setenv("YR_CONFIG_PATH", str(cfg_path))

    native_client = MagicMock()
    native_client.mget_h2d_from_multi_buffers.return_value = []
    hetero_client = MagicMock(return_value=native_client)
    monkeypatch.setattr(sys.modules["yr.datasystem.hetero_client"], "HeteroClient", hetero_client)
    backend_module = sys.modules[YuanrongBackend.__module__]
    monkeypatch.setattr(backend_module, "split_host_port", lambda _: ("127.0.0.1", 31501))

    backend = YuanrongBackend(MagicMock())

    hetero_client.assert_called_once_with(
        "127.0.0.1",
        31501,
        connect_timeout_ms=12000,
        req_timeout_ms=8000,
        enable_remote_h2d=False,
    )
    native_client.init.assert_called_once_with()
    backend.get(["k1"], [[100]], [[10]])
    native_client.mget_h2d_from_multi_buffers.assert_called_once_with(["k1"], [[100]], [[10]], 3000)
