#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from vllm_ascend.model_loader.netloader.netloader import DRAFT_PORT_OFFSET, ModelNetLoaderElastic


class DummyDeviceConfig:
    device = "cuda"
    device_type = "cuda"


class DummyParallelConfig:
    tensor_parallel_size = 1
    pipeline_parallel_size = 1


class DummyVllmConfig:
    device_config = DummyDeviceConfig()
    parallel_config = DummyParallelConfig()
    additional_config = None
    quant_config = None
    speculative_config: object | None = None


class DummyModelConfig:
    model = "dummy-model"
    dtype = torch.float32
    runner_type: str | None = None


class DummyDraftModelConfig(DummyModelConfig):
    model = "draft-model"
    runner_type = "draft"


@pytest.fixture
def default_load_config():
    class DummyLoadConfig:
        model_loader_extra_config = None
        load_format = "default"

    return DummyLoadConfig()


def make_loader_with_config(extra):
    class DummyLoadConfig:
        model_loader_extra_config = extra
        load_format = "default"

    return ModelNetLoaderElastic(DummyLoadConfig())


def test_init_with_extra_config_file(tmp_path, monkeypatch):
    # Generate test JSON file
    config_content = {
        "SOURCE": [{"device_id": 0}],
        "MODEL": "foo-model",
        "LISTEN_PORT": 5001,
        "INT8_CACHE": "hbm",
        "OUTPUT_PREFIX": str(tmp_path),
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_content))

    dummy_logger = MagicMock()
    monkeypatch.setattr("vllm.logger.logger", dummy_logger)
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.utils.is_valid_path_prefix", lambda x: True)

    extra = {"CONFIG_FILE": str(config_file)}
    loader = make_loader_with_config(extra)
    assert loader.model_path == "foo-model"
    assert loader.source == [{"device_id": 0}]
    assert loader.listen_port == 5001
    assert loader.int8_cache == "hbm"
    assert loader.output_prefix == str(tmp_path)


def test_init_with_extra_config(monkeypatch):
    dummy_logger = MagicMock()
    monkeypatch.setattr("vllm.logger.logger", dummy_logger)
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.utils.is_valid_path_prefix", lambda x: True)

    extra = {
        "SOURCE": [{"device_id": 0}],
        "MODEL": "foo",
        "LISTEN_PORT": "4000",
        "INT8_CACHE": "dram",
        "OUTPUT_PREFIX": "/tmp/",
    }
    loader = make_loader_with_config(extra)
    assert loader.model_path == "foo"
    assert loader.listen_port == 4000
    assert loader.int8_cache == "dram"
    assert loader.output_prefix == "/tmp/"
    assert loader.source == [{"device_id": 0}]


def test_init_with_invalid_config(monkeypatch):
    dummy_logger = MagicMock()
    monkeypatch.setattr("vllm.logger.logger", dummy_logger)
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.utils.is_valid_path_prefix", lambda x: False)
    # c
    extra = {
        "SOURCE": None,
        "MODEL": None,
        "LISTEN_PORT": None,
        "INT8_CACHE": "something",
        "OUTPUT_PREFIX": None,
    }
    loader = make_loader_with_config(extra)
    assert loader.model_path is None
    assert loader.listen_port is None
    assert loader.int8_cache == "no"
    assert loader.output_prefix is None


def test_clear_static_forward_context_clears_current_vllm_config(monkeypatch):
    class DummyCompilationConfig:
        def __init__(self):
            self.static_forward_context = {}

    class ConfigWithCompilation:
        def __init__(self):
            self.compilation_config = DummyCompilationConfig()

    passed_config = ConfigWithCompilation()
    current_config = ConfigWithCompilation()
    passed_config.compilation_config.static_forward_context["passed.layer"] = object()
    current_config.compilation_config.static_forward_context["current.layer"] = object()
    monkeypatch.setattr(
        "vllm_ascend.model_loader.netloader.netloader.get_current_vllm_config",
        lambda: current_config,
    )

    ModelNetLoaderElastic._clear_static_forward_context(passed_config)

    assert passed_config.compilation_config.static_forward_context == {}
    assert current_config.compilation_config.static_forward_context == {}


@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_load_model_elastic_success(mock_logger, monkeypatch, tmp_path):
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)

    class FakeContext:
        def __enter__(self):
            pass

        def __exit__(self, a, b, c):
            pass

    monkeypatch.setattr("torch.device", lambda d: FakeContext())
    # patch deep copy
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.deepcopy", lambda x: x)
    # patch set_default_torch_dtype
    monkeypatch.setattr(
        "vllm_ascend.model_loader.netloader.netloader.set_default_torch_dtype", lambda dtype: FakeContext()
    )
    # patch initialize_model
    dummy_model = MagicMock(spec=nn.Module)
    dummy_model.eval.return_value = dummy_model
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.initialize_model", lambda **kwargs: dummy_model)
    # patch elastic_load
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", lambda **kwargs: dummy_model)
    # patch process_weights_after_loading
    monkeypatch.setattr(
        "vllm_ascend.model_loader.netloader.netloader.process_weights_after_loading", lambda *a, **k: None
    )
    # patch get_ip
    monkeypatch.setattr("vllm.utils.network_utils.get_ip", lambda: "127.0.0.1")
    # patch find_free_port
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.find_free_port", lambda: 8888)

    # patch ElasticServer
    class DummyElasticServer:
        def __init__(*a, **k):
            pass

        def start(self):
            pass

    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.ElasticServer", DummyElasticServer)
    # write output_prefix to the temporary directory
    extra = {
        "SOURCE": [{"device_id": 0}],
        "MODEL": "foo",
        "LISTEN_PORT": 5555,
        "OUTPUT_PREFIX": str(tmp_path) + "/output_",
        "INT8_CACHE": "no",
    }
    loader = make_loader_with_config(extra)
    vllm_config = DummyVllmConfig()
    model_config = DummyModelConfig()
    result = loader.load_model(vllm_config, model_config)
    assert isinstance(result, nn.Module)
    # Check file
    written_file = tmp_path / "output_0.txt"
    assert written_file.exists()


def _patch_loader_common(monkeypatch):
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)

    class FakeContext:
        def __enter__(self):
            pass

        def __exit__(self, a, b, c):
            pass

    monkeypatch.setattr("torch.device", lambda d: FakeContext())
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.deepcopy", lambda x: x)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.netloader.netloader.set_default_torch_dtype", lambda dtype: FakeContext()
    )
    dummy_model = MagicMock(spec=nn.Module)
    dummy_model.eval.return_value = dummy_model
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.initialize_model", lambda **kwargs: dummy_model)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.netloader.netloader.process_weights_after_loading", lambda *a, **k: None
    )
    monkeypatch.setattr("vllm.utils.network_utils.get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.find_free_port", lambda: 8888)
    return dummy_model


@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_target_model_waits_for_all_netloader_ranks_before_draft(mock_logger, monkeypatch):
    dummy_model = _patch_loader_common(monkeypatch)
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", lambda **kwargs: dummy_model)

    class DummyElasticServer:
        def __init__(*a, **k):
            pass

        def start(self):
            pass

    barrier_calls = []
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.ElasticServer", DummyElasticServer)
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.barrier", lambda: barrier_calls.append("barrier"))

    extra = {
        "SOURCE": [{"device_id": 0, "sources": ["127.0.0.1:5000"]}],
        "MODEL": "dummy-model",
        "LISTEN_PORT": 5555,
        "INT8_CACHE": "dram",
    }
    loader = make_loader_with_config(extra)
    vllm_config = DummyVllmConfig()
    vllm_config.speculative_config = object()
    loader.load_model(vllm_config, DummyModelConfig())

    assert barrier_calls == ["barrier"]


@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_failed_target_model_participates_in_barrier_before_error(mock_logger, monkeypatch):
    _patch_loader_common(monkeypatch)
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", lambda **kwargs: None)
    monkeypatch.setattr(
        ModelNetLoaderElastic,
        "revert_to_default",
        lambda self, *args, **kwargs: (None, False),
    )

    barrier_calls = []
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.barrier", lambda: barrier_calls.append("barrier"))

    extra = {
        "SOURCE": [{"device_id": 0, "sources": ["127.0.0.1:5000"]}],
        "MODEL": "dummy-model",
        "LISTEN_PORT": 5555,
        "INT8_CACHE": "dram",
    }
    loader = make_loader_with_config(extra)
    vllm_config = DummyVllmConfig()
    vllm_config.speculative_config = object()

    with pytest.raises(RuntimeError, match="NetLoader elastic loads model fails"):
        loader.load_model(vllm_config, DummyModelConfig())

    assert barrier_calls == ["barrier"]


@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_draft_model_does_not_wait_for_target_netloader_barrier(mock_logger, monkeypatch):
    dummy_model = _patch_loader_common(monkeypatch)
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", lambda **kwargs: dummy_model)

    class DummyElasticServer:
        def __init__(*a, **k):
            pass

        def start(self):
            pass

    barrier_calls = []
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.ElasticServer", DummyElasticServer)
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.barrier", lambda: barrier_calls.append("barrier"))

    extra = {
        "SOURCE": [{"device_id": 0, "sources": ["127.0.0.1:5000"]}],
        "MODEL": "draft-model",
        "LISTEN_PORT": 5555,
        "INT8_CACHE": "dram",
    }
    loader = make_loader_with_config(extra)
    vllm_config = DummyVllmConfig()
    vllm_config.speculative_config = object()
    loader.load_model(vllm_config, DummyDraftModelConfig())

    assert barrier_calls == []


@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_load_draft_model_elastic_success(mock_logger, monkeypatch, tmp_path):
    dummy_model = _patch_loader_common(monkeypatch)
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", lambda **kwargs: dummy_model)

    elastic_server_instances = []

    class DummyElasticServer:
        def __init__(self, *args, **kwargs):
            elastic_server_instances.append(self)
            self.int8_cache = args[7]
            self.group_name = kwargs.get("group_name")

        def start(self):
            pass

    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.ElasticServer", DummyElasticServer)

    extra = {
        "SOURCE": [{"device_id": 0, "sources": ["127.0.0.1:5000"]}],
        "MODEL": "draft-model",
        "LISTEN_PORT": 5555,
        "OUTPUT_PREFIX": str(tmp_path) + "/output_",
        "INT8_CACHE": "no",
    }
    loader = make_loader_with_config(extra)
    result = loader.load_model(DummyVllmConfig(), DummyDraftModelConfig())

    assert isinstance(result, nn.Module)
    assert loader._draft_elastic_server is elastic_server_instances[0]
    assert loader._draft_elastic_server.int8_cache == "no"
    assert loader._draft_elastic_server.group_name == "netloader_draft"
    assert not (tmp_path / "output_0.txt").exists()


@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_load_draft_model_uses_hbm_when_int8_cache_is_dram(mock_logger, monkeypatch):
    dummy_model = _patch_loader_common(monkeypatch)
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", lambda **kwargs: dummy_model)

    captured = {}

    class DummyElasticServer:
        def __init__(self, *args, **kwargs):
            captured["int8_cache"] = args[7]
            captured["group_name"] = kwargs.get("group_name")

        def start(self):
            pass

    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.ElasticServer", DummyElasticServer)

    extra = {
        "SOURCE": [{"device_id": 0, "sources": ["127.0.0.1:5000"]}],
        "MODEL": "draft-model",
        "LISTEN_PORT": 5555,
        "INT8_CACHE": "dram",
    }
    loader = make_loader_with_config(extra)
    loader.load_model(DummyVllmConfig(), DummyDraftModelConfig())

    assert captured["int8_cache"] == "hbm"
    assert captured["group_name"] == "netloader_draft"


@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_load_draft_model_port_offset_and_group_name(mock_logger, monkeypatch, tmp_path):
    dummy_model = _patch_loader_common(monkeypatch)
    captured = {}

    def capture_elastic_load(**kwargs):
        captured.update(kwargs)
        return dummy_model

    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", capture_elastic_load)

    class DummyElasticServer:
        def __init__(*a, **k):
            pass

        def start(self):
            pass

    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.ElasticServer", DummyElasticServer)

    extra = {
        "SOURCE": [{"device_id": 0, "sources": ["127.0.0.1:5000", "10.0.0.1:6000"]}],
        "MODEL": "draft-model",
        "LISTEN_PORT": 5555,
        "INT8_CACHE": "no",
    }
    loader = make_loader_with_config(extra)
    loader.load_model(DummyVllmConfig(), DummyDraftModelConfig())

    assert captured["group_name"] == "netloader_draft"
    assert captured["model_path"] == "draft-model"
    assert captured["sources"] == [
        {
            "device_id": 0,
            "sources": [
                f"127.0.0.1:{5000 + DRAFT_PORT_OFFSET}",
                f"10.0.0.1:{6000 + DRAFT_PORT_OFFSET}",
            ],
        }
    ]
    assert loader.listen_port == 5555 + DRAFT_PORT_OFFSET


@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_load_draft_model_skips_invalid_source_addresses(mock_logger, monkeypatch):
    dummy_model = _patch_loader_common(monkeypatch)
    captured = {}

    def capture_elastic_load(**kwargs):
        captured.update(kwargs)
        return dummy_model

    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", capture_elastic_load)

    class DummyElasticServer:
        def __init__(*a, **k):
            pass

        def start(self):
            pass

    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.ElasticServer", DummyElasticServer)

    extra = {
        "SOURCE": [{"device_id": 0, "sources": ["127.0.0.1:5000", "invalid", "10.0.0.1:not_port"]}],
        "MODEL": "draft-model",
        "LISTEN_PORT": 5555,
        "INT8_CACHE": "no",
    }
    loader = make_loader_with_config(extra)
    loader.load_model(DummyVllmConfig(), DummyDraftModelConfig())

    assert captured["sources"] == [
        {"device_id": 0, "sources": [f"127.0.0.1:{5000 + DRAFT_PORT_OFFSET}"]},
    ]


if __name__ == "__main__":
    pytest.main()
