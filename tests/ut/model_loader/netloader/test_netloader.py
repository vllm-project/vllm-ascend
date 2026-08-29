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
from functools import wraps
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from vllm_ascend.model_loader.netloader.netloader import (
    DRAFT_PORT_OFFSET,
    ModelNetLoaderElastic,
    pre_transfer_weight_processing,
)


class DummyDeviceConfig:
    device = "cuda"
    device_type = "cuda"


class DummyParallelConfig:
    tensor_parallel_size = 1
    pipeline_parallel_size = 1
    local_world_size = 1


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


class _DummyElasticServer:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass

    def register_transfer_manifest(self, model):
        pass


def _install_elastic_server(monkeypatch, server_cls=_DummyElasticServer):
    monkeypatch.setattr(
        "vllm_ascend.model_loader.netloader.netloader.ElasticServer",
        server_cls,
    )


def _recording_elastic_server(calls: list[str]):
    class RecordingElasticServer:
        def __init__(self, *args, **kwargs):
            calls.append("server_init")

        def register_transfer_manifest(self, model):
            calls.append("register")

        def start(self):
            calls.append("server_start")

    return RecordingElasticServer


def _capturing_elastic_server(instances: list):
    class CapturingElasticServer:
        def __init__(self, *args, **kwargs):
            instances.append(self)
            self.int8_cache = args[7]
            self.group_name = kwargs.get("group_name")

        def start(self):
            pass

        def register_transfer_manifest(self, model):
            pass

    return CapturingElasticServer


def _patch_dist_barrier(monkeypatch) -> list[str]:
    barrier_calls: list[str] = []
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)

    def _barrier(*args, **kwargs):
        barrier_calls.append("barrier")

    monkeypatch.setattr("torch.distributed.barrier", _barrier)
    return barrier_calls


def _patch_loader_common(monkeypatch):
    ModelNetLoaderElastic._target_elastic_fallback = False
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
    monkeypatch.setattr(
        "vllm_ascend.model_loader.netloader.netloader.cache_processed_layout_transfer_manifest",
        lambda model: 0,
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.netloader.netloader.synchronize_npu",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("vllm.utils.network_utils.get_ip", lambda: "127.0.0.1")
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.find_free_port", lambda: 8888)
    return dummy_model


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


def test_remove_new_static_forward_context_keys_preserves_baseline(monkeypatch):
    class DummyCompilationConfig:
        def __init__(self):
            self.static_forward_context = {}

    class ConfigWithCompilation:
        def __init__(self):
            self.compilation_config = DummyCompilationConfig()

    passed_config = ConfigWithCompilation()
    current_config = ConfigWithCompilation()
    target_layer = object()
    draft_layer = object()
    passed_config.compilation_config.static_forward_context["target.layer"] = target_layer
    current_config.compilation_config.static_forward_context["current.target"] = target_layer
    monkeypatch.setattr(
        "vllm_ascend.model_loader.netloader.netloader.get_current_vllm_config_or_none",
        lambda: current_config,
    )

    snapshots = ModelNetLoaderElastic._snapshot_static_forward_context_keys(passed_config)
    passed_config.compilation_config.static_forward_context["draft.layer"] = draft_layer
    current_config.compilation_config.static_forward_context["current.draft"] = draft_layer

    ModelNetLoaderElastic._remove_new_static_forward_context_keys(passed_config, snapshots)

    assert passed_config.compilation_config.static_forward_context == {"target.layer": target_layer}
    assert current_config.compilation_config.static_forward_context == {"current.target": target_layer}


def test_pre_transfer_weight_processing_unwraps_and_restores_quant_methods():
    import vllm_ascend.ops.fused_moe.fused_moe as fused_moe_module

    class _FakeAscendMoERunner:
        def __init__(self, quant_method):
            self._quant_method = quant_method

    calls = []

    def original_process_weights(*args, **kwargs):
        calls.append("original")

    @wraps(original_process_weights)
    def wrapped_process_weights(*args, **kwargs):
        calls.append("wrapped")
        original_process_weights(*args, **kwargs)

    quant_method = SimpleNamespace(process_weights_after_loading=wrapped_process_weights)
    moe_runner = _FakeAscendMoERunner(quant_method)
    other_layer = SimpleNamespace()

    class _FakeModule:
        def modules(self):
            return iter([self, moe_runner, other_layer])

    fake_module = _FakeModule()

    missing_marker = object()
    real_moe_runner_cls = getattr(fused_moe_module, "AscendMoERunner", missing_marker)
    fused_moe_module.AscendMoERunner = _FakeAscendMoERunner
    try:
        with pre_transfer_weight_processing(fake_module):
            calls.clear()
            quant_method.process_weights_after_loading()
            assert calls == ["original"]

        calls.clear()
        quant_method.process_weights_after_loading()
        assert calls == ["wrapped", "original"]

        calls.clear()
        with pytest.raises(RuntimeError, match="boom"), pre_transfer_weight_processing(fake_module):
            quant_method.process_weights_after_loading()
            assert calls == ["original"]
            raise RuntimeError("boom")

        calls.clear()
        quant_method.process_weights_after_loading()
        assert calls == ["wrapped", "original"]
    finally:
        if real_moe_runner_cls is missing_marker:
            delattr(fused_moe_module, "AscendMoERunner")
        else:
            fused_moe_module.AscendMoERunner = real_moe_runner_cls


@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_load_model_elastic_success(mock_logger, monkeypatch, tmp_path):
    dummy_model = _patch_loader_common(monkeypatch)
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", lambda **kwargs: dummy_model)
    _install_elastic_server(monkeypatch)
    extra = {
        "SOURCE": [{"device_id": 0}],
        "MODEL": "foo",
        "LISTEN_PORT": 5555,
        "OUTPUT_PREFIX": str(tmp_path) + "/output_",
        "INT8_CACHE": "no",
    }
    result = make_loader_with_config(extra).load_model(DummyVllmConfig(), DummyModelConfig())
    assert isinstance(result, nn.Module)
    assert (tmp_path / "output_0.txt").exists()


@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_target_elastic_failure_sets_fallback_flag(mock_logger, monkeypatch):
    dummy_model = _patch_loader_common(monkeypatch)
    ModelNetLoaderElastic._target_elastic_fallback = False
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", lambda **kwargs: None)
    monkeypatch.setattr(
        ModelNetLoaderElastic,
        "revert_to_default",
        lambda self, *args, **kwargs: (dummy_model, False),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic._snapshot_static_forward_context_keys",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.netloader.netloader.ModelNetLoaderElastic._remove_new_static_forward_context_keys",
        lambda *args, **kwargs: None,
    )
    _install_elastic_server(monkeypatch)

    extra = {
        "SOURCE": [{"device_id": 0, "sources": ["127.0.0.1:5000"]}],
        "MODEL": "dummy-model",
        "LISTEN_PORT": 5555,
        "INT8_CACHE": "dram",
    }
    loader = make_loader_with_config(extra)
    loader.load_model(DummyVllmConfig(), DummyModelConfig())

    assert ModelNetLoaderElastic._target_elastic_fallback is True
    ModelNetLoaderElastic._target_elastic_fallback = False


@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_draft_skips_elastic_when_target_fell_back(mock_logger, monkeypatch):
    dummy_model = _patch_loader_common(monkeypatch)
    ModelNetLoaderElastic._target_elastic_fallback = True
    elastic_calls = []

    def capture_elastic_load(**kwargs):
        elastic_calls.append(kwargs)
        return dummy_model

    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", capture_elastic_load)
    monkeypatch.setattr(
        ModelNetLoaderElastic,
        "revert_to_default",
        lambda self, *args, **kwargs: (dummy_model, False),
    )
    _install_elastic_server(monkeypatch)

    extra = {
        "SOURCE": [{"device_id": 0, "sources": ["127.0.0.1:5000"]}],
        "MODEL": "draft-model",
        "LISTEN_PORT": 5555,
        "INT8_CACHE": "dram",
    }
    loader = make_loader_with_config(extra)
    try:
        result = loader.load_model(DummyVllmConfig(), DummyDraftModelConfig())
        assert result is dummy_model
        assert elastic_calls == []
    finally:
        ModelNetLoaderElastic._target_elastic_fallback = False


@pytest.mark.parametrize(
    "int8_cache,expected_order",
    [
        ("no", ["process", "elastic_load"]),
        ("dram", ["elastic_load", "process"]),
    ],
)
@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_elastic_load_process_weights_order_depends_on_int8_cache(mock_logger, monkeypatch, int8_cache, expected_order):
    dummy_model = _patch_loader_common(monkeypatch)
    calls = []

    def capture_process_weights(*args, **kwargs):
        calls.append("process")

    def capture_elastic_load(**kwargs):
        calls.append("elastic_load")
        assert kwargs["int8_cache"] == int8_cache
        return dummy_model

    monkeypatch.setattr(
        "vllm_ascend.model_loader.netloader.netloader.process_weights_after_loading",
        capture_process_weights,
    )
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", capture_elastic_load)
    _install_elastic_server(monkeypatch)

    extra = {
        "SOURCE": [{"device_id": 0, "sources": ["127.0.0.1:5000"]}],
        "MODEL": "dummy-model",
        "LISTEN_PORT": 5555,
        "INT8_CACHE": int8_cache,
    }
    loader = make_loader_with_config(extra)
    loader.load_model(DummyVllmConfig(), DummyModelConfig())

    assert calls == expected_order


@pytest.mark.parametrize(
    "int8_cache,expected_calls",
    [
        ("no", ["process", "server_init", "register", "server_start"]),
        ("dram", ["server_init", "server_start", "process"]),
    ],
)
@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_seed_process_weights_order_depends_on_int8_cache(mock_logger, monkeypatch, int8_cache, expected_calls):
    dummy_model = _patch_loader_common(monkeypatch)
    calls = []

    def capture_process_weights(*args, **kwargs):
        calls.append("process")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.netloader.netloader.process_weights_after_loading",
        capture_process_weights,
    )
    _install_elastic_server(monkeypatch, _recording_elastic_server(calls))
    monkeypatch.setattr(
        ModelNetLoaderElastic,
        "revert_to_default",
        lambda self, *args, **kwargs: (dummy_model, True),
    )

    extra = {
        "SOURCE": None,
        "MODEL": "dummy-model",
        "LISTEN_PORT": 5555,
        "INT8_CACHE": int8_cache,
    }
    loader = make_loader_with_config(extra)
    loader.load_model(DummyVllmConfig(), DummyModelConfig())

    assert calls == expected_calls


@pytest.mark.parametrize("int8_cache", ["dram", "no"])
@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_target_model_barrier_before_draft(mock_logger, monkeypatch, int8_cache):
    dummy_model = _patch_loader_common(monkeypatch)
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", lambda **kwargs: dummy_model)
    _install_elastic_server(monkeypatch)
    barrier_calls = _patch_dist_barrier(monkeypatch)

    extra = {
        "SOURCE": [{"device_id": 0, "sources": ["127.0.0.1:5000"]}],
        "MODEL": "dummy-model",
        "LISTEN_PORT": 5555,
        "INT8_CACHE": int8_cache,
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
    barrier_calls = _patch_dist_barrier(monkeypatch)

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
    _install_elastic_server(monkeypatch)
    barrier_calls = _patch_dist_barrier(monkeypatch)

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
def test_target_model_uses_node_local_barrier(mock_logger, monkeypatch):
    dummy_model = _patch_loader_common(monkeypatch)
    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", lambda **kwargs: dummy_model)
    _install_elastic_server(monkeypatch)

    created_groups: list[list[int]] = []
    barrier_groups: list[Any] = []

    def fake_new_group(ranks=None, **kwargs):
        rank_list = list(ranks)
        created_groups.append(rank_list)
        return f"pg-{rank_list[0]}"

    def fake_barrier(*args, **kwargs):
        barrier_groups.append(kwargs.get("group", args[0] if args else None))

    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 8)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 2)
    monkeypatch.setattr("torch.distributed.new_group", fake_new_group)
    monkeypatch.setattr("torch.distributed.barrier", fake_barrier)

    extra = {
        "SOURCE": [{"device_id": 2, "sources": ["127.0.0.1:5000"]}],
        "MODEL": "dummy-model",
        "LISTEN_PORT": 5555,
        "INT8_CACHE": "no",
    }
    loader = make_loader_with_config(extra)
    vllm_config = DummyVllmConfig()
    vllm_config.parallel_config = DummyParallelConfig()
    vllm_config.parallel_config.local_world_size = 4
    vllm_config.speculative_config = object()
    loader.load_model(vllm_config, DummyModelConfig())

    assert created_groups == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert barrier_groups == ["pg-0"]


@pytest.mark.parametrize(
    "int8_cache,expected_load_cache",
    [
        ("no", "no"),
        ("dram", "hbm"),
    ],
)
@patch("vllm_ascend.model_loader.netloader.netloader.logger")
def test_load_draft_model_port_offset_and_group_name(
    mock_logger, monkeypatch, tmp_path, int8_cache, expected_load_cache
):
    dummy_model = _patch_loader_common(monkeypatch)
    captured = {}
    elastic_server_instances: list[Any] = []

    def capture_elastic_load(**kwargs):
        captured.update(kwargs)
        return dummy_model

    monkeypatch.setattr("vllm_ascend.model_loader.netloader.netloader.elastic_load", capture_elastic_load)
    _install_elastic_server(monkeypatch, _capturing_elastic_server(elastic_server_instances))

    extra = {
        "SOURCE": [{"device_id": 0, "sources": ["127.0.0.1:5000", "10.0.0.1:6000"]}],
        "MODEL": "draft-model",
        "LISTEN_PORT": 5555,
        "OUTPUT_PREFIX": str(tmp_path) + "/output_",
        "INT8_CACHE": int8_cache,
    }
    loader = make_loader_with_config(extra)
    result = loader.load_model(DummyVllmConfig(), DummyDraftModelConfig())

    assert isinstance(result, nn.Module)
    assert loader._draft_elastic_server is elastic_server_instances[0]
    assert loader._draft_elastic_server.int8_cache == expected_load_cache
    assert loader._draft_elastic_server.group_name == "netloader_draft"
    assert not (tmp_path / "output_0.txt").exists()
    assert captured["group_name"] == "netloader_draft"
    assert captured["int8_cache"] == expected_load_cache
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
    _install_elastic_server(monkeypatch)

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
