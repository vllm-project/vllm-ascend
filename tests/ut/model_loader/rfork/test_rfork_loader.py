#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.model_loader.rfork.rfork_loader import (
    RForkModelLoader,
    _get_rfork_worker_attr,
    _is_draft_model,
    _is_layer_sharding_enabled,
    _make_fallback_load_config,
)


class DummyLoadConfig:
    device = None
    load_format = "rfork"

    def __init__(self, model_loader_extra_config):
        self.model_loader_extra_config = model_loader_extra_config


@pytest.mark.parametrize("config_value", [True, False])
def test_rfork_seed_timeout_bool_falls_back_to_env(monkeypatch, config_value):
    monkeypatch.setenv("RFORK_SEED_TIMEOUT_SEC", "7.5")

    loader = RForkModelLoader(
        DummyLoadConfig(
            {
                "rfork_seed_timeout_sec": config_value,
            }
        )
    )

    assert loader.seed_timeout_sec == 7.5


@pytest.mark.parametrize("config_value", [True, False])
def test_rfork_seed_timeout_bool_falls_back_to_default(monkeypatch, config_value):
    monkeypatch.delenv("RFORK_SEED_TIMEOUT_SEC", raising=False)

    loader = RForkModelLoader(
        DummyLoadConfig(
            {
                "rfork_seed_timeout_sec": config_value,
            }
        )
    )

    assert loader.seed_timeout_sec == 5.0


def _vllm_config(model_config=None, scheduler_config=None):
    return SimpleNamespace(
        additional_config=None,
        device_config=SimpleNamespace(device="cpu"),
        model_config=model_config or SimpleNamespace(),
        scheduler_config=scheduler_config or SimpleNamespace(),
    )


@pytest.mark.parametrize(
    "model_config",
    [
        SimpleNamespace(runner_type="draft"),
        SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_mtp")),
        SimpleNamespace(hf_config=SimpleNamespace(architectures=["DeepSeekV4MTPModel"])),
        SimpleNamespace(hf_text_config=SimpleNamespace(architectures=["OpenPanguMTPModel"])),
    ],
)
def test_rfork_detects_draft_model(model_config):
    assert _is_draft_model(_vllm_config(model_config=model_config))


def test_rfork_detects_draft_model_from_scheduler_config():
    scheduler_config = SimpleNamespace(runner_type="draft")

    assert _is_draft_model(_vllm_config(scheduler_config=scheduler_config))


def test_rfork_does_not_treat_target_model_as_draft():
    target_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="deepseek_v4",
            architectures=["DeepSeekV4ForCausalLM"],
        )
    )

    assert not _is_draft_model(_vllm_config(model_config=target_model_config))


def test_rfork_detects_explicit_draft_model_config():
    target_vllm_config = _vllm_config(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v4",
                architectures=["DeepSeekV4ForCausalLM"],
            )
        )
    )
    draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="deepseek_mtp",
            architectures=["DeepSeekV4MTPModel"],
        )
    )

    assert _is_draft_model(target_vllm_config, draft_model_config)


def test_rfork_uses_separate_worker_attr_for_explicit_draft_model_config():
    target_vllm_config = _vllm_config(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v4",
                architectures=["DeepSeekV4ForCausalLM"],
            )
        )
    )
    draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="deepseek_mtp",
            architectures=["DeepSeekV4MTPModel"],
        )
    )

    assert _get_rfork_worker_attr(target_vllm_config, target_vllm_config.model_config) == "rfork_worker"
    assert _get_rfork_worker_attr(target_vllm_config, draft_model_config) == "rfork_draft_worker"


def test_rfork_fallback_load_config_copy_does_not_mutate_original():
    original_extra_config = {"model_url": "model", "model_deploy_strategy_name": "tp8"}
    load_config = DummyLoadConfig(original_extra_config)

    fallback_load_config = _make_fallback_load_config(load_config)

    assert fallback_load_config is not load_config
    assert fallback_load_config.load_format == "auto"
    assert fallback_load_config.model_loader_extra_config == {}
    assert load_config.load_format == "rfork"
    assert load_config.model_loader_extra_config == original_extra_config


def test_rfork_detects_layer_sharding_config():
    assert _is_layer_sharding_enabled(
        SimpleNamespace(
            additional_config={
                "layer_sharding": ["o_proj"],
            }
        )
    )
    assert not _is_layer_sharding_enabled(SimpleNamespace(additional_config={}))
    assert not _is_layer_sharding_enabled(SimpleNamespace(additional_config=None))


def test_rfork_layer_sharding_uses_default_loader(monkeypatch):
    import vllm.model_executor.model_loader as model_loader

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "tp8"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test")
    vllm_config = _vllm_config(model_config=model_config)
    vllm_config.additional_config = {"layer_sharding": ["o_proj"]}

    def fail_if_rfork_worker_is_created(*args, **kwargs):
        raise AssertionError("RFork worker should not be initialized when layer_sharding is enabled.")

    expected_model = SimpleNamespace()
    captured = {}

    def fake_get_model(**kwargs):
        captured.update(kwargs)
        return expected_model

    monkeypatch.setattr(loader, "_ensure_rfork_worker", fail_if_rfork_worker_is_created)
    monkeypatch.setattr(model_loader, "get_model", fake_get_model)

    model = loader.load_model(vllm_config=vllm_config, model_config=model_config)

    assert model is expected_model
    assert captured["vllm_config"] is vllm_config
    assert captured["model_config"] is model_config
    assert captured["prefix"] == ""
    assert captured["load_config"] is not load_config
    assert captured["load_config"].load_format == "auto"
    assert captured["load_config"].model_loader_extra_config == {}
