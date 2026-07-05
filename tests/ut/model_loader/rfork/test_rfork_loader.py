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
    _get_ep_rank,
    _get_pp_rank,
    _get_rfork_worker_attr,
    _is_draft_model,
    _is_dynamic_eplb_enabled,
    _is_layer_sharding_enabled,
    _make_fallback_load_config,
)
from vllm_ascend.model_loader.rfork.seed_protocol import get_local_seed_key


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


def _parallel_config(
    *,
    enable_eplb=False,
    enable_expert_parallel=False,
    pipeline_parallel_size=1,
    is_moe_model=True,
):
    return SimpleNamespace(
        enable_eplb=enable_eplb,
        enable_expert_parallel=enable_expert_parallel,
        pipeline_parallel_size=pipeline_parallel_size,
        is_moe_model=is_moe_model,
    )


def _vllm_config(model_config=None, scheduler_config=None, parallel_config=None):
    return SimpleNamespace(
        additional_config=None,
        device_config=SimpleNamespace(device="cpu"),
        model_config=model_config or SimpleNamespace(),
        parallel_config=parallel_config or _parallel_config(),
        scheduler_config=scheduler_config or SimpleNamespace(),
    )


def _parallel_vllm_config(
    *,
    enable_expert_parallel=False,
    pipeline_parallel_size=1,
    is_moe_model=True,
):
    return SimpleNamespace(
        parallel_config=_parallel_config(
            enable_expert_parallel=enable_expert_parallel,
            pipeline_parallel_size=pipeline_parallel_size,
            is_moe_model=is_moe_model,
        )
    )


def test_rfork_ep_rank_is_not_added_when_expert_parallel_is_disabled(monkeypatch):
    def fail_if_ep_group_is_accessed():
        pytest.fail("EP group should not be accessed when expert parallelism is disabled.")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ep_group",
        fail_if_ep_group_is_accessed,
    )

    assert _get_ep_rank(_parallel_vllm_config()) is None


def test_rfork_ep_rank_comes_from_ep_group(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ep_group",
        lambda: SimpleNamespace(rank_in_group=7),
    )

    assert _get_ep_rank(_parallel_vllm_config(enable_expert_parallel=True)) == 7


def test_rfork_ep_rank_is_not_added_for_dense_model(monkeypatch):
    def fail_if_ep_group_is_accessed():
        pytest.fail("EP group should not be accessed for a dense model.")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ep_group",
        fail_if_ep_group_is_accessed,
    )

    assert _get_ep_rank(_parallel_vllm_config(enable_expert_parallel=True, is_moe_model=False)) is None


def test_rfork_requires_initialized_ep_group(monkeypatch):
    def raise_uninitialized_ep_group():
        raise AssertionError("expert parallel group is not initialized")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ep_group",
        raise_uninitialized_ep_group,
    )

    with pytest.raises(RuntimeError, match="EP group is not initialized"):
        _get_ep_rank(_parallel_vllm_config(enable_expert_parallel=True))


def test_rfork_pp_rank_is_not_added_when_pipeline_parallelism_is_disabled(monkeypatch):
    def fail_if_pp_group_is_accessed():
        pytest.fail("PP group should not be accessed when pipeline parallelism is disabled.")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_pp_group",
        fail_if_pp_group_is_accessed,
    )

    assert _get_pp_rank(_parallel_vllm_config()) is None


def test_rfork_pp_rank_comes_from_pp_group(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_pp_group",
        lambda: SimpleNamespace(rank_in_group=3),
    )

    assert _get_pp_rank(_parallel_vllm_config(pipeline_parallel_size=2)) == 3


def test_rfork_requires_initialized_pp_group(monkeypatch):
    def raise_uninitialized_pp_group():
        raise AssertionError("pipeline parallel group is not initialized")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_pp_group",
        raise_uninitialized_pp_group,
    )

    with pytest.raises(RuntimeError, match="PP group is not initialized"):
        _get_pp_rank(_parallel_vllm_config(pipeline_parallel_size=2))


def test_rfork_seed_key_preserves_non_ep_format():
    assert (
        get_local_seed_key(
            disaggregation_mode="kv_consumer",
            node_rank=0,
            tp_rank=3,
            model_url="/models/dsv4",
            model_deploy_strategy_name="decode",
        )
        == "/models/dsv4$decode$kv_consumer$0$3"
    )


def test_rfork_seed_key_isolated_by_ep_rank():
    common_config = {
        "disaggregation_mode": "kv_consumer",
        "node_rank": 0,
        "tp_rank": 0,
        "model_url": "/models/dsv4",
        "model_deploy_strategy_name": "decode",
    }

    assert get_local_seed_key(**common_config, ep_rank=0) == "/models/dsv4$decode$kv_consumer$0$0$ep0"
    assert get_local_seed_key(**common_config, ep_rank=1) == "/models/dsv4$decode$kv_consumer$0$0$ep1"


def test_rfork_seed_key_isolated_by_pp_rank():
    common_config = {
        "disaggregation_mode": "kv_consumer",
        "node_rank": 0,
        "tp_rank": 0,
        "ep_rank": 0,
        "model_url": "/models/dsv4",
        "model_deploy_strategy_name": "decode",
    }

    assert get_local_seed_key(**common_config, pp_rank=0) == "/models/dsv4$decode$kv_consumer$0$pp0$0$ep0"
    assert get_local_seed_key(**common_config, pp_rank=1) == "/models/dsv4$decode$kv_consumer$0$pp1$0$ep0"


def test_rfork_seed_key_distinguishes_parallel_rank_types():
    common_config = {
        "disaggregation_mode": "kv_consumer",
        "node_rank": 0,
        "model_url": "/models/dsv4",
        "model_deploy_strategy_name": "decode",
    }

    pp_key = get_local_seed_key(**common_config, pp_rank=3, tp_rank=1)
    ep_key = get_local_seed_key(**common_config, tp_rank=3, ep_rank=1)

    assert pp_key == "/models/dsv4$decode$kv_consumer$0$pp3$1"
    assert ep_key == "/models/dsv4$decode$kv_consumer$0$3$ep1"
    assert pp_key != ep_key


def test_rfork_draft_seed_key_isolated_by_ep_rank():
    assert (
        get_local_seed_key(
            disaggregation_mode="kv_consumer",
            node_rank=0,
            tp_rank=0,
            model_url="/models/dsv4",
            model_deploy_strategy_name="decode",
            is_draft_worker=True,
            ep_rank=5,
        )
        == "/models/dsv4$decode$kv_consumer$0$0$ep5$draft"
    )


def test_rfork_worker_receives_parallel_ranks(monkeypatch):
    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace()
    vllm_config = SimpleNamespace(
        kv_transfer_config=None,
        model_config=model_config,
        scheduler_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(node_rank=2),
    )
    captured = {}
    expected_worker = SimpleNamespace()

    def fake_rfork_worker(**kwargs):
        captured.update(kwargs)
        return expected_worker

    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader.RForkWorker", fake_rfork_worker)
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader._get_pp_rank", lambda config: 3)
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader._get_ep_rank", lambda config: 7)
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader.get_tensor_model_parallel_rank", lambda: 5)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 11)

    worker = loader._ensure_rfork_worker(vllm_config, model_config)

    assert worker is expected_worker
    assert captured["node_rank"] == 2
    assert captured["tp_rank"] == 5
    assert captured["pp_rank"] == 3
    assert captured["ep_rank"] == 7
    assert captured["device_id"] == 11


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


def test_rfork_detects_dynamic_eplb_config():
    assert _is_dynamic_eplb_enabled(
        SimpleNamespace(
            parallel_config=SimpleNamespace(enable_eplb=True),
            additional_config=None,
        )
    )
    assert _is_dynamic_eplb_enabled(
        SimpleNamespace(
            parallel_config=SimpleNamespace(enable_eplb=False),
            additional_config={
                "eplb_config": {
                    "dynamic_eplb": True,
                }
            },
        )
    )
    assert _is_dynamic_eplb_enabled(
        SimpleNamespace(
            parallel_config=SimpleNamespace(enable_eplb=False),
            additional_config={
                "eplb_config": {
                    "expert_map_record_path": "/tmp/expert-map.json",
                }
            },
        )
    )
    assert not _is_dynamic_eplb_enabled(
        SimpleNamespace(
            parallel_config=SimpleNamespace(enable_eplb=False),
            additional_config={"eplb_config": {}},
        )
    )
    assert not _is_dynamic_eplb_enabled(
        SimpleNamespace(
            parallel_config=SimpleNamespace(enable_eplb=False),
            additional_config=None,
        )
    )


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


def test_rfork_dynamic_eplb_uses_default_loader(monkeypatch):
    import vllm.model_executor.model_loader as model_loader

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "tp8"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test")
    vllm_config = _vllm_config(model_config=model_config)
    vllm_config.additional_config = {"eplb_config": {"dynamic_eplb": True}}

    def fail_if_rfork_worker_is_created(*args, **kwargs):
        raise AssertionError("RFork worker should not be initialized when dynamic EPLB is enabled.")

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


def test_rfork_native_eplb_uses_default_loader(monkeypatch):
    import vllm.model_executor.model_loader as model_loader

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "tp8"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test")
    vllm_config = _vllm_config(
        model_config=model_config,
        parallel_config=_parallel_config(enable_eplb=True),
    )
    vllm_config.additional_config = None

    def fail_if_rfork_worker_is_created(*args, **kwargs):
        raise AssertionError("RFork worker should not be initialized when native EPLB is enabled.")

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
