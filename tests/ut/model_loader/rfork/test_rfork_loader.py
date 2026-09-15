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

from contextlib import nullcontext
from functools import wraps
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.model_loader.rfork.rfork_loader import (
    RForkModelLoader,
    _get_ep_rank,
    _get_pp_rank,
    _get_rfork_session_attr,
    _is_draft_model,
    _is_dynamic_eplb_enabled,
    _make_fallback_load_config,
    _reset_process_global_model_state,
    _rfork_pre_transfer_weight_processing,
    _rfork_skip_unquantized_moe_post_load_processing,
    _start_rfork_seed_service,
)
from vllm_ascend.model_loader.rfork.types import (
    RForkFallbackCleanupResult,
    RForkSeedServiceStartResult,
)


class DummyLoadConfig:
    device = None
    load_format = "rfork"
    rfork_session: object | None = None

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

    assert loader.rfork_config.seed_timeout_sec == 7.5


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

    assert loader.rfork_config.seed_timeout_sec == 5.0


def test_rfork_environment_values_are_used_as_fallbacks(monkeypatch):
    monkeypatch.setenv("MODEL_URL", "env-model")
    monkeypatch.setenv("RFORK_REQUEST_TIMEOUT_SEC", "4.0")

    loader = RForkModelLoader(DummyLoadConfig({}))

    assert loader.rfork_config.model_url == "env-model"
    assert loader.rfork_config.request_timeout_sec == 4.0


def test_rfork_explicit_config_takes_priority_over_environment(monkeypatch):
    monkeypatch.setenv("MODEL_URL", "env-model")
    monkeypatch.setenv("MODEL_DEPLOY_STRATEGY_NAME", "env-strategy")
    monkeypatch.setenv("RFORK_SCHEDULER_URL", "http://env-planner")
    monkeypatch.setenv("RFORK_SEED_TIMEOUT_SEC", "7.0")
    monkeypatch.setenv("RFORK_REQUEST_TIMEOUT_SEC", "8.0")
    monkeypatch.setenv("RFORK_SEED_BIND_HOST", "env-host")
    monkeypatch.setenv("RFORK_SEED_ADVERTISE_HOST", "env-advertise")

    loader = RForkModelLoader(
        DummyLoadConfig(
            {
                "model_url": "config-model",
                "model_deploy_strategy_name": "config-strategy",
                "rfork_scheduler_url": "http://config-planner",
                "rfork_seed_timeout_sec": 1.0,
                "rfork_request_timeout_sec": 2.0,
                "rfork_seed_bind_host": "config-host",
                "rfork_seed_advertise_host": "config-advertise",
            }
        )
    )

    assert loader.rfork_config.model_url == "config-model"
    assert loader.rfork_config.model_deploy_strategy_name == "config-strategy"
    assert loader.rfork_config.planner_url == "http://config-planner"
    assert loader.rfork_config.seed_timeout_sec == 1.0
    assert loader.rfork_config.request_timeout_sec == 2.0
    assert loader.rfork_config.seed_bind_host == "config-host"
    assert loader.rfork_config.seed_advertise_host == "config-advertise"


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


def test_rfork_session_receives_parallel_ranks(monkeypatch):
    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace()
    vllm_config = SimpleNamespace(
        model_config=model_config,
        scheduler_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(),
    )
    captured = {}
    expected_session = SimpleNamespace()

    def fake_rfork_session(config, identity):
        captured["config"] = config
        captured["identity"] = identity
        return expected_session

    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader.RForkSession", fake_rfork_session)
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader._get_pp_rank", lambda config: 3)
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader._get_ep_rank", lambda config: 7)
    monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader.get_tensor_model_parallel_rank", lambda: 5)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 11)

    session = loader._ensure_rfork_session(vllm_config, model_config)

    assert session is expected_session
    assert captured["identity"].tp_rank == 5
    assert captured["identity"].pp_rank == 3
    assert captured["identity"].ep_rank == 7
    assert captured["identity"].global_rank == 11


def test_rfork_target_registered_blocks_not_collected_for_target_model():
    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    load_config.rfork_session = SimpleNamespace(
        transfer_backend=SimpleNamespace(registered_weight_blocks=[(128, 4096)])
    )
    target_model_config = SimpleNamespace()
    vllm_config = _vllm_config(model_config=target_model_config)

    assert loader._get_target_registered_blocks(vllm_config, target_model_config) == []


def test_rfork_target_registered_blocks_collected_for_draft_model():
    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5_mtp"))
    target_blocks = [(128, 4096), (8192, 1024)]
    load_config.rfork_session = SimpleNamespace(
        transfer_backend=SimpleNamespace(registered_weight_blocks=target_blocks)
    )
    vllm_config = _vllm_config(model_config=draft_model_config)

    blocks = loader._get_target_registered_blocks(vllm_config, draft_model_config)

    assert blocks == target_blocks
    assert blocks is not target_blocks


def test_rfork_draft_load_passes_target_registered_blocks_to_session(monkeypatch):
    import vllm.model_executor.model_loader as model_loader

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    draft_model_config = SimpleNamespace(
        dtype=torch.float32,
        model="/models/test",
        hf_config=SimpleNamespace(model_type="qwen3_5_mtp"),
    )
    vllm_config = _vllm_config(model_config=draft_model_config)
    target_blocks = [(128, 4096)]
    load_config.rfork_session = SimpleNamespace(
        transfer_backend=SimpleNamespace(registered_weight_blocks=target_blocks)
    )
    captured_blocks = []
    events = []

    class _DraftSession:
        def acquire_seed(self):
            events.append("seed")
            return True

        def transfer_from_seed(self, model, processed_layout, exclude_blocks):
            events.append("transfer")
            captured_blocks.append(list(exclude_blocks))
            return True

        def start_seed_service(self, model, processed_layout, exclude_blocks=None):
            events.append("start_seed_service")
            captured_blocks.append(list(exclude_blocks or []))
            return True

        def prepare_for_fallback(self):
            return True

    draft_session = _DraftSession()
    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda vc, mc: draft_session)
    monkeypatch.setattr(loader, "_requires_processed_layout_transfer", lambda mc: False)

    class _Model:
        def eval(self):
            return self

    expected_model = _Model()

    def fake_get_model(**kwargs):
        return expected_model

    monkeypatch.setattr(model_loader, "get_model", fake_get_model)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.initialize_model",
        lambda **kwargs: expected_model,
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.process_weights_after_loading",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader._rfork_skip_unquantized_moe_post_load_processing",
        lambda model: nullcontext(),
    )

    model = loader.load_model(vllm_config=vllm_config, model_config=draft_model_config)

    assert model is expected_model
    assert captured_blocks == [target_blocks, target_blocks]


@pytest.mark.parametrize("processed_layout", [False, True])
def test_rfork_acquires_seed_after_model_preparation(monkeypatch, processed_layout):
    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(
        dtype=torch.float32,
        model="/models/test",
        quantization="ascend" if processed_layout else None,
    )
    vllm_config = _vllm_config(model_config=model_config)
    events = []

    class _Model(torch.nn.Module):
        pass

    model = _Model()

    class _Session:
        def acquire_seed(self):
            events.append("acquire")
            return True

        def transfer_from_seed(self, model, processed_layout, exclude_blocks=None):
            events.append("transfer")
            return True

        def start_seed_service(self, model, processed_layout, exclude_blocks=None):
            events.append("start_seed_service")
            return True

    session = _Session()
    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda vc, mc: session)
    monkeypatch.setattr(loader, "_requires_processed_layout_transfer", lambda mc: processed_layout)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.initialize_model",
        lambda **kwargs: (events.append("initialize") or model),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.process_weights_after_loading",
        lambda *args, **kwargs: events.append("layout" if processed_layout else "post_load"),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader._rfork_skip_unquantized_moe_post_load_processing",
        lambda model: nullcontext(),
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda: events.append("synchronize"), empty_cache=lambda: None),
        raising=False,
    )

    assert loader.load_model(vllm_config=vllm_config, model_config=model_config) is model

    if processed_layout:
        assert events[:5] == ["initialize", "layout", "synchronize", "acquire", "transfer"]
    else:
        assert events[:4] == ["initialize", "acquire", "transfer", "post_load"]


@pytest.mark.parametrize("failure_stage", ["initialize", "layout"])
def test_rfork_model_preparation_failure_does_not_acquire_seed(monkeypatch, failure_stage):
    import vllm.model_executor.model_loader as model_loader

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(
        dtype=torch.float32,
        model="/models/test",
        quantization=failure_stage == "layout",
    )
    vllm_config = _vllm_config(model_config=model_config)
    fallback_model = torch.nn.Module()
    acquire_calls = []
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None),
        raising=False,
    )

    class _Session:
        def acquire_seed(self):
            acquire_calls.append(True)
            raise AssertionError("seed acquisition must happen after model preparation")

        def prepare_for_fallback(self):
            return True

        def start_seed_service(self, model, processed_layout, exclude_blocks=None):
            return True

    session = _Session()
    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda vc, mc: session)
    monkeypatch.setattr(loader, "_requires_processed_layout_transfer", lambda mc: failure_stage == "layout")
    monkeypatch.setattr(model_loader, "get_model", lambda **kwargs: fallback_model)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )

    if failure_stage == "initialize":

        def fail_initialize(**kwargs):
            raise RuntimeError("initialize failed")

        monkeypatch.setattr("vllm_ascend.model_loader.rfork.rfork_loader.initialize_model", fail_initialize)
    else:
        model = torch.nn.Module()
        monkeypatch.setattr(
            "vllm_ascend.model_loader.rfork.rfork_loader.initialize_model",
            lambda **kwargs: model,
        )

        def fail_layout(*args, **kwargs):
            raise RuntimeError("layout failed")

        monkeypatch.setattr(
            "vllm_ascend.model_loader.rfork.rfork_loader.process_weights_after_loading",
            fail_layout,
        )

    assert loader.load_model(vllm_config=vllm_config, model_config=model_config) is fallback_model
    assert acquire_calls == []


@pytest.mark.parametrize(
    ("quantization", "weight_nz_mode", "hardware_policy", "expected"),
    [
        ("ascend", 0, "CONFIGURABLE", True),
        (None, 2, "CONFIGURABLE", True),
        (None, 0, "FORCE_NZ", True),
        (None, 0, "CONFIGURABLE", False),
    ],
)
def test_rfork_processed_layout_covers_quantization_and_nz_modes(
    monkeypatch, quantization, weight_nz_mode, hardware_policy, expected
):
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(weight_nz_mode=weight_nz_mode),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_current_hardware_profile",
        lambda: SimpleNamespace(weight_layout_policy=SimpleNamespace(name=hardware_policy)),
    )
    loader = RForkModelLoader(DummyLoadConfig({}))

    assert loader._requires_processed_layout_transfer(SimpleNamespace(quantization=quantization)) is expected


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


def test_rfork_uses_separate_session_attr_for_explicit_draft_model_config():
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

    assert _get_rfork_session_attr(target_vllm_config, target_vllm_config.model_config) == "rfork_session"
    assert _get_rfork_session_attr(target_vllm_config, draft_model_config) == "rfork_draft_session"


def test_rfork_fallback_load_config_copy_does_not_mutate_original():
    original_extra_config = {"model_url": "model", "model_deploy_strategy_name": "tp8"}
    load_config = DummyLoadConfig(original_extra_config)

    fallback_load_config = _make_fallback_load_config(load_config)

    assert fallback_load_config is not load_config
    assert fallback_load_config.load_format == "auto"
    assert fallback_load_config.model_loader_extra_config == {}
    assert load_config.load_format == "rfork"
    assert load_config.model_loader_extra_config == original_extra_config


def test_rfork_detects_dynamic_eplb_config(monkeypatch):
    # Native Model Runner V2 EPLB is represented by ParallelConfig and does
    # not require the AscendConfig singleton.

    def fail_singleton_read():
        raise AssertionError("singleton should not be read")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        fail_singleton_read,
    )
    assert _is_dynamic_eplb_enabled(
        SimpleNamespace(
            parallel_config=SimpleNamespace(enable_eplb=True),
            additional_config=None,
        )
    )

    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=False),
        # A conflicting raw value verifies that RFork consumes only the typed
        # singleton after AscendConfig initialization.
        additional_config={"eplb_config": {"dynamic_eplb": False}},
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=True, expert_map_record_path=None)),
    )
    assert _is_dynamic_eplb_enabled(vllm_config)

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(
            eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path="/tmp/expert-map.json")
        ),
    )
    assert _is_dynamic_eplb_enabled(vllm_config)

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    assert not _is_dynamic_eplb_enabled(vllm_config)


def test_rfork_dynamic_eplb_uses_default_loader(monkeypatch):
    import vllm.model_executor.model_loader as model_loader

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "tp8"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test")
    vllm_config = _vllm_config(model_config=model_config)
    vllm_config.additional_config = {"eplb_config": {"dynamic_eplb": True}}

    def fail_if_rfork_session_is_created(*args, **kwargs):
        raise AssertionError("RFork session should not be initialized when dynamic EPLB is enabled.")

    expected_model = SimpleNamespace()
    captured = {}

    def fake_get_model(**kwargs):
        captured.update(kwargs)
        return expected_model

    monkeypatch.setattr(loader, "_ensure_rfork_session", fail_if_rfork_session_is_created)
    monkeypatch.setattr(model_loader, "get_model", fake_get_model)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=True, expert_map_record_path=None)),
    )

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

    def fail_if_rfork_session_is_created(*args, **kwargs):
        raise AssertionError("RFork session should not be initialized when native EPLB is enabled.")

    expected_model = SimpleNamespace()
    captured = {}

    def fake_get_model(**kwargs):
        captured.update(kwargs)
        return expected_model

    monkeypatch.setattr(loader, "_ensure_rfork_session", fail_if_rfork_session_is_created)
    monkeypatch.setattr(model_loader, "get_model", fake_get_model)

    model = loader.load_model(vllm_config=vllm_config, model_config=model_config)

    assert model is expected_model
    assert captured["vllm_config"] is vllm_config
    assert captured["model_config"] is model_config
    assert captured["prefix"] == ""
    assert captured["load_config"] is not load_config
    assert captured["load_config"].load_format == "auto"
    assert captured["load_config"].model_loader_extra_config == {}


def test_rfork_session_construction_failure_falls_back_without_starting_seed(monkeypatch):
    import vllm.model_executor.model_loader as model_loader

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test", quantization=None)
    vllm_config = _vllm_config(model_config=model_config)
    expected_model = SimpleNamespace()
    start_calls = []

    def fail_session(*args, **kwargs):
        raise RuntimeError("TransferEngine unavailable")

    monkeypatch.setattr(loader, "_ensure_rfork_session", fail_session)
    monkeypatch.setattr(model_loader, "get_model", lambda **kwargs: expected_model)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )

    class _UnexpectedSession:
        def start_seed_service(self, *args, **kwargs):
            start_calls.append((args, kwargs))

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.RForkSession",
        lambda *args, **kwargs: _UnexpectedSession(),
    )

    assert loader.load_model(vllm_config=vllm_config, model_config=model_config) is expected_model
    assert start_calls == []


def test_rfork_seed_start_failure_returns_valid_model_without_disk_reload(monkeypatch):
    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test", quantization=None)
    vllm_config = _vllm_config(model_config=model_config)
    events = []
    warnings = []

    class _Model(torch.nn.Module):
        def eval(self):
            events.append("eval")
            return super().eval()

    model = _Model()

    class _Session:
        def acquire_seed(self):
            events.append("seed")
            return True

        def transfer_from_seed(self, model, processed_layout, exclude_blocks=None):
            events.append("transfer")
            return True

        def start_seed_service(self, model, processed_layout, exclude_blocks=None):
            events.append("start_seed_service")
            return RForkSeedServiceStartResult.FAILED

        def prepare_for_fallback(self):
            return True

    session = _Session()
    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda vc, mc: session)
    monkeypatch.setattr(loader, "_requires_processed_layout_transfer", lambda mc: False)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.initialize_model",
        lambda **kwargs: model,
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.process_weights_after_loading",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader._rfork_skip_unquantized_moe_post_load_processing",
        lambda model: nullcontext(),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.logger.warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    result = loader.load_model(vllm_config=vllm_config, model_config=model_config)

    assert result is model
    assert events.index("eval") < events.index("start_seed_service")
    assert any("not currently advertised as a seed" in args[0] for args in warnings)


def test_rfork_fallback_seed_is_deferred_when_only_lease_release_is_pending(monkeypatch):
    import vllm.model_executor.model_loader as model_loader

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "strategy"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test", quantization=None)
    vllm_config = _vllm_config(model_config=model_config)
    rfork_model = torch.nn.Module()
    fallback_model = torch.nn.Module()
    seed_start_models = []

    class _Session:
        def acquire_seed(self):
            return True

        def transfer_from_seed(self, model, processed_layout, exclude_blocks=None):
            return False

        def prepare_for_fallback(self):
            return RForkFallbackCleanupResult(service_stopped=True, lease_released=False, memory_reset=True)

        def start_seed_service(self, model, processed_layout, exclude_blocks=None):
            seed_start_models.append(model)
            return RForkSeedServiceStartResult.DEFERRED

    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda vc, mc: _Session())
    monkeypatch.setattr(loader, "_requires_processed_layout_transfer", lambda mc: False)
    monkeypatch.setattr(model_loader, "get_model", lambda **kwargs: fallback_model)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.initialize_model",
        lambda **kwargs: rfork_model,
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.process_weights_after_loading",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader._rfork_skip_unquantized_moe_post_load_processing",
        lambda model: nullcontext(),
    )

    assert loader.load_model(vllm_config=vllm_config, model_config=model_config) is fallback_model
    assert seed_start_models == [fallback_model]


def test_rfork_seed_start_exception_does_not_escape(monkeypatch):
    exceptions = []

    class _Session:
        identity = SimpleNamespace(is_draft_model=False)

        def start_seed_service(self, *args, **kwargs):
            raise RuntimeError("seed server failed")

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.logger.exception",
        lambda *args, **kwargs: exceptions.append(args),
    )

    assert not _start_rfork_seed_service(
        _Session(),  # type: ignore[arg-type]
        object(),
        False,
        [],
        load_source="fallback",
    )
    assert any("seed service startup raised" in args[0] for args in exceptions)


def test_rfork_fallback_clears_only_failed_model_state_before_reinit(monkeypatch):
    """Fallback re-init in the same process must first clear stale layer registries."""
    import vllm.model_executor.layers.rotary_embedding as rotary_embedding
    import vllm.model_executor.model_loader as model_loader

    monkeypatch.setattr(rotary_embedding, "_ROPE_DICT", {})
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "tp8"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test", quantization="ascend")
    vllm_config = _vllm_config(model_config=model_config)

    class _FakeModule:
        pass

    stale_attention = _FakeModule()
    stale_moe = _FakeModule()
    unrelated_layer = _FakeModule()
    fallback_down_proj = _FakeModule()
    vllm_config.compilation_config = SimpleNamespace(
        static_forward_context={"unrelated.layer": unrelated_layer},
        static_all_moe_layers=["unrelated.layer"],
    )
    rope_key = ("identity", 1.0, 32768)
    rope_value = object()
    _ROPE_DICT[rope_key] = rope_value

    class _DiscardedModel:
        def modules(self):
            return iter([self, stale_attention, stale_moe])

    rfork_model = _DiscardedModel()
    expected_model = SimpleNamespace()
    get_model_calls = []

    def fake_get_model(**kwargs):
        get_model_calls.append(kwargs)
        assert vllm_config.compilation_config.static_forward_context == {
            "unrelated.layer": unrelated_layer,
        }
        assert vllm_config.compilation_config.static_all_moe_layers == ["unrelated.layer"]
        assert {rope_key: rope_value} == _ROPE_DICT
        vllm_config.compilation_config.static_forward_context["model.layers.0.mlp.down_proj"] = fallback_down_proj
        return expected_model

    rfork_session = SimpleNamespace(
        acquire_seed=lambda: True,
        transfer_from_seed=lambda model, processed_layout, exclude_blocks=None: False,
        prepare_for_fallback=lambda: True,
        start_seed_service=lambda model, processed_layout, exclude_blocks=None: True,
    )

    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda vc, mc: rfork_session)
    monkeypatch.setattr(model_loader, "get_model", fake_get_model)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )

    def fake_initialize_model(**kwargs):
        vllm_config.compilation_config.static_forward_context.update(
            {
                "model.layers.0.self_attn.indexer.k_cache": stale_attention,
            }
        )
        vllm_config.compilation_config.static_all_moe_layers.extend(
            [stale_moe, "model.layers.0.self_attn.indexer.k_cache"]
        )
        return rfork_model

    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.initialize_model",
        fake_initialize_model,
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.process_weights_after_loading",
        lambda *args, **kwargs: None,
    )

    model = loader.load_model(vllm_config=vllm_config, model_config=model_config)

    assert model is expected_model
    assert len(get_model_calls) == 1
    assert vllm_config.compilation_config.static_forward_context == {
        "unrelated.layer": unrelated_layer,
        "model.layers.0.mlp.down_proj": fallback_down_proj,
    }
    assert vllm_config.compilation_config.static_all_moe_layers == ["unrelated.layer"]
    assert {rope_key: rope_value} == _ROPE_DICT


def test_rfork_seed_miss_fallback_preserves_existing_process_global_state(monkeypatch):
    import vllm.model_executor.layers.rotary_embedding as rotary_embedding
    import vllm.model_executor.model_loader as model_loader

    monkeypatch.setattr(rotary_embedding, "_ROPE_DICT", {})
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

    load_config = DummyLoadConfig({"model_url": "model", "model_deploy_strategy_name": "tp8"})
    loader = RForkModelLoader(load_config)
    model_config = SimpleNamespace(dtype=torch.float32, model="/models/test", quantization="ascend")
    vllm_config = _vllm_config(model_config=model_config)
    existing_layer = SimpleNamespace()
    vllm_config.compilation_config = SimpleNamespace(
        static_forward_context={"existing.layer": existing_layer},
        static_all_moe_layers=["existing.layer"],
    )
    rope_key = ("identity", 1.0, 32768)
    rope_value = object()
    _ROPE_DICT[rope_key] = rope_value

    existing_rope = _ROPE_DICT[rope_key]

    class _DiscardedModel:
        def modules(self):
            return iter([self])

    rfork_model = _DiscardedModel()
    expected_model = SimpleNamespace()

    def fake_get_model(**kwargs):
        assert vllm_config.compilation_config.static_forward_context == {
            "existing.layer": existing_layer,
        }
        assert vllm_config.compilation_config.static_all_moe_layers == ["existing.layer"]
        assert _ROPE_DICT[rope_key] is existing_rope
        return expected_model

    rfork_session = SimpleNamespace(
        acquire_seed=lambda: False,
        prepare_for_fallback=lambda: True,
        start_seed_service=lambda model, processed_layout, exclude_blocks=None: True,
    )

    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda vc, mc: rfork_session)
    monkeypatch.setattr(model_loader, "get_model", fake_get_model)
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.initialize_model",
        lambda **kwargs: rfork_model,
    )
    monkeypatch.setattr(
        "vllm_ascend.model_loader.rfork.rfork_loader.process_weights_after_loading",
        lambda *args, **kwargs: _ROPE_DICT.__setitem__(("late", 2.0, 65536), object()),
    )

    model = loader.load_model(vllm_config=vllm_config, model_config=model_config)

    assert model is expected_model
    assert vllm_config.compilation_config.static_forward_context == {
        "existing.layer": existing_layer,
    }
    assert vllm_config.compilation_config.static_all_moe_layers == ["existing.layer"]
    assert _ROPE_DICT[rope_key] is rope_value
    assert ("late", 2.0, 65536) not in _ROPE_DICT


def test_reset_process_global_model_state_is_safe_when_attrs_missing():
    vllm_config = SimpleNamespace(compilation_config=SimpleNamespace())
    _reset_process_global_model_state(vllm_config)


def test_rfork_pre_transfer_weight_processing_unwraps_and_restores_quant_methods(monkeypatch):
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
    fused_moe_layer = _FakeAscendMoERunner(quant_method)
    other_layer = SimpleNamespace()

    class _FakeModule:
        def modules(self):
            return iter([self, fused_moe_layer, other_layer])

    fake_module = _FakeModule()
    monkeypatch.setattr(fused_moe_module, "AscendMoERunner", _FakeAscendMoERunner)

    with _rfork_pre_transfer_weight_processing(fake_module):
        assert quant_method.process_weights_after_loading is original_process_weights
        quant_method.process_weights_after_loading()
    assert quant_method.process_weights_after_loading is wrapped_process_weights
    assert calls == ["original"]

    # Restoration must happen even when the wrapped block raises.
    with pytest.raises(RuntimeError, match="boom"), _rfork_pre_transfer_weight_processing(fake_module):
        assert quant_method.process_weights_after_loading is original_process_weights
        raise RuntimeError("boom")
    assert quant_method.process_weights_after_loading is wrapped_process_weights


def test_rfork_skips_only_unquantized_moe_post_load_processing(monkeypatch):
    import vllm_ascend.ops.fused_moe.fused_moe as fused_moe_module
    import vllm_ascend.ops.fused_moe.routed_experts as routed_experts_module

    class _FakeAscendUnquantizedFusedMoEMethod:
        def __init__(self, process_weights_after_loading):
            self.process_weights_after_loading = process_weights_after_loading

    class _FakeAscendMoERunner:
        def __init__(self, quant_method):
            self._quant_method = quant_method

    calls = []

    def unquantized_process(*args, **kwargs):
        calls.append("unquantized")

    def quantized_process(*args, **kwargs):
        calls.append("quantized")

    unquantized_method = _FakeAscendUnquantizedFusedMoEMethod(unquantized_process)
    quantized_method = SimpleNamespace(process_weights_after_loading=quantized_process)
    unquantized_layer = _FakeAscendMoERunner(unquantized_method)
    quantized_layer = _FakeAscendMoERunner(quantized_method)
    duplicate_unquantized_layer = _FakeAscendMoERunner(unquantized_method)

    class _FakeModule:
        def modules(self):
            return iter(
                [
                    self,
                    unquantized_layer,
                    quantized_layer,
                    duplicate_unquantized_layer,
                ]
            )

    monkeypatch.setattr(
        routed_experts_module,
        "AscendUnquantizedFusedMoEMethod",
        _FakeAscendUnquantizedFusedMoEMethod,
    )
    monkeypatch.setattr(fused_moe_module, "AscendMoERunner", _FakeAscendMoERunner)

    with _rfork_skip_unquantized_moe_post_load_processing(_FakeModule()):
        assert unquantized_method.process_weights_after_loading() is None
        quantized_method.process_weights_after_loading()
        assert calls == ["quantized"]
    assert unquantized_method.process_weights_after_loading is unquantized_process
    assert quantized_method.process_weights_after_loading is quantized_process

    with pytest.raises(RuntimeError, match="boom"), _rfork_skip_unquantized_moe_post_load_processing(_FakeModule()):
        raise RuntimeError("boom")
    assert unquantized_method.process_weights_after_loading is unquantized_process
