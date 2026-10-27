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

import copy
import os
from unittest.mock import patch

import pytest
from vllm.config import VllmConfig

from vllm_ascend.ascend_config import clear_ascend_config, init_ascend_config
from vllm_ascend.utils import clear_enable_sp, enable_sp


def _sp_config(additional_config):
    config = VllmConfig()
    config.parallel_config.enable_expert_parallel = True
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.all2all_backend = "allgather_reducescatter"
    config.additional_config = additional_config
    return config


def test_additional_config_false_overrides_enabled_environment_variable():
    clear_ascend_config()
    clear_enable_sp()
    config = _sp_config({"enable_flashcomm1": False})

    with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"}):
        init_ascend_config(config)

    assert not enable_sp(config)
    assert config.parallel_config.all2all_backend == "flashinfer_all2allv"


def test_additional_config_true_overrides_disabled_environment_variable():
    clear_ascend_config()
    clear_enable_sp()
    config = _sp_config({"enable_flashcomm1": True})

    with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "0"}):
        init_ascend_config(config)

    assert enable_sp(config)
    assert config.parallel_config.all2all_backend == "allgather_reducescatter"


def test_environment_variable_remains_migration_fallback():
    clear_ascend_config()
    clear_enable_sp()
    config = _sp_config({})

    with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"}):
        init_ascend_config(config)

    assert enable_sp(config)
    assert config.parallel_config.all2all_backend == "allgather_reducescatter"


def test_serialized_parallel_config_preserves_sp_decision_for_worker_subprocess():
    clear_ascend_config()
    clear_enable_sp()
    config = _sp_config({"enable_flashcomm1": True})

    with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "0"}):
        init_ascend_config(config)

    worker_config = copy.deepcopy(config.parallel_config)

    assert worker_config.use_sequence_parallel_moe


@pytest.mark.parametrize("enabled", [False, True])
def test_enable_sp_with_explicit_worker_config_needs_no_current_context(enabled):
    clear_ascend_config()
    config = _sp_config({"enable_flashcomm1": enabled})
    with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": str(int(not enabled))}):
        init_ascend_config(config)
    worker_config = copy.deepcopy(config)

    with patch(
        "vllm.config.get_current_vllm_config", side_effect=AssertionError("Current vLLM config is not set")
    ) as current:
        assert enable_sp(worker_config) is enabled
        current.assert_not_called()
        assert enable_sp() is False


@pytest.mark.parametrize("legacy_value", [None, "0"])
def test_flashcomm_defaults_to_disabled(legacy_value):
    clear_ascend_config()
    config = _sp_config({})
    with patch.dict(os.environ):
        os.environ.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
        if legacy_value is not None:
            os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = legacy_value
        init_ascend_config(config)
    assert not enable_sp(config)


@pytest.mark.parametrize("enabled", [False, True])
def test_conflicting_legacy_value_logs_explicit_override(enabled):
    clear_ascend_config()
    config = _sp_config({"enable_flashcomm1": enabled})
    legacy_value = str(int(not enabled))
    with (
        patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": legacy_value}),
        patch("vllm_ascend.ascend_config.logger.warning_once") as warning,
    ):
        init_ascend_config(config)
    warning.assert_any_call(
        "additional_config.enable_flashcomm1=%s overrides VLLM_ASCEND_ENABLE_FLASHCOMM1=%s.",
        enabled,
        legacy_value,
    )
