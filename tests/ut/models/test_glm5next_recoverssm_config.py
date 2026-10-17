# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_ascend import envs as ascend_envs
from vllm_ascend.models.glm5next.recoverssm import validate_recoverssm_config


def make_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        use_v2_model_runner=False,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1),
        speculative_config=SimpleNamespace(method="deepseek_mtp", num_speculative_tokens=5),
    )


def test_supported_execution_contract():
    validate_recoverssm_config(make_config())


@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True), ("2", None)])
def test_experimental_flag_defaults_off_and_rejects_invalid_values(monkeypatch, value, expected):
    name = "VLLM_ASCEND_ENABLE_EXPERIMENTAL_KDA_RECOVERSSM"
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)
    read_flag = ascend_envs.env_variables[name]
    if expected is None:
        with pytest.raises(ValueError, match="must be either"):
            read_flag()
    else:
        assert read_flag() is expected


@pytest.mark.parametrize(
    "path,value",
    [
        ("model_config.enforce_eager", False),
        ("scheduler_config.async_scheduling", True),
        ("use_v2_model_runner", True),
        ("cache_config.enable_prefix_caching", True),
        ("kv_transfer_config", object()),
        ("parallel_config.pipeline_parallel_size", 2),
        ("parallel_config.data_parallel_size", 2),
        ("speculative_config", None),
        ("speculative_config.method", "eagle3"),
        ("speculative_config.num_speculative_tokens", 0),
    ],
)
def test_unsupported_execution_fails_before_inference(path, value):
    config = make_config()
    owner = config
    parts = path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], value)
    with pytest.raises(ValueError, match="Experimental KDA RecoverSSM"):
        validate_recoverssm_config(config)
