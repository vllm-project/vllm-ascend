# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest

from tests.e2e.nightly.multi_node.internal_dp.scripts.multi_node_config import MultiNodeConfigLoader


@pytest.fixture
def deployment() -> dict:
    return {
        "envs": {"SERVER_PORT": 8080},
        "server_cmd": "vllm serve model --enable-expert-parallel",
        "runner_configs": {
            "v1": {
                "envs": {"DYNAMIC_EPLB": True},
                "server_cmd_suffix": '--additional-config \'{"eplb_config":{"dynamic_eplb":true}}\'',
            },
            "v2": {
                "envs": {"DYNAMIC_EPLB": False},
                "server_cmd_suffix": "--enable-eplb --eplb-config.use_async true",
            },
        },
    }


@pytest.mark.parametrize(
    ("use_v2_model_runner", "expected_env", "expected_arg", "unexpected_arg"),
    [
        (False, True, "dynamic_eplb", "--enable-eplb"),
        (True, False, "--enable-eplb", "dynamic_eplb"),
    ],
)
def test_resolve_runner_deployment(
    deployment: dict,
    use_v2_model_runner: bool,
    expected_env: bool,
    expected_arg: str,
    unexpected_arg: str,
) -> None:
    cmd, envs = MultiNodeConfigLoader._resolve_runner_deployment(deployment, use_v2_model_runner=use_v2_model_runner)

    assert cmd.startswith("vllm serve model")
    assert expected_arg in cmd
    assert unexpected_arg not in cmd
    assert envs["DYNAMIC_EPLB"] is expected_env
    assert envs["SERVER_PORT"] == 8080
    assert deployment["envs"] == {"SERVER_PORT": 8080}


def test_resolve_runner_deployment_without_overrides() -> None:
    deployment = {
        "envs": {"SERVER_PORT": 8080},
        "server_cmd": "vllm serve model",
    }

    assert MultiNodeConfigLoader._resolve_runner_deployment(deployment, True) == (
        "vllm serve model",
        {"SERVER_PORT": 8080},
    )


def test_resolve_runner_deployment_requires_selected_runner(deployment: dict) -> None:
    del deployment["runner_configs"]["v2"]

    with pytest.raises(KeyError, match=r"runner_configs\.v2"):
        MultiNodeConfigLoader._resolve_runner_deployment(deployment, True)
