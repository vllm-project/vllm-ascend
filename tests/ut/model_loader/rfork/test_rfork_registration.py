from unittest.mock import patch

import vllm_ascend
import vllm_ascend.envs as envs_ascend


def test_register_model_loader_registers_netloader_and_rforkloader() -> None:
    with patch("vllm_ascend.model_loader.netloader.register_netloader") as netloader_register:
        with patch("vllm_ascend.model_loader.rfork.register_rforkloader") as rfork_register:
            vllm_ascend.register_model_loader()
            netloader_register.assert_called_once()
            rfork_register.assert_called_once()


def test_envs_exposes_rfork_related_variables() -> None:
    assert hasattr(envs_ascend, "VLLM_RFORK_ENABLED")
    assert hasattr(envs_ascend, "RFORK_SCHEDULER_URL")
    assert hasattr(envs_ascend, "RFORK_SEED_KEY_SEPARATOR")
