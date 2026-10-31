import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.patch.platform import patch_dp_device_ids
from vllm_ascend.utils import vllm_version_is

pytestmark = pytest.mark.skipif(not vllm_version_is("0.23.0"), reason="CatCCOS DP visibility patch targets vLLM 0.23")


def _vllm_config(backend="catccos", dp_size=4):
    return SimpleNamespace(
        additional_config={"fused_mc2_backend": backend},
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            data_parallel_size_local=dp_size,
            distributed_executor_backend="mp",
            data_parallel_backend="mp",
            nnodes_within_dp=1,
            assigned_physical_gpu_ids=None,
            world_size=1,
        ),
    )


def test_catccos_dp_keeps_all_local_devices_visible(monkeypatch):
    device_env = patch_dp_device_ids.current_platform.device_control_env_var
    called = []

    @contextmanager
    def original_context(vllm_config, local_dp_rank):
        called.append(local_dp_rank)
        with patch.dict("os.environ", {device_env: "3"}):
            yield

    monkeypatch.setattr(patch_dp_device_ids, "_original_set_device_control_env_var", original_context)
    monkeypatch.setenv(device_env, "0,1,2,3")

    with patch_dp_device_ids._set_device_control_env_var(_vllm_config(), local_dp_rank=2):
        assert os.environ[device_env] == "0,1,2,3"

    assert called == []
    assert os.environ[device_env] == "0,1,2,3"


def test_non_catccos_dp_uses_upstream_device_isolation(monkeypatch):
    device_env = patch_dp_device_ids.current_platform.device_control_env_var
    called = []

    @contextmanager
    def original_context(vllm_config, local_dp_rank):
        called.append(local_dp_rank)
        with patch.dict("os.environ", {device_env: "2"}):
            yield

    monkeypatch.setattr(patch_dp_device_ids, "_original_set_device_control_env_var", original_context)
    monkeypatch.setenv(device_env, "0,1,2,3")

    with patch_dp_device_ids._set_device_control_env_var(_vllm_config(backend="deepmegamoe"), local_dp_rank=2):
        assert os.environ[device_env] == "2"

    assert called == [2]
    assert os.environ[device_env] == "0,1,2,3"


def test_catccos_dp_rejects_incomplete_visible_device_list(monkeypatch):
    device_env = patch_dp_device_ids.current_platform.device_control_env_var
    monkeypatch.setenv(device_env, "0")

    with (
        pytest.raises(RuntimeError, match="needs 4 visible devices"),
        patch_dp_device_ids._set_device_control_env_var(_vllm_config(), local_dp_rank=1),
    ):
        pass
