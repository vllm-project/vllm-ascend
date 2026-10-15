#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Configuration and device initialization coverage for RoPE UB sizing."""

from unittest.mock import MagicMock

import pytest

from vllm_ascend.ascend_config import AscendConfig, SparseKVOffloadConfig
from vllm_ascend.ops.triton import triton_utils


def test_rope_ub_config_default():
    assert AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig()).rope_ub_size_kb == 0


@pytest.mark.parametrize("value", [0, 128, 192, 256])
def test_rope_ub_config_values(value):
    assert (
        AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig(), rope_ub_size_kb=value).rope_ub_size_kb == value
    )


@pytest.mark.parametrize("value", [-1, 1.5, "invalid"])
def test_rope_ub_config_invalid(value):
    with pytest.raises(ValueError, match="rope_ub_size_kb"):
        AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig(), rope_ub_size_kb=value)


@pytest.fixture
def mock_device(monkeypatch):
    monkeypatch.setattr(triton_utils, "HAS_TRITON", True)
    for name in ("_NUM_AICORE", "_NUM_VECTORCORE", "_UB_SIZE_BYTES"):
        monkeypatch.setattr(triton_utils, name, -1)
    driver = MagicMock()
    monkeypatch.setattr(triton_utils, "triton", driver)
    monkeypatch.setattr(triton_utils.torch.npu, "current_device", lambda: 0)
    return driver.runtime.driver.active.utils.get_device_properties


@pytest.mark.parametrize("detected", [None, 1, 128 * 1024])
@pytest.mark.parametrize("override", [0, 256])
def test_rope_ub_device_initialization(mock_device, monkeypatch, detected, override):
    # The removed environment variable must not affect either config path.
    monkeypatch.setenv("VLLM_ASCEND_ROPE_UB_SIZE_KB", "512")
    mock_device.return_value = {"num_aicore": 20, "num_vectorcore": 40, "ub_size": detected}
    config = AscendConfig(sparse_kv_offload_config=SparseKVOffloadConfig(), rope_ub_size_kb=override)
    triton_utils.init_device_properties_triton(rope_ub_size_kb=config.rope_ub_size_kb)
    expected = override * 1024 if override else (detected if detected and detected >= 4096 else 192 * 1024)
    assert triton_utils.get_ub_size_bytes() == expected
    assert triton_utils.get_aicore_num() == 20
    assert triton_utils.get_vectorcore_num() == 40


def test_override_after_early_initialization(mock_device):
    mock_device.return_value = {"num_aicore": 20, "num_vectorcore": 40, "ub_size": 192 * 1024}
    triton_utils.init_device_properties_triton()
    triton_utils.init_device_properties_triton(rope_ub_size_kb=128)
    triton_utils.init_device_properties_triton()
    assert triton_utils.get_ub_size_bytes() == 128 * 1024
    mock_device.assert_called_once()
