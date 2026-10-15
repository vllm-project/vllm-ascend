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

"""Device detection and fallback coverage for RoPE UB sizing."""

from unittest.mock import MagicMock

import pytest

from vllm_ascend.ops.triton import triton_utils


@pytest.fixture
def mock_device(monkeypatch):
    monkeypatch.setattr(triton_utils, "HAS_TRITON", True)
    for name in ("_NUM_AICORE", "_NUM_VECTORCORE", "_UB_SIZE_BYTES"):
        monkeypatch.setattr(triton_utils, name, -1)
    driver = MagicMock()
    monkeypatch.setattr(triton_utils, "triton", driver)
    monkeypatch.setattr(triton_utils.torch.npu, "current_device", lambda: 0)
    return driver.runtime.driver.active.utils.get_device_properties


@pytest.mark.parametrize("detected", [None, 0, 1, 128 * 1024, 256 * 1024])
def test_rope_ub_device_initialization(mock_device, monkeypatch, detected):
    # A stale deployment environment must not override device detection.
    monkeypatch.setenv("VLLM_ASCEND_ROPE_UB_SIZE_KB", "512")
    mock_device.return_value = {"num_aicore": 20, "num_vectorcore": 40, "ub_size": detected}
    triton_utils.init_device_properties_triton()
    expected = detected if detected and detected >= 4096 else triton_utils.DEFAULT_UB_SIZE_BYTES
    assert triton_utils.get_ub_size_bytes() == expected
    assert triton_utils.get_aicore_num() == 20
    assert triton_utils.get_vectorcore_num() == 40


@pytest.mark.parametrize("key", triton_utils._UB_SIZE_PROPERTY_KEYS)
def test_rope_ub_device_property_keys(mock_device, key):
    mock_device.return_value = {"num_aicore": 20, "num_vectorcore": 40, key: 128 * 1024}
    triton_utils.init_device_properties_triton()
    assert triton_utils.get_ub_size_bytes() == 128 * 1024


def test_repeated_initialization_preserves_detected_size(mock_device):
    mock_device.return_value = {"num_aicore": 20, "num_vectorcore": 40, "ub_size": 128 * 1024}
    triton_utils.init_device_properties_triton()
    triton_utils.init_device_properties_triton()
    assert triton_utils.get_ub_size_bytes() == 128 * 1024
    mock_device.assert_called_once()
