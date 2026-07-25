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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.ascend_forward_context import _cann_megamoe_supported_by_config
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    _CANN_ACL_INT4,
    _CANN_ACL_INT8,
    _CANN_MEGA_MOE_QUANT_MODE_INT8,
    _CANN_MEGA_MOE_QUANT_MODE_MX,
    _CANN_TORCH_FLOAT8_E4M3FN,
    _get_cann_mega_moe_quant_settings,
)
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize(
    ("quant_type", "expected"),
    [
        (QuantType.W8A8, (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT8)),
        (QuantType.W4A8, (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT4)),
        (QuantType.W8A8MXFP, (_CANN_MEGA_MOE_QUANT_MODE_MX, _CANN_TORCH_FLOAT8_E4M3FN, None)),
        (QuantType.W4A8MXFP, (_CANN_MEGA_MOE_QUANT_MODE_MX, _CANN_TORCH_FLOAT8_E4M3FN, None)),
    ],
)
def test_cann_mega_moe_quant_settings(quant_type, expected):
    assert _get_cann_mega_moe_quant_settings(quant_type) == expected


def test_cann_mega_moe_quant_settings_rejects_unsupported_type():
    with pytest.raises(RuntimeError, match="Unsupported quant type"):
        _get_cann_mega_moe_quant_settings(QuantType.W4A16)


def _make_vllm_config(hidden_size):
    hf_text_config = SimpleNamespace(hidden_size=hidden_size)
    model_config = MagicMock()
    model_config.hf_text_config = hf_text_config
    model_config.get_hidden_size = MagicMock(return_value=hidden_size)
    return SimpleNamespace(model_config=model_config)


@pytest.mark.parametrize("hidden_size", [1024, 1536, 4096, 6144, 7168, 8192])
def test_cann_mega_moe_supported_hidden_sizes(hidden_size):
    config = _make_vllm_config(hidden_size)
    assert _cann_megamoe_supported_by_config(config, "w8a8_dynamic")


@pytest.mark.parametrize("hidden_size", [512, 896, 1023, 1025, 7000, 8704, 9216])
def test_cann_mega_moe_rejects_unsupported_hidden_sizes(hidden_size):
    config = _make_vllm_config(hidden_size)
    assert not _cann_megamoe_supported_by_config(config, "w8a8_dynamic")


def test_cann_mega_moe_rejects_unsupported_quantization():
    config = _make_vllm_config(6144)
    assert not _cann_megamoe_supported_by_config(config, "fp16")


@pytest.mark.parametrize("quant_name", ["w8a8", "w4a8", "w8a8_dynamic", "w4a8_dynamic"])
def test_cann_mega_moe_supported_quantization_names(quant_name):
    config = _make_vllm_config(6144)
    assert _cann_megamoe_supported_by_config(config, quant_name)
