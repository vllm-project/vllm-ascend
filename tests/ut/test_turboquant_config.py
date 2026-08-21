#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.platform import _configure_turboquant_cache
from vllm_ascend.utils import AscendDeviceType


def _make_config(
    *,
    dtype: torch.dtype = torch.bfloat16,
    kv_lora_rank: int = 512,
    rope_head_dim: int = 64,
):
    hf_text_config = SimpleNamespace(
        index_topk=2048,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope_head_dim,
    )
    model_config = SimpleNamespace(
        dtype=dtype,
        hf_config=SimpleNamespace(),
        hf_text_config=hf_text_config,
    )
    return SimpleNamespace(
        additional_config=None,
        cache_config=SimpleNamespace(cache_dtype="turboquant_4bit_nc"),
        model_config=model_config,
    )


def test_configure_turboquant_cache_sets_flag_before_ascend_config_init() -> None:
    config = _make_config()

    with patch("vllm_ascend.platform.get_ascend_device_type", return_value=AscendDeviceType.A2):
        _configure_turboquant_cache(config)

    assert config.additional_config == {"enable_sparse_sfa_turboquant": True}


@pytest.mark.parametrize(
    ("device_type", "dtype", "kv_lora_rank", "rope_head_dim", "message"),
    [
        (AscendDeviceType.A5, torch.bfloat16, 512, 64, "only supported on Ascend A2 and A3"),
        (AscendDeviceType.A2, torch.float16, 512, 64, "requires bfloat16"),
        (AscendDeviceType.A2, torch.bfloat16, 256, 64, "requires kv_lora_rank=512"),
        (AscendDeviceType.A2, torch.bfloat16, 512, 32, "requires qk_rope_head_dim=64"),
    ],
)
def test_configure_turboquant_cache_rejects_unsupported_contract(
    device_type: AscendDeviceType,
    dtype: torch.dtype,
    kv_lora_rank: int,
    rope_head_dim: int,
    message: str,
) -> None:
    config = _make_config(dtype=dtype, kv_lora_rank=kv_lora_rank, rope_head_dim=rope_head_dim)

    with (
        patch("vllm_ascend.platform.get_ascend_device_type", return_value=device_type),
        pytest.raises(ValueError, match=message),
    ):
        _configure_turboquant_cache(config)


def test_configure_turboquant_cache_rejects_non_sfa_model() -> None:
    config = _make_config()
    del config.model_config.hf_text_config.index_topk

    with pytest.raises(ValueError, match="only supported by SFA sparse models"):
        _configure_turboquant_cache(config)
