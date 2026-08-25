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
from vllm.config.compilation import CompilationMode, CUDAGraphMode

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.platform import _update_compilation_modes, _validate_turboquant_cache


def _make_config(
    *,
    dtype: torch.dtype = torch.bfloat16,
    kv_lora_rank: int = 512,
    rope_head_dim: int = 64,
    cache_dtype: str = "turboquant_4bit_nc",
    additional_config: dict[str, object] | None = None,
    use_v2_model_runner: bool = False,
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
        additional_config=additional_config,
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        model_config=model_config,
        use_v2_model_runner=use_v2_model_runner,
    )


def test_validate_turboquant_cache_does_not_inject_internal_flag() -> None:
    config = _make_config()

    with patch(
        "vllm_ascend.platform.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A2),
    ):
        _validate_turboquant_cache(config)

    assert config.additional_config is None


def test_compilation_update_preserves_turboquant_cache_dtype() -> None:
    config = _make_config()
    config.compilation_config = SimpleNamespace(
        mode=CompilationMode.NONE,
        splitting_ops=[],
        cudagraph_mode=CUDAGraphMode.NONE,
    )
    config.speculative_config = None
    config.model_config.enforce_eager = False
    config.model_config.is_encoder_decoder = False
    ascend_config = SimpleNamespace(
        ascend_compilation_config={},
        xlite_graph_config=SimpleNamespace(enabled=False, full_mode=False),
    )

    with patch(
        "vllm_ascend.platform.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A2),
    ):
        _validate_turboquant_cache(config)
    _update_compilation_modes(config, ascend_config)

    assert config.cache_config.cache_dtype == "turboquant_4bit_nc"


def test_validate_turboquant_cache_rejects_direct_enable_flag() -> None:
    config = _make_config(
        cache_dtype="auto",
        additional_config={"enable_sparse_sfa_turboquant": True},
    )

    with pytest.raises(ValueError, match="enable_sparse_sfa_turboquant"):
        _validate_turboquant_cache(config)


def test_validate_turboquant_cache_rejects_explicit_sparse_sfa_c8() -> None:
    config = _make_config(additional_config={"enable_sparse_sfa_c8": True})

    with pytest.raises(
        ValueError,
        match="turboquant_4bit_nc and enable_sparse_sfa_c8 cannot be enabled together",
    ):
        _validate_turboquant_cache(config)


def test_validate_turboquant_cache_rejects_xlite() -> None:
    config = _make_config(additional_config={"xlite_graph_config": {"enabled": True}})

    with pytest.raises(ValueError, match="does not support xLite graph mode"):
        _validate_turboquant_cache(config)


@pytest.mark.parametrize(
    "additional_config",
    [
        {"enable_sparse_sfa_c8": "false"},
        {"xlite_graph_config": {"enabled": "false"}},
    ],
)
def test_validate_turboquant_cache_coerces_false_strings(
    additional_config: dict[str, object],
) -> None:
    config = _make_config(additional_config=additional_config)

    with patch(
        "vllm_ascend.platform.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A2),
    ):
        _validate_turboquant_cache(config)


def test_validate_turboquant_cache_rejects_model_runner_v2() -> None:
    config = _make_config(use_v2_model_runner=True)

    with pytest.raises(ValueError, match="only supports Model Runner V1"):
        _validate_turboquant_cache(config)


def test_validate_turboquant_cache_allows_model_runner_v2_for_other_cache_dtypes() -> None:
    config = _make_config(cache_dtype="auto", use_v2_model_runner=True)

    _validate_turboquant_cache(config)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("tq_key_quant_mode", 3),
        ("tq_value_quant_mode", 3),
        ("tq_tile_size", 128),
    ],
)
def test_validate_turboquant_cache_rejects_internal_parameters(
    option: str,
    value: object,
) -> None:
    config = _make_config(additional_config={option: value})

    with pytest.raises(ValueError, match=option):
        _validate_turboquant_cache(config)


@pytest.mark.parametrize(
    ("device_type", "dtype", "kv_lora_rank", "rope_head_dim", "message"),
    [
        (AscendDeviceType.A5, torch.bfloat16, 512, 64, "only supported on Ascend A2 and A3"),
        (AscendDeviceType.A2, torch.float16, 512, 64, "requires bfloat16"),
        (AscendDeviceType.A2, torch.bfloat16, 256, 64, "requires kv_lora_rank=512"),
        (AscendDeviceType.A2, torch.bfloat16, 512, 32, "requires qk_rope_head_dim=64"),
    ],
)
def test_validate_turboquant_cache_rejects_unsupported_contract(
    device_type: AscendDeviceType,
    dtype: torch.dtype,
    kv_lora_rank: int,
    rope_head_dim: int,
    message: str,
) -> None:
    config = _make_config(dtype=dtype, kv_lora_rank=kv_lora_rank, rope_head_dim=rope_head_dim)

    with (
        patch(
            "vllm_ascend.platform.get_current_hardware_profile",
            return_value=get_hardware_profile(device_type),
        ),
        pytest.raises(ValueError, match=message),
    ):
        _validate_turboquant_cache(config)


def test_validate_turboquant_cache_rejects_non_sfa_model() -> None:
    config = _make_config()
    del config.model_config.hf_text_config.index_topk

    with pytest.raises(ValueError, match="only supported by SFA sparse models"):
        _validate_turboquant_cache(config)
