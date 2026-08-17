# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.quantization.nvfp4 import (
    AscendNvFp4Config,
    dequantize_nvfp4,
    unpack_nvfp4,
)


def test_unpack_nvfp4_uses_low_nibble_first():
    packed = torch.tensor([[0x10, 0xB7, 0xEF]], dtype=torch.uint8)

    unpacked = unpack_nvfp4(packed)

    expected = torch.tensor([[0.0, 0.5, 6.0, -1.5, -4.0, -6.0]])
    torch.testing.assert_close(unpacked, expected)


def test_dequantize_nvfp4_applies_block_and_global_scales():
    packed = torch.tensor([[0x21, 0x43, 0x65, 0x87]], dtype=torch.uint8)
    block_scale = torch.tensor([[2.0, 4.0]])
    global_scale = torch.tensor(0.25)

    result = dequantize_nvfp4(packed, block_scale, global_scale, group_size=4)

    values = torch.tensor([[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0]])
    expected = values * torch.tensor([[0.5] * 4 + [1.0] * 4])
    torch.testing.assert_close(result, expected)


def test_dequantize_nvfp4_rejects_invalid_scale_shape():
    packed = torch.zeros((2, 8), dtype=torch.uint8)

    with pytest.raises(ValueError, match="weight_scale shape"):
        dequantize_nvfp4(packed, torch.ones((2, 2)), torch.tensor(1.0))


def test_nvfp4_config_parses_modelopt_layout():
    config = AscendNvFp4Config.from_config(
        {
            "quantization": {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "exclude_modules": ["lm_head"],
            }
        }
    )

    assert config.group_size == 16
    assert config.ignore == ["lm_head"]


def test_nvfp4_config_rejects_nonstandard_group_size():
    with pytest.raises(ValueError, match="group_size=16"):
        AscendNvFp4Config(group_size=32)
