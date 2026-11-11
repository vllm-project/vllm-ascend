# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""A8W8 per-token quant requires fp32 dequant scales.

Qwen3.5/3.6 W8A8 checkpoints ship fp16 scales. Passing them to
``npu_quant_matmul`` / ``npu_grouped_matmul`` fails with aclnn 161002
("scale[0] dtype FLOAT16 and y dtype FLOAT16 are not supported when A8W8
per-token quant"). These helpers pick the fp32 scale the loader already builds
and normalise per-expert grouped-matmul scales.
"""

import torch

from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    _dequant_scale,
    _fused_scale_list,
)


class _LayerWithFp32:
    def __init__(self):
        self.weight_scale = torch.ones(4, 1, dtype=torch.float16)
        self.weight_scale_fp32 = self.weight_scale.to(torch.float32)


class _LayerWithoutFp32:
    def __init__(self):
        self.weight_scale = torch.ones(4, 1, dtype=torch.float16)


def test_prefers_fp32_scale_when_available():
    scale = _dequant_scale(_LayerWithFp32())
    assert scale.dtype == torch.float32


def test_falls_back_to_weight_scale_when_fp32_copy_is_absent():
    scale = _dequant_scale(_LayerWithoutFp32())
    assert scale.dtype == torch.float16


def test_fused_scale_list_converts_every_expert():
    scales = [torch.ones(2, dtype=torch.float16) for _ in range(3)]
    converted = _fused_scale_list(scales)
    assert len(converted) == 3
    assert all(scale.dtype == torch.float32 for scale in converted)


def test_fused_scale_list_keeps_per_expert_shape():
    scales = [torch.ones((experts, 1), dtype=torch.float16) for experts in (2, 3)]
    assert [tuple(scale.shape) for scale in _fused_scale_list(scales)] == [(2, 1), (3, 1)]
