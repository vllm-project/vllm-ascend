# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Factories for the Ascend ``draft_lm_head_quantization`` formats.

`NPUPlatform.get_draft_lm_head_quantization_methods` maps each format name
to a factory qualified name in this module; the upstream lm_head copy
mechanism resolves and calls the factory when it builds the drafter's
quantized copy of a shared target LM head:

- ``mxfp8``: per-group MXFP8 weights on the native MXFP8 matmul
  (A5-class, FP8_OPTIMIZED device family).
- ``int8``: per-channel INT8 weights on the W8A8 matmul (A2/A3-class,
  STANDARD device family).
- ``fp8``/``nvfp4``: the core formats, which have no Ascend execution
  path; shadowed with factories that raise a clear error instead of
  failing inside a CUDA-kernel initialization.

310P (COMPATIBILITY family) declares no Ascend format.
"""

from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLmHeadMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLmHeadMethod


def create_mxfp8_lm_head_method() -> AscendEmbeddingMethod:
    """Quantize the draft copy into per-group MXFP8 (A5-class devices)."""
    return AscendEmbeddingMethod(AscendW8A8MXFP8DynamicLmHeadMethod())


def create_int8_lm_head_method() -> AscendEmbeddingMethod:
    """Quantize the draft copy into per-channel INT8 (A2/A3-class devices)."""
    return AscendEmbeddingMethod(AscendW8A8DynamicLmHeadMethod())


def _unsupported_format_error(format_name: str) -> ValueError:
    return ValueError(
        f'draft_lm_head_quantization="{format_name}" has no Ascend execution '
        'path; use "mxfp8" on A5-class devices (FP8_OPTIMIZED family) or '
        '"int8" on A2/A3-class devices (STANDARD family).'
    )


def create_unsupported_fp8_method() -> AscendEmbeddingMethod:
    """Reject the core ``fp8`` format with a pointer to the Ascend formats."""
    raise _unsupported_format_error("fp8")


def create_unsupported_nvfp4_method() -> AscendEmbeddingMethod:
    """Reject the core ``nvfp4`` format with a pointer to the Ascend formats."""
    raise _unsupported_format_error("nvfp4")
