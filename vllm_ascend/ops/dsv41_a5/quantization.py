# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 indexer quantization entry point."""

from __future__ import annotations

import torch

from vllm_ascend.ops.triton.quantize_mxfp4_indexer import quantize_mxfp4_indexer


def mxfp4_quantize_e8m0(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_mxfp4_indexer(x)
