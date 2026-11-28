# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Eligibility helpers shared by A5 GMM-SwiGLU-OAI-MX-Quant callers."""

import math

import torch

from vllm_ascend.utils import enable_custom_op

MX_GROUP_SIZE = 32
MX_ALIGNMENT = 128
MAX_LOCAL_EXPERTS = 1024
MAX_TILING_DIM = (1 << 32) - 1
MIN_FP8_ROWS_PER_EXPERT = 513


def has_gmm_swiglu_oai_mx_quant() -> bool:
    return enable_custom_op() and hasattr(torch.ops._C_ascend, "gmm_swiglu_oai_mx_quant")


def supports_gmm_swiglu_oai_shape(*, experts: int, n: int, k: int) -> bool:
    return (
        1 <= experts <= MAX_LOCAL_EXPERTS
        and 0 < n <= MAX_TILING_DIM
        and 0 < k <= MAX_TILING_DIM
        and n % MX_ALIGNMENT == 0
        and k % MX_ALIGNMENT == 0
    )


def supports_gmm_swiglu_oai_attrs(*, limit: float, alpha: float, beta: float) -> bool:
    limit_f = float(limit)
    alpha_f = float(alpha)
    beta_f = float(beta)
    return math.isfinite(limit_f) and limit_f > 0.0 and math.isfinite(alpha_f) and math.isfinite(beta_f)


def get_layer_swiglu_oai_attrs(layer: torch.nn.Module) -> tuple[float, float, float]:
    moe_config = getattr(layer, "moe_config", None)
    source = moe_config if moe_config is not None else layer
    limit = getattr(source, "swiglu_limit", 0.0) or 0.0
    alpha = getattr(source, "swiglu_alpha", 1.0) or 1.0
    beta = getattr(source, "swiglu_beta", 0.0) or 0.0
    return float(limit), float(alpha), float(beta)
