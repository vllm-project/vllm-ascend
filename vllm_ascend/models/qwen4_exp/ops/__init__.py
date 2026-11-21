"""Model-local Ascend kernels for Qwen4Exp/Qwen3.8 Flash-Next."""

from .reference import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
from .split_qkv_rmsnorm_mrope import qwen4_exp_split_qkv_rmsnorm_mrope_out

__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_combine_norm",
    "hc_gate_mix",
    "hc_silu",
    "qwen4_exp_split_qkv_rmsnorm_mrope_out",
]
