# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LeapQuant: window-boundary quantization of linear-attention recurrent state.

See ``vllm_ascend/quantization/methods/mamba_state/leapquant/state.py`` for
the algorithm reference and RFC vllm-ascend#21097 for the rollout plan.
"""

from vllm_ascend.quantization.methods.mamba_state.leapquant.runtime import (
    leapquant_config_or_none,
    leapquant_flush_state_rows,
)
from vllm_ascend.quantization.methods.mamba_state.leapquant.state import (
    decode_flush_mask,
    fit_compensators,
    leap_window_flush,
    mgs_orthonormal_columns,
    prefill_flush_mask,
    quantize_residual,
    reconstruct_state,
    start_basis,
)

__all__ = [
    "decode_flush_mask",
    "fit_compensators",
    "leap_window_flush",
    "leapquant_config_or_none",
    "leapquant_flush_state_rows",
    "mgs_orthonormal_columns",
    "prefill_flush_mask",
    "quantize_residual",
    "reconstruct_state",
    "start_basis",
]
