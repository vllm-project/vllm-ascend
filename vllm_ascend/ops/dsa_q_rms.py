# SPDX-License-Identifier: Apache-2.0
"""Keep eager Q RMS arithmetic behind a graph-safe boundary."""

import torch


@torch.library.custom_op("vllm_ascend::fxrt_dsa_q_rms", mutates_args=())
def fxrt_dsa_q_rms(q: torch.Tensor, epsilon: float) -> torch.Tensor:
    # Lazy import keeps registration independent of Triton availability.
    from vllm_ascend.ops.triton.rms_norm import triton_q_rms

    return triton_q_rms(q, epsilon)


@fxrt_dsa_q_rms.register_fake
def _fxrt_dsa_q_rms_fake(q: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.empty_like(q)
