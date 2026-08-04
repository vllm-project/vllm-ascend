"""Explicit entry points for Kimi K3 target-verification fusion kernels.

These functions require a speculative-decoding implementation that owns
per-candidate convolution and KDA state snapshots. The v0.26.0rc K3 runtime
does not expose that cache lifecycle, so ordinary K3 prefill/decode must not
call them.
"""

from typing import Any

from vllm.triton_utils import HAS_TRITON


def _require_triton(op_name: str) -> None:
    if not HAS_TRITON:
        raise RuntimeError(f"{op_name} requires Triton on Ascend.")


def causal_conv1d_linear_verify_npu(*args: Any, **kwargs: Any):
    """Run the supplied causal-convolution verifier with state snapshots."""
    _require_triton("Kimi K3 causal-convolution target verification")
    from vllm_ascend.ops.triton.kimi_k3.causal_conv1d_verify import (
        causal_conv1d_linear_verify_npu as run,
    )

    return run(*args, **kwargs)


def kda_target_verify_npu(*args: Any, **kwargs: Any):
    """Run the supplied KDA verifier with per-step state snapshots."""
    _require_triton("Kimi K3 KDA target verification")
    from vllm_ascend.ops.triton.kimi_k3.kda_target_verify import (
        kda_target_verify_npu as run,
    )

    return run(*args, **kwargs)


__all__ = ["causal_conv1d_linear_verify_npu", "kda_target_verify_npu"]
