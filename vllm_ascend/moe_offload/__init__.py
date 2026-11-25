"""
MoE expert weight offload module.

Provides per-layer MoE expert weight offloading: weights are stored on CPU
pinned memory and H2D'd to NPU 2-slot rotation on demand, reducing NPU HBM
footprint for large MoE models.

Compatible with:
- FlashComm1 (only affects dense layers, not MoE routed experts)
- FUSED_MC2=1 (dispatch_ffn_combine_w4_a8 takes per-expert lists directly)
- FUSED_MC2=0 (grouped_matmul_swiglu_quant_v2 for W4A8 per-channel)
- INT8 KV cache (orthogonal to MoE weight offload)
"""

import logging

# Ensure INFO-level messages from this package are not swallowed.
# The vllm_ascend root logger has propagate=False and its own handler,
# but in spawned worker processes the handler configuration timing can
# leave child loggers at the Python default (WARNING). Setting the
# level here guarantees INFO records from manager/eplb_d2d are emitted.
logging.getLogger(__name__).setLevel(logging.INFO)

from vllm_ascend.moe_offload.manager import MoEOffloadManager  # noqa: E402

__all__ = ["MoEOffloadManager"]
