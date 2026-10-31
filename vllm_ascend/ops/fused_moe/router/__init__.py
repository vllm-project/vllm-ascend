from vllm_ascend.ops.fused_moe.router.fused_topk_router import (
    AscendFusedTopKRouter as AscendFusedMoERouter,
)
from vllm_ascend.ops.fused_moe.router.router_factory import create_ascend_fused_moe_router

__all__ = [
    "AscendFusedMoERouter",
    "create_ascend_fused_moe_router",
]
