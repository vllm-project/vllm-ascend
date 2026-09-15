"""FXRT-safe Python custom operator boundary for MoE prefill."""
import os

import torch
from vllm.distributed.parallel_state import get_dp_group

from vllm_ascend.utils import fxrt_prefill_decompose_enabled


@torch.library.custom_op("vllm_ascend::fxrt_moe_gating_top_k_hash", mutates_args=())
def fxrt_moe_gating_top_k_hash(
    x: torch.Tensor, k: int, bias: torch.Tensor | None,
    input_ids: torch.Tensor | None, tid2eid: torch.Tensor | None,
    k_group: int, group_count: int, routed_scaling_factor: float,
    eps: float, group_select_mode: int, renorm: int, norm_type: int,
    out_flag: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # The reduced dummy model uses padded DP metadata during prefill.  In a
    # 2DP run this can leave input_ids DP-gathered while x remains local to a
    # DP rank.  The C++ hash op requires one id per local router row.  Keep
    # this compatibility adjustment explicitly opt-in: real W8A8 execution
    # keeps the original all-gathered input_ids unchanged.
    if (
        os.getenv("VLLM_ASCEND_FXRT_DUMMY_QUANT") == "1"
        and input_ids is not None
        and input_ids.numel() != x.shape[0]
    ):
        dp_group = get_dp_group()
        rows = x.shape[0]
        start = dp_group.rank_in_group * rows
        input_ids = input_ids.flatten()[start : start + rows].contiguous()
    return torch.ops._C_ascend.moe_gating_top_k_hash(
        x, k, bias, input_ids, tid2eid, k_group, group_count,
        routed_scaling_factor, eps, group_select_mode, renorm, norm_type,
        out_flag)


@fxrt_moe_gating_top_k_hash.register_fake
def _fake(x, k, bias, input_ids, tid2eid, k_group, group_count,
          routed_scaling_factor, eps, group_select_mode, renorm, norm_type,
          out_flag):
    rows, experts = x.shape
    return (torch.empty((rows, k), device=x.device, dtype=x.dtype),
            torch.empty((rows, k), device=x.device, dtype=torch.int32),
            torch.empty((rows, experts), device=x.device, dtype=torch.float32))


def moe_gating_top_k_hash_for_prefill(*args, **kwargs):
    if fxrt_prefill_decompose_enabled():
        return fxrt_moe_gating_top_k_hash(*args, **kwargs)
    return torch.ops._C_ascend.moe_gating_top_k_hash(*args, **kwargs)
