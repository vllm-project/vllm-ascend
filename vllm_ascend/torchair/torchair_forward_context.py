from enum import Enum

import vllm_ascend.envs as envs_ascend


class FusedMoEState(Enum):
    AllGather = 0
    All2All = 1
    MC2 = 2
    AllGatherEP = 3
    NaiveMulticast = 4
    All2AllSeq = 5


def get_fused_moe_state(ep_size: int, with_prefill: bool,
                        is_deepseek_v3_r1: bool):
    # the fusion operator torch_npu.npu_grouped_matmul_finalize_routing called by allgather ep
    # only supports deepseek v3/r1
    if (envs_ascend.VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP and ep_size > 1
            and is_deepseek_v3_r1):
        return FusedMoEState.AllGatherEP
    elif ep_size == 1:
        if with_prefill:
            return FusedMoEState.NaiveMulticast
        else:
            return FusedMoEState.AllGather
    # NOTE: mc2 need ep_size >= 16 & all2all can't use in torchair graph.
    elif ep_size < 16 or with_prefill:
        return FusedMoEState.All2All
    else:
        return FusedMoEState.MC2