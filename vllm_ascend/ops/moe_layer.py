import os
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.distributed import (GroupCoordinator, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.distributed.parallel_state import (get_dp_group, get_ep_group,
                                             get_tp_group)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import \
    FusedMoEConfig  # isort: skip
from vllm.model_executor.layers.fused_moe.config import \
    FusedMoEParallelConfig  # isort: skip
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod, determine_expert_map)
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import FusedMoEState
from vllm_ascend.distributed.communication_op import \
    data_parallel_reduce_scatter
from vllm_ascend.distributed.parallel_state import get_mc2_group
MOE_ALL2ALL_BUFFER: bool = envs_ascend.MOE_ALL2ALL_BUFFER
SELECT_GATING_TOPK_SOTFMAX_EXPERTS: bool = envs_ascend.SELECT_GATING_TOPK_SOTFMAX_EXPERTS


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
 
    def __init__(self, moe: MoEConfig = None):

        super().__init__(moe=moe)
        vllm_config = get_current_vllm_config()

        self.global_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled

        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(
                local_rank)
        except AttributeError:
            self.moe_all_to_all_group_name = None

        select_experts_dict = {}
        token_dispatcher_dict = {}

        self.select_experts = UnquantizedSelectExperts()
        self.token_dispatcher = UnquantizedTokenDispatcherWithMC2()


    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod,
              self).process_weights_after_loading(layer)
        layer.w13_weight = torch.nn.Parameter(self._maybe_pad_weight(
            layer.w13_weight.data),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(self._maybe_pad_weight(
            layer.w2_weight.data),
                                             requires_grad=False)

    def apply(self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
        enable_force_load_balance: bool = False,
        shared_experts: Optional[Any] = None,
        log2phy=None,
        **kwargs,
    ) -> torch.Tensor:
        self.select_experts

        self.token_dispatcher.token_permutation()

        apply_mlp()

        self.token_dispatcher.token_unpermutation()