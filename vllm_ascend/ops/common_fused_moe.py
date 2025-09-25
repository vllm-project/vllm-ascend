#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os.path
from typing import Callable, Optional

import torch
import torch_npu
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.distributed import (get_dp_group, get_ep_group, get_tp_group,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod, determine_expert_map)
from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.eplb.core.eplb_utils import (determine_default_expert_map,
                                              determine_default_log2phy_map)
from vllm_ascend.ops.expert_load_balancer import ExpertLoadBalancer
from vllm_ascend.ops.moe.experts_selector import select_experts
from vllm_ascend.ops.moe.moe_comm_method import setup_moe_comm_method
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_310p, npu_stream_switch

original_unquantized_fused_moe_init_func = UnquantizedFusedMoEMethod.__init__


def unquantized_fused_moe_init_func(self, *args, **kwargs):
    original_unquantized_fused_moe_init_func(self, *args, **kwargs)

    # NOTE: Currently, this self.use_aclgraph is only used in
    # UnquantizedFusedMoEMethod.forward_oot to decide whether to use in
    # ops/fused_moe.py:568 to circumvent torch.randint_like not supported issue.
    # Once torch.randint_like is supported or removed, this flag can be removed.
    vllm_config = get_current_vllm_config()
    ascend_config = get_ascend_config()
    if ascend_config.torchair_graph_config.enabled:
        self.use_aclgraph = False
    else:
        self.use_aclgraph = (vllm_config.compilation_config.level
                             == CompilationLevel.PIECEWISE
                             and not vllm_config.model_config.enforce_eager)
    self.transpose = True


def forward_oot(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None) -> torch.Tensor:

    topk_weights, topk_ids, row_idx = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        global_num_experts=global_num_experts)

    moe_comm_method = get_forward_context().moe_comm_method
    return moe_comm_method.fused_experts(hidden_states=x,
                                         w1=layer.w13_weight,
                                         w2=layer.w2_weight,
                                         topk_weights=topk_weights,
                                         topk_ids=topk_ids,
                                         row_idx=row_idx,
                                         global_num_experts=global_num_experts,
                                         expert_map=expert_map)


def process_weights_after_loading(self, layer):
    super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)
    if self.transpose:
        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(
            1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(
            1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

        self.transpose = False
    else:
        w13_data = self._maybe_pad_weight(layer.w13_weight.data)
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data)
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

    if not is_310p():
        layer.w13_weight.data = torch_npu.npu_format_cast(
            layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
        layer.w2_weight.data = torch_npu.npu_format_cast(
            layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)


class AscendFusedMoE(FusedMoE):
    moe_counter = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        AscendFusedMoE.moe_counter += 1
        self.moe_instance_id = AscendFusedMoE.moe_counter
        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.mc2_group = get_mc2_group()
        ascend_config = get_ascend_config()
        self.dynamic_eplb = ascend_config.dynamic_eplb
        self.expert_map_path = ascend_config.expert_map_path
        self.global_redundant_expert_num = ascend_config.init_redundancy_expert
        # static eplb initializing with expert_map_path
        if self.expert_map_path and os.path.exists(
                self.expert_map_path) and os.access(self.expert_map_path,
                                                    os.R_OK):
            self.expert_load_balancer = ExpertLoadBalancer(
                self.expert_map_path, self.global_num_experts)
            self.local_num_experts, self.expert_map = (
                self.expert_load_balancer.get_rank_placement_map(
                    self.moe_instance_id, self.ep_rank))
            self.log2phy = self.expert_load_balancer.get_rank_log2phy_map(
                self.moe_instance_id, self.ep_rank).npu()
            self.global_redundant_expert_num = (
                self.expert_load_balancer.get_global_redundant_expert_num())
        else:
            # init moe.
            self.local_num_experts, self.expert_map = determine_expert_map(
                self.ep_size, self.ep_rank, self.global_num_experts)
            # dynamic eplb initializing with not expert_map_path
            if self.dynamic_eplb:
                self.global_redundant_expert_num = ascend_config.init_redundancy_expert
                self.local_num_experts, self.expert_map = determine_default_expert_map(
                    self.global_num_experts, self.ep_size, self.ep_rank,
                    self.global_redundant_expert_num)
                self.log2phy = determine_default_log2phy_map(
                    self.global_num_experts, self.ep_size, self.ep_rank,
                    self.global_redundant_expert_num)
        local_num_experts = (torch.sum(
            self.expert_map != -1) if self.expert_map is not None else
                             self.global_num_experts)
        if self.dynamic_eplb:
            self.moe_load = torch.zeros(local_num_experts, dtype=torch.int64)

        setup_moe_comm_method(self.moe_config)

    def update_expert_map(self, new_expert_map):
        self.expert_map = new_expert_map

    def get_map(self):
        return self.expert_map

    def get_log2phy_map(self):
        return self.logical_to_physical_map

    def clear_moe_load(self):
        if self.moe_load is not None:
            self.moe_load.zero_()

    def maybe_all_reduce_tensor_model_parallel(
            self, final_hidden_states: torch.Tensor):
        """NOTE(Yizhou): This is to override the parent class method. In `mc2commimpl`,
        and `alltoallcommimpl`, we do not need to all-reduce the final outputs since
        the outputs are already aggregated across tensor parallel ranks in the
        `finalize` function. In `allgathercommimpl`, we still need to all-reduce the
        outputs since each rank only has partial outputs.
        """
        forward_context = get_forward_context()
        moe_comm_type = forward_context.moe_comm_type
        if moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2}:
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        assert self.quant_method is not None

        forward_context = get_forward_context()
        hidden_states, router_logits = forward_context.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=forward_context.sp_enabled)

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            enable_eplb=self.enable_eplb,
            expert_load_view=self.expert_load_view,
            logical_to_physical_map=self.logical_to_physical_map,
            logical_replica_count=self.logical_replica_count,
        )
        if isinstance(final_hidden_states, tuple):
            final_hidden_states, group_list_type, expert_tokens = final_hidden_states

        if self.dynamic_eplb:
            self.moe_load += expert_tokens if group_list_type else \
                torch.cat([expert_tokens[:1], expert_tokens[1:] - expert_tokens[:-1]])

        final_hidden_states = forward_context.moe_comm_method.finalize(
            hidden_states=final_hidden_states,
            reduce_results=self.reduce_results)

        return final_hidden_states

    def transpose_weight(self, loaded_weight, expert_data, shard_dim):
        # Ensure training and inference weight shapes match during RL weight updates
        if (
            loaded_weight.shape[1] != expert_data.shape[1] and \
            loaded_weight.shape[0] != expert_data.shape[0]
        ):
            shard_dim = int(not shard_dim)
            loaded_weight = loaded_weight.transpose(0, 1).contiguous()
        return loaded_weight, shard_dim

    def _load_w13(self,
                  expert_data: torch.Tensor,
                  shard_dim: int,
                  shard_id: str,
                  loaded_weight: torch.Tensor,
                  tp_rank: int,
                  load_full: bool = False):
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        loaded_weight, shard_dim = self.transpose_weight(
            loaded_weight, expert_data, shard_dim)
        shard_size = expert_data.shape[shard_dim] // 2
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim,
                                                 shard_size * tp_rank,
                                                 shard_size)
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(self,
                 expert_data: torch.Tensor,
                 shard_dim: int,
                 loaded_weight: torch.Tensor,
                 tp_rank: int,
                 load_full: bool = False):
        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        loaded_weight, shard_dim = self.transpose_weight(
            loaded_weight, expert_data, shard_dim)
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim,
                                                 shard_size * tp_rank,
                                                 shard_size)
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)


class AscendSharedFusedMoE(SharedFusedMoE, AscendFusedMoE):

    def __init__(
        self,
        shared_experts: torch.nn.Module,
        use_overlapped: bool = True,
        **kwargs,
    ):
        AscendFusedMoE.__init__(self, **kwargs)
        self._shared_experts = shared_experts
        self.use_overlapped = use_overlapped
        self.shared_expert_stream = None
        ascend_config = get_ascend_config()
        self.multistream_overlap_shared_expert = ascend_config.multistream_overlap_shared_expert
        if self.multistream_overlap_shared_expert:
            self.shared_expert_stream = torch.npu.Stream()

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Make sure the shared experts stream begins after hidden_states are ready.
        if self.multistream_overlap_shared_expert:
            self.shared_expert_stream.wait_stream(  # type: ignore
                torch.npu.current_stream())
        with npu_stream_switch(self.shared_expert_stream,
                               enabled=self.multistream_overlap_shared_expert):
            # Use a separate stream to run shared experts.
            shared_out = self._shared_experts(hidden_states)

            # NOTE: This is exactly the opposite of `maybe_all_reduce_tensor_model_parallel`
            forward_context = get_forward_context()
            moe_comm_type = forward_context.moe_comm_type
            if moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2}:
                shared_out = tensor_model_parallel_all_reduce(shared_out)

        _, fused_out = AscendFusedMoE.forward(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        # Make sure the default stream waits for the shared experts stream to finish.
        if self.multistream_overlap_shared_expert:
            torch.npu.current_stream().wait_stream(self.shared_expert_stream)
        return shared_out, fused_out

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        shared_output = torch.empty(1)
        fused_output = AscendFusedMoE.forward_impl(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return shared_output, fused_output


UnquantizedFusedMoEMethod.__init__ = unquantized_fused_moe_init_func
UnquantizedFusedMoEMethod.process_weights_after_loading = process_weights_after_loading
UnquantizedFusedMoEMethod.forward_oot = forward_oot
