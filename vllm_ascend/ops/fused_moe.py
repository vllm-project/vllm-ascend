# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/kernels/test_moe.py

import os
from typing import Any, Callable, Optional, Union

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
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

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.eplb.core.eplb_utils import (determine_default_expert_map,
                                              determine_default_log2phy_map)
from vllm_ascend.ops.expert_load_balancer import ExpertLoadBalancer
from vllm_ascend.ops.moe.experts_selector import select_experts
from vllm_ascend.ops.moe.moe_comm_method import setup_moe_comm_method
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ,
                               get_all_reduce_merge_state,
                               get_rm_router_logits_state, is_310p,
                               vllm_version_is)

def compute_identity_kernel_npu(top_k, hidden_states, expert_scales, output, hidden_dim, scales_stride):
    """
    使用PyTorch操作实现compute_identity_kernel的功能，适用于Ascend NPU。
    """
    try:
        num_tokens = hidden_states.size(0)
        # 将expert_scales扩展为与hidden_states相同的形状
        expert_scales_expanded = expert_scales.view(num_tokens, top_k, 1).expand(-1, -1, hidden_dim)
        # 将hidden_states扩展为与expert_scales相同的形状
        hidden_states_expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1)
        # 计算每个expert的贡献
        expert_contributions = hidden_states_expanded * expert_scales_expanded
        # 求和所有expert的贡献
        output = expert_contributions.sum(dim=1)
        return output
    except Exception as e:
        # 如果出现异常，返回一个零张量作为降级处理
        print(f"Error in compute_identity_kernel_npu: {e}")
        return torch.zeros_like(hidden_states)

def zero_experts_compute_npu(expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states):
    """
    使用PyTorch操作实现zero_experts_compute_triton的功能，适用于Ascend NPU。
    """
    try:
        # 确保所有张量都在同一设备上
        device = hidden_states.device
        expert_indices = expert_indices.to(device)
        expert_scales = expert_scales.to(device)
        
        N = expert_indices.numel()
        top_k = expert_indices.size(-1)
        
        # 创建副本以避免修改原始张量
        expert_indices = expert_indices.clone()
        expert_scales = expert_scales.clone()
        
        if zero_expert_type == "identity":
            # 创建一个掩码，标记哪些专家索引小于num_experts（即需要置零的专家）
            zero_expert_mask = expert_indices < num_experts
            # 将需要置零的专家对应的scale置为0，其他保持不变
            zero_expert_scales = torch.where(zero_expert_mask, 
                                            torch.zeros_like(expert_scales), 
                                            expert_scales)
        else:
            zero_expert_scales = expert_scales

        # 对于normal_expert_mask，我们将不在有效范围内的专家索引和scale都置为0
        normal_expert_mask = expert_indices >= num_experts
        expert_indices = torch.where(normal_expert_mask, 
                                    torch.zeros_like(expert_indices), 
                                    expert_indices)
        expert_scales = torch.where(normal_expert_mask, 
                                   torch.zeros_like(expert_scales), 
                                   expert_scales)
        
        # 初始化输出张量
        output = torch.zeros_like(hidden_states)
        hidden_dim = hidden_states.size(-1)
        num_tokens = hidden_states.size(0)
        
        # 调用NPU优化的实现
        output = compute_identity_kernel_npu(
            top_k,
            hidden_states,
            zero_expert_scales,
            output,
            hidden_dim,
            zero_expert_scales.stride(0)
        )
        
        return output
    except Exception as e:
        # 如果出现异常，返回一个零张量作为降级处理
        print(f"Error in zero_experts_compute_npu: {e}")
        return torch.zeros_like(hidden_states)

def fused_topk_bias(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    """Ascend NPU兼容的带偏置topk选择函数"""
    n_routed_experts = gating_output.shape[-1]
    scores = gating_output.softmax(dim=-1)
    scores_for_choice = scores.view(
        -1, n_routed_experts) + e_score_correction_bias.unsqueeze(0)
    topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1,
                              sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_indices.to(torch.int32)


# 有区别，这里没有像jsb一样改造
def unified_fused_experts_eager(hidden_states: torch.Tensor,
                                w1: torch.Tensor,
                                w2: torch.Tensor,
                                topk_weights: torch.Tensor,
                                topk_ids: torch.Tensor,
                                row_idx: torch.Tensor,
                                expert_map: Optional[torch.Tensor] = None,
                                log2phy: Optional[torch.Tensor] = None,
                                global_redundant_expert_num: int = 0,
                                w1_scale: Optional[torch.Tensor] = None,
                                w1_scale_bias: Optional[torch.Tensor] = None,
                                w2_scale: Optional[torch.Tensor] = None,
                                w2_scale_bias: Optional[torch.Tensor] = None,
                                shared_experts: Optional[torch.Tensor] = None,
                                shared_gate_up: Optional[Any] = None,
                                shared_dequant_scale: Optional[Any] = None,
                                mc2_mask: Optional[torch.Tensor] = None,
                                apply_router_weight_on_input: bool = False,
                                with_quant: bool = False):
    token_dispatcher = get_forward_context().token_dispatcher

    results = token_dispatcher.token_dispatch(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        row_idx=row_idx,
        expert_map=expert_map,
        log2phy=log2phy,
        global_redundant_expert_num=global_redundant_expert_num,
        shared_experts=shared_experts,
        shared_gate_up=shared_gate_up,
        shared_dequant_scale=shared_dequant_scale,
        mc2_mask=mc2_mask,
        apply_router_weight_on_input=apply_router_weight_on_input,
        with_quant=with_quant)

    expert_output = unified_apply_mlp(
        hidden_states=results["hidden_states"],
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        group_list=results["group_list"],
        dynamic_scale=results.get("dynamic_scale"),
        group_list_type=results.get("group_list_type"),
        w1_scale_bias=w1_scale_bias,
        w2_scale_bias=w2_scale_bias,
        topk_scales=results.get("topk_scales"),
        with_quant=with_quant)
    final_hidden_states = token_dispatcher.token_combine(expert_output)
    return final_hidden_states

class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self, moe: FusedMoEConfig = None):

        super().__init__(moe=moe)
        vllm_config = get_current_vllm_config()

        self.global_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len
        get_ascend_config()
        self.dynamic_eplb = get_ascend_config().dynamic_eplb

        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(
                local_rank)
        except AttributeError:
            self.moe_all_to_all_group_name = None

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod,
              self).process_weights_after_loading(layer)
        layer.w13_weight = torch.nn.Parameter(self._maybe_pad_weight(
            layer.w13_weight.data),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(self._maybe_pad_weight(
            layer.w2_weight.data),
                                             requires_grad=False)
        if not is_310p():
            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        zero_expert_num: Optional[int] = 0,
        zero_expert_type: Optional[str] = None,
        is_prefill: bool = False,
        enable_force_load_balance: bool = False,
        shared_experts: Optional[Any] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # 初始化零专家输出
        zero_expert_result = None

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
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts)

        # 处理LongCat Flash的零专家逻辑
        if zero_expert_num != 0 and zero_expert_type is not None:
            zero_expert_result = zero_experts_compute_npu(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=global_num_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        topk_weights = topk_weights.to(x.dtype)
        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance and not self.use_aclgraph:
            topk_ids = torch.randint_like(topk_ids, 0, global_num_experts)

        moe_comm_method = get_forward_context().moe_comm_method
        expert_output = moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            row_idx=row_idx,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            shared_experts=shared_experts,
            need_trans=True,
            dynamic_eplb=self.dynamic_eplb)
        
        # 如果有零专家输出，返回元组；否则返回单个结果
        if zero_expert_result is not None:
            return expert_output, zero_expert_result
        else:
            return expert_output


class AscendFusedMoE(FusedMoE):

    # The moe_counter parameter is required during the initialization of EPLB
    # to identify the current layer index within the MOE model.
    moe_counter = -1

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        routed_scaling_factor: float = 1.0,
        enable_eplb: bool = False,
        zero_expert_num: Optional[int] = 0,
        zero_expert_type: Optional[str] = None,
    ):
        # TODO: This could not initialize FusedMoE baseclass,
        # fixme and make __init__() of AscendFusedMoE more clear
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=reduce_results,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            quant_config=quant_config,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            prefix=prefix,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            routed_scaling_factor=routed_scaling_factor,
            enable_eplb=enable_eplb,
            zero_expert_num=zero_expert_num,
            zero_expert_type=zero_expert_type,
        )
        AscendFusedMoE.moe_counter += 1
        self.moe_instance_id = AscendFusedMoE.moe_counter

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        vllm_config = get_current_vllm_config()

        self.moe_parallel_config = FusedMoEParallelConfig.make(
            tp_size_=(tp_size if tp_size is not None else
                      get_tensor_model_parallel_world_size()),
            dp_size_=(dp_size
                      if dp_size is not None else get_dp_group().world_size),
            vllm_parallel_config=vllm_config.parallel_config)

        self.top_k = top_k
        self.num_experts = num_experts
        self.global_num_experts = num_experts
        assert intermediate_size % self.tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        self.expert_map = None
        self.activation = activation
        self.log2phy = None
        self.global_redundant_expert_num = 0

        is_deepseek_v3_r1 = self.global_num_experts == 256
        self.rm_router_logits = get_rm_router_logits_state(
            self.moe_parallel_config.ep_size, self.dp_size, is_deepseek_v3_r1)
        self.all_reduce_merge = get_all_reduce_merge_state(
            self.moe_parallel_config.ep_size, is_deepseek_v3_r1)

        ascend_config = get_ascend_config()
        self.dynamic_eplb = ascend_config.dynamic_eplb
        self.expert_map_path = ascend_config.expert_map_path
        self.global_redundant_expert_num = ascend_config.init_redundancy_expert
        self.global_num_experts = num_experts + self.global_redundant_expert_num
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
        local_num_experts = (torch.sum(self.expert_map != -1)
                             if self.expert_map is not None else num_experts)
        if self.dynamic_eplb:
            self.moe_load = torch.zeros(local_num_experts, dtype=torch.int64)

        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")
        if vllm_version_is("0.10.2"):
            moe = FusedMoEConfig.make(
                num_experts=self.global_num_experts,
                experts_per_token=top_k,
                hidden_dim=hidden_size,
                num_local_experts=self.local_num_experts,
                moe_parallel_config=self.moe_parallel_config,
                # TODO (bnell): this needs to be fixed for quantized types.
                in_dtype=params_dtype,
                quant_config=quant_config)
        else:
            moe = FusedMoEConfig(
                num_experts=self.global_num_experts,
                experts_per_token=top_k,
                hidden_dim=hidden_size,
                num_local_experts=self.local_num_experts,
                moe_parallel_config=self.moe_parallel_config,
                in_dtype=params_dtype,
            )
        self.moe_config = moe
        # TODO: The self.moe_config.tp_size here is not correct, fixme soon

        if quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod(moe)
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)

        assert self.quant_method is not None

        local_num_experts = torch.sum(self.expert_map != -1) \
            if self.expert_map is not None else num_experts

        self.moe_load = None

        if self.dynamic_eplb:
            self.moe_load = torch.zeros(local_num_experts, dtype=torch.int64)

        moe_quant_params = {
            "num_experts": local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if (self.quant_method.__class__.__name__
                in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.ep_group = get_ep_group()
        # NOTE: self.tp_group is not expert_tp_group
        self.tp_group = get_tp_group().device_group
        self.quant_method.create_weights(layer=self, **moe_quant_params)

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.mc2_group = get_mc2_group()
        self.moe_config.num_global_redundant_experts = self.global_redundant_expert_num

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

    def forward(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                is_prefill: bool,
                enable_force_load_balance: bool = False,
                top_k: Optional[int] = None,
                shared_experts: Optional[Any] = None,
                gate=None,
                replace_allreduce: bool = False):

        assert self.quant_method is not None

        if top_k:
            real_top_k = top_k
        else:
            real_top_k = self.top_k

        forward_context = get_forward_context()
        mc2_mask = forward_context.mc2_mask
        # For w8a8 dynamic we can do npu_dynamic_quant and gate in parallel.
        quantized_x_for_share, dynamic_scale_for_share = None, None

        if shared_experts:
            # When all_reduce_merge is in progress, shared_experts does not do all_reduce in mlp, but waits until shared_experts+router_experts are completed before doing all_reduce
            shared_hidden_states = shared_experts(hidden_states)

        if forward_context.sp_enabled:
            replace_allreduce = True

        hidden_states, router_logits = forward_context.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            enable_shared_expert_dp=self.enable_shared_expert_dp,
            rm_router_logits=self.rm_router_logits,
            replace_allreduce=replace_allreduce,
            gate=gate)

        # Matrix multiply.
        moe_result = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=real_top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            is_prefill=is_prefill,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            shared_experts=None,
            mc2_mask=mc2_mask,
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
            zero_expert_num=self.zero_expert_num,
            zero_expert_type=self.zero_expert_type,
        )
        
        # 处理零专家结果
        zero_expert_result = None
        if isinstance(moe_result, tuple) and len(moe_result) == 2:
            e_hidden_states, zero_expert_result = moe_result
            if zero_expert_result is not None:
                e_hidden_states += (
                    zero_expert_result[:e_hidden_states.size(0)])
        else:
            e_hidden_states = moe_result

        group_list_type = None

        if shared_experts:
            if isinstance(e_hidden_states,
                          tuple) and len(e_hidden_states) == 2:
                e_hidden_states, shared_hidden_states = e_hidden_states

        if isinstance(e_hidden_states, tuple) and len(e_hidden_states) == 3:
            e_hidden_states, group_list_type, expert_tokens = e_hidden_states

        if self.dynamic_eplb and group_list_type is not None:
            self.moe_load += expert_tokens if group_list_type else \
                torch.cat([expert_tokens[:1], expert_tokens[1:] - expert_tokens[:-1]])

        final_hidden_states = forward_context.moe_comm_method.finalize(
            hidden_states=e_hidden_states,
            reduce_results=(not self.all_reduce_merge))

        if shared_experts:
            return final_hidden_states, shared_hidden_states
        else:
            return final_hidden_states

    # ----------------------------------------- TBO-related --------------------------------------------

    def _forward_ms_fused_moe_comp(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_prefill: bool,
        real_top_k,
        enable_force_load_balance: bool = False,
    ):
        hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=real_top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            is_prefill=is_prefill,
            enable_force_load_balance=enable_force_load_balance,
        )

        return hidden_states
