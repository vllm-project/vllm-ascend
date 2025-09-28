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
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.distributed import (get_context_model_parallel_world_size,
                              get_tensor_model_parallel_rank,
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

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import FusedMoEState
from vllm_ascend.distributed.parallel_state import get_mc2_group, is_sp_enabled
from vllm_ascend.ops.expert_load_balancer import ExpertLoadBalancer
from vllm_ascend.ops.moe.experts_selector import select_experts
from vllm_ascend.ops.moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.sequence_parallel import MetadataForPadding
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, dispose_tensor,
                               get_all_reduce_merge_state,
                               get_rm_router_logits_state, is_310p)


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
                                with_quant: bool = False,
                                fusion_mlp: bool = False):
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
        with_quant=with_quant,
        fusion=fusion_mlp)
    final_hidden_states = token_dispatcher.token_combine(expert_output)
    return final_hidden_states


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self, moe: FusedMoEConfig = None):

        super().__init__(moe=moe)
        vllm_config = get_current_vllm_config()

        self.global_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len
        get_ascend_config()

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
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
        enable_force_load_balance: bool = False,
        shared_experts: Optional[Any] = None,
        **kwargs,
    ) -> torch.Tensor:

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

        topk_weights = topk_weights.to(x.dtype)
        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance and not self.use_aclgraph:
            topk_ids = torch.randint_like(topk_ids, 0, global_num_experts)

        return unified_fused_experts_eager(hidden_states=x,
                                           w1=layer.w13_weight,
                                           w2=layer.w2_weight,
                                           topk_weights=topk_weights,
                                           topk_ids=topk_ids,
                                           row_idx=row_idx,
                                           expert_map=expert_map,
                                           shared_experts=shared_experts,
                                           mc2_mask=kwargs.get(
                                               "mc2_mask", None),
                                           with_quant=False)


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
            cp_size: Optional[int] = None,
            prefix: str = "",
            custom_routing_function: Optional[Callable] = None,
            scoring_func: str = "softmax",
            e_score_correction_bias: Optional[torch.Tensor] = None,
            activation: str = "silu",
            apply_router_weight_on_input: bool = False):
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
            cp_size_=(cp_size if cp_size is not None else
                      get_context_model_parallel_world_size()),
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
        expert_map_path = ascend_config.expert_map_path
        if expert_map_path and os.path.exists(expert_map_path):
            # moe expert load balance
            expert_load_balancer = ExpertLoadBalancer(expert_map_path,
                                                      self.global_num_experts)
            self.local_num_experts, self.expert_map = \
                                expert_load_balancer.get_rank_placement_map(
                                                self.moe_instance_id,
                                                get_ep_group().rank_in_group)
            self.log2phy = expert_load_balancer.get_rank_log2phy_map(
                self.moe_instance_id,
                get_ep_group().rank_in_group)
            self.global_redundant_expert_num = \
                        expert_load_balancer.get_global_redundant_expert_num()
        else:
            # Create a tensor of size num_experts filled with -1
            self.local_num_experts, self.expert_map = determine_expert_map(
                self.ep_size,
                get_ep_group().rank_in_group, self.global_num_experts)

        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")
        moe = FusedMoEConfig.make(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            # TODO (bnell): this needs to be fixed for quantized types.
            in_dtype=params_dtype,
            quant_config=quant_config)

        self.moe_config = moe

        if quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod(moe)
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)

        assert self.quant_method is not None

        local_num_experts = torch.sum(self.expert_map != -1) \
            if self.expert_map is not None else num_experts

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
        self.token_dispatcher = None

        ep_size = (get_ep_group().world_size if
                   vllm_config.parallel_config.enable_expert_parallel else 1)
        from vllm_ascend.ops.moe.token_dispatcher import \
            setup_token_dispatchers
        setup_token_dispatchers(
            ep_size,
            top_k=self.top_k,
            num_experts=self.global_num_experts,
            num_global_redundant_experts=self.global_redundant_expert_num,
            num_local_experts=self.local_num_experts)

    def naive_multicast(self, x: torch.Tensor,
                        cu_tokens_across_dp_cpu: torch.Tensor):
        assert (len(x.shape) == 2)
        buffer = torch.empty((cu_tokens_across_dp_cpu[-1], x.size(1)),
                             device=x.device,
                             dtype=x.dtype)
        start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
            self.dp_rank - 1]
        end = cu_tokens_across_dp_cpu[self.dp_rank]
        buffer[start:end, :].copy_(x)
        for idx in range(self.dp_size):
            start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
            end = cu_tokens_across_dp_cpu[idx]
            get_dp_group().broadcast(buffer[start:end, :], idx)
        return buffer

    def forward(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                is_prefill: bool,
                enable_force_load_balance: bool = False,
                top_k: Optional[int] = None,
                shared_experts: Optional[Any] = None,
                gate=None,
                replace_allreduce: bool = False,
                _metadata_for_padding: Optional[MetadataForPadding] = None):

        assert self.quant_method is not None

        if top_k:
            real_top_k = top_k
        else:
            real_top_k = self.top_k

        num_tokens, hidden_size = hidden_states.shape

        forward_context = get_forward_context()
        fused_moe_state = forward_context.fused_moe_state
        self.enable_sp = is_sp_enabled()
        mc2_mask = forward_context.mc2_mask
        # For w8a8 dynamic we can do npu_dynamic_quant and gate in parallel.
        quantized_x_for_share, dynamic_scale_for_share = None, None

        if shared_experts:
            # When all_reduce_merge is in progress, shared_experts does not do all_reduce in mlp, but waits until shared_experts+router_experts are completed before doing all_reduce
            shared_hidden_states = shared_experts(hidden_states)

        enable_sp = _metadata_for_padding is not None and _metadata_for_padding.not_dummy_and_is_prefill
        tp_size = get_tensor_model_parallel_world_size()
        # NOTE enable_sp: qwen3-moe/gqa sp, self.enable_sp: deepseek-v2/mla sp
        # TODO need to use the same enable_sp switch for deepseek & qwen3-moe sp
        if enable_sp or (self.enable_sp and is_prefill):
            replace_allreduce = True

        if (fused_moe_state not in [
                FusedMoEState.AllGather, FusedMoEState.AllGatherEP,
                FusedMoEState.NaiveMulticast
        ] and not replace_allreduce):
            if fused_moe_state in {FusedMoEState.MC2}:
                padding_size = forward_context.padded_num_tokens
            else:
                # TODO: Determine if we can remove the padding
                padding_size = tp_size
            if num_tokens < padding_size and not self.enable_shared_expert_dp:
                hidden_states = nn.functional.pad(
                    hidden_states, (0, 0, 0, padding_size - num_tokens))
                router_logits = nn.functional.pad(
                    router_logits, (0, 0, 0, padding_size - num_tokens))
            if tp_size > 1:
                tp_rank = get_tensor_model_parallel_rank()
                if not self.enable_shared_expert_dp:
                    chunk_hidden_states = torch.tensor_split(hidden_states,
                                                             tp_size,
                                                             dim=0)
                    chunk_router_logits = torch.tensor_split(router_logits,
                                                             tp_size,
                                                             dim=0)
                    hidden_states = chunk_hidden_states[tp_rank]
                    router_logits = chunk_router_logits[tp_rank]

                chunk_mc2_mask = torch.tensor_split(mc2_mask, tp_size, dim=0)
                mc2_mask = chunk_mc2_mask[tp_rank]

        if self.dp_size > 1:
            if fused_moe_state == FusedMoEState.AllGather:
                # NOTE: When in torchair graph, it has been padded in model_runner_v1
                max_tokens_across_dp = forward_context.max_tokens_across_dp
                if num_tokens < max_tokens_across_dp:
                    hidden_states = nn.functional.pad(
                        hidden_states,
                        (0, 0, 0, max_tokens_across_dp - num_tokens))
                    if not self.rm_router_logits:
                        router_logits = nn.functional.pad(
                            router_logits,
                            (0, 0, 0, max_tokens_across_dp - num_tokens))
                hidden_states = get_dp_group().all_gather(hidden_states, 0)
                if self.rm_router_logits:
                    router_logits, _ = gate(hidden_states)
                else:
                    router_logits = get_dp_group().all_gather(router_logits, 0)

            elif fused_moe_state == FusedMoEState.NaiveMulticast:
                cu_tokens_across_dp_cpu = get_forward_context(
                ).dp_metadata.cu_tokens_across_dp_cpu
                hidden_states = self.naive_multicast(hidden_states,
                                                     cu_tokens_across_dp_cpu)
                if self.rm_router_logits:
                    router_logits, _ = gate(hidden_states)
                else:
                    router_logits = self.naive_multicast(
                        router_logits, cu_tokens_across_dp_cpu)

        # Matrix multiply.
        e_hidden_states = self.quant_method.apply(
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
            token_dispatcher=self.token_dispatcher,
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
        )

        if shared_experts:
            if isinstance(e_hidden_states, tuple):
                e_hidden_states, shared_hidden_states = e_hidden_states

        if (fused_moe_state not in [
                FusedMoEState.AllGather, FusedMoEState.AllGatherEP,
                FusedMoEState.NaiveMulticast
        ] and not replace_allreduce and not self.enable_shared_expert_dp):
            if tp_size > 1:
                dist.all_gather(list(chunk_hidden_states), e_hidden_states,
                                self.tp_group)
                final_hidden_states = torch.cat(chunk_hidden_states, dim=0)
                dispose_tensor(e_hidden_states)
            else:
                final_hidden_states = e_hidden_states
            if num_tokens < padding_size:
                final_hidden_states = final_hidden_states[:num_tokens]
        elif self.dp_size > 1 and not self.enable_shared_expert_dp:
            if fused_moe_state == FusedMoEState.NaiveMulticast:
                start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
                    self.dp_rank - 1]
                end = cu_tokens_across_dp_cpu[self.dp_rank]
                final_hidden_states = get_dp_group().all_reduce(
                    e_hidden_states)
                final_hidden_states = final_hidden_states[start:end, :]
                dispose_tensor(e_hidden_states)
            elif fused_moe_state == FusedMoEState.AllGather:
                final_hidden_states = get_dp_group().reduce_scatter(
                    e_hidden_states, 0)
                final_hidden_states = final_hidden_states[:num_tokens]
                dispose_tensor(e_hidden_states)
            else:
                final_hidden_states = e_hidden_states
        else:
            final_hidden_states = e_hidden_states

        if tp_size > 1 and not self.all_reduce_merge and fused_moe_state in [
                FusedMoEState.AllGather, FusedMoEState.AllGatherEP,
                FusedMoEState.NaiveMulticast
        ]:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

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
