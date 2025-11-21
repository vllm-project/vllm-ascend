# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch
import torch_npu
from omni.models.common.config.model_config import model_extra_config
from vllm.distributed import (get_dp_group, get_ep_group, get_pp_group,
                              get_tp_group, get_world_group)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.utils import set_weight_attrs


class FusedMoeWeightScaleSupported(Enum):
    CHANNEL = 'channel'


@dataclass
class FusedMoEParallelConfig:
    tp_size: int
    dp_size: int
    ep_size: int
    tp_rank: int
    dp_rank: int
    ep_rank: int

    use_ep: bool  # whether to use EP or not

    @staticmethod
    def make(tp_size_: int, dp_size_: int) -> 'FusedMoEParallelConfig':
        rank = get_ep_group().rank_in_group
        world_size = get_ep_group().world_size
        return FusedMoEParallelConfig(tp_size=1,
                                      tp_rank=0,
                                      dp_size=world_size,
                                      dp_rank=rank,
                                      ep_size=world_size,
                                      ep_rank=rank,
                                      use_ep=True)


@dataclass
class MoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    intermediate_size_per_partition: int

    num_local_experts: int
    moe_parallel_config: FusedMoEParallelConfig

    in_dtype: torch.dtype  # the activation type

    block_size: int = 128

    max_num_tokens: Optional[int] = None

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep


class FusedMoEMethodBase(QuantizeMethodBase):

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        raise NotImplementedError

    @abstractmethod
    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              router_logits: torch.Tensor,
              top_k: int,
              renormalize: bool,
              use_grouped_topk: bool = False,
              topk_group: Optional[int] = None,
              num_expert_group: Optional[int] = None,
              global_num_experts: int = -1,
              expert_range: List[int] = None,
              custom_routing_function: Optional[Callable] = None,
              scoring_func: str = 'softmax',
              e_score_correction_bias: Optional[torch.Tensor] = None,
              apply_router_weight_on_input: bool = False,
              activation: str = 'silu') -> torch.Tensor:
        raise NotImplementedError


class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
    """MoE method without quantization."""

    def __init__(self, moe: MoEConfig):
        super().__init__()
        self.moe = moe
        if model_extra_config.operator_opt_config.use_prefetch:
            self.w13_prefetch_size = model_extra_config.operator_opt_config.expert_gate_up_prefetch * 1024 * 1024  # 24
            self.w2_prefetch_size = model_extra_config.operator_opt_config.expert_down_prefetch * 1024 * 1024  # 12

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter('w13_weight', w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter('w2_weight', w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        w13 = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        w2 = layer.w2_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)

        if model_extra_config.operator_opt_config.gmm_nz:
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight.data, 29).contiguous()
            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight.data, 29).contiguous()

    def alltoall_prefill(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            router_logits: torch.Tensor,
            top_k: int,
            renormalize: bool,
            use_grouped_topk: bool = False,
            topk_group: Optional[int] = None,
            num_expert_group: Optional[int] = None,
            global_num_experts: int = -1,
            expert_range: List[int] = None,
            custom_routing_function: Optional[Callable] = None,
            scoring_func: str = 'softmax',
            e_score_correction_bias: Optional[torch.Tensor] = None,
            apply_router_weight_on_input: bool = False,
            activation: str = 'silu'):
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=None)
        topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.int()
        expanded_x, expanded_row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
            x,
            topk_ids,
            scale=None,
            offset=None,
            active_num=topk_ids.numel(),
            expert_num=global_num_experts,
            expert_capacity=-1,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, global_num_experts],
            quant_mode=-1,
            row_idx_type=0,
        )

        import torch.distributed as dist
        tokens_per_expert_group = tokens_per_expert.new_empty(
            tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)

        combine_tokens = torch.stack(
            [tokens_per_expert_group, tokens_per_expert], dim=0)
        ep_size = get_ep_group().world_size

        combine_tokens = combine_tokens.view(2, ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        # all2all input splits, 大小为当前rank路由到其它rank的token数总和
        input_splits = combine_tokens_cpu[1]
        # all2all output splits, 每个rank拿到的其它卡的token数
        output_splits = combine_tokens_cpu[0]
        # all2all output, 展开成一维，大小为其它卡路由到当前rank的token数总和
        gathered_tokens = expanded_x.new_empty(all_tokens.item(),
                                               expanded_x.shape[1])
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits,
                               input_splits)

        # 按专家归并后的tokens， 按专家归并后的scales, 给FinalizeRouting用的索引, 每个专家处理的token数
        hidden_states_sorted_by_experts, _, gathered_idxs_unsort, tokens_per_local_expert = torch_npu.npu_moe_re_routing(
            gathered_tokens, tokens_per_expert_group.view(ep_size, -1))
        group_list = tokens_per_local_expert.to(torch.int64)

        gate_up_proj = torch_npu.npu_grouped_matmul(
            [hidden_states_sorted_by_experts], [layer.w13_weight],
            bias=None,
            group_list=group_list,
            split_item=3,
            output_dtype=x.dtype,
            group_type=0,
            group_list_type=1)[0]

        inter_states = torch_npu.npu_swiglu(gate_up_proj)

        hidden_states_ordered_by_experts = torch_npu.npu_grouped_matmul(
            [inter_states], [layer.w2_weight],
            bias=None,
            group_list=group_list,
            split_item=3,
            output_dtype=x.dtype,
            group_type=0,
            group_list_type=1)[0]

        new_x = torch_npu.npu_moe_finalize_routing(
            hidden_states_ordered_by_experts.float(),
            skip1=None,
            skip2=None,
            bias=None,
            scales=None,
            expanded_src_to_dst_row=gathered_idxs_unsort.to(
                torch.float32).argsort().to(torch.int32),
            export_for_source_row=None,
            drop_pad_mode=2)

        new_x = new_x.to(torch.bfloat16)
        gathered_tokens = new_x.new_empty(*expanded_x.shape)

        dist.all_to_all_single(gathered_tokens, new_x, input_splits,
                               output_splits)

        y = torch_npu.npu_moe_finalize_routing(
            expanded_permuted_rows=gathered_tokens,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,  # 数据类型要求与y一致
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=2)

        return y

    def apply_all2all_decode(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = 'softmax',
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=None)
        topk_ids = topk_ids.int()

        if model_extra_config.operator_opt_config.use_prefetch:
            flag_expert_prefetch = topk_ids
            if self.w13_prefetch_size > 0:
                torch_npu.npu_prefetch(layer.w13_weight, flag_expert_prefetch,
                                       self.w13_prefetch_size)
            if self.w2_prefetch_size > 0:
                torch_npu.npu_prefetch(layer.w2_weight, flag_expert_prefetch,
                                       self.w2_prefetch_size)

        tp_world_size = 1
        expand_x, dynamic_scales, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts, expand_scales = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=topk_ids,
            group_ep=layer.moe_all_to_all_group_name,
            group_tp=layer.moe_rs_group_name,
            ep_world_size=layer.all2all_world_size,
            tp_world_size=tp_world_size,
            ep_rank_id=layer.all2all_global_rank // tp_world_size,
            tp_rank_id=layer.all2all_global_rank % tp_world_size,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=global_num_experts,
            scales=None,
            quant_mode=0,  # 0: 非量化; 1: 静态量化; 2: 动态量化
            global_bs=0)
        group_list = expert_token_nums.to(torch.int64)

        gate_up_proj = torch_npu.npu_grouped_matmul([expand_x],
                                                    [layer.w13_weight],
                                                    bias=None,
                                                    group_list=group_list,
                                                    split_item=3,
                                                    output_dtype=x.dtype,
                                                    group_type=0,
                                                    group_list_type=1)[0]
        inter_states = torch_npu.npu_swiglu(gate_up_proj)
        hidden_states_ordered_by_experts = torch_npu.npu_grouped_matmul(
            [inter_states], [layer.w2_weight],
            bias=None,
            group_list=group_list,
            split_item=3,
            output_dtype=x.dtype,
            group_type=0,
            group_list_type=1)[0]

        output_combine = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=hidden_states_ordered_by_experts,
            expert_ids=topk_ids,
            assist_info_for_combine=expand_idx,
            ep_send_counts=ep_recv_counts,
            tp_send_counts=tp_recv_counts,
            expert_scales=topk_weights.to(torch.float32),
            group_ep=layer.moe_all_to_all_group_name,
            group_tp=layer.moe_rs_group_name,
            ep_world_size=layer.all2all_world_size,
            tp_world_size=tp_world_size,
            ep_rank_id=layer.all2all_global_rank // tp_world_size,
            tp_rank_id=layer.all2all_global_rank % tp_world_size,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=global_num_experts,
            global_bs=0,
        )

        return output_combine

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_range: List[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = 'softmax',
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = 'silu',
        is_prefill: bool = False,
    ) -> torch.Tensor:
        if is_prefill:
            if model_extra_config.operator_opt_config.prefill_moe_all_to_all:
                return self.alltoall_prefill(
                    layer, x, router_logits, top_k, renormalize,
                    use_grouped_topk, topk_group, num_expert_group,
                    global_num_experts, expert_range, custom_routing_function,
                    scoring_func, e_score_correction_bias,
                    apply_router_weight_on_input, activation)
            else:
                assert len(x.shape) == 2
                assert len(router_logits.shape) == 2
                n_tokens = x.shape[0]
                n_tokens_tensor = torch.Tensor([n_tokens]).int().npu()
                n_tokens_list = get_ep_group().all_gather(n_tokens_tensor,
                                                          dim=0).tolist()
                x_output_list = [
                    torch.empty((n, x.shape[1]),
                                dtype=x.dtype,
                                device=x.device) for n in n_tokens_list
                ]
                router_logits_output_list = [
                    torch.empty((n, router_logits.shape[1]),
                                dtype=router_logits.dtype,
                                device=router_logits.device)
                    for n in n_tokens_list
                ]
                get_ep_group().all_gather_v(x_output_list, x)
                get_ep_group().all_gather_v(router_logits_output_list,
                                            router_logits)
                x = torch.cat(x_output_list)
                router_logits = torch.cat(router_logits_output_list)
        else:
            if model_extra_config.operator_opt_config.decode_moe_dispatch_combine:
                return self.apply_all2all_decode(
                    layer, x, router_logits, top_k, renormalize,
                    use_grouped_topk, topk_group, num_expert_group,
                    global_num_experts, custom_routing_function, scoring_func,
                    e_score_correction_bias)
            else:
                x = get_ep_group().all_gather(x, dim=0)
                router_logits = get_ep_group().all_gather(router_logits, dim=0)

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=None)
        sorted_tokens, expanded_x_idx, expert_tokens, _ = torch_npu.npu_moe_init_routing_v2(
            x,
            topk_ids,
            scale=None,
            offset=None,
            active_num=topk_ids.numel(),
            expert_num=global_num_experts,
            expert_capacity=-1,
            drop_pad_mode=0,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=expert_range,
            quant_mode=-1,
            row_idx_type=1)
        gate_up_proj = torch_npu.npu_grouped_matmul(
            [sorted_tokens], [layer.w13_weight],
            bias=None,
            group_list=expert_tokens,
            split_item=3,
            output_dtype=sorted_tokens.dtype,
            group_type=0,
            group_list_type=1)[0]
        x = torch_npu.npu_swiglu(gate_up_proj)
        y = torch_npu.npu_grouped_matmul([x], [layer.w2_weight],
                                         bias=None,
                                         group_list=expert_tokens,
                                         split_item=3,
                                         output_dtype=x.dtype,
                                         group_type=0,
                                         group_list_type=1)[0]

        # 将不在本rank的专家的topk_weights置为0
        valid_mask = (topk_ids >= expert_range[0]) & (topk_ids
                                                      < expert_range[1])
        topk_weights = topk_weights * valid_mask.to(topk_weights.dtype)

        topk = topk_weights.shape[1]
        tmp_n_tokens = topk_weights.shape[0]
        token_ids = expanded_x_idx // topk
        expert_ids = expanded_x_idx % topk
        target_indices = expert_ids * tmp_n_tokens + token_ids
        another_expanded_idx = torch.zeros_like(expanded_x_idx)
        another_expanded_idx.scatter_(dim=0,
                                      index=target_indices.to(torch.long),
                                      src=torch.arange(
                                          expanded_x_idx.shape[0],
                                          dtype=expanded_x_idx.dtype,
                                          device=expanded_x_idx.device))

        y = torch_npu.npu_moe_finalize_routing(
            y,
            None,
            None,
            None,
            topk_weights,  # 数据类型要求与y一致
            another_expanded_idx,
            topk_ids,
        )

        if is_prefill:
            assert len(y.shape) == 2
            y_list = list(torch.split(y, n_tokens_list))
            y_output = torch.empty((n_tokens, y.shape[1]),
                                   dtype=y.dtype,
                                   device=y.device)
            get_ep_group().reduce_scatter_v(y_output, y_list)
            y = y_output
        else:
            y = get_ep_group().reduce_scatter(y)

        return y


def determine_expert_range(ep_size: int, ep_rank: int,
                           global_num_experts: int) -> Tuple[int, List[int]]:
    """
    Calculates the start expert idx and end expert idx for this tp rank.

    Args:
        ep_size (int): The size of the expert parallel group.
        global_num_experts (int): The total number of experts in the model.
    """
    assert ep_size > 0
    if ep_size == 1:
        return (0, global_num_experts)

    local_num_experts = global_num_experts // ep_size
    start_expert_id = ep_rank * local_num_experts
    end_expert_id = start_expert_id + local_num_experts

    return local_num_experts, [start_expert_id, end_expert_id]


class FusedMoE(torch.nn.Module):
    """
    FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    ZERO_CORRECTION_BIAS = None

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
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
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ):
        super().__init__()
        self.prefix = prefix

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        assert tp_size is None or tp_size == 1
        self.moe_parallel_config: FusedMoEParallelConfig = (
            FusedMoEParallelConfig.make(
                tp_size_=1,
                dp_size_=(dp_size if dp_size is not None else
                          get_tp_group().world_size)))

        self.global_num_experts = num_experts

        moe_dispatch_combine = os.environ.get('MOE_DISPATCH_COMBINE',
                                              '0') == '1'
        if moe_dispatch_combine:
            # 适配dispatch_combine算子
            self.all2all_ep_size = get_ep_group().world_size
            self.all2all_global_rank = get_world_group().rank_in_group
            self.all2all_world_size = get_world_group().world_size

            self.moe_all_to_all_group = get_world_group().device_group
            self.moe_all_to_all_group_name = self.moe_all_to_all_group._get_backend(
                torch.device('npu')).get_hccl_comm_name(
                    self.all2all_global_rank)
            self.moe_rs_group = get_pp_group().device_group
            self.moe_rs_group_rank = get_pp_group().rank_in_group
            self.moe_rs_group_name = self.moe_rs_group._get_backend(
                torch.device('npu')).get_hccl_comm_name(self.moe_rs_group_rank)

        # Determine expert maps
        if self.use_ep:
            self.local_num_experts, self.expert_range = determine_expert_range(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts)
        else:
            self.local_num_experts, self.expert_range = (
                self.global_num_experts, None)

        self.top_k = top_k

        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")

        moe = MoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            intermediate_size_per_partition=self.
            intermediate_size_per_partition,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            # TODO (bnell): this needs to be fixed for quantized types.
            in_dtype=params_dtype,
            max_num_tokens=None,
        )
        self.moe_config = moe
        self.quant_config = quant_config

        if FusedMoE.ZERO_CORRECTION_BIAS is None:
            FusedMoE.ZERO_CORRECTION_BIAS = torch.zeros(
                (self.global_num_experts, ), dtype=torch.float, device='npu')

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        quant_method: Optional[QuantizeMethodBase] = None

        if quant_config is None or prefix in quant_config.ignore:
            quant_method = UnquantizedFusedMoEMethod(moe)
        else:
            quant_method = quant_config.get_quant_method(self, prefix)

        assert quant_method is not None
        assert isinstance(quant_method, FusedMoEMethodBase)
        self.quant_method = quant_method

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
        }

        self.quant_method.create_weights(layer=self, **moe_quant_params)

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    def _load_model_weight_or_group_weight_scale(self,
                                                 shard_dim: int,
                                                 expert_data: torch.Tensor,
                                                 shard_id: str,
                                                 loaded_weight: torch.Tensor,
                                                 tp_rank: int,
                                                 load_full_w2: bool = False):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(shard_dim=shard_dim,
                          loaded_weight=loaded_weight,
                          expert_data=expert_data,
                          tp_rank=tp_rank,
                          load_full=load_full_w2)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def _load_per_channel_weight_scale(self, expert_data: torch.Tensor,
                                       shard_dim: int, shard_id: str,
                                       loaded_weight: torch.Tensor,
                                       tp_rank: int):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def _load_w13(self, expert_data: torch.Tensor, shard_dim: int,
                  shard_id: str, loaded_weight: torch.Tensor, tp_rank: int):

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        shard_size = expert_data.shape[shard_dim] // 2
        loaded_weight = loaded_weight.narrow(shard_dim, shard_size * tp_rank,
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
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(shard_dim,
                                                 shard_size * tp_rank,
                                                 shard_size)
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int):
        if self.expert_range is not None:
            if expert_id < self.expert_range[
                    0] or expert_id >= self.expert_range[1]:
                return None
            else:
                return expert_id - self.expert_range[0]
        else:
            return expert_id

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int) -> None:

        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id is None:
            return
        if shard_id not in ('w1', 'w2', 'w3'):
            raise ValueError(
                f"shard_id must be in ['w1', 'w2', 'w3'], but got {shard_id}")

        WEIGHT_SCALE_SUPPORTED = [
            e.value for e in FusedMoeWeightScaleSupported
        ]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        expert_data = param.data if full_load else param.data[expert_id]

        # Case weight scales
        if "scale" in weight_name:
            self._load_per_channel_weight_scale(shard_id=shard_id,
                                                shard_dim=shard_dim,
                                                loaded_weight=loaded_weight,
                                                expert_data=expert_data,
                                                tp_rank=self.tp_rank)
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank)
            return

    @staticmethod
    def select_experts(hidden_states: torch.Tensor,
                       router_logits: torch.Tensor,
                       top_k: int,
                       use_grouped_topk: bool,
                       renormalize: bool,
                       topk_group: Optional[int] = None,
                       num_expert_group: Optional[int] = None,
                       custom_routing_function: Optional[Callable] = None,
                       scoring_func: str = "softmax",
                       e_score_correction_bias: Optional[torch.Tensor] = None,
                       routed_scaling_factor: Optional[torch.Tensor] = None,
                       indices_type: Optional[torch.dtype] = None):
        # DeepSeekV2 uses grouped_top_k
        if e_score_correction_bias is None:
            e_score_correction_bias = FusedMoE.ZERO_CORRECTION_BIAS
        if use_grouped_topk:
            if topk_group is None or num_expert_group is None:
                raise ValueError(
                    "topk_group and num_expert_group cannot be None when use_grouped_topk."
                )
            if renormalize:
                raise ValueError("renormalize cannot be True.")
            if scoring_func not in ['softmax', 'sigmoid']:
                raise ValueError(
                    f"Unsupported scoring function {scoring_func}.")
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits.float(),
                k=top_k,
                bias=e_score_correction_bias.float()
                if e_score_correction_bias is not None else None,
                k_group=topk_group,
                group_count=num_expert_group,
                group_select_mode=1,
                renorm=1 if renormalize else 0,
                norm_type=1 if scoring_func == 'sigmoid' else 0,
                routed_scaling_factor=routed_scaling_factor or 1.0,
                eps=float(1e-20))
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)
        elif custom_routing_function is None:
            if scoring_func != 'softmax':
                raise ValueError("Only supported softmax.")

            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k_softmax(
                router_logits.float(), k=top_k)
            if renormalize:
                topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)

        return topk_weights, topk_ids

    def forward(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                is_prefill: bool = False):

        assert self.quant_method is not None

        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_range=self.expert_range,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            is_prefill=is_prefill)

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
            cls, ckpt_gate_proj_name: str, ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int) -> list[tuple[str, str, int, str]]:

        return [
            # (param_name, weight_name, expert_id, shard_id)
            ("experts.w13_" if weight_name
             in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.w2_",
             f"experts.{expert_id}.{weight_name}.", expert_id, shard_id)
            for expert_id in range(num_experts) for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def extra_repr(self) -> str:

        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"ep_size={self.ep_size}, "
            f"renormalize={self.renormalize}, "
            f"use_grouped_topk={self.use_grouped_topk}")

        if self.use_grouped_topk:
            s += f", num_expert_group={self.num_expert_group}, topk_group={self.topk_group}"  # noqa: E501

        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"  # noqa: E501

        return s
