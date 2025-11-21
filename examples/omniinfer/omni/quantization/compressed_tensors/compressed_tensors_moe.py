# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
from typing import Optional

import torch
import torch_npu
from omni.adaptors.vllm.distributed.parallel_state import GroupCoordinator
from omni.models.common.config.model_config import model_extra_config
from omni.models.common.layers.moe.fused_moe.fused_moe import (
    fused_experts_moe_dispatch_combine, fused_experts_w4a8_allgather_ep,
    fused_experts_w8a8_allgather_ep, fused_experts_w8a8_allgather_ep_a2,
    moe_infer_fusion)
from vllm.attention import AttentionMetadata
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import \
    CompressedTensorsMoEMethod
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

SEQ_SPLIT_LENGTH = 4096
torch.npu.config.allow_internal_format = True


class AscendCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsMoEMethod):

    def __init__(self):
        self.warm_up = True
        self.n_routed_experts = None
        self.smooth_scale = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=torch.int8),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=torch.int8),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})

        w13_scale = torch.nn.Parameter(torch.ones(
            num_experts,
            2 * intermediate_size_per_partition,
            dtype=torch.float32
            if params_dtype == torch.float16 else torch.bfloat16),
                                       requires_grad=False)
        w13_offset = torch.nn.Parameter(torch.zeros(
            num_experts,
            2 * intermediate_size_per_partition,
            dtype=torch.float32
            if params_dtype == torch.float16 else torch.bfloat16),
                                        requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_scale)
        layer.register_parameter("w13_weight_offset", w13_offset)
        set_weight_attrs(w13_scale, extra_weight_attrs)
        set_weight_attrs(w13_offset, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(torch.ones(
            num_experts,
            hidden_size,
            dtype=torch.float32
            if params_dtype == torch.float16 else torch.bfloat16),
                                      requires_grad=False)
        w2_offset = torch.nn.Parameter(torch.zeros(
            num_experts,
            hidden_size,
            dtype=torch.float32
            if params_dtype == torch.float16 else torch.bfloat16),
                                       requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_scale)
        layer.register_parameter("w2_weight_offset", w2_offset)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight.transpose(
            1, 2).contiguous(),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight.transpose(
            1, 2).contiguous(),
                                             requires_grad=False)
        if model_extra_config.operator_opt_config.gmm_nz:
            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight, 29)
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight, 29)
        if model_extra_config.operator_opt_config.pd_seperate_prefill:
            layer.w2_weight_scale = torch.nn.Parameter(
                layer.w2_weight_scale.to(torch.bfloat16), requires_grad=False)
        elif not model_extra_config.operator_opt_config.opt_w2_scale_cast:
            layer.w2_weight_scale = torch.nn.Parameter(
                layer.w2_weight_scale.to(torch.float32), requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(layer.w13_weight_scale.to(
            torch.float32),
                                                    requires_grad=False)
        self.n_routed_experts = len(layer.w13_weight)
        self.local_expert_indices_offset = (get_ep_group().rank_in_group *
                                            self.n_routed_experts)
        self.local_expert_indices = [
            self.local_expert_indices_offset + i
            for i in range(self.n_routed_experts)
        ]
        self.smooth_scale = torch.ones(
            (self.n_routed_experts, layer.w13_weight_scale.shape[-1] // 2),
            dtype=torch.float32,
            device=current_platform.device_type)
        torch._dynamo.mark_static(self.smooth_scale)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              topk_weights: torch.Tensor,
              topk_ids: torch.Tensor,
              pertoken_scale: torch.Tensor,
              attn_metadata: AttentionMetadata,
              comm_group: Optional[GroupCoordinator] = None) -> torch.Tensor:
        max_num_deployed_expert_per_rank = self.n_routed_experts
        if model_extra_config.operator_opt_config.use_omni_placement and layer.planner.is_moe_layer(
                layer.moe_layer_idx):
            max_num_deployed_expert_per_rank = layer.planner.get_max_num_deployed_expert_per_rank(
            )

        if get_ep_group().world_size > 1:
            is_prefill = attn_metadata is None or attn_metadata.prefill is not None
            if model_extra_config.operator_opt_config.prefill_moe_all_to_all or (
                    model_extra_config.operator_opt_config.
                    decode_moe_dispatch_combine and not is_prefill):
                if is_prefill:
                    out = moe_infer_fusion(layer,
                                           x,
                                           topk_ids,
                                           topk_weights,
                                           self.warm_up,
                                           is_prefill,
                                           comm_group=comm_group)
                else:
                    out = fused_experts_moe_dispatch_combine(
                        layer,
                        x,
                        topk_weights,
                        topk_ids,
                        max_num_deployed_expert=max_num_deployed_expert_per_rank
                        * get_ep_group().world_size,
                        is_prefill=is_prefill,
                        is_route_expert=True)
            else:
                if os.getenv("ASCEND_PLATFORM", "A3") == "A2":
                    out = fused_experts_w8a8_allgather_ep_a2(
                        hidden_states=x,
                        pertoken_scale=pertoken_scale,
                        w1=layer.w13_weight,
                        w2=layer.w2_weight,
                        w1_scale=layer.w13_weight_scale,
                        w2_scale=layer.w2_weight_scale,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        n_routed_experts=self.n_routed_experts,
                        is_prefill=is_prefill,
                        max_num_deployed_expert_per_rank=
                        max_num_deployed_expert_per_rank,
                        # ENABLE_OMNI_PLANNER
                        smooth_scale=self.smooth_scale)
                else:
                    out = fused_experts_w8a8_allgather_ep(
                        hidden_states=x,
                        pertoken_scale=pertoken_scale,
                        w1=layer.w13_weight,
                        w2=layer.w2_weight,
                        w1_scale=layer.w13_weight_scale,
                        w2_scale=layer.w2_weight_scale,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        n_routed_experts=self.n_routed_experts,
                        is_prefill=is_prefill,
                        max_num_deployed_expert_per_rank=
                        max_num_deployed_expert_per_rank
                        # ENABLE_OMNI_PLANNER
                    )
            if self.warm_up:
                self.warm_up = False
            return out
        else:
            row_idx = torch.arange(topk_ids.numel(),
                                   device=current_platform.device_type,
                                   dtype=torch.int32).view(
                                       -1, x.shape[0]).transpose(0, 1)
            token_num = x.shape[0]
            if token_num > SEQ_SPLIT_LENGTH:  # Split seq to reduce memory usage
                x_list = x.split(SEQ_SPLIT_LENGTH)
                topk_weights_list = topk_weights.split(SEQ_SPLIT_LENGTH)
                topk_ids_list = topk_ids.split(SEQ_SPLIT_LENGTH)
                out = []
                for i in range(len(x_list)):
                    split_token, top_k = topk_weights_list[i].shape
                    row_idx = torch.arange(split_token * top_k).to(
                        torch.int32).view(
                            (top_k, split_token)).T.contiguous().npu()
                    out.append(
                        fused_experts_w8a8(
                            x_list[i], layer.w13_weight, layer.w2_weight,
                            layer.w13_weight_scale, layer.w2_weight_scale,
                            layer.w13_weight_offset, layer.w2_weight_offset,
                            topk_weights_list[i], topk_ids_list[i], row_idx))
                return torch.concat(out)
            return fused_experts_w8a8(x, layer.w13_weight, layer.w2_weight,
                                      layer.w13_weight_scale,
                                      layer.w2_weight_scale,
                                      layer.w13_weight_offset,
                                      layer.w2_weight_offset, topk_weights,
                                      topk_ids, row_idx)


def fused_experts_w8a8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_offset: torch.Tensor,
    w2_offset: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    row_idx: torch.Tensor,
):
    num_tokens, hidden_size = hidden_states.shape
    n_routed_experts = len(w1)
    sorted_tokens, expanded_src_to_dst_row, expanded_expert_idx = \
        torch_npu.npu_moe_init_routing(hidden_states, row_idx, topk_ids, num_tokens)
    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, n_routed_experts).to(torch.int64)
    act_dtype = hidden_states.dtype
    w1_scale = w1_scale.to(torch.bfloat16)
    w2_scale = w2_scale.to(torch.bfloat16)
    sorted_tokens, pertoken_scale = torch_npu.npu_dynamic_quant(sorted_tokens)
    gate_up_proj = \
        torch_npu.npu_grouped_matmul([sorted_tokens], [w1], scale=[w1_scale], per_token_scale=[pertoken_scale],
                                     bias=None, group_list=expert_tokens, split_item=3, output_dtype=act_dtype,
                                     group_type=0,
                                     group_list_type=0)[0]

    gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)
    gate_up_proj, pertoken_scale = torch_npu.npu_dynamic_quant(
        gate_up_proj)  # , smooth_scales=scale_2)

    out = torch_npu.npu_grouped_matmul([gate_up_proj], [w2],
                                       scale=[w2_scale],
                                       per_token_scale=[pertoken_scale],
                                       bias=None,
                                       group_list=expert_tokens,
                                       split_item=3,
                                       output_dtype=act_dtype,
                                       group_type=0,
                                       group_list_type=0)[0]
    out = out.float()
    return torch_npu.npu_moe_finalize_routing(out, None, None, None,
                                              topk_weights,
                                              expanded_src_to_dst_row,
                                              topk_ids).to(torch.bfloat16)


class AscendCompressedTensorsW4A8Int8MoEMethod(CompressedTensorsMoEMethod):
    LAST_SEQ_LEN = None
    BEST_EXPERT_TOKENS = None

    def __init__(
        self,
        quant_config: "AscendCompressedTensorsConfig"  # type: ignore # noqa E501
    ):

        STORAGE_BITS_NPU = 8
        WEIGHT_BITS = 4

        self.quant_config = quant_config
        self.weight_quant = self.quant_config.target_scheme_map["Linear"].get(
            "weights")
        self.input_quant = self.quant_config.target_scheme_map["Linear"].get(
            "input_activations")
        self.initialized = False
        self.warm_up = True
        self.n_total_experts = None
        self.pack_factor = STORAGE_BITS_NPU // WEIGHT_BITS
        self.group_size = self.weight_quant.group_size

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update({
            "is_transposed": False,
            "quant_method": self.weight_quant.strategy
        })
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition // self.pack_factor,
            hidden_size,
            dtype=torch.int8),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size // self.pack_factor,
            intermediate_size_per_partition,
            dtype=torch.int8),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.weight_quant.strategy == "group":
            w13_weight_int4_scale = torch.nn.Parameter(torch.ones(
                num_experts,
                hidden_size // self.group_size,
                2 * intermediate_size_per_partition,
                dtype=torch.int64),
                                                       requires_grad=False)

            w2_weight_int4_scale = torch.nn.Parameter(torch.ones(
                num_experts,
                intermediate_size_per_partition // self.group_size,
                hidden_size,
                dtype=torch.int64),
                                                      requires_grad=False)
        elif self.weight_quant.strategy == "channel":
            w13_weight_int4_scale = torch.nn.Parameter(torch.ones(
                num_experts,
                1,
                2 * intermediate_size_per_partition,
                dtype=torch.int64),
                                                       requires_grad=False)

            w2_weight_int4_scale = torch.nn.Parameter(torch.ones(
                num_experts, 1, hidden_size, dtype=torch.int64),
                                                      requires_grad=False)
        else:
            raise ValueError(
                f"Do not support quant strategy {self.weight_quant.strategy} !"
            )

        w13_weight_bias = torch.nn.Parameter(torch.ones(
            num_experts,
            2 * intermediate_size_per_partition,
            dtype=torch.float32),
                                             requires_grad=False)

        w2_weight_bias = torch.nn.Parameter(torch.ones(num_experts,
                                                       hidden_size,
                                                       dtype=torch.float32),
                                            requires_grad=False)
        # w13_weight_int4_offset = torch.nn.Parameter(torch.zeros(num_experts,
        #                                                         2 * intermediate_size_per_partition,
        #                                                         dtype=torch.float32
        #                                                         if params_dtype == torch.float16 else torch.bfloat16),
        #                                             requires_grad=False)

        layer.register_parameter("w13_weight_int4_scale",
                                 w13_weight_int4_scale)
        layer.register_parameter("w13_weight_bias", w13_weight_bias)
        # layer.register_parameter("w13_weight_int4_offset", w13_weight_int4_offset)
        set_weight_attrs(w13_weight_int4_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_bias, extra_weight_attrs)
        # set_weight_attrs(w13_weight_int4_offset, extra_weight_attrs)

        # w2_weight_int4_offset = torch.nn.Parameter(torch.zeros(num_experts,
        #                                                        hidden_size,
        #                                                        dtype=torch.float32
        #                                                        if params_dtype == torch.float16 else torch.bfloat16),
        #                                            requires_grad=False)

        layer.register_parameter("w2_weight_int4_scale", w2_weight_int4_scale)
        layer.register_parameter("w2_weight_bias", w2_weight_bias)
        # layer.register_parameter("w2_weight_int4_offset", w2_weight_int4_offset)
        set_weight_attrs(w2_weight_int4_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_bias, extra_weight_attrs)
        # set_weight_attrs(w2_weight_int4_offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight.transpose(
            1, 2).contiguous(),
                                              requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight.transpose(
            1, 2).contiguous(),
                                             requires_grad=False)

        if model_extra_config.operator_opt_config.gmm_nz:
            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight, 29)
            if "Ascend910B" not in torch.npu.get_device_name(0):
                layer.w2_weight.data = torch_npu.npu_format_cast(
                    layer.w2_weight, 29)

        layer.w13_weight.data = layer.w13_weight.data.view(
            torch.int32).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.view(
            torch.int32).contiguous()

        self.n_routed_experts = len(layer.w13_weight)

        experts_start_idx = get_ep_group(
        ).rank_in_group * self.n_routed_experts
        experts_end_idx = experts_start_idx + self.n_routed_experts
        self.expert_range = [experts_start_idx, experts_end_idx]
        self.n_total_experts = self.n_routed_experts * get_ep_group(
        ).world_size
        self.local_expert_indices_offset = (experts_start_idx)
        self.local_expert_indices = [
            self.local_expert_indices_offset + i
            for i in range(self.n_routed_experts)
        ]
        self.initialized = True

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              topk_weights: torch.Tensor,
              topk_ids: torch.Tensor,
              pertoken_scale: torch.Tensor,
              attn_metadata: AttentionMetadata,
              comm_group: Optional[GroupCoordinator] = None) -> torch.Tensor:

        # ENABLE_OMNI_PLANNER
        max_num_deployed_expert_per_rank = self.n_routed_experts
        # if model_extra_config.operator_opt_config.use_omni_planner:
        #     max_num_deployed_expert_per_rank = layer.planner.get_max_num_deployed_expert_per_rank()

        is_prefill = is_prefill = attn_metadata is None or attn_metadata.prefill is not None

        if model_extra_config.operator_opt_config.enable_moe_expert_parallel:
            is_prefill = attn_metadata is None or attn_metadata.prefill is not None
            if model_extra_config.operator_opt_config.prefill_moe_all_to_all or (
                    model_extra_config.operator_opt_config.
                    decode_moe_dispatch_combine and not is_prefill):
                if is_prefill:
                    out = moe_infer_fusion(layer,
                                           x,
                                           topk_ids,
                                           topk_weights,
                                           self.warm_up,
                                           is_prefill,
                                           comm_group=comm_group)
                else:
                    out = fused_experts_moe_dispatch_combine(
                        layer,
                        x,
                        topk_weights,
                        topk_ids,
                        max_num_deployed_expert=max_num_deployed_expert_per_rank
                        * get_ep_group().world_size,
                        is_prefill=is_prefill,
                        is_route_expert=True)
            else:
                out = fused_experts_w4a8_allgather_ep(
                    hidden_states=x,
                    pertoken_scale=pertoken_scale,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    w1_scale=layer.w13_weight_int4_scale,
                    w2_scale=layer.w2_weight_int4_scale,
                    w1_bias=layer.w13_weight_bias,
                    w2_bias=layer.w2_weight_bias,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    n_routed_experts=self.n_routed_experts,
                    attn_metadata=attn_metadata,
                    max_num_deployed_expert_per_rank=
                    max_num_deployed_expert_per_rank
                    # ENABLE_OMNI_PLANNER
                )
            if self.warm_up:
                self.warm_up = False
            return out
        else:
            raise NotImplementedError(
                "w4a8 does not support enable_moe_expert_parallel = False")
