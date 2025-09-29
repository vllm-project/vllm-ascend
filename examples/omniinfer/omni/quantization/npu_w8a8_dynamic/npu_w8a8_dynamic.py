#
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
#
# By using quantization case, this file is called before worker patch achieve,
from typing import Any, Callable, Dict, List, Optional
import os
import torch
import torch.distributed as dist
import torch_npu
import torchair
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped
)
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    ChannelQuantScaleParameter
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.distributed import get_tp_group, get_dp_group, get_ep_group

from omni.models.common.layers.fused_mlp import FusedMLP, FusedMLPMethodBase
from omni.models.common.layers.linear import (
    FlashCommLinearMethodBase,
    UnquantizedFlashCommLinearMethod,
    FlashCommLinearBase
)
from omni.models.qwen.fused_moe import FusedMoE, FusedMoEMethodBase
from omni.adaptors.vllm.utils import NPU_W8A8_DYNAMIC
from omni.adaptors.vllm.distributed.parallel_state import(
    GroupCoordinator,
    get_scale_parallel_group
)

from omni.models.common.config.model_config import model_extra_config

logger = init_logger(__name__)


@register_quantization_config(NPU_W8A8_DYNAMIC)
class NpuW8A8DynamicConfig(QuantizationConfig):
    """Config class for NPU W8A8 Dynamic."""

    def __init__(
        self,
        ignored_layers: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None
    ) -> None:
        self.ignored_layers = ignored_layers or []
        self.ignore = ignore

    @classmethod
    def get_name(cls) -> str:
        return NPU_W8A8_DYNAMIC

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        quant_method = hf_quant_cfg['quant_method']
        if torch.npu.is_available() and quant_method == 'npu_w8a8_dynamic':
            return NPU_W8A8_DYNAMIC
        return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NpuW8A8DynamicConfig":
        quant_method = cls.get_from_keys(config, ['quant_method'])
        ignored_layers = cls.get_from_keys_or(config, ['ignored_layers'], None)
        ignore = cls.get_from_keys_or(config, ['ignore'], [])
        return cls(ignored_layers=ignored_layers, ignore=ignore)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, FlashCommLinearBase) or isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedFlashCommLinearMethod()
            return NpuW8A8DynamicLinearMethod(self)
        elif isinstance(layer, FusedMLP):
            return NpuW8A8DynamicFusedMLPMethod(self)
        elif isinstance(layer, FusedMoE):
            return NpuW8A8DynamicFusedMoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []
    
    def get_cache_scale(self, name: str) -> Optional[str]:
        return None


class NpuW8A8DynamicLinearMethod(FlashCommLinearMethodBase):
    """Linear method for NPU W8A8 Dynamic.
    
    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: NpuW8A8DynamicConfig):
        self.quant_config = quant_config

    def create_weights(self,
                       layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: List[int],
                       input_size: int,
                       output_size: int,
                       params_dtype: torch.dtype,
                       **extra_weight_attrs):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get('weight_loader')

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight_dtype = torch.int8

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader
        )
        layer.register_parameter('weight', weight)

        logger.info('NpuW8A8LinearMethod params_dtype=%s', params_dtype)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(sum(output_partition_sizes), dtype=params_dtype),
            output_dim=0,
            weight_loader=weight_loader
        )
        layer.register_parameter('weight_scale', weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_data = torch_npu.npu_format_cast(layer.weight.data.t().contiguous(), 29)
        layer.weight = torch.nn.Parameter(weight_data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data.view(-1), requires_grad=False)

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            module_name: Optional[str] = "",
            x_transform: Optional[str] = None,
            is_prefill: Optional[bool] = True
    ) -> torch.Tensor:
        if isinstance(x, Dict):
            x, x_scale = x["x_int8"], x["pertoken_scale"]
        else:
            x, x_scale = torch_npu.npu_dynamic_quant(x, smooth_scales=None)

        scale_parallel = os.environ.get('SCALE_PARALLEL', '0') == '1'
        if x_transform == 'AG':
            if is_prefill or (not scale_parallel):
                x_scale = get_tp_group().all_gather(x_scale, dim=0)
            else:
                with torchair.scope.npu_stream_switch('sacle'):  # CANN包多流接口
                    x_scale = get_scale_parallel_group().all_gather(x_scale, dim=0)
            x = get_tp_group().all_gather(x, dim=0)
        elif x_transform == 'A2A':
            if is_prefill or (not scale_parallel):
                x_scale = get_tp_group().all_to_all(x_scale, transpose=False)
            else:
                with torchair.ops.NpuStreamSwitch('11'):  # CANN包多流接口
                    x_scale = get_tp_group().all_to_all(x_scale, transpose=False)
            x = get_tp_group().all_to_all(x)
        y = torch_npu.npu_quant_matmul(
            x1=x,
            x2=layer.weight,
            scale=layer.weight_scale,
            pertoken_scale=x_scale,
            bias=bias,
            output_dtype=layer.orig_dtype
        )
        return y


class NpuW8A8DynamicFusedMLPMethod(FusedMLPMethodBase):
    """Apply dequant_swiglu_quant fused kernel.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: NpuW8A8DynamicConfig):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.gate_up_proj.weight_scale = torch.nn.Parameter(
            layer.gate_up_proj.weight_scale.data.float(), requires_grad=False)

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            x_transform: str = None,
            is_prefill: bool = True,
    ) -> torch.Tensor:
        bias = layer.gate_up_proj.bias if not layer.gate_up_proj.skip_bias_add else None
        x, x_scale = torch_npu.npu_dynamic_quant(x, smooth_scales=None)
        scale_parallel = os.environ.get('SCALE_PARALLEL', '0') == '1'
        if x_transform is not None:
            if x_transform == 'AG':
                if is_prefill or (not scale_parallel):
                    x_scale = get_tp_group().all_gather(x_scale, dim=0)
                else:
                    with torchair.ops.NpuStreamSwitch('scale'):  # CANN包多流接口
                        x_scale = get_tp_group().all_gather(x_scale, dim=0)
                x = get_tp_group().all_gather(x, dim=0)
            elif x_transform == 'A2A':
                if is_prefill or (not scale_parallel):
                    x_scale = get_tp_group().all_to_all(x_scale, transpose=False)
                else:
                    with torchair.ops.NpuStreamSwitch('scale'):  # CANN包多流接口
                        x_scale = get_tp_group().all_to_all(x_scale, transpose=False)
                x = get_tp_group().all_to_all(x)
        y_int32 = torch_npu.npu_quant_matmul(
            x1=x,
            x2=layer.gate_up_proj.weight,
            scale=layer.gate_up_proj.weight_scale,
            pertoken_scale=None,
            bias=None,
            output_dtype=torch.int32
        )
        int_int32, int_scale = torch_npu.npu_dequant_swiglu_quant(
            y_int32,
            weight_scale=layer.gate_up_proj.weight_scale,
            activation_scale=x_scale,
            bias=bias,
            activate_left=True,
            quant_mode=1,
        )

        bias = None if (layer.down_proj.tp_rank > 0 or layer.down_proj.skip_bias_add) else layer.down_proj.bias
        output = torch_npu.npu_quant_matmul(
            x1=int_int32,
            x2=layer.down_proj.weight,
            scale=layer.down_proj.weight_scale,
            pertoken_scale=int_scale,
            bias=bias,
            output_dtype=layer.down_proj.orig_dtype
        )

        return output


class NpuW8A8DynamicFusedMoEMethod(FusedMoEMethodBase):
    """Fused MoE method for NPU W8A8 Dynamic.

    Args:
        quant_config: The quantization config.
    """

    ONES_SCALE = None

    def __init__(self, quant_config: NpuW8A8DynamicConfig):
        self.quant_config = quant_config
        if model_extra_config.operator_opt_config.use_prefetch:
            self.w13_prefetch_size = 2 * 12 * 1024 * 1024
            self.w2_prefetch_size = 2 * 6 * 1024 * 1024

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                    2 * intermediate_size_per_partition,
                                                    hidden_size,
                                                    dtype=torch.int8),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                  2 * intermediate_size_per_partition,
                                                  dtype=torch.float32
                                                  if params_dtype == torch.float16 else torch.bfloat16),
                                        requires_grad=False)
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(num_experts,
                                                   hidden_size,
                                                   intermediate_size_per_partition,
                                                   dtype=torch.int8),
                                        requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(torch.ones(num_experts,
                                                 hidden_size,
                                                 dtype=torch.float32
                                                 if params_dtype == torch.float16 else torch.bfloat16),
                                        requires_grad=False)
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)

        if NpuW8A8DynamicFusedMoEMethod.ONES_SCALE is None:
            NpuW8A8DynamicFusedMoEMethod.ONES_SCALE = torch.ones(
                (layer.local_num_experts, layer.intermediate_size_per_partition), dtype=torch.float32, device='npu'
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        w13 = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        w2 = layer.w2_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.data.float(), requires_grad=False
        )
        if model_extra_config.operator_opt_config.gmm_nz:
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29).contiguous()
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29).contiguous()

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
        moe_dispatch_combine = os.environ.get('MOE_DISPATCH_COMBINE', '0') == '1'
        if moe_dispatch_combine:
            return self.apply_all2all(
                layer,
                x,
                router_logits,
                top_k,
                renormalize,
                use_grouped_topk,
                topk_group,
                num_expert_group,
                global_num_experts,
                expert_range,
                custom_routing_function,
                scoring_func,
                e_score_correction_bias,
                apply_router_weight_on_input,
                activation,
                is_prefill
            )
        else:
            return self.apply_ag_rs(
                layer,
                x,
                router_logits,
                top_k,
                renormalize,
                use_grouped_topk,
                topk_group,
                num_expert_group,
                global_num_experts,
                expert_range,
                custom_routing_function,
                scoring_func,
                e_score_correction_bias,
                apply_router_weight_on_input,
                activation,
                is_prefill
            )

    def apply_all2all(
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
                indices_type=None
            )
            topk_weights = topk_weights.to(x.dtype)
            topk_ids = topk_ids.int()

            expanded_x, expanded_row_idx, tokens_per_expert, expanded_scale = torch_npu.npu_moe_init_routing_v2(
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
                quant_mode=1,
                row_idx_type=0
            )

            tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)

            combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
            ep_size = get_ep_group().world_size

            combine_tokens = combine_tokens.view(2, ep_size, -1).sum(2)
            all_tokens = combine_tokens[0].sum()
            combine_tokens_cpu = combine_tokens.cpu().tolist()
            # all2all input splits, 大小为当前rank路由到其它rank的token数总和
            input_splits = combine_tokens_cpu[1]
            # all2all output splits, 每个rank拿到的其它卡的token数
            output_splits = combine_tokens_cpu[0]
            # all2all output, 展开成一维，大小为其它卡路由到当前rank的token数总和
            gathered_tokens = expanded_x.new_empty(
                all_tokens.item(), expanded_x.shape[1]
            )
            dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits)
            gathered_pertoken_scale = expanded_scale.new_empty(gathered_tokens.shape[0])
            dist.all_to_all_single(gathered_pertoken_scale, expanded_scale, output_splits, input_splits)
            # reroute
            # 按专家归并后的tokens，按专家归并后的scales，给FinalizeRouting用的索引，每个专家处理的token数
            hidden_states_sorted_by_experts, gathered_pertoken_scale, gathered_idxs_unsort, tokens_per_local_expert = torch_npu.npu_moe_re_routing(
                gathered_tokens,
                tokens_per_expert_group.view(ep_size, -1),
                per_token_scales=gathered_pertoken_scale
            )
            group_list = tokens_per_local_expert.to(torch.int64)

            gate_up_proj = torch_npu.npu_grouped_matmul(
                [hidden_states_sorted_by_experts],
                [layer.w13_weight],
                bias=None,
                group_list=group_list,
                split_item=3,
                output_dtype=torch.int32,
                group_type=0,
                group_list_type=1
            )[0]
            inter_states, inter_states_scale = torch_npu.npu_dequant_swiglu_quant(
                gate_up_proj,
                weight_scale=layer.w13_weight_scale,
                activation_scale=gathered_pertoken_scale,
                bias=None,
                quant_scale=NpuW8A8DynamicFusedMoEMethod.ONES_SCALE,
                quant_offset=None,
                group_index=group_list,
                activate_left=True,
                quant_mode=1
            )
            hidden_states_ordered_by_experts = torch_npu.npu_grouped_matmul(
                [inter_states],
                [layer.w2_weight],
                scale=[layer.w2_weight_scale],
                per_token_scale=[inter_states_scale],
                bias=None,
                group_list=group_list,
                split_item=3,
                output_dtype=x.dtype,
                group_type=0,
                group_list_type=1
            )[0]

            new_x = torch_npu.npu_moe_finalize_routing(
                hidden_states_ordered_by_experts.float(),
                skip1=None,
                skip2=None,
                bias=None,
                scales=None,
                expanded_src_to_dst_row=gathered_idxs_unsort.to(torch.float32).argsort().to(torch.int32),
                export_for_source_row=None,
                drop_pad_mode=2
            )
            new_x = new_x.to(torch.bfloat16)
            gathered_tokens = new_x.new_empty(*expanded_x.shape)

            dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits)

            y = torch_npu.npu_moe_finalize_routing(
                gathered_tokens,
                None,
                None,
                None,
                topk_weights,  # 数据类型要求与y一致
                expanded_row_idx,
                topk_ids,
                drop_pad_mode=2
            )

            return y
        else:
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
                indices_type=None
            )
            topk_ids = topk_ids.int()

            if model_extra_config.operator_opt_config.use_prefetch:
                flag_expert_prefetch = topk_ids
                if self.w13_prefetch_size > 0:
                    torch_npu.npu_prefetch(layer.w13_weight, flag_expert_prefetch, self.w13_prefetch_size)
                if self.w2_prefetch_size > 0:
                    torch_npu.npu_prefetch(layer.w2_weight, flag_expert_prefetch, self.w2_prefetch_size)

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
                quant_mode=2,  # 0: 非量化; 1: 静态量化; 2: 动态量化
                global_bs=0
            )
            group_list = expert_token_nums.to(torch.int64)

            gate_up_proj = torch_npu.npu_grouped_matmul(
                [expand_x],
                [layer.w13_weight],
                bias=None,
                group_list=group_list,
                split_item=3,
                output_dtype=torch.int32,
                group_type=0,
                group_list_type=1
            )[0]
            # ONES_SCALE为该类内部变量，故在此处mark_static
            torch._dynamo.mark_static(NpuW8A8DynamicFusedMoEMethod.ONES_SCALE)
            inter_states, inter_states_scale = torch_npu.npu_dequant_swiglu_quant(
                gate_up_proj,
                weight_scale=layer.w13_weight_scale,
                activation_scale=dynamic_scales,
                bias=None,
                quant_scale=NpuW8A8DynamicFusedMoEMethod.ONES_SCALE,
                quant_offset=None,
                group_index=group_list,
                activate_left=True,
                quant_mode=1
            )
            hidden_states_ordered_by_experts = torch_npu.npu_grouped_matmul(
                [inter_states],
                [layer.w2_weight],
                scale=[layer.w2_weight_scale],
                per_token_scale=[inter_states_scale],
                bias=None,
                group_list=group_list,
                split_item=3,
                output_dtype=x.dtype,
                group_type=0,
                group_list_type=1
            )[0]

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

    def apply_ag_rs(
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
            indices_type=None
        )

        x_int8, x_scale = torch_npu.npu_dynamic_quant(x)

        if is_prefill:
            assert len(x_int8.shape) == 2
            n_tokens = x_int8.shape[0]
            n_tokens_tensor = torch.Tensor([n_tokens]).int().npu()
            n_tokens_list = get_ep_group().all_gather(n_tokens_tensor, dim=0).tolist()

            x_int8_output_list = [torch.empty((n, x_int8.shape[1]), dtype=x_int8.dtype, device=x_int8.device) for n in
                                  n_tokens_list]
            get_ep_group().all_gather_v(x_int8_output_list, x_int8)
            x_int8 = torch.cat(x_int8_output_list)

            x_scale_output_list = [torch.empty((n,), dtype=x_scale.dtype, device=x_scale.device) for n in n_tokens_list]
            get_ep_group().all_gather_v(x_scale_output_list, x_scale)
            x_scale = torch.cat(x_scale_output_list)

            topk_weights_output_list = [
                torch.empty((n, topk_weights.shape[1]), dtype=topk_weights.dtype, device=topk_weights.device) for n in
                n_tokens_list
            ]
            get_ep_group().all_gather_v(topk_weights_output_list, topk_weights)
            topk_weights = torch.cat(topk_weights_output_list)

            topk_ids_output_list = [torch.empty((n, topk_ids.shape[1]), dtype=topk_ids.dtype, device=topk_ids.device)
                                    for n in n_tokens_list]
            get_ep_group().all_gather_v(topk_ids_output_list, topk_ids)
            topk_ids = torch.cat(topk_ids_output_list)
        else:
            x_int8 = get_ep_group().all_gather(x_int8, dim=0)
            x_scale = get_ep_group().all_gather(x_scale, dim=0)
            topk_weights = get_ep_group().all_gather(topk_weights, dim=0)
            topk_ids = get_ep_group().all_gather(topk_ids, dim=0)

        sorted_tokens, expanded_x_idx, expert_tokens, expanded_scale = torch_npu.npu_moe_init_routing_v2(
                x_int8,
                topk_ids,
                scale=x_scale,
                offset=None,
                active_num=topk_ids.numel(),
                expert_num=global_num_experts,
                expert_capacity=-1,
                drop_pad_mode=0,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=expert_range,
                quant_mode=-1,
                row_idx_type=0
        )
        expanded_x_idx = torch.clamp(expanded_x_idx, min=0, max=expanded_x_idx.shape[0] - 1)
        gate_up_proj = torch_npu.npu_grouped_matmul(
            [sorted_tokens],
            [layer.w13_weight],
            bias=None,
            group_list=expert_tokens,
            split_item=3,
            output_dtype=torch.int32,
            group_type=0,
            group_list_type=1
        )[0]
        inter_states, inter_states_scale = torch_npu.npu_dequant_swiglu_quant(
            gate_up_proj,
            weight_scale=layer.w13_weight_scale,
            activation_scale=expanded_scale,
            bias=None,
            quant_scale=NpuW8A8DynamicFusedMoEMethod.ONES_SCALE,
            quant_offset=None,
            group_index=expert_tokens,
            activate_left=True,
            quant_mode=1
        )
        y = torch_npu.npu_grouped_matmul(
            [inter_states],
            [layer.w2_weight],
            scale=[layer.w2_weight_scale],
            per_token_scale=[inter_states_scale],
            bias=None,
            group_list=expert_tokens,
            split_item=3,
            output_dtype=x.dtype,
            group_type=0,
            group_list_type=1
        )[0]

        # 将不在本rank的专家的topk_weights置为0
        valid_mask = (topk_ids >= expert_range[0]) & (topk_ids < expert_range[1])
        topk_weights = topk_weights * valid_mask.to(topk_weights.dtype)

        y = torch_npu.npu_moe_finalize_routing(
            y.float(),
            None,
            None,
            None,
            topk_weights.float(),  # 数据类型要求与y一致
            expanded_x_idx,
            topk_ids,
            drop_pad_mode=2
        ).to(x.dtype)

        if is_prefill:
            assert len(y.shape) == 2
            y_list = list(torch.split(y, n_tokens_list))
            y_output = torch.empty((n_tokens, y.shape[1]), dtype=y.dtype, device=y.device)
            get_ep_group().reduce_scatter_v(y_output, y_list)
            y = y_output
        else:
            y = get_ep_group().reduce_scatter(y)

        return y
