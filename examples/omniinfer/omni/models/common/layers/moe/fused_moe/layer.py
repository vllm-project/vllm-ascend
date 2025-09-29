# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Optional, Callable
import os
import torch, torch_npu
import torchair as tng
import torch.distributed as dist
from vllm.distributed import get_world_group, get_pp_group, get_ep_group, get_tp_group
from vllm.attention import AttentionMetadata
from vllm.platforms import current_platform
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE as GPUFusedMoE
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod as GPUUnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.layer import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from omni.adaptors.vllm.distributed.parallel_state import GroupCoordinator
from omni.models.common.config.model_config import model_extra_config
from omni.models.common.layers.moe.fused_moe.fused_moe import (
    fused_topk,
    grouped_topk
)

UNQUANT_MODE = 0
STATIC_QUANT_MODE = 1
DYNAMIC_QUANT_MODE = 2


class UnquantizedFusedMoEMethod(GPUUnquantizedFusedMoEMethod):
    LAST_SEQ_LEN = None
    BEST_EXPERT_TOKENS = None

    def __init__(self):
        super().__init__(None)
        self.warm_up = True

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            pertoken_scale: torch.Tensor,
            attn_metadata: AttentionMetadata,
            comm_group: Optional[GroupCoordinator]
    ) -> torch.Tensor:
        is_prefill = attn_metadata is None or attn_metadata.prefill is not None
        out = self.moe_infer_fusion(layer,
                                    x,
                                    topk_ids,
                                    topk_weights,
                                    layer.w13_weight,
                                    layer.w2_weight,
                                    is_prefill)
        if self.warm_up:
            self.warm_up = False
        return out

    def moe_infer_fusion(self, layer, x, topk_ids, topk_weight, w1, w2, is_prefill=True):
        _, h = x.shape
        hidden_states = x.view(-1, h)
        topk_weight = topk_weight.to(x.dtype)
        if self.warm_up:
            # This is forced balancing, the goal is to reduce peak memory
            global_rank = get_world_group().rank_in_group
            step = hidden_states.shape[0] * 8  # topk 8 expert
            cur_topk_list = [
                (i + global_rank // 1) % 256 for i in range(
                    global_rank // 1 * step, (global_rank // 1 + 1) * step)]
            topk_ids = torch.Tensor(cur_topk_list).int().view(hidden_states.shape[0], -1).npu()
        else:
            topk_ids = topk_ids.int()
        max_num_deployed_expert = 256
        if model_extra_config.operator_opt_config.use_omni_placement and layer.moe_layer_idx < 58:
            max_num_deployed_expert = layer.planner.get_max_num_deployed_expert_per_rank() * get_world_group().world_size
        expert_range = [0, max_num_deployed_expert]
        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=topk_ids,
            scale=None,
            expert_num=max_num_deployed_expert,
            active_expert_range=expert_range,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_num=topk_ids.numel(),
            drop_pad_mode=0,
            row_idx_type=0,
            quant_mode=-1)
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group,
                               tokens_per_expert)  # (total_experts,) --> (total_ranks * n_routed_experts_per_rank)

        # combine tensors, do reduceSum and D2H toghter
        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        # view: EP, E//EP
        # sum: EP, the number of tokens each rank receives from other cards
        ep_size = get_ep_group().world_size
        combine_tokens = combine_tokens.view(2, ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        # alltoall input splits, the total number of tokens routed from the current rank to other ranks
        input_splits = combine_tokens_cpu[1]
        # alltoall output splits, the number of tokens each rank receives from other cards
        output_splits = combine_tokens_cpu[0]
        # alltoall output, unfolded into one dimension, the size is the sum of the number of tokens routed from other cards to the current rank.
        gathered_tokens = expanded_x.new_empty(
            all_tokens.item(), expanded_x.shape[1]
        )
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits)
        # reroute
        # Tokens merged by experts, scales merged by experts, indices for FinalizeRouting, number of tokens processed by each expert
        hidden_states_sorted_by_experts, _, gathered_idxs_unsort, tokens_per_local_expert = torch_npu.npu_moe_re_routing(
            gathered_tokens,
            tokens_per_expert_group.view(ep_size, -1),
            per_token_scales=None
        )
        group_list = tokens_per_local_expert.to(torch.int64)
        if model_extra_config.operator_opt_config.use_omni_placement:
            layer.planner.record_activation(layer.moe_layer_idx, group_list,
                                            support_multi_stream=model_extra_config.operator_opt_config.moe_multi_stream_tune and (
                                                not is_prefill))
        mm1_mm3 = torch_npu.npu_grouped_matmul([hidden_states_sorted_by_experts], [w1],
                                               group_list=group_list, split_item=3, group_type=0,
                                               group_list_type=1)[0]
        intermediate_h = torch_npu.npu_swiglu(mm1_mm3)
        # gmm2: down
        hidden_states_ordered_by_experts = torch_npu.npu_grouped_matmul([intermediate_h], [w2], bias=None,
                                                                        group_list=group_list, split_item=3,
                                                                        group_type=0,
                                                                        group_list_type=1)[0]
        new_x = torch.index_select(hidden_states_ordered_by_experts, 0,
                                   gathered_idxs_unsort.to(torch.float32).argsort().to(torch.int32))
        gathered_tokens = new_x.new_empty(*expanded_x.shape)

        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits)

        return hidden_states, gathered_tokens, topk_weight, expanded_row_idx

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight.transpose(1, 2).contiguous(), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight.transpose(1, 2).contiguous(), requires_grad=False)
        if model_extra_config.operator_opt_config.gmm_nz:
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 29)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 29)
        self.n_routed_experts = len(layer.w13_weight)
        self.local_expert_indices_offset = (
                get_ep_group().rank_in_group * self.n_routed_experts
        )
        self.local_expert_indices = [
            self.local_expert_indices_offset + i for i in range(self.n_routed_experts)
        ]
        self.initialized = True


class FusedMoE(torch.nn.Module):
    _load_w13 = GPUFusedMoE._load_w13
    _load_w2 = GPUFusedMoE._load_w2
    _load_single_value = GPUFusedMoE._load_single_value
    _load_g_idx = GPUFusedMoE._load_g_idx
    make_expert_params_mapping = GPUFusedMoE.make_expert_params_mapping
    _load_per_tensor_weight_scale = GPUFusedMoE._load_per_tensor_weight_scale
    _load_model_weight_or_group_weight_scale = GPUFusedMoE._load_model_weight_or_group_weight_scale

    # _load_fp8_scale = GPUFusedMoE._load_fp8_scale

    def __init__(
            self,
            num_experts: int,
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
            prefix: str = "",
            custom_routing_function: Optional[Callable] = None,
            scoring_func: str = "softmax",
            e_score_correction_bias: Optional[torch.Tensor] = None,
            first_k_dense_replace: int = 3,
            **kwargs
    ):
        super().__init__()
        # OMNI_PLANNER: import omni planner instance, all layers share the same instance(singleton instance)
        self.planner = kwargs.get("planner", None)
        self.moe_layer_idx = kwargs.get("moe_layer_idx", None)
        self.expert_mapping = kwargs.get("expert_mapping", None)
        ep_size = get_ep_group().world_size
        if ep_size > 1:
            ep_size = ep_size - model_extra_config.parall_config.redundancy_shared_expert_num
            num_experts = int(num_experts / ep_size)
            tp_size = 1

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (tp_size if tp_size is not None else
                        get_ep_group().world_size)
        self.top_k = top_k
        self.num_experts = num_experts
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            if num_expert_group is None or topk_group is None:
                raise RuntimeError("num_expert_group and topk_group must not be None")
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.is_prefill_instance = os.environ.get("ROLE", "") == "prefill"
        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                UnquantizedFusedMoEMethod())
            self.quant_mode = UNQUANT_MODE
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
            self.quant_mode = DYNAMIC_QUANT_MODE  # static_quant_mode is not supported now
        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")

        # ENABLE_OMNI_PLANNER
        num_of_redundant_experts = 0
        if model_extra_config.operator_opt_config.use_omni_placement:
            num_of_redundant_experts = self.planner.get_num_of_redundant_experts(moe_layer_idx=self.moe_layer_idx,
                                                                                 num_expert_per_device_origin=num_experts,
                                                                                 rank_device=get_ep_group().rank_in_group - model_extra_config.parall_config.redundancy_shared_expert_num)
        self.quant_method.create_weights(
            layer=self,
            num_experts=num_experts + num_of_redundant_experts,  # ENABLE_OMNI_PLANNER
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader)
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")

        if model_extra_config.operator_opt_config.decode_moe_dispatch_combine:
            # Adapt the dispatch combine operator
            self.ep_size = get_ep_group().world_size
            self.global_rank = get_world_group().rank_in_group
            self.world_size = get_world_group().world_size
            # self.n_shared_experts = n_shared_experts

            self.moe_all_to_all_group = get_world_group().device_group
            self.moe_all_to_all_group_name = self.moe_all_to_all_group._get_backend(
                torch.device(current_platform.device_type)).get_hccl_comm_name(
                self.global_rank)
            self.moe_rs_group = get_pp_group().device_group
            self.moe_rs_group_rank = get_pp_group().rank_in_group
            self.moe_rs_group_name = self.moe_rs_group._get_backend(
                torch.device(current_platform.device_type)).get_hccl_comm_name(
                self.moe_rs_group_rank)

    def cal_best_topk(self, tokens):
        world_size = get_world_group().world_size
        batch_size = tokens * world_size
        step = batch_size // world_size * self.top_k
        global_rank = get_world_group().rank_in_group
        cur_topk_list = [i % 256 for i in range(global_rank * step, (global_rank + 1) * step)]
        return torch.Tensor(cur_topk_list).to(dtype=torch.int32, device=current_platform.device_type, non_blocking=True).view(batch_size // world_size, -1)

    def apply_expert_load_balance(
            self,
            topk_ids: torch.Tensor,
            best_topk_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # omni placement
        if self.planner is not None:
            _, topk_ids, _ = self.planner.plan(
                layer_idx_moe=self.moe_layer_idx,
                tokens=None,
                token_expert_ids=topk_ids,
                token_expert_scores=None,
                expert_mapping=self.expert_mapping
            )

        # Forced load balance
        if model_extra_config.operator_opt_config.best_ep:
            if self.is_prefill_instance:
                topk_ids = self.cal_best_topk(topk_ids.shape[0])
            elif best_topk_ids is not None:
                if model_extra_config.operator_opt_config.moe_multi_stream_tune:
                    topk_ids = tng.scope.npu_wait_tensor(best_topk_ids, topk_ids)
                else:
                    topk_ids = best_topk_ids

        return topk_ids

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
                       layer: torch.nn.Module = None
                       ):
        # For vllm v1, metadata is a dict {layer_name: metadata}
        attn_metadata = get_forward_context().attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[next(iter(attn_metadata))]
        is_prefill = attn_metadata is None or attn_metadata.prefill is not None
        # DeekSeekv2 uses grouped_top_k
        # adapt: When num_expert_group=1, it degenerates to fused_topk.
        if use_grouped_topk:  # and num_expert_group != 1:
            # adapt end.
            if topk_group is None:
                raise ValueError(f"Unsupported topk_group is None")
            if num_expert_group is None:
                raise ValueError(f"Unsupported num_expert_group is None")

            if e_score_correction_bias is None and attn_metadata is None:
                topk_weights, topk_ids, row_idx = grouped_topk(
                    hidden_states=hidden_states,
                    gating_output=router_logits,
                    topk=top_k,
                    renormalize=renormalize,
                    num_expert_group=num_expert_group,
                    topk_group=topk_group,
                    scoring_func=scoring_func,
                    e_score_correction_bias=e_score_correction_bias)
                topk_weights = topk_weights * routed_scaling_factor
            else:
                topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                    router_logits.float(),
                    k=top_k,  # topk is currently 8
                    bias=e_score_correction_bias,  # float32
                    k_group=topk_group,  # fix: 4
                    group_count=num_expert_group,  # fix 8
                    group_select_mode=1,  # 0: maximum in group; 1: topk2.sum(fix)
                    renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
                    norm_type=1,  # 0: softmax; 1: sigmoid(fix)
                    routed_scaling_factor=routed_scaling_factor,
                    eps=float(1e-20))
            row_idx = torch.arange(topk_ids.numel(), device=current_platform.device_type, dtype=torch.int32).view(
                -1, router_logits.shape[
                    0]).transpose(0, 1)
        elif custom_routing_function is None:
            topk_weights, topk_ids, row_idx = fused_topk(gating_output=router_logits,
                                                         topk=top_k,
                                                         renormalize=renormalize)
        else:
            topk_weights, topk_ids, row_idx = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)

        return topk_weights, topk_ids, row_idx

    def _load_per_channel_weight_scale(self, expert_data: torch.Tensor,
                                       shard_dim: int, shard_id: str,
                                       loaded_weight: torch.tensor,
                                       tp_rank: int):
        # adapt loaded_weight shape
        loaded_weight = loaded_weight.squeeze(-1)
        # adapt end
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def forward(self, hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                pertoken_scale: torch.Tensor,
                attn_metadata: AttentionMetadata,
                comm_group: Optional[GroupCoordinator] = None
                ):
        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=pertoken_scale,
            attn_metadata=attn_metadata,
            comm_group=comm_group
        )

        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = get_ep_group().all_reduce(final_hidden_states)

        return final_hidden_states

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int) -> None:

        if get_ep_group().world_size > 1:
            ep_rank = get_ep_group().rank_in_group - model_extra_config.parall_config.redundancy_shared_expert_num
            # ENABLE_OMNI_PLANNER
            if model_extra_config.operator_opt_config.use_omni_placement:
                # OMNI_PLANNER: determine the expert deployment based on the pattern
                exists_locally, local_pos = self.planner.is_expert_on_current_rank(self.moe_layer_idx, expert_id,
                                                                                   ep_rank, self.num_experts)
                # if the re-deployed expert is not on the current rank, then skip the weight_loader
                if not exists_locally:
                    return
                # if the re-deployed expert is on the current rank, then update the id of the expert
                else:
                    expert_id = ep_rank * self.num_experts + local_pos
            else:
                if expert_id < ep_rank * self.num_experts or expert_id >= (ep_rank + 1) * self.num_experts:
                    return
            tp_rank = 0
            expert_id -= ep_rank * self.num_experts
        else:
            tp_rank = get_tp_group().rank_in_group
        # compressed-tensors checkpoints with packed weights are stored flipped
        loaded_weight = loaded_weight.t().contiguous() if (
                self.quant_method.__class__.__name__
                == "CompressedTensorsWNA16MoEMethod") else loaded_weight

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but "
                             f"got {shard_id}.")

        WEIGHT_SCALE_SUPPORTED = [
            e.value for e in FusedMoeWeightScaleSupported
        ]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        expert_data = param.data[expert_id]

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = ~shard_dim

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if param.data[expert_id] != 1 and (param.data[expert_id] -
                                               loaded_weight).abs() > 1e-5:
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}")

            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(shard_dim=0,
                             shard_id=shard_id,
                             loaded_weight=loaded_weight,
                             expert_data=expert_data,
                             tp_rank=tp_rank)
            return

        # Case weight scales and zero_points
        if ("scale" in weight_name or "zero" in weight_name or "offset" in weight_name or "bias" in weight_name):
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                if "int4_scale" in weight_name:
                    shard_dim = 1
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank)
            elif quant_method == FusedMoeWeightScaleSupported.GROUP.value:
                shard_dim = 1
                if "bias" in weight_name:
                    shard_dim = 0
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank)
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(shard_id=shard_id,
                                                   param=param,
                                                   loaded_weight=loaded_weight,
                                                   expert_id=expert_id)
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}")
            return
        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank)
            return
