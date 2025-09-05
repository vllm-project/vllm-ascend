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

from typing import Any, Callable, Optional

import torch
import torch_npu
from vllm.config import CompilationLevel, get_current_vllm_config
from vllm.distributed import (get_dp_group, get_ep_group, get_tp_group,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import \
    FusedMoEParallelConfig  # isort: skip
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod)
from vllm_ascend.ops.moe.experts_selector import select_experts
from vllm_ascend.ops.moe.moe_comm_method import (AllGatherCommImpl,
                                                 AlltoAllCommImpl, MC2CommImpl,
                                                 NaiveMulticastCommImpl)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, is_310p

original_unquantized_fused_moe_init_func = UnquantizedFusedMoEMethod.__init__



class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
                                 == CompilationLevel.PIECEWISE and
                                 not vllm_config.model_config.enforce_eager)

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(
            1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(
            1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

        if not is_310p():
            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)

    def fused_experts(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_int8_w8a8: bool = False,
        use_int4_w4a8: bool = False,
        global_num_experts: Optional[int] = None,
        expert_map: Optional[torch.Tensor] = None,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        w1_scale_bias: torch.Tensor = None,
        w2_scale_bias: torch.Tensor = None,
        # For TorchAir graph
        is_torchair: bool = False,
        # For Cube/Vector parallel
        shared_experts: Optional[Any] = None,
        quantized_x_for_share: Optional[Any] = None,
        dynamic_scale_for_share: Optional[Any] = None,
        # For load balance
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
    ) -> torch.Tensor:
        # Check constraints
        assert hidden_states.shape[1] == w1.shape[1], (
            f"Hidden size mismatch {hidden_states.shape[1]} != {w1.shape[1]}")
        assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
        assert hidden_states.is_contiguous(
        ), "Hidden_states must be contiguous"
        assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
        assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
        assert hidden_states.dtype in [
            torch.float32, torch.float16, torch.bfloat16
        ]
        if (use_int8_w8a8 or use_int4_w4a8):
            assert w1_scale is not None and w2_scale is not None, \
                "INT8 quantization requires weight scales."

        w1_scale = w1_scale.to(torch.float32)
        down_scale = [w2_scale]
        down_output_dtype = w2_scale.dtype
    else:
        down_scale = None
        down_output_dtype = None

    moe_comm_method = get_forward_context().moe_comm_method
    assert moe_comm_method is not None, "Missing communication context"

    num_experts = w1.shape[0]

    permuted_hidden_states, expert_tokens, dynamic_scale, group_list_type = moe_comm_method.permute(
        hidden_states, topk_ids, topk_weights, expert_map, num_experts,
        use_int8_w8a8 or use_int4_w4a8)

    gate_up_output = torch_npu.npu_grouped_matmul(
        x=[permuted_hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=torch.int32 if use_int8_w8a8 else None,
    )[0]

    if (use_int8_w8a8 or use_int4_w4a8):
        activated_output, activated_output_scale = torch_npu.npu_dequant_swiglu_quant(
            x=gate_up_output,
            weight_scale=w1_scale,
            activation_scale=dynamic_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=expert_tokens,
            activate_left=True,
            quant_mode=1,
        )
        activated_output_scale = [activated_output_scale]
    else:
        activated_output = torch_npu.npu_swiglu(gate_up_output)
        activated_output_scale = None

    down_output = torch_npu.npu_grouped_matmul(
        x=[activated_output],
        weight=[w2],
        scale=down_scale,
        per_token_scale=activated_output_scale,
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=down_output_dtype,
    )[0]

    moe_comm_method.unpermute(down_output, hidden_states)

    return hidden_states


def fused_experts_moge(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    moe_parallel_config: FusedMoEParallelConfig,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    expert_map: torch.Tensor = None,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """

    Args:
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        w1: Expert weights1 of shape (num_experts, intermediate_size * 2, hidden_size).
        w2: Expert weights2 of shape (num_experts, hidden_size, intermediate_size).
        topk_weights: Routing weights of shape (num_tokens, top_k).
        topk_ids: Selected expert IDs of shape (num_tokens, top_k).
        top_k: Number of experts to select.
        expert_map: Expert mapping of shape (num_experts,).
    Args:
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        w1: Expert weights1 of shape (num_experts, intermediate_size * 2, hidden_size).
        w2: Expert weights2 of shape (num_experts, hidden_size, intermediate_size).
        topk_weights: Routing weights of shape (num_tokens, top_k).
        topk_ids: Selected expert IDs of shape (num_tokens, top_k).
        top_k: Number of experts to select.
        expert_map: Expert mapping of shape (num_experts,).

    Returns:
        hidden_states: Hidden states after routing.
    """
    ep_size = moe_parallel_config.ep_size
    local_num_experts = global_num_experts // ep_size
    local_num_group = top_k // ep_size
    Returns:
        hidden_states: Hidden states after routing.
    """
    ep_size = moe_parallel_config.ep_size
    local_num_experts = global_num_experts // ep_size
    local_num_group = top_k // ep_size

    bsz, _ = hidden_states.shape
    flatten_topk_ids = topk_ids.view(-1)
    sorted_topk_ids = torch.argsort(flatten_topk_ids.float())
    sorted_topk_ids = sorted_topk_ids.to(torch.int32)
    sorted_hidden_states = hidden_states.index_select(
        0, sorted_topk_ids // local_num_group)

    experts_id = torch.arange(0,
                              local_num_experts,
                              dtype=topk_ids.dtype,
                              device=topk_ids.device)
    num_tokens_per_expert = (flatten_topk_ids.unsqueeze(-1) == experts_id).to(
        torch.float32).sum(0)
    topk_scales = topk_weights.view(-1).index_select(
        0, sorted_topk_ids).unsqueeze(-1)
    group_list = num_tokens_per_expert.cumsum(dim=0).to(torch.int64)
    experts_id = torch.arange(0,
                              local_num_experts,
                              dtype=topk_ids.dtype,
                              device=topk_ids.device)
    num_tokens_per_expert = (flatten_topk_ids.unsqueeze(-1) == experts_id).to(
        torch.float32).sum(0)
    topk_scales = topk_weights.view(-1).index_select(
        0, sorted_topk_ids).unsqueeze(-1)
    group_list = num_tokens_per_expert.cumsum(dim=0).to(torch.int64)

    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[sorted_hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
    )[0]
    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[sorted_hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
    )[0]

    if is_310p():
        gate_up_out = torch_npu.npu_swiglu(gate_up_out.to(torch.float32)).to(
            torch.float16)
    else:
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)
    gate_up_out *= topk_scales
    if is_310p():
        gate_up_out = torch_npu.npu_swiglu(gate_up_out.to(torch.float32)).to(
            torch.float16)
    else:
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)
    gate_up_out *= topk_scales

    down_out_list = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
    )[0]
    down_out_list = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=group_list,
    )[0]

    unsorted_topk_ids = torch.argsort(sorted_topk_ids.float()).to(torch.int32)
    unsorted_hidden_states = down_out_list.index_select(0, unsorted_topk_ids)
    final_hidden_states = unsorted_hidden_states.reshape(
        bsz, top_k // ep_size, -1).sum(1)
    unsorted_topk_ids = torch.argsort(sorted_topk_ids.float()).to(torch.int32)
    unsorted_hidden_states = down_out_list.index_select(0, unsorted_topk_ids)
    final_hidden_states = unsorted_hidden_states.reshape(
        bsz, top_k // ep_size, -1).sum(1)

    return final_hidden_states
    return final_hidden_states


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NOTE: Currently, this self.use_aclgraph is only used in
        # UnquantizedFusedMoEMethod.forward_oot to decide whether to use in
        # ops/fused_moe.py:568 to circumvent torch.randint_like not supported issue.
        # Once torch.randint_like is supported or removed, this flag can be removed.
        vllm_config = get_current_vllm_config()
        self.use_aclgraph = (vllm_config.compilation_config.level
                             == CompilationLevel.PIECEWISE
                             and not vllm_config.model_config.enforce_eager)

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(
            1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(
            1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

        if not is_310p():
            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NOTE: Currently, this self.use_aclgraph is only used in
        # UnquantizedFusedMoEMethod.forward_oot to decide whether to use in
        # ops/fused_moe.py:568 to circumvent torch.randint_like not supported issue.
        # Once torch.randint_like is supported or removed, this flag can be removed.
        vllm_config = get_current_vllm_config()
        self.use_aclgraph = (vllm_config.compilation_config.level
                             == CompilationLevel.PIECEWISE
                             and not vllm_config.model_config.enforce_eager)
    self.transpose = True

    def forward_oot_normal(
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
            logical_replica_count: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        topk_weights, topk_ids, row_idx = select_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.mc2_group = get_mc2_group()

        for method in {
                AllGatherCommImpl, AlltoAllCommImpl, MC2CommImpl,
                NaiveMulticastCommImpl
        }:
            setattr(
                self, method.__name__.lower(),
                method(moe_config=self.moe_config))  # type: ignore[abstract]

    def maybe_all_reduce_tensor_model_parallel(
            self, final_hidden_states: torch.Tensor):
        """NOTE(Yizhou): This is to override the parent class method. In `mc2commimpl`,
        and `alltoallcommimpl`, we do not need to all-reduce the final outputs since
        the outputs are already aggregated across tensor parallel ranks in the
        `finalize` function. In `allgathercommimpl`, we still need to all-reduce the
        outputs since each rank only has partial outputs.
        """
        forward_context = get_forward_context()
        moe_comm_method_name = forward_context.moe_comm_method_name
        if moe_comm_method_name in {"alltoallcommimpl", "mc2commimpl"}:
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        assert self.quant_method is not None

        forward_context = get_forward_context()
        moe_comm_method_name = forward_context.moe_comm_method_name

        forward_context.moe_comm_method = getattr(self, moe_comm_method_name)

        hidden_states, router_logits = forward_context.moe_comm_method.prepare(
            hidden_states=hidden_states, router_logits=router_logits)

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


class AscendSharedFusedMoE(AscendFusedMoE):

    def __init__(
        self,
        shared_experts: torch.nn.Module,
        use_overlapped: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._shared_experts = shared_experts
        self.use_overlapped = use_overlapped

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_out = self._shared_experts(hidden_states)

        # NOTE: This is exactly the opposite of `maybe_all_reduce_tensor_model_parallel`
        forward_context = get_forward_context()
        moe_comm_method_name = forward_context.moe_comm_method_name
        if moe_comm_method_name in {"alltoallcommimpl", "mc2commimpl"}:
            shared_out = tensor_model_parallel_all_reduce(shared_out)

        fused_out = super().forward(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return shared_out, fused_out


UnquantizedFusedMoEMethod.__init__ = unquantized_fused_moe_init_func
UnquantizedFusedMoEMethod.process_weights_after_loading = process_weights_after_loading
UnquantizedFusedMoEMethod.forward_oot = forward_oot
