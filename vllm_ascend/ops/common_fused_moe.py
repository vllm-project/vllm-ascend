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
from typing import Any, Callable, Optional

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import (get_dp_group, get_ep_group, get_tp_group,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod, determine_expert_map,
    get_compressed_expert_map)
from vllm.model_executor.layers.shared_fused_moe import SharedFusedMoE

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.eplb.core.eplb_utils import determine_default_log2phy_map
from vllm_ascend.ops.expert_load_balancer import ExpertLoadBalancer
from vllm_ascend.ops.moe.experts_selector import select_experts
from vllm_ascend.ops.moe.moe_comm_method import setup_moe_comm_method
from vllm_ascend.quantization.w8a8_dynamic import \
    AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.utils import (ACL_FORMAT_FRACTAL_NZ, enable_sp, is_310p,
                               is_enable_nz, npu_stream_switch,
                               shared_expert_dp_enabled,
                               shared_experts_compute_stream)


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self, moe: FusedMoEConfig = None):

        super().__init__(moe=moe)
        self.dynamic_eplb = get_ascend_config().dynamic_eplb
        self.transpose = True

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod,
              self).process_weights_after_loading(layer)
        if self.transpose:
            w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(
                1, 2).contiguous()
            layer.w13_weight = torch.nn.Parameter(w13_data,
                                                  requires_grad=False)

            w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(
                1, 2).contiguous()
            layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

            self.transpose = False
        else:
            w13_data = self._maybe_pad_weight(layer.w13_weight.data)
            layer.w13_weight = torch.nn.Parameter(w13_data,
                                                  requires_grad=False)

            w2_data = self._maybe_pad_weight(layer.w2_weight.data)
            layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

        if not is_310p() and is_enable_nz(layer.w13_weight.data.dtype):
            layer.w13_weight.data = torch_npu.npu_format_cast(
                layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)

    def apply(self,
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
              enable_force_load_balance: bool = False,
              shared_experts: Optional[Any] = None,
              **kwargs) -> torch.Tensor:

        topk_weights, topk_ids = select_experts(
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

        topk_weights = topk_weights.to(x.dtype)
        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            topk_ids = torch.randint_like(topk_ids, 0, global_num_experts)

        moe_comm_method = get_forward_context().moe_comm_method
        return moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            shared_experts=shared_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            dynamic_eplb=self.dynamic_eplb)


class AscendFusedMoE(FusedMoE):
    moe_counter = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_experts = kwargs["num_experts"]
        intermediate_size = kwargs["intermediate_size"]

        AscendFusedMoE.moe_counter += 1
        self.moe_instance_id = AscendFusedMoE.moe_counter

        self.expert_map = None
        self.log2phy = None

        if self.quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod(
                self.moe_config)
        else:
            self.quant_method = self.quant_config.get_quant_method(
                self, self.layer_name)

        assert self.quant_method is not None

        self.moe_config.tp_group = get_tp_group()
        self.moe_config.dp_group = get_dp_group()
        self.moe_config.ep_group = get_ep_group()
        self.moe_config.mc2_group = get_mc2_group()
        ascend_config = get_ascend_config()
        self.dynamic_eplb = ascend_config.dynamic_eplb or ascend_config.expert_map_record_path
        self.expert_map_path = ascend_config.expert_map_path
        self.global_redundant_expert_num = ascend_config.init_redundancy_expert
        self.global_num_experts = num_experts + self.global_redundant_expert_num
        # TODO: Flag for static expert placement. This is a temporary workaround
        # to allow dynamic EPLB with float weights by skipping quantization checks.
        self.static_eplb_enabled = False
        if self.custom_routing_function is None and self.e_score_correction_bias is not None:
            vllm_config = get_current_vllm_config()
            self.e_score_correction_bias.data = self.e_score_correction_bias.data.to(
                dtype=vllm_config.model_config.dtype)
        # static eplb initializing with expert_map_path
        init_eplb_enable = False
        if self.expert_map_path and os.path.exists(
                self.expert_map_path) and os.access(self.expert_map_path,
                                                    os.R_OK):
            self.expert_load_balancer = ExpertLoadBalancer(
                self.expert_map_path, num_experts)
            self.expert_load_balancer.check_expert_map_tensor()
            self.global_redundant_expert_num = (
                self.expert_load_balancer.get_global_redundant_expert_num())
            self.global_num_experts = num_experts + self.global_redundant_expert_num
            try:
                self.local_num_experts, self.expert_map = (
                    self.expert_load_balancer.get_rank_placement_map(
                        self.moe_instance_id, self.ep_rank))
                self.log2phy = self.expert_load_balancer.get_rank_log2phy_map(
                    self.moe_instance_id, self.ep_rank).npu()
                init_eplb_enable = True
            except Exception as e:
                logger.warning(
                    f"Init expert map of mtp/eagle when using sample.{e}")
                self.local_num_experts, self.expert_map = determine_expert_map(
                    self.ep_size, self.ep_rank, self.global_num_experts)
                self.log2phy = determine_default_log2phy_map(
                    self.global_num_experts, self.ep_size, self.ep_rank).npu()
        else:
            # init moe.
            self.local_num_experts, self.expert_map = determine_expert_map(
                self.ep_size, self.ep_rank, self.global_num_experts)
            # dynamic eplb initializing with not expert_map_path
            if self.dynamic_eplb:
                self.log2phy = determine_default_log2phy_map(
                    self.global_num_experts, self.ep_size, self.ep_rank).npu()
        if self.expert_map is not None and isinstance(self.expert_map,
                                                      torch.Tensor):
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.", self.ep_rank, self.ep_size, self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self.expert_map))
        local_num_experts = (torch.sum(
            self.expert_map != -1) if self.expert_map is not None else
                             self.global_num_experts)
        if self.dynamic_eplb:
            self.moe_load = torch.zeros(local_num_experts,
                                        dtype=torch.int64).npu()

        if init_eplb_enable and (
                not hasattr(self.quant_method, "quant_method")
                or not isinstance(self.quant_method.quant_method,
                                  AscendW8A8DynamicFusedMoEMethod)):
            raise ValueError("Eplb supports only w8a8_dynamic quantization.")

        self.moe_config.num_experts = self.global_num_experts
        self.moe_config.num_local_experts = self.local_num_experts
        self.moe_config.original_num_experts = num_experts

        moe_quant_params = {
            "num_experts": local_num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size_per_partition":
            self.intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if (self.quant_method.__class__.__name__
                in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod")):
            moe_quant_params["intermediate_size_full"] = intermediate_size
        self.quant_method.create_weights(layer=self, **moe_quant_params)

        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp

        setup_moe_comm_method(self.moe_config)

    def update_expert_map(self, new_expert_map):
        self.expert_map = new_expert_map

    def get_map(self):
        return self.expert_map

    def get_log2phy_map(self):
        return self.log2phy

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
        return torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states)

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        assert self.quant_method is not None

        # For w8a8 dynamic we can do npu_dynamic_quant and gate in parallel.
        quantized_x_for_share, dynamic_scale_for_share = None, None

        forward_context = get_forward_context()

        # Load balancing for token distribution among experts in dummy_run
        # TODO: The community only considers load balancing when DP > 1.
        # This approach may overlook some extreme scenarios.
        enable_force_load_balance = forward_context.in_profile_run

        hidden_states, router_logits = forward_context.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=forward_context.sp_enabled,
            enable_shared_expert_dp=self.enable_shared_expert_dp)

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
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
            shared_experts=None,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num)

        if isinstance(final_hidden_states, tuple):
            final_hidden_states, group_list_type, expert_tokens = final_hidden_states

            if self.dynamic_eplb:
                self.moe_load += expert_tokens if group_list_type == 1 else \
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
        if enable_sp():
            logger.info_once(
                "Sequence parallelism is enabled, shared experts are replicated for best performance."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared_out, fused_out = AscendFusedMoE.forward(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return shared_out, fused_out

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        # Make sure the shared experts stream begins after hidden_states are ready.
        if self.multistream_overlap_shared_expert:
            shared_experts_compute_stream().wait_stream(  # type: ignore
                torch.npu.current_stream())
        with npu_stream_switch(shared_experts_compute_stream(),
                               enabled=self.multistream_overlap_shared_expert):
            # Use a separate stream to run shared experts.
            # Note that currently we only support calculations in separate streams with aclgraph.
            # Communication operations in another stream might cause unknown errors.
            shared_out = self._shared_experts(hidden_states)

        fused_output = AscendFusedMoE.forward_impl(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        # Make sure the default stream waits for the shared experts stream to finish.
        if self.multistream_overlap_shared_expert:
            torch.npu.current_stream().wait_stream(
                shared_experts_compute_stream())
        # NOTE: This is exactly the opposite of `maybe_all_reduce_tensor_model_parallel`
        forward_context = get_forward_context()
        moe_comm_type = forward_context.moe_comm_type
        if moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2} \
                and not shared_expert_dp_enabled():
            shared_out = tensor_model_parallel_all_reduce(shared_out)
        return shared_out, fused_output
