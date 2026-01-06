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
from vllm.config import get_current_vllm_config
from vllm.distributed import (get_dp_group, get_ep_group, get_tp_group,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod, get_compressed_expert_map)
from vllm.model_executor.layers.fused_moe.shared_fused_moe import \
    SharedFusedMoE

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
from vllm_ascend.eplb.utils import moe_load_async_stream
from vllm_ascend.flash_common3_context import (get_flash_common3_context,
                                               set_flash_common3_context)
from vllm_ascend.ops.fused_moe.experts_selector import (select_experts,
                                                        zero_experts_compute)
from vllm_ascend.ops.fused_moe.moe_comm_method import (AllGatherCommImpl,
                                                       FusedExpertsResult,
                                                       setup_moe_comm_method)
from vllm_ascend.ops.fused_moe.prepare_finalize import QuantType
from vllm_ascend.quantization.w4a8_dynamic import \
    AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.quantization.w8a8_dynamic import \
    AscendW8A8DynamicFusedMoEMethod
from vllm_ascend.utils import (AscendDeviceType, enable_sp,
                               get_ascend_device_type, maybe_trans_nz,
                               npu_stream_switch, shared_expert_dp_enabled,
                               shared_experts_calculation_stream)


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self, moe: FusedMoEConfig = None):

        super().__init__(moe=moe)
        self.dynamic_eplb = get_ascend_config().dynamic_eplb

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod,
              self).process_weights_after_loading(layer)

        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(
            1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(
            1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

        if get_ascend_device_type() != AscendDeviceType._310P:
            layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
            layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)

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
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)
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

        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
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
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0),
                                       global_num_experts,
                                       device=topk_ids.device)
            topk_ids = torch.argsort(
                random_matrix, dim=1)[:, :topk_ids.size(1)].to(topk_ids.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        final_hidden_states = moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            shared_experts=shared_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            dynamic_eplb=self.dynamic_eplb,
            mc2_mask=kwargs.get("mc2_mask", None))
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        return final_hidden_states


class AscendFusedMoE(FusedMoE):
    moe_counter = -1
    gate_stream: Optional[torch.npu.Stream] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_experts = kwargs["num_experts"]
        intermediate_size = kwargs["intermediate_size"]

        AscendFusedMoE.moe_counter += 1
        self.moe_instance_id = AscendFusedMoE.moe_counter

        self._expert_map = None
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
        self.moe_config.supports_eplb = self.quant_method.supports_eplb
        ascend_config = get_ascend_config()
        # flashcommon3 gate stream
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
        if self.multistream_overlap_gate and AscendFusedMoE.gate_stream is None:
            AscendFusedMoE.gate_stream = torch.npu.Stream()
        if self.custom_routing_function is None and self.e_score_correction_bias is not None:
            vllm_config = get_current_vllm_config()
            self.e_score_correction_bias.data = self.e_score_correction_bias.data.to(
                dtype=vllm_config.model_config.dtype)

        # init moe
        self._expert_map, self.log2phy, self.global_redundant_expert_num = init_eplb_config(
            ascend_config, self.moe_instance_id, self.moe_config)
        self.global_num_experts = num_experts + self.global_redundant_expert_num
        self.dynamic_eplb = (ascend_config.dynamic_eplb
                             or ascend_config.expert_map_record_path) and (
                                 self.log2phy is not None)
        self.local_num_experts = (torch.sum(
            self._expert_map != -1).item() if self._expert_map is not None else
                                  self.global_num_experts)
        if self._expert_map is not None:
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.", self.ep_rank, self.ep_size, self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self._expert_map))
        if self.dynamic_eplb:
            self.moe_load = torch.zeros(self.local_num_experts,
                                        dtype=torch.int64).npu()

        self.moe_config.num_experts = self.global_num_experts
        self.moe_config.num_local_experts = self.local_num_experts
        self.moe_config.global_redundant_expert_num = self.global_redundant_expert_num

        moe_quant_params = {
            "num_experts": self.local_num_experts,
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
        self.quant_type = self._get_quant_type()

    def _get_quant_type(self) -> QuantType:
        quant_method = self.quant_method
        if not hasattr(quant_method,
                       "quant_method") or quant_method.quant_method is None:
            return QuantType.NONE

        method = quant_method.quant_method

        if isinstance(method, AscendW8A8DynamicFusedMoEMethod):
            return QuantType.W8A8
        elif isinstance(method, AscendW4A8DynamicFusedMoEMethod):
            return QuantType.W4A8
        else:
            return QuantType.NONE

    def update_expert_map(self, new_expert_map):
        self._expert_map = new_expert_map

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

        forward_context = get_forward_context()
        if self.multistream_overlap_gate:
            assert AscendFusedMoE.gate_stream is not None
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            AscendFusedMoE.gate_stream.wait_stream(torch.npu.current_stream())
            with npu_stream_switch(AscendFusedMoE.gate_stream,
                                   enabled=self.multistream_overlap_gate):
                # share_expert
                assert fc3_context.shared_experts is not None
                shared_out = fc3_context.shared_experts(hidden_states)
                # NOTE: This is exactly the opposite of `maybe_all_reduce_tensor_model_parallel`
                moe_comm_type = forward_context.moe_comm_type
                if moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2} \
                        and not shared_expert_dp_enabled():
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
                set_flash_common3_context(shared_out=shared_out)

                topk_weights, topk_ids = select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    top_k=self.top_k,
                    use_grouped_topk=self.use_grouped_topk,
                    renormalize=self.renormalize,
                    topk_group=self.topk_group,
                    num_expert_group=self.num_expert_group,
                    custom_routing_function=self.custom_routing_function,
                    scoring_func=self.scoring_func,
                    routed_scaling_factor=self.routed_scaling_factor,
                    e_score_correction_bias=self.e_score_correction_bias,
                    global_num_experts=self.global_num_experts)

                if isinstance(forward_context.moe_comm_method,
                              AllGatherCommImpl):
                    topk_weights = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                        topk_weights, True, True)
                    topk_ids = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                        topk_ids, True, True)

                set_flash_common3_context(topk_weights=topk_weights,
                                          topk_ids=topk_ids)

        hidden_states, router_logits, mc2_mask, context_metadata = forward_context.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=forward_context.sp_enabled,
            enable_shared_expert_dp=self.enable_shared_expert_dp,
            quant_type=self.quant_type)

        # Make sure the default stream waits for the gate stream to finish.
        if self.multistream_overlap_gate:
            torch.npu.current_stream().wait_stream(AscendFusedMoE.gate_stream)

        if isinstance(hidden_states, tuple):
            hidden_states, pertoken_scale = hidden_states
        else:
            pertoken_scale = None

        # Matrix multiply.
        fused_experts_results: FusedExpertsResult = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            pertoken_scale=pertoken_scale,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self._expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
            shared_experts=None,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            mc2_mask=mc2_mask)

        if self.dynamic_eplb:
            expert_tokens = fused_experts_results.expert_tokens
            group_list_type = fused_experts_results.group_list_type
            assert expert_tokens is not None and group_list_type is not None, \
                "expert_tokens and group_list_type should not be None when dynamic_eplb is enabled."
            moe_load_stream = moe_load_async_stream()
            cur_stream = torch.npu.current_stream()
            moe_load_stream.wait_stream(cur_stream)
            with npu_stream_switch(moe_load_stream):
                self.moe_load += expert_tokens if group_list_type == 1 else \
                    torch.cat([expert_tokens[:1], expert_tokens[1:] - expert_tokens[:-1]])
            cur_stream.wait_stream(moe_load_stream)

        routed_out = forward_context.moe_comm_method.finalize(
            hidden_states=fused_experts_results.routed_out,
            reduce_results=self.reduce_results,
            context_metadata=context_metadata)

        return routed_out


class AscendSharedFusedMoE(SharedFusedMoE, AscendFusedMoE):

    def __init__(
        self,
        shared_experts: torch.nn.Module,
        gate: Optional[torch.nn.Module] = None,
        use_overlapped: bool = True,
        **kwargs,
    ):
        AscendFusedMoE.__init__(self, **kwargs)

        self._shared_experts = shared_experts
        self.use_overlapped = use_overlapped
        self.shared_expert_stream = None
        ascend_config = get_ascend_config()
        self.multistream_overlap_shared_expert = ascend_config.multistream_overlap_shared_expert
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
        if enable_sp():
            logger.info_once(
                "Sequence parallelism is enabled, shared experts are replicated for best performance."
            )

        self._gate = gate

    @property
    def gate(self) -> Optional[torch.nn.Module]:
        return self._gate if self.use_overlapped else None

    @property
    def is_internal_router(self) -> bool:
        return False

    @property
    def use_dp_chunking(self) -> bool:
        """This func routes to the chunked forward path using the FlashInfer Cutlass kernel
        only when data parallelism (DP) is enabled. Thus just returning False in vllm-ascend
        """
        return False

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
        shared_out = None
        if not self.multistream_overlap_gate:
            # Make sure the shared experts stream begins after hidden_states are ready.
            if self.multistream_overlap_shared_expert:
                shared_experts_calculation_stream(
                ).wait_stream(  # type: ignore
                    torch.npu.current_stream())
            with npu_stream_switch(
                    shared_experts_calculation_stream(),
                    enabled=self.multistream_overlap_shared_expert):
                # Use a separate stream to run shared experts.
                shared_out = self._shared_experts(hidden_states)
        else:
            set_flash_common3_context(shared_experts=self._shared_experts)

        routed_out = AscendFusedMoE.forward_impl(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        if not self.multistream_overlap_gate:
            # Make sure the default stream waits for the shared experts stream to finish.
            if self.multistream_overlap_shared_expert:
                torch.npu.current_stream().wait_stream(
                    shared_experts_calculation_stream())

            # NOTE: This is exactly the opposite of `maybe_all_reduce_tensor_model_parallel`
            forward_context = get_forward_context()
            moe_comm_type = forward_context.moe_comm_type
            if moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2} \
                    and not shared_expert_dp_enabled():
                shared_out = tensor_model_parallel_all_reduce(shared_out)
        else:
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            shared_out = fc3_context.shared_out

        return shared_out, routed_out
