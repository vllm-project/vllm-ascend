#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
from vllm_ascend.lora.fused_moe import sync_lora_context
from vllm_ascend.ops.fused_moe.experts_selector import select_experts, zero_experts_compute
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.methods.base import get_moe_num_logical_experts
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, maybe_trans_nz


@dataclass
class FusedMoEResult:
    routed_out: torch.Tensor
    before_dispatch_evt: torch.npu.Event | None = None
    before_gmm2_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    swiglu_limit: float = 0.0
    swiglu_alpha: float = 1.0
    swiglu_beta: float = 0.0


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig = None, tid2eid=None):
        super().__init__(moe=moe)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        self.tid2eid = tid2eid
        self.lora_context = None

    def set_lora_context(self, lora_context) -> None:
        self.lora_context = lora_context

    @property
    def is_monolithic(self) -> bool:
        return False

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # Ascend uses its own MoE communication and forward_impl path.
        # Do not let upstream modular-kernel initialization replace it.
        return None

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)

        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

        # TODO: Current dispatch_ffn_combine/mega_moe fusion operator ONLY supports NZ format.
        # Therefore, we must cast weights to NZ when fusion is enabled.
        # Once the underlying dispatch_ffn_combine/mega_moe operator is updated to support
        # ND format (or other formats), remove this specific 'if' check and the forced
        # npu_format_cast. At that point, the operator should be able to handle weights
        # in their native format without explicit casting here.
        enable_fused_mc2 = get_ascend_config().enable_fused_mc2
        if enable_fused_mc2:
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
            if enable_fused_mc2 == 1 and self.dynamic_eplb:
                layer.w13_weight_list = [weight.clone() for weight in layer.w13_weight.data.unbind(dim=0)]
                layer.w2_weight_list = [weight.clone() for weight in layer.w2_weight.data.unbind(dim=0)]
                del layer.w13_weight
                del layer.w2_weight
                torch.npu.empty_cache()
        else:
            layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
            layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: torch.Tensor | None = None,
        mc2_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)
        input_ids = getattr(get_forward_context(), "input_ids", None)
        num_shared_experts = getattr(layer, "n_shared_experts", 0)
        if num_shared_experts is None:
            num_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
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
            num_experts=num_logical_experts,
            tid2eid=self.tid2eid,
            input_ids=input_ids,
        )
        try:
            _vllm_config = get_current_vllm_config()
        except AssertionError:
            _vllm_config = None
        model_config = None if _vllm_config is None else _vllm_config.model_config
        if model_config is not None and model_config.enable_return_routed_experts:
            capturer = getattr(layer, "_ascend_routed_experts_capturer", None)
            if capturer is not None:
                capturer.capture(layer_id=layer.layer_id, topk_ids=topk_ids)

        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=num_logical_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        topk_weights = topk_weights.to(x.dtype)
        # This is a naive implementation for experts load balance so as to
        # avoid accumulating too much tokens on a single rank. It is only
        # activated when doing profile runs.
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        if getattr(layer, "swigluoai_uninterleave", False):
            activation = "swigluoai_uninterleave"

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        w13_weight_list = getattr(layer, "w13_weight_list", None)
        w2_weight_list = getattr(layer, "w2_weight_list", None)
        has_split_weight_lists = isinstance(w13_weight_list, list) and isinstance(w2_weight_list, list)
        if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
            if self.dynamic_eplb and not has_split_weight_lists:
                logger.warning_once(
                    "FUSED_MC2 is enabled with dynamic EPLB, but unquantized MoE weights are not split into "
                    "tensor lists. This may cause accuracy issues or communication hangs."
                )
            w1 = w13_weight_list if isinstance(w13_weight_list, list) else [layer.w13_weight]
            w2 = w2_weight_list if isinstance(w2_weight_list, list) else [layer.w2_weight]
            w1_scale = [torch.tensor([], dtype=torch.int64)]
            w2_scale = [torch.tensor([], dtype=torch.int64)]
            w1_scale_bias = [torch.tensor([], dtype=torch.float32)]
            w2_scale_bias = [torch.tensor([], dtype=torch.float32)]
        else:
            w1 = w13_weight_list if isinstance(w13_weight_list, list) else layer.w13_weight
            w1_scale = None
            w2 = w2_weight_list if isinstance(w2_weight_list, list) else layer.w2_weight
            w2_scale = None
            w1_scale_bias = None
            w2_scale_bias = None

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                w1_bias=layer.w13_bias if self.moe.has_bias else None,
                w2_bias=layer.w2_bias if self.moe.has_bias else None,
                quant_type=QuantType.NONE,
                dynamic_eplb=self.dynamic_eplb,
                expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy,
                pertoken_scale=pertoken_scale,
                activation=activation,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                swiglu_limit=layer.swiglu_limit,
                swiglu_alpha=getattr(layer, "swiglu_alpha", 1.0),
                swiglu_beta=getattr(layer, "swiglu_beta", 0.0),
                lora_context=getattr(layer, "_ascend_moe_lora_context", None),
            )
        )
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states.routed_out += zero_expert_result
        return final_hidden_states


def use_multistage_eplb_load(dynamic_eplb: bool, policy_type: int, collection_interval: int) -> bool:
    """Whether EPLB should retain a separate expert-load vector per step."""
    return dynamic_eplb and policy_type == 3 and collection_interval > 1


def make_eplb_placement_config(eplb_config, num_redundant_experts: int) -> SimpleNamespace:
    """Build the minimal config view consumed by init_eplb_config."""
    return SimpleNamespace(
        expert_map_path=eplb_config.expert_map_path,
        dynamic_eplb=eplb_config.dynamic_eplb,
        num_redundant_experts=num_redundant_experts,
    )


class AscendRoutedExperts(RoutedExperts):  # type: ignore[no-redef]
    """Ascend-owned routed expert container.

    This keeps Ascend quant-method selection with the expert weight owner,
    matching upstream vLLM's RoutedExperts responsibility split.
    """

    moe_counter = -1

    def __init__(self, *args, tid2eid=None, n_shared_experts: int = 0, **kwargs):
        object.__setattr__(self, "tid2eid", tid2eid)
        super().__init__(*args, **kwargs)
        ascend_config = get_ascend_config()
        self.enable_npugraph_ex_static_kernel = ascend_config.ascend_compilation_config.enable_static_kernel
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self.dynamic_eplb = False
        self.multi_stage = False
        self.load_counter = None
        self.num_iter = None
        self.moe_load: torch.Tensor | None = None
        self.ascend_expert_map = None
        self.log2phy = None
        self.global_redundant_expert_num = 0
        self.init_eplb(n_shared_experts)

        vllm_config = get_current_vllm_config()

        if (
            self.custom_routing_function is None
            and self.e_score_correction_bias is not None
            and not vllm_config.model_config.is_deepseek_mla
        ):
            self.e_score_correction_bias.data = self.e_score_correction_bias.data.to(
                dtype=vllm_config.model_config.dtype
            )

    def init_eplb(self, n_shared_experts):
        ascend_config = get_ascend_config()
        mix_placement = getattr(ascend_config, "mix_placement", False)

        # EPLB initialization (Ascend-specific; mirrors old AscendFusedMoE logic).
        AscendRoutedExperts.moe_counter += 1
        self.moe_instance_id = AscendRoutedExperts.moe_counter

        eplb_config = get_ascend_config().eplb_config

        # The upstream FusedMoE factory has already included redundant expert
        # slots in moe_config and allocated RoutedExperts weights accordingly.
        # Ascend's placement builder operates on logical expert IDs, so give it
        # a shallow config view with the logical count.
        placement_moe_config = copy(self.moe_config)
        placement_moe_config.num_experts = self.moe_config.num_logical_experts + (
            n_shared_experts if mix_placement else 0
        )
        allocated_redundancy = self.moe_config.num_experts - self.moe_config.num_logical_experts
        if eplb_config.num_redundant_experts not in (0, allocated_redundancy):
            raise ValueError(
                "Conflicting EPLB redundant expert counts: "
                f"allocated={allocated_redundancy}, Ascend={eplb_config.num_redundant_experts}."
            )
        placement_eplb_config = make_eplb_placement_config(eplb_config, allocated_redundancy)
        vllm_config = get_current_vllm_config()
        (
            self.global_expert_map,
            self.ascend_expert_map,
            self.log2phy,
            self.global_redundant_expert_num,
        ) = init_eplb_config(
            placement_eplb_config,
            AscendRoutedExperts.moe_counter,
            placement_moe_config,
            mix_placement,
            n_shared_experts,
            tp_size=vllm_config.parallel_config.tensor_parallel_size,
        )

        self.moe_config.global_redundant_expert_num = self.global_redundant_expert_num
        local_num_experts = self.moe_config.num_local_experts
        expected_local_num_experts = (
            placement_moe_config.num_experts + self.global_redundant_expert_num
        ) // self.moe_config.ep_size
        if local_num_experts != expected_local_num_experts:
            raise ValueError(
                "EPLB local expert capacity mismatch: "
                f"allocated={local_num_experts}, placement={expected_local_num_experts}. "
                "Ensure vLLM and Ascend use the same redundant expert count."
            )
        # Keep ExpertMapManager's physical-expert map until checkpoint loading
        # finishes. The upstream loader uses it to place both original and
        # redundant physical experts. Ascend execution uses ascend_expert_map,
        # which maps logical expert IDs to the local physical slots.

        self.dynamic_eplb = eplb_config.dynamic_eplb and (self.log2phy is not None)
        self.multi_stage = False
        self.load_counter = None
        self.num_iter = None
        self.moe_load = torch.zeros(local_num_experts, dtype=torch.int64).npu()
        # Only FlashLB consumes a time series of expert loads. Other EPLB
        # policies (including the default SwiftBalance policy) expect one load
        # vector per layer and rank. Using the collection interval alone here
        # adds an unexpected window dimension and produces
        # [layer, rank, interval, expert] after all-gather.
        if use_multistage_eplb_load(
            self.dynamic_eplb,
            eplb_config.eplb_policy_type,
            eplb_config.expert_heat_collection_interval,
        ):
            self.multi_stage = True
            self.load_counter = torch.tensor(0, dtype=torch.int32, device="npu")
            self.num_iter = eplb_config.expert_heat_collection_interval
            self.moe_load = torch.zeros((self.num_iter, local_num_experts), dtype=torch.int32, device="npu")

        # Register this MoE layer with EPLB for PP compatibility.
        # PPMissingLayer (nn.Identity) never calls AscendFusedMoE.__init__,
        # so only real MoE layers on this rank are registered.
        VllmEplbAdaptor.register_layer(self)

    def _get_quant_method(self, prefix, quant_config, moe_config):
        if quant_config is None:
            return AscendUnquantizedFusedMoEMethod(moe_config, tid2eid=self.tid2eid)
        return quant_config.get_quant_method(self, prefix, tid2eid=self.tid2eid)

    def get_eplb_parameter(self, name: str):
        """Return an expert parameter from the refactored weight owner."""
        return getattr(self, name)

    def update_expert_map(self, new_expert_map: torch.Tensor) -> None:
        """Update both the runner and routed-expert map references."""
        self.ascend_expert_map = new_expert_map
        self.expert_map_manager._expert_map = new_expert_map

    def get_log2phy_map(self) -> torch.Tensor | None:
        return self.log2phy

    @property
    def ep_rank(self) -> int:
        return self.moe_config.ep_rank

    def clear_moe_load(self) -> None:
        assert self.moe_load is not None
        self.moe_load.zero_()
        if self.multi_stage:
            assert self.load_counter is not None
            self.load_counter.zero_()

    @property
    def quant_type(self) -> QuantType:
        quant_type = QuantType.NONE
        method = getattr(self.quant_method, "quant_method", None)
        if method is not None:
            quant_type = getattr(method, "quant_type", QuantType.NONE)
        return quant_type

    def no_shared_forward_impl(
        self,
        *,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        return_with_event: bool = False,
    ):
        forward_context = get_forward_context()
        # When static kernels are enabled, the forward pass runs twice
        # (compilation + capture), causing moe_layer_index to overflow.
        if self.enable_npugraph_ex_static_kernel and forward_context.all_moe_layers:
            moe_layer_index = forward_context.moe_layer_index % (len(forward_context.all_moe_layers))
            forward_context.moe_layer_index = moe_layer_index

        # Load balancing for token distribution among experts in dummy_run.
        enable_force_load_balance = _EXTRA_CTX.in_profile_run

        lora_context = getattr(self, "_ascend_moe_lora_context", None)
        if lora_context is not None:
            sync_lora_context(self.quant_method, lora_context)

        prepare_output = _EXTRA_CTX.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=_EXTRA_CTX.flash_comm_v1_enabled,
            enable_shared_expert_dp=self.enable_shared_expert_dp,
            quant_type=self.quant_type,
        )
        hidden_states = prepare_output.hidden_states
        router_logits = prepare_output.router_logits
        mc2_mask = prepare_output.mc2_mask
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape
        pertoken_scale = prepare_output.pertoken_scale

        fused_experts_results: FusedExpertsResult = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            pertoken_scale=pertoken_scale,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            num_experts=self.moe_config.num_experts,
            expert_map=self.ascend_expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            mc2_mask=mc2_mask,
        )

        if self.dynamic_eplb and _EXTRA_CTX.eplb_heat_collection_status:
            expert_tokens = fused_experts_results.expert_tokens
            group_list_type = fused_experts_results.group_list_type
            assert expert_tokens is not None and group_list_type is not None, (
                "expert_tokens and group_list_type should not be None when dynamic_eplb is enabled."
            )
            local_load = (
                expert_tokens
                if group_list_type == 1
                else torch.cat([expert_tokens[:1], expert_tokens[1:] - expert_tokens[:-1]])
            )
            assert self.moe_load is not None
            if self.multi_stage:
                assert self.load_counter is not None and self.num_iter is not None
                cur_iter = torch.remainder(self.load_counter, self.num_iter)
                self.moe_load.index_add_(
                    dim=0, index=cur_iter, source=local_load.to(torch.int32, non_blocking=True).view(1, -1)
                )
                self.load_counter.add_(1)
            else:
                self.moe_load.add_(local_load)

        routed_out = _EXTRA_CTX.moe_comm_method.finalize(
            hidden_states=fused_experts_results.routed_out,
            reduce_results=isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl),
            padded_hidden_states_shape=padded_hidden_states_shape,
        )

        # Clear per-forward LoRA state from long-lived singletons.
        if lora_context is not None:
            sync_lora_context(self.quant_method, None)

        if return_with_event:
            return FusedMoEResult(
                routed_out=routed_out,
                before_dispatch_evt=fused_experts_results.before_dispatch_evt,
                before_gmm2_evt=fused_experts_results.before_gmm2_evt,
                before_combine_evt=fused_experts_results.before_combine_evt,
                swiglu_limit=fused_experts_results.swiglu_limit,
                swiglu_alpha=getattr(fused_experts_results, "swiglu_alpha", 1.0),
                swiglu_beta=getattr(fused_experts_results, "swiglu_beta", 0.0),
            )

        # The vLLM FusedMoE forward_impl does not return events.
        return routed_out
