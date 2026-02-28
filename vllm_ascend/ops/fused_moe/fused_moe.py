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
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group, tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, UnquantizedFusedMoEMethod, get_compressed_expert_map
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE

from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.16.0"):
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase  # type: ignore
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter  # type: ignore
    from vllm.model_executor.layers.fused_moe.runner.default_moe_runner import DefaultMoERunner  # type: ignore

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
from vllm_ascend.flash_common3_context import get_flash_common3_context, set_flash_common3_context
from vllm_ascend.ops.fused_moe.experts_selector import select_experts, zero_experts_compute
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult, setup_moe_comm_method
from vllm_ascend.quantization.methods.base import QuantType
from vllm_ascend.utils import (
    enable_sp,
    maybe_trans_nz,
    npu_stream_switch,
    shared_expert_dp_enabled,
    shared_experts_calculation_stream,
    vllm_version_is,
)


@dataclass
class FusedMoEResult:
    routed_out: torch.Tensor
    before_dispatch_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None


@dataclass
class FusedMoEEvents:
    before_routed_experts: torch.npu.Event
    before_dispatch: torch.npu.Event | None = field(default=None)
    before_combine: torch.npu.Event | None = field(default=None)


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig = None):
        super().__init__(moe=moe)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)

        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        layer.w13_weight = torch.nn.Parameter(w13_data, requires_grad=False)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        layer.w2_weight = torch.nn.Parameter(w2_data, requires_grad=False)

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
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
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
            global_num_experts=global_num_experts,
        )
        if layer.vllm_config.model_config is not None and layer.vllm_config.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.capture(
                    layer_id=layer.layer_id,
                    topk_ids=topk_ids,
                )

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
            random_matrix = torch.rand(topk_ids.size(0), global_num_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        final_hidden_states = moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_bias=layer.w13_bias if self.moe.has_bias else None,
            w2_bias=layer.w2_bias if self.moe.has_bias else None,
            activation=activation,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            dynamic_eplb=self.dynamic_eplb,
            log2phy=log2phy,
            mc2_mask=kwargs.get("mc2_mask"),
        )
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        return final_hidden_states


if not vllm_version_is("0.16.0"):
    # Please remove this inheritance after extending vllm, todo(wxs)
    class AscendMoERunner(DefaultMoERunner):
        """
        Default implementation of the MoE runner for executing Mixture of Experts layers.

        This class provides a comprehensive implementation for running MoE computations
        with support for:
        - Expert routing and token dispatching
        - Shared experts computation with optional parallel execution using CUDA streams
        - Data parallel (DP) chunking for large batch processing
        - Tensor model parallel and expert parallel operations
        - Various quantization methods and custom operators
        - Both monolithic and decomposed expert execution paths

        The runner handles the complete MoE forward pass including routing tokens to
        experts, executing expert computations, and combining results. It supports
        advanced features like overlapped execution of shared experts and optimized
        kernels for different parallel execution modes.

        Eventually, this class will be split up and specialized for different
        configurations, e.g. the presence or absence of shared experts, a gate, etc.
        """

        def __init__(
            self,
            layer: torch.nn.Module,
            moe_config: FusedMoEConfig,
            router: FusedMoERouter,
            routed_input_transform: torch.nn.Module | None,
            gate: torch.nn.Module | None,
            shared_experts: torch.nn.Module | None,
            quant_method: FusedMoEMethodBase,
            reduce_results: bool,
            enable_dbo: bool,
        ):
            super().__init__(
                layer,
                moe_config,
                router,
                routed_input_transform,
                gate,
                shared_experts,
                quant_method,
                reduce_results,
                enable_dbo,
            )
            if self.shared_experts is None:
                self.moe_forward = torch.ops.vllm.moe_forward
            else:
                self.moe_forward = torch.ops.vllm.moe_forward_shared

        def forward_impl(
            self,
            layer: torch.nn.Module,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            shared_input: torch.Tensor | None,
        ):
            """
            Override the default forward_impl to use Ascend-specific implementation.
            This delegates to the layer's forward_impl method which contains the
            Ascend-specific MoE computation logic.
            """
            result = layer.forward_impl(hidden_states, router_logits)
            # If the layer has shared experts, forward_impl returns a tuple (shared_out, routed_out)
            # Otherwise, it returns just routed_out
            # The torch op expects the same return type based on whether it's moe_forward or moe_forward_shared
            return result


class AscendFusedMoE(FusedMoE):
    moe_counter = -1
    gate_stream: torch.npu.Stream | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_experts = kwargs["num_experts"]
        intermediate_size = kwargs["intermediate_size"]

        AscendFusedMoE.moe_counter += 1
        self.moe_instance_id = AscendFusedMoE.moe_counter

        self._expert_map = None
        self.log2phy = None

        if self.quant_config is None:
            self.quant_method = AscendUnquantizedFusedMoEMethod(self.moe_config)
        else:
            self.quant_method = self.quant_config.get_quant_method(self, self.layer_name)

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
                dtype=vllm_config.model_config.dtype
            )

        # init moe
        eplb_config = ascend_config.eplb_config
        self.global_expert_map, self._expert_map, self.log2phy, self.global_redundant_expert_num = init_eplb_config(
            eplb_config, self.moe_instance_id, self.moe_config
        )
        self.global_num_experts = num_experts + self.global_redundant_expert_num
        self.dynamic_eplb = eplb_config.dynamic_eplb and (self.log2phy is not None)
        self.local_num_experts = (
            torch.sum(self._expert_map != -1).item() if self._expert_map is not None else self.global_num_experts
        )
        if self._expert_map is not None:
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.",
                self.ep_rank,
                self.ep_size,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self._expert_map),
            )
        if self.dynamic_eplb:
            self.moe_load = torch.zeros(self.local_num_experts, dtype=torch.int64).npu()

        self.moe_config.num_experts = self.global_num_experts
        self.moe_config.num_local_experts = self.local_num_experts
        self.moe_config.global_redundant_expert_num = self.global_redundant_expert_num

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": self.hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if self.quant_method.__class__.__name__ in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod"):
            moe_quant_params["intermediate_size_full"] = intermediate_size
        self.quant_method.create_weights(layer=self, **moe_quant_params)

        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp

        setup_moe_comm_method(self.moe_config)
        self.quant_type = self._get_quant_type()
        if not vllm_version_is("0.16.0"):
            self.runner = self._init_runner()

    if not vllm_version_is("0.16.0"):

        def _init_runner(self):
            # Storing the runner in the FusedMoE is an intermediate state, eventually
            # the runner will own the FusedMoE layer and provide the execution interface
            # for MoE ops.
            return AscendMoERunner(
                layer=self,
                moe_config=self.moe_config,
                router=self.router,
                routed_input_transform=self._routed_input_transform,
                gate=self.gate,
                shared_experts=self.shared_experts,
                quant_method=self.quant_method,
                reduce_results=self.reduce_results,
                enable_dbo=self.vllm_config.parallel_config.enable_dbo,
            )

    def _get_quant_type(self) -> QuantType:
        quant_type = QuantType.NONE
        method = getattr(self.quant_method, "quant_method", None)

        if method is not None:
            quant_type = getattr(method, "quant_type", QuantType.NONE)

        return quant_type

    def update_expert_map(self, new_expert_map):
        self._expert_map = new_expert_map

    def get_log2phy_map(self):
        return self.log2phy

    def clear_moe_load(self):
        if self.moe_load is not None:
            self.moe_load.zero_()

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """NOTE(Yizhou): This is to override the parent class method. In `mc2commimpl`,
        and `alltoallcommimpl`, we do not need to all-reduce the final outputs since
        the outputs are already aggregated across tensor parallel ranks in the
        `finalize` function. In `allgathercommimpl`, we still need to all-reduce the
        outputs since each rank only has partial outputs.
        """
        return torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(final_hidden_states)

    if not vllm_version_is("0.16.0"):

        def forward(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            self.ensure_moe_quant_config_init()
            return self.runner.forward(
                hidden_states,
                router_logits,
            )

    def forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, return_with_event: bool = False
    ) -> torch.Tensor | FusedMoEResult:
        assert self.quant_method is not None

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
            with npu_stream_switch(AscendFusedMoE.gate_stream, enabled=self.multistream_overlap_gate):
                # share_expert
                assert fc3_context.shared_experts is not None
                shared_out = fc3_context.shared_experts(hidden_states)
                # NOTE: This is exactly the opposite of `maybe_all_reduce_tensor_model_parallel`
                moe_comm_type = forward_context.moe_comm_type
                if (
                    moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
                    and not shared_expert_dp_enabled()
                ):
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
                    global_num_experts=self.global_num_experts,
                )

                if isinstance(forward_context.moe_comm_method, AllGatherCommImpl):
                    topk_weights = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(topk_weights, True, True)
                    topk_ids = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(topk_ids, True, True)

                set_flash_common3_context(topk_weights=topk_weights, topk_ids=topk_ids)

        hidden_states, router_logits, mc2_mask, context_metadata = forward_context.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=forward_context.flash_comm_v1_enabled,
            enable_shared_expert_dp=self.enable_shared_expert_dp,
            quant_type=self.quant_type,
        )

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
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            mc2_mask=mc2_mask,
        )

        if self.dynamic_eplb:
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
            self.moe_load.add_(local_load)
        routed_out = forward_context.moe_comm_method.finalize(
            hidden_states=fused_experts_results.routed_out,
            reduce_results=self.reduce_results,
            context_metadata=context_metadata,
        )

        if return_with_event:
            return FusedMoEResult(
                routed_out=routed_out,
                before_dispatch_evt=fused_experts_results.before_dispatch_evt,
                before_combine_evt=fused_experts_results.before_combine_evt,
            )
        else:
            # The vLLM FusedMoE forward_impl does not return events.
            return routed_out


class AscendSharedFusedMoE(SharedFusedMoE, AscendFusedMoE):
    def __init__(
        self,
        shared_experts: torch.nn.Module,
        gate: torch.nn.Module | None = None,
        use_overlapped: bool = True,
        routed_input_transform: torch.nn.Module | None = None,
        **kwargs,
    ):
        AscendFusedMoE.__init__(self, **kwargs)

        self._routed_input_transform = routed_input_transform
        self._shared_experts = shared_experts
        self.use_overlapped = use_overlapped
        self.shared_expert_stream = None
        ascend_config = get_ascend_config()
        self.multistream_overlap_shared_expert = (
            ascend_config.multistream_overlap_shared_expert and self._shared_experts is not None
        )
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate and self._shared_experts is not None
        if enable_sp():
            logger.info_once("Sequence parallelism is enabled, shared experts are replicated for best performance.")

        self._gate = gate
        if not vllm_version_is("0.16.0"):
            # Recreate the runner with the correct shared_experts parameter
            # The parent class created the runner before self._shared_experts was set
            self.runner = self._init_runner()

        if self.multistream_overlap_shared_expert:
            # Wrap the quant_method's process_weights_after_loading to validate that
            # splitting shared expert computation (gate_up projection + activation,
            # then down projection) yields identical results to integrated
            # computation after weight loading.
            original_process_weights = self.quant_method.process_weights_after_loading

            @wraps(original_process_weights)
            def wrapped_process_weights(*args, **kwargs):
                result = original_process_weights(*args, **kwargs)
                self._validate_shared_expert_consistency()
                return result

            self.quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore

    def _shared_experts_part1(self, hidden_states: torch.Tensor):
        shared_gate_up, _ = self._shared_experts.gate_up_proj(hidden_states)  # type: ignore
        return shared_gate_up

    def _shared_experts_part2(self, hidden_states: torch.Tensor, shared_gate_up: torch.Tensor):
        shared_act = self._shared_experts.act_fn(shared_gate_up)  # type: ignore
        shared_out, _ = self._shared_experts.down_proj(shared_act)  # type: ignore

        # Qwen3-Next specific gating mechanism
        if hasattr(self._shared_experts, "expert_gate") and self._shared_experts.expert_gate is not None:
            gate_out, _ = self._shared_experts.expert_gate(hidden_states)  # type: ignore
            shared_out = F.sigmoid(gate_out) * shared_out
        return shared_out

    def _validate_shared_expert_consistency(self):
        """Validate that split shared expert computation matches integrated
        computation."""
        test_input = (
            torch.rand(10, self.hidden_size, device="npu", dtype=self.moe_config.in_dtype) * 2 - 1
        )  # Random input for testing, scoped to [-1, 1]

        integrated_out = self._shared_experts(test_input)
        part1_out = self._shared_experts_part1(test_input)
        split_out = self._shared_experts_part2(test_input, part1_out)

        if not torch.allclose(integrated_out, split_out):
            diff = (integrated_out - split_out).abs()
            logger.error("SharedFusedMoE shared experts split computation does not match the integrated computation.")
            logger.error(f"Max absolute difference: {diff.max().item()}")
            logger.error(
                "Integrated output - sum: %s, norm: %s", integrated_out.sum().item(), integrated_out.norm().item()
            )
            logger.error("Split output - sum: %s, norm: %s", split_out.sum().item(), split_out.norm().item())
            raise ValueError(
                "SharedFusedMoE shared experts split computation does not match the integrated computation."
            )
        logger.info_once("SharedFusedMoE shared experts split computation matches the integrated computation.")

    @property
    def gate(self) -> torch.nn.Module | None:
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
        if self._shared_experts is None:
            fused_out = AscendFusedMoE.forward(
                self,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            shared_out = None
            return shared_out, fused_out
        shared_out, fused_out = AscendFusedMoE.forward(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return shared_out, fused_out

    def _forward_shared_experts(self, hidden_states: torch.Tensor, fused_moe_evts: FusedMoEEvents):
        if self._shared_experts is None:
            return None

        def maybe_wait_event(evt: torch.npu.Event | None):
            if evt is not None:
                torch.npu.current_stream().wait_event(evt)

        with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap_shared_expert):
            # Ensure the shared experts wait for hidden_states to be ready.
            torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
            # Execute the gate projection and activation concurrently with the
            # dispatch communication.
            maybe_wait_event(fused_moe_evts.before_dispatch)
            part1_out = self._shared_experts_part1(hidden_states)
            # Execute the down projection concurrently with the combine
            # communication.
            maybe_wait_event(fused_moe_evts.before_combine)
            shared_out = self._shared_experts_part2(hidden_states, part1_out)

        # Make sure the default stream waits for the shared experts stream to
        # finish.
        if self.multistream_overlap_shared_expert:
            torch.npu.current_stream().wait_stream(shared_experts_calculation_stream())

        # NOTE: This is exactly the opposite of
        # `maybe_all_reduce_tensor_model_parallel`
        forward_context = get_forward_context()
        moe_comm_type = forward_context.moe_comm_type
        if (
            moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
            and not shared_expert_dp_enabled()
        ):
            shared_out = tensor_model_parallel_all_reduce(shared_out)
        return shared_out

    def forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ):
        if self.multistream_overlap_gate:
            set_flash_common3_context(shared_experts=self._shared_experts)

        before_routed_experts = torch.npu.current_stream().record_event()
        fused_moe_results = AscendFusedMoE.forward_impl(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
            return_with_event=True,
        )
        routed_out = fused_moe_results.routed_out

        if self._shared_experts is None:
            return routed_out

        if self.multistream_overlap_gate:
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            shared_out = fc3_context.shared_out
        else:
            shared_out = self._forward_shared_experts(
                hidden_states,
                FusedMoEEvents(
                    before_routed_experts=before_routed_experts,
                    before_dispatch=fused_moe_results.before_dispatch_evt,
                    before_combine=fused_moe_results.before_combine_evt,
                ),
            )

        return shared_out, routed_out
