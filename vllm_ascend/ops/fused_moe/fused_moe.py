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
# ruff: noqa: E501
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group, tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,  # noqa: F401
    MoERunner,
)

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
from vllm_ascend.flash_common3_context import get_flash_common3_context, set_flash_common3_context
from vllm_ascend.ops.fused_moe.experts_selector import select_experts, zero_experts_compute
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult, setup_moe_comm_method
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.methods.base import get_moe_num_logical_experts
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_NZ,
    maybe_trans_nz,
    npu_stream_switch,
    shared_expert_dp_enabled,
    shared_experts_calculation_stream,
    vllm_version_is,
)

from vllm.model_executor.layers.fused_moe.layer import FusedMoE  # noqa: F401
from vllm.model_executor.layers.fused_moe.layer import MoERunner

if vllm_version_is("0.23.0"):
    from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod
else:
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    global_indices = torch.where(expert_map != -1)[0]
    local_indices = expert_map[global_indices]
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices)
    )


@dataclass
class FusedMoEResult:
    routed_out: torch.Tensor
    before_dispatch_evt: torch.npu.Event | None = None
    before_gmm2_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    swiglu_limit: float = 0.0


@dataclass
class FusedMoEEvents:
    before_routed_experts: torch.npu.Event
    after_routed_experts: torch.npu.Event | None = field(default=None)
    before_dispatch: torch.npu.Event | None = field(default=None)
    before_gmm2: torch.npu.Event | None = field(default=None)
    before_combine: torch.npu.Event | None = field(default=None)
    swiglu_limit: float = 0.0


def mock_false():
    return False


def mock_true():
    return True


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig = None, tid2eid=None):
        super().__init__(moe=moe)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        self.tid2eid = tid2eid

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

        # TODO: Current dispatch_ffn_combine fusion operator ONLY supports NZ format.
        # Therefore, we must cast weights to NZ when fusion is enabled.
        # Once the underlying dispatch_ffn_combine operator is updated to support
        # ND format (or other formats), remove this specific 'if' check and the forced
        # npu_format_cast. At that point, the operator should be able to handle weights
        # in their native format without explicit casting here.
        if get_ascend_config().enable_fused_mc2:
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ)
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
        if not vllm_version_is("0.23.0"):
            try:
                _vllm_config = get_current_vllm_config()
            except AssertionError:
                _vllm_config = None
            if (
                _vllm_config is not None
                and _vllm_config.model_config is not None
                and _vllm_config.model_config.enable_return_routed_experts
            ):
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
        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            random_matrix = torch.rand(topk_ids.size(0), num_logical_experts, device=topk_ids.device)
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        # NOTE: In the MoECommType.FUSED_MC2 branch, we wrap weights (w1, w2) into lists
        # and provide dummy scales (w1_scale, w2_scale). This is required because:
        # The underlying Ascend fused operator (e.g., dispatch_ffn_combine) expects
        # inputs in a list format.
        # TODO: Passing an empty tensor as scale for float (BF16) cases is semantically
        # incorrect. The ideal solution is to pass None. However, if the underlying
        # dispatch_ffn_combine C++ operator does not support None for the scale argument
        # (due to signature constraints), we are forced to use a placeholder empty tensor.
        # This TODO tracks the requirement to update the C++ operator to accept Optional[Tensor]
        # or None for scales in non-quantized scenarios.
        if _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2:
            w1 = [layer.w13_weight]
            w1_scale = [torch.tensor([], dtype=torch.int64)]
            w2 = [layer.w2_weight]
            w2_scale = [torch.tensor([], dtype=torch.int64)]
            w1_scale_bias = [torch.tensor([], dtype=torch.float32)]
            w2_scale_bias = [torch.tensor([], dtype=torch.float32)]
        else:
            w1 = layer.w13_weight
            w1_scale = None
            w2 = layer.w2_weight
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
            )
        )
        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result
        return final_hidden_states


if vllm_version_is("0.23.0"):
    from vllm_ascend.ops.fused_moe.fused_moe_0_23_0 import AscendFusedMoE, AscendMoERunner

    AscendFusedMoE.__module__ = __name__
    AscendMoERunner.__module__ = __name__

else:

    class AscendMoERunner(MoERunner):  # type: ignore[no-redef]
        moe_counter = -1
        gate_stream: torch.npu.Stream | None = None

        def __init__(
            self,
            layer_name,
            moe_config,
            router,
            routed_experts,
            enable_dbo=False,
            gate=None,
            shared_experts=None,
            shared_expert_gate=None,
            routed_input_transform=None,
            routed_output_transform=None,
            routed_scaling_factor=1,
            tid2eid=None,
            n_shared_experts: int = 0,
        ):
            super().__init__(
                layer_name,
                moe_config,
                router,
                routed_experts,
                enable_dbo,
                gate,
                shared_experts,
                shared_expert_gate,
                routed_input_transform,
                routed_output_transform,
                routed_scaling_factor,
            )
            self.top_k = moe_config.experts_per_token
            self._gate = gate
            self.hidden_size = moe_config.hidden_dim

            # Routing params — all stored on routed_experts by the factory function.
            self.use_grouped_topk = routed_experts.use_grouped_topk
            self.renormalize = routed_experts.renormalize
            self.topk_group = routed_experts.topk_group
            self.num_expert_group = routed_experts.num_expert_group
            self.custom_routing_function = routed_experts.custom_routing_function
            self.scoring_func = routed_experts.scoring_func
            self._original_routed_scaling_factor = routed_experts.routed_scaling_factor
            self.e_score_correction_bias = routed_experts.e_score_correction_bias
            self.apply_router_weight_on_input = routed_experts.apply_router_weight_on_input

            # Ascend-specific: not stored on RoutedExperts, passed via runner_args.
            self.tid2eid = tid2eid

            # Replace quant_method on routed_experts with the Ascend version.
            # Must NOT set self.quant_method (instance attr) — vllm loader scans all
            # nn.Module children for quant_method and would call process_weights_after_loading
            # on the runner with wrong expectations. Set on routed_experts so that
            # self._quant_method (the MoERunner property) returns the Ascend version.
            if routed_experts.quant_config is None:
                routed_experts.quant_method = AscendUnquantizedFusedMoEMethod(self.moe_config, tid2eid=self.tid2eid)
            else:
                routed_experts.quant_method = routed_experts.quant_config.get_quant_method(
                    routed_experts, self.layer_name, tid2eid=self.tid2eid
                )

            self.quant_type = self._get_quant_type()
            # Can be removed after vllm fixes the issue.
            if self._needs_routed_expert_parameter_aliases():
                self._schedule_routed_expert_parameter_aliases()

            self.moe_config.tp_group = get_tp_group()
            self.moe_config.dp_group = get_dp_group()
            if self.moe_config.ep_size > 1:
                self.moe_config.ep_group = get_ep_group()
                self.moe_config.mc2_group = get_mc2_group()

            ascend_config = get_ascend_config()
            self._shared_experts = shared_experts
            has_shared_experts = self._shared_experts is not None

            self.shared_multistream_overlap_gate = ascend_config.multistream_overlap_gate and has_shared_experts
            self.enable_npugraph_ex_static_kernel = ascend_config.ascend_compilation_config.enable_static_kernel

            # flashcommon3 gate stream
            self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
            if self.multistream_overlap_gate and AscendMoERunner.gate_stream is None:
                AscendMoERunner.gate_stream = torch.npu.Stream()

            vllm_config = get_current_vllm_config()

            if (
                self.custom_routing_function is None
                and self.e_score_correction_bias is not None
                and not vllm_config.model_config.is_deepseek_mla
            ):
                self.e_score_correction_bias.data = self.e_score_correction_bias.data.to(
                    dtype=vllm_config.model_config.dtype
                )

            self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
            self.multistream_overlap_shared_expert = (
                ascend_config.multistream_overlap_shared_expert and shared_experts is not None
            )
            mix_placement = getattr(ascend_config, "mix_placement", False)

            # EPLB initialization (Ascend-specific; mirrors old AscendFusedMoE logic).
            AscendMoERunner.moe_counter += 1
            self.moe_instance_id = AscendMoERunner.moe_counter

            eplb_config = ascend_config.eplb_config

            if mix_placement:
                moe_config.num_experts += n_shared_experts

            (
                self.global_expert_map,
                self._expert_map,
                self.log2phy,
                self.global_redundant_expert_num,
            ) = init_eplb_config(
                eplb_config,
                AscendMoERunner.moe_counter,
                moe_config,
                mix_placement,
                n_shared_experts,
                tp_size=vllm_config.parallel_config.tensor_parallel_size,
            )

            moe_config.global_redundant_expert_num = self.global_redundant_expert_num
            local_num_experts = (moe_config.num_experts + self.global_redundant_expert_num) // moe_config.ep_size
            moe_config.num_local_experts = local_num_experts
            routed_experts.expert_map_manager._local_num_experts = local_num_experts
            routed_experts.expert_map_manager._expert_map = self._expert_map

            self.dynamic_eplb = eplb_config.dynamic_eplb and (self.log2phy is not None)
            self.multi_stage = False
            self.moe_load = torch.zeros(local_num_experts, dtype=torch.int64).npu()
            if self.dynamic_eplb and eplb_config.expert_heat_collection_interval > 1:
                self.multi_stage = True
                self.load_counter = torch.tensor(0, dtype=torch.int32, device="npu")
                self.num_iter = eplb_config.expert_heat_collection_interval
                self.moe_load = torch.zeros((self.num_iter, local_num_experts), dtype=torch.int32, device="npu")

            setup_moe_comm_method(self.moe_config)
            if self.multistream_overlap_shared_expert:
                # Wrap the quant_method's process_weights_after_loading to validate that
                # splitting shared expert computation (gate_up projection + activation,
                # then down projection) yields identical results to integrated
                # computation after weight loading.
                original_process_weights = self._quant_method.process_weights_after_loading

                @wraps(original_process_weights)
                def wrapped_process_weights(*args, **kwargs):
                    result = original_process_weights(*args, **kwargs)
                    self._validate_shared_expert_consistency()
                    return result

                self._quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore

            # Register this MoE layer with EPLB for PP compatibility.
            # PPMissingLayer (nn.Identity) never calls AscendFusedMoE.__init__,
            # so only real MoE layers on this rank are registered.
            VllmEplbAdaptor.register_layer(self)

        def _validate_shared_expert_consistency(self):
            """Validate that split shared expert computation matches integrated computation."""
            test_input = (
                torch.rand(10, self.hidden_size, device="npu", dtype=self.moe_config.in_dtype) * 2 - 1
            )  # Random input for testing, scoped to [-1, 1]

            assert self._shared_experts is not None
            integrated_out = self._shared_experts(test_input)
            part1_out = self._shared_experts_part1(test_input)
            split_out = self._shared_experts_part2(test_input, part1_out)

            if not torch.allclose(integrated_out, split_out):
                diff = (integrated_out - split_out).abs()
                logger.error(
                    "[fused_moe/layer] Shared expert split computation validation failed."
                    " The split-path computation does not match the integrated-path result."
                    " max_abs_diff=%s, integrated_sum=%s, integrated_norm=%s,"
                    " split_sum=%s, split_norm=%s, hidden_size=%s, dtype=%s.",
                    diff.max().item(),
                    integrated_out.sum().item(),
                    integrated_out.norm().item(),
                    split_out.sum().item(),
                    split_out.norm().item(),
                    self.hidden_size,
                    self.moe_config.in_dtype,
                )
                raise ValueError("FusedMoE shared experts split computation does not match the integrated computation.")
            logger.info_once(
                "[fused_moe/layer] Shared expert split computation validation passed."
                " Integrated and split-path results are consistent."
            )

        def _shared_experts_part1(self, hidden_states: torch.Tensor):
            shared_gate_up, _ = self._shared_experts.gate_up_proj(hidden_states)  # type: ignore
            return shared_gate_up

        def _shared_experts_part2(self, hidden_states: torch.Tensor, shared_gate_up: torch.Tensor):
            shared_act = self._shared_experts.act_fn(shared_gate_up)  # type: ignore
            shared_out, _ = self._shared_experts.down_proj(shared_act)  # type: ignore

            # Qwen3-Next specific gating mechanism
            assert self._shared_experts is not None
            if hasattr(self._shared_experts, "expert_gate") and self._shared_experts.expert_gate is not None:
                gate_out, _ = self._shared_experts.expert_gate(hidden_states)  # type: ignore
                shared_out = F.sigmoid(gate_out) * shared_out
            return shared_out

        def _get_quant_type(self) -> QuantType:
            quant_type = QuantType.NONE
            method = getattr(self._quant_method, "quant_method", None)

            if method is not None:
                quant_type = getattr(method, "quant_type", QuantType.NONE)

            return quant_type

        def _schedule_routed_expert_parameter_aliases(self) -> None:
            alias_names = [name for name, _ in self.routed_experts.named_parameters(recurse=False)]

            original_process_weights = self._quant_method.process_weights_after_loading

            @wraps(original_process_weights)
            def wrapped_process_weights(layer, *args, **kwargs):
                result = original_process_weights(layer, *args, **kwargs)
                for name in alias_names:
                    param = getattr(self.routed_experts, name)
                    alias_param = torch.nn.Parameter(param.data, requires_grad=param.requires_grad)
                    alias_param.__dict__.update(param.__dict__)
                    self.register_parameter(name, alias_param)
                return result

            self._quant_method.process_weights_after_loading = wrapped_process_weights  # type: ignore[method-assign]

        def _needs_routed_expert_parameter_aliases(self) -> bool:
            # test_gpt_oss_distributed_tp2
            # test_qwen3_moe_routing_replay[Qwen/Qwen3.5-35B-A3B]
            # test_multimodal_reasoning_pp_full_decode_only
            vllm_config = get_current_vllm_config()
            hf_config = getattr(vllm_config.model_config, "hf_config", None)
            model_type = getattr(hf_config, "model_type", None)
            architectures = getattr(hf_config, "architectures", ()) or ()
            if model_type in {"qwen3_5_mtp", "step3p5_mtp"} or set(architectures) & {
                "Qwen3_5MoeMTP",
                "Step3p5MTP",
            }:
                return False
            if self.layer_name.startswith("mtp.") or ".mtp." in self.layer_name:
                return False
            return model_type in {
                "aria",
                "aria_text",
                "gpt_oss",
                "qwen3_5_moe",
                "qwen3_vl_moe",
                "qwen3_vl_moe_text",
                "step3_text",
                "step3_vl",
                "step3p5",
            }

        @property
        def is_internal_router(self) -> bool:
            gate = self.gate
            return gate is not None and hasattr(gate, "weight_fp32")

        @property
        def use_dp_chunking(self) -> bool:
            """Ascend uses its own forward_impl path, not the FlashInfer Cutlass
            chunked path. Always return False to stay on forward_impl."""
            return False

        @property
        def _fused_output_is_reduced(self) -> bool:
            # For MC2/ALLTOALL/FUSED_MC2 comm types, finalize() already includes
            # TP all-reduce for the routed output, and _forward_shared_experts
            # handles it for the shared output. Signal this to the upstream
            # MoERunner.forward() so _maybe_reduce_final_output does not apply a
            # second TP all-reduce (which would double-count the contributions).
            moe_comm_type = _EXTRA_CTX.moe_comm_type
            return moe_comm_type in {
                MoECommType.ALLTOALL,
                MoECommType.MC2,
                MoECommType.FUSED_MC2,
            } or (moe_comm_type == MoECommType.ALLGATHER and _EXTRA_CTX.flash_comm_v1_enabled)

        def _maybe_reduce_shared_expert_output(
            self,
            shared_output: torch.Tensor | None,
        ) -> torch.Tensor | None:
            # _forward_shared_experts already handles shared expert TP all-reduce
            # for MC2/ALLTOALL/FUSED_MC2. For AllGather the reduction is done
            # via _maybe_reduce_final_output on the combined (shared + routed)
            # output. Skip any additional reduction here.
            return shared_output

        def _maybe_reduce_final_output(
            self,
            states: torch.Tensor,
            trunc_size: int,
        ) -> torch.Tensor:
            states = torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(states)
            return states[..., :trunc_size]

        def no_shared_forward_impl(  # type: ignore[override]
            self, hidden_states: torch.Tensor, router_logits: torch.Tensor, return_with_event: bool = False
        ) -> torch.Tensor | FusedMoEResult:
            forward_context = get_forward_context()
            # When static kernels are enabled, the forward pass runs twice (compilation + capture),
            # causing moe_layer_index to overflow. Wrap the index to prevent out-of-bounds errors.
            if self.enable_npugraph_ex_static_kernel and forward_context.all_moe_layers:
                moe_layer_index = forward_context.moe_layer_index % (len(forward_context.all_moe_layers))
                forward_context.moe_layer_index = moe_layer_index

            # Load balancing for token distribution among experts in dummy_run
            # TODO: The community only considers load balancing when DP > 1.
            # This approach may overlook some extreme scenarios.
            enable_force_load_balance = _EXTRA_CTX.in_profile_run
            forward_context = get_forward_context()
            if self.multistream_overlap_gate:
                fc3_context = get_flash_common3_context()
                assert fc3_context is not None
                assert AscendMoERunner.gate_stream is not None
                AscendMoERunner.gate_stream.wait_stream(torch.npu.current_stream())
                with npu_stream_switch(AscendMoERunner.gate_stream, enabled=self.multistream_overlap_gate):
                    # share_expert
                    assert fc3_context.shared_experts is not None
                    shared_out = fc3_context.shared_experts(hidden_states)
                    # NOTE: This is exactly the opposite of `maybe_all_reduce_tensor_model_parallel`
                    moe_comm_type = _EXTRA_CTX.moe_comm_type
                    if (
                        moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
                        and not shared_expert_dp_enabled()
                    ):
                        shared_out = tensor_model_parallel_all_reduce(shared_out)
                    set_flash_common3_context(shared_out=shared_out)
                    input_ids = getattr(get_forward_context(), "input_ids", None)

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
                        routed_scaling_factor=self._original_routed_scaling_factor,
                        e_score_correction_bias=self.e_score_correction_bias,
                        num_experts=self.moe_config.num_experts,
                        input_ids=input_ids,
                        tid2eid=self.tid2eid,
                    )

                    if isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl):
                        topk_weights = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(topk_weights, True, True)
                        topk_ids = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(topk_ids, True, True)

                    set_flash_common3_context(topk_weights=topk_weights, topk_ids=topk_ids)

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

            # Make sure the default stream waits for the gate stream to finish.
            if self.multistream_overlap_gate:
                assert AscendMoERunner.gate_stream is not None
                torch.npu.current_stream().wait_stream(AscendMoERunner.gate_stream)

            # Matrix multiply.
            # apply() expects a RoutedExperts-like layer for weight access
            # (w13_weight, w2_weight, swiglu_limit, etc.). Pass routed_experts,
            # not self; the routing params come through the other kwargs.
            fused_experts_results: FusedExpertsResult = self._quant_method.apply(
                layer=self.routed_experts,
                x=hidden_states,
                router_logits=router_logits,
                pertoken_scale=pertoken_scale,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                num_experts=self.moe_config.num_experts,
                expert_map=self._expert_map,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self._original_routed_scaling_factor,
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
                if self.multi_stage:
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

            if return_with_event:
                return FusedMoEResult(
                    routed_out=routed_out,
                    before_dispatch_evt=fused_experts_results.before_dispatch_evt,
                    before_gmm2_evt=fused_experts_results.before_gmm2_evt,
                    before_combine_evt=fused_experts_results.before_combine_evt,
                    swiglu_limit=fused_experts_results.swiglu_limit,
                )
            else:
                # The vLLM FusedMoE forward_impl does not return events.
                return routed_out

        def _forward_shared_experts(self, hidden_states: torch.Tensor, fused_moe_evts: FusedMoEEvents):
            if self._shared_experts is None:
                return None

            def maybe_wait_event(evt: torch.npu.Event | None):
                if evt is not None:
                    torch.npu.current_stream().wait_event(evt)

            with npu_stream_switch(shared_experts_calculation_stream(), enabled=self.multistream_overlap_shared_expert):
                # Only used for int quantization
                has_quantized_shared = hasattr(self._shared_experts.gate_up_proj, "weight_scale") and hasattr(
                    self._shared_experts.down_proj, "weight_scale"
                )
                if has_quantized_shared and self.quant_type in (QuantType.W8A8, QuantType.W4A8):
                    original_dtype = hidden_states.dtype
                    # Execute dynamic quant concurrently with MoE gate.
                    torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                    quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
                    # Execute the gate projection and activation concurrently with the
                    # dispatch communication.
                    maybe_wait_event(fused_moe_evts.after_routed_experts)
                    hidden_states = torch_npu.npu_quant_matmul(
                        quantized_x,
                        self._shared_experts.gate_up_proj.weight,
                        self._shared_experts.gate_up_proj.weight_scale,
                        pertoken_scale=None,
                        bias=None,
                        output_dtype=torch.int32,
                    )
                    # Execute activation concurrently with gmm2.

                    maybe_wait_event(fused_moe_evts.before_gmm2)
                    quantized_x, swiglu_out_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
                        x=hidden_states,
                        weight_scale=self._shared_experts.gate_up_proj.weight_scale_fp32,
                        activation_scale=pertoken_scale,
                        bias=None,
                        quant_scale=None,
                        quant_offset=None,
                        group_index=None,
                        activate_left=True,
                        quant_mode=1,
                        swiglu_mode=1,
                        clamp_limit=fused_moe_evts.swiglu_limit,
                    )
                    # Execute the down projection concurrently with the combine
                    # communication.
                    maybe_wait_event(fused_moe_evts.before_combine)
                    shared_out = torch_npu.npu_quant_matmul(
                        quantized_x,
                        self._shared_experts.down_proj.weight,
                        self._shared_experts.down_proj.weight_scale,
                        pertoken_scale=swiglu_out_scale,
                        bias=None,
                        output_dtype=original_dtype,
                    )
                elif has_quantized_shared and self.quant_type == QuantType.W4A8MXFP:
                    original_dtype = hidden_states.dtype
                    # Execute dynamic quant concurrently with MoE gate.
                    torch.npu.current_stream().wait_event(fused_moe_evts.before_routed_experts)
                    quantized_x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                        hidden_states, dst_type=torch.float8_e4m3fn
                    )
                    # Execute the gate projection and activation concurrently with the
                    # dispatch communication.
                    maybe_wait_event(fused_moe_evts.before_dispatch)
                    hidden_states = self._shared_experts.gate_up_proj((quantized_x, pertoken_scale))[0]
                    # Execute activation concurrently with gmm2.
                    maybe_wait_event(fused_moe_evts.before_gmm2)
                    quantized_x, swiglu_out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
                        hidden_states,
                        topk_weight=None,
                        group_index=None,
                        dst_type=torch.float8_e4m3fn,
                        quant_mode=2,
                        clamp_value=fused_moe_evts.swiglu_limit,
                    )
                    # Execute the down projection concurrently with the combine
                    # communication.
                    maybe_wait_event(fused_moe_evts.before_combine)
                    shared_out = self._shared_experts.down_proj((quantized_x, swiglu_out_scale))[0]
                else:
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
            moe_comm_type = _EXTRA_CTX.moe_comm_type
            if (
                moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
                and not shared_expert_dp_enabled()
            ):
                shared_out = tensor_model_parallel_all_reduce(shared_out)
            return shared_out

        def shared_forward_impl(  # type: ignore[override]
            self, hidden_states: torch.Tensor, router_logits: torch.Tensor
        ):
            if self.shared_multistream_overlap_gate:
                set_flash_common3_context(shared_experts=self._shared_experts)

            if self.is_internal_router:
                gate = self.gate
                assert gate is not None
                # NOTE(Angazenn): To make this cast explicitly, the hbm usage might
                # increase with extra hidden states. We also assume that all gate
                # linear is unquantized so that we the weight is pre-casted in
                # process_weights_after_loading of AscendUnquantizedLinearMethod.
                hidden_states_fp32 = hidden_states.float()
                before_routed_experts = torch.npu.current_stream().record_event()
                router_logits = F.linear(hidden_states_fp32, gate.weight_fp32)
                after_routed_experts = torch.npu.current_stream().record_event()
            else:
                before_routed_experts = torch.npu.current_stream().record_event()
                after_routed_experts = None

            fused_moe_results = self.no_shared_forward_impl(
                hidden_states,
                router_logits,
                return_with_event=True,
            )
            routed_out = fused_moe_results.routed_out

            if self._shared_experts is None:
                return routed_out

            if self.shared_multistream_overlap_gate:
                fc3_context = get_flash_common3_context()
                assert fc3_context is not None
                shared_out = fc3_context.shared_out
            else:
                shared_out = self._forward_shared_experts(
                    hidden_states,
                    FusedMoEEvents(
                        after_routed_experts=after_routed_experts,
                        before_routed_experts=before_routed_experts,
                        before_dispatch=fused_moe_results.before_dispatch_evt,
                        before_gmm2=fused_moe_results.before_gmm2_evt,
                        before_combine=fused_moe_results.before_combine_evt,
                        swiglu_limit=fused_moe_results.swiglu_limit,
                    ),
                )
            return shared_out, routed_out

        def _forward_impl(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            shared_experts_input: torch.Tensor | None,
            input_ids: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            with self._sequence_parallel_context():
                if self.shared_experts is None:
                    return self.no_shared_forward_impl(hidden_states, router_logits)
                else:
                    return self.shared_forward_impl(hidden_states, router_logits)
