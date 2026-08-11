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
from collections.abc import Iterable
from copy import copy
from types import SimpleNamespace

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed.utils import is_weak_contiguous
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoERouter, RoutedExperts, SharedExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
from vllm.model_executor.utils import replace_parameter

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_utils import init_eplb_config
from vllm_ascend.lora.fused_moe import sync_lora_context
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.eplb import record_local_expert_load
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult
from vllm_ascend.ops.fused_moe.moe_utils import get_moe_num_logical_experts
from vllm_ascend.ops.fused_moe.shared_experts import FusedMoEEvents
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, maybe_trans_nz


class AscendUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig = None, tid2eid=None):
        super().__init__(moe=moe)
        vllm_config = get_current_vllm_config()
        self.dynamic_eplb = False if vllm_config.use_v2_model_runner else get_ascend_config().eplb_config.dynamic_eplb
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

    @staticmethod
    def get_eplb_weight_views(layer) -> list[torch.Tensor]:
        weights = [layer.w13_weight, layer.w2_weight]
        if layer.w13_bias is not None:
            weights.append(layer.w13_bias)
        if layer.w2_bias is not None:
            weights.append(layer.w2_bias)
        return weights

    def process_weights_after_loading(self, layer):
        super(UnquantizedFusedMoEMethod, self).process_weights_after_loading(layer)

        # Keep expert-aware loaders attached for later online weight updates.
        w13_data = self._maybe_pad_weight(layer.w13_weight.data).transpose(1, 2).contiguous()
        replace_parameter(layer, "w13_weight", w13_data)

        w2_data = self._maybe_pad_weight(layer.w2_weight.data).transpose(1, 2).contiguous()
        replace_parameter(layer, "w2_weight", w2_data)

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
        layer: "AscendRoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        activation = getattr(layer, "activation", "silu")
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
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=activation,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                lora_context=getattr(layer, "_ascend_moe_lora_context", None),
            )
        )
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


class EplbExpertTensorList(list[torch.Tensor]):
    """Per-expert tensors exposed through the upstream EPLB weight contract."""

    @property
    def shape(self) -> torch.Size:
        return torch.Size((len(self), *self[0].shape))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is torch.empty_like:
            source = args[0]
            return cls(torch.empty_like(tensor, **(kwargs or {})) for tensor in source)
        return NotImplemented


class AscendRoutedExperts(RoutedExperts):  # type: ignore[no-redef]
    """Ascend-owned routed expert container.

    This keeps Ascend quant-method selection with the expert weight owner,
    matching upstream vLLM's RoutedExperts responsibility split.
    """

    moe_counter = -1

    def __init__(
        self,
        *args,
        tid2eid=None,
        n_shared_experts: int = 0,
        **kwargs,
    ):
        object.__setattr__(self, "tid2eid", tid2eid)
        super().__init__(*args, **kwargs)
        if self.quant_config is None:
            # Preserve the pre-refactor BF16 lifecycle: let upstream create
            # weights first, then install the Ascend execution method.
            self._replace_quant_method(
                AscendUnquantizedFusedMoEMethod(
                    self.moe_config,
                    tid2eid=self.tid2eid,
                )
            )
        self.router: FusedMoERouter | None = None
        ascend_config = get_ascend_config()
        vllm_config = get_current_vllm_config()
        self.n_shared_experts = n_shared_experts
        self.mix_placement = getattr(ascend_config, "mix_placement", False)
        self.enable_npugraph_ex_static_kernel = ascend_config.ascend_compilation_config.enable_static_kernel
        self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp
        self._use_v2_model_runner = bool(vllm_config.use_v2_model_runner)
        self.dynamic_eplb = False
        self.multi_stage = False
        self.load_counter = None
        self.num_iter = None
        self.moe_load: torch.Tensor | None = None
        self.ascend_expert_map: torch.Tensor | None = None
        self.log2phy: torch.Tensor | None = None
        self.global_redundant_expert_num: int = 0
        self.ascend_pertoken_scale: torch.Tensor | None = None
        self.ascend_mc2_mask: torch.Tensor | None = None
        if not self._use_v2_model_runner:
            self.init_eplb(n_shared_experts)
        self.return_with_event = False

        if (
            self.custom_routing_function is None
            and self.e_score_correction_bias is not None
            and not vllm_config.model_config.is_deepseek_mla
        ):
            self.e_score_correction_bias.data = self.e_score_correction_bias.data.to(
                dtype=vllm_config.model_config.dtype
            )

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        try:
            get_weight_views = self.quant_method.get_eplb_weight_views
        except AttributeError as exc:
            raise NotImplementedError(
                f"{self.quant_method.__class__.__name__} must implement get_eplb_weight_views() for Ascend EPLB."
            ) from exc
        weights = list(get_weight_views(self))
        if not weights:
            raise NotImplementedError(f"EPLB weight views are not defined for {self.quant_method.__class__.__name__}.")
        flattened_weights = []
        for weight in weights:
            if isinstance(weight, (list, tuple)):
                if len(weight) != self.local_num_experts:
                    raise ValueError(
                        "Every EPLB expert tensor list must contain "
                        f"local_num_experts ({self.local_num_experts}) tensors, got {len(weight)}."
                    )
                if not all(is_weak_contiguous(expert_weight) for expert_weight in weight):
                    raise ValueError("Every tensor in an Ascend EPLB expert tensor list must be weakly contiguous.")
                flattened_weights.append(EplbExpertTensorList(weight))
                continue
            if weight.shape[0] != self.local_num_experts:
                raise ValueError(
                    "The first dimension of every EPLB weight view must equal "
                    f"local_num_experts ({self.local_num_experts}), got {tuple(weight.shape)}."
                )
            if not is_weak_contiguous(weight):
                raise ValueError("Every Ascend EPLB weight view must be weakly contiguous.")
            try:
                flattened_weights.append(weight.view(self.local_num_experts, -1))
            except RuntimeError as exc:
                raise ValueError("Every Ascend EPLB expert row must be flattenable without a copy.") from exc
        return flattened_weights

    def init_eplb(self, n_shared_experts):
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
            n_shared_experts if self.mix_placement else 0
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
            self.mix_placement,
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

        # Level-2 sleep discards NPU tensors that are not parameters/buffers.
        # Register Ascend runtime EPLB NPU state as named buffers for wake restore.
        # ascend_expert_map stays a plain CPU attribute and does not need promotion.
        self._promote_attr_to_buffer("log2phy")
        if self.dynamic_eplb:
            self._promote_attr_to_buffer("moe_load")
            if self.multi_stage:
                self._promote_attr_to_buffer("load_counter")

        # Register this MoE layer with EPLB for PP compatibility.
        # PPMissingLayer (nn.Identity) never calls AscendFusedMoE.__init__,
        # so only real MoE layers on this rank are registered.
        VllmEplbAdaptor.register_layer(self)

    def _promote_attr_to_buffer(self, name: str) -> None:
        """Move an existing tensor attribute onto a Level-2 restorable named buffer."""
        tensor = getattr(self, name, None)
        if tensor is None:
            return
        delattr(self, name)
        self.register_buffer(name, tensor)

    def _get_quant_method(self, prefix, quant_config, moe_config):
        if quant_config is None:
            return super()._get_quant_method(prefix, quant_config, moe_config)
        return quant_config.get_quant_method(self, prefix, tid2eid=self.tid2eid)

    def get_eplb_parameter(self, name: str):
        """Return an expert parameter from the refactored weight owner."""
        return getattr(self, name)

    @property
    def ascend_expert_map(self) -> torch.Tensor | None:
        """Return the global-to-local map used by Ascend MoE execution."""
        if getattr(self, "_use_v2_model_runner", False):
            return self.expert_map
        return getattr(self, "_ascend_expert_map", None)

    @ascend_expert_map.setter
    def ascend_expert_map(self, expert_map: torch.Tensor | None) -> None:
        object.__setattr__(self, "_ascend_expert_map", expert_map)

    def update_expert_map(self, new_expert_map: torch.Tensor | None = None) -> None:
        """Update the upstream map or preserve the legacy Ascend update API."""
        if new_expert_map is None:
            super().update_expert_map()
            return
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

    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_force_load_balance: bool,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.router is None:
            raise RuntimeError("AscendRoutedExperts requires a router for expert selection.")
        topk_weights, topk_ids = self.router._select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            input_ids=input_ids,
        )
        if self.log2phy is not None:
            topk_ids = self.log2phy[topk_ids]

        try:
            _vllm_config = get_current_vllm_config()

            model_config = None if _vllm_config is None else _vllm_config.model_config
            if model_config is not None and model_config.enable_return_routed_experts:
                capturer = getattr(self, "_ascend_routed_experts_capturer", None)
                if capturer is not None:
                    capturer.capture(layer_id=self.layer_id, topk_ids=topk_ids)
        except Exception:
            pass

        num_shared_experts = self.n_shared_experts
        if num_shared_experts is None:
            num_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            self,
            self.moe_config.num_experts,
            global_redundant_expert_num=self.global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )

        if getattr(self, "mix_placement", False):
            batch_size = topk_ids.shape[0]
            shared_expert_ids = torch.arange(
                num_logical_experts,
                num_logical_experts + num_shared_experts,
                dtype=topk_ids.dtype,
                device=topk_ids.device,
            ).repeat(batch_size, 1)
            shared_expert_weights = torch.ones(
                topk_weights.shape[0],
                num_shared_experts,
                dtype=topk_weights.dtype,
                device=topk_weights.device,
            )
            topk_ids = torch.cat([topk_ids, shared_expert_ids], dim=1)
            topk_weights = torch.cat([topk_weights, shared_expert_weights], dim=1)

        topk_weights = topk_weights.to(hidden_states.dtype)
        # This is a naive implementation for experts load balance so as to
        # avoid accumulating too much tokens on a single rank. It is only
        # activated when doing profile runs.
        if enable_force_load_balance:
            random_matrix = torch.rand(
                topk_ids.size(0),
                num_logical_experts,
                device=topk_ids.device,
            )
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        return topk_weights, topk_ids

    def forward_impl(
        self,
        *,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
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
        if self.router is None:
            raise RuntimeError("AscendRoutedExperts requires a router for expert selection.")
        topk_weights, topk_ids = self._select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            enable_force_load_balance=enable_force_load_balance,
            input_ids=input_ids,
        )
        self.ascend_pertoken_scale = pertoken_scale
        self.ascend_mc2_mask = mc2_mask
        try:
            fused_experts_results: FusedExpertsResult = self.quant_method.apply(
                layer=self,
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts=None,
                shared_experts_input=None,
            )
        finally:
            self.ascend_pertoken_scale = None
            self.ascend_mc2_mask = None

        if self._use_v2_model_runner and self.router.eplb_state is not None:
            expert_tokens = fused_experts_results.expert_tokens
            group_list_type = fused_experts_results.group_list_type
            assert expert_tokens is not None and group_list_type is not None, (
                "expert_tokens and group_list_type must be returned when Model Runner V2 EPLB is enabled."
            )
            eplb_state = self.router.eplb_state
            assert eplb_state.expert_load_view is not None
            record_local_expert_load(
                expert_tokens=expert_tokens,
                group_list_type=group_list_type,
                expert_load_view=eplb_state.expert_load_view,
                ep_rank=self.moe_config.ep_rank,
                ep_size=self.moe_config.ep_size,
            )
        elif self.dynamic_eplb and _EXTRA_CTX.eplb_heat_collection_status:
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

        if self.return_with_event:
            return routed_out, FusedMoEEvents(
                before_routed_experts=None,
                after_routed_experts=None,
                before_dispatch=fused_experts_results.before_dispatch_evt,
                before_gmm2=fused_experts_results.before_gmm2_evt,
                before_combine=fused_experts_results.before_combine_evt,
            )

        # The vLLM FusedMoE forward_impl does not return events.
        return routed_out
