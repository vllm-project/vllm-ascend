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
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEMlpComputeInput,
    MoEPrepareOutput,
    build_mlp_compute_input,
    build_token_dispatch_input,
)
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalize,
    PrepareAndFinalizeWithAll2All,
    PrepareAndFinalizeWithAllGather,
    PrepareAndFinalizeWithMC2,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    MoETokenDispatcher,
    TokenDispatcherWithAll2AllV,
    TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2,
)
from vllm_ascend.quantization.quant_type import QuantType

_MoECommMethods: dict[MoECommType | None, MoECommMethod] = {}

_CANN_ACL_INT8 = 258
_CANN_ACL_INT4 = 285
_CANN_TORCH_FLOAT8_E4M3FN = 24

_CANN_MEGA_MOE_QUANT_MODE_INT8 = 2
_CANN_MEGA_MOE_QUANT_MODE_MX = 4

_DISPATCH_FFN_COMBINE_MODE = 1
_CANN_MEGA_MOE_FUSED_MC2_MODE = 2
_CANN_MEGA_MOE_MODULE_NAME = "cann_ops_transformer.ops.mega_moe"


def _as_tensor_list(value: torch.Tensor | list[torch.Tensor], name: str) -> list[torch.Tensor]:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} cannot be an empty list for CANN MegaMoe.")
        return value
    return [value]


def _as_optional_tensor_list(value: torch.Tensor | list[torch.Tensor] | None) -> list[torch.Tensor] | None:
    if value is None:
        return None
    return _as_tensor_list(value, "optional CANN MegaMoe tensor list")


def _pick_mega_moe_bias(
    scale_bias: torch.Tensor | list[torch.Tensor] | None,
    fallback_bias: torch.Tensor | list[torch.Tensor] | None,
) -> list[torch.Tensor] | None:
    """Pick the correct per-expert bias list to forward to CANN MegaMoe.

    W4A8 emits a real per-expert ``w*_scale_bias`` list (the 8 * sum
    correction term computed in AscendW4A8DynamicFusedMoEMethod) that must
    be passed to MegaMoe as ``l*_bias``. The older ``dispatch_ffn_combine``
    path (``enable_fused_mc2 == 1``) historically stuffed
    ``w*_scale_bias`` with empty placeholder tensors so the C++ op's
    positional signature was satisfied — MegaMoe does NOT tolerate that
    placeholder, so empty entries are treated as "no bias" here.

    Order of preference:
        1. ``scale_bias`` if it is a non-empty list of real (numel > 0) tensors.
        2. ``fallback_bias`` (the model's ``w*_bias``, rarely set for MoE).
        3. ``None``.
    """
    if scale_bias is not None:
        items = scale_bias if isinstance(scale_bias, list) else [scale_bias]
        if items and not all(getattr(t, "numel", lambda: 0)() == 0 for t in items):
            return list(items)
    return _as_optional_tensor_list(fallback_bias)


def _infer_intermediate_hidden(weight1: torch.Tensor, weight2: torch.Tensor, hidden: int) -> int:
    if weight2.dim() >= 2:
        if weight2.shape[-1] == hidden:
            return int(weight2.shape[-2])
        if weight2.shape[-2] == hidden:
            return int(weight2.shape[-1])
    if weight1.dim() < 2:
        return hidden
    if weight1.shape[-1] == hidden:
        combined_intermediate = int(weight1.shape[-2])
    elif weight1.shape[-2] == hidden:
        combined_intermediate = int(weight1.shape[-1])
    else:
        combined_intermediate = int(weight1.shape[-2])
    if combined_intermediate > 1 and combined_intermediate % 2 == 0:
        return combined_intermediate // 2
    return combined_intermediate


def _normalize_cann_activation(activation) -> str:
    activation_value = getattr(activation, "value", activation)
    activation_value = "swiglu" if activation_value is None else str(activation_value)
    return "swiglu" if activation_value in ("silu", "swiglu") else activation_value


def _get_cann_mega_moe_quant_settings(quant_type: QuantType) -> tuple[int, int | None, int | None]:
    # CANN 9.1 docs name this argument dispatch_quant_out_dtype and show
    # torch.int8, while the installed Python wrapper still accepts ACL dtype
    # enum ints. Keep the enum values until the wrapper schema changes.
    if quant_type == QuantType.W8A8:
        return (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT8)
    if quant_type == QuantType.W4A8:
        return (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT4)
    if quant_type == QuantType.MXFP8:
        return (_CANN_MEGA_MOE_QUANT_MODE_MX, _CANN_TORCH_FLOAT8_E4M3FN, None)
    if quant_type == QuantType.W4A8MXFP:
        return (_CANN_MEGA_MOE_QUANT_MODE_MX, _CANN_TORCH_FLOAT8_E4M3FN, None)
    raise RuntimeError(
        "CANN 9.1 MegaMoe integration supports W8A8/W4A8 INT on A2/A3 and MXFP on FP8-capable "
        "MegaMoe platforms. "
        f"Unsupported quant type: {quant_type}."
    )


def get_moe_comm_method(moe_comm_type: MoECommType | None) -> MoECommMethod | None:
    return _MoECommMethods.get(moe_comm_type)


def setup_moe_comm_method(moe_config):
    if moe_config.ep_size > 1:
        _MoECommMethods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)
        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)
        _MoECommMethods[MoECommType.MC2] = MC2CommImpl(moe_config)
        _MoECommMethods[MoECommType.FUSED_MC2] = FusedMC2CommImpl(moe_config)
    else:
        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)


def set_gmmswigluquant_method():
    from vllm_ascend.ascend_config import get_ascend_config

    ascend_config = get_ascend_config()
    return ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant


@dataclass
class FusedExpertsResult:
    routed_out: torch.Tensor
    # This field is for shared experts and should be set by the MoE
    # communication method that supports shared experts in parallel with routed
    # experts.
    before_dispatch_evt: torch.npu.Event | None = None
    before_gmm2_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    # For dynamic_eplb
    group_list_type: int = 1
    expert_tokens: torch.Tensor | None = None
    swiglu_limit: float = 0.0


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()
        self.use_fusion_ops = set_gmmswigluquant_method()

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        return self.prepare_finalize.prepare(
            hidden_states,
            router_logits,
            enable_shared_expert_dp,
            replace_allreduce,
            quant_type,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        hidden_states = self.prepare_finalize.finalize(hidden_states, reduce_results, padded_hidden_states_shape)
        return hidden_states

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        # Check constraints
        assert fused_experts_input.hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.float8_e4m3fn,
        ], f"Unsupported hidden_states dtype: {fused_experts_input.hidden_states.dtype}"

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        assert moe_comm_method is not None, "Missing communication context"

        before_dispatch_evt = torch.npu.current_stream().record_event()
        routed_topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            routed_topk_ids = fused_experts_input.routing.log2phy[routed_topk_ids]

        token_dispatch_input = build_token_dispatch_input(
            fused_experts_input=fused_experts_input,
            topk_ids=routed_topk_ids,
        )
        token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)

        mlp_compute_input = build_mlp_compute_input(
            fused_experts_input=fused_experts_input,
            token_dispatch_output=token_dispatch_output,
            use_fusion_ops=self.use_fusion_ops,
        )

        mlp_output, before_gmm2_evt = self._apply_mlp(mlp_compute_input)

        before_combine_evt = torch.npu.current_stream().record_event()
        routed_out = self.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            combine_metadata=token_dispatch_output.combine_metadata,
        )

        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=token_dispatch_output.group_list_type,
            expert_tokens=token_dispatch_output.group_list,
            swiglu_limit=fused_experts_input.swiglu_limit,
        )

    def _apply_mlp(self, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
        return unified_apply_mlp(mlp_compute_input=mlp_compute_input)

    @abstractmethod
    def _get_token_dispatcher(self) -> MoETokenDispatcher:
        raise NotImplementedError("_get_token_dispatcher function not implemented.")

    @abstractmethod
    def _get_prepare_finalize(self) -> PrepareAndFinalize:
        raise NotImplementedError("_get_prepare_finalize function not implemented.")


class AllGatherCommImpl(MoECommMethod):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.

    NOTE(Yizhou): TBH, it is really weird that we were supposed to use
    `torch_npu.npu_moe_init_routing_v2` and `torch_npu.npu_moe_finalize_routing`
    or `torch_npu.npu_moe_token_permute` and `torch_npu.npu_moe_token_unpermute`
    for pre-processing and post-processing, respectively.
    But `npu_moe_finalize_routing` will lead to accuracy issues so we have to
    use `torch_npu.npu_moe_token_unpermute` instead.
    This is a workaround and should be removed after the issue is fixed.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAllGather(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAllGather(self.moe_config)


class MC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)


class AlltoAllCommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_grouped_matmul` is available.

    This implementation uses all-to-all communication to exchange tokens
    between data parallel ranks before and after the MLP computation. It should
    have better performance than AllGatherCommImpl when DP size > 1.
    """

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)


class FusedMC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def __init__(self, moe_config):
        super().__init__(moe_config)
        self._cann_mega_moe_ops = None
        self._cann_symm_buffers = {}
        if get_ascend_config().enable_fused_mc2 == _DISPATCH_FFN_COMBINE_MODE:
            self.expert_token_nums = torch.zeros([self.moe_config.num_local_experts], dtype=torch.int32, device="npu")
        else:
            self.expert_token_nums = None

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)

    def _load_cann_mega_moe_ops(self):
        if self._cann_mega_moe_ops is None:
            try:
                module = importlib.import_module(_CANN_MEGA_MOE_MODULE_NAME)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to import CANN 9.1 official MegaMoe. Source "
                    "/usr/local/Ascend/cann-9.1.0/set_env.sh before starting vLLM."
                ) from exc
            self._cann_mega_moe_ops = (module.get_symm_buffer_for_mega_moe, module.mega_moe)
        return self._cann_mega_moe_ops

    def _get_cann_symm_buffer(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        topk_ids: torch.Tensor,
        weight1: list[torch.Tensor],
        weight2: list[torch.Tensor],
        dispatch_quant_mode: int,
        dispatch_quant_out_dtype: int | None,
    ):
        get_symm_buffer_for_mega_moe, _ = self._load_cann_mega_moe_ops()
        group = get_mc2_group().device_group
        hidden = int(fused_experts_input.hidden_states.shape[-1])
        intermediate_hidden = _infer_intermediate_hidden(weight1[0], weight2[0], hidden)
        if self.token_dispatcher.global_bs > 0:
            num_max_tokens_per_rank = max(
                1,
                int(self.token_dispatcher.global_bs // self.token_dispatcher.ep_world_size),
            )
        else:
            num_max_tokens_per_rank = max(
                1,
                int(
                    getattr(self.token_dispatcher, "max_num_tokens_per_rank", 0)
                    or fused_experts_input.hidden_states.shape[0]
                ),
            )
        num_topk = int(topk_ids.shape[-1])
        redundant_experts = int(fused_experts_input.routing.global_redundant_expert_num)
        if fused_experts_input.routing.expert_map is not None:
            num_experts = int(fused_experts_input.routing.expert_map.numel()) + redundant_experts
        else:
            num_experts = int(self.moe_config.num_experts) + redundant_experts
        expert_per_rank = max(1, num_experts // int(self.token_dispatcher.ep_world_size))
        max_recv_token_num = max(
            1,
            num_max_tokens_per_rank * int(self.token_dispatcher.ep_world_size) * min(num_topk, expert_per_rank),
        )
        key = (
            id(group),
            num_experts,
            num_max_tokens_per_rank,
            max_recv_token_num,
            num_topk,
            hidden,
            intermediate_hidden,
            dispatch_quant_mode,
            dispatch_quant_out_dtype,
        )
        if key not in self._cann_symm_buffers:
            self._cann_symm_buffers[key] = get_symm_buffer_for_mega_moe(
                group,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                max_recv_token_num=max_recv_token_num,
                dispatch_quant_mode=dispatch_quant_mode,
                dispatch_quant_out_dtype=dispatch_quant_out_dtype,
                combine_quant_mode=0,
                comm_alg="",
            )
        return self._cann_symm_buffers[key]

    def _apply_cann_mega_moe(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        topk_ids: torch.Tensor,
    ):
        assert fused_experts_input.weights.w1_scale is not None
        assert fused_experts_input.weights.w2_scale is not None

        _, mega_moe = self._load_cann_mega_moe_ops()
        dispatch_quant_mode, dispatch_quant_out_dtype, weight_type = _get_cann_mega_moe_quant_settings(
            fused_experts_input.quant.quant_type
        )
        weight1 = _as_tensor_list(fused_experts_input.weights.w1, "w1")
        weight2 = _as_tensor_list(fused_experts_input.weights.w2, "w2")
        weight_scales1 = _as_tensor_list(fused_experts_input.weights.w1_scale, "w1_scale")
        weight_scales2 = _as_tensor_list(fused_experts_input.weights.w2_scale, "w2_scale")
        sym_buffer = self._get_cann_symm_buffer(
            fused_experts_input,
            topk_ids,
            weight1,
            weight2,
            dispatch_quant_mode,
            dispatch_quant_out_dtype,
        )
        activation_clamp = fused_experts_input.swiglu_limit if fused_experts_input.swiglu_limit > 0 else None
        x_active_mask = None
        if self.token_dispatcher.global_bs == 0 and fused_experts_input.routing.mc2_mask is not None:
            x_active_mask = fused_experts_input.routing.mc2_mask.to(torch.int8).contiguous()
        out, expert_tokens = mega_moe(
            fused_experts_input.hidden_states,
            topk_ids.to(torch.int32),
            fused_experts_input.topk_weights.to(torch.float32),
            weight1,
            weight2,
            sym_buffer,
            l1_weights_sf=weight_scales1,
            l2_weights_sf=weight_scales2,
            l1_bias=_pick_mega_moe_bias(
                fused_experts_input.weights.w1_scale_bias,
                fused_experts_input.weights.w1_bias,
            ),
            l2_bias=_pick_mega_moe_bias(
                fused_experts_input.weights.w2_scale_bias,
                fused_experts_input.weights.w2_bias,
            ),
            x_active_mask=x_active_mask,
            activation=_normalize_cann_activation(fused_experts_input.activation),
            activation_clamp=activation_clamp,
            weight1_type=weight_type,
            weight2_type=weight_type,
        )
        self.expert_token_nums = expert_tokens
        return out, expert_tokens

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        assert not (fused_experts_input.weights.w1_scale is None or fused_experts_input.weights.w2_scale is None), (
            "w1_scale and w2_scale cannot be None for FusedMC2CommImpl."
        )

        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2), (
            "token_dispatcher must be an instance of TokenDispatcherWithMC2."
        )

        # Apply log2phy if needed
        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]

        expert_tokens = None
        if get_ascend_config().enable_fused_mc2 == _DISPATCH_FFN_COMBINE_MODE:
            assert not (
                fused_experts_input.weights.w1_scale_bias is None or fused_experts_input.weights.w2_scale_bias is None
            ), "w1_scale_bias and w2_scale_bias cannot be None when enable_fused_mc2=1."

            out = torch.empty_like(fused_experts_input.hidden_states)
            torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
                x=fused_experts_input.hidden_states,
                weight1=fused_experts_input.weights.w1,
                weight2=fused_experts_input.weights.w2,
                expert_idx=topk_ids,
                scale1=fused_experts_input.weights.w1_scale,
                scale2=fused_experts_input.weights.w2_scale,
                bias1=fused_experts_input.weights.w1_scale_bias,
                bias2=fused_experts_input.weights.w2_scale_bias,
                probs=fused_experts_input.topk_weights.to(torch.float32),
                group=self.token_dispatcher.moe_all_to_all_group_name,
                max_output_size=131072,
                swiglu_limit=fused_experts_input.swiglu_limit,
                x_active_mask=fused_experts_input.routing.mc2_mask,
                out=out,
                expert_token_nums=self.expert_token_nums,
            )
            expert_tokens = self.expert_token_nums
        elif get_ascend_config().enable_fused_mc2 == _CANN_MEGA_MOE_FUSED_MC2_MODE:
            out, expert_tokens = self._apply_cann_mega_moe(fused_experts_input, topk_ids)
        else:
            raise ValueError(f"Wrong value of {get_ascend_config().enable_fused_mc2=}")
        return FusedExpertsResult(
            routed_out=out, expert_tokens=expert_tokens, swiglu_limit=fused_experts_input.swiglu_limit
        )
