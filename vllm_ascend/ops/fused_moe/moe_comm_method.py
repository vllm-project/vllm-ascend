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
from vllm.logger import logger
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
from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe

_MoECommMethods: dict[MoECommType | None, MoECommMethod] = {}

_CANN_ACL_INT8 = 258
_CANN_ACL_INT4 = 285
_CANN_TORCH_FLOAT8_E4M3FN = 24

_CANN_MEGA_MOE_QUANT_MODE_INT8 = 2
_CANN_MEGA_MOE_QUANT_MODE_MX = 4

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


def _to_float32_bias_list(bias: list[torch.Tensor] | None) -> list[torch.Tensor] | None:
    """Coerce l1_bias/l2_bias entries to FLOAT32 as required by CANN mega_moe.

    The A8W4-INT precision-compensation biases (B1/B2) must be float32 per the
    op interface doc. Skip the cast when a tensor is already float32 so no extra
    kernel is launched in the common case.
    """
    if bias is None:
        return None
    return [t if t.dtype == torch.float32 else t.to(torch.float32) for t in bias]


def _infer_intermediate_hidden(weight1: torch.Tensor, weight2: torch.Tensor, hidden: int) -> int:
    if weight2.dim() >= 2:
        if weight2.shape[-1] == hidden:
            return int(weight2.shape[-2])
        if weight2.shape[-2] == hidden:
            return int(weight2.shape[-1])
        # Quantized down-projection weights are stored as
        # [intermediate, hidden // pack_factor]: int4 packs the contraction
        # dim (//2 for two-per-int8 storage, //8 for eight-per-int32), so the
        # last dim no longer equals hidden. The row dim (-2) is the real
        # intermediate size and is never packed, so prefer it whenever the
        # last dim evenly divides hidden.
        if weight2.shape[-1] and hidden % int(weight2.shape[-1]) == 0:
            return int(weight2.shape[-2])
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

def _get_cann_mega_moe_quant_settings(quant_type: QuantType) -> tuple[int, int | None, int | None]:
    # Returns (dispatch_quant_mode, dispatch_quant_out_dtype, weight_type).
    # The current custom op package still requires explicit INT4 for W4A8
    # packed weights; otherwise it derives W4A8's packed N as an INT8 N and
    # rejects weight2.
    #
    # dispatch_quant_out_dtype: the doc types this as torch.dtype (torch.int8 /
    # torch.float8_e4m3fn). We pass the ACL enum ints (258 / 24) because W8A8
    # was validated end-to-end this way in PD; switching W4A8 to torch.int8 did
    # NOT fix the W4A8 accuracy issue and slowed graph capture (see bug_a3.md),
    # so keep the working values until the W4A8 accuracy root cause is found on
    # the operator side.
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
        self._mega_moe_symm_buffer = None
        if get_ascend_config().enable_fused_mc2 == 1:
            self.expert_token_nums = torch.zeros([self.moe_config.num_local_experts], dtype=torch.int32, device="npu")
        else:
            self.expert_token_nums = None

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)

    def _init_mega_moe_symm_buffer(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        topk_ids: torch.Tensor,
        weight1: list[torch.Tensor],
        weight2: list[torch.Tensor],
        dispatch_quant_mode: int,
        dispatch_quant_out_dtype: int | None,
    ):
        # FusedMC2CommImpl always builds a TokenDispatcherWithMC2 (see
        # setup_moe_comm_method), which is where global_bs / ep_world_size live.
        # Assert it so mypy resolves those attributes off the base dispatcher.
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        group = get_mc2_group().device_group
        hidden = int(fused_experts_input.hidden_states.shape[-1])
        intermediate_hidden = _infer_intermediate_hidden(weight1[0], weight2[0], hidden)
        # The sym buffer is allocated by get_symm_buffer_for_mega_moe, a
        # collective handshake over the EP (mc2) group. Its shape params —
        # especially num_max_tokens_per_rank — MUST be identical on every EP
        # rank, otherwise ranks allocate mismatched buffers / at different
        # times and HCCL aborts (SUSPECT REMOTE ERROR 507057). So this value
        # must be derived ONLY from rank-invariant, compile-time config,
        # NEVER from the current forward's per-rank token count.
        if self.token_dispatcher.global_bs > 0:
            # global_bs = num_tokens_per_tp_rank * ep_world_size (compile-time).
            num_max_tokens_per_rank = max(
                1,
                int(self.token_dispatcher.global_bs // self.token_dispatcher.ep_world_size),
            )
        else:
            # num_tokens_per_tp_rank, set once in TokenDispatcherWithMC2.__init__
            # from scheduler/graph config — rank-invariant.
            rank_invariant_cap = getattr(self.token_dispatcher, "max_num_tokens_per_rank", 0)
            assert rank_invariant_cap and int(rank_invariant_cap) > 0, (
                "CANN MegaMoe sym buffer needs a rank-invariant token cap "
                "(token_dispatcher.max_num_tokens_per_rank). Falling back to a "
                "per-forward token count would desync the EP-group collective "
                "(HCCL 507057). Got: "
                f"{rank_invariant_cap!r}"
            )
            num_max_tokens_per_rank = max(1, int(rank_invariant_cap))
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

        logger.info(
            "CANN MegaMoe sym-buffer alloc (must match across all EP ranks): "
            "ep_rank=%s ep_world=%s global_bs=%s",
            getattr(self.token_dispatcher, "ep_rank_id", "?"),
            getattr(self.token_dispatcher, "ep_world_size", "?"),
            self.token_dispatcher.global_bs,
        )
        self._mega_moe_symm_buffer = get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            max_recv_token_num=max_recv_token_num,
            dispatch_quant_mode=dispatch_quant_mode,
            dispatch_quant_out_dtype=dispatch_quant_out_dtype,
        )

    def _apply_cann_mega_moe(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        topk_ids: torch.Tensor,
    ):
        assert fused_experts_input.weights.w1_scale is not None
        assert fused_experts_input.weights.w2_scale is not None
        # TokenDispatcherWithMC2 carries global_bs (used below for the mc2_mask
        # branch); assert the subtype so mypy resolves it off the base class.
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)

        dispatch_quant_mode, dispatch_quant_out_dtype, weight_type = _get_cann_mega_moe_quant_settings(
            fused_experts_input.quant.quant_type
        )
        weight1 = _as_tensor_list(fused_experts_input.weights.w1, "w1")
        weight2 = _as_tensor_list(fused_experts_input.weights.w2, "w2")
        # A8W4-INT MegaMoe reads N from weight1.storageShape.lastDim treated as int8 (N = lastDim*2)
        # and checks weight2.dim0 == N/2, so the weights MUST be int8-shaped (two int4 per byte), NOT
        # the eight-int4-per-int32 packing (that makes the op read N four times too small and fail
        # CheckWeight2Input). The op prototype also REQUIRES FRACTAL_NZ per expert. The W4A8 quant
        # method therefore builds per-expert int8 + FRACTAL_NZ lists (cann_mega_moe_*_weight_list) and
        # they are passed through as-is here. W8A8 weights are already int8 + FRACTAL_NZ, also as-is.
        weight_scales1 = _as_tensor_list(fused_experts_input.weights.w1_scale, "w1_scale")
        weight_scales2 = _as_tensor_list(fused_experts_input.weights.w2_scale, "w2_scale")
        # MegaMoe requires per-expert weight scales to be 1-D. The W4A8 method
        # squeezes w13 scales but leaves w2 scales as [1, hidden]; drop the
        # leading singleton dim so CheckWeightScaleInput passes. Guarded to the
        # [1, N] per-channel case to avoid flattening genuine per-group scales.
        weight_scales1 = [t.squeeze(0) if (t.dim() == 2 and t.shape[0] == 1) else t for t in weight_scales1]
        weight_scales2 = [t.squeeze(0) if (t.dim() == 2 and t.shape[0] == 1) else t for t in weight_scales2]

        if self._mega_moe_symm_buffer is None:
            self._init_mega_moe_symm_buffer(
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
            # mc2_mask comes from the reserved bool buffer in
            # ascend_forward_context.set_mc2_mask. MegaMoe wants int8 as
            # the per-token active mask, so cast only when the dtype does
            # not already match — saves the kernel launch when an upstream
            # change ever flips the reserved buffer to int8.
            raw_mask = fused_experts_input.routing.mc2_mask
            if raw_mask.dtype == torch.int8:
                x_active_mask = raw_mask.contiguous()
            else:
                x_active_mask = raw_mask.to(torch.int8).contiguous()
        # A8W4-INT precision-compensation biases B1/B2 (l1_bias/l2_bias). The doc
        # requires FLOAT32, ND, per-expert shapes (2*intermediate_hidden,) for B1
        # and (hidden,) for B2 — which is exactly the ``8 * sum`` correction term
        # the W4A8 quant method emits as w13_scale_bias / w2_scale_bias. Cast to
        # float32 defensively so a non-fp32 scale_bias (e.g. from the modelslim
        # new_quant_version transpose+sum path) still satisfies the op contract.
        l1_bias = _to_float32_bias_list(
            _pick_mega_moe_bias(
                fused_experts_input.weights.w1_scale_bias,
                fused_experts_input.weights.w1_bias,
            )
        )
        l2_bias = _to_float32_bias_list(
            _pick_mega_moe_bias(
                fused_experts_input.weights.w2_scale_bias,
                fused_experts_input.weights.w2_bias,
            )
        )
        out, expert_tokens = mega_moe(
            fused_experts_input.hidden_states,
            topk_ids.to(torch.int32),
            fused_experts_input.topk_weights.to(torch.float32),
            weight1,
            weight2,
            self._mega_moe_symm_buffer,
            l1_weights_sf=weight_scales1,
            l2_weights_sf=weight_scales2,
            l1_bias=l1_bias,
            l2_bias=l2_bias,
            x_active_mask=x_active_mask,
            activation_clamp=activation_clamp,
            weight1_type=weight_type,
            weight2_type=weight_type,
        )
        # NOTE: self.expert_token_nums is only used by the
        # dispatch_ffn_combine path (enable_fused_mc2 == 1) as a
        # pre-allocated in/out buffer. The MegaMoe op returns a fresh
        # expert_tokens tensor that is consumed by the caller via the
        # return value, so there is nothing to keep on the instance.
        return out, expert_tokens

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        # FIX(mega all-route): the shared expert (and any unquantized MoE) is QuantType.NONE and
        # cannot go through MegaMoe (which handles W8A8/W4A8/MXFP only). All-route now sends every
        # MoE layer here, so delegate unquantized layers to the generic base path
        # (dispatch -> unified_apply_mlp -> combine), which uses their intact weights. Quantized
        # routed experts (whose standard weights are freed for MegaMoe) still take the path below.
        if fused_experts_input.quant.quant_type == QuantType.NONE:
            return MoECommMethod.fused_experts(self, fused_experts_input)
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
        if get_ascend_config().enable_fused_mc2 == 1:
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
        elif get_ascend_config().enable_fused_mc2 == 2:
            assert fused_experts_input.routing.expert_map is not None, "expert_map cannot be None."
            out, expert_tokens = torch.ops._C_ascend.dispatch_gmm_combine_decode(  # type: ignore
                x=fused_experts_input.hidden_states,
                expert_ids=topk_ids,
                gmm1_permuted_weight=fused_experts_input.weights.w1,
                gmm1_permuted_weight_scale=fused_experts_input.weights.w1_scale,
                gmm2_weight=fused_experts_input.weights.w2,
                gmm2_weight_scale=fused_experts_input.weights.w2_scale,
                expert_smooth_scales=None,
                expert_scales=fused_experts_input.topk_weights.to(torch.float32),
                group_ep=self.token_dispatcher.moe_all_to_all_group_name,
                ep_rank_size=self.token_dispatcher.ep_world_size,
                ep_rank_id=self.token_dispatcher.ep_rank_id,
                moe_expert_num=self.moe_config.num_experts,
                global_bs=self.token_dispatcher.global_bs,
            )
        elif get_ascend_config().enable_fused_mc2 == 3:
            out, expert_tokens = self._apply_cann_mega_moe(fused_experts_input, topk_ids)
        else:
            raise ValueError(f"Wrong value of {get_ascend_config().enable_fused_mc2=}")
        return FusedExpertsResult(
            routed_out=out, expert_tokens=expert_tokens, swiglu_limit=fused_experts_input.swiglu_limit
        )