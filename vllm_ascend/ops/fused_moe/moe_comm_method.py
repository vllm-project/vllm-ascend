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

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_config import _MEGA_MOE_SUPPORTED, get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe import moe_utils
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEFusedExpertsInput
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput, build_mlp_compute_input
from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import build_token_dispatch_input
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
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
from vllm_ascend.utils import AscendDeviceType, enable_dsa_cp, get_ascend_device_type

_MoECommMethods: dict[MoECommType | None, MoECommMethod] = {}


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


@dataclass
class FusedExpertsResult:
    routed_out: torch.Tensor | None
    # DSA-CP MoE DBO can leave C1 in flight while independent full-batch
    # shared-expert work runs.
    finish_routed_out: Callable[[], torch.Tensor] | None = None
    # This field is for shared experts and should be set by the MoE
    # communication method that supports shared experts in parallel with routed
    # experts.
    before_dispatch_evt: torch.npu.Event | None = None
    before_gmm2_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    # For dynamic_eplb
    group_list_type: int = 1
    expert_tokens: torch.Tensor | None = None


def _split_fused_experts_input(
    fused_experts_input: MoEFusedExpertsInput,
) -> tuple[MoEFusedExpertsInput, MoEFusedExpertsInput]:
    """Split token-indexed MoE inputs while preserving original token order."""

    num_tokens = fused_experts_input.hidden_states.shape[0]
    split_point = (num_tokens + 1) // 2
    token_slices = (slice(0, split_point), slice(split_point, num_tokens))
    micro_batches = []
    for token_slice in token_slices:
        pertoken_scale = fused_experts_input.routing.pertoken_scale
        if pertoken_scale is not None:
            pertoken_scale = pertoken_scale[token_slice]
        micro_batches.append(
            replace(
                fused_experts_input,
                hidden_states=fused_experts_input.hidden_states[token_slice],
                topk_ids=fused_experts_input.topk_ids[token_slice],
                topk_weights=fused_experts_input.topk_weights[token_slice],
                routing=replace(
                    fused_experts_input.routing,
                    pertoken_scale=pertoken_scale,
                ),
            )
        )
    return micro_batches[0], micro_batches[1]


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()
        self.lora_context = None

    def set_lora_context(self, lora_context) -> None:
        self.lora_context = lora_context
        self.prepare_finalize.set_lora_context(lora_context)
        self.token_dispatcher.set_lora_context(lora_context)

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        return self.prepare_finalize.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=replace_allreduce,
            quant_type=quant_type,
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
            torch.uint8,
        ], f"Unsupported hidden_states dtype: {fused_experts_input.hidden_states.dtype}"

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        assert moe_comm_method is not None, "Missing communication context"

        before_dispatch_evt = torch.npu.current_stream().record_event()

        token_dispatch_input = build_token_dispatch_input(
            fused_experts_input=fused_experts_input,
        )
        token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)

        mlp_compute_input = build_mlp_compute_input(
            fused_experts_input=fused_experts_input,
            token_dispatch_output=token_dispatch_output,
            moe_config=self.moe_config,
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

    def _local_dsa_cp_moe_dbo_candidate(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ) -> tuple[bool, str]:
        ascend_config = get_ascend_config()
        vllm_config = get_current_vllm_config()
        hidden_states = fused_experts_input.hidden_states
        num_tokens = hidden_states.shape[0] if hidden_states.ndim > 0 else 0

        if get_ascend_device_type() != AscendDeviceType.A5:
            return False, "only A5 is supported"
        if not enable_dsa_cp():
            return False, "DSA-CP is disabled"
        if _EXTRA_CTX.moe_comm_type != MoECommType.ALLTOALL:
            return False, "the selected MoE communication method is not AllToAllV"
        if not vllm_config.model_config.enforce_eager:
            return False, "only eager execution is supported"
        if self.moe_config.ep_size <= 1:
            return False, "EP size must be greater than one"
        if fused_experts_input.lora_context is not None or self.lora_context is not None:
            return False, "LoRA is unsupported"
        if fused_experts_input.dynamic_eplb:
            return False, "dynamic EPLB is unsupported"
        if ascend_config.multistream_overlap_shared_expert:
            return False, "shared-expert multistream overlap is unsupported"
        if hidden_states.ndim != 2:
            return False, "hidden states must be token-major 2-D"
        if fused_experts_input.topk_ids.shape[0] != num_tokens:
            return False, "top-k ids are not token-aligned"
        if fused_experts_input.topk_weights.shape[0] != num_tokens:
            return False, "top-k weights are not token-aligned"
        if fused_experts_input.routing.mc2_mask is not None:
            return False, "MC2 mask is incompatible with AllToAllV DBO"
        pertoken_scale = fused_experts_input.routing.pertoken_scale
        if pertoken_scale is not None and pertoken_scale.shape[0] != num_tokens:
            return False, "per-token scale is not token-aligned"
        if num_tokens < ascend_config.dsa_cp_moe_dbo_token_threshold:
            return False, "local token count is below the configured threshold"
        split_point = (num_tokens + 1) // 2
        if split_point <= 0 or split_point >= num_tokens:
            return False, "the second micro-batch would be empty"
        return True, "eligible"

    def _dsa_cp_moe_dbo_eligible(self, fused_experts_input: MoEFusedExpertsInput) -> bool:
        if not get_ascend_config().enable_dsa_cp_moe_dbo:
            return False

        forward_context = get_forward_context()
        cache_attr = "_ascend_dsa_cp_moe_dbo_eligible"
        cached = getattr(forward_context, cache_attr, None)
        if isinstance(cached, bool):
            return cached

        local_eligible, local_reason = self._local_dsa_cp_moe_dbo_candidate(fused_experts_input)
        eligibility = torch.tensor(
            int(local_eligible),
            dtype=torch.int32,
            device=fused_experts_input.hidden_states.device,
        )
        # This synchronization occurs once per forward and is cached. It is
        # required so every EP rank chooses the same collective protocol even
        # when DSA-CP gives ranks different local token payloads.
        dist.all_reduce(
            eligibility,
            op=dist.ReduceOp.MIN,
            group=self.token_dispatcher.ep_group,
        )
        globally_eligible = bool(eligibility.item())
        setattr(forward_context, cache_attr, globally_eligible)
        if globally_eligible:
            logger.info_once(
                "DSA-CP MoE DBO is active: ep_size=%d, local_tokens=%d, collective_order=D0,D1,C0,C1",
                self.moe_config.ep_size,
                fused_experts_input.hidden_states.shape[0],
            )
        else:
            logger.debug(
                "DSA-CP MoE DBO falls back to synchronous AllToAllV. local_reason=%s",
                local_reason,
            )
        return globally_eligible

    def _fused_experts_dsa_cp_moe_dbo(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ) -> FusedExpertsResult:
        assert isinstance(self.token_dispatcher, TokenDispatcherWithAll2AllV)
        before_dispatch_evt = torch.npu.current_stream().record_event()
        micro_batch0, micro_batch1 = _split_fused_experts_input(fused_experts_input)

        # Every rank launches the same global collective order: D0, D1, C0,
        # C1. Payload sizes are allowed to differ across DSA-CP ranks.
        dispatch0 = self.token_dispatcher.dispatch_start(build_token_dispatch_input(fused_experts_input=micro_batch0))
        dispatch1 = self.token_dispatcher.dispatch_start(build_token_dispatch_input(fused_experts_input=micro_batch1))

        dispatch_output0 = self.token_dispatcher.dispatch_finish(dispatch0)
        mlp_output0, before_gmm2_evt = self._apply_mlp(
            build_mlp_compute_input(
                fused_experts_input=micro_batch0,
                token_dispatch_output=dispatch_output0,
                moe_config=self.moe_config,
            )
        )

        # D1 overlaps MLP0. C0 then overlaps MLP1.
        dispatch_output1 = self.token_dispatcher.dispatch_finish(dispatch1)
        before_combine_evt = torch.npu.current_stream().record_event()
        combine0 = self.token_dispatcher.combine_start(
            mlp_output0,
            dispatch_output0.combine_metadata,
        )
        mlp_output1, _ = self._apply_mlp(
            build_mlp_compute_input(
                fused_experts_input=micro_batch1,
                token_dispatch_output=dispatch_output1,
                moe_config=self.moe_config,
            )
        )
        combine1 = self.token_dispatcher.combine_start(
            mlp_output1,
            dispatch_output1.combine_metadata,
        )
        routed_out0 = self.token_dispatcher.combine_finish(combine0)

        routed_out: torch.Tensor | None = None

        def finish_routed_out() -> torch.Tensor:
            nonlocal routed_out
            if routed_out is None:
                routed_out1 = self.token_dispatcher.combine_finish(combine1)
                routed_out = torch.cat((routed_out0, routed_out1), dim=0)
            return routed_out

        defer_final_combine = get_ascend_config().enable_dsa_cp_moe_dbo_shared_expert_overlap
        if not defer_final_combine:
            routed_out = finish_routed_out()

        return FusedExpertsResult(
            routed_out=routed_out,
            finish_routed_out=finish_routed_out if defer_final_combine else None,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=dispatch_output0.group_list_type,
            expert_tokens=dispatch_output0.group_list + dispatch_output1.group_list,
        )

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        if not self._dsa_cp_moe_dbo_eligible(fused_experts_input):
            return super().fused_experts(fused_experts_input)
        return self._fused_experts_dsa_cp_moe_dbo(fused_experts_input)


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
        if _MEGA_MOE_SUPPORTED:
            self.mega_moe_symm_buffer = None
            self.get_symm_buffer_for_mega_moe, self.mega_moe = moe_utils.load_cann_mega_moe_ops()
        if get_ascend_config().enable_fused_mc2 == 1:
            self.expert_token_nums = torch.zeros([self.moe_config.num_local_experts], dtype=torch.int32, device="npu")
        else:
            self.expert_token_nums = None

        self.swiglu_limit = 0.0 if moe_config.swiglu_limit is None else moe_config.swiglu_limit
        self.swiglu_alpha = 1.0 if moe_config.swiglu_alpha is None else moe_config.swiglu_alpha
        self.swiglu_beta = 0.0 if moe_config.swiglu_beta is None else moe_config.swiglu_beta

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)

    def _init_mega_moe_symm_buffer(
        self,
        dispatch_quant_mode: int = 0,
        dispatch_quant_out_dtype: torch.dtype | None = None,
    ):
        # FusedMC2CommImpl always builds a TokenDispatcherWithMC2 (see
        # setup_moe_comm_method), which is where global_bs / ep_world_size live.
        # Assert it so mypy resolves those attributes off the base dispatcher.
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        group = get_mc2_group().device_group
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
            num_max_tokens_per_rank = max(1, int(rank_invariant_cap))
        num_topk = self.moe_config.experts_per_token
        num_experts = self.moe_config.num_experts
        expert_per_rank = max(1, num_experts // int(self.token_dispatcher.ep_world_size))
        max_recv_token_num = max(
            1,
            num_max_tokens_per_rank * int(self.token_dispatcher.ep_world_size) * min(num_topk, expert_per_rank),
        )

        logger.info(
            "CANN MegaMoe sym-buffer alloc (must match across all EP ranks): ep_rank=%s ep_world=%s global_bs=%s",
            getattr(self.token_dispatcher, "ep_rank_id", "?"),
            getattr(self.token_dispatcher, "ep_world_size", "?"),
            self.token_dispatcher.global_bs,
        )

        return self.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden=self.moe_config.hidden_dim,
            intermediate_hidden=2 * self.moe_config.intermediate_size_per_partition,
            max_recv_token_num=max_recv_token_num,
            dispatch_quant_mode=dispatch_quant_mode,
            dispatch_quant_out_dtype=dispatch_quant_out_dtype,
        )

    def _apply_cann_mega_moe(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        # TokenDispatcherWithMC2 carries global_bs (used below for the mc2_mask
        # branch); assert the subtype so mypy resolves it off the base class.
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)

        def to_list(x):
            return x if isinstance(x, list) else [x]

        weight1 = to_list(fused_experts_input.weights.w1)
        weight2 = to_list(fused_experts_input.weights.w2)
        # A8W4-INT MegaMoe reads N from weight1.storageShape.lastDim treated as int8 (N = lastDim*2)
        # and checks weight2.dim0 == N/2, so the weights MUST be int8-shaped (two int4 per byte), NOT
        # the eight-int4-per-int32 packing (that makes the op read N four times too small and fail
        # CheckWeight2Input). The op prototype also REQUIRES FRACTAL_NZ per expert. The W4A8 quant
        # method therefore builds per-expert int8 + FRACTAL_NZ lists (cann_mega_moe_*_weight_list) and
        # they are passed through as-is here. W8A8 weights are already int8 + FRACTAL_NZ, also as-is.
        weight_scales1 = fused_experts_input.weights.w1_scale
        weight_scales2 = fused_experts_input.weights.w2_scale
        dispatch_quant_mode, dispatch_quant_out_dtype, weight_type = moe_utils._get_cann_mega_moe_quant_settings(
            fused_experts_input.quant.quant_type
        )

        if self.mega_moe_symm_buffer is None:
            self.mega_moe_symm_buffer = self._init_mega_moe_symm_buffer(
                dispatch_quant_mode,
                dispatch_quant_out_dtype,
            )
        else:
            self.mega_moe_symm_buffer.dispatch_quant_mode = dispatch_quant_mode
            self.mega_moe_symm_buffer.dispatch_quant_out_dtype = dispatch_quant_out_dtype

        activation_clamp = self.swiglu_limit if self.swiglu_limit > 0 else None
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
        # A8W4-INT precision-compensation biases B1/B2 (l1_bias/l2_bias).
        l1_bias = fused_experts_input.weights.w1_scale_bias
        l2_bias = fused_experts_input.weights.w2_scale_bias

        out, expert_tokens = self.mega_moe(
            fused_experts_input.hidden_states,
            fused_experts_input.topk_ids.to(torch.int32),
            fused_experts_input.topk_weights.to(torch.float32),
            weight1,
            weight2,
            self.mega_moe_symm_buffer,
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
        # mega_moe path (enable_fused_mc2 == 1) as a
        # pre-allocated in/out buffer. The MegaMoe op returns a fresh
        # expert_tokens tensor that is consumed by the caller via the
        # return value, so there is nothing to keep on the instance.
        return out, expert_tokens

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2), (
            "token_dispatcher must be an instance of TokenDispatcherWithMC2."
        )

        expert_tokens = None
        if get_ascend_config().enable_fused_mc2 == 1:
            if _MEGA_MOE_SUPPORTED:
                out, expert_tokens = self._apply_cann_mega_moe(fused_experts_input)
            else:
                assert not (
                    fused_experts_input.weights.w1_scale_bias is None
                    or fused_experts_input.weights.w2_scale_bias is None
                ), "w1_scale_bias and w2_scale_bias cannot be None when enable_fused_mc2=1."

                out = torch.empty_like(fused_experts_input.hidden_states)
                torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
                    x=fused_experts_input.hidden_states,
                    weight1=fused_experts_input.weights.w1,
                    weight2=fused_experts_input.weights.w2,
                    expert_idx=fused_experts_input.topk_ids,
                    scale1=fused_experts_input.weights.w1_scale,
                    scale2=fused_experts_input.weights.w2_scale,
                    bias1=fused_experts_input.weights.w1_scale_bias,
                    bias2=fused_experts_input.weights.w2_scale_bias,
                    probs=fused_experts_input.topk_weights.to(torch.float32),
                    group=self.token_dispatcher.moe_all_to_all_group_name,
                    max_output_size=get_ascend_config().mega_moe_max_tokens,
                    swiglu_limit=self.swiglu_limit,
                    x_active_mask=fused_experts_input.routing.mc2_mask,
                    out=out,
                    expert_token_nums=self.expert_token_nums,
                )
                expert_tokens = self.expert_token_nums
        else:
            raise ValueError(f"Wrong value of {get_ascend_config().enable_fused_mc2=}")
        return FusedExpertsResult(
            routed_out=out,
            expert_tokens=expert_tokens,
        )
