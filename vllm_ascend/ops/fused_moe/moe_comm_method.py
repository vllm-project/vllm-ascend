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
from dataclasses import dataclass

from npu_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group, get_mega_group
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
_MEGA_MOE_DISPATCH_QUANT_MODE_None = 0
_MEGA_MOE_DISPATCH_QUANT_MODE_INT = 2
_MEGA_MOE_DISPATCH_QUANT_MODE_MX = 4
_MEGA_MOE_DISPATCH_QUANT_OUT_TYPE_E5M2 = 23
_MEGA_MOE_DISPATCH_QUANT_OUT_TYPE_E4M3FN = 24


def _is_mega_moe_supported(fused_experts_input: MoEFusedExpertsInput) -> bool:
    if fused_experts_input.quant.is_mxfp:
        return True
    return fused_experts_input.quant.quant_type in (QuantType.NONE, QuantType.W8A8, QuantType.W4A8)

def _get_dispatch_quant_out_type(dtype: torch.dtype | None) -> int:
    if dtype == torch.float8_e5m2:
        return _MEGA_MOE_DISPATCH_QUANT_OUT_TYPE_E5M2
    return _MEGA_MOE_DISPATCH_QUANT_OUT_TYPE_E4M3FN


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
    swiglu_limit: int = 0


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
        ]

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


mega_moe_symm_buffer = None


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
        if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
            self.expert_token_nums = torch.zeros([self.moe_config.num_local_experts], dtype=torch.int32, device="npu")
        else:
            self.expert_token_nums = None

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)

    def _get_mega_buffer(self, fused_experts_input: MoEFusedExpertsInput):
        assert not (fused_experts_input.weights.w1 is None or fused_experts_input.weights.w2 is None), (
            "w1 and w2 cannot be None for mega_moe."
        )

        _dispatch_quant_mode = None
        _dispatch_quant_out_type = None
        if not fused_experts_input.quant.is_quant:
            assert (fused_experts_input.weights.w1_scale is None 
                and fused_experts_input.weights.w2_scale is None
                and fused_experts_input.weights.w1_bias is None
                and fused_experts_input.weights.w2_bias is None), (
                "w1 scale, w2 scale, w1 bias, w2 bias must be None for A16W16 mega_moe."
            )

            _dispatch_quant_mode = _MEGA_MOE_DISPATCH_QUANT_MODE_None
            _dispatch_quant_out_type = None

        elif fused_experts_input.quant.is_mxfp:
            assert not (fused_experts_input.weights.w1_scale is None or fused_experts_input.weights.w2_scale is None), (
                "w1 scale and w2 scale can not be None for MXFP mega_moe."
            )
            assert (fused_experts_input.weights.w1_bias is None and fused_experts_input.weights.w2_bias is None), (
                "w1 bias and w2 bias must be None for MXFP mega_moe."
            )
            assert fused_experts_input.quant.mxfp is not None, "mxfp params are required for MXFP mega_moe."

            _dispatch_quant_mode = _MEGA_MOE_DISPATCH_QUANT_MODE_MX
            _dispatch_quant_out_type = _get_dispatch_quant_out_type(
                fused_experts_input.quant.mxfp.act_quant_type
            )
        
        elif fused_experts_input.quant.quant_type == QuantType.W8A8:
            assert not (fused_experts_input.weights.w1_scale is None or fused_experts_input.weights.w2_scale is None), (
                "w1 scale and w2 scale can not be None for W8A8 INT mega_moe."
            )
            assert (fused_experts_input.weights.w1_bias is None and fused_experts_input.weights.w2_bias is None), (
                "w1 bias and w2 bias must be None for W8A8 INT mega_moe."
            )

            _dispatch_quant_mode.dispatch_quant_mode = _MEGA_MOE_DISPATCH_QUANT_MODE_INT
            _dispatch_quant_out_type.dispatch_quant_out_type = torch.int8

        elif fused_experts_input.quant.quant_type == QuantType.W4A8:
            assert (fused_experts_input.weights.w1_scale is not None 
                and fused_experts_input.weights.w2_scale is not None
                and fused_experts_input.weights.w1_bias is not None
                and fused_experts_input.weights.w2_bias is not None), (
                "w1 scale, w2 scale, w1 bias, w2 bias can not be None for W4A8 mega_moe."
            )            

            _dispatch_quant_mode.dispatch_quant_mode = _MEGA_MOE_DISPATCH_QUANT_MODE_INT
            _dispatch_quant_out_type.dispatch_quant_out_type = torch.int8
        
        else:
            raise AssertionError(f"this experts input quant type is not supported yet.")

        global mega_moe_symm_buffer

        mega_moe_symm_buffer = get_symm_buffer_for_mega_moe(
            group=get_mega_group().device_group,
            num_experts=self.moe_config.num_experts,
            num_max_tokens_per_rank=0,
            num_topk=self.moe_config.experts_per_token,
            hidden=self.moe_config.hidden_dim,
            intermediate_hidden=2 * self.moe_config.intermediate_size_per_partition,
            dispatch_quant_mode=_dispatch_quant_mode,
            ispatch_quant_out_type=_dispatch_quant_out_type,
        )

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        # assert not (fused_experts_input.weights.w1_scale is None or fused_experts_input.weights.w2_scale is None), (
            # "w1_scale and w2_scale cannot be None for FusedMC2CommImpl."
        # )

        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2), (
            "token_dispatcher must be an instance of TokenDispatcherWithMC2."
        )

        # Apply log2phy if needed
        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]

        expert_tokens = None
        if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
            if _is_mega_moe_supported(fused_experts_input):
                global mega_moe_symm_buffer
                if mega_moe_symm_buffer is None:
                    self._get_mega_buffer(fused_experts_input)

                out, expert_tokens = mega_moe(
                    x=fused_experts_input.hidden_states,
                    topk_ids=topk_ids,
                    topk_weights=fused_experts_input.topk_weights,
                    l1_weights=fused_experts_input.weights.w1,
                    l2_weights=fused_experts_input.weights.w2,
                    sym_buffer=mega_moe_symm_buffer,
                    l1_weights_sf=fused_experts_input.weights.w1_scale,
                    l2_weights_sf=fused_experts_input.weights.w2_scale,
                    l1_bias=fused_experts_input.weights.w1_bias,
                    l2_bias=fused_experts_input.weights.w2_bias,
                )

            else:
                assert not (
                    fused_experts_input.weights.w1_scale_bias is None
                    or fused_experts_input.weights.w2_scale_bias is None
                ), "w1_scale_bias and w2_scale_bias cannot be None for FusedMC2CommImpl (enable_fused_mc2=1)."
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
        else:
            raise ValueError(f"Wrong value of {get_ascend_config().enable_fused_mc2=}")
        return FusedExpertsResult(
            routed_out=out, expert_tokens=expert_tokens, swiglu_limit=fused_experts_input.swiglu_limit
        )
