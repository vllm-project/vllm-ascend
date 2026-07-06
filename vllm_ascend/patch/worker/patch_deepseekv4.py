#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is part of the vllm-ascend project.
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
"""AFD (Attention-FFN Disaggregation) patches for DeepSeek V4.

This module splits ``DeepseekV2DecoderLayer.forward`` into an attention-side
``compute_attn_output`` and an FFN-side ``compute_ffn_output`` so that the two
halves can run on separate workers and exchange intermediates through the AFD
connector (``NPUP2PAFDConnector``).
"""

import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import Any, Optional

import torch
import torch.nn.functional as F
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config  # noqa: F401  # kept for downstream patches
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.sequence import IntermediateTensors

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.models.deepseek_v4 import (
    AscendDeepseekV4ForCausalLM,
    DeepseekV2DecoderLayer,
    DeepseekV4Model,
    DeepseekV4MoE,
    get_spec_layer_idx_from_weight_name,
)
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.triton.mul_add import muls_add_triton
from vllm_ascend.quantization.methods.base import get_moe_num_logical_experts
from vllm_ascend.utils import enable_dsa_cp



# ---------------------------------------------------------------------------
# DeepseekV4MoE.afd_forward
# ---------------------------------------------------------------------------
@torch.compiler.disable
def afd_forward(
    self: DeepseekV4MoE,
    hidden_states: torch.Tensor,
    router_logits: Optional[torch.Tensor] = None,
    group_list: Optional[torch.Tensor] = None,
    dynamic_scales: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    row_idx: Optional[torch.Tensor] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    cam_p2p_ep_name: str = "",
) -> torch.Tensor:
    """FFN-side MoE forward driven by the AFD connector.

    Mirrors ``DeepseekV4MoE.forward`` but delegates the expert computation to
    ``afd_connector.compute_moe`` so that routing tensors produced on the
    attention side can be consumed here. The shared-output merge, scaling and
    sequence-parallel/all-reduce handling match the regular forward path.
    """
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    if self.is_sequence_parallel:
        from vllm.model_executor.models.utils import sequence_parallel_chunk
        hidden_states = sequence_parallel_chunk(hidden_states)

    forward_ctx = get_forward_context()
    afd_connector = forward_ctx.afd_metadata.afd_connector
    fused_moe_out = afd_connector.compute_moe(
        experts=self.experts,
        hidden_states=hidden_states,
        router_logits=router_logits,
        group_list=group_list,
        dynamic_scales=dynamic_scales,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        row_idx=row_idx,
        x_active_mask=x_active_mask,
        cam_p2p_ep_name=cam_p2p_ep_name,
        connector_name=getattr(self, "connector_name", None),
    )

    fused_moe_out_is_tuple = isinstance(fused_moe_out, tuple)
    if fused_moe_out_is_tuple:
        shared_output, final_hidden_states = fused_moe_out
        if self.shared_experts is None:
            assert shared_output is None

        # Fix FP16 overflow (see DeepseekV2DecoderLayer for details).
        if hidden_states.dtype != torch.float16:
            if not self.is_rocm_aiter_moe_enabled:
                if self.shared_experts is not None:
                    assert shared_output is not None
                    final_hidden_states = muls_add_triton(
                        final_hidden_states, shared_output, self.routed_scaling_factor
                    )
                else:
                    final_hidden_states *= self.routed_scaling_factor
        elif self.shared_experts is not None:
            assert shared_output is not None
            final_hidden_states = muls_add_triton(
                shared_output, final_hidden_states, 1.0 / self.routed_scaling_factor
            )
    else:
        # AFD 路径: compute_moe 只返回 routed_out (非 tuple), 需要手动计算
        # shared_experts 并合并, 与 DeepseekV4MoE.forward (非 AFD 路径) 保持
        # 一致。否则缺少 shared_experts 贡献和 routed_scaling_factor, 导致
        # 输出乱码。
        final_hidden_states = fused_moe_out
        if self.shared_experts is not None:
            from vllm_ascend.ascend_forward_context import MoECommType
            from vllm_ascend.utils import shared_expert_dp_enabled
            from vllm.distributed import tensor_model_parallel_all_reduce
            shared_output = self.shared_experts(hidden_states)
            fwd_ctx = get_forward_context()
            moe_comm_type = getattr(fwd_ctx, "moe_comm_type", None)
            if (
                moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2,
                                  MoECommType.FUSED_MC2}
                and not shared_expert_dp_enabled()
            ):
                shared_output = tensor_model_parallel_all_reduce(shared_output)
            if hidden_states.dtype != torch.float16:
                if not self.is_rocm_aiter_moe_enabled:
                    final_hidden_states = muls_add_triton(
                        final_hidden_states, shared_output,
                        self.routed_scaling_factor
                    )
            else:
                final_hidden_states = muls_add_triton(
                    shared_output, final_hidden_states,
                    1.0 / self.routed_scaling_factor
                )

    if self.is_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]
    elif self.tp_size > 1 and fused_moe_out_is_tuple:
        # Legacy tuple outputs are reduced here. Tensor outputs from the
        # upstream MoERunner have already gone through its final reduction.
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states
        )

    return final_hidden_states.view(num_tokens, hidden_dim)


# ---------------------------------------------------------------------------
# AscendFusedMoE.afd_ffn_compute
# ---------------------------------------------------------------------------
@torch.compiler.disable
def afd_ffn_compute(
    self: AscendFusedMoE,
    layer: AscendFusedMoE,
    hidden_states: torch.Tensor,
    router_logits: Optional[torch.Tensor] = None,
    group_list: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    row_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FFN-side MoE expert computation for AFD.

    Mirrors ``AscendFusedMoE.forward_impl``. When ``compute_gate_on_attention``
    is enabled, ``topk_weights`` / ``topk_ids`` are pre-computed on the
    attention side and received via P2P, so ``select_experts`` is skipped.
    When disabled (``topk_ids is None``), the FFN side computes routing
    locally using the gate weight, exactly like ``forward_impl``.
    """
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
    from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl
    from vllm_ascend.ops.fused_moe.moe_runtime_args import (
        build_fused_experts_input)

    moe_comm_method = _EXTRA_CTX.moe_comm_method

    # 与 attention 侧 (forward_m2n) 保持一致：通过 compute_gate_on_attention 配置判断
    # attention 侧: afd_config.compute_gate_on_attention=True → 计算 routing
    # FFN 侧:      compute_gate_on_attention=False → 在此计算 routing
    _fwd_ctx = get_forward_context()
    _afd_connector = getattr(getattr(_fwd_ctx, "afd_metadata", None),
                             "afd_connector", None)
    _afd_config = getattr(_afd_connector, "afd_config", None)
    ffn_side_gating = (
        _afd_config is None
        or not getattr(_afd_config, "compute_gate_on_attention", False)
    )
    if ffn_side_gating:
        assert router_logits is None and topk_ids is None, \
            "FFN-side gating expects router_logits/topk_ids to be None"
        # 精度诊断: hidden_states 维度 (正常路径期望 2D, P2P 接收可能是 3D)
        logger.info(
            "[FFN_GATING_ENTRY] hs dim=%d shape=%s dtype=%s "
            "mean=%.6f std=%.6f",
            hidden_states.dim(), tuple(hidden_states.shape),
            hidden_states.dtype,
            hidden_states.float().mean().item(),
            hidden_states.float().std().item(),
        )
        # 与 shared_forward_impl 保持一致：当 gate 存在预转换的 float32 权重
        # (weight_fp32) 时使用它计算 router_logits, 避免精度损失导致乱码。
        gate = self.gate
        has_fp32 = gate is not None and hasattr(gate, "weight_fp32")
        logger.info(
            "[FFN_GATE] gate is_none=%s has_weight_fp32=%s "
            "weight dtype=%s",
            gate is None, has_fp32,
            gate.weight.dtype if gate is not None else None,
        )
        if has_fp32:
            router_logits = F.linear(hidden_states.float(), gate.weight_fp32)
        else:
            router_logits = F.linear(hidden_states.float(), gate.weight)
        logger.info(
            "[FFN_ROUTER_LOGITS] shape=%s dtype=%s mean=%.6f std=%.6f",
            tuple(router_logits.shape), router_logits.dtype,
            router_logits.float().mean().item(),
            router_logits.float().std().item(),
        )
    else:
        assert topk_ids is not None, \
            "Attention-side gating expects topk_ids to be pre-computed"

    # 1. prepare — handles dispatch / padding (same as forward_impl).
    prepare_output = moe_comm_method.prepare(
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

    # 1b. FFN-side gating: call select_experts with prepared tensors.
    #     This mirrors forward_impl where select_experts runs inside
    #     quant_method.apply AFTER prepare. The resulting topk_ids
    #     naturally matches prepared hidden_states — no pad/split needed.
    if ffn_side_gating:
        # input_ids 已由 recv_attn_output 从 attention 侧通过 P2P 接收并
        # 设置到 forward_context 中, 供 tid2eid (logical→physical expert 映射) 使用。
        # select_experts 内部会从 forward_context.input_ids 读取并自行调用
        # pad_and_split_input_ids / all_gather_input_id_with_dp_group 处理 pad,
        # 因此这里无需再调用 maybe_all_gather_and_maybe_unpad (该 op 在
        # flash_comm_v1_enabled=False 时为 no-op, 调用无效)。
        fwd_ctx = get_forward_context()
        input_ids = getattr(fwd_ctx, "input_ids", None)
        # 与 quant_method.apply 保持一致：使用 num_logical_experts 而非
        # moe_config.num_experts, 避免 tid2eid 映射越界导致乱码。
        num_shared_experts = getattr(self, "n_shared_experts", 0)
        if num_shared_experts is None:
            num_shared_experts = 0
        num_logical_experts = get_moe_num_logical_experts(
            self,
            self.moe_config.num_experts,
            global_redundant_expert_num=self.global_redundant_expert_num,
            num_shared_experts=num_shared_experts,
        )
        logger.info(
            "[FFN_SELECT_INPUT] hs shape=%s router_logits shape=%s "
            "input_ids is_none=%s shape=%s tid2eid is_none=%s shape=%s "
            "num_logical=%d moe_num_experts=%d "
            "global_redundant=%d n_shared=%d",
            tuple(hidden_states.shape), tuple(router_logits.shape),
            input_ids is None,
            tuple(input_ids.shape) if input_ids is not None else None,
            self.tid2eid is None,
            tuple(self.tid2eid.shape) if self.tid2eid is not None else None,
            num_logical_experts, self.moe_config.num_experts,
            self.global_redundant_expert_num, num_shared_experts,
        )
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
            num_experts=num_logical_experts,
            input_ids=input_ids,
            tid2eid=self.tid2eid,
        )
        topk_weights = topk_weights.to(torch.float)
        logger.info(
            "[FFN_SELECT_OUTPUT] topk_ids shape=%s first16=%s "
            "topk_weights shape=%s mean=%.6f",
            tuple(topk_ids.shape),
            topk_ids.flatten().tolist()[:16],
            tuple(topk_weights.shape),
            topk_weights.mean().item(),
        )

    # 1c. Attention 侧预计算的 topk_ids 需要 pad/split 对齐。
    # FFN 侧 gating (ffn_side_gating=True) 时,select_experts 在 prepare 之后
    # 调用,topk_ids 已与 prepared hidden_states 形状一致,跳过此块。
    # 仅当 compute_gate_on_attention=True (attention 侧算好 topk_ids) 时执行:
    #   topk_ids 在 Attention 侧、FFN 的 prepare 之前计算,通过 P2P 传输,
    #   仍保持原始 num_tokens 行数,需同步 prepare 的 pad/split。
    prepare_finalize = moe_comm_method.prepare_finalize
    if (not ffn_side_gating
            and not _EXTRA_CTX.flash_comm_v1_enabled
            and not self.enable_shared_expert_dp):
        target_pad_length = _EXTRA_CTX.padded_num_tokens
        pad_size = target_pad_length - prepare_finalize.num_tokens
        if pad_size > 0:
            topk_ids = F.pad(topk_ids, (0, 0, 0, pad_size))
            topk_weights = F.pad(topk_weights, (0, 0, 0, pad_size))
        if prepare_finalize.tp_size > 1:
            topk_ids = torch.tensor_split(
                topk_ids, prepare_finalize.tp_size, dim=0)[prepare_finalize.tp_rank]
            topk_weights = torch.tensor_split(
                topk_weights, prepare_finalize.tp_size, dim=0)[prepare_finalize.tp_rank]

    fused_scale_flag = (
        _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2
        and get_ascend_config().enable_fused_mc2 == 1
    )
    if self.dynamic_eplb:
        w1 = layer.w13_weight_list
        w1_scale = (layer.fused_w1_scale_list if fused_scale_flag
                    else layer.w13_weight_scale_fp32_list)
        w2 = layer.w2_weight_list
        w2_scale = (layer.fused_w2_scale_list if fused_scale_flag
                    else layer.w2_weight_scale_list)
    else:
        w1 = [layer.w13_weight]
        w1_scale = ([layer.fused_w1_scale] if fused_scale_flag
                    else [layer.w13_weight_scale_fp32])
        w2 = [layer.w2_weight]
        w2_scale = ([layer.fused_w2_scale] if fused_scale_flag
                    else [layer.w2_weight_scale])
    w1_scale_bias = ([torch.tensor([], dtype=torch.float32)]
                     if fused_scale_flag else None)
    w2_scale_bias = ([torch.tensor([], dtype=torch.float32)]
                     if fused_scale_flag else None)

    # 3. Build MoEFusedExpertsInput with pre-computed topk (skip select_experts).
    fused_experts_input = build_fused_experts_input(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1=w1,
        w2=w2,
        quant_type=self.quant_type,
        dynamic_eplb=self.dynamic_eplb,
        expert_map=self._expert_map,
        global_redundant_expert_num=self.global_redundant_expert_num,
        mc2_mask=mc2_mask,
        apply_router_weight_on_input=self.apply_router_weight_on_input,
        log2phy=self.log2phy,
        pertoken_scale=pertoken_scale,
        activation=self.activation,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_scale_bias=w1_scale_bias,
        w2_scale_bias=w2_scale_bias,
        swiglu_limit=self.swiglu_limit,
    )

    # 4. fused_experts — run expert GEMM with pre-computed routing tensors.
    fused_experts_results = moe_comm_method.fused_experts(
        fused_experts_input=fused_experts_input)

    # 诊断: MoE 专家计算输出 (dispatch+MLP+combine 之后, finalize 之前)
    _moe_out = fused_experts_results.routed_out
    logger.info(
        "[FFN_MOE_OUT] routed_out shape=%s dtype=%s mean=%.6f std=%.6f "
        "min=%.6f max=%.6f has_nan=%s has_inf=%s",
        tuple(_moe_out.shape), _moe_out.dtype,
        _moe_out.float().mean().item(),
        _moe_out.float().std().item(),
        _moe_out.float().min().item(),
        _moe_out.float().max().item(),
        torch.isnan(_moe_out).any().item(),
        torch.isinf(_moe_out).any().item(),
    )

    # 5. finalize — all-reduce / unpad.
    routed_out = moe_comm_method.finalize(
        hidden_states=fused_experts_results.routed_out,
        reduce_results=isinstance(moe_comm_method, AllGatherCommImpl),
        padded_hidden_states_shape=padded_hidden_states_shape,
    )

    # 诊断: finalize 输出 (all-reduce / unpad 之后, 返回给 afd_forward 之前)
    logger.info(
        "[FFN_FINALIZE_OUT] routed_out shape=%s dtype=%s mean=%.6f std=%.6f "
        "min=%.6f max=%.6f has_nan=%s has_inf=%s",
        tuple(routed_out.shape), routed_out.dtype,
        routed_out.float().mean().item(),
        routed_out.float().std().item(),
        routed_out.float().min().item(),
        routed_out.float().max().item(),
        torch.isnan(routed_out).any().item(),
        torch.isinf(routed_out).any().item(),
    )

    return routed_out


# ---------------------------------------------------------------------------
# DeepseekV2DecoderLayer.compute_attn_output / compute_ffn_output
# ---------------------------------------------------------------------------
def compute_attn_output(
    self: DeepseekV2DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    llama_4_scaling: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention half of ``DeepseekV2DecoderLayer.forward``.

    Runs ``hc_pre -> input_layernorm -> self_attn -> hc_post`` and returns the
    attention output that will be sent to the FFN worker.
    """
    residual = hidden_states.clone()
    hidden_states, post, comb = self.hc_pre(
        hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
    )
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
        llama_4_scaling=llama_4_scaling,
    )
    hidden_states = self.hc_post(hidden_states, residual, post, comb)
    return hidden_states


def compute_ffn_output(
    self: DeepseekV2DecoderLayer,
    layer_idx: int,
    hidden_states: torch.Tensor,
    router_logits: Optional[torch.Tensor] = None,
    group_list: Optional[torch.Tensor] = None,
    dynamic_scales: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    row_idx: Optional[torch.Tensor] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    cam_p2p_ep_name: str = "",
) -> torch.Tensor:
    """FFN half of ``DeepseekV2DecoderLayer.forward``.

    Runs ``hc_pre -> post_attention_layernorm -> mlp.afd_forward -> hc_post``
    using the routing tensors received from the attention worker.
    """
    # 诊断: FFN 侧 hc_pre 前 (从 attention 侧 P2P 接收的 3D 张量)
    logger.info(
        "[FFN_HC_PRE_IN] layer=%d hs dim=%d shape=%s dtype=%s "
        "mean=%.6f std=%.6f min=%.6f max=%.6f has_nan=%s",
        layer_idx, hidden_states.dim(), tuple(hidden_states.shape),
        hidden_states.dtype,
        hidden_states.float().mean().item(),
        hidden_states.float().std().item(),
        hidden_states.float().min().item(),
        hidden_states.float().max().item(),
        torch.isnan(hidden_states).any().item(),
    )
    residual = hidden_states.clone()
    hidden_states, post, comb = self.hc_pre(
        hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
    )
    # 诊断: FFN 侧 hc_pre 后 (3D→2D fold), 即将进入 post_attention_layernorm + MoE
    logger.info(
        "[FFN_HC_PRE_OUT] layer=%d hs dim=%d shape=%s mean=%.6f std=%.6f "
        "min=%.6f max=%.6f has_nan=%s | post shape=%s comb shape=%s",
        layer_idx, hidden_states.dim(), tuple(hidden_states.shape),
        hidden_states.float().mean().item(),
        hidden_states.float().std().item(),
        hidden_states.float().min().item(),
        hidden_states.float().max().item(),
        torch.isnan(hidden_states).any().item(),
        tuple(post.shape) if post is not None else None,
        tuple(comb.shape) if comb is not None else None,
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp.afd_forward(
        hidden_states,
        router_logits=router_logits,
        group_list=group_list,
        dynamic_scales=dynamic_scales,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        row_idx=row_idx,
        x_active_mask=x_active_mask,
        cam_p2p_ep_name=cam_p2p_ep_name,
    )
    # 诊断: FFN 侧 hc_post 前 (MoE 输出 2D), 即将进入 hc_post (2D→3D unfold)
    logger.info(
        "[FFN_HC_POST_IN] layer=%d hs dim=%d shape=%s mean=%.6f std=%.6f "
        "min=%.6f max=%.6f has_nan=%s",
        layer_idx, hidden_states.dim(), tuple(hidden_states.shape),
        hidden_states.float().mean().item(),
        hidden_states.float().std().item(),
        hidden_states.float().min().item(),
        hidden_states.float().max().item(),
        torch.isnan(hidden_states).any().item(),
    )
    hidden_states = self.hc_post(hidden_states, residual, post, comb)
    # 诊断: FFN 侧 hc_post 后 (3D unfold), 即将通过 P2P 回传给 attention 侧
    logger.info(
        "[FFN_HC_POST_OUT] layer=%d hs dim=%d shape=%s mean=%.6f std=%.6f "
        "min=%.6f max=%.6f has_nan=%s",
        layer_idx, hidden_states.dim(), tuple(hidden_states.shape),
        hidden_states.float().mean().item(),
        hidden_states.float().std().item(),
        hidden_states.float().min().item(),
        hidden_states.float().max().item(),
        torch.isnan(hidden_states).any().item(),
    )
    return hidden_states


# ---------------------------------------------------------------------------
# DeepseekV4Model.forward_m2n / forward (AFD dispatch)
# ---------------------------------------------------------------------------
@torch.compiler.disable
def forward_m2n(
    self: DeepseekV4Model,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    positions: torch.Tensor,
    afd_metadata: Any,
    llama_4_scaling: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Attention-side layer loop that ships intermediates to the FFN worker.

    For each decoder layer:
      1. (layer > 0) receive the previous layer's FFN output,
      2. compute the attention output,
      3. optionally compute the router gate + ``select_experts`` when
         ``compute_gate_on_attention`` is enabled,
      4. send the attention output (and routing tensors) to the FFN worker.
    After the loop, the final FFN output is received.
    """
    afd_connector = afd_metadata.afd_connector
    # NOTE: avoid calling get_current_vllm_config() here because torch dynamo
    # compilation runs outside set_current_vllm_config() context. The connector
    # already caches afd_config at init time.
    afd_config = getattr(afd_connector, "afd_config", None)

    for layer in islice(self.layers, self.start_layer, self.end_layer):
        if layer.layer_idx > 0:
            hidden_states = afd_connector.recv_ffn_output(
                hidden_states=hidden_states, metadata=None
            )

        hidden_states = layer.compute_attn_output(
            positions, hidden_states, residual, llama_4_scaling
        )

        router_logits = None
        topk_weights = None
        topk_ids = None
        hidden_states_2d = None
        if afd_config is not None and afd_config.compute_gate_on_attention:
            # moe_gating_top_k requires 2D router_logits. Use a local 2D view
            # for gate projection only — do NOT reshape hidden_states itself,
            # because the original shape is sent to FFN where hc_pre expects
            # 3D/4D input.
            #
            # compute_attn_output returns 3D (bs, hc, d) after hc_post. On the
            # FFN side, hc_pre folds hc → 1 producing (bs, d). Routing must be
            # computed from the same token count (bs, not bs*hc), otherwise
            # topk_ids has hc times more rows than post-hc_pre hidden_states,
            # causing "xShape's dim0 not equal to expertIdShape's dim0" in
            # npu_moe_distribute_dispatch_v2. Average over hc (dim=-2) to
            # approximate the post-hc_pre representation.
            hidden_dim = hidden_states.shape[-1]
            if hidden_states.dim() >= 3:
                hidden_states_2d = hidden_states.mean(dim=-2).reshape(
                    -1, hidden_dim)
            else:
                hidden_states_2d = hidden_states.view(-1, hidden_dim)
            router_logits = F.linear(hidden_states_2d.float(),
                                     layer.mlp.gate.weight)
            topk_weights, topk_ids = afd_connector.select_experts(
                hidden_states=hidden_states_2d,
                router_logits=router_logits,
                top_k=layer.mlp.experts.top_k,
                use_grouped_topk=True,
                renormalize=getattr(self.config, "norm_topk_prob", True),
                topk_group=getattr(self.config, "topk_group", 1),
                num_expert_group=getattr(self.config, "n_group", 1),
                e_score_correction_bias=layer.mlp.gate.e_score_correction_bias,
            )
            topk_weights = topk_weights.to(torch.float)

        # When compute_gate_on_attention=False, the FFN side computes routing
        # locally and needs input_ids for tid2eid (logical→physical expert
        # mapping). Send input_ids via P2P alongside hidden_states.
        p2p_input_ids = None
        if afd_config is not None and not afd_config.compute_gate_on_attention:
            p2p_input_ids = getattr(get_forward_context(), "input_ids", None)

        # P2P 传输完整性诊断: 记录发送前的张量状态, 与 FFN 侧 recv 后对比
        logger.info(
            "[ATTN_SEND_PRE] layer=%d hs dim=%d shape=%s dtype=%s "
            "mean=%.6f std=%.6f",
            layer.layer_idx,
            hidden_states.dim(), tuple(hidden_states.shape),
            hidden_states.dtype,
            hidden_states.float().mean().item(),
            hidden_states.float().std().item(),
        )
        if p2p_input_ids is not None:
            logger.info(
                "[ATTN_SEND_INPUT_IDS] shape=%s dtype=%s first8=%s",
                tuple(p2p_input_ids.shape), p2p_input_ids.dtype,
                p2p_input_ids.flatten().tolist()[:8],
            )
        else:
            logger.info("[ATTN_SEND_INPUT_IDS] is_none=True")
        if router_logits is not None:
            logger.info(
                "[ATTN_SEND_ROUTER] router_logits shape=%s dtype=%s "
                "mean=%.6f std=%.6f",
                tuple(router_logits.shape), router_logits.dtype,
                router_logits.float().mean().item(),
                router_logits.float().std().item(),
            )
        if topk_ids is not None:
            logger.info(
                "[ATTN_SEND_TOPK] topk_ids shape=%s first16=%s "
                "topk_weights shape=%s mean=%.6f",
                tuple(topk_ids.shape), topk_ids.flatten().tolist()[:16],
                tuple(topk_weights.shape), topk_weights.mean().item(),
            )

        afd_connector.send_attn_output(
            hidden_states=hidden_states,
            metadata=None,
            router_logits=router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            input_ids=p2p_input_ids,
        )
        logger.info("[ATTN_SEND_DONE] layer=%d send_attn_output completed",
                     layer.layer_idx)

    hidden_states = afd_connector.recv_ffn_output(
        hidden_states=hidden_states, metadata=afd_metadata
    )
    return hidden_states, residual


def afd_model_forward(
    self: DeepseekV4Model,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: Optional[IntermediateTensors],
    inputs_embeds: Optional[torch.Tensor] = None,
) -> torch.Tensor | IntermediateTensors:
    """Replacement ``DeepseekV4Model.forward`` with an AFD dispatch branch.

    When ``afd_metadata`` is present in the forward context, the decoder layer
    loop is replaced by ``forward_m2n``; otherwise the original layer-by-layer
    path is used. The MTP hidden-state stash and ``hc_head``/``norm`` epilogue
    are preserved unchanged.
    """
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = None

    # Compute llama 4 scaling once per forward pass if enabled
    llama_4_scaling_config = None
    llama_4_scaling: torch.Tensor | None
    if llama_4_scaling_config is not None:
        from vllm.model_executor.models.deepseek_v2 import _get_llama_4_scaling

        llama_4_scaling = _get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config[
                "original_max_position_embeddings"
            ],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    if get_pp_group().is_first_rank:
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

    forward_ctx = get_forward_context()
    afd_metadata = forward_ctx.afd_metadata if forward_ctx is not None else None

    if afd_metadata is not None:
        hidden_states, residual = self.forward_m2n(
            hidden_states, residual, positions, afd_metadata, llama_4_scaling
        )
    else:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions, hidden_states, residual, llama_4_scaling
            )

    # Stash pre-hc_head residual for the MTP draft (captured copy_).
    if forward_ctx is not None and forward_ctx.flash_comm_v1_enabled:
        h_states_flat = tensor_model_parallel_all_gather(
            hidden_states.flatten(1), dim=0
        )
        pad_size = forward_ctx.pad_size
        if pad_size > 0:
            h_states_flat = h_states_flat[:-pad_size]
        num_tokens = h_states_flat.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(h_states_flat)
    else:
        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states})

    hidden_states = self.hc_head(
        hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
    )
    hidden_states = self.norm(hidden_states)
    return hidden_states


# ---------------------------------------------------------------------------
# AscendDeepseekV4ForCausalLM AFD helpers
# ---------------------------------------------------------------------------
def is_moe_weight(self: AscendDeepseekV4ForCausalLM, name: str) -> bool:
    """Return True for expert / shared-expert / router-gate weights."""
    if (
        "shared_experts" in name
        or "experts" in name
        or "gate" in name
        or "up" in name
        or "down" in name
    ):
        return True
    return False


def is_common_weight(self: AscendDeepseekV4ForCausalLM, name: str) -> bool:
    """Return True for weights required by both AFD roles.

    Layernorms and hc_* structural parameters are needed on both sides because
    ``hc_pre``/``hc_post`` are invoked in both ``compute_attn_output`` and
    ``compute_ffn_output``.
    """
    if (
        "lm_head" in name
        or "model.norm.weight" in name
        or "embed_tokens" in name
        or "input_layernorm" in name
        or "post_attention_layernorm" in name
        or "hc_" in name
    ):
        return True
    return False


def model_compute_ffn_output(
    self: AscendDeepseekV4ForCausalLM,
    hidden_states: torch.Tensor,
    layer_idx: int,
    router_logits: Optional[torch.Tensor] = None,
    group_list: Optional[torch.Tensor] = None,
    dynamic_scales: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    row_idx: Optional[torch.Tensor] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    cam_p2p_ep_name: str = "",
) -> torch.Tensor:
    """Model-level FFN entry point used by ``NPUFFNModelRunner``."""
    if self.afd_config is not None and self.afd_config.compute_gate_on_attention:
        hidden_states = self.model.layers[layer_idx].compute_ffn_output(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            router_logits=router_logits,
            group_list=group_list,
            dynamic_scales=dynamic_scales,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            row_idx=row_idx,
            x_active_mask=x_active_mask,
            cam_p2p_ep_name=cam_p2p_ep_name,
        )
    else:
        hidden_states = self.model.layers[layer_idx].compute_ffn_output(
            layer_idx=layer_idx, hidden_states=hidden_states
        )
    return hidden_states


def afd_load_weights(
    self: AscendDeepseekV4ForCausalLM, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    """AFD-aware ``load_weights``.

    * attention role: skip MoE expert weights; load the router gate when
      ``compute_gate_on_attention`` is enabled.
    * ffn role: skip non-MoE / non-common weights; skip the router gate when
      ``compute_gate_on_attention`` is enabled (attention side owns it).
    """
    rocm_aiter_moe_shared_expert_enabled = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
    rocm_aiter_moe_shared_expert_enabled = getattr(get_ascend_config(), "mix_placement", False)
    stacked_params_mapping = [
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        self.model,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.n_routed_experts
        + (self.config.n_shared_experts if rocm_aiter_moe_shared_expert_enabled else 0),
        num_redundant_experts=self.num_redundant_experts,
    )

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    heads_per_rank = self.config.num_attention_heads // tp_size
    head_start = tp_rank * heads_per_rank

    for name, loaded_weight in weights:
        spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
        if spec_layer is not None:
            continue  # skip spec decode layers for main model

        if not name.startswith("model"):
            name = f"model.{name}"

        if ".w1." in name:
            name = name.replace(".w1.", ".gate_proj.")
        if ".w2." in name:
            name = name.replace(".w2.", ".down_proj.")
        if ".w3." in name:
            name = name.replace(".w3.", ".up_proj.")

        if "model.head." in name and "model.lm_head." not in name:
            name = name.replace("model.head.", "lm_head.")
        if "model.lm_head." in name:
            name = name.replace("model.lm_head.", "lm_head.")
        if "embed." in name and "embed_token." not in name:
            name = name.replace("embed.", "embed_tokens.")
        if "attn" in name and "self_attn" not in name:
            name = name.replace(".attn.", ".self_attn.")
        if ".ffn." in name:
            name = name.replace(".ffn.", ".mlp.")
        if ".ffn_norm." in name:
            name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
        if ".attn_norm." in name:
            name = name.replace(".attn_norm.", ".input_layernorm.")
        if name.endswith(".scale"):
            name = name.replace(".scale", ".weight_scale")

        if "rotary_emb.inv_freq" in name:
            continue
        if ".gate.bias" in name:
            name = name.replace(".gate.bias", ".gate.e_score_correction_bias")

        # ---------------------------------------------------------------
        # AFD role filtering
        # ---------------------------------------------------------------
        # Attention role: load the router gate when compute_gate_on_attention
        # is enabled (the gate lives at ``mlp.gate.*`` in this model).
        if (
            self.afd_role == "attention"
            and self.afd_config is not None
            and self.afd_config.compute_gate_on_attention
            and "mlp.gate." in name
        ):
            if not is_pp_missing_parameter(name, self) and name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
            continue

        # Attention role: skip MoE expert weights.
        if self.afd_role == "attention" and self.is_moe_weight(name):
            continue

        # FFN role: skip the router gate when compute_gate_on_attention is
        # enabled (the attention side owns it).
        if (
            self.afd_role == "ffn"
            and self.afd_config is not None
            and self.afd_config.compute_gate_on_attention
            and "mlp.gate." in name
        ):
            continue
        # ---------------------------------------------------------------

        if "sink" in name:
            if is_pp_missing_parameter(name, self):
                continue
            param = params_dict[name]
            if enable_dsa_cp():
                param.data.copy_(loaded_weight)
            else:
                narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow_weight)
            loaded_params.add(name)
            continue

        is_fusion_moe_shared_experts_layer = (
            rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)
        )

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            if is_fusion_moe_shared_experts_layer:
                continue
            name_mapped = name.replace(weight_name, param_name)

            if (param_name == "fused_qkv_a_proj") and name_mapped not in params_dict:
                continue
            else:
                name = name_mapped
            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            is_expert_weight = False

            num_chunks = 1
            if is_fusion_moe_shared_experts_layer:
                num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                split_dim = 1 if "down_proj.weight" in name else 0
                total = loaded_weight.shape[split_dim]
                assert total % num_chunks == 0, (
                    f"Shared expert weight dim {total} not divisible by num_chunks {num_chunks}"
                )
                chunk_size = total // num_chunks

            for j in range(num_chunks):
                chunk_name = name
                weight_to_load = loaded_weight

                if is_fusion_moe_shared_experts_layer:
                    if split_dim == 0:
                        weight_to_load = loaded_weight[j * chunk_size : (j + 1) * chunk_size, :]
                    else:
                        weight_to_load = loaded_weight[:, j * chunk_size : (j + 1) * chunk_size]
                    chunk_name = name.replace(
                        "mlp.shared_experts",
                        f"mlp.experts.{self.config.n_routed_experts + j}",
                    )

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in chunk_name:
                        continue

                    is_expert_weight = True
                    name_mapped = chunk_name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                    success = weight_loader(
                        param,
                        weight_to_load,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        if not is_fusion_moe_shared_experts_layer:
                            name = name_mapped
                        else:
                            loaded_params.add(name_mapped)
                        break
                else:
                    # FFN role: skip non-MoE, non-common weights (e.g. attn).
                    if (
                        self.afd_role == "ffn"
                        and not self.is_moe_weight(name)
                        and not self.is_common_weight(name)
                    ):
                        continue
                    if is_expert_weight:
                        continue
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)

    return loaded_params


# ---------------------------------------------------------------------------
# Apply monkey-patches
# ---------------------------------------------------------------------------
# DeepseekV4MoE gains an afd_forward entry point for the FFN worker.
DeepseekV4MoE.afd_forward = afd_forward  # type: ignore[assignment]

# AscendFusedMoE gains an afd_ffn_compute entry point that runs expert GEMM
# with pre-computed routing tensors received from the attention side.
AscendFusedMoE.afd_ffn_compute = afd_ffn_compute  # type: ignore[assignment]

# Decoder layer split into attention / FFN halves.
DeepseekV2DecoderLayer.compute_attn_output = compute_attn_output  # type: ignore[assignment]
DeepseekV2DecoderLayer.compute_ffn_output = compute_ffn_output  # type: ignore[assignment]

# Model-level AFD dispatch.
DeepseekV4Model.forward_m2n = forward_m2n  # type: ignore[assignment]
DeepseekV4Model.forward = afd_model_forward  # type: ignore[assignment]

# CausalLM AFD helpers + weight loading.
_orig_deepseekv4_init = AscendDeepseekV4ForCausalLM.__init__


def _afd_deepseekv4_init(self: AscendDeepseekV4ForCausalLM, *, vllm_config, prefix: str = ""):
    _orig_deepseekv4_init(self, vllm_config=vllm_config, prefix=prefix)
    self.afd_config = getattr(vllm_config, "afd_config", None)
    self.afd_role = (
        self.afd_config.afd_role if self.afd_config is not None else None
    )


AscendDeepseekV4ForCausalLM.__init__ = _afd_deepseekv4_init  # type: ignore[assignment]
AscendDeepseekV4ForCausalLM.is_moe_weight = is_moe_weight  # type: ignore[assignment]
AscendDeepseekV4ForCausalLM.is_common_weight = is_common_weight  # type: ignore[assignment]
AscendDeepseekV4ForCausalLM.compute_ffn_output = model_compute_ffn_output  # type: ignore[assignment]
AscendDeepseekV4ForCausalLM.load_weights = afd_load_weights  # type: ignore[assignment]
