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
#
"""W4A4 dynamic fused-MoE scheme backed by the fused "mega" kernel.

The routed experts are INT4 (per-output-channel symmetric), activations are
INT4 dynamic per-token, and a block-diagonal Hadamard rotation -- baked into the
``gate_up`` weights offline at quantization time -- is applied online to the
activations inside the kernel (Stage 1), so the two cancel. The whole expert
path (Hadamard -> quant+scatter -> gate_up -> SwiGLU+requant -> down ->
combine) runs in a single launch; see ``vllm_ascend.ops.mega_moe_w4a4``.

Weight loading reuses the W4A8 MoE parameter declarations (int8-stored INT4
``[E, 2I, H]`` / ``[E, H, I]`` + per-channel fp32 scales); only the post-load
repack and the forward differ, so this subclasses the W4A8 MoE method.
"""

from collections.abc import Callable

import torch

from vllm_ascend.ops import mega_moe_w4a4 as _mega
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult

from .base import get_moe_num_logical_experts
from .registry import register_scheme
from .w4a8 import AscendW4A8DynamicFusedMoEMethod


@register_scheme("W4A4_DYNAMIC", "moe")
class AscendW4A4DynamicFusedMoEMethod(AscendW4A8DynamicFusedMoEMethod):
    """W4A4 (INT4 weight + INT4 activation) MoE via the fused mega kernel.

    Inherits the W4A8 weight-parameter layout; forces per-channel weights
    (``group_size == 0``) and overrides the post-load repack + forward.
    """

    # No dedicated QuantType enum for W4A4; the scheme is selected by the
    # ("W4A4_DYNAMIC", "moe") registry key from the checkpoint's quant config.
    # quant_type is only read by the parent's apply(), which this overrides.

    # The fused kernel sorts the global topk_ids against the full local expert set
    # and returns only routed_out, so it supports neither expert parallelism nor
    # the dynamic-EPLB bookkeeping (which asserts expert_tokens). Reject both
    # rather than silently producing wrong results (see apply / __init__).
    supports_eplb = False

    def __init__(self):
        super().__init__()
        # W4A4 checkpoints quantize weights per output channel.
        self.group_size = 0
        self.is_per_channel_weight = True
        # The parent __init__ sets self.supports_eplb = True; override it (the class
        # attribute alone is shadowed by that instance assignment).
        self.supports_eplb = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack the loaded INT4 expert weights into the FRACTAL_NZ int4 layout
        the cube matmuls consume, and keep per-channel scales as fp32 (the
        kernel uint64-packs them once on first use). The vendor int8 buffers are
        freed -- only the mega kernel runs.

        The offline block-diagonal Hadamard is already baked into ``gate_up``;
        the kernel's Stage-1 online Hadamard cancels it, so nothing is rotated
        here.
        """
        # Loaded layout: w13_weight [E, 2I, H] int8, scale [E, 2I, 1];
        #                w2_weight  [E, H,  I] int8, scale [E, H,  1].
        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data
        E = w13.shape[0]
        device = w13.device

        # Transpose to K-major (kernel wants K first), then pack to FRACTAL_NZ int4.
        # gate_up: [E, 2I, H] -> [E, H, 2I]   (K=H, N=2I)
        # down:    [E, H,  I] -> [E, I, H]    (K=I, N=H)
        w13_kn = w13.transpose(1, 2).contiguous()
        w2_kn = w2.transpose(1, 2).contiguous()
        # The mega kernel is compiled with fixed per-rank H/I (KERNEL_H_DIM/I_DIM) and
        # assumes the UNPACKED expert layout (gate_up N == 2*I, down N == H). A
        # different model/TP size, or a packed ModelSlim layout (quant_description
        # version 1.0.0 packs gate_up as [E,I,H] / down as [E,H/2,I]), would pass a
        # K-only check yet leave the kernel indexing with the wrong N stride -> the
        # up half uninitialized and the down matmul reading past the weight. Validate
        # both K (shape[1]) and N (shape[2]) and fail fast.
        H_dim, I_dim = w13_kn.shape[1], w2_kn.shape[1]
        n_gu, n_dn = w13_kn.shape[2], w2_kn.shape[2]
        if (H_dim, I_dim) != (_mega.KERNEL_H_DIM, _mega.KERNEL_I_DIM) or n_gu != 2 * I_dim or n_dn != H_dim:
            raise NotImplementedError(
                "W4A4 mega kernel expects the unpacked per-rank layout gate_up "
                f"[E, H, 2I] and down [E, I, H] with H={_mega.KERNEL_H_DIM}, "
                f"I={_mega.KERNEL_I_DIM} (Qwen3.x-MoE at TP=4); got gate_up K={H_dim} "
                f"N={n_gu} (want N={2 * I_dim}), down K={I_dim} N={n_dn} (want N={H_dim}). "
                "A packed ModelSlim layout or a different model/TP size is not supported."
            )
        layer.w13_nz = _mega.pack_nz_int4(w13_kn)
        layer.w2_nz = _mega.pack_nz_int4(w2_kn)
        layer.w13_scale_mega = layer.w13_weight_scale.data.squeeze(-1).to(torch.float32).contiguous()
        layer.w2_scale_mega = layer.w2_weight_scale.data.squeeze(-1).to(torch.float32).contiguous()
        layer.mega_dims = (E, w13_kn.shape[1], w2_kn.shape[1], w13_kn.shape[2])  # (E, H, I, 2I)

        # Free the vendor int8 copies; keep tiny placeholders so vLLM parameter
        # introspection (e.g. the expert-count ``shape[0]`` lookup) still works.
        layer.w13_weight.data = torch.empty(E, 1, 1, dtype=torch.int8, device=device)
        layer.w2_weight.data = torch.empty(E, 1, 1, dtype=torch.int8, device=device)
        layer.w13_weight_scale.data = torch.empty(E, 1, dtype=torch.float32, device=device)
        layer.w2_weight_scale.data = torch.empty(E, 1, dtype=torch.float32, device=device)
        import gc

        gc.collect()
        if torch.npu.is_available():
            torch.npu.empty_cache()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: torch.Tensor | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: torch.Tensor | None = None,
    ) -> FusedExpertsResult:
        # The kernel sorts the global topk_ids against the full local expert set;
        # it does not consume expert-parallel routing metadata. Under EP/EPLB,
        # tokens routed to non-local experts would be dropped or combined with the wrong expert, so
        # reject it explicitly rather than return wrong results.
        if expert_map is not None or log2phy is not None:
            raise NotImplementedError(
                "W4A4 mega kernel does not support expert parallelism / EPLB "
                "(expert_map/log2phy). Serve with expert parallelism disabled (TP only)."
            )
        # apply_router_weight_on_input pre-scales the hidden states by the routing weight
        # BEFORE the expert MLP (the top-1 dispatcher path). The kernel applies topk_weights
        # only in the final combine, so honoring this flag would require MLP(x*w) but the
        # kernel computes MLP(x)*w — different for the nonlinear SwiGLU experts. Qwen3.x-MoE
        # is top-8 (flag stays False); reject rather than silently produce wrong output.
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "W4A4 mega kernel does not support apply_router_weight_on_input=True "
                "(pre-scaling hidden states before the expert MLP). It applies the routing "
                "weight in the combine instead; the two differ for SwiGLU experts."
            )
        T, H = x.shape[0], x.shape[-1]
        x_2d = x.view(T, H) if x.dim() > 2 else x
        outer_dtype = x_2d.dtype
        # The INT4 expert path is fp16-only on 910B; the kernel consumes x as half*.
        # bf16 is the validated production dtype (Qwen3.x linear-attn requires a bf16 model),
        # cast to fp16 here — lossless in effect since activations are int4 dynamic-quantized
        # in the kernel. Reject any other dtype (e.g. fp32): it's untested and would be
        # reinterpreted as fp16 garbage. Matches this method's fail-fast stance on
        # unsupported configs (EP/EPLB, packed layout, wrong dims).
        if outer_dtype == torch.bfloat16:
            x_2d = x_2d.to(torch.float16)
        elif outer_dtype != torch.float16:
            raise NotImplementedError(
                f"W4A4 mega MoE supports fp16/bf16 hidden states; got {outer_dtype}. "
                "Serve with dtype=bfloat16 (the validated Qwen3.x config) or float16."
            )

        num_shared = getattr(layer, "n_shared_experts", 0) or 0
        num_logical_experts = get_moe_num_logical_experts(
            layer,
            num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=num_shared,
        )
        topk_weights, topk_ids = select_experts(
            hidden_states=x_2d,
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
            tid2eid=tid2eid,
        )

        E, kgu, kdn, n_gu = layer.mega_dims  # (E, H, I, 2I)
        if enable_force_load_balance and E > 0:
            rnd = torch.rand(topk_ids.shape[0], E, device=topk_ids.device)
            topk_ids = torch.argsort(rnd, dim=1)[:, : topk_ids.shape[1]].to(topk_ids.dtype)

        group_list, expanded_row_idx, sort_idx = _mega.routing_prep(topk_ids.to(torch.int32), E)
        out = _mega.mega_moe_forward(
            x_2d,
            layer.w13_nz,
            layer.w13_scale_mega,
            layer.w2_nz,
            layer.w2_scale_mega,
            group_list,
            expanded_row_idx,
            sort_idx,
            topk_weights.to(torch.float16),
            top_k=top_k,
            H=kgu,
            i_dim=kdn,
            n_gu=n_gu,
        ).clone()  # detach from the reused workspace

        if outer_dtype != torch.float16:
            out = out.to(outer_dtype)
        out = out.view_as(x) if x.dim() > 2 else out
        return FusedExpertsResult(routed_out=out)
