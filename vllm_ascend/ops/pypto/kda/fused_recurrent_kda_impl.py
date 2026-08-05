# Integrated production kernel for fused_recurrent_kda (consolidated from modules/fused_recurrent_kda_module1_impl.py).
# Exports fused_recurrent_kda_wrapper. Same kernel logic as the final staged module.
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# =============================================================================
# fused_recurrent_kda_impl.py
# State-maintaining recurrent Key-Delta-Attention (decode / recurrent mode).
#
# Token-by-token sequential recurrence maintaining fp32 state matrix S [H, V, K]:
#     S = S * exp(g)                                # gate decay
#     S += outer(beta * k, v - (k . S))            # KDA delta update
#     o = q . S                                    # output
#
# Supports varlen (cu_seqlens), inplace state (ssm_state_indices + NULL slot 0),
# and in-kernel q/k L2norm.
#
# Production reference: models/qwen3_next/gated_delta_rule_impl.py
# (structurally isomorphic recurrent delta-rule kernel — single fused @jit
# + in-kernel pypto.loop(seq -> head[parallel] -> token[step=1]) + persistent
# state buffer + [:] / assemble writeback).
#
# Layers A-L mapping:
#   A  docstring (this block)
#   B  (N/A — reference_l2norm lives in golden)
#   C  (N/A — golden in separate file)
#   D  Constants: D=128, K=128, V=128
#   E  (N/A — forward-only inference kernel)
#   F  (N/A — no backward)
#   G  (N/A — layout adaptation handled in Layer K wrapper)
#   H  operations inlined directly in kernel body (INCORE_2)
#   I  Kernel impl (inlined in @jit body — all pypto.loop)
#   J  @pypto.frontend.jit entry (factory closure pattern)
#   K  fused_recurrent_kda_impl (host wrapper, torch-only)
#   L  (N/A — test produced by verifier)
# =============================================================================

import math
import pypto
import torch
import torch_npu  # noqa: F401  required for NPU device init
from torch._dynamo import allow_in_graph
from torch._subclasses.fake_tensor import FakeTensor


# =============================================================================
# Layer D — Constants (compile-time)
# =============================================================================

# Dynamic axes (DESIGN.md front matter dynamic_axes: ["T", "S"]).
# T = token axis (q/k/v/g/beta/o first dim), S = state slot axis (state_buf first dim).


# =============================================================================
# Layer H — operations inlined directly in kernel body (INCORE_2: eliminate
# function-call boundaries for better compiler scheduling across stages).
# =============================================================================


# =============================================================================
# Layer I + J — Kernel implementation + JIT entry (factory closure pattern)
# =============================================================================
# Following the production reference gated_delta_rule_impl.py, closure params
# (H, D, in_dtype, scale, eps, flags) are captured via a factory function that
# returns the @pypto.frontend.jit kernel. The kernel body is inlined directly
# (no separate _kernel_impl helper) to avoid is_loop_begin/is_loop_end parser
# issues and to match the production reference structure.

_kernel_cache = {}


def _create_fused_recurrent_kda_kernel(H, D, in_dtype, scale, eps,
                                       use_qk_l2norm_in_kernel,
                                       inplace_final_state,
                                       is_spec_decoding=False):
    K = D
    V = D

    # T / S are module-level pypto.DYNAMIC aliases (see Layer D). Shape lists
    # are inlined directly into the Tensor annotations below so OL31 can
    # resolve the dynamic axes (T = token axis, S = state-slot axis) via the
    # module-level alias scan. (Factory-local aliases would be invisible to
    # the lint AST walker.)
    # SIG_IMPL: snapshot marker — factory creates kernel with closure params
    @pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU,
                                         "stitch_function_max_num": 256,
                                         "device_sched_mode": 2,
                                         "launch_sched_aicpu_num": 3,
                                         },
                         pass_options={"vec_nbuffer_setting": {"DEFAULT": 8},
                                       "cube_nbuffer_setting": {-1: 2}}
                         )
    def fused_recurrent_kda_kernel_npu(
        q: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D], in_dtype),
        k: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D], in_dtype),
        v: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D], in_dtype),
        g: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D], pypto.DT_FP32),
        beta: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_FP32),
        state_buf_in: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, V, K], pypto.DT_FP32),
        cu_seqlens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
        ssm_state_indices: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_INT32),
        num_accepted_tokens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
        o: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D], in_dtype),
        state_buf_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, V, K], pypto.DT_FP32),
    ):
        """Fused recurrent KDA kernel (NPU).

        Loop nesting: seq -> head(parallel) -> token(step=1, strictly sequential).
        State S_buf [V, K] FP32 is loop-carried via persistent buffer + [:] (F7).
        Layout matches Triton (V, K); matmuls use b_trans=True.
        """
        # SIG_JIT: snapshot marker — JIT entry signature
        N = cu_seqlens.shape[0] - 1  # SymbolicScalar: number of sequences
        NH = N * H  # total parallel units = sequences * heads
        pypto.experimental.set_operation_options(combine_axis=True)

        # ── LOOP_NH: merged seq×head parallel loop ──
        for nh_idx in pypto.loop(NH, name="LOOP_NH", idx_name="nh_idx",
                                  parallel=True):
            seq_idx = nh_idx // H
            h_idx = nh_idx % H
            s = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]  # seq token count
            b_ofs = cu_seqlens[seq_idx]                        # seq start token

            if inplace_final_state:
                if is_spec_decoding:
                    init_t = num_accepted_tokens[seq_idx] - 1
                    init_slot = ssm_state_indices[seq_idx, init_t]
                else:
                    init_slot = ssm_state_indices[seq_idx, 0]
            else:
                init_slot = b_ofs

            # Persistent buffer (F7: tensor outside loop) — [V, K] layout (Triton-equivalent)
            S_buf = pypto.tensor([V, K], pypto.DT_FP32)
            pypto.set_vec_tile_shapes(V, K)

            if is_spec_decoding:
                S_buf[:] = state_buf_in[init_slot, h_idx]
            elif not inplace_final_state:
                # Non-inplace: init once outside loop, carry via [:] across tokens
                S_buf[:] = state_buf_in[init_slot, h_idx]

            # ── LOOP_T: token loop (sequential) ──
            for t_idx in pypto.loop(0, s, 1, name="LOOP_T",
                                    idx_name="t_idx",
                                    unroll_list=[1]):
                t_global = b_ofs + t_idx

                pypto.set_pass_options(sg_set_scope=1)

                q_i = pypto.view(q, [1, 1, D], [t_global, h_idx, 0])
                k_i = pypto.view(k, [1, 1, D], [t_global, h_idx, 0])
                v_i = pypto.view(v, [1, 1, D], [t_global, h_idx, 0])
                g_i = pypto.view(g, [1, 1, D], [t_global, h_idx, 0])
                beta_i = pypto.view(beta, [1, 1], [t_global, h_idx])

                pypto.set_vec_tile_shapes(1, D)
                q_i = pypto.cast(pypto.reshape(q_i, [1, D]), pypto.DT_FP32)
                k_i = pypto.cast(pypto.reshape(k_i, [1, D]), pypto.DT_FP32)
                v_i = pypto.cast(pypto.reshape(v_i, [1, D]), pypto.DT_FP32)
                g_i = pypto.reshape(g_i, [1, D])

                q_sq = pypto.mul(q_i, q_i)
                q_sum = pypto.sum(q_sq, -1, True)
                q_sum = pypto.add(q_sum, eps)
                q_rsqrt = pypto.rsqrt(q_sum)
                q_i = pypto.mul(q_i, q_rsqrt)
                k_sq = pypto.mul(k_i, k_i)
                k_sum = pypto.sum(k_sq, -1, True)
                k_sum = pypto.add(k_sum, eps)
                k_rsqrt = pypto.rsqrt(k_sum)
                k_i = pypto.mul(k_i, k_rsqrt)

                if inplace_final_state and not is_spec_decoding:
                    # Inplace: re-read from state_buf_in each iter (in/out split
                    # avoids RW cycle; same slot read+write = correct carry)
                    S_buf[:] = state_buf_in[init_slot, h_idx]

                pypto.set_vec_tile_shapes(V, K)
                gate = pypto.exp(g_i)  # [1, K] — broadcasts over V directly
                S_buf[:] = pypto.mul(S_buf, gate)  # [V, K] * [1, K] → [V, K]
                bk = pypto.mul(beta_i, k_i)
                kS_prod = pypto.mul(k_i, S_buf)  # [1,K]*[V,K]→[V,K] (broadcast)
                kS = pypto.sum(kS_prod, -1, True)  # [V, 1]
                kS = pypto.reshape(kS, [1, V])  # [1, V]
                delta = pypto.sub(v_i, kS)
                delta_col = pypto.reshape(delta, [V, 1])
                bk_row = pypto.reshape(bk, [1, K])
                upd = pypto.mul(delta_col, bk_row)  # [V, 1] * [1, K] → [V, K]
                S_buf[:] = pypto.add(S_buf, upd)
                o_prod = pypto.mul(q_i, S_buf)  # [1,K]*[V,K]→[V,K] (broadcast)
                o_i = pypto.sum(o_prod, -1, True)  # [V, 1]
                o_i = pypto.reshape(o_i, [1, V])  # [1, V]
                o_i = pypto.mul(o_i, scale)

                o_i_cast = pypto.cast(o_i, in_dtype)
                o_i_3d = pypto.reshape(o_i_cast, [1, 1, D])
                pypto.assemble(o_i_3d, [t_global, h_idx, 0], o)

                S_buf_4d = pypto.reshape(S_buf, [1, 1, V, K])
                if inplace_final_state:
                    if is_spec_decoding:
                        final_slot = ssm_state_indices[seq_idx, t_idx]
                        pypto.assemble(S_buf_4d, [final_slot, h_idx, 0, 0], state_buf_out)
                    else:
                        pypto.assemble(S_buf_4d, [init_slot, h_idx, 0, 0], state_buf_out)
                else:
                    pypto.assemble(S_buf_4d, [t_global, h_idx, 0, 0], state_buf_out)

                pypto.set_pass_options(sg_set_scope=-1)

    return fused_recurrent_kda_kernel_npu


def _get_fused_recurrent_kda_kernel(H, D, in_dtype, scale, eps,
                                    use_qk_l2norm_in_kernel,
                                    inplace_final_state,
                                    is_spec_decoding=False):
    key = (H, D, in_dtype, scale, eps, use_qk_l2norm_in_kernel, inplace_final_state, is_spec_decoding)
    if key not in _kernel_cache:
        _kernel_cache[key] = _create_fused_recurrent_kda_kernel(
            H, D, in_dtype, scale, eps,
            use_qk_l2norm_in_kernel, inplace_final_state, is_spec_decoding)
    return _kernel_cache[key]


_TORCH_TO_PYPTO_DTYPE = {
    torch.float16: pypto.DT_FP16,
    torch.bfloat16: pypto.DT_BF16,
    torch.float32: pypto.DT_FP32,
}


def fused_recurrent_kda_wrapper(
    q,
    k,
    v,
    g,
    beta=None,
    scale=None,
    initial_state=None,
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    **kwargs,
):
    B, T, H, D = q.shape
    V = D
    K = D

    # Resolve scale default
    if scale is None:
        scale = D ** -0.5

    # Default cu_seqlens: single sequence covering all tokens
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=q.device)

    # ── Input validation ──
    assert B == 1, f"q batch must be 1, got B={B}"
    assert q.ndim == 4, f"q must be 4D [B,T,H,D], got ndim={q.ndim}"
    assert k.shape == (B, T, H, K), f"k shape mismatch: expected {(B,T,H,K)}, got {tuple(k.shape)}"
    assert v.shape == (B, T, H, V), f"v shape mismatch: expected {(B,T,H,V)}, got {tuple(v.shape)}"
    assert g.shape == (B, T, H, K), f"g shape mismatch: expected {(B,T,H,K)}, got {tuple(g.shape)}"
    assert beta is not None, "beta must not be None"
    assert beta.shape == (B, T, H), f"beta shape mismatch: expected {(B,T,H)}, got {tuple(beta.shape)}"
    assert scale is not None, "scale must not be None"
    assert initial_state is not None, "initial_state must not be None"
    assert cu_seqlens is not None, "cu_seqlens must not be None"
    assert cu_seqlens.ndim == 1, f"cu_seqlens must be 1D, got ndim={cu_seqlens.ndim}"
    if ssm_state_indices is not None:
        assert ssm_state_indices.ndim in (1, 2), f"ssm_state_indices must be 1D or 2D, got ndim={ssm_state_indices.ndim}"
    if num_accepted_tokens is not None:
        assert num_accepted_tokens.ndim == 1, f"num_accepted_tokens must be 1D, got ndim={num_accepted_tokens.ndim}"

    # ── Dtype validation (all params) ──
    # q/k/v: kernel entry in_dtype, must be consistent fp16/bf16
    assert q.dtype in (torch.float16, torch.bfloat16), \
        f"q dtype must be fp16/bf16, got {q.dtype}"
    assert k.dtype == q.dtype, f"k dtype must match q ({q.dtype}), got {k.dtype}"
    assert v.dtype == q.dtype, f"v dtype must match q ({q.dtype}), got {v.dtype}"
    in_dtype = _TORCH_TO_PYPTO_DTYPE[q.dtype]
    # g/beta: kernel entry DT_FP32, only fp32 supported
    assert g.dtype == torch.float32, f"g dtype must be fp32, got {g.dtype}"
    assert beta.dtype == torch.float32, f"beta dtype must be fp32, got {beta.dtype}"
    # initial_state: fp32
    assert initial_state.dtype == torch.float32, \
        f"initial_state must be fp32, got {initial_state.dtype}"
    # cu_seqlens/ssm_state_indices: int32
    assert cu_seqlens.dtype == torch.int32, \
        f"cu_seqlens must be int32, got {cu_seqlens.dtype}"
    if ssm_state_indices is not None:
        assert ssm_state_indices.dtype == torch.int32, \
            f"ssm_state_indices must be int32, got {ssm_state_indices.dtype}"
    if num_accepted_tokens is not None:
        assert num_accepted_tokens.dtype == torch.int32, \
            f"num_accepted_tokens must be int32, got {num_accepted_tokens.dtype}"

    # 1. Fold B=1 into T (varlen T = total token count)
    q2d = q.reshape(T, H, D).contiguous()
    k2d = k.reshape(T, H, D).contiguous()
    v2d = v.reshape(T, H, D).contiguous()
    g2d = g.reshape(T, H, D).contiguous()
    beta2d = beta.reshape(T, H).contiguous()

    # State layout: [S, H, V, K] fp32 — caller must provide contiguous tensor
    assert initial_state.is_contiguous(), "initial_state must be contiguous"

    # 2. ssm_state_indices fallback (reshape to 2D for kernel)
    if ssm_state_indices is not None and ssm_state_indices.numel() > 0:
        if ssm_state_indices.ndim == 1:
            ssm32 = ssm_state_indices.unsqueeze(-1)  # [N] -> [N, 1]
        else:
            ssm32 = ssm_state_indices  # [N, T] as-is
    else:
        ssm32 = torch.zeros((1, 1), dtype=torch.int32, device=q.device)

    # num_accepted_tokens fallback
    is_spec_decoding = num_accepted_tokens is not None and num_accepted_tokens.numel() > 0
    if is_spec_decoding:
        nat32 = num_accepted_tokens
    else:
        nat32 = torch.empty(0, dtype=torch.int32, device=q.device)

    # HOST_WRAPPER_INSPECT_ALLOC: snapshot marker — output allocation
    # 3. Allocate output (OL58: use torch.* not pypto.* for host allocation)
    o_out = torch.empty(T, H, D, dtype=q.dtype, device=q.device)

    # Get cached kernel (closure params captured via factory)
    kernel = _get_fused_recurrent_kda_kernel(
        H, D, in_dtype, scale, 1e-6,
        use_qk_l2norm_in_kernel, inplace_final_state, is_spec_decoding)

    # HOST_WRAPPER_INSPECT_PASS: snapshot marker — kernel call + return
    if not inplace_final_state:
        ht_vk = initial_state.clone()
        kernel(q2d, k2d, v2d, g2d, beta2d, ht_vk, cu_seqlens, ssm32, nat32, o_out, ht_vk)
        ht = ht_vk.contiguous()
        return o_out.reshape(B, T, H, D), ht
    else:
        kernel(q2d, k2d, v2d, g2d, beta2d, initial_state, cu_seqlens, ssm32, nat32, o_out, initial_state)
        return o_out.reshape(B, T, H, D), initial_state


# Alias for convenience (task description uses fused_recurrent_kda_impl)
fused_recurrent_kda_impl = fused_recurrent_kda_wrapper


@allow_in_graph
def fused_recurrent_kda(
    q,
    k,
    v,
    g,
    beta=None,
    scale=None,
    initial_state=None,
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    num_accepted_tokens=None,
    **kwargs,
):
    """aclgraph-compatible entry point for fused_recurrent_kda.

    Handles FakeTensor during graph capture by returning zero tensors of
    the correct shape. Falls through to the eager wrapper for real execution.
    See docs/zh/tutorials/network_integration/pytorch_integration.md
    """
    if isinstance(q, FakeTensor) or isinstance(initial_state, FakeTensor):
        B, T, H, D = q.shape
        o_out = torch.zeros(B, T, H, D, dtype=q.dtype, device=q.device)
        state_out = torch.zeros_like(initial_state)
        return o_out, state_out
    return fused_recurrent_kda_wrapper(
        q, k, v, g, beta, scale, initial_state,
        inplace_final_state, use_qk_l2norm_in_kernel,
        cu_seqlens, ssm_state_indices, num_accepted_tokens, **kwargs)


# =============================================================================
# torch.library integration — register fused_recurrent_kda as a custom op
# (torch.ops.pypto.fused_recurrent_kda) for torch.compile / Inductor / graph
# capture. Pattern mirrors models/deepseek_v4/compressor_impl.py and
# custom/sas_forward/sas_forward_impl.py.
#
# Schema note: scale (float) and the two bool flags are passed as concrete
# scalars; the caller resolves scale=None → D**-0.5 and ssm_state_indices=None
# → empty int32 tensor before dispatching to torch.ops. cu_seqlens is a
# required Tensor in the schema (wrapper's None fallback is handled by the
# _graph convenience function below).
# =============================================================================

_pyptolib = torch.library.Library("pypto", "FRAGMENT")
_pyptolib.define(
    "fused_recurrent_kda("
    "Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
    "Tensor initial_state, Tensor cu_seqlens, Tensor ssm_state_indices, "
    "Tensor num_accepted_tokens, "
    "float scale, bool inplace_final_state, bool use_qk_l2norm_in_kernel"
    ") -> (Tensor, Tensor)"
)


@torch.library.impl(_pyptolib, "fused_recurrent_kda", "Meta")
def _fused_recurrent_kda_meta(
    q, k, v, g, beta, initial_state, cu_seqlens, ssm_state_indices,
    num_accepted_tokens,
    scale, inplace_final_state, use_qk_l2norm_in_kernel,
):
    B, T, H, D = q.shape
    o = torch.empty((B, T, H, D), dtype=q.dtype, device=q.device)
    ht = torch.empty_like(initial_state)
    return o, ht


try:
    @torch.library.impl(_pyptolib, "fused_recurrent_kda", "NPU")
    def _fused_recurrent_kda_npu(
        q, k, v, g, beta, initial_state, cu_seqlens, ssm_state_indices,
        num_accepted_tokens,
        scale, inplace_final_state, use_qk_l2norm_in_kernel,
    ):
        return fused_recurrent_kda_wrapper(
            q, k, v, g, beta,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
        )
except Exception as e:
    if "could not parse dispatch key: NPU" in str(e):
        print(f"Skip: torchair not installed, skip NPU registration for operator 'fused_recurrent_kda'")
    else:
        print(f"Skip: Unexpected error : {e}")


def fused_recurrent_kda_graph(
    q, k, v, g, beta=None, scale=None, initial_state=None,
    inplace_final_state=True, use_qk_l2norm_in_kernel=True,
    cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
):
    """Graph mode entry — calls torch.ops.pypto.fused_recurrent_kda.

    Resolves Python-level defaults (scale=None -> D**-0.5, cu_seqlens=None ->
    [0,T], ssm_state_indices=None -> empty int32, num_accepted_tokens=None ->
    empty int32) into concrete schema values before dispatching to the
    registered custom op.
    """
    B, T, H, D = q.shape
    if scale is None:
        scale = D ** -0.5
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=q.device)
    if ssm_state_indices is None:
        ssm_state_indices = torch.empty(0, dtype=torch.int32, device=q.device)
    if num_accepted_tokens is None:
        num_accepted_tokens = torch.empty(0, dtype=torch.int32, device=q.device)
    return torch.ops.pypto.fused_recurrent_kda(
        q, k, v, g, beta, initial_state, cu_seqlens, ssm_state_indices,
        num_accepted_tokens,
        scale, inplace_final_state, use_qk_l2norm_in_kernel,
    )
