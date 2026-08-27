import vllm.model_executor.layers.mamba.ops.causal_conv1d as _cc1d
import vllm.third_party.flash_linear_attention.ops as fla_ops
import vllm.third_party.flash_linear_attention.ops.fused_recurrent as fla_fused_recurrent
import vllm.third_party.flash_linear_attention.ops.layernorm_guard as fla_layernorm_guard
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend._310p.ops.causal_conv1d import (
    causal_conv1d_fn as _npu_causal_conv1d_fn_impl,
)
from vllm_ascend._310p.ops.causal_conv1d import causal_conv1d_update as _npu_causal_conv1d_update_impl
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.layernorm_guard import LayerNormFn

triton.next_power_of_2 = next_power_of_2


def _npu_causal_conv1d_update(*args, **kwargs):
    for key in (
        "null_block_id",
        "block_idx_first_scheduled_token",
        "block_idx_last_scheduled_token",
        "initial_state_idx",
        "num_computed_tokens",
        "block_size_to_align",
        "validate_data",
        "max_query_len",
        "out",
        "metadata",
    ):
        kwargs.pop(key, None)
    return _npu_causal_conv1d_update_impl(*args, **kwargs)


# npu_kda_causal_conv1d_bind: CUDA causal_conv1d kernels use tl.extra.cuda.gdc_wait
# which Ascend Triton does not provide. Swap the NPU update kernel and a PyTorch
# prefill fn, then rebind GLM KDA from-imports.
_CC1D_FN_DROP = (
    "null_block_id",
    "block_idx_first_scheduled_token",
    "block_idx_last_scheduled_token",
    "initial_state_idx",
    "num_computed_tokens",
    "block_size_to_align",
    "validate_data",
)


def _npu_causal_conv1d_fn(*args, metadata=None, **kwargs):
    for key in _CC1D_FN_DROP:
        kwargs.pop(key, None)
    return _npu_causal_conv1d_fn_impl(*args, **kwargs)


_cc1d.causal_conv1d_update = _npu_causal_conv1d_update
_cc1d.causal_conv1d_fn = _npu_causal_conv1d_fn
try:
    import vllm.models.glm5next.nvidia.kda as _glm_kda

    _glm_kda.causal_conv1d_update = _npu_causal_conv1d_update
    _glm_kda.causal_conv1d_fn = _npu_causal_conv1d_fn
except ImportError:
    pass

fla_layernorm_guard.LayerNormFn = LayerNormFn
fla_ops.chunk_gated_delta_rule = chunk_gated_delta_rule

# GLM-5.3-Flash (and Kimi KDA) import fused_recurrent_kda / chunk_kda_with_fused_gate
# from this FLA module. Swap them for the NPU Triton implementations before the
# model is constructed. Missing on older vLLM trees that predate KDA.
try:
    import vllm.third_party.flash_linear_attention.ops.kda as fla_kda

    from vllm_ascend.ops.triton.kda.kda import (
        chunk_kda as _npu_chunk_kda,
    )
    from vllm_ascend.ops.triton.kda.kda import (
        chunk_kda_with_fused_gate as _npu_chunk_kda_with_fused_gate,
    )
    from vllm_ascend.ops.triton.kda.kda import (
        fused_kda_gate as _npu_fused_kda_gate,
    )
    from vllm_ascend.ops.triton.kda.kda import (
        fused_recurrent_kda as _npu_fused_recurrent_kda,
    )

    fla_kda.fused_recurrent_kda = _npu_fused_recurrent_kda
    fla_kda.chunk_kda = _npu_chunk_kda
    fla_kda.chunk_kda_with_fused_gate = _npu_chunk_kda_with_fused_gate
    fla_kda.fused_kda_gate = _npu_fused_kda_gate
except ImportError:
    pass

# GLM-5.3-Flash vision tower (and any other NVIDIA fused Q/K RMSNorm) launches a
# CUDA Triton kernel that references tl.extra.cuda.gdc_wait. Ascend Triton has
# no such symbol; the AST visitor raises even when launch_pdl is False.
try:
    import torch_npu
    import vllm.models.common.ops as _common_ops
    import vllm.models.common.ops.fused_qk_rmsnorm as _fused_qk_mod

    def _npu_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps):
        # Same math as the CUDA kernel: RMS in fp32, one cast at store.
        q_out, _ = torch_npu.npu_rms_norm(qr, q_weight, eps)
        kv_out, _ = torch_npu.npu_rms_norm(kv, kv_weight, eps)
        return q_out, kv_out

    _fused_qk_mod.fused_q_kv_rmsnorm = _npu_fused_q_kv_rmsnorm
    _common_ops.fused_q_kv_rmsnorm = _npu_fused_q_kv_rmsnorm
    # from_import_sweep: the GLM vision tower is imported during model
    # registration, before this patch runs, so its from-import already
    # holds a direct reference to the CUDA kernel. Rebinding the defining
    # module is not enough; rebind every module that captured the name.
    import sys as _sys

    for _m in list(_sys.modules.values()):
        if getattr(_m, "fused_q_kv_rmsnorm", None) not in (None, _npu_fused_q_kv_rmsnorm):
            _m.fused_q_kv_rmsnorm = _npu_fused_q_kv_rmsnorm
except ImportError:
    pass

# On NPU platforms without an active Triton backend (e.g. 310P), replace the
# Triton-based fused_post_conv_prep with a pure-PyTorch fallback so that
# qwen_gdn_linear_attn's from-import picks up the replacement before model
# load.
if not HAS_TRITON:
    import torch
    import torch.nn.functional as _F

    def _fused_post_conv_prep_pytorch(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads,
        head_k_dim,
        head_v_dim,
        apply_l2norm=True,
        output_g_exp=False,
    ):
        L = conv_output.shape[0]
        H, K, V = num_k_heads, head_k_dim, head_v_dim
        HV = A_log.shape[0]

        q = conv_output[:, : H * K].reshape(L, H, K)
        k = conv_output[:, H * K : 2 * H * K].reshape(L, H, K)
        v = conv_output[:, 2 * H * K :].reshape(L, HV, V)

        if apply_l2norm:
            # x / sqrt(sum(x^2) + eps) — matches Triton kernel, in fp32
            def _l2norm(t):
                t_f = t.float()
                return (t_f / torch.sqrt((t_f * t_f).sum(-1, keepdim=True) + 1e-6)).to(t.dtype)

            q, k = _l2norm(q), _l2norm(k)

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        x = (a + dt_bias.unsqueeze(0)).float()
        g = -torch.exp(A_log.float().unsqueeze(0)) * _F.softplus(x)
        if output_g_exp:
            g = torch.exp(g)

        return q, k, v, g, torch.sigmoid(b.float())

    fla_ops.fused_post_conv_prep = _fused_post_conv_prep_pytorch

    def _fused_recurrent_packed_decode_pytorch(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel=False,
    ):
        B = mixed_qkv.shape[0]
        HV, V, K = initial_state.shape[-3:]
        H = (mixed_qkv.shape[1] - HV * V) // (2 * K)
        ratio = HV // H

        q = mixed_qkv[:, : H * K].reshape(B, H, K)
        k = mixed_qkv[:, H * K : 2 * H * K].reshape(B, H, K)
        v = mixed_qkv[:, 2 * H * K :].reshape(B, HV, V)

        SOFTPLUS_THRESHOLD = 20.0
        x = (a + dt_bias.unsqueeze(0)).float()
        softplus_x = torch.where(x <= SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(x)), x)
        g = -torch.exp(A_log.float().unsqueeze(0)) * softplus_x  # [B, HV]
        beta = torch.sigmoid(b.float())  # [B, HV]

        for n in range(B):
            state_idx = int(ssm_state_indices[n].item())
            if state_idx <= 0:
                out[n, 0] = 0
                continue

            h = initial_state[state_idx].float()  # [HV, V, K]
            q_n = q[n].float().repeat_interleave(ratio, dim=0)  # [HV, K]
            k_n = k[n].float().repeat_interleave(ratio, dim=0)  # [HV, K]
            v_n = v[n].float()  # [HV, V]

            if use_qk_l2norm_in_kernel:

                def _l2norm(t):
                    t_f = t.float()
                    return t_f / torch.sqrt((t_f * t_f).sum(-1, keepdim=True) + 1e-6)

                q_n, k_n = _l2norm(q_n), _l2norm(k_n)
            q_n = q_n * scale

            h = h * torch.exp(g[n]).view(HV, 1, 1)
            v_n = v_n - torch.einsum("hvk,hk->hv", h, k_n)
            v_n = v_n * beta[n].view(HV, 1)
            h = h + torch.einsum("hv,hk->hvk", v_n, k_n)
            out[n, 0] = torch.einsum("hvk,hk->hv", h, q_n).to(out.dtype)
            initial_state[state_idx] = h.to(initial_state.dtype)

        return out, initial_state

    fla_fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode = _fused_recurrent_packed_decode_pytorch


# npu_mhc_fused_rmsnorm: the tilelang mHC pre kernel fuses the layer's input
# RMSNorm into layer_input, but mhc_pre_torch (which forward_native/forward_oot
# use on NPU) ignores norm_weight/norm_eps entirely, so every layer would run
# without its input norm. Reapply it here to match the CUDA semantics:
#   layer_input = layer_input * rsqrt(mean(layer_input^2) + norm_eps) * norm_weight
try:
    import torch as _t
    from vllm.model_executor.kernels.mhc.torch import (
        mhc_post_torch as _mhc_post_torch,
    )
    from vllm.model_executor.kernels.mhc.torch import (
        mhc_pre_torch as _mhc_pre_torch,
    )
    from vllm.model_executor.layers import mhc as _mhc_mod

    def _mhc_rms_norm(x, weight, eps):
        xf = x.float()
        var = xf.square().mean(dim=-1, keepdim=True)
        return (xf * _t.rsqrt(var + eps) * weight.float()).to(x.dtype)

    def _mhc_pre_npu(
        self,
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        norm_weight=None,
        norm_eps=0.0,
    ):
        post_mix, comb_mix, layer_input = _mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        if norm_weight is not None:
            layer_input = _mhc_rms_norm(layer_input, norm_weight, norm_eps)
        return post_mix, comb_mix, layer_input

    def _mhc_fused_post_pre_npu(
        self,
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits=1,
        tile_n=1,
        norm_weight=None,
        norm_eps=0.0,
    ):
        residual_cur = _mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)
        post_mix_cur, comb_mix_cur, layer_input_cur = _mhc_pre_torch(
            residual_cur,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        if norm_weight is not None:
            layer_input_cur = _mhc_rms_norm(layer_input_cur, norm_weight, norm_eps)
        return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur

    _mhc_mod.MHCPreOp.forward_oot = _mhc_pre_npu
    _mhc_mod.MHCPreOp.forward_native = _mhc_pre_npu
    _mhc_mod.MHCFusedPostPreOp.forward_oot = _mhc_fused_post_pre_npu
    _mhc_mod.MHCFusedPostPreOp.forward_native = _mhc_fused_post_pre_npu
except ImportError:
    pass

# npu_kda_causal_conv1d_triton: the 310P PyTorch fallback calls .item() per
# request, which aborts ACL graph capture. The upstream NPU Triton kernel has
# no host sync and accepts the spec-decode kwargs the fallback had to drop.
try:
    from vllm_ascend.ops.triton.mamba.causal_conv1d import (
        causal_conv1d_update_npu as _cc1d_update_npu,
    )

    _cc1d.causal_conv1d_update = _cc1d_update_npu
    try:
        import vllm.models.glm5next.nvidia.kda as _glm_kda_triton

        _glm_kda_triton.causal_conv1d_update = _cc1d_update_npu
    except ImportError:
        pass
    print("[npu_kda_causal_conv1d_triton] bound NPU Triton causal_conv1d_update", flush=True)
except Exception as _cc1d_err:
    print("[npu_kda_causal_conv1d_triton] FAILED:", repr(_cc1d_err), flush=True)

# npu_mamba_state_ops: gather_initial_states / scatter_states assert
# state.is_cuda and launch Triton kernels that reference
# tl.extra.cuda.gdc_wait, neither of which holds on Ascend. Both are
# plain index ops, so express them in torch and keep them ACL-graph
# safe (no host sync). A from-import in the already-loaded GLM KDA
# module needs the same sweep as the vision-tower RMSNorm above.
try:
    import sys as _sys_ms

    import torch as _t_ms
    import vllm.model_executor.layers.mamba.ops.gather_initial_states as _gis_mod
    import vllm.model_executor.layers.mamba.ops.scatter_states as _scs_mod

    def _npu_gather_initial_states(state, indices, has_initial_state):
        idx = indices.to(_t_ms.int64) * has_initial_state.to(_t_ms.int64)
        out = state.index_select(0, idx)
        keep = has_initial_state.view([-1] + [1] * (state.dim() - 1)).to(out.dtype)
        return out * keep

    def _npu_scatter_states(state, src, indices):
        state.index_copy_(0, indices.to(_t_ms.int64), src)

    _gis_mod.gather_initial_states = _npu_gather_initial_states
    _scs_mod.scatter_states = _npu_scatter_states
    for _m in list(_sys_ms.modules.values()):
        if getattr(_m, "gather_initial_states", None) not in (None, _npu_gather_initial_states):
            _m.gather_initial_states = _npu_gather_initial_states
        if getattr(_m, "scatter_states", None) not in (None, _npu_scatter_states):
            _m.scatter_states = _npu_scatter_states
except ImportError:
    pass
