import vllm.model_executor.layers.fla.ops
import vllm.model_executor.layers.mamba.ops.causal_conv1d
import vllm.v1.worker.gpu.sample.gumbel
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule
from vllm_ascend.ops.triton.fla.layernorm_guard import LayerNormFn
from vllm_ascend.ops.triton.fla.sigmoid_gating import fused_recurrent_gated_delta_rule_fwd_kernel
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update_npu
from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.20.2"):
    triton.next_power_of_2 = next_power_of_2

vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update = causal_conv1d_update_npu
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn = causal_conv1d_fn
vllm.model_executor.layers.fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule_fwd_kernel = (
    fused_recurrent_gated_delta_rule_fwd_kernel
)
vllm.model_executor.layers.fla.ops.layernorm_guard.LayerNormFn = LayerNormFn
vllm.model_executor.layers.fla.ops.chunk_gated_delta_rule = chunk_gated_delta_rule

# On NPU platforms without an active Triton backend (e.g. 310P), replace the
# Triton-based fused_post_conv_prep with a pure-PyTorch fallback so that
# qwen_gdn_linear_attn's from-import picks up the replacement before model
# load.  fused_post_conv_prep was introduced alongside Qwen3-Next GDN support
# and does not exist in v0.20.2.
if not HAS_TRITON and not vllm_version_is("0.20.2"):
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

    vllm.model_executor.layers.fla.ops.fused_post_conv_prep = _fused_post_conv_prep_pytorch

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

    vllm.model_executor.layers.fla.ops.fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode = (
        _fused_recurrent_packed_decode_pytorch
    )
