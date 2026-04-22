#
# Pure PyTorch fallback for Qwen3.5 GDN (Gated Delta Network) on Ascend NPU.
#
# This file provides a correct (but slower) implementation of the GDN computation
# when triton kernels produce incorrect numerical results on NPU.
#
# The triton kernels (chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,
# fused_sigmoid_gating_delta_rule_update, causal_conv1d_fn, causal_conv1d_update)
# all produce incorrect results on Ascend 910 with Triton 3.2.0 + NPU backend.
#
# This fallback implements the same algorithms in pure PyTorch.

import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.utils import enable_sp

# Use the non-triton causal_conv1d_ops if available (some builds have PyTorch versions)
_CONV_TRITON_AVAILABLE = False  # triton conv1d is broken on NPU


def _softplus(x, beta=1.0, threshold=20.0):
    """Numerically stable softplus implementation."""
    bx = beta * x
    return torch.where(
        bx <= threshold,
        (1.0 / beta) * torch.log1p(torch.exp(bx)),
        x,
    )


def _pytorch_causal_conv1d_decode(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    """
    Causal 1D convolution for single-token decode.

    x: (num_tokens, dim) - input tokens
    conv_state: (num_slots, dim, kernel_size - 1) - convolution state
    weight: (dim, kernel_size) - convolution weights
    bias: (dim,) - bias
    activation: str - activation function name
    """
    # For decode, each token just does: out = x * weight[0] + state * weight[1:] + bias
    K = weight.shape[1]  # kernel size

    # x: (num_tokens, D), weight[:, 0]: (D,)
    out = x * weight[:, 0]  # (num_tokens, D)

    # Add contribution from conv_state
    # conv_state: (num_slots, D, K-1), weight[:, 1:]: (D, K-1)
    for i in range(K - 1):
        out = out + conv_state[:, :, i] * weight[:, i + 1]

    # Add bias
    if bias is not None:
        out = out + bias.unsqueeze(0)

    # Apply activation
    if activation == "silu":
        out = F.silu(out)
    elif activation == "gelu":
        out = F.gelu(out)

    return out


def _pytorch_causal_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str,
    conv_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> torch.Tensor:
    """
    Causal 1D convolution for prefill with varlen batching.

    x: (D, total_tokens) - input tokens (already transposed)
    weight: (D, K) - convolution weights
    bias: (D,) - bias
    activation: str
    conv_states: (num_slots, D, K-1) - convolution state storage
    cache_indices: (num_sequences,) - indices into conv_states
    query_start_loc: (num_sequences+1,) - cumulative sequence lengths
    has_initial_state: (num_sequences,) - whether each sequence has initial state
    """
    D, T = x.shape
    K = weight.shape[1]
    N = len(cache_indices)

    # Output
    out = torch.zeros_like(x)  # (D, T)

    for n in range(N):
        bos = query_start_loc[n].item()
        eos = query_start_loc[n + 1].item()
        seq_len = eos - bos
        slot_idx = cache_indices[n].item()

        if slot_idx < 0 or seq_len == 0:
            continue

        # Get this sequence's input: (D, seq_len)
        x_seq = x[:, bos:eos]

        if has_initial_state[n].item():
            # Prepend conv_state to input
            state = conv_states[slot_idx]  # (D, K-1)
            x_padded = torch.cat([state, x_seq], dim=1)  # (D, K-1+seq_len)
        else:
            # Pad with zeros
            x_padded = F.pad(x_seq, (K - 1, 0))  # (D, K-1+seq_len)

        # Apply convolution: out[d][t] = sum_{k=0}^{K-1} weight[d][k] * x_padded[d][t + K-1 - k]
        # This is equivalent to F.conv1d with groups=D
        x_padded_4d = x_padded.unsqueeze(0)  # (1, D, L_padded)
        weight_3d = weight.unsqueeze(1)  # (D, 1, K)
        conv_out = F.conv1d(x_padded_4d, weight_3d, groups=D).squeeze(0)  # (D, seq_len)

        # Add bias
        if bias is not None:
            conv_out = conv_out + bias.unsqueeze(1)

        # Apply activation
        if activation == "silu":
            conv_out = F.silu(conv_out)
        elif activation == "gelu":
            conv_out = F.gelu(conv_out)

        out[:, bos:eos] = conv_out

        # Update conv_state with last K-1 values of this sequence
        if K > 1:
            conv_states[slot_idx] = x_seq[:, -(K - 1) :].float().to(conv_states.dtype)

    return out


def _compute_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
):
    """
    Compute GDN gating values: g (decay) and beta (sigmoid).

    A_log: (HV,) - log decay rate
    a: (num_tokens, HV) - a input
    b: (num_tokens, HV) - b input (for sigmoid)
    dt_bias: (HV,) - dt bias

    Returns:
        g: (1, num_tokens, HV) - decay values
        beta: (1, num_tokens, HV) - beta values
    """
    # g = -exp(A_log) * softplus(a + dt_bias)
    x = a.float() + dt_bias.float().unsqueeze(0)  # (T, HV)
    sp = _softplus(x, softplus_beta, softplus_threshold)  # (T, HV)
    g = -torch.exp(A_log.float()).unsqueeze(0) * sp  # (1, T, HV) broadcasting

    # beta = sigmoid(b)
    beta = torch.sigmoid(b.float()).unsqueeze(0)  # (1, T, HV)

    return g, beta


def _pytorch_recurrent_gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    use_l2norm: bool,
    softplus_beta: float,
    softplus_threshold: float,
    use_sigmoid_gating: bool,
):
    """
    Pure PyTorch implementation of GDN decode (single token per sequence).

    If use_sigmoid_gating is True, computes g and beta internally from A_log, dt_bias, a, b.
    Otherwise, uses precomputed g and beta.

    State layout: (num_slots, HV, V, K) - matching vllm's standard kernel.

    Returns:
        output: (1, T, HV, V) - attention output
    """
    T = q.shape[1]
    H = q.shape[2]
    HV = v.shape[2]
    V = v.shape[3]
    kv_ratio = HV // H
    N = len(cu_seqlens) - 1

    output = torch.zeros(1, T, HV, V, dtype=v.dtype, device=v.device)

    for n in range(N):
        bos = cu_seqlens[n].item()
        eos = cu_seqlens[n + 1].item()
        state_idx = state_indices[n].item()

        if state_idx < 0 or bos >= eos:
            continue

        # Load state: (HV, V, K) in float32
        h = ssm_state[state_idx].float()  # (HV, V, K)

        for t in range(bos, eos):
            # Get inputs for this token
            q_t = q[0, t].float()  # (H, K)
            k_t = k[0, t].float()  # (H, K)
            v_t = v[0, t].float()  # (HV, V)

            # L2 normalize q and k
            if use_l2norm:
                q_t = q_t / (q_t.norm(dim=-1, keepdim=True) + 1e-6)
                k_t = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)
            q_t = q_t * scale

            # Expand key/value to all value heads (GQA)
            k_expanded = k_t.repeat_interleave(kv_ratio, dim=0)  # (HV, K)
            q_expanded = q_t.repeat_interleave(kv_ratio, dim=0)  # (HV, K)

            # Compute gating
            if use_sigmoid_gating:
                # Compute g and beta from A_log, a, b, dt_bias
                a_t = a[t].float()  # (HV,)
                b_t = b[t].float()  # (HV,)
                x = a_t + dt_bias.float()
                sp = _softplus(x, softplus_beta, softplus_threshold)
                g_t = -torch.exp(A_log.float()) * sp  # (HV,)
                beta_t = torch.sigmoid(b_t)  # (HV,)
            else:
                g_t = g[0, t]  # (HV,)
                beta_t = beta[0, t]  # (HV,)

            # Decay state: h *= exp(g)
            exp_g = torch.exp(g_t).unsqueeze(1).unsqueeze(2)  # (HV, 1, 1)
            h = h * exp_g  # (HV, V, K)

            # Delta rule: v -= h @ k
            # h: (HV, V, K), k_expanded: (HV, K)
            v_corr = torch.bmm(h, k_expanded.unsqueeze(2)).squeeze(2)  # (HV, V)
            v_t = v_t - v_corr

            # Beta gating
            v_t = v_t * beta_t.unsqueeze(1)  # (HV, V)

            # Update state: h += v outer k
            h = h + torch.bmm(v_t.unsqueeze(2), k_expanded.unsqueeze(1))  # (HV, V, K)

            # Output: o = h @ q
            o_t = torch.bmm(h, q_expanded.unsqueeze(2)).squeeze(2)  # (HV, V)

            output[0, t] = o_t.to(v.dtype)

        # Store state back
        ssm_state[state_idx] = h.to(ssm_state.dtype)

    return output


def _pytorch_recurrent_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    use_l2norm: bool,
):
    """
    Pure PyTorch implementation of GDN prefill (sequential token processing).

    Uses precomputed g and beta (from fused_gdn_gating_patch).

    State layout: (HV, V, K) - matching vllm's standard kernel.

    Returns:
        output: (1, T, HV, V)
        final_states: dict mapping sequence index to final state tensor
    """
    T = q.shape[1]
    H = q.shape[2]
    HV = v.shape[2]
    V = v.shape[3]
    kv_ratio = HV // H
    N = len(cu_seqlens) - 1

    output = torch.zeros(1, T, HV, V, dtype=v.dtype, device=v.device)

    for n in range(N):
        bos = cu_seqlens[n].item()
        eos = cu_seqlens[n + 1].item()
        state_idx = state_indices[n].item()

        if state_idx < 0 or bos >= eos:
            continue

        # Load initial state: (HV, V, K) in float32
        h = ssm_state[state_idx].float()  # (HV, V, K)

        for t in range(bos, eos):
            q_t = q[0, t].float()  # (H, K)
            k_t = k[0, t].float()  # (H, K)
            v_t = v[0, t].float()  # (HV, V)

            if use_l2norm:
                q_t = q_t / (q_t.norm(dim=-1, keepdim=True) + 1e-6)
                k_t = k_t / (k_t.norm(dim=-1, keepdim=True) + 1e-6)
            q_t = q_t * scale

            k_expanded = k_t.repeat_interleave(kv_ratio, dim=0)  # (HV, K)
            q_expanded = q_t.repeat_interleave(kv_ratio, dim=0)  # (HV, K)

            g_t = g[0, t].float()  # (HV,)
            beta_t = beta[0, t].float()  # (HV,)

            # Decay
            exp_g = torch.exp(g_t).unsqueeze(1).unsqueeze(2)
            h = h * exp_g

            # Delta rule: v -= h @ k
            v_corr = torch.bmm(h, k_expanded.unsqueeze(2)).squeeze(2)
            v_t = v_t - v_corr

            # Beta gating
            v_t = v_t * beta_t.unsqueeze(1)

            # Update state
            h = h + torch.bmm(v_t.unsqueeze(2), k_expanded.unsqueeze(1))

            # Output
            o_t = torch.bmm(h, q_expanded.unsqueeze(2)).squeeze(2)
            output[0, t] = o_t.to(v.dtype)

        # Store final state
        ssm_state[state_idx] = h.to(ssm_state.dtype)

    return output


def _rearrange_mixed_qkv(mixed_qkv, head_k_dim, head_v_dim):
    """Split mixed_qkv into query, key, value tensors."""
    if mixed_qkv is None:
        return None, None, None
    # Use a simpler approach for splitting mixed_qkv
    # mixed_qkv: (T, key_dim + key_dim + value_dim) from conv output
    D = mixed_qkv.shape[-1]
    # Split proportionally: 2*key_dim + value_dim = D
    # key_dim = num_k_heads * head_k_dim, value_dim = num_v_heads * head_v_dim
    # For Qwen3.5 GDN: num_k_heads = num_v_heads, head_k_dim = head_v_dim
    # So D = 3 * key_dim, split into 3 equal parts? No...
    # Actually the conv output is key_dim * 2 + value_dim
    # For Qwen3.5: key_dim = num_k_heads * head_k_dim, value_dim = num_v_heads * head_v_dim
    # If num_k_heads == num_v_heads and head_k_dim == head_v_dim: D = 3 * key_dim
    # Split: [key_dim, key_dim, value_dim]
    # But we need the head_k_dim and num_k_heads to split correctly

    # The split should be [key_dim, key_dim, value_dim]
    # where key_dim = num_k_heads * head_k_dim, value_dim = num_v_heads * head_v_dim
    # For simplicity, split into 3 equal parts if they're equal
    third = D // 3
    query, key, value = torch.split(mixed_qkv, [third, third, D - 2 * third], dim=-1)
    num_k_heads = third // head_k_dim
    num_v_heads = (D - 2 * third) // head_v_dim

    query = query.view(*query.shape[:-1], num_k_heads, head_k_dim).unsqueeze(0)  # (1, T, H, K)
    key = key.view(*key.shape[:-1], num_k_heads, head_k_dim).unsqueeze(0)  # (1, T, H, K)
    value = value.view(*value.shape[:-1], num_v_heads, head_v_dim).unsqueeze(0)  # (1, T, HV, V)

    return query.contiguous(), key.contiguous(), value.contiguous()


class PyTorchQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """
    Pure PyTorch implementation of Qwen3.5 GDN for Ascend NPU.
    Replaces all triton kernels with correct (but slower) PyTorch code.
    """

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        num_actual_tokens = attn_metadata.num_actual_tokens
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        has_initial_state = attn_metadata.has_initial_state

        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)  # (..., K-1, D) -> (..., D, K-1)
        ssm_state = self_kv_cache[1]  # (num_pages, HV, V, K)

        if not enable_sp():
            mixed_qkv = mixed_qkv[:num_actual_tokens]
            b = b[:num_actual_tokens]
            a = a[:num_actual_tokens]

        # Convolution weights
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))

        # Split into spec and non-spec tokens
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # ---- Convolution ----
        # Spec tokens: decode convolution
        if spec_sequence_masks is not None and mixed_qkv_spec is not None:
            mixed_qkv_spec = _pytorch_causal_conv1d_decode(
                x=mixed_qkv_spec,
                conv_state=conv_state,
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # Non-spec tokens
        if attn_metadata.num_prefills > 0 and mixed_qkv_non_spec is not None:
            # Prefill convolution
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.T.contiguous()  # (D, T)
            out = _pytorch_causal_conv1d_prefill(
                x=mixed_qkv_non_spec_T,
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                has_initial_state=has_initial_state,
            )
            mixed_qkv_non_spec = out.T.contiguous()  # (T, D)
        elif attn_metadata.num_decodes > 0 and mixed_qkv_non_spec is not None:
            # Decode convolution
            mixed_qkv_non_spec = _pytorch_causal_conv1d_decode(
                x=mixed_qkv_non_spec,
                conv_state=conv_state,
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # ---- Rearrange QKV ----
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)

        scale = self.head_k_dim**-0.5

        # ---- GDN Recurrent Computation ----
        if attn_metadata.num_prefills > 0 or spec_sequence_masks is not None:
            # Path 1: Prefill + spec decode
            # Compute gating values for all actual tokens
            a_for_gating = a[:num_actual_tokens] if not enable_sp() else a
            b_for_gating = b[:num_actual_tokens] if not enable_sp() else b
            g_all, beta_all = _compute_gdn_gating(self.A_log, a_for_gating, b_for_gating, self.dt_bias)

            # Split gating for spec and non-spec tokens
            if spec_sequence_masks is not None:
                if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                    g_spec = g_all
                    beta_spec = beta_all
                    g_non_spec = None
                    beta_non_spec = None
                else:
                    g_spec = g_all[:, spec_token_indx, :]
                    beta_spec = beta_all[:, spec_token_indx, :]
                    g_non_spec = g_all[:, non_spec_token_indx, :]
                    beta_non_spec = beta_all[:, non_spec_token_indx, :]
                a_spec = a.index_select(0, spec_token_indx) if mixed_qkv_spec is not None else None
                b_spec = b.index_select(0, spec_token_indx) if mixed_qkv_spec is not None else None
                a_non_spec = a.index_select(0, non_spec_token_indx) if mixed_qkv_non_spec is not None else None
                b_non_spec = b.index_select(0, non_spec_token_indx) if mixed_qkv_non_spec is not None else None
            else:
                # No spec decode: all tokens are non-spec
                g_spec = None
                beta_spec = None
                g_non_spec = g_all
                beta_non_spec = beta_all
                a_spec = None
                b_spec = None
                a_non_spec = a_for_gating
                b_non_spec = b_for_gating

            # Spec decode: sequential GDN
            core_attn_out_spec = None
            if spec_sequence_masks is not None and query_spec is not None:
                core_attn_out_spec = _pytorch_recurrent_gdn_decode(
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    g=g_spec,
                    beta=beta_spec,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    a=a_spec,
                    b=b_spec,
                    ssm_state=ssm_state,
                    state_indices=spec_state_indices_tensor,
                    cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                    scale=scale,
                    use_l2norm=True,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    use_sigmoid_gating=False,
                )

            # Non-spec: prefill or decode
            core_attn_out_non_spec = None
            if attn_metadata.num_prefills > 0 and query_non_spec is not None:
                # Prefill: sequential GDN with precomputed g, beta
                core_attn_out_non_spec = _pytorch_recurrent_gdn_prefill(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    ssm_state=ssm_state,
                    state_indices=non_spec_state_indices_tensor,
                    cu_seqlens=non_spec_query_start_loc,
                    scale=scale,
                    use_l2norm=True,
                )
            elif attn_metadata.num_decodes > 0 and query_non_spec is not None:
                # Decode: sequential GDN with precomputed g, beta
                core_attn_out_non_spec = _pytorch_recurrent_gdn_decode(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    a=a_non_spec,
                    b=b_non_spec,
                    ssm_state=ssm_state,
                    state_indices=non_spec_state_indices_tensor,
                    cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],
                    scale=scale,
                    use_l2norm=True,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    use_sigmoid_gating=False,
                )

        elif attn_metadata.num_decodes > 0 and query_non_spec is not None:
            # Path 2: Decode only - sigmoid gating
            core_attn_out_non_spec = _pytorch_recurrent_gdn_decode(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=None,
                beta=None,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                a=a,
                b=b,
                ssm_state=ssm_state,
                state_indices=non_spec_state_indices_tensor,
                cu_seqlens=non_spec_query_start_loc,
                scale=scale,
                use_l2norm=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                use_sigmoid_gating=True,
            )
            core_attn_out_spec = None
        else:
            core_attn_out_spec = None
            core_attn_out_non_spec = None

        # ---- Merge output ----
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),  # type: ignore[union-attr]
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)[:num_actual_tokens]
        elif spec_sequence_masks is not None and core_attn_out_spec is not None:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)[:num_actual_tokens]
        elif core_attn_out_non_spec is not None:
            if not enable_sp():
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
            else:
                core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)[:num_actual_tokens]


# Apply the monkey-patch
Qwen3_5GatedDeltaNet._forward_core = PyTorchQwen3_5GatedDeltaNet._forward_core
