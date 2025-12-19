# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# mypy: ignore-errors
import warnings
from typing import Optional

import torch
from einops import rearrange
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from vllm.model_executor.layers.fla.ops.utils import SUPPRESS_LEVEL
from vllm.forward_context import get_forward_context

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_delta_hupdate import chunk_gated_delta_rule_fwd_hupdate
from .chunk_o import chunk_fwd_o
from .chunk_o_update import chunk_fwd_o_update
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .solve_tril import solve_tril
from .utils import input_guard
from .wy_fast import recompute_w_u_fwd
from vllm_ascend.utils import prefill_context_parallel_enable

from .utils import prepare_final_chunk_indices

# isort: off
if prefill_context_parallel_enable():
    from vllm.distributed import get_pcp_group, get_tp_group


def chunk_gated_delta_rule_fwd(q: torch.Tensor,
                               k: torch.Tensor,
                               v: torch.Tensor,
                               g: torch.Tensor,
                               beta: torch.Tensor,
                               scale: float,
                               initial_state: torch.Tensor,
                               output_final_state: bool,
                               cu_seqlens: Optional[torch.LongTensor] = None):

    forward_context = get_forward_context()
    num_decodes = 0
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is not None and isinstance(attn_metadata, dict):
        attn_metadata = next(iter(attn_metadata.values()), None)
    if attn_metadata is not None:
        num_decodes = attn_metadata.num_decodes
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(k=k,
                                 beta=beta,
                                 g_cumsum=g,
                                 cu_seqlens=cu_seqlens,
                                 output_dtype=torch.float32)
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # if get_pcp_group().rank_in_group == 1 and get_tp_group().rank_in_group == 1:
    #     print(">>>>>>>>", k, k.shape)
    # if get_tp_group().rank_in_group == 1:
    #     print(">>>>>>>>", k[:, -5:, :, :], k.shape)

    if get_pcp_group().world_size > 1:
        h_update = chunk_gated_delta_rule_fwd_hupdate(
            k=k,
            w=w,
            u=u,
            g=g,
            cu_seqlens=cu_seqlens,
            num_decodes=num_decodes,
        )
        # h_update = h.new_empty(h.shape[0], h.shape[1] + 1, h.shape[2], h.shape[3], h.shape[4])
        # final_state torch.Size([4, 8, 128, 128])
        if get_pcp_group().rank_in_group == 0:
            first_final_state = final_state
        else:
            first_final_state = torch.empty_like(final_state)
        get_pcp_group().broadcast(first_final_state, src=0)

        final_chunk_indices = prepare_final_chunk_indices(cu_seqlens, 64)
        final_h_update = h_update[:, final_chunk_indices, :, :, :]
        all_final_h_update = get_pcp_group().all_gather(final_h_update, 0)

        updated_state = final_state.new_empty(get_pcp_group().world_size, *final_state.shape)
        updated_state[0, ...] = first_final_state
        for i in range(1, get_pcp_group().world_size):
            updated_final_state = final_state + torch.matmul(all_final_h_update[i, ...], updated_state[i - 1, ...])
            updated_state[i, ...] = updated_final_state

        final_state = updated_state[-1, ...]
        # if get_pcp_group().rank_in_group == 1 and get_tp_group().rank_in_group == 1:
        #     print(">>>>>>>>", final_state, final_state.shape)

        if get_pcp_group().rank_in_group == 0:
            updated_h_state = torch.zeros_like(final_state)
        else:
            updated_h_state = updated_state[get_pcp_group().rank_in_group - 1, ...]

        h = chunk_fwd_o_update(
            q=q,
            v=v_new,
            h=h,
            h_update=h_update,
            updated_h_state=updated_h_state,
            cu_seqlens=cu_seqlens,
        )
    # print()
    # if get_tp_group().rank_in_group == 1:
    #     print(">>>>>>>>", final_state, final_state.shape)

    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    # if get_pcp_group().rank_in_group == 0 and get_tp_group().rank_in_group == 0:
    #     print(">>>>>>>>>>>>>>S", torch.isnan(final_state).any())
    #     print(">>>>>>>>>>>>>>v_new", torch.isnan(v_new).any())
    #     print(">>>>>>>>>>>>>>h", torch.isnan(h).any())
    #     print(">>>>>>>>>>>>>>h_update", torch.isnan(h_update).any())
        # print(">>>>>>>>>>>>>>out", o[0, 1, 0, :])

    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                g: torch.Tensor,
                beta: torch.Tensor,
                scale: float,
                initial_state: torch.Tensor,
                output_final_state: bool,
                cu_seqlens: Optional[torch.LongTensor] = None,
                use_qk_l2norm_in_kernel: bool = False):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)
        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_gated_delta_rule(q: torch.Tensor,
                           k: torch.Tensor,
                           v: torch.Tensor,
                           g: torch.Tensor,
                           beta: torch.Tensor,
                           scale: float = None,
                           initial_state: torch.Tensor = None,
                           output_final_state: bool = False,
                           cu_seqlens: Optional[torch.LongTensor] = None,
                           head_first: bool = False,
                           use_qk_l2norm_in_kernel: bool = False):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(
        beta.shape
    ) == 3, "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
            stacklevel=2)
        q, k, v, beta, g = map(
            lambda x: rearrange(x, 'b h t ... -> b t h ...'),
            (q, k, v, beta, g))
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
            stacklevel=2)
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.")
        if initial_state is not None and initial_state.shape[0] != len(
                cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1]**-0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens,
        use_qk_l2norm_in_kernel)
    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    return o, final_state