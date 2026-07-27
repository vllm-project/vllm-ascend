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
import torch
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.25.1"):
    from vllm.model_executor.layers.fla.ops.utils import SUPPRESS_LEVEL  # type: ignore[import-not-found]
else:
    from vllm.third_party.flash_linear_attention.ops.utils import SUPPRESS_LEVEL

from .chunk_delta_hupdate import chunk_gated_delta_rule_fwd_hupdate
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .solve_tril import solve_tril as solve_tril_fast
from .utils import input_guard, prepare_chunk_indices, prepare_final_chunk_indices


def _as_host_tuple(values):
    if values is None:
        return None
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(int(v) for v in values)
    if isinstance(values, torch.Tensor):
        return tuple(int(v) for v in values.detach().cpu().reshape(-1).tolist())
    return tuple(int(v) for v in values)


def _prepare_chunk_indices_if_needed(cu_seqlens, chunk_indices, chunk_size: int):
    if cu_seqlens is None or chunk_indices is not None:
        return chunk_indices
    if isinstance(cu_seqlens, torch.Tensor):
        return prepare_chunk_indices(cu_seqlens, chunk_size)
    return None


def solve_tril(
    A: torch.Tensor,
    cu_seqlens=None,
    chunk_indices_large_block=None,
    chunk_indices_bt=None,
    chunk_indices_32=None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    output_dtype = A.dtype if output_dtype is None else output_dtype
    A_for_kernel = A.to(output_dtype).contiguous()
    if cu_seqlens is not None and not isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = torch.tensor(cu_seqlens, device=A.device, dtype=torch.int64)
    prebuilt_indices = {
        "1216": chunk_indices_large_block,
        "32": chunk_indices_32,
        str(A_for_kernel.shape[-1]): chunk_indices_bt,
    }
    return solve_tril_fast(
        A_for_kernel,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=prebuilt_indices,
        output_dtype=output_dtype,
    )


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens=None,
    chunk_indices=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run recompute with the kernel-native BHTD/BHTC layout."""
    chunk_size = A.shape[-1]
    chunk_indices = _prepare_chunk_indices_if_needed(cu_seqlens, chunk_indices, chunk_size)
    return torch.ops._C_ascend.npu_recompute_wu_fwd(
        k,
        v,
        beta.to(g_cumsum.dtype),
        A,
        g_cumsum,
        cu_seqlens=_as_host_tuple(cu_seqlens),
        chunk_indices=_as_host_tuple(chunk_indices),
        chunk_size=chunk_size,
    )


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fwd-h with the kernel-native BHTD layout."""
    del chunk_offsets
    chunk_indices = _prepare_chunk_indices_if_needed(cu_seqlens, chunk_indices, chunk_size)
    return torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k.to(torch.bfloat16).contiguous(),
        w.to(torch.bfloat16).contiguous(),
        u.to(torch.bfloat16).contiguous(),
        g=None if g is None else g.contiguous(),
        gk=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
        cu_seqlens=_as_host_tuple(cu_seqlens),
        chunk_indices=_as_host_tuple(chunk_indices),
        use_exp2=False,
        transpose_state_layout=False,
    )


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens=None,
    chunk_size: int = 64,
    chunk_indices=None,
    chunk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fwd-o with the kernel-native BHTD layout."""
    del chunk_offsets
    if scale is None:
        scale = k.shape[-1] ** -0.5
    chunk_indices = _prepare_chunk_indices_if_needed(cu_seqlens, chunk_indices, chunk_size)
    return torch.ops._C_ascend.chunk_fwd_o(
        q.to(torch.bfloat16).contiguous(),
        k.to(torch.bfloat16).contiguous(),
        v.contiguous(),
        h,
        scale,
        g=None if g is None else g.contiguous(),
        g_gamma=None,
        cu_seqlens=_as_host_tuple(cu_seqlens),
        chunk_indices=_as_host_tuple(chunk_indices),
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    prebuilt_meta=None,
):
    forward_context = get_forward_context()
    num_decodes = 0
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is not None and isinstance(attn_metadata, dict):
        attn_metadata = next(iter(attn_metadata.values()), None)
    if attn_metadata is not None:
        num_decodes = attn_metadata.num_decodes
    chunk_size = 64
    block_indices_cumsum = None if prebuilt_meta is None else prebuilt_meta.block_indices_cumsum
    cu_seqlens_host = None if prebuilt_meta is None else prebuilt_meta.cu_seqlens_host
    chunk_indices_chunk64 = None if prebuilt_meta is None else prebuilt_meta.chunk_indices_chunk64
    chunk_indices_chunk64_host = None if prebuilt_meta is None else prebuilt_meta.chunk_indices_chunk64_host
    chunk_offsets_chunk64 = None if prebuilt_meta is None else prebuilt_meta.chunk_offsets_chunk64
    update_chunk_offsets_chunk64 = None if prebuilt_meta is None else prebuilt_meta.update_chunk_offsets_chunk64
    final_chunk_indices_chunk64 = None if prebuilt_meta is None else prebuilt_meta.final_chunk_indices_chunk64
    chunk_indices_large_block = None if prebuilt_meta is None else prebuilt_meta.chunk_indices_large_block
    chunk_indices_chunk32 = None if prebuilt_meta is None else getattr(prebuilt_meta, "chunk_indices_chunk32", None)

    cu_seqlens = None if cu_seqlens is None else cu_seqlens.to(torch.int64)
    if cu_seqlens is not None and chunk_indices_chunk64 is None and chunk_indices_chunk64_host is None:
        chunk_indices_chunk64 = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunk_indices = None if chunk_indices_chunk64 is None else chunk_indices_chunk64.to(torch.int64)
    if cu_seqlens_host is None and cu_seqlens is not None:
        cu_seqlens_host = _as_host_tuple(cu_seqlens)
    if chunk_indices_chunk64_host is None and chunk_indices is not None:
        chunk_indices_chunk64_host = _as_host_tuple(chunk_indices)
    # Compact zero-length segments for the AscendC kernels (see
    # _compact_empty_segments).  chunk_indices_chunk64 is already compact-
    # ranked and is reused as-is; only cu_seqlens / initial_state need
    # compacting.
    if prebuilt_meta is not None and hasattr(prebuilt_meta, "keep_meta"):
        cu_seqlens_kern = cu_seqlens_host if prebuilt_meta.cu_seqlens_kern is None else prebuilt_meta.cu_seqlens_kern
        keep_meta = prebuilt_meta.keep_meta
        initial_state_kern = (
            initial_state[keep_meta] if initial_state is not None and keep_meta is not None else initial_state
        )
    else:
        cu_seqlens_kern, initial_state_kern = cu_seqlens_host, initial_state
        keep_meta = None
    # Conv1D, KKT, and the AscendC recompute/fwd-h/fwd-o kernels consume BHTD.
    # g/beta retain the BTH layout emitted by fused_gdn_gating. Only the
    # AscendC gate input requires a materialized layout conversion.
    g_bth = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        block_indices=block_indices_cumsum,
    )
    q_bhtd = q.to(torch.bfloat16)
    k_bhtd = k.to(torch.bfloat16)
    v_bhtd = v
    g_bht = g_bth.movedim(1, 2).contiguous()

    # Obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k_bhtd,
        beta=beta,
        g_cumsum=g_bth,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices_large_block=chunk_indices_large_block,
        chunk_indices_bt=chunk_indices_chunk64,
        chunk_indices_32=chunk_indices_chunk32,
        output_dtype=k.dtype,
    )
    # KKT and solve-tril preserve BHTC, matching the ACLNN recompute input.
    A_bhtc = A
    beta_bht = beta.movedim(1, 2).contiguous()

    w, u = recompute_w_u_fwd(
        k=k_bhtd,
        v=v_bhtd,
        beta=beta_bht,
        A=A_bhtc,
        g_cumsum=g_bht,
        cu_seqlens=cu_seqlens_host,
        chunk_indices=chunk_indices_chunk64_host,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k_bhtd,
        w=w,
        u=u,
        g=g_bht,
        initial_state=initial_state_kern,
        output_final_state=True,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_kern,
        chunk_indices=chunk_indices_chunk64_host,
    )
    if keep_meta is not None:
        # Scatter the compacted final_state back to the original [N, H, K, V]
        # layout the PCP state recursion expects; empty segments keep their
        # initial state.
        _fs_full = initial_state.clone()
        _fs_full[keep_meta] = final_state
        final_state = _fs_full

    if get_pcp_group().world_size > 1:
        # When integrating mtp, since `mix_qkv` has been split, `num_decode`
        # cannot be directly obtained from the metadata and needs to be recalculated.
        actual_num_decodes = getattr(prebuilt_meta, "num_decodes", None)
        if actual_num_decodes is None:
            actual_num_decodes = num_decodes
        # The PCP h-update kernel still has a BTHD interface.
        k_bthd = k_bhtd.movedim(1, 2).contiguous()
        h_update = chunk_gated_delta_rule_fwd_hupdate(
            k=k_bthd,
            w=w.movedim(1, 2).contiguous(),
            u=u.movedim(1, 2).contiguous(),
            g=g_bth,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices_chunk64,
            chunk_offsets=chunk_offsets_chunk64,
            update_chunk_offsets=update_chunk_offsets_chunk64,
            num_decodes=actual_num_decodes,
        )
        all_final_state = get_pcp_group().all_gather(final_state.unsqueeze(0), 0)
        final_chunk_indices = final_chunk_indices_chunk64
        if final_chunk_indices is None:
            final_chunk_indices = prepare_final_chunk_indices(cu_seqlens, chunk_size)
        final_h_update = h_update[:, final_chunk_indices, :, :, :]
        all_final_h_update = get_pcp_group().all_gather(final_h_update, 0)

        updated_state = final_state.new_empty(get_pcp_group().world_size, *final_state.shape)
        updated_state[0, ...] = all_final_state[0]
        for i in range(1, get_pcp_group().world_size):
            # correct_i = all_final_state[i] + Φ_i · (correct_{i-1} - s0) = Φ_i · correct_{i-1} + p_i
            updated_final_state = all_final_state[i] + torch.matmul(
                all_final_h_update[i, ...], updated_state[i - 1, ...] - initial_state
            )
            updated_state[i, ...] = updated_final_state

        final_state = updated_state[-1, ...]

        if get_pcp_group().rank_in_group == 0:
            updated_h_state = torch.zeros_like(final_state)
        else:
            updated_h_state = updated_state[get_pcp_group().rank_in_group - 1, ...]

        if get_pcp_group().rank_in_group > 0:
            rerun_initial_state = initial_state.clone()
            prefill_seq_offset = actual_num_decodes
            prefill_slice = slice(prefill_seq_offset, final_state.shape[0])
            rerun_initial_state[prefill_slice] = updated_h_state[prefill_slice]
            h, v_new, _ = chunk_gated_delta_rule_fwd_h(
                k=k_bhtd,
                w=w,
                u=u,
                g=g_bht,
                initial_state=rerun_initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices_chunk64,
                chunk_offsets=chunk_offsets_chunk64,
            )

    o = (
        chunk_fwd_o(
            q=q_bhtd,
            k=k_bhtd,
            v=v_new,
            h=h,
            g=g_bht,
            scale=scale,
            cu_seqlens=cu_seqlens_host,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices_chunk64_host,
            chunk_offsets=chunk_offsets_chunk64,
        )
        .movedim(1, 2)
        .contiguous()
    )

    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    return (
        g,
        o,
        A,
        final_state,
        w.movedim(1, 2).contiguous(),
        h.movedim(1, 2).contiguous(),
        v_new.movedim(1, 2).contiguous(),
    )


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        prebuilt_meta=None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
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
            prebuilt_meta=prebuilt_meta,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    prebuilt_meta=None,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    core_attn_out: torch.Tensor | None = None,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, H, T, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
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
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, H, T, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, H, T, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, H, T, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v = map(lambda x: rearrange(x, 'b h t d -> 1 h (b t) d'), (q, k, v))
        >>> beta, g = map(lambda x: rearrange(x, 'b t h -> 1 (b t) h'), (beta, g))
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
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H]."

    if (
        q.shape[0] != g.shape[0]
        or q.shape[2] != g.shape[1]
        or k.shape[0] != g.shape[0]
        or k.shape[2] != g.shape[1]
        or v.shape[0] != g.shape[0]
        or v.shape[2] != g.shape[1]
        or beta.shape != g.shape
    ):
        raise ValueError("chunk_gated_delta_rule expects BHTD Q/K/V and BTH g/beta inputs.")
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"chunk_gated_delta_rule: The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"chunk_gated_delta_rule: The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        prebuilt_meta,
        use_qk_l2norm_in_kernel,
    )
    return o, final_state
