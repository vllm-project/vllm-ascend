# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paged FP32 GDN checkpoints and accepted-prefix recovery on Ascend.

Q/K are normalized by the caller, matching the existing recurrent path.
Verification never overwrites the checkpoint. All records use the effective
post-normalization key and post-activation gates seen by verification.
"""

from dataclasses import dataclass

import torch
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton

PAGE_DESCRIPTOR_FIELDS: tl.constexpr = 5  # address followed by up to four strides
GDN_PAGE_COUNT: tl.constexpr = 5  # conv, checkpoint, correction, key, log-decay


@triton.jit
def _verify_kernel(
    Q,
    K,
    V,
    G,
    Beta,
    State,
    Correction,
    Keys,
    Gates,
    Starts,
    Indices,
    Out,
    stride_qt: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kt: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vt: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_gt: tl.constexpr,
    stride_gh: tl.constexpr,
    stride_bt: tl.constexpr,
    stride_bh: tl.constexpr,
    stride_s: tl.constexpr,
    stride_c: tl.constexpr,
    stride_k: tl.constexpr,
    stride_g: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_starts: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    L: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    block = tl.load(Indices + row * stride_indices).to(tl.int64)
    bos = tl.load(Starts + row * stride_starts)
    eos = tl.load(Starts + (row + 1) * stride_starts)
    vv = tile * BV + tl.arange(0, BV)
    if block <= 0:
        # Capture/profile rows can have a non-empty query but a null page.
        # Keep their outputs deterministic without touching checkpoint/records.
        for pos in range(tl.minimum(eos - bos, L)):
            tl.store(Out + ((bos + pos) * HV + head) * DV + vv, 0.0, vv < DV)
        return
    if bos == eos:
        return
    kh = head // (HV // HK)
    kk = tl.arange(0, BK)
    mask = (vv[:, None] < DV) & (kk[None, :] < DK)
    state = tl.load(State + block * stride_s + head * DV * DK + vv[:, None] * DK + kk[None, :], mask=mask, other=0).to(
        tl.float32
    )
    for pos in range(tl.minimum(eos - bos, L)):
        token = bos + pos
        q = tl.load(Q + token * stride_qt + kh * stride_qh + kk, kk < DK, 0).to(tl.float32)
        k = tl.load(K + token * stride_kt + kh * stride_kh + kk, kk < DK, 0).to(tl.float32)
        v = tl.load(V + token * stride_vt + head * stride_vh + vv, vv < DV, 0).to(tl.float32)
        g = tl.load(G + token * stride_gt + head * stride_gh).to(tl.float32)
        beta = tl.load(Beta + token * stride_bt + head * stride_bh).to(tl.float32)
        state = state * tl.exp(g)
        delta = (v - tl.sum(state * k[None, :], 1)) * beta
        state = state + delta[:, None] * k[None, :]
        out = tl.sum(state * q[None, :], 1) * (DK**-0.5)
        tl.store(Out + (token * HV + head) * DV + vv, out, vv < DV)
        tl.store(Correction + block * stride_c + (head * L + pos) * DV + vv, delta, vv < DV)
        if tile == 0:
            tl.store(Gates + block * stride_g + head * L + pos, g)
            if head % (HV // HK) == 0:
                tl.store(Keys + block * stride_k + (kh * L + pos) * DK + kk, k, kk < DK)


@triton.jit
def _commit_state_kernel(
    Pages,
    Indices,
    Starts,
    Accepted,
    Requests,
    stride_indices: tl.constexpr,
    stride_starts: tl.constexpr,
    stride_accepted: tl.constexpr,
    stride_requests: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    L: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_REQUESTS: tl.constexpr,
):
    row, head, layer_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    layer = layer_tile // tl.cdiv(DV, BV)
    tile = layer_tile % tl.cdiv(DV, BV)
    block = tl.load(Indices + row * stride_indices).to(tl.int64)
    request = row
    if HAS_REQUESTS:
        request = tl.load(Requests + row * stride_requests)
    bos = tl.load(Starts + row * stride_starts)
    eos = tl.load(Starts + (row + 1) * stride_starts)
    count = tl.minimum(tl.maximum(tl.load(Accepted + request * stride_accepted), 0), tl.minimum(eos - bos, L))
    if block <= 0 or count == 0:
        return
    # Each page descriptor contains address and four strides. Record/state
    # inner layouts are dense; block strides may include allocator padding.
    desc = Pages + layer * GDN_PAGE_COUNT * PAGE_DESCRIPTOR_FIELDS
    state_ptr = tl.load(desc + PAGE_DESCRIPTOR_FIELDS).to(tl.pointer_type(tl.float32))
    correction_ptr = tl.load(desc + 2 * PAGE_DESCRIPTOR_FIELDS).to(tl.pointer_type(tl.float32))
    key_ptr = tl.load(desc + 3 * PAGE_DESCRIPTOR_FIELDS).to(tl.pointer_type(tl.float32))
    gate_ptr = tl.load(desc + 4 * PAGE_DESCRIPTOR_FIELDS).to(tl.pointer_type(tl.float32))
    ss = tl.load(desc + PAGE_DESCRIPTOR_FIELDS + 1)
    cs = tl.load(desc + 2 * PAGE_DESCRIPTOR_FIELDS + 1)
    ks = tl.load(desc + 3 * PAGE_DESCRIPTOR_FIELDS + 1)
    gs = tl.load(desc + 4 * PAGE_DESCRIPTOR_FIELDS + 1)
    kk = tl.arange(0, BK)
    vv = tile * BV + tl.arange(0, BV)
    mask = (vv[:, None] < DV) & (kk[None, :] < DK)
    address = state_ptr + block * ss + head * DV * DK + vv[:, None] * DK + kk[None, :]
    state = tl.load(address, mask, 0)
    kh = head // (HV // HK)
    # Forward composition preserves verification's update order and avoids
    # re-evaluating the state-dependent residual during accepted-state recovery.
    for pos in range(count):
        delta = tl.load(correction_ptr + block * cs + (head * L + pos) * DV + vv, vv < DV, 0)
        key = tl.load(key_ptr + block * ks + (kh * L + pos) * DK + kk, kk < DK, 0)
        gate = tl.load(gate_ptr + block * gs + head * L + pos)
        state = state * tl.exp(gate) + delta[:, None] * key[None, :]
    tl.store(address, state, mask)


@triton.jit
def _commit_conv_kernel(
    ConvRef,
    Pages,
    Indices,
    Starts,
    Accepted,
    Requests,
    stride_indices: tl.constexpr,
    stride_starts: tl.constexpr,
    stride_accepted: tl.constexpr,
    stride_requests: tl.constexpr,
    CHANNELS: tl.constexpr,
    HISTORY: tl.constexpr,
    L: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_REQUESTS: tl.constexpr,
):
    row, tile, layer = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    block = tl.load(Indices + row * stride_indices).to(tl.int64)
    request = row
    if HAS_REQUESTS:
        request = tl.load(Requests + row * stride_requests)
    length = tl.load(Starts + (row + 1) * stride_starts) - tl.load(Starts + row * stride_starts)
    count = tl.minimum(tl.maximum(tl.load(Accepted + request * stride_accepted), 0), tl.minimum(length, L))
    if block <= 0 or count == 0:
        return
    desc = Pages + layer * GDN_PAGE_COUNT * PAGE_DESCRIPTOR_FIELDS
    conv = tl.load(desc).to(tl.pointer_type(ConvRef.dtype.element_ty))
    block_stride, dim_stride, token_stride = tl.load(desc + 1), tl.load(desc + 2), tl.load(desc + 3)
    channels = tile * BLOCK + tl.arange(0, BLOCK)
    base = conv + block * block_stride + channels * dim_stride
    # Forward, channel-disjoint copies are safe for an overlapping left shift.
    # 1D loads also support strided channels on A3's Triton backend.
    for pos in tl.static_range(HISTORY):
        values = tl.load(base + (count - 1 + pos) * token_stride, channels < CHANNELS, 0)
        tl.store(base + pos * token_stride, values, channels < CHANNELS)


def _check_dense_page(tensor, shape, device):
    if tensor.shape != shape or tensor.dtype != torch.float32 or tensor.device != device:
        raise ValueError(f"GDN RecoverSSM requires FP32 page {shape} on {device}")
    stride = 1
    for dim in range(tensor.ndim - 1, 0, -1):
        if tensor.stride(dim) != stride:
            raise ValueError("GDN RecoverSSM page inner dimensions must be contiguous")
        stride *= tensor.shape[dim]
    if tensor.stride(0) < stride:
        raise ValueError("GDN RecoverSSM pages must not overlap")


def gdn_recoverssm_verify(
    q, k, v, g, beta, checkpoint, corrections, keys, gates, query_start_loc, state_indices, spec_query_len
):
    """Return GDN outputs and compact records, leaving the checkpoint intact."""
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != q.shape:
        raise ValueError("GDN RecoverSSM Q/K must have matching [1,T,HK,K] shapes")
    _, tokens, hk, dk = q.shape
    if v.ndim != 4 or v.shape[:2] != (1, tokens):
        raise ValueError("GDN RecoverSSM V must have shape [1,T,HV,V]")
    hv, dv = v.shape[2:]
    if hk < 1 or hv % hk or spec_query_len < 1:
        raise ValueError("GDN RecoverSSM requires integral GQA and a positive window")
    if g.shape != (1, tokens, hv) or beta.shape != g.shape:
        raise ValueError("GDN RecoverSSM requires one gate and beta per value head")
    tensors = (q, k, v, g, beta, checkpoint, corrections, keys, gates, query_start_loc, state_indices)
    if any(t.device != q.device for t in tensors):
        raise ValueError("GDN RecoverSSM inputs must share a device")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("GDN RecoverSSM Q/K/V channels must be contiguous")
    blocks = checkpoint.shape[0]
    _check_dense_page(checkpoint, (blocks, hv, dv, dk), q.device)
    _check_dense_page(corrections, (blocks, hv, spec_query_len, dv), q.device)
    _check_dense_page(keys, (blocks, hk, spec_query_len, dk), q.device)
    _check_dense_page(gates, (blocks, hv, spec_query_len), q.device)
    rows = state_indices.numel()
    if state_indices.ndim != 1 or query_start_loc.shape != (rows + 1,):
        raise ValueError("GDN RecoverSSM request indices and offsets do not match")
    if tokens > rows * spec_query_len:
        raise ValueError("GDN RecoverSSM input exceeds its record capacity")
    if any(t.dtype not in (torch.int32, torch.int64) for t in (state_indices, query_start_loc)):
        raise ValueError("GDN RecoverSSM request metadata must use integer tensors")
    out = torch.empty_like(v, memory_format=torch.contiguous_format)
    if rows and tokens:
        _verify_kernel[(rows, hv, triton.cdiv(dv, 16))](
            q,
            k,
            v,
            g,
            beta,
            checkpoint,
            corrections,
            keys,
            gates,
            query_start_loc,
            state_indices,
            out,
            q.stride(1),
            q.stride(2),
            k.stride(1),
            k.stride(2),
            v.stride(1),
            v.stride(2),
            g.stride(1),
            g.stride(2),
            beta.stride(1),
            beta.stride(2),
            checkpoint.stride(0),
            corrections.stride(0),
            keys.stride(0),
            gates.stride(0),
            state_indices.stride(0),
            query_start_loc.stride(0),
            HK=hk,
            HV=hv,
            DK=dk,
            DV=dv,
            L=spec_query_len,
            BK=triton.next_power_of_2(dk),
            BV=16,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@dataclass
class GDNRecoverSSMCommitContext:
    pages: tuple
    descriptors: torch.Tensor
    spec_query_len: int
    max_num_reqs: int
    conv_history_len: int

    @classmethod
    def create(cls, layers, *, spec_query_len, max_num_reqs):
        if not layers or spec_query_len < 1 or max_num_reqs < 1:
            raise ValueError("GDN RecoverSSM requires layers and positive capacities")
        pages = []
        descriptors = []
        for layer in layers:
            if len(layer.kv_cache) != GDN_PAGE_COUNT:
                raise ValueError("GDN RecoverSSM requires conv, checkpoint, correction, key and gate pages")
            conv, state, correction, keys, gates = layer.kv_cache
            conv = conv if is_conv_state_dim_first() else conv.transpose(-1, -2)
            blocks, hv, dv, dk = state.shape
            hk = keys.shape[1]
            if hk < 1 or hv % hk:
                raise ValueError("GDN RecoverSSM requires integral GQA")
            _check_dense_page(state, (blocks, hv, dv, dk), state.device)
            _check_dense_page(correction, (blocks, hv, spec_query_len, dv), state.device)
            _check_dense_page(keys, (blocks, hk, spec_query_len, dk), state.device)
            _check_dense_page(gates, (blocks, hv, spec_query_len), state.device)
            channels = 2 * hk * dk + hv * dv
            if conv.ndim != 3 or conv.shape[:2] != (blocks, channels) or conv.device != state.device:
                raise ValueError("GDN RecoverSSM convolution page has incompatible layout")
            history = conv.shape[2] - spec_query_len + 1
            if history < 1:
                raise ValueError("GDN RecoverSSM convolution history is too short")
            page = (conv, state, correction, keys, gates)
            if pages and any(
                (a.shape, a.dtype, a.device) != (b.shape, b.dtype, b.device) for a, b in zip(page, pages[0])
            ):
                raise ValueError("GDN RecoverSSM layers must share shapes, dtypes and device")
            pages.append(page)
            for tensor in page:
                descriptors.append(
                    [tensor.data_ptr(), *tensor.stride(), *([0] * (PAGE_DESCRIPTOR_FIELDS - 1 - tensor.ndim))]
                )
        return cls(
            tuple(pages),
            torch.tensor(descriptors, dtype=torch.int64, device=pages[0][1].device),
            spec_query_len,
            max_num_reqs,
            history,
        )

    def commit(
        self,
        num_accepted_tokens,
        state_indices,
        query_start_loc,
        *,
        request_indices=None,
        block_table=None,
        num_computed_tokens=None,
        mamba_block_size=None,
    ):
        if block_table is not None or num_computed_tokens is not None or mamba_block_size is not None:
            raise ValueError("GDN RecoverSSM currently supports Mamba cache mode none only")
        batch = state_indices.numel()
        if batch > self.max_num_reqs or state_indices.ndim != 1 or query_start_loc.shape != (batch + 1,):
            raise ValueError("GDN RecoverSSM commit request metadata exceeds capacity or has incompatible shape")
        if request_indices is not None and request_indices.shape != (batch,):
            raise ValueError("GDN RecoverSSM sampler mapping must have one row per request")
        integer_tensors = (num_accepted_tokens, state_indices, query_start_loc)
        if request_indices is not None:
            integer_tensors += (request_indices,)
        if any(t.ndim != 1 or t.dtype not in (torch.int32, torch.int64) for t in integer_tensors):
            raise ValueError("GDN RecoverSSM commit metadata must be one-dimensional integer tensors")
        if request_indices is None and num_accepted_tokens.numel() < batch:
            raise ValueError("GDN RecoverSSM acceptance counts are too short")
        device = self.descriptors.device
        if any(t.device != device for t in (num_accepted_tokens, state_indices, query_start_loc)):
            raise ValueError("GDN RecoverSSM commit tensors must share the cache device")
        if request_indices is not None and request_indices.device != device:
            raise ValueError("GDN RecoverSSM sampler mapping must be on the cache device")
        if not batch:
            return
        conv, state, _, keys, _ = self.pages[0]
        _, hv, dv, dk = state.shape
        args = (
            self.descriptors,
            state_indices,
            query_start_loc,
            num_accepted_tokens,
            request_indices,
            state_indices.stride(0),
            query_start_loc.stride(0),
            num_accepted_tokens.stride(0),
            request_indices.stride(0) if request_indices is not None else 0,
        )
        _commit_state_kernel[(batch, hv, len(self.pages) * triton.cdiv(dv, 16))](
            *args,
            HK=keys.shape[1],
            HV=hv,
            DK=dk,
            DV=dv,
            L=self.spec_query_len,
            BK=triton.next_power_of_2(dk),
            BV=16,
            HAS_REQUESTS=request_indices is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _commit_conv_kernel[(batch, triton.cdiv(conv.shape[1], 128), len(self.pages))](
            conv,
            *args,
            CHANNELS=conv.shape[1],
            HISTORY=self.conv_history_len,
            L=self.spec_query_len,
            BLOCK=128,
            HAS_REQUESTS=request_indices is not None,
            num_warps=4,
        )
