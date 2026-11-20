# SPDX-License-Identifier: Apache-2.0
"""Owner-scoped DCP warmup using the production launch constructors."""

from collections import defaultdict

import torch
from vllm.model_executor.warmup.jit_warmup_triton_helper import TritonWarmupTensor
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.dcp.dcp_a2a import (
    _fused_dcp_lse_combine_kernel,
    _launch_fused_dcp_lse_combine,
    _launch_pack_dcp_output_lse,
    _lse_pack_dim,
    _pack_dcp_output_lse_kernel,
    _use_fast_exchange,
)
from vllm_ascend.ops.triton.dcp.dcp_a2a_batched import (
    _fused_dcp_lse_combine_batched_kernel,
    _pack_dcp_output_lse_batched_kernel,
)
from vllm_ascend.ops.triton.dcp.dcp_jit import DcpJitKernel
from vllm_ascend.ops.triton.dcp.sfa_dcp_exchange import (
    _FULL_PACK_VECTOR_ELEMENTS,
    _PACK_BLOCK_ELEMENTS,
    _launch_exchange_t1_pack,
    _launch_exchange_tokens_pack,
    _pack_t1_peer,
    _pack_token_peer,
    _pack_token_peer_tiled,
)
from vllm_ascend.ops.triton.dcp.sfa_dcp_merge import _fused_merge_kernel, _launch_fused_merge
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

# Fixed wrappers reuse the original JITFunction objects and their native cache.
_DCP_KERNELS = (
    DcpJitKernel(_pack_dcp_output_lse_kernel),
    DcpJitKernel(_pack_dcp_output_lse_batched_kernel),
    DcpJitKernel(_fused_dcp_lse_combine_kernel),
    DcpJitKernel(_fused_dcp_lse_combine_batched_kernel),
    DcpJitKernel(_pack_t1_peer),
    DcpJitKernel(_pack_token_peer, ("tokens",)),
    DcpJitKernel(_pack_token_peer_tiled, ("tokens",)),
    DcpJitKernel(
        _fused_merge_kernel,
        ("lse_stride_rank", "lse_stride_axis1", "head_count", "total_tiles", "num_programs"),
    ),
)


class _Tensor(TritonWarmupTensor):
    @property
    def ndim(self):
        return len(self.shape)


def _token_representatives(max_tokens, heads, local_heads, head_dim, vector_cores):
    """Sample all scalar-specialization residues within each dispatch interval.

    Every varying stride/size is an integer multiple of T. Between dispatch
    thresholds, exact-one and T modulo 16 determine all ordinary scalar and
    pointer-alignment keys. No capture-size list or Cartesian stride product is
    needed. Include integer-width boundaries for large configured capacities.
    """
    cuts = {
        1,
        2,
        128,
        triton.cdiv(8 * vector_cores, heads),
        triton.cdiv(4 * vector_cores, local_heads),
        triton.cdiv(16 * vector_cores, local_heads),
        _FULL_PACK_VECTOR_ELEMENTS // local_heads + 1,
        max_tokens + 1,
    }
    for power in range(_PACK_BLOCK_ELEMENTS.bit_length()):
        cuts.add((1 << power) // local_heads + 1)
    factors = (
        1,
        heads,
        local_heads,
        head_dim,
        max(1, head_dim // 2),
        head_dim + 1,
        head_dim + 4,
        local_heads * (head_dim + 1),
        local_heads * (head_dim + 4),
        local_heads * (head_dim // 2 + 1),
        local_heads * (head_dim // 2 + 1) * 2,
        local_heads * triton.cdiv(head_dim, min(triton.next_power_of_2(head_dim), 512)),
    )
    cuts.update(triton.cdiv(1 << 31, factor) for factor in factors)
    ordered = sorted(cut for cut in cuts if 1 <= cut <= max_tokens + 1)
    return [token for lo, hi in zip(ordered, ordered[1:]) for token in range(lo, min(hi, lo + 16))]


def _output_and_lse(dtype, tokens, heads, head_dim, layout):
    output_strides = (heads * head_dim, head_dim, 1)
    lse_strides = (heads, 1, 1)
    if layout == "head_major":
        output_strides = (head_dim, tokens * head_dim, 1)
    if layout in ("head_major", "head_major_lse"):
        lse_strides = (1, tokens, 1)
    return (
        _Tensor(dtype, shape=(tokens, heads, head_dim), strides=output_strides),
        _Tensor(torch.float32, shape=(tokens, heads, 1), strides=lse_strides),
    )


def _record_exchange(output, lse, softmax_sum, ranks, vector_cores, launch):
    tokens, heads, head_dim = output.shape
    local_heads = heads // ranks
    head_words = head_dim // 2
    output_words = local_heads * head_words
    peer_words = tokens * (output_words + local_heads)
    send = _Tensor(torch.int32, shape=(ranks, peer_words))
    if tokens == 1:
        words = _Tensor(torch.int32, shape=(heads * head_words,))
        _launch_exchange_t1_pack(words, lse, softmax_sum, send, local_heads, head_words, ranks, launch)
        parts = _Tensor(
            output.dtype,
            shape=(ranks, local_heads, 1, head_dim),
            strides=(peer_words * 2, head_dim, head_dim, 1),
        )
        stats = _Tensor(
            torch.float32,
            aligned=(output_words * 4) % 16 == 0,
            shape=(ranks, local_heads, 1),
            strides=(peer_words, 1, 1),
        )
        token_dim = 2
    else:
        words = _Tensor(
            torch.int32,
            shape=(tokens, heads, head_words),
            strides=(output.stride(0) // 2, output.stride(1) // 2, 1),
        )
        _launch_exchange_tokens_pack(
            words, lse, softmax_sum, send, tokens, local_heads, head_words, ranks, lambda: vector_cores, launch
        )
        parts = _Tensor(
            output.dtype,
            shape=(ranks, tokens, local_heads, head_dim),
            strides=(peer_words * 2, local_heads * head_dim, head_dim, 1),
        )
        stats = _Tensor(
            torch.float32,
            aligned=(tokens * output_words * 4) % 16 == 0,
            shape=(ranks, tokens, local_heads),
            strides=(peer_words, local_heads, 1),
        )
        token_dim = 1
    result = _Tensor(torch.float32, shape=(tokens, local_heads, head_dim))
    _launch_fused_merge(parts, stats, result, token_dim, vector_cores, launch)


def _record_dcp_layout(
    dtype, tokens, local_heads, head_dim, scatter_size, pcp_size, from_stats, mla, layout, vector_cores, record
):
    """One owner layout, routed through the production launch constructors."""
    heads = local_heads * scatter_size
    partial, lse = _output_and_lse(dtype, tokens, heads, head_dim, layout)
    softmax_sum = lse if from_stats else None
    if _use_fast_exchange(partial, lse, scatter_size, 1, None if pcp_size == 1 else True, False, False):
        _record_exchange(partial, lse, softmax_sum, scatter_size, vector_cores, record)
        if not mla:
            return
    # SFA uses this path when fast exchange is ineligible. MLA's split
    # history always packs, including shapes eligible for fast exchange.
    packed_dim = head_dim + _lse_pack_dim(dtype)
    send = _Tensor(dtype, shape=(scatter_size, local_heads, tokens, packed_dim))
    _launch_pack_dcp_output_lse(partial, lse, softmax_sum, send, 1, vector_cores, record)
    recv = _Tensor(dtype, shape=(scatter_size * pcp_size, local_heads, tokens, packed_dim))
    output = _Tensor(dtype, shape=(tokens, local_heads, head_dim))
    _launch_fused_dcp_lse_combine(recv, output, head_dim, 1, False, None, None, vector_cores, record)
    if mla:
        local_output, local_lse = _output_and_lse(dtype, tokens, local_heads, head_dim, layout)
        _launch_fused_dcp_lse_combine(recv, output, head_dim, 1, False, local_output, local_lse, vector_cores, record)


def register_dcp_warmup(*, dtype, local_heads, head_dim, scatter_size, pcp_size, max_tokens, from_stats, mla=False):
    """Register SFA and MLA owners' reachable output-merge layouts.

    Native/QSFA allocates contiguous TND outputs and FP32 statistics. MLA also
    uses transposed FlashMLA output/LSE and singleton-batch BSND LSE views.
    The split MLA route defers history combine and may add one local result.
    """
    init_device_properties_triton()
    vector_cores = get_vectorcore_num()
    heads = local_heads * scatter_size
    owners = {owner.kernel: owner for owner in _DCP_KERNELS}
    keys: dict[DcpJitKernel, dict[DcpJitKernel.CompileKey, None]] = defaultdict(dict)

    def record(kernel, grid, *args, **kwargs):
        owner = owners[kernel]
        keys[owner][owner.key_from_launch(*args, **kwargs)] = None

    layouts = ("token_major", "head_major_lse", "head_major") if mla else ("token_major",)
    for tokens in _token_representatives(max_tokens, heads, local_heads, head_dim, vector_cores):
        for layout in layouts:
            _record_dcp_layout(
                dtype,
                tokens,
                local_heads,
                head_dim,
                scatter_size,
                pcp_size,
                from_stats,
                mla,
                layout,
                vector_cores,
                record,
            )
    for owner, owner_keys in keys.items():
        owner.register_warmup(tuple(owner_keys))
