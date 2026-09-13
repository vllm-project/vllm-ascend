# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch references and static fixtures for DeepSeek V4.1 tests."""

from typing import Any

import torch
import torch.nn.functional as F

from vllm_ascend.core.deepseek_v41 import (
    STATE_RING_ROWS,
    DeepseekV41CompressorStateSpec,
    DeepseekV41FullSpec,
    DeepseekV41IndexerSpec,
    DeepseekV41SWASpec,
)
from vllm_ascend.models.deepseek_v41.compressor import _read, text_config_of
from vllm_ascend.models.deepseek_v41.model import build_layer_plan


def build_v41_cache_specs(config: Any, vllm_config: Any, prefix: str = "model"):
    """Describe the source-shared cache graph for allocation tests."""
    config = text_config_of(config)
    block_size = vllm_config.cache_config.block_size
    if block_size <= 0 or block_size % 2:
        raise ValueError("V4.1 logical block_size must be a positive multiple of two")
    width = _read(config, "head_dim")
    index_width = _read(config, "index_head_dim")
    window = _read(config, "sliding_window")
    specs = {}
    for role in build_layer_plan(config).layers:
        attn_prefix = f"{prefix}.layers.{role.layer_idx}.self_attn"
        specs[f"{attn_prefix}.swa_cache"] = DeepseekV41SWASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=width,
            dtype=torch.bfloat16,
            sliding_window=window,
        )
        if not role.is_kv_source:
            continue
        specs[f"{attn_prefix}.long_kv_cache"] = DeepseekV41FullSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=width,
            dtype=torch.bfloat16,
            tokens_per_state=role.compress_ratio,
            storage_block_size=block_size // role.compress_ratio,
        )
        specs[f"{attn_prefix}.indexer.k_cache"] = DeepseekV41IndexerSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=index_width,
            dtype=torch.int8,
            tokens_per_state=role.compress_ratio,
            storage_block_size=block_size // role.compress_ratio,
            scale_dim=1,
            scale_dtype=torch.float16,
        )
        if role.compress_ratio == 2:
            specs[f"{attn_prefix}.compressor.state_cache"] = DeepseekV41CompressorStateSpec(
                block_size=STATE_RING_ROWS,
                num_kv_heads=1,
                head_size=2 * width,
                dtype=torch.float32,
            )
    return specs


def scatter_cache(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
    """PyTorch reference for writing flat slots into a paged cache."""
    cache = cache.squeeze(-2)
    slots = slots[: values.shape[0]].long()
    valid = slots >= 0
    physical = slots.clamp_min(0)
    pages = torch.div(physical, cache.shape[1], rounding_mode="floor")
    rows = physical.remainder(cache.shape[1])
    write_values = torch.where(
        valid.view((-1,) + (1,) * (values.ndim - 1)),
        values,
        torch.zeros_like(values),
    )
    cache[pages, rows] = write_values.to(cache.dtype)


def gather_cache_rows(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for reading rows from a block-strided cache view."""
    cache = cache.squeeze(-2)
    slots = slots.long()
    pages = torch.div(slots, cache.shape[1], rounding_mode="floor")
    rows = slots.remainder(cache.shape[1])
    return cache[pages, rows]


def paged_prefix(cache, block_table, length, block_size):
    """Materialize one request's logical prefix from a paged cache."""
    if length <= 0:
        return cache.new_empty((0, cache.shape[-1]))
    blocks = (length + block_size - 1) // block_size
    page_ids = block_table[:blocks].long()
    return cache.squeeze(-2).index_select(0, page_ids).flatten(0, 1)[:length]


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """PyTorch reference for level-one candidate block selection."""
    width = logits.shape[-1]
    if width == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(-1)
    num_blocks = scores.shape[-1]
    if not torch.is_tensor(compress_lens):
        compress_lens = torch.tensor(compress_lens, device=logits.device)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last,
        torch.inf,
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def select_index_topk(logits, compress_lens, index_topk):
    """PyTorch reference for chronological level-two position TopK."""
    width = logits.shape[-1]
    if width == 0:
        return torch.empty((*logits.shape[:-1], 0), dtype=torch.int32, device=logits.device)
    topk = min(index_topk, width)
    indices = logits.topk(topk, dim=-1, sorted=False).indices.sort(-1).values
    return torch.where(indices < compress_lens, indices, -1).int()


def small_op_attention(
    q,
    positions,
    swa_cache,
    swa_metadata,
    *,
    source_cache=None,
    source_metadata=None,
    compress_ratio=0,
    window_size=128,
    index_topk=512,
    compressed_indices=None,
    sinks=None,
    softmax_scale=1.0,
):
    """PyTorch reference for SWA plus selected compressed KV attention."""
    query_starts = swa_metadata.query_start_loc.tolist()
    seq_lens = swa_metadata.seq_lens.tolist()
    outputs = []
    for req_idx, (q_start, q_end) in enumerate(zip(query_starts[:-1], query_starts[1:])):
        seq_len = int(seq_lens[req_idx])
        local = paged_prefix(
            swa_cache,
            swa_metadata.block_table[req_idx],
            seq_len,
            swa_metadata.storage_block_size,
        )
        compressed = None
        if source_cache is not None:
            compressed_len = int(source_metadata.cache_seq_lens[req_idx])
            compressed = paged_prefix(
                source_cache,
                source_metadata.block_table[req_idx],
                compressed_len,
                source_metadata.storage_block_size,
            )
        for token_idx in range(q_start, q_end):
            position = int(positions[token_idx])
            local_start = max(0, position - window_size + 1)
            keys = local[local_start : position + 1]
            if compressed is not None:
                visible = (position + 1) // compress_ratio
                if compressed_indices is None:
                    selected = compressed[max(0, visible - index_topk) : visible]
                else:
                    indices = compressed_indices[token_idx].long()
                    indices = indices[(indices >= 0) & (indices < visible)]
                    selected = compressed.index_select(0, indices)
                keys = torch.cat((keys, selected))
            logits = torch.einsum("hd,kd->hk", q[token_idx].float(), keys.float())
            logits *= softmax_scale
            if sinks is not None:
                logits = torch.cat((logits, sinks.float().unsqueeze(-1)), -1)
                probs = logits.softmax(-1)[..., :-1]
            else:
                probs = logits.softmax(-1)
            outputs.append(torch.einsum("hk,kd->hd", probs, keys.float()))
    return torch.stack(outputs).to(q.dtype)


def compressor_ratio2_reference(compressor, x, start_pos: int, state_cache, state_block_table):
    """Reference ratio-2 compressor over one request's private FP32 ring."""
    if start_pos < 0 or x.ndim != 2:
        raise ValueError("Expected nonnegative start_pos and [tokens, hidden] input")
    if (
        state_cache is None
        or state_cache.ndim != 3
        or state_cache.shape[-1] != 2 * compressor.width
        or state_cache.dtype != torch.float32
    ):
        raise ValueError("Ratio2 requires paged FP32 [pages, block_size, 2*head_dim] state")
    if not isinstance(state_block_table, (list, tuple)):
        raise ValueError("Reference compressor requires a host list/tuple state_block_table")
    block_size = state_cache.shape[1]
    if block_size != STATE_RING_ROWS or len(state_block_table) != 1:
        raise ValueError("State requires one 32-row ring block per request")

    def state_row(position):
        offset = position % block_size
        physical_block = state_block_table[0]
        if not isinstance(physical_block, int) or not 0 < physical_block < state_cache.shape[0]:
            raise ValueError("Compressor state refers to an absent/null/out-of-range page")
        return state_cache[physical_block, offset]

    if x.shape[0]:
        first = start_pos - start_pos % compressor.ratio
        for position in range(first, start_pos + x.shape[0]):
            state_row(position)
    kv = compressor.wkv(x.float())
    score = compressor.wgate(x.float())
    completed = []
    for token in range(x.shape[0]):
        position = start_pos + token
        row = state_row(position)
        row[: compressor.width] = kv[token]
        row[compressor.width :] = score[token]
        if (position + 1) % compressor.ratio == 0:
            group = torch.stack([state_row(position - 1), row])
            pooled = (group[:, : compressor.width] * group[:, compressor.width :].softmax(dim=0)).sum(dim=0)
            completed.append(pooled)
    latent = torch.stack(completed).to(x.dtype) if completed else x.new_empty((0, compressor.width))
    return compressor.norm(latent)


def hc_mixes_reference(layer, x, hc_fn, hc_scale, hc_base):
    """Reference mHC coefficient construction."""
    x_float = x.float()
    flat = x_float.flatten(-2)
    mixes = F.linear(flat, hc_fn)
    mixes *= torch.rsqrt(flat.square().mean(-1, keepdim=True) + layer.norm_eps)
    pre, post, comb = mixes.split([layer.hc_mult, layer.hc_mult, layer.hc_mult * layer.hc_mult], -1)
    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[: layer.hc_mult]) + layer.hc_eps
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[layer.hc_mult : 2 * layer.hc_mult])
    comb = comb.unflatten(-1, (layer.hc_mult, layer.hc_mult))
    comb = comb * hc_scale[2] + hc_base[2 * layer.hc_mult :].view(layer.hc_mult, layer.hc_mult)
    comb = comb.softmax(-1) + layer.hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + layer.hc_eps)
    for _ in range(layer.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + layer.hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + layer.hc_eps)
    return pre, post, comb


def hc_post_reference(x, residual, post, comb):
    """Reference mHC residual expansion."""
    y = post.unsqueeze(-1) * x.unsqueeze(-2)
    y += (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(-3)
    return y.to(x.dtype)
