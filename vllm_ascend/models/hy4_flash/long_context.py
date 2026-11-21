# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Eager paged sparse attention for the HY4 Flash text route."""

import torch
import torch_npu
from torch import nn
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.hy_v4_flash.nvidia.indexpool_ops import (
    INDEX_DIM,
    expand_groups,
    fwht_quantize,
    gated_pool,
    rotate_indexer,
    select_groups,
)

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import attention_transfer_window
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND

_SCORE_CHUNK_SIZE = 128


class AscendHYV4FlashPagedBackend(AscendAttentionBackend):
    @classmethod
    def is_sparse(cls) -> bool:
        # SFA consumes separate contiguous K and V allocations.
        return True

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str=""):
        # The sparse ACLNN kernel consumes block-major BSND pages.
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_head_size_v(cls, head_size: int, head_size_v: int) -> bool:
        return (head_size, head_size_v) in ((512, 64), (INDEX_DIM, INDEX_DIM))


def make_paged_cache(vllm_config, prefix, key_dim, value_dim):
    return Attention(
        1,
        key_dim,
        1.0,
        num_kv_heads=1,
        head_size_v=value_dim,
        cache_config=vllm_config.cache_config,
        prefix=prefix,
        attn_backend=AscendHYV4FlashPagedBackend,
    )


def _cache_rows(cache, block_table, positions):
    block_size = cache.shape[1]
    blocks = block_table[(positions // block_size).long()].long()
    return cache[blocks, positions % block_size, 0]


def _write_cache_rows(cache, slots, values):
    block_size = cache.shape[1]
    cache[slots // block_size, slots % block_size, 0] = values


def _quantized_vectors(vectors):
    quantized, scales = fwht_quantize(vectors.reshape(-1, INDEX_DIM).bfloat16())
    return (quantized.float() * scales[:, None]).to(torch.bfloat16).view_as(vectors)


def _swa_indices(positions, window):
    first = (positions + 1 - window).clamp_min(0)
    recent = first[:, None] + torch.arange(window, device=positions.device)
    return recent.masked_fill(recent > positions[:, None], -1).int()


class AscendHYV4FlashIndexPool(nn.Module):
    """Use trained HY4 pooling parameters with request-scoped paged buffers."""

    def __init__(self, vllm_config, config, prefix):
        super().__init__()
        self.ratio = config.compressed_indexer_compress_ratio
        self.group_budget = config.compressed_indexer_group_topk - 1
        self.num_heads = config.compressed_indexer_num_attention_heads
        self.rope_dim = config.qk_rope_head_dim
        self.rope_parameters = config.rope_parameters
        self.eps = config.rms_norm_eps
        for name, input_size, output_size in (
            ("query_projection", config.q_lora_rank, self.num_heads * INDEX_DIM),
            ("head_weight_projection", config.hidden_size, self.num_heads),
            ("token_key_projection", config.hidden_size, INDEX_DIM),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    input_size,
                    output_size,
                    bias=False,
                    quant_config=vllm_config.quant_config,
                    prefix=f"{prefix}.{name}",
                ),
            )
        self.token_key_norm = nn.RMSNorm(INDEX_DIM, eps=self.eps)
        self.key_value_projection = nn.Linear(INDEX_DIM, INDEX_DIM, bias=False, dtype=torch.float32)
        self.key_gate_projection = nn.Linear(INDEX_DIM, INDEX_DIM, bias=False, dtype=torch.float32)
        self.key_norm = nn.RMSNorm(INDEX_DIM, eps=self.eps, dtype=torch.float32)
        self.intra_group_ape = nn.Parameter(torch.empty(self.ratio, INDEX_DIM, dtype=torch.float32))
        self.cache = make_paged_cache(vllm_config, f"{prefix}.paged_keys", INDEX_DIM, INDEX_DIM)

    def _write_completed_groups(self, raw_cache, pool_cache, block_tables, ranges, device):
        owners = []
        group_ids = []
        for request, _, _, context, length in ranges:
            first, end = context // self.ratio, length // self.ratio
            group_ids.extend(range(first, end))
            owners.extend([request] * (end - first))
        if not group_ids:
            return
        owners = torch.tensor(owners, device=device)
        group_ids = torch.tensor(group_ids, device=device)
        # The CUDA implementation anchors each group at its first token.
        starts = group_ids * self.ratio
        group_tokens = starts[:, None] + torch.arange(self.ratio, device=device)
        block_size = raw_cache.shape[1]
        raw_blocks = block_tables[owners[:, None], group_tokens // block_size].long()
        raw = raw_cache[raw_blocks, group_tokens % block_size, 0].float()
        pooled = gated_pool(
            self.key_value_projection(raw),
            self.key_gate_projection(raw),
            self.intra_group_ape,
            self.key_norm.weight,
            self.eps,
        ).bfloat16()
        pooled = _quantized_vectors(rotate_indexer(pooled, starts, self.rope_dim, self.rope_parameters))
        block_size = pool_cache.shape[1]
        blocks = block_tables[owners, group_ids // block_size].long()
        pool_cache[blocks, group_ids % block_size, 0] = pooled

    def _select_indices(self, query, weights, keys, positions):
        output = []
        for start in range(0, query.shape[0], _SCORE_CHUNK_SIZE):
            end = start + _SCORE_CHUNK_SIZE
            q, w, pos = query[start:end], weights[start:end], positions[start:end]
            if keys.shape[0] > self.group_budget:
                projected = q.reshape(-1, INDEX_DIM) @ keys.T
                scores = (projected.view(q.shape[0], self.num_heads, -1).float().relu_() * w[..., None]).sum(1)
                chosen = select_groups(scores, pos // self.ratio, self.group_budget)
            else:
                # Every complete group fits in the selection budget.
                chosen = torch.arange(self.group_budget, device=q.device, dtype=torch.int32).expand(q.shape[0], -1)
            output.append(expand_groups(chosen, pos, self.ratio))
        return torch.cat(output)

    def forward(self, hidden, q_c, positions):
        metadata = get_forward_context().attn_metadata[self.cache.layer_name]
        count = metadata.num_actual_tokens
        positions = positions[:count]
        hidden, q_c = hidden[:count], q_c[:count]
        raw_cache, pool_cache = self.cache.kv_cache
        raw_keys = self.token_key_norm(self.token_key_projection(hidden)[0]).bfloat16()
        _write_cache_rows(raw_cache, metadata.slot_mapping[:count].long(), raw_keys)
        ranges = []
        start = 0
        for request, (end, length) in enumerate(zip(metadata.actual_seq_lengths_q, metadata.seq_lens_list)):
            if end > start:
                ranges.append((request, start, end, length - (end - start), length))
            start = end
        block_tables = metadata.block_tables
        self._write_completed_groups(raw_cache, pool_cache, block_tables, ranges, q_c.device)
        width = (self.group_budget + 1) * self.ratio
        if max(metadata.seq_lens_list) <= width:
            # Every history group and the incomplete tail fit in this route.
            return _swa_indices(positions, width)

        query = self.query_projection(q_c)[0].reshape(-1, self.num_heads, INDEX_DIM)
        query = _quantized_vectors(rotate_indexer(query, positions, self.rope_dim, self.rope_parameters))
        weights = self.head_weight_projection(hidden)[0].float() * (self.num_heads * INDEX_DIM) ** -0.5
        outputs = []
        for request, start, end, _, length in ranges:
            key_positions = torch.arange(length // self.ratio, device=q_c.device)
            keys = _cache_rows(pool_cache, block_tables[request], key_positions)
            outputs.append(self._select_indices(query[start:end], weights[start:end], keys, positions[start:end]))
        return torch.cat(outputs)


def absorb_kv_projection(attention, dtype):
    weight = attention.kv_b_proj.weight
    if weight.dtype != dtype:
        raise ValueError(f"HY4 Flash long-context MLA needs a dense {dtype} kv_b_proj weight.")
    heads = attention.num_local_heads
    weight = torch_npu.npu_format_cast(weight.detach(), ACL_FORMAT_FRACTAL_ND)
    weight = weight.reshape(heads, attention.qk_nope_head_dim + attention.v_head_dim, attention.kv_lora_rank)
    key, value = weight.split((attention.qk_nope_head_dim, attention.v_head_dim), dim=1)
    attention._key_projection = key.contiguous()
    attention._value_projection = value.transpose(1, 2).contiguous()


def run_paged_sparse_attention(attention, hidden, q_c, positions, q, latent, rope, output):
    metadata_by_layer = get_forward_context().attn_metadata
    if metadata_by_layer is None:
        output.zero_()
        return
    metadata = metadata_by_layer[attention.attn.layer_name]
    tokens = metadata.num_actual_tokens
    output.zero_()
    if tokens == 0:
        return
    # Text-only MRoPE supplies identical coordinates on each axis.
    positions = positions[0, :tokens] if positions.ndim == 2 else positions[:tokens]
    if attention.is_window_layer:
        topk = _swa_indices(positions, attention.config.sliding_window)
    else:
        topk = attention.indexer(hidden[:tokens], q_c[:tokens], positions)
    kv_cache, rope_cache = attention.attn.kv_cache
    with attention_transfer_window():
        slots = metadata.slot_mapping[:tokens].long()
        _write_cache_rows(kv_cache, slots, latent[:tokens])
        _write_cache_rows(rope_cache, slots, rope[:tokens, 0])
        q_nope, q_rope = q[:tokens].split((attention.qk_nope_head_dim, attention.qk_rope_head_dim), dim=-1)
        q_absorbed = torch.bmm(q_nope.transpose(0, 1), attention._key_projection).transpose(0, 1).contiguous()
        attended, softmax_max, softmax_sum = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=q_absorbed,
            key=kv_cache,
            value=kv_cache,
            sparse_indices=topk[:, None, :].contiguous(),
            scale_value=attention.scaling,
            sparse_block_size=1,
            block_table=metadata.block_tables,
            actual_seq_lengths_query=metadata.query_start_loc[1:].contiguous(),
            actual_seq_lengths_kv=metadata.seq_lens_gpu[: len(metadata.seq_lens_list)].contiguous(),
            query_rope=q_rope.contiguous(),
            key_rope=rope_cache,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=True,
        )
        heads = attention.num_local_heads
        lse = softmax_max.reshape(tokens, heads) + softmax_sum.reshape(tokens, heads).log()
        sink_weight = (lse - attention.learnable_sink_param[None]).sigmoid()
        value = torch.bmm(attended.transpose(0, 1), attention._value_projection).transpose(0, 1)
        output[:tokens].copy_((value.float() * sink_weight[..., None]).flatten(1))
