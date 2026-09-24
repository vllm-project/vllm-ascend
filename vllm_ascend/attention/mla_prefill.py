# SPDX-License-Identifier: Apache-2.0
"""Bounded non-absorbed MLA prefill using the compressed persistent cache.

The backend owns compressed KV writes, Q/KV compression, RoPE and O projection.
This module consumes their outputs and never changes the permanent cache layout.

The current vllm-ascend branch stores the K3 MLA cache as a ``(latent, position)``
BF16 pair ([blocks, 128, 1, 512] and [blocks, 128, 1, 64]); :func:`full_prefill`
accepts that pair as well as the equivalent first-axis-strided single view.
"""

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from vllm.logger import logger


def _flash_binding_source() -> str:
    """Name the package that will serve ``cann_ops_transformer.ops.flash_attn``.

    The operator is served by the CANN-bundled ``cann_ops_transformer``; the
    ``flash_attn_manifest.json`` check is a leftover of the former csrc overlay
    build and reports the resolved package directory only.
    """
    import cann_ops_transformer

    package_dir = os.path.dirname(os.path.abspath(cann_ops_transformer.__file__))
    manifest = os.path.join(os.path.dirname(package_dir), "flash_attn_manifest.json")
    if os.path.isfile(manifest):
        return f"the self-compiled csrc overlay at {package_dir} (manifest: {manifest})"
    return f"the CANN-bundled package at {package_dir} (no csrc manifest)"


@dataclass(frozen=True)
class HistoryChunk:
    requests: tuple[int, ...]
    starts: tuple[int, ...]
    lengths: tuple[int, ...]
    query_lengths: tuple[int, ...]
    query_indices: tuple[int, ...] | None

    @property
    def num_tokens(self) -> int:
        return sum(self.lengths)


@dataclass(frozen=True)
class PrefillPlan:
    query_lengths: tuple[int, ...]
    context_lengths: tuple[int, ...]
    chunks: tuple[HistoryChunk, ...]
    num_decode_tokens: int
    workspace_tokens: int
    kernel_block_size: int = 128

    @property
    def num_tokens(self) -> int:
        return sum(self.query_lengths)


def _cumulative(lengths: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return tuple(offsets)


def build_prefill_plan(
    query_start_loc_cpu: Sequence[int],
    num_decodes: int,
    exact_prefill_context_lengths: Sequence[int],
    workspace_tokens: int,
    kernel_block_size: int = 128,
) -> PrefillPlan:
    """Plan only the prefill suffix using accurate runner-owned CPU counts.

    `exact_prefill_context_lengths` contains *only* prefill requests. In
    particular this function never accepts optimistic decode sequence lengths.
    The chunk width matches AscendMLAMetadataBuilder.build_chunked_metadata.
    """
    if isinstance(query_start_loc_cpu, torch.Tensor) and query_start_loc_cpu.device.type != "cpu":
        raise ValueError("query_start_loc_cpu must already be on CPU")
    if isinstance(exact_prefill_context_lengths, torch.Tensor) and exact_prefill_context_lengths.device.type != "cpu":
        raise ValueError("prefill context lengths must already be on CPU")
    offsets = tuple(int(x) for x in query_start_loc_cpu)
    contexts = tuple(int(x) for x in exact_prefill_context_lengths)
    if not 0 <= num_decodes < len(offsets):
        raise ValueError("invalid decode request prefix")
    query_lengths = tuple(b - a for a, b in zip(offsets[num_decodes:], offsets[num_decodes + 1 :]))
    if len(contexts) != len(query_lengths) or any(x < 0 for x in contexts) or any(x <= 0 for x in query_lengths):
        raise ValueError("prefill query/context metadata mismatch")
    if kernel_block_size != 128 or workspace_tokens < kernel_block_size:
        raise ValueError("the compressed cache uses kernel block128 and needs at least one workspace page")
    local_offsets = _cumulative(query_lengths)
    positive = sum(x > 0 for x in contexts)
    # Normally one group, as in the existing metadata builder. Splitting request
    # groups also keeps the allocation bounded if the budget is smaller than B pages.
    group_capacity = max(1, workspace_tokens // kernel_block_size)
    width = max(kernel_block_size, workspace_tokens // max(1, positive) // kernel_block_size * kernel_block_size)
    chunks = []
    for start in range(0, max(contexts, default=0), width):
        active = tuple(i for i, length in enumerate(contexts) if length > start)
        for group_start in range(0, len(active), group_capacity):
            requests = active[group_start : group_start + group_capacity]
            lengths = tuple(min(width, contexts[i] - start) for i in requests)
            indices = None
            if len(requests) != len(query_lengths):
                indices = tuple(t for i in requests for t in range(local_offsets[i], local_offsets[i + 1]))
            chunks.append(
                HistoryChunk(
                    requests, (start,) * len(requests), lengths, tuple(query_lengths[i] for i in requests), indices
                )
            )
    return PrefillPlan(query_lengths, contexts, tuple(chunks), offsets[num_decodes], workspace_tokens)


@dataclass
class AttentionCall:
    query_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]
    query_cu: torch.Tensor
    kv_cu: torch.Tensor
    query_used: torch.Tensor
    kv_used: torch.Tensor
    mask_mode: int
    schedule: torch.Tensor | None = None


@dataclass
class DeviceHistoryChunk:
    plan: HistoryChunk
    request_indices: torch.Tensor | None
    query_indices: torch.Tensor | None
    starts: torch.Tensor
    call: AttentionCall


@dataclass
class PrefillMetadata:
    plan: PrefillPlan
    current: AttentionCall
    history: tuple[DeviceHistoryChunk, ...]


def prepare_metadata(plan: PrefillPlan, device: torch.device, schedule: Callable | None = None) -> PrefillMetadata:
    """CPU→device metadata is prepared once per batch, outside model layers."""

    calls: dict[tuple[tuple[int, ...], tuple[int, ...], int], AttentionCall] = {}

    def tensor(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device=device)

    def call(query_lengths, kv_lengths, mode):
        key = (query_lengths, kv_lengths, mode)
        if key in calls:
            return calls[key]
        item = AttentionCall(
            query_lengths,
            kv_lengths,
            tensor(_cumulative(query_lengths)),
            tensor(_cumulative(kv_lengths)),
            tensor(query_lengths),
            tensor(kv_lengths),
            mode,
        )
        if schedule is not None:
            item.schedule = schedule(item)
        # Absolute history offsets live in DeviceHistoryChunk.starts. These
        # read-only lengths and schedules can be shared by equal-sized chunks.
        calls[key] = item
        return item

    history = []
    for chunk in plan.chunks:
        partial = chunk.query_indices is not None
        history.append(
            DeviceHistoryChunk(
                chunk,
                tensor(chunk.requests, torch.int64) if partial else None,
                tensor(chunk.query_indices, torch.int64) if partial else None,
                tensor(chunk.starts),
                call(chunk.query_lengths, chunk.lengths, 0),
            )
        )
    return PrefillMetadata(plan, call(plan.query_lengths, plan.query_lengths, 3), tuple(history))


def canonical_lse(lse: torch.Tensor, tokens: int, heads: int) -> torch.Tensor:
    # Native FlashAttn returns H,T; FIA can return T,H,1 on B060. Never
    # transpose a 3-D LSE: doing so mixes query/head merge weights.
    if lse.dim() == 2:
        lse = lse.transpose(0, 1).unsqueeze(-1)
    if tuple(lse.shape) != (tokens, heads, 1):
        raise ValueError(f"unexpected LSE shape {tuple(lse.shape)}")
    return lse.float()


def full_prefill(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    current_latent: torch.Tensor,
    current_k_pe: torch.Tensor,
    cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    block_table: torch.Tensor,
    metadata: PrefillMetadata,
    kv_b_proj: Callable,
    attention: Callable,
    kv_cache_load: Callable,
    attention_update: Callable,
    *,
    query: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute only prefill rows; online merging bounds history output memory.

    `cache` is either the first-axis-strided BF16 [pages,128,576] view or the
    persistent BF16 pair (latent [pages,128,1,512], positional keys
    [pages,128,1,64]); both describe the same physical BF16 blocks.
    """
    tokens, heads, nope_dim = q_nope.shape
    if isinstance(cache, tuple):
        if len(cache) != 2:
            raise ValueError("MLA prefill requires a matching latent/positional cache pair")
        latent_cache, position_cache = cache
        if latent_cache.dtype != q_nope.dtype or position_cache.dtype != q_nope.dtype:
            raise ValueError("BF16 prefill requires model-dtype latent and positional keys")
    else:
        if cache.shape[1:] != (128, 576):
            raise ValueError("BF16 prefill cache layout mismatch")
        latent_cache, position_cache = cache[..., :512].unsqueeze(2), cache[..., 512:].unsqueeze(2)
    if (
        tokens != metadata.plan.num_tokens
        or latent_cache.shape[1:] != (128, 1, 512)
        or position_cache.shape[1:] != (128, 1, 64)
        or latent_cache.shape[0] != position_cache.shape[0]
    ):
        raise ValueError("prefill token count or compressed cache layout mismatch")
    if query is None:
        query = torch.cat((q_nope, q_pe), dim=-1)

    def expand_attention(latent, position, packed_query, call):
        projected = kv_b_proj(latent.reshape(-1, 512))
        if isinstance(projected, tuple):
            projected = projected[0]
        kv = projected.view(-1, heads, nope_dim + 128)
        key_nope, value = kv.split((nope_dim, 128), dim=-1)
        # Keep the projection's split views: the adapter packs K/V from these
        # views itself, so materializing K192 here would only duplicate them.
        out, lse = attention(packed_query, key_nope, value, position, call)
        return out, canonical_lse(lse, packed_query.shape[0], heads)

    # Every prefill query has at least itself in current KV. This starts with
    # finite LSE, including requests with no prefix, before any history merge.
    cumulative_out, cumulative_lse = expand_attention(current_latent, current_k_pe, query, metadata.current)
    if not metadata.history:
        return cumulative_out, cumulative_lse
    cumulative_out = cumulative_out.float()
    query_view_requests = None
    query_view = None
    for chunk in metadata.history:
        gather_tokens = chunk.plan.num_tokens
        table = block_table if chunk.request_indices is None else block_table.index_select(0, chunk.request_indices)
        latent = torch.empty((gather_tokens, 1, 512), dtype=latent_cache.dtype, device=latent_cache.device)
        position = torch.empty((gather_tokens, 1, 64), dtype=position_cache.dtype, device=position_cache.device)
        kv_cache_load(
            latent_cache,
            position_cache,
            table,
            chunk.call.kv_used,
            chunk.starts,
            key=latent,
            value=position,
        )
        if chunk.query_indices is None:
            chunk_query = query
            previous_out, previous_lse = cumulative_out, cumulative_lse
        else:
            # Reuse the packed query for repeated history segments with the
            # same active requests; empty-history rows never enter the kernel.
            if query_view_requests != chunk.plan.requests:
                query_view = query.index_select(0, chunk.query_indices)
                query_view_requests = chunk.plan.requests
            chunk_query = query_view
            previous_out = cumulative_out.index_select(0, chunk.query_indices)
            previous_lse = cumulative_lse.index_select(0, chunk.query_indices)
        chunk_out, chunk_lse = expand_attention(latent, position, chunk_query, chunk.call)
        chunk_out = chunk_out.float()
        merged_out, merged_lse = attention_update(
            (previous_lse.reshape(-1), chunk_lse.reshape(-1)),
            (previous_out.reshape(-1, 128), chunk_out.reshape(-1, 128)),
            1,
        )
        if chunk.query_indices is None:
            cumulative_out = merged_out.view(tokens, heads, 128)
            cumulative_lse = merged_lse.view(tokens, heads, 1)
        else:
            cumulative_out.index_copy_(0, chunk.query_indices, merged_out.view(-1, heads, 128))
            cumulative_lse.index_copy_(0, chunk.query_indices, merged_lse.view(-1, heads, 1))
    return cumulative_out.to(q_nope.dtype), cumulative_lse


def native_flash_adapters(
    num_heads: int,
    scale: float,
    attn_mask: torch.Tensor,
    *,
    bf16_prepare: bool = False,
):
    """Use a matched CANN 192/128 binding/runtime; no custom backend or switch.

    The binding must expose ``head_dim_v`` so that QK192 can attend V128; the
    CANN-bundled ``cann_ops_transformer`` provides it, and
    :func:`_flash_binding_source` reports which package serves the call.
    """
    from cann_ops_transformer.ops import flash_attn, flash_attn_metadata

    logger.info_once(
        "A5 Flash MLA prefill FlashAttn is served by %s, which carries the head_dim_v "
        "ABI needed for QK192/V128; the CANN-bundled operator keeps head_dim_v == head_dim.",
        _flash_binding_source(),
    )

    if bf16_prepare:
        from vllm_ascend.ops.flash_mla_bf16_prepare import prepare_flash_mla_bf16

    def kwargs(call):
        return dict(
            cu_seqlens_q=call.query_cu,
            cu_seqlens_kv=call.kv_cu,
            seqused_q=call.query_used,
            seqused_kv=call.kv_used,
            max_seqlen_q=max(call.query_lengths),
            max_seqlen_kv=max(call.kv_lengths),
            mask_mode=call.mask_mode,
            layout_q="TND",
            layout_kv="TND",
            layout_out="TND",
        )

    def schedule(call):
        return flash_attn_metadata(
            num_heads, num_heads, 192, head_dim_v=128, batch_size=len(call.query_lengths), **kwargs(call)
        )

    def attention(query, key_nope, value, key_rope, call):
        if bf16_prepare:
            key, value = prepare_flash_mla_bf16(key_nope, value, key_rope)
        else:
            key = torch.cat((key_nope, key_rope.reshape(-1, 1, 64).expand(-1, num_heads, -1)), dim=-1)
            value = value.contiguous()
        return flash_attn(
            query,
            key,
            value,
            metadata=call.schedule,
            attn_mask=attn_mask if call.mask_mode else None,
            softmax_scale=scale,
            return_softmax_lse=True,
            **kwargs(call),
        )

    return schedule, attention
