from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group, is_v1_kv_transfer_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend.device.utils import (
    FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE,
    _gather_paged_kv_to_dense,
)
from vllm_ascend.utils import (
    AscendDeviceType,
    get_ascend_config,
    get_ascend_device_type,
    is_pd_decode_recompute_scheduler_enabled,
)
from vllm_ascend.worker.kvcomp_utils import KVCompMetaData


@dataclass
class PagedAttentionGraphParam:
    """Mark PA params when PA and FIA share one graph replay list."""

    params: tuple
    layer_name: str | None

    def __iter__(self):
        return iter(self.params)


def expand_paged_kv_to_per_query(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_speculative_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand per-seq block_table/seq_lens to per-query for MTP verify.

    MTP verify: target processes K+1 query tokens per seq (positions c..c+K).
    seq_lens[i] = c + K + 1 (KV already written for all K+1 tokens). The PA
    kernel is per-query-row: a single context_len makes all K+1 query rows
    attend the full KV, so token0 sees draft1's KV (future leak) -> logits[0]
    polluted. Expand so token j (j=0..K) attends positions 0..c+j:
        context_len = seq_lens - K + j   (i.e. [s-K, s-K+1, ..., s])
        block_table row repeated K+1 times (same blocks, context_len truncates).
    No-op when shapes already match (num_tokens == num_seqs) or K == 0.
    """
    k = num_speculative_tokens
    num_seqs = seq_lens.shape[0]
    num_tokens = num_seqs * (k + 1)
    if k <= 0 or block_table.shape[0] == num_tokens:
        return block_table, seq_lens
    base = seq_lens.to(torch.int32) - k
    offsets = torch.arange(k + 1, dtype=torch.int32, device=seq_lens.device)
    context_lens = (base.unsqueeze(1) + offsets.unsqueeze(0)).reshape(-1)
    block_table = block_table.repeat_interleave(k + 1, dim=0)
    return block_table, context_lens


def update_paged_attention_graph_param(
    update_stream,
    handle,
    event,
    param: PagedAttentionGraphParam,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    (
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        num_heads,
        scale,
        _captured_block_table,
        _captured_seq_lens,
        output,
    ) = param.params
    workspace = torch_npu._npu_paged_attention_get_workspace(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        scale_value=scale,
        block_table=block_table,
        context_lens=seq_lens,
        out=output,
    )
    torch.npu.graph_task_update_begin(update_stream, handle)
    torch_npu._npu_paged_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
        scale_value=scale,
        block_table=block_table,
        context_lens=seq_lens,
        out=output,
        workspace=workspace,
    )
    torch.npu.graph_task_update_end(update_stream)
    event.record(update_stream)


def cache_graph_workspace(
    graph_params,
    num_tokens: int,
    candidate_workspace: torch.Tensor,
    *,
    use_max_workspace: bool,
) -> torch.Tensor:
    # Most models keep the original first-workspace cache behavior. Models with
    # mixed attention layer shapes may need the largest workspace for a graph
    # size because layers can require different FIA workspace sizes.
    current_workspace = graph_params.workspaces.get(num_tokens)
    if use_max_workspace:
        if current_workspace is None or (
            candidate_workspace.numel() * candidate_workspace.element_size()
            > current_workspace.numel() * current_workspace.element_size()
        ):
            graph_params.workspaces[num_tokens] = candidate_workspace
    elif current_workspace is None:
        graph_params.workspaces[num_tokens] = candidate_workspace
    return graph_params.workspaces[num_tokens]


@lru_cache(maxsize=1)
def needs_layer_aware_fia_graph_replay() -> bool:
    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    text_config = getattr(hf_config, "text_config", None)
    model_types = (
        getattr(hf_config, "model_type", None),
        getattr(hf_text_config, "model_type", None),
        getattr(text_config, "model_type", None),
    )
    return any(model_type in {"gemma4", "gemma4_text"} for model_type in model_types)


def ascend_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
    scheduler_config = vllm_config.scheduler_config
    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config

    chunked_prefill_workspace_size = min(
        # Make sure there is enough for 8 full length request or at least
        # 4 pages of cache per request
        max(8 * model_config.max_model_len, 4 * scheduler_config.max_num_seqs * cache_config.block_size),
        # For long-context models try not to over-allocate limiting
        # kv-cache space, limiting it to 128k tokens,
        # which would result in the workspace being:
        #   2*(576)*(128*1024) = 288mb
        # (assuming 576 MLA head dim, and fp16)
        # which would result in up-projected context being
        #   2*(192*128)*(128*1024) = 6gb
        # (assuming 192 QK head dim, 128 heads, and fp16)
        128 * 1024,
    )

    chunked_prefill_workspace_size = max(
        chunked_prefill_workspace_size,
        scheduler_config.max_num_seqs * cache_config.block_size,
    )

    return chunked_prefill_workspace_size


def using_paged_attention(runtime_shape: int, vllm_config: VllmConfig, head_size: int | None = None) -> bool:
    if get_ascend_device_type() == AscendDeviceType.A5:
        return False
    # A2/A3 FIA (TND) does not support head_dim=512 (Gemma4 global attention):
    # 512-dim decode must go through PagedAttention even with MTP enabled,
    # because FIA TND is semantically/numerically wrong for 512 and drops
    # MTP pos0 acceptance from ~80% to ~60%.
    # TODO: Remove this fallback when A2/A3 FIA TND supports Gemma4's
    # 512-dim global attention heads. Prefill is handled by the device adaptor.
    if head_size == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE:
        return True
    # Non-512 heads: keep PA disabled under MTP (original behavior).
    if vllm_config.speculative_config is not None:
        return False
    from vllm.config.compilation import CUDAGraphMode

    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    if cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        return False

    return runtime_shape in get_ascend_config().pa_shape_list


@lru_cache(maxsize=1)
def enable_cp():
    prefill_config = get_current_vllm_config().parallel_config
    return prefill_config.prefill_context_parallel_size > 1 or prefill_config.decode_context_parallel_size > 1


@dataclass
class AscendPrefillContextParallelMetadata:
    """
    Metadata for Prefill Context Parallelism (PCP) in CommonAttentionMetadata.

    Contains index tensors and sequence lengths for PCP operations.
    """

    pcp_allgather_restore_idx: torch.Tensor = None

    num_actual_tokens_pcp_padded: int = 0

    num_computed_tokens_of_pcp_dcp: list[list[list[int]]] | None = None

    q_head_idx_tensor: torch.Tensor = None

    q_tail_idx_tensor: torch.Tensor = None

    kv_with_q_head_nomask_idx_tensor: torch.Tensor = None

    kv_with_q_head_mask_idx_tensor: torch.Tensor = None

    kv_with_q_tail_nomask_idx_tensor: torch.Tensor = None

    kv_with_q_tail_mask_idx_tensor: torch.Tensor = None

    kv_tail_proj_idx_tensor: torch.Tensor = None

    kv_with_q_head_attn_idx_in_tail_tensor: torch.Tensor = None

    kv_with_q_tail_attn_idx_in_tail_tensor: torch.Tensor = None

    attn_mask_seqlens: torch.Tensor = None

    head_attn_nomask_seqlens: torch.Tensor = None

    tail_attn_nomask_seqlens: torch.Tensor = None

    head_actual_seq_lengths_kv: list[int] | None = None

    tail_actual_seq_lengths_kv: list[int] | None = None

    q_full_idx: torch.Tensor = None

    # original query_lens before pcp split
    query_lens_pcp_full_cpu: torch.Tensor = None

    # original max_query_len before pcp split
    max_query_len_pcp_full: int = 0

    # the following attributes are specifically used in hybrid-attn models.
    pcp_use_hybrid_attn: bool = False

    pcp_unpad_mask: torch.Tensor = None

    # to get the right order of query in prefill per rank
    pcp_fa_query_idx: torch.Tensor = None

    # restore the full sequence across all pcp ranks
    # when entering from linear-attention to attention
    pcp_enter_fa_restore_idx: torch.Tensor = None

    # scatter the full sequence across all pcp ranks
    # when exiting from attention to linear-attention
    pcp_exit_fa_scatter_idx: torch.Tensor = None

    # the number of tokens padded in linear-attn per rank
    pcp_padded_tokens_fla: int = 0

    # the max number of unpadded tokens in all ranks
    max_num_tokens_across_pcp: int = 0

    # the number of scheduled tokens on the current rank before padding
    total_num_scheduled_tokens: int = 0

    # Because the sequence shard in linear attention layers does not include padding,
    # the full attention layers cannot obtain the correct query_lens with pcp pad for
    # chunked prefill calculation. Therefore, this value needs to be passed to the backend.
    # TODO:To be refactored.
    attn_chunk_seqlens: torch.Tensor = None
    dcp_mtp_attn_mask: torch.Tensor = None


@dataclass
class AscendCommonAttentionMetadata(CommonAttentionMetadata):
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.

    For many of the tensors we keep both NPU and CPU versions.
    """

    # CPU tensor of sequence lengths for host-side operations.
    # E.g., tensor([128, 256, 64]) for 3 requests with different seq lengths.
    seq_lens_cpu: torch.Tensor = None

    # CPU tensor of already computed tokens count per request.
    # E.g., tensor([100, 200, 50]) means req0 has 100 tokens already computed.
    num_computed_tokens_cpu: torch.Tensor = None

    # Number of decode tokens per request, used for speculative decoding.
    # E.g., 1 for normal decoding, >1 for speculative decoding.
    decode_token_per_req: int = 1

    # Actual query sequence lengths for each token in the batch (CPU list).
    # E.g., [1, 1, 1, 128] for 3 decode tokens and 1 prefill with 128 tokens.
    actual_seq_lengths_q: list[int] = field(default_factory=list)

    # NPU tensor of position indices for rotary embeddings computation.
    # E.g., tensor([0, 1, 2, ...]) indicating token positions in sequence.
    positions: torch.Tensor = None
    positions_cpu: torch.Tensor = None

    # CPU tensor of slot mapping for host-side operations.
    slot_mapping_cpu: torch.Tensor = None

    # Current attention state (e.g., ChunkedPrefill, DecodeOnly).
    attn_state: Any = None

    # Padding size for graph capture, -1 means not in graph mode.
    graph_pad_size: int = -1

    # Total number of tokens including padding, used for padding operations.
    num_input_tokens: int = 0

    # Metadata for Prefill Context Parallelism (PCP) operations.
    prefill_context_parallel_metadata: AscendPrefillContextParallelMetadata | None = None
    kvcomp_metadata: KVCompMetaData | None = None

    # TODO: Remove it when vLLM no longer uses this function.
    def unpadded(self, num_actual_tokens: int, num_actual_reqs: int) -> "AscendCommonAttentionMetadata":
        # This only use to eagle now. It will be use to enforce_eager in future.
        # Helper to slice optional per-request tensors to ``num_actual_reqs``.
        def _slice_reqs(x):
            return x[:num_actual_reqs] if x is not None else None

        return AscendCommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: num_actual_reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: num_actual_reqs + 1],
            seq_lens=self.seq_lens[:num_actual_reqs],
            seq_lens_cpu=_slice_reqs(self.seq_lens_cpu),
            num_computed_tokens_cpu=_slice_reqs(self.num_computed_tokens_cpu),
            num_reqs=num_actual_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=self.max_query_len,
            decode_token_per_req=self.decode_token_per_req,
            # NOTE: keep all tokens for block_table_tensor and slot_mapping otherwise
            # there will be error about shape mismatch during reshape and cache.
            # This is really strange since vLLM slices them as well
            block_table_tensor=self.block_table_tensor,
            slot_mapping=self.slot_mapping,
            slot_mapping_cpu=self.slot_mapping_cpu,
            causal=self.causal,
            actual_seq_lengths_q=self.actual_seq_lengths_q[:num_actual_tokens],
            positions=self.positions,
            positions_cpu=self.positions_cpu,
            attn_state=self.attn_state,
            graph_pad_size=-1,  # It should be -1 when not run in fullgraph mode.
            num_input_tokens=self.num_input_tokens,
            prefill_context_parallel_metadata=self.prefill_context_parallel_metadata,
            seq_lens_cpu_upper_bound=self.seq_lens_cpu_upper_bound[:num_actual_reqs]
            if self.seq_lens_cpu_upper_bound is not None
            else None,
            max_seq_len=self.max_seq_len,
            # Propagate parent-class fields so the unpadded view is a
            # faithful sub-batch of the original. Missing any of these
            # would silently break downstream consumers (e.g. NPU
            # backends preferring ``_seq_lens_cpu`` over ``seq_lens_cpu``,
            # DCP backends needing ``dcp_local_seq_lens(_cpu)``,
            # encoder-decoder layers needing ``encoder_seq_lens``, the
            # mamba ``is_prefilling`` flag, and FastPrefill's
            # ``logits_indices_padded`` / ``num_logits_indices``).
            _seq_lens_cpu=_slice_reqs(self._seq_lens_cpu),
            _num_computed_tokens_cpu=_slice_reqs(self._num_computed_tokens_cpu),
            dcp_local_seq_lens=_slice_reqs(self.dcp_local_seq_lens),
            dcp_local_seq_lens_cpu=_slice_reqs(self.dcp_local_seq_lens_cpu),
            is_prefilling=_slice_reqs(self.is_prefilling),
            encoder_seq_lens=_slice_reqs(self.encoder_seq_lens),
            encoder_seq_lens_cpu=_slice_reqs(self.encoder_seq_lens_cpu),
            logits_indices_padded=self.logits_indices_padded,
            num_logits_indices=self.num_logits_indices,
        )


def filter_chunked_req_indices(
    seq_len: torch.Tensor,
    mask_for_non_zero_chunk: list[bool] | None,
) -> torch.Tensor:
    """
    filter the reqs which are doing real chunk_prefill.

    Args:
        seq_len: contains multi-req length: [req0_len, req1_len, ...]
        mask_for_non_zero_chunk: [True, False, True, False, ...]
    Returns:
        filtered_indices: the real chunked req's indices
    """
    assert mask_for_non_zero_chunk is not None and len(seq_len) == len(mask_for_non_zero_chunk)
    offsets = torch.cumsum(torch.cat([torch.tensor([0]), seq_len[:-1]]), dim=0)
    filtered_indices = torch.cat(
        [
            torch.arange(offsets[i], offsets[i] + seq_len[i])
            for i in range(len(mask_for_non_zero_chunk))
            if mask_for_non_zero_chunk[i]
        ]
    )
    return filtered_indices


def split_decodes_and_prefills(
    common_attn_metadata: AscendCommonAttentionMetadata,
    decode_threshold: int = 1,
    require_uniform: bool = False,
    treat_short_extends_as_decodes: bool = True,
) -> tuple[int, int, int, int]:
    """
    Assuming a reordered batch, finds the boundary between prefill and decode
    requests.
    While pcp > 1, query_lens is split across pcp ranks, so we pass in the
    original query_lens and max_query_len to distinguish prefills and decodes.

    The batch is expected to be ordered as:
    decode -> short_extend -> long_extend -> prefill

    Args:
        common_attn_metadata: AscendCommonAttentionMetadata object containing the
            batch metadata.
        decode_threshold: The maximum query length to be considered a decode.
        require_uniform: If True, requires that all decode requests have the
            same query length. When set, some queries may be considered
            prefills even if they are <= decode_threshold, in order to ensure
            uniformity.
        treat_short_extends_as_decodes: If True (default), short extends
            (query_len <= threshold but still prefilling) are counted as
            decodes. If False, they are counted as prefills.

    Returns:
        num_decodes: The number of decode requests.
        num_prefills: The number of prefill requests.
        num_decode_tokens: The number of tokens in the decode requests.
        num_prefill_tokens: The number of tokens in the prefill requests.
    """
    long_seq_metadata = common_attn_metadata.prefill_context_parallel_metadata
    query_lens_pcp_full = long_seq_metadata.query_lens_pcp_full_cpu if long_seq_metadata else None
    max_query_len_pcp_full = long_seq_metadata.max_query_len_pcp_full if long_seq_metadata else 0
    max_query_len = common_attn_metadata.max_query_len if max_query_len_pcp_full == 0 else max_query_len_pcp_full
    num_reqs = common_attn_metadata.num_reqs
    if num_reqs == 0:
        return 0, 0, 0, 0

    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc_cpu

    # PD D + RecomputeScheduler: num_computed may be N-1 after KV recv while
    # this step is MTP decode (max_query_len <= threshold).
    if is_pd_decode_recompute_scheduler_enabled():
        treat_short_extends_as_decodes = True

    if (
        max_query_len <= decode_threshold
        and (not require_uniform or decode_threshold <= 1)
        and treat_short_extends_as_decodes
    ):
        return num_reqs, 0, num_tokens, 0

    query_lens_sharded = query_start_loc[1:] - query_start_loc[:-1]
    query_lens = query_lens_sharded if query_lens_pcp_full is None else query_lens_pcp_full
    if query_lens[0].item() > decode_threshold:
        return 0, num_reqs, 0, num_tokens

    if require_uniform:
        if torch.all((query_lens == query_lens[0]) | (query_lens == 0)):
            return num_reqs, 0, num_tokens, 0
        is_prefill = query_lens != query_lens[0]
    else:
        is_prefill = query_lens > decode_threshold

    if not treat_short_extends_as_decodes:
        assert common_attn_metadata.is_prefilling is not None
        raw_is_prefilling = common_attn_metadata.is_prefilling
        is_prefilling = raw_is_prefilling[: query_lens.shape[0]]
        if is_prefilling.shape[0] < query_lens.shape[0]:
            is_prefilling = F.pad(
                is_prefilling,
                (0, query_lens.shape[0] - is_prefilling.shape[0]),
                value=False,
            )
        is_prefill |= is_prefilling

    if not torch.any(is_prefill):
        return num_reqs, 0, num_tokens, 0

    first_prefill = is_prefill.int().argmax(dim=-1).item()
    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    num_decode_tokens = query_start_loc[first_prefill].item()
    num_prefill_tokens = num_tokens - num_decode_tokens
    return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)


def wait_for_kv_layer_from_connector(layer_name: str):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    # TODO: assert ascendMetadata
    connector.wait_for_layer_load(layer_name)


def maybe_save_kv_layer_to_connector(
    layer_name: str,
    kv_cache_layer: list[torch.Tensor],
):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    # TODO: assert ascendMetadata
    connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata)


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def trans_rope_weight(weight, rope_dim):
    if rope_dim == 0:
        return weight.contiguous()
    nope_part = weight[..., :-rope_dim, :]
    rope_part = weight[..., -rope_dim:, :]
    reordered_rope_part = torch.cat((rope_part[..., ::2, :], rope_part[..., 1::2, :]), dim=-2)
    return torch.cat((nope_part, reordered_rope_part), dim=-2).contiguous()


def transdata(nd_mat, block_size: tuple = (16, 16)):
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, (0, r_pad, 0, c_pad))
    nz_mat = torch.permute(
        torch.reshape(
            nd_mat,
            (r // block_size[0], block_size[0], c // block_size[1], block_size[1]),
        ),
        [2, 0, 1, 3],
    )
    nz_mat = torch.reshape(nz_mat, (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
    return nz_mat


def enabling_mlapo(vllm_config: VllmConfig) -> bool:
    config_val = get_ascend_config().enable_mlapo
    if get_ascend_device_type() == AscendDeviceType.A5:
        return bool(config_val)

    is_decode_instance = (
        vllm_config.kv_transfer_config is not None
        and vllm_config.kv_transfer_config.is_kv_consumer
        and not vllm_config.kv_transfer_config.is_kv_producer
    )
    return bool(config_val and is_decode_instance)


# ---------------------------------------------------------------------------
# Gemma4 MTP KV-sharing helpers
#
# Draft-model attention layers in Gemma4 MTP share K/V with the corresponding
# target-model layers (kv_sharing_target_layer_name).  On Ascend, the draft
# layers are Q-only: Gemma4MTPAttention.forward() passes a torch.empty dummy
# as K/V, so the draft cannot rely on its own (empty) cache.  These helpers
# gather the shared K/V from the target layer's paged cache and run a manual
# PyTorch SDPA cross-attention, because Ascend FIA (npu_fusion_attention)
# cannot handle cross-attention where actual_seq_qlen != actual_seq_kvlen.
#
# Kept in attention/utils.py (not attention_v1.py) per code-review
# requirement: attention_v1.py only calls maybe_kv_share_prefill /
# should_skip_draft_kv_write / notify_kv_cache_written.
# ---------------------------------------------------------------------------


def notify_kv_cache_written(layer_name: str = ""):
    """Notify the KV-transfer connector that KV cache has been written.

    No-op when there is no v1 KV-transfer group (the common case for
    Gemma4 MTP, which uses in-process KV-sharing via kv_sharing_target_layer_name
    rather than a distributed connector).  Restored from the revert of PR #11021
    (commit 44312516) — only the no-op stub is needed here, not the rest of
    the Layerwise KV Pooling machinery.
    """
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return
    connector = get_kv_transfer_group()
    on_kv_cache_written = getattr(connector, "on_kv_cache_written", None)
    if on_kv_cache_written is not None:
        on_kv_cache_written(layer_name)


def _forward_shared_kv_prefill_attention(
    impl,
    query: torch.Tensor,
    shared_key: torch.Tensor,
    shared_value: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
) -> torch.Tensor:
    """Manual PyTorch attention with already-dense shared KV from block_table.

    Ascend FIA (npu_fusion_attention) cannot handle cross-attention where
    actual_seq_qlen differs from actual_seq_kvlen — it either crashes with
    mask shape errors or produces zero output.  Use PyTorch's
    scaled_dot_product_attention instead, which correctly supports
    cross-attention with arbitrary Q/KV lengths and GQA (grouped-query
    attention).
    """
    num_tokens = attn_metadata.actual_seq_lengths_q[-1]
    q = query[:num_tokens]  # [T, H, D]
    k = shared_key  # [S, Hkv, D]
    v = shared_value  # [S, Hkv, D]

    # Build a block-diagonal causal mask that respects per-request
    # boundaries.  The flattened [num_tokens, S] batch concatenates
    # requests; a single global causal mask would let row i of request r
    # attend to KV columns belonging to request r' < r (cross-request
    # leak).
    #
    # Per-request Q lengths come from actual_seq_lengths_q (cumulative,
    # so diff to get per-request).  Per-request KV lengths come from
    # seq_lens_list.  Q and KV lengths differ for cross-attention (MTP
    # draft decode: Q=1 new token, KV=full sequence), so we must track
    # them independently.
    S = k.shape[0]
    cum_q = attn_metadata.actual_seq_lengths_q
    if cum_q and len(cum_q) > 1:
        q_lens = [cum_q[i] - cum_q[i - 1] for i in range(1, len(cum_q))]
    else:
        q_lens = [num_tokens]
    kv_lens = attn_metadata.seq_lens_list or [S]
    # Pair Q and KV lengths per request; if counts mismatch (padding),
    # zip stops at the shorter — both lists should have the same number
    # of real requests.
    mask = torch.full((num_tokens, S), float("-inf"), dtype=q.dtype, device=q.device)
    q_off = 0
    kv_off = 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        if q_len <= 0 or kv_len <= 0:
            q_off += q_len
            kv_off += kv_len
            continue
        # Query row j (0-indexed in this request's Q block) is the
        # (kv_len - q_len + j)-th token of the full sequence.  It attends
        # to KV columns [0, kv_len - q_len + j] (causal), restricted to
        # the sliding window if configured.
        offset = kv_len - q_len
        for j in range(q_len):
            causal_pos = offset + j  # position in the full sequence
            window_start = (
                max(0, causal_pos - impl.sliding_window + 1)
                if impl.sliding_window is not None and impl.sliding_window < kv_len
                else 0
            )
            mask[q_off + j, kv_off + window_start : kv_off + causal_pos + 1] = 0
        q_off += q_len
        kv_off += kv_len
    attn_mask = mask

    # Handle GQA: expand KV heads to match Q heads.
    # Ascend NPU's scaled_dot_product_attention does not broadcast
    # head dimension, so we must explicitly repeat KV heads.
    if q.shape[1] != k.shape[1]:
        n_rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(n_rep, dim=1)  # [S, Hkv, D] -> [S, Hq, D]
        v = v.repeat_interleave(n_rep, dim=1)  # [S, Hkv, D] -> [S, Hq, D]

    # Always use 4D format [B, H, L, D] for Ascend NPU.
    q_4d = q.unsqueeze(0).transpose(1, 2)  # [T, H, D] -> [1, H, T, D]
    k_4d = k.unsqueeze(0).transpose(1, 2)  # [S, H, D] -> [1, H, S, D]
    v_4d = v.unsqueeze(0).transpose(1, 2)  # [S, H, D] -> [1, H, S, D]
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [T, S] -> [1, 1, T, S]
    attn_output = F.scaled_dot_product_attention(
        q_4d,
        k_4d,
        v_4d,
        attn_mask=attn_mask,
        scale=impl.scale,
    )  # [1, H, T, D]
    attn_output = attn_output.squeeze(0).transpose(0, 1)  # [T, H, D]

    output[:num_tokens] = attn_output
    return output


def _get_current_token_shared_kv(
    impl,
    attn_metadata,
) -> tuple:
    """Gather current-token KV from the producer layer's shared cache."""
    if impl.key_cache is None or impl.value_cache is None:
        return None, None
    num_tokens = attn_metadata.actual_seq_lengths_q[-1]
    if attn_metadata.slot_mapping is None or attn_metadata.slot_mapping.numel() < num_tokens:
        return None, None
    slots = attn_metadata.slot_mapping[:num_tokens].long()
    key = impl.key_cache.reshape(-1, impl.num_kv_heads, impl.head_size).index_select(0, slots)
    value = impl.value_cache.reshape(-1, impl.num_kv_heads, impl.head_size).index_select(0, slots)
    return key, value


def _get_shared_kv_from_block_table(
    impl,
    attn_metadata,
) -> tuple:
    """Gather K/V from the shared target cache using block tables.

    Used when slot_mapping is not available (e.g., during speculative
    decoding where the draft model inherits attn_metadata from the
    target but slot_mapping may not be populated for draft layers).

    For KV-sharing draft layers, impl.key_cache points to the draft
    model's own (empty) cache.  We must swap to the target layer's
    cache via _kv_share_target_impl, mirroring the PA path fix.
    """
    _tgt_impl = getattr(impl, "_kv_share_target_impl", None)
    if _tgt_impl is not None and _tgt_impl.key_cache is not None:
        read_kc = _tgt_impl.key_cache
        read_vc = _tgt_impl.value_cache
    else:
        read_kc = impl.key_cache
        read_vc = impl.value_cache

    if read_kc is None or read_vc is None:
        return None, None

    # Per-group block-table routing: draft layers share KV with target
    # layers that may be in DIFFERENT KV cache groups.  attn_metadata.block_tables
    # is the common (gid=0) table; using it for layers whose target is in gid≠0
    # reads from the wrong pool.  Route each layer to its per-group block_table
    # via _kv_share_gid (set by _store_gids_on_impls) + _per_group_bt_ref
    # (the {gid: block_table} dict set by set_per_group_block_table).
    # Per-group block-table routing: draft layers share KV with target
    # layers that may be in DIFFERENT KV cache groups.  attn_metadata.block_tables
    # is the common (gid=0) table; using it for layers whose target is in gid!=0
    # reads from the wrong pool.  Route each layer to its per-group block_table
    # via _kv_share_gid (set by _store_gids_on_impls) + _per_group_bt_ref
    # (the {gid: block_table} dict set by set_per_group_block_table).
    _my_gid = getattr(impl, "_kv_share_gid", None)
    _per_group_bt = getattr(impl, "_per_group_bt_ref", None)
    _routed_bt = None
    if _my_gid is not None and _per_group_bt is not None and _my_gid in _per_group_bt:
        _routed_bt = _per_group_bt[_my_gid]
    block_table = _routed_bt if _routed_bt is not None else attn_metadata.block_tables
    seq_lens = attn_metadata.seq_lens_list
    if block_table is None or not seq_lens:
        return None, None

    # Explicit gid -> block_table routing above is the intended path.  We
    # deliberately do NOT probe other groups' block tables by reading KV
    # tensor means: that would force an NPU->CPU sync (.item()) on the
    # attention hot path, use a data-dependent zero/non-zero heuristic,
    # and a broad `except Exception` that hides real routing/shape bugs.
    # If the routed block table is wrong, fail loudly so the routing is
    # fixed at the source rather than masked here.
    dense_key, dense_value = _gather_paged_kv_to_dense(
        read_kc,
        read_vc,
        block_table,
        seq_lens,
        impl.num_kv_heads,
        impl.head_size,
    )
    return dense_key, dense_value


def maybe_kv_share_prefill(
    impl,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache,
    attn_metadata,
    output: torch.Tensor,
):
    """Intercept entry point for KV-sharing draft layers.

    Called at the top of AscendAttentionBackendImpl.forward_impl.  Returns
    the attention output tensor if this layer is a KV-sharing draft layer
    whose prefill should be computed from the shared target cache via
    PyTorch SDPA; returns None to let the caller fall through to the
    normal FIA / PA path.

    Also initialises impl.key_cache / impl.value_cache from the kv_cache
    tuple for KV-sharing layers (they do not own a private cache).
    """
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    # Ensure self.key_cache / self.value_cache are initialised from the
    # kv_cache tuple BEFORE the shared-KV lookup, otherwise they will be
    # None (draft layers do not own a private cache).
    if (
        impl.kv_sharing_target_layer_name is not None
        and impl.key_cache is None
        and kv_cache is not None
        and len(kv_cache) >= 2
    ):
        impl.key_cache, impl.value_cache = kv_cache[0], kv_cache[1]

    # Dual-graph (VLLM_ASCEND_GEMMA4_DRAFT_GRAPH=1): under graph capture or
    # FULL replay, draft KV-sharing layers must NOT take the SDPA path here.
    # SDPA gathers paged KV into a dense tensor using Python-side
    # seq_lens.tolist() (frozen at capture time) and builds the mask in Python,
    # which is incompatible with graph replay (upstream issue #48503 class).
    # Return None to let forward_impl route to PA (head_dim=512) / FIA
    # (head_dim=256), which consume tensorised block_table/seq_lens and can be
    # updated via graph_task_update. Eager mode keeps the SDPA path.
    _capturing = getattr(_EXTRA_CTX, "capturing", False)
    _is_draft = getattr(_EXTRA_CTX, "is_draft_model", False)
    if _is_draft and impl.kv_sharing_target_layer_name is not None and _capturing:
        return None

    _kv_prefill_eligible = (
        impl.kv_sharing_target_layer_name is not None
        and key is not None
        and value is not None
        and query.shape[0] == key.shape[0]
        and attn_metadata.attn_state
        in (
            AscendAttentionState.PrefillNoCache,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.SpecDecoding,
        )
    )
    if not _kv_prefill_eligible:
        return None

    # For SpecDecoding / draft-model layers, slot_mapping points to
    # empty/wrong positions (draft layers do not write KV).  Skip the
    # slot-based lookup and go straight to the block-table gather.
    if attn_metadata.attn_state != AscendAttentionState.SpecDecoding and not getattr(
        _EXTRA_CTX, "is_draft_model", False
    ):
        shared_key, shared_value = _get_current_token_shared_kv(impl, attn_metadata)
    else:
        shared_key, shared_value = None, None

    if shared_key is None or shared_value is None:
        shared_key, shared_value = _get_shared_kv_from_block_table(impl, attn_metadata)

    if shared_key is None or shared_value is None:
        return None

    return _forward_shared_kv_prefill_attention(
        impl,
        query,
        shared_key,
        shared_value,
        attn_metadata,
        output,
    )


def should_skip_draft_kv_write(impl) -> bool:
    """True for Q-only draft KV-sharing layers.

    Gemma4MTPAttention.forward() creates a dummy K/V via torch.empty() and
    passes it as key/value to self.attn().  Writing this uninitialized
    memory back via reshape_and_cache would corrupt the shared target KV
    cache, causing progressive degradation across sequential loop steps.
    Skip the write for draft KV-shared layers — they only READ.
    """
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX

    return getattr(impl, "kv_sharing_target_layer_name", None) is not None and getattr(
        _EXTRA_CTX, "is_draft_model", False
    )


def maybe_route_512_capture(impl, attn_metadata) -> bool:
    """True if a head_dim=512 non-sliding layer should route to PagedAttention
    during graph capture.

    FIA TND does not support head_dim=512 (Gemma4 global attention).  During
    graph capture the eager device-adaptor fallback
    (npu_large_head_prefill_attention) is bypassed, and forward_impl's PA
    routing requires attn_state==DecodeOnly which excludes MTP's SpecDecoding
    capture step.  Route 512-dim non-sliding layers to the paged-attention
    graph path here, mirroring using_paged_attention(head_size=512) in
    forward_impl.  KV-sharing draft 512 layers are intercepted earlier by
    maybe_kv_share_prefill and never reach here.

    A5 gate: on A5 (950) full_graph_pa segfaults in atb::_npu_paged_attention
    during capture for this layer; A5's original path (full_graph_fia) works.
    This fix targets A2/A3 (910B4) where FIA TND raises error 561002.
    """
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.utils import is_950

    return (
        getattr(impl, "head_size", None) == FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE
        and getattr(impl, "sliding_window", None) is None
        and not getattr(_EXTRA_CTX, "is_draft_model", False)
        and not is_950()
    )


def maybe_skip_reshape_for_kv_share(impl, attn_metadata) -> bool:
    """True if reshape_and_cache should be skipped for a KV-sharing target layer.

    KV-sharing target layers (e.g. Gemma4 MTP draft) consume the producer
    layer's cache.  Re-caching here would overwrite the shared KV slots
    before attention reads it.  When True the caller must still record the
    producer's reshape_cache_event (if this layer is a producer) and return
    early.
    """
    return getattr(impl, "kv_sharing_target_layer_name", None) is not None
