# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 DSA metadata and fused attention execution.

The model file owns the network topology and projection modules.  This module
owns the attention execution boundary: it gathers every cache plane's metadata
before running the compressor, indexer and sparse-attention operators without
moving cache or scheduler knowledge back into the model.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    max_decode_query_len,
)
from vllm.v1.kv_cache_interface import AttentionSpec, CircularBufferSpec

from vllm_ascend.ascend_config import PcpConfig, get_ascend_config
from vllm_ascend.attention.context_parallel.pcp_owner_timing import enter as _timing_enter
from vllm_ascend.attention.context_parallel.pcp_owner_timing import exit as _timing_exit
from vllm_ascend.attention.dsa_v1 import AscendDSABackend, build_dspark_swa_indices, dsv4_dsa_overlap_stream
from vllm_ascend.attention.utils import (
    KVCacheRuntimeShapes,
    maybe_save_kv_layer_to_connector,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    get_kv_cache_compression_ratio,
    get_storage_block_size,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.models.deepseek_v41.cache_config import STATE_RING_ROWS
from vllm_ascend.ops.rope_dsv4 import (
    get_cos_and_sin_dsa,
    get_full_cos_and_sin_dsa_for_layer,
)
from vllm_ascend.ops.triton.c2_ring_metadata import build_c2_ring_metadata
from vllm_ascend.utils import (
    get_potential_max_tokens,
    npu_stream_switch,
    oproj_tp_enable,
)
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
    wait_for_device_metadata,
)

V41_METADATA_BUFFER_SIZE = 1024


@eager_break_during_capture
def dsa_v41_forward(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Execute V4.1 attention behind an explicit graph side-effect boundary."""
    forward_context = get_forward_context()
    attn = forward_context.no_compile_layers[layer_name]
    attn.v41_impl.forward(attn, None, hidden_states, output)


def dsa_v41_forward_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="dsa_v41_forward",
    op_func=dsa_v41_forward,
    mutates_args=["output"],
    fake_impl=dsa_v41_forward_fake,
    dispatch_key="PrivateUse1",
)


@eager_break_during_capture
def dsa_v41_write_kv(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Write a V4.1 layer's KV caches without running attention or its MLP."""
    forward_context = get_forward_context()
    attn = forward_context.no_compile_layers[layer_name]
    metadata = attn.v41_impl._get_layer_metadata(forward_context.attn_metadata)
    wait_for_kv_layer_from_connector(layer_name)
    hidden_states = hidden_states[: metadata.swa.num_actual_tokens]
    attn.v41_impl._update_caches(attn, hidden_states, metadata)
    # The following cut-layer forward writes retained rows again. Its normal
    # save hook records the final-write event without advancing twice here.


def dsa_v41_write_kv_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="dsa_v41_write_kv",
    op_func=dsa_v41_write_kv,
    mutates_args=["output"],
    fake_impl=dsa_v41_write_kv_fake,
    dispatch_key="PrivateUse1",
)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read one field from either an HF config object or a raw config dict."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def build_replay_swa_visible_lens(
    window_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    replay_start: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """How many ori KV entries each query token may actually see.

    Upstream's SWA kernels clamp the window per request (``start_pos =
    max(start_pos, replay_start)``), and the clamp is a correctness requirement
    rather than a refinement: a replayed request recomputes the hit's last
    window, so the sliding-window group holds no KV of its own below
    ``replay_start``. Those blocks were retired and replaced by the null block,
    whose KV upstream documents as all-zero -- the window would not read another
    request's data, it would read phantom keys that score zero and take weight
    in the replayed token's softmax.

    The band-mode operator bounds a window with one scalar for the whole call
    (``ori_win_left``), which cannot express a per-token floor, so the floor
    travels through ``ori_topk_length`` in band mode: "the number of ori KV
    entries this query token may see". Entries are counted back from the token's
    own right edge, so a count of ``position - floor + 1`` is exactly "keep the
    window ending here, but not past the floor". A5 uses an explicit list
    and builds its indices with the same count.

    Positions are derived the way the operator derives its own, from
    ``seqused_ori_kv - actS1Size + s1``, so the two coordinate systems cannot
    drift. Device-side only: no ``.item()``, no host round trip. The caller
    guarantees ``num_tokens`` is the sum of the query lengths, as
    :func:`build_dspark_swa_indices` does for the other user of this channel.
    """
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    req_ids = torch.repeat_interleave(
        torch.arange(query_lens.shape[0], device=query_start_loc.device, dtype=torch.long),
        query_lens,
        output_size=num_tokens,
    )
    token_offsets = torch.arange(num_tokens, device=query_start_loc.device) - query_start_loc[req_ids]
    positions = seq_lens[req_ids] - query_lens[req_ids] + token_offsets
    lower = replay_start[: query_lens.shape[0]].to(positions.dtype)[req_ids]
    start_positions = torch.maximum(positions - int(window_size) + 1, lower).clamp_min(0)
    return (positions - start_positions + 1).to(torch.int32).unsqueeze(1)


@dataclass
class AscendDSAV41Metadata(AttentionMetadata):
    """Scheduler and cache-plane contract for one V4.1 cache resource.

    ``seq_lens``/``query_start_loc`` always stay in original-token
    coordinates, matching the common vLLM metadata. The ``cache_*`` fields
    describe the rows visible to the concrete cache plane. Keeping both
    coordinate systems here lets future fused kernels replace the eager path
    without rebuilding scheduling metadata in the model.
    """

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    compress_ratio: int
    storage_block_size: int
    is_compressor_state: bool
    flat_slot_mapping: torch.Tensor | None = None
    cache_kind: str = "unknown"
    # MRV2 producer ownership follows this forward's metadata, including capture.
    device_metadata_executor: Any = None
    positions: torch.Tensor | None = None
    cos: Any = None
    sin: Any = None
    num_actual_tokens: int = 0
    num_input_tokens: int = 0
    num_reqs: int = 0
    num_actual_reqs: int = 0
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    logical_block_size: int = 0
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    cache_seq_lens: torch.Tensor | None = None
    max_query_len: int = 0
    max_seq_len: int = 0
    max_cache_seq_len: int = 0
    attn_state: Any = None
    is_prefilling: torch.Tensor | None = None
    causal: bool | torch.Tensor = True
    ori_sparse_indices: torch.Tensor | None = None
    ori_topk_length: torch.Tensor | None = None
    ori_mask_mode: int = 4
    ori_win_left: int = 0
    ori_win_right: int = 0
    replay_seq_lens: torch.Tensor | None = None
    replay_block_table: torch.Tensor | None = None
    replay_cache_blocks: torch.Tensor | None = None
    replay_cache_offsets: torch.Tensor | None = None
    smla_metadata: torch.Tensor | None = None
    a5_smla_plans: dict[int, torch.Tensor] | None = None
    qli_metadata: torch.Tensor | None = None
    cmp_residual: torch.Tensor | None = None
    c2_ring_metadata: torch.Tensor | None = None
    c2_complete_mask: torch.Tensor | None = None
    c2_source_positions: torch.Tensor | None = None
    c2_source_cos: torch.Tensor | None = None
    c2_source_sin: torch.Tensor | None = None
    c2_metadata_group_id: int | None = None
    # Per-forward staging cache, shared through the builder's batch dictionary
    # so every cache group in the same forward resolves the same instance.
    swa_stage_cache: Any = None
    global_metadata: "AscendDSAV41Metadata | None" = None


@dataclass(frozen=True)
class DeepseekV41CompressorMetadata:
    """V4-shaped cache/state bundle consumed by the compressor stage."""

    cache: AscendDSAV41Metadata
    state: AscendDSAV41Metadata | None = None


@dataclass(frozen=True)
class DeepseekV41IndexerMetadata:
    """V4-shaped source cache bundle consumed by the indexer stage."""

    cache: AscendDSAV41Metadata


@dataclass(frozen=True)
class DeepseekV41LayerMetadata:
    """All metadata consumed by one V4.1 attention layer invocation."""

    attention: AscendDSAV41Metadata | None
    swa: AscendDSAV41Metadata
    compressor: DeepseekV41CompressorMetadata | None
    indexer: DeepseekV41IndexerMetadata | None
    # Per-forward shared cache for the bounded SWA staging view. Every layer
    # resolves it from the same attention-metadata mapping, so planning and
    # operator metadata are shared across the cache group instead of per layer.
    swa_stage_cache: "_SwaChunkStageCache | None" = None

    @property
    def positions(self) -> torch.Tensor:
        return self.swa.positions

    def rope(self, layer_name: str, num_tokens: int):
        return self.swa.cos[layer_name][:num_tokens], self.swa.sin[layer_name][:num_tokens]


@dataclass
class DeepseekV41PreparedIndexer:
    """Query-local projections and the event guarding auxiliary quantization."""

    query: torch.Tensor | None
    weights: torch.Tensor
    quantized_query: torch.Tensor | None = None
    query_scale: torch.Tensor | None = None
    quantize_done: Any = None


def compressed_slot_mapping(slot_mapping: torch.Tensor, ratio: int) -> torch.Tensor:
    """Convert original-token physical slots to completed compressed slots.

    Logical block sizes must be divisible by ratio. Negative/padded slots and
    incomplete compression groups never produce a write.
    """
    valid = (slot_mapping >= 0) & ((slot_mapping + 1) % ratio == 0)
    return torch.where(valid, slot_mapping // ratio, -1)


def _request_counts(common: Any, num_reqs: int):
    """Return V4-shaped request counters without synchronizing the NPU."""
    is_prefilling = getattr(common, "is_prefilling", None)
    query_start_loc_cpu = getattr(common, "query_start_loc_cpu", None)
    if (
        is_prefilling is None
        or query_start_loc_cpu is None
        or getattr(is_prefilling, "device", None) is None
        or is_prefilling.device.type != "cpu"
    ):
        return 0, 0, 0, 0
    flags = is_prefilling[:num_reqs].bool()
    num_flags = min(flags.numel(), query_start_loc_cpu.numel() - 1)
    flags = flags[:num_flags]
    query_lens_cpu = query_start_loc_cpu[1 : num_flags + 1] - query_start_loc_cpu[:num_flags]
    num_prefills = int(flags.sum().item())
    num_decodes = num_flags - num_prefills
    num_prefill_tokens = int(query_lens_cpu[flags].sum().item())
    num_decode_tokens = int(query_lens_cpu[~flags].sum().item())
    return num_decodes, num_decode_tokens, num_prefills, num_prefill_tokens


def scatter_cache_sk(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Store rows using builder-prepared coordinates and V4's Ascend op.

    V4.1 cache planes can be views into a larger layer-outermost slot, so the
    physical page stride is not necessarily the contiguous stride implied by
    the plane shape. ``npu_scatter_nd_update_sk`` preserves that stride and
    treats the builder's ``[-1, -1]`` coordinates as skipped rows, matching V4.
    """
    cache = cache.squeeze(-2)
    indices = slot_mapping[: values.shape[0]]
    updates = values.to(cache.dtype).contiguous()
    # A5 PP mirrors move packed bytes verbatim. The scatter operator accepts
    # signed 8-bit elements, so use same-size views without converting data.
    if cache.dtype == torch.uint8:
        cache = cache.view(torch.int8)
        updates = updates.view(torch.int8)
    torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, indices, updates)


def pad_sparse_indices(indices: torch.Tensor, topk: int) -> torch.Tensor:
    """Convert V4.1's compact [T, K] selection into SMLA [T, 1, topk]."""
    if indices.shape[-1] < topk:
        indices = F.pad(indices, (0, topk - indices.shape[-1]), value=-1)
    return indices.unsqueeze(1).contiguous().int()


class _SwaChunkPlan:
    """Immutable per-step view of one cache group's bounded SWA blocks.

    ``window_block_table`` lists, per local chunk row, exactly the slab blocks
    that the sliding window can reach. Staging turns them into a dense local
    range; ``replay_seq_lens`` rebases the operator's causal coordinates onto
    that range. ``segments`` is the shared, batch-level segment plan that both
    the rebased table and the operator metadata signature derive from.
    """

    __slots__ = (
        "window_block_table",
        "counts",
        "replay_seq_lens",
        "total_blocks",
        "max_replay_seq_len",
        "prepared",
        "segments",
    )

    def __init__(
        self, window_block_table, counts, replay_seq_lens, total_blocks, max_replay_seq_len, prepared, segments
    ):
        self.window_block_table = window_block_table
        self.counts = counts
        self.replay_seq_lens = replay_seq_lens
        self.total_blocks = total_blocks
        self.max_replay_seq_len = max_replay_seq_len
        # Dense stage table and flat source-block list, shared by every layer
        # in this group; only the copied data differs per layer.
        self.prepared = prepared
        self.segments = segments


class _SwaChunkStageCache:
    """Per-forward staging caches shared by every layer of one batch.

    ``segments`` depends only on the rank-local CPU sequence lengths, so the
    segment plan is computed once per batch regardless of how many layers read
    it. ``smla_cache`` is keyed by a value signature of the operator inputs:
    layers with identical batch geometry share one ``npu_sparse_flash_mla_metadata``
    call, while every differing geometry still pays its own call.
    """

    __slots__ = ("segments", "segments_key", "smla_cache")

    def __init__(self):
        self.segments_key: tuple | None = None
        self.segments: list | None = None
        self.smla_cache: dict[tuple, torch.Tensor | bool] = {}


_SWA_CHUNK_STAGE_CACHE_KEY = "_swa_chunk_stage_cache"


def _cpu_prefix(value, size: int) -> tuple[int, ...] | None:
    """Read a CPU-side length prefix without a device synchronization."""
    if value is None:
        return None
    try:
        values = value.tolist() if hasattr(value, "tolist") else value
        return tuple(int(item) for item in list(values)[:size])
    except (TypeError, ValueError):
        return None


def _swa_chunk_stage_cache(metadata) -> _SwaChunkStageCache:
    """Return the batch-scoped staging cache attached to ``metadata``."""
    cache = getattr(metadata, _SWA_CHUNK_STAGE_CACHE_KEY, None)
    if not isinstance(cache, _SwaChunkStageCache):
        cache = _SwaChunkStageCache()
        try:
            setattr(metadata, _SWA_CHUNK_STAGE_CACHE_KEY, cache)
        except (AttributeError, TypeError):
            # A slotted metadata object cannot carry the shared cache; this
            # step then plans per call rather than failing.
            return cache
    return cache


def _build_swa_chunk_plan(metadata, block_table, num_reqs, block_size, window, num_blocks, *, stage_cache=None):
    """Build the bounded SWA view for one cache group.

    The segment plan runs on the CPU-side sequence lengths only: no device
    synchronization and no full-prefix gather. It is cached on the shared
    per-forward ``stage_cache`` under its actual CPU inputs, so every layer
    reuses one plan while each cache group still gathers its own block table.
    Returns ``None`` for any batch the bounded view cannot represent exactly,
    in which case the caller keeps the direct peer-read path.
    """
    from vllm_ascend.attention.context_parallel.pcp_owner_stage import (
        build_stage_tables_fixed,
        plan_swa_chunk_segments,
    )

    if stage_cache is None:
        stage_cache = _swa_chunk_stage_cache(metadata)
    starts = _cpu_prefix(getattr(metadata, "query_start_loc_cpu", None), num_reqs + 1)
    seqs = _cpu_prefix(getattr(metadata, "seq_lens_cpu", None), num_reqs)
    key = (starts, seqs, int(num_reqs), int(block_size), int(window))
    if stage_cache.segments_key == key:
        segments = stage_cache.segments
    else:
        segments = plan_swa_chunk_segments(starts, seqs, num_reqs, block_size, window)
        stage_cache.segments_key = key
        stage_cache.segments = segments
    plan = None
    if segments is not None and block_table.shape[1] > 0:
        max_blocks = max((segment.num_blocks for segment in segments), default=0)
        if max_blocks > 0 and max_blocks <= int(block_table.shape[1]):
            device = block_table.device
            indices = torch.full((num_reqs, max_blocks), -1, dtype=torch.int64, device=device)
            for row, segment in enumerate(segments):
                indices[row, : segment.num_blocks] = torch.arange(
                    segment.first_block,
                    segment.first_block + segment.num_blocks,
                    dtype=torch.int64,
                    device=device,
                )
            valid = indices >= 0
            gathered = block_table[:num_reqs].gather(1, indices.clamp_min(0))
            window_block_table = torch.where(valid, gathered, torch.full_like(gathered, -1))
            counts = [segment.num_blocks for segment in segments]
            replay_seq_lens = torch.tensor(
                [segment.replay_seq_len for segment in segments],
                dtype=metadata.seq_lens.dtype,
                device=metadata.seq_lens.device,
            )
            prepared = build_stage_tables_fixed(window_block_table, counts, num_blocks=num_blocks)
            plan = _SwaChunkPlan(
                window_block_table=window_block_table,
                counts=counts,
                replay_seq_lens=replay_seq_lens,
                total_blocks=sum(counts),
                max_replay_seq_len=max(segment.replay_seq_len for segment in segments),
                prepared=prepared,
                segments=segments,
            )
    return plan


def _smla_signature_value(value):
    """Normalize one operator input into a hashable signature component."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        # Length tensors affect the tiling, so they cannot be keyed by value
        # without a device sync. The same tensor object across layers shares
        # the identical contents; a different allocation rebuilds.
        return ("tensor", int(value.data_ptr()), tuple(value.shape))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _swa_chunk_smla_signature(
    attn, swa_metadata, operator_metadata, plan, *, has_compressed, ratio, cmp_topk, num_reqs
) -> tuple:
    """Value signature of every input that changes the operator tiling."""
    cmp_mask_mode = 3 if has_compressed else 0
    return (
        int(attn.n_local_heads),
        int(attn.head_dim),
        int(num_reqs),
        int(swa_metadata.max_query_len),
        int(plan.max_replay_seq_len),
        int(operator_metadata.max_cache_seq_len) if has_compressed else 0,
        _smla_signature_value(swa_metadata.ori_topk_length),
        int(cmp_topk) if has_compressed else 0,
        int(ratio) if has_compressed else 0,
        int(swa_metadata.ori_mask_mode),
        cmp_mask_mode,
        int(swa_metadata.ori_win_left),
        int(swa_metadata.ori_win_right),
        "TND",
        "PA_BBND",
        True,
        bool(has_compressed),
        tuple((int(segment.replay_start), int(segment.replay_seq_len)) for segment in plan.segments),
    )


def _build_swa_chunk_smla_metadata(
    attn,
    swa_metadata,
    operator_metadata,
    plan,
    *,
    stage_cache=None,
    has_compressed,
    ratio,
    cmp_seq_lens,
    cmp_residual,
    cmp_topk,
    num_reqs,
):
    """Rebuild the operator tiling metadata for the rebased bounded view.

    The builder produced it from the unbounded sequence lengths; the kernel
    must see the same batch with ``seqused_ori_kv`` shortened to each segment's
    replay length. Memoized on the shared per-forward cache by a value
    signature of the operator inputs, so identical batch geometry across layers
    pays one metadata op per forward while differing geometry still rebuilds.
    """
    if stage_cache is None:
        stage_cache = _swa_chunk_stage_cache(swa_metadata)
    key = _swa_chunk_smla_signature(
        attn,
        swa_metadata,
        operator_metadata,
        plan,
        has_compressed=has_compressed,
        ratio=ratio,
        cmp_topk=cmp_topk,
        num_reqs=num_reqs,
    )
    if key in stage_cache.smla_cache:
        cached = stage_cache.smla_cache[key]
        if cached is False:
            return None
        # Every layer gets its own tensor even though the op ran once; a
        # kernel or caller that mutates the tiling cannot affect its peers.
        return cached.clone() if isinstance(cached, torch.Tensor) else cached
    try:
        value = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
            attn.n_local_heads,
            1,
            attn.head_dim,
            cu_seqlens_q=swa_metadata.query_start_loc[: num_reqs + 1].int(),
            seqused_ori_kv=plan.replay_seq_lens,
            seqused_cmp_kv=cmp_seq_lens if has_compressed else None,
            cmp_residual_kv=cmp_residual if has_compressed else None,
            batch_size=num_reqs,
            max_seqlen_q=int(swa_metadata.max_query_len),
            max_seqlen_ori_kv=int(plan.max_replay_seq_len),
            max_seqlen_cmp_kv=(int(operator_metadata.max_cache_seq_len) if has_compressed else 0),
            ori_topk=0,
            ori_topk_length=swa_metadata.ori_topk_length,
            cmp_topk=cmp_topk if has_compressed else 0,
            cmp_ratio=ratio if has_compressed else 0,
            ori_mask_mode=swa_metadata.ori_mask_mode,
            cmp_mask_mode=3 if has_compressed else 0,
            ori_win_left=swa_metadata.ori_win_left,
            ori_win_right=swa_metadata.ori_win_right,
            layout_q="TND",
            layout_kv="PA_BBND",
            has_ori_kv=True,
            has_cmp_kv=has_compressed,
        )
    except Exception:  # noqa: BLE001 - fail open to the unbounded path
        from vllm.logger import logger

        logger.warning_once(
            "PCP SWA chunk staging: rebuilding sparse-MLA metadata failed; "
            "keeping the direct peer-read path for this step.",
            exc_info=True,
        )
        stage_cache.smla_cache[key] = False
        return None
    # The operator may return a tensor backed by an internal reusable buffer;
    # cache an owned copy and hand out a separate copy per caller.
    stored = value.clone() if isinstance(value, torch.Tensor) else value
    stage_cache.smla_cache[key] = stored
    return stored.clone() if isinstance(stored, torch.Tensor) else stored


def _maybe_stage_owner_cmp_cache(
    source_cache, block_table, metadata, num_reqs, pcp_config, seq_lens=None, block_size=None
):
    """Copy a pure-prefill batch's long-KV blocks into local scratch.

    Only applies to owner-sharded caches: every PCP rank writes one quarter of
    the slab locally and maps the rest over peer memory, where one-byte-grain
    reads run at a third of local bandwidth.  Copying the referenced blocks
    once per layer turns that into a bulk gather and leaves the attention
    kernel on local HBM.  Decode steps and non-prefill batches keep the
    original table; overflow falls back to direct peer reads.
    """
    _pcp_stage_count("long_kv", "attempts")
    if source_cache is None or block_table is None:
        _pcp_stage_count("long_kv", "skipped_config")
        return None
    # Pure-prefill steps only.  Decode rows address the whole history, so a
    # per-step bulk copy would cost more than the peer reads it replaces; a
    # mixed step keeps direct peer reads for every row.
    if getattr(metadata, "num_prefills", 0) <= 0:
        _pcp_stage_count("long_kv", "skipped_nonprefill")
        return None
    if getattr(metadata, "num_decodes", 0) > 0:
        _pcp_stage_count("long_kv", "skipped_mixed")
        return None
    from vllm_ascend.attention.context_parallel.pcp_owner_stage import get_stage_cache

    if not pcp_config.owner_stage:
        _pcp_stage_count("long_kv", "skipped_config")
        return None
    from vllm_ascend.attention.context_parallel.pcp_owner_vmm import owner_slab_for_tensor

    if owner_slab_for_tensor(source_cache) is None:
        _pcp_stage_count("long_kv", "skipped_nonowner")
        return None
    # Consumers of one long-KV source share the same cache and block table:
    # the source layer owns both, and every later layer in the group reads
    # them unchanged.  Memoizing on the (per-group, per-step) metadata object
    # turns 38 copies per forward into 4 without any ordering hazard, because
    # the source layer's own write happens before its first stage call.
    memo = getattr(metadata, "_pcp_stage_memo", None)
    if memo is not None and memo[0] == source_cache.data_ptr():
        _pcp_stage_count("long_kv", "memo_hit")
        return memo[1]
    cache = get_stage_cache(source_cache.device, source_cache, budget_bytes=pcp_config.owner_stage_bytes)
    result = cache.stage(
        source_cache,
        block_table,
        seq_lens=seq_lens,
        block_size=block_size,
        enable_timing=pcp_config.enable_timing,
    )
    if result is None:
        _pcp_stage_count("long_kv", "skipped_overflow")
        return None
    metadata._pcp_stage_memo = (source_cache.data_ptr(), result)
    _pcp_stage_count("long_kv", "ok")
    _PCP_STAGE_STEP["long_kv_max_blocks"] = max(
        _PCP_STAGE_STEP["long_kv_max_blocks"], int(cache.last_total_blocks)
    )
    _PCP_STAGE_STEP["long_kv_capacity"] = int(cache.capacity_blocks)
    return result


# --- owner staging attribution -------------------------------------------
# Gated fallbacks (mixed/decode batches, non-owner caches, plan or metadata
# failures, overflow) silently keep direct peer reads.  Without counters a
# high-TTFT step looks like a staging win, so count every decision and log one
# summary per fixed number of forward steps.  The batch-scoped stage cache is
# rebuilt per step, which gives the step boundary without threading a step id
# through the layer forwards.
_PCP_STAGE_STATS: dict[str, dict[str, int]] = {}
_PCP_STAGE_STEP: dict[str, Any] = {
    # Keyed on the step's attention-metadata mapping, not on the stage cache:
    # the builder keeps one batch dictionary per worker, so the stage cache
    # (and its signature caches) survives across steps by design.
    "token": None,
    "stage_cache": None,
    # Step-scoped negative memo: the stages that can fail (bounded plan,
    # metadata rebuild, overflow) depend on batch geometry only, so a failed
    # step must not re-run their table/metadata work in every layer.
    "negative": None,
    "steps": 0,
    "smla_ops": 0,
    "step_smla_ops": 0,
    "swa_max_blocks": 0,
    "swa_capacity": 0,
    "long_kv_max_blocks": 0,
    "long_kv_capacity": 0,
}
_PCP_STAGE_REPORT_EVERY = 128


def _pcp_stage_count(kind: str, name: str, amount: int = 1) -> None:
    stats = _PCP_STAGE_STATS.setdefault(kind, {})
    stats[name] = stats.get(name, 0) + amount


def _pcp_stage_mark_negative(stage_cache, reason: str) -> None:
    if stage_cache is not None:
        _PCP_STAGE_STEP["negative"] = (stage_cache, reason)


def _pcp_stage_negative(stage_cache) -> str | None:
    negative = _PCP_STAGE_STEP["negative"]
    if stage_cache is not None and negative is not None and negative[0] is stage_cache:
        return negative[1]
    return None


def _pcp_stage_note_step(step_token, stage_cache) -> None:
    """Report the previous forward step once the step metadata object changes."""
    state = _PCP_STAGE_STEP
    state["stage_cache"] = stage_cache
    if step_token is None:
        return
    token = state["token"]
    if token is step_token:
        return
    if token is not None:
        cache = state["stage_cache"]
        entries = len(getattr(cache, "smla_cache", {}) or {})
        state["steps"] += 1
        if state["steps"] % _PCP_STAGE_REPORT_EVERY == 0:
            from vllm.logger import logger

            logger.info(
                "PCP stage counters: steps=%d swa=%s long_kv=%s smla_ops=%d step_smla_ops=%d "
                "step_smla_entries=%d swa_blocks=%d/%d long_kv_blocks=%d/%d",
                state["steps"],
                dict(sorted(_PCP_STAGE_STATS.get("swa", {}).items())),
                dict(sorted(_PCP_STAGE_STATS.get("long_kv", {}).items())),
                state["smla_ops"],
                state["step_smla_ops"],
                entries,
                state["swa_max_blocks"],
                state["swa_capacity"],
                state["long_kv_max_blocks"],
                state["long_kv_capacity"],
            )
        # The operator tiling cache survives in the builder's batch dictionary
        # across steps, while its signature cannot see the per-request
        # compressed lengths (only max_cache_seq_len).  Drop it every step so
        # a step with the same max but a different distribution cannot reuse
        # stale metadata; the CPU-keyed segment plan is safe to keep.
        smla_cache = getattr(cache, "smla_cache", None)
        if isinstance(smla_cache, dict):
            smla_cache.clear()
        state["step_smla_ops"] = 0
        state["negative"] = None
    state["token"] = step_token


class AscendDSAV41Impl:
    """V4.1 execution boundary backed by fused Ascend operators.

    Projection, compressor and indexer modules remain registered by the model,
    while this object resolves the complete per-layer metadata bundle and owns
    their invocation order.  That is the same separation used by ``dsa_v1``:
    model construction is independent from cache-aware attention execution.
    """

    def _start_owner_swa_stage_async(self, attn, metadata):
        """Launch bounded SWA staging on a side stream right after the cache writes.

        The table build, metadata rebuild and copy are all device work, so they
        overlap ``_select_sparse_indices`` on the main stream.  The result is
        stashed on this layer's SWA metadata and consumed by
        ``_forward_attention``; any non-prefill batch, non-owner cache or
        overflow keeps the synchronous path.
        """
        swa_metadata = metadata.swa
        # Never let a stash from an earlier branch reach the kernel.
        swa_metadata._pcp_swa_async = None
        stage_cache = getattr(metadata, "swa_stage_cache", None)
        _pcp_stage_note_step(get_forward_context().attn_metadata, stage_cache)
        _pcp_stage_count("swa", "attempts")
        config = self.pcp_config
        if not config.owner_stage_swa_chunk:
            _pcp_stage_count("swa", "skipped_config")
            return
        if swa_metadata.replay_block_table is not None:
            _pcp_stage_count("swa", "skipped_replay")
            return
        if getattr(attn, "packed_cache_ops", None) is not None:
            _pcp_stage_count("swa", "skipped_packed")
            return
        if swa_metadata.num_decodes > 0:
            _pcp_stage_count("swa", "skipped_mixed")
            return
        if swa_metadata.num_prefills <= 0:
            _pcp_stage_count("swa", "skipped_nonprefill")
            return
        if swa_metadata.ori_sparse_indices is not None:
            _pcp_stage_count("swa", "skipped_sparse")
            return
        if swa_metadata.ori_topk_length is not None:
            _pcp_stage_count("swa", "skipped_topk")
            return
        if not swa_metadata.storage_block_size or swa_metadata.ori_win_left < 0:
            _pcp_stage_count("swa", "skipped_window")
            return
        negative = _pcp_stage_negative(stage_cache)
        if negative is not None:
            _pcp_stage_count("swa", f"skipped_{negative}")
            return
        from vllm_ascend.attention.context_parallel.pcp_owner_stage import (
            get_swa_chunk_stage_cache,
        )
        from vllm_ascend.attention.context_parallel.pcp_owner_vmm import (
            owner_slab_for_tensor,
        )

        swa_cache = attn.dsa_attn.swa_cache_layer.kv_cache
        slab = owner_slab_for_tensor(swa_cache)
        if slab is None or not slab.active:
            _pcp_stage_count("swa", "skipped_nonowner")
            return
        has_compressed = self.role.compress_ratio in (1, 2)
        num_reqs = swa_metadata.num_reqs
        ori_block_table = swa_metadata.block_table[:num_reqs]
        cache = get_swa_chunk_stage_cache(
            swa_cache.device,
            swa_cache,
            budget_bytes=config.owner_stage_swa_chunk_bytes,
        )
        # Recorded after ``_preprocess`` issued the SWA/long-KV writes on the
        # main stream (the multistream path joins its side stream before
        # returning), so the copy can never read a half-written cache.
        start = torch.npu.current_stream().record_event()
        stream = cache.staging_stream()
        issue = _timing_enter("stage_swa_chunk_issue", config.enable_timing)
        staged = None
        rebuilt = None
        plan = None
        with npu_stream_switch(stream, enabled=True):
            stream.wait_event(start)
            plan_token = _timing_enter("stage_swa_plan", config.enable_timing)
            plan = _build_swa_chunk_plan(
                swa_metadata,
                ori_block_table,
                num_reqs,
                int(swa_metadata.storage_block_size),
                int(swa_metadata.ori_win_left) + 1,
                int(swa_cache.shape[0]),
                stage_cache=stage_cache,
            )
            _timing_exit(plan_token)
            if plan is not None and plan.total_blocks > 0:
                smla_token = _timing_enter("stage_swa_smla", config.enable_timing)
                before_entries = len(stage_cache.smla_cache) if stage_cache is not None else 0
                rebuilt = _build_swa_chunk_smla_metadata(
                    attn,
                    swa_metadata,
                    metadata.attention if has_compressed else swa_metadata,
                    plan,
                    stage_cache=stage_cache,
                    has_compressed=has_compressed,
                    ratio=self.role.compress_ratio if has_compressed else 0,
                    cmp_seq_lens=(metadata.attention.cache_seq_lens[:num_reqs] if has_compressed else None),
                    cmp_residual=(metadata.attention.cmp_residual if has_compressed else None),
                    cmp_topk=self.topology.index_topk if has_compressed else 0,
                    num_reqs=num_reqs,
                )
                _timing_exit(smla_token)
                if stage_cache is not None:
                    missed = max(0, len(stage_cache.smla_cache) - before_entries)
                    _PCP_STAGE_STEP["smla_ops"] += missed
                    _PCP_STAGE_STEP["step_smla_ops"] += missed
                if rebuilt is not None:
                    staged = cache.stage_async(
                        swa_cache,
                        plan.window_block_table,
                        counts=plan.counts,
                        prepared=plan.prepared,
                        enable_timing=config.enable_timing,
                        start_event=start,
                    )
        _timing_exit(issue)
        if plan is None or plan.total_blocks <= 0:
            _pcp_stage_count("swa", "skipped_plan")
            _pcp_stage_mark_negative(stage_cache, "plan")
            return
        if rebuilt is None:
            _pcp_stage_count("swa", "skipped_metadata")
            _pcp_stage_mark_negative(stage_cache, "metadata")
            return
        if staged is None:
            _pcp_stage_count("swa", "skipped_overflow")
            _pcp_stage_mark_negative(stage_cache, "overflow")
            return
        _pcp_stage_count("swa", "ok")
        _PCP_STAGE_STEP["swa_max_blocks"] = max(_PCP_STAGE_STEP["swa_max_blocks"], int(plan.total_blocks))
        _PCP_STAGE_STEP["swa_capacity"] = int(cache.capacity_blocks)
        swa_metadata._pcp_swa_async = (staged[2], (staged[0], staged[1]), plan, rebuilt, has_compressed)

    def _start_owner_long_kv_stage_async(self, attn, metadata):
        """Launch the long-KV source copy on a side stream after the writes.

        Mirrors ``_maybe_stage_owner_cmp_cache`` but issues the copy early:
        ``_forward_attention`` only joins the done event before the kernel, so
        the copy overlaps the indexer selection.  One launch per source group
        and step (the memo below), decode/mixed/non-owner/overflow keep the
        synchronous path.
        """
        if not self.role.has_long_context:
            return
        cmp_metadata = metadata.attention
        if cmp_metadata is None or getattr(attn, "packed_cache_ops", None) is not None:
            return
        _pcp_stage_count("long_kv", "async_attempts")
        if getattr(cmp_metadata, "num_prefills", 0) <= 0:
            _pcp_stage_count("long_kv", "async_skipped_nonprefill")
            return
        if getattr(cmp_metadata, "num_decodes", 0) > 0:
            _pcp_stage_count("long_kv", "async_skipped_mixed")
            return
        config = self.pcp_config
        if not config.owner_stage:
            _pcp_stage_count("long_kv", "async_skipped_config")
            return
        from vllm_ascend.attention.context_parallel.pcp_owner_stage import get_stage_cache
        from vllm_ascend.attention.context_parallel.pcp_owner_vmm import owner_slab_for_tensor

        source_cache = get_forward_context().no_compile_layers[self.long_kv_source_prefix].kv_cache
        if source_cache is None:
            _pcp_stage_count("long_kv", "async_skipped_config")
            return
        pointer = int(source_cache.data_ptr())
        memo = getattr(cmp_metadata, "_pcp_stage_memo", None)
        if memo is not None and memo[0] == pointer:
            _pcp_stage_count("long_kv", "async_memo_hit")
            return
        if getattr(cmp_metadata, "_pcp_stage_async", None) is not None:
            _pcp_stage_count("long_kv", "async_pending")
            return
        if owner_slab_for_tensor(source_cache) is None:
            _pcp_stage_count("long_kv", "async_skipped_nonowner")
            return
        num_reqs = metadata.swa.num_reqs
        block_table = cmp_metadata.block_table[:num_reqs]
        seq_lens = cmp_metadata.cache_seq_lens[:num_reqs]
        cache = get_stage_cache(source_cache.device, source_cache, budget_bytes=config.owner_stage_bytes)
        start = torch.npu.current_stream().record_event()
        timing = _timing_enter("stage_long_kv_issue", config.enable_timing)
        staged = cache.stage_async(
            source_cache,
            block_table,
            seq_lens=seq_lens,
            block_size=getattr(cmp_metadata, "storage_block_size", None),
            enable_timing=config.enable_timing,
            start_event=start,
        )
        _timing_exit(timing)
        if staged is None:
            _pcp_stage_count("long_kv", "async_skipped_overflow")
            return
        cmp_metadata._pcp_stage_async = (pointer, (staged[0], staged[1]), staged[2])
        _pcp_stage_count("long_kv", "async_ok")
        _PCP_STAGE_STEP["long_kv_max_blocks"] = max(
            _PCP_STAGE_STEP["long_kv_max_blocks"], int(cache.last_total_blocks)
        )
        _PCP_STAGE_STEP["long_kv_capacity"] = int(cache.capacity_blocks)

    def __init__(self, prefix, role, topology, long_kv_source_prefix, index_k_source_prefix):
        ascend_config = get_ascend_config()
        self.multistream_dsv4_dsa_overlap = ascend_config.multistream_dsv4_dsa_overlap
        self._pcp_config = getattr(ascend_config, "pcp_config", None) or PcpConfig()
        self.prefix = prefix
        self.layer_name = f"{prefix}.attn"
        self.role = role
        self.topology = topology
        self.swa_prefix = f"{prefix}.swa_cache"
        self.long_kv_source_prefix = long_kv_source_prefix
        self.index_k_source_prefix = index_k_source_prefix
        self.compressor_state_prefix = (
            f"{prefix}.compressor.state_cache" if role.is_kv_source and role.compress_ratio == 2 else None
        )

    @property
    def pcp_config(self) -> PcpConfig:
        """Typed PCP settings; tests that bypass ``__init__`` get the defaults."""
        config = getattr(self, "_pcp_config", None)
        if config is None:
            config = PcpConfig()
            self._pcp_config = config
        return config

    def _get_layer_metadata(self, metadata) -> DeepseekV41LayerMetadata:
        swa = metadata[self.swa_prefix]
        long_kv = metadata[self.long_kv_source_prefix] if self.long_kv_source_prefix is not None else None
        index_k = metadata[self.index_k_source_prefix] if self.index_k_source_prefix is not None else None
        compressor_state = metadata[self.compressor_state_prefix] if self.compressor_state_prefix is not None else None
        # The builder publishes one staging cache per forward through the
        # shared batch dictionary; every cache group receives the same
        # instance, so planning and operator metadata are shared per batch.
        stage_cache = getattr(swa, "swa_stage_cache", None)
        if not isinstance(stage_cache, _SwaChunkStageCache):
            stage_cache = _swa_chunk_stage_cache(swa)
        return DeepseekV41LayerMetadata(
            attention=long_kv,
            swa=swa,
            compressor=(
                DeepseekV41CompressorMetadata(long_kv, compressor_state)
                if self.role.is_kv_source and long_kv is not None
                else None
            ),
            indexer=(DeepseekV41IndexerMetadata(index_k) if index_k is not None else None),
            swa_stage_cache=stage_cache,
        )

    @staticmethod
    def _apply_rotary(attn, value, cos, sin, *, inverse=False):
        return DeviceOperator.apply_partial_rotary_inplace(
            value, cos, sin, start=attn.nope_head_dim, end=attn.head_dim, inverse=inverse
        )

    @staticmethod
    def _write_swa_cache(attn, metadata, value, pcp_config: PcpConfig | None = None):
        if not value.shape[0]:
            return
        cache = attn.dsa_attn.swa_cache_layer.kv_cache
        backend = getattr(attn, "packed_cache_ops", None)
        if backend is not None:
            backend.write_attention_cache(
                cache,
                metadata.flat_slot_mapping[: value.shape[0]],
                value,
                kind="win",
            )
            return
        scatter_cache_sk(cache, metadata.slot_mapping, value)

    @staticmethod
    def _write_swa_cache_on_main_stream(attn, metadata):
        # Replicated-cache collectives and packaged A5 prefill/mixed writes
        # run on the current stream after the independent Q/KV work joins.
        # A missing prefill classification also cannot qualify as pure decode.
        return metadata.global_metadata is not None or (
            getattr(attn, "packed_cache_ops", None) is not None
            and (metadata.num_prefills > 0 or metadata.num_decodes != metadata.num_reqs)
        )

    @classmethod
    def _project_kv(cls, attn, hidden_states, cos, sin):
        kv = attn.kv_norm(attn.wkv(hidden_states)).view(-1, 1, attn.head_dim)
        cls._apply_rotary(attn, kv, cos, sin)
        return kv.squeeze(1)

    def _update_caches(self, attn, hidden_states, metadata):
        if hidden_states.shape[0] == 0:
            return
        cos, sin = metadata.rope(attn.rotary_emb.layername, hidden_states.shape[0])
        kv = self._project_kv(attn, hidden_states, cos, sin)
        AscendDSAV41Impl._write_swa_cache(attn, metadata.swa, kv, self.pcp_config)
        if self.role.is_kv_source:
            self._compress_and_write_caches(attn, hidden_states, cos, sin, metadata, metadata)

    def _prepare_inputs_and_caches(self, attn, hidden_states, metadata, metadata_by_prefix):
        return self._preprocess(attn, hidden_states, metadata, metadata)

    def _preprocess(self, attn, hidden_states, local_metadata, global_metadata):
        """Prepare rank-local queries and replicated caches.

        All projections use the same local hidden states and RoPE. Global
        metadata addresses the gathered caches; without PCP both views coincide.
        """
        hidden_states = hidden_states[: local_metadata.swa.num_actual_tokens]
        cos, sin = (
            local_metadata.rope(attn.rotary_emb.layername, hidden_states.shape[0])
            if hidden_states.shape[0]
            else (None, None)
        )
        if hidden_states.shape[0] and self.multistream_dsv4_dsa_overlap:
            q, qr = self._multistream_qkv(attn, hidden_states, cos, sin, local_metadata.swa)
        else:
            if hidden_states.shape[0]:
                qr = attn.q_norm(attn.wq_a(hidden_states))
                q = attn.wq_b(qr).unflatten(-1, (-1, attn.head_dim))
                kv = self._project_kv(attn, hidden_states, cos, sin)
            else:
                q = qr = None
                kv = hidden_states.new_empty((0, attn.head_dim))
            # Empty PCP ranks still participate in the replicated-cache gather.
            self._write_swa_cache(attn, local_metadata.swa, kv, self.pcp_config)
        prepared_indexer = None
        if q is not None:
            self._apply_rotary(attn, q, cos, sin)
            # Consume local Query RoPE before compression can reuse source tables.
            prepared_indexer = self._prepare_indexer_inputs(attn, hidden_states, qr, cos, sin)
        self._prepare_compressed_cache(attn, hidden_states, cos, sin, local_metadata, global_metadata, prepared_indexer)
        return q.to(hidden_states.dtype) if q is not None else None, prepared_indexer

    def _prepare_compressed_cache(
        self, attn, hidden_states, cos, sin, local_metadata, global_metadata, prepared_indexer
    ):
        """Prepare long KV and Indexer K after the local query RoPE has been consumed."""
        # Empty PCP source ranks must still enter the cache collectives.
        if self.role.is_kv_source:
            self._compress_and_write_caches(
                attn,
                hidden_states,
                cos,
                sin,
                local_metadata,
                global_metadata,
                prepared_indexer=prepared_indexer,
            )
        else:
            self._quantize_indexer_query(attn, prepared_indexer, local_metadata)

    def _prepare_indexer_inputs(self, attn, hidden_states, qr, cos, sin):
        if not self.role.has_long_context or not self.role.is_index_source:
            return None
        indexer = attn.indexer
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()

        query = indexer.project_query(qr)
        weights_start = main_stream.record_event()
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(weights_start)
            weights = indexer.project_weights(hidden_states)
            weights_done = aux_stream.record_event()
        indexer.apply_query_rope(query, cos, sin)
        main_stream.wait_event(weights_done)
        return DeepseekV41PreparedIndexer(query=query, weights=weights)

    @staticmethod
    def _should_quantize_indexer(prepared, metadata):
        return (
            prepared is not None
            and prepared.quantized_query is None
            and prepared.query is not None
            and prepared.query.shape[0] > 0
            and metadata.indexer.cache.max_cache_seq_len > 0
        )

    def _quantize_indexer_query(self, attn, prepared, metadata):
        if not self._should_quantize_indexer(prepared, metadata):
            return
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()
        quantize_start = main_stream.record_event()
        with npu_stream_switch(aux_stream, enabled=True):
            aux_stream.wait_event(quantize_start)
            prepared.quantized_query, prepared.query_scale = attn.indexer.quantize_query(prepared.query)
            prepared.quantize_done = aux_stream.record_event()

    def _project_output(self, attn, output, hidden_states, *, projected):
        padded = output
        if output.shape[0] != hidden_states.shape[0]:
            padded = output.new_zeros((hidden_states.shape[0], output.shape[1], output.shape[2]))
            padded[: output.shape[0]] = output
        attn.dsa_attn.dsa_attn.impl._forward_o_proj(padded, projected)
        return projected

    def _multistream_qkv(self, attn, hidden_states, cos, sin, swa_metadata):
        """Overlap Q Vector work with KV Cube work, then reverse their roles.

        Reuse V1's stream and projection wrappers. V4.1 keeps floating-point
        qr for its indexer and has no post-Wq_b Q RMSNorm. Stage events serialize
        the Cube matmuls; the final join makes SWA writes visible to attention.
        PCP gathers KV through its cache-write override on the current stream.
        """
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()
        v1_impl = attn.dsa_attn.dsa_attn.impl
        wq_a, wkv, wq_b = v1_impl.cv_wq_a, v1_impl.cv_wkv, v1_impl.cv_wq_b
        # ModelSlim wraps different quantization schemes in the same adapter.
        q_scheme = getattr(wq_a._quant_method, "quant_method", wq_a._quant_method)
        kv_scheme = getattr(wkv._quant_method, "quant_method", wkv._quant_method)
        share_quant = type(q_scheme) is type(kv_scheme) and wq_a._has_communication == wkv._has_communication
        # Part 1: Q_a matmul (Cube) overlaps independent KV quantization (Vector).
        q_quant, q_scale = wq_a.quantize(hidden_states)
        kv_quant_done = None
        if share_quant:
            kv_quant, kv_scale = q_quant, q_scale
        else:
            q_quant_done = main_stream.record_event()
            with npu_stream_switch(aux_stream):
                aux_stream.wait_event(q_quant_done)
                kv_quant, kv_scale = wkv.quantize(hidden_states)
                kv_quant_done = aux_stream.record_event()
        q_a = wq_a.matmul(q_quant, q_scale, bias=attn.wq_a.bias)

        # Part 2: Q normalization/quantization (Vector) overlaps KV matmul (Cube).
        part2_start = main_stream.record_event()
        if kv_quant_done is not None:
            main_stream.wait_event(kv_quant_done)
        with npu_stream_switch(aux_stream):
            aux_stream.wait_event(part2_start)
            kv = wkv.matmul(kv_quant, kv_scale, bias=attn.wkv.bias)
            kv_matmul_done = aux_stream.record_event()
        qr = attn.q_norm(q_a)
        q_b_quant, q_b_scale = wq_b.quantize(qr)

        # Part 3: Q_b matmul (Cube) overlaps KV norm and RoPE (Vector).
        # A5 prefill/mixed cache publication follows the stream join below.
        # Forward the typed PCP config: every rank in a PCP group must make the
        # same local-vs-gathered write decision. Omitting it here would force
        # this rank to gather while an empty or non-multistream peer writes
        # locally, leaving the cache collectives unmatched.
        write_cache_on_main = self._write_swa_cache_on_main_stream(attn, swa_metadata)
        part3_start = main_stream.record_event()
        main_stream.wait_event(kv_matmul_done)
        with npu_stream_switch(aux_stream):
            aux_stream.wait_event(part3_start)
            kv = attn.kv_norm(kv).view(-1, 1, attn.head_dim)
            self._apply_rotary(attn, kv, cos, sin)
            if not write_cache_on_main:
                self._write_swa_cache(attn, swa_metadata, kv.squeeze(1), self.pcp_config)
        q = wq_b.matmul(q_b_quant, q_b_scale, bias=attn.wq_b.bias).unflatten(-1, (-1, attn.head_dim))
        main_stream.wait_stream(aux_stream)
        if write_cache_on_main:
            self._write_swa_cache(attn, swa_metadata, kv.squeeze(1), self.pcp_config)
        return q, qr

    def _compress_and_write_caches(
        self, attn, hidden_states, cos, sin, local_metadata, global_metadata, *, prepared_indexer=None
    ):
        if not hidden_states.shape[0]:
            return
        compressor = attn.compressor
        compressor_metadata = local_metadata.compressor
        ratio = self.role.compress_ratio
        if ratio == 1:
            self._quantize_indexer_query(attn, prepared_indexer, local_metadata)
            latent = compressor(hidden_states)
            # C1 source positions are the current token positions. Reuse the
            # query RoPE selected by the SWA metadata builder instead of
            # indexing the global table a second time.
            source_cos = cos
            source_sin = sin
        else:
            state_metadata = compressor_metadata.state
            if state_metadata.c2_metadata_group_id is not None:
                wait_for_device_metadata(DeviceMetadataStage.COMPRESSOR, state_metadata.c2_metadata_group_id)
            hidden_states_fp32 = hidden_states.float()
            # Finish the input cast before query quantization starts on the
            # auxiliary stream, so the WKV matmul can overlap that quantization.
            self._quantize_indexer_query(attn, prepared_indexer, local_metadata)
            kv = compressor.wkv(hidden_states_fp32)
            score = compressor.wgate(hidden_states_fp32)
            latent = compressor.pool_projected(kv, score, state_metadata, prev_rows=None, prev_row_indices=None)
            source_cos = state_metadata.c2_source_cos
            source_sin = state_metadata.c2_source_sin
            if source_cos is None or source_sin is None:
                fallback_cos, fallback_sin = get_cos_and_sin_dsa(state_metadata.c2_source_positions)
                source_cos = fallback_cos[attn.rotary_emb.layername]
                source_sin = fallback_sin[attn.rotary_emb.layername]

        self._write_compressed_caches(attn, latent, source_cos, source_sin, global_metadata)

    def _write_compressed_caches(self, attn, latent, source_cos, source_sin, metadata):
        compressor_metadata = metadata.compressor
        indexer_metadata = metadata.indexer
        num_tokens = latent.shape[0]
        # PCP restoration may remove padding retained by the source RoPE tables.
        source_cos = source_cos[:num_tokens]
        source_sin = source_sin[:num_tokens]
        index_slots = indexer_metadata.cache.slot_mapping[:num_tokens]
        long_slots = compressor_metadata.cache.slot_mapping[:num_tokens]
        attn.indexer.update_keys(
            latent,
            index_slots,
            source_cos,
            source_sin,
        )
        latent = latent.view(-1, 1, attn.head_dim)
        AscendDSAV41Impl._apply_rotary(attn, latent, source_cos, source_sin)
        backend = getattr(attn, "packed_cache_ops", None)
        if backend is not None:
            backend.write_attention_cache(
                attn.long_kv_cache.kv_cache,
                compressor_metadata.cache.flat_slot_mapping[:num_tokens],
                latent.squeeze(1),
                kind="cmp",
            )
        else:
            scatter_cache_sk(
                attn.long_kv_cache.kv_cache,
                long_slots,
                latent.squeeze(1),
            )

    def _select_sparse_indices(self, attn, positions, metadata, *, prepared_indexer):
        num_tokens = positions.shape[0]
        if not self.role.has_long_context:
            return None
        shared = attn.shared_state
        if not self.role.is_index_source:
            return shared.topk_indices[:num_tokens]

        context = get_forward_context().no_compile_layers
        source_layer = context[self.index_k_source_prefix]
        source_cache = source_layer.kv_cache
        if self.role.uses_candidate_filter and getattr(attn.indexer, "packed_cache_ops", None) is not None:
            folded_name = self.index_k_source_prefix + "_folded"
            folded_cache = context[folded_name].kv_cache
            source_cache = (*source_cache, folded_cache)
        assert prepared_indexer is not None
        if prepared_indexer.quantize_done is not None:
            torch.npu.current_stream().wait_event(prepared_indexer.quantize_done)
        selected, candidates = attn.indexer.select_projected(
            prepared_indexer.query,
            prepared_indexer.weights,
            positions,
            source_cache,
            metadata.indexer.cache,
            quantized_query=prepared_indexer.quantized_query,
            query_scale=prepared_indexer.query_scale,
            is_candidate_source=self.role.is_candidate_source,
            uses_candidate_filter=self.role.uses_candidate_filter,
            candidate_topk_blocks=self.topology.candidate_topk_blocks,
            candidate_block_size=self.topology.candidate_block_size,
            candidates=shared.candidates[:num_tokens],
            candidate_lengths=(None if shared.candidate_lengths is None else shared.candidate_lengths[:num_tokens]),
            topk_lengths=(None if shared.topk_lengths is None else shared.topk_lengths[:num_tokens]),
            indices_output=shared.topk_indices[:num_tokens],
        )
        # An empty long-context cache does not write the stable output buffer.
        if selected.shape[1] == 0:
            shared.topk_indices[: selected.shape[0]].fill_(-1)
        elif not getattr(attn, "uses_a5_packed_cache", False):
            shared.topk_indices[: selected.shape[0]].copy_(selected)
        if self.role.is_candidate_source:
            shared.candidates[: candidates.shape[0]].copy_(candidates)
        return shared.topk_indices[:num_tokens]

    def _forward_attention(self, attn, q, metadata, compressed_indices, *, source_cache=None):
        """Run SparseFlashMla with the same PA metadata for both operator stages."""
        if source_cache is None and self.role.has_long_context:
            source_cache = get_forward_context().no_compile_layers[self.long_kv_source_prefix].kv_cache
        backend = getattr(attn, "packed_cache_ops", None)
        if backend is not None:
            compressed_lengths = None
            if self.role.has_long_context and attn.shared_state.topk_lengths is not None:
                compressed_lengths = attn.shared_state.topk_lengths[: q.shape[0]]
            return backend.qsmla(
                q,
                attn.dsa_attn.swa_cache_layer.kv_cache,
                source_cache,
                metadata,
                compressed_indices,
                compressed_lengths=compressed_lengths,
                window_size=attn.window_size,
                sinks=attn.attn_sink,
                softmax_scale=attn.softmax_scale,
            )
        has_compressed = self.role.compress_ratio in (1, 2)
        ratio = self.role.compress_ratio if has_compressed else 0
        num_reqs = metadata.swa.num_reqs
        query_start_loc = metadata.swa.query_start_loc[: num_reqs + 1]
        seq_lens = metadata.swa.seq_lens[:num_reqs]
        ori_block_table = metadata.swa.block_table[:num_reqs]
        cmp_block_table = None
        cmp_seq_lens = None
        cmp_residual = None
        cmp_indices = None
        cmp_topk = 0
        if has_compressed:
            cmp_block_table = metadata.attention.block_table[:num_reqs]
            cmp_seq_lens = metadata.attention.cache_seq_lens[:num_reqs]
            cmp_residual = metadata.attention.cmp_residual
            cmp_topk = self.topology.index_topk
            cmp_indices = pad_sparse_indices(compressed_indices, cmp_topk)
            timing = _timing_enter("stage", self.pcp_config.enable_timing)
            pending = getattr(metadata.attention, "_pcp_stage_async", None)
            if pending is not None and source_cache is not None and pending[0] == int(source_cache.data_ptr()):
                metadata.attention._pcp_stage_async = None
                wait_timing = _timing_enter("stage_long_kv_wait", self.pcp_config.enable_timing)
                torch.npu.current_stream().wait_event(pending[2])
                _timing_exit(wait_timing)
                staged = pending[1]
                metadata.attention._pcp_stage_memo = (pending[0], staged)
                _pcp_stage_count("long_kv", "async_consumed")
            else:
                staged = _maybe_stage_owner_cmp_cache(
                    source_cache,
                    cmp_block_table,
                    metadata.attention,
                    num_reqs,
                    self.pcp_config,
                    seq_lens=cmp_seq_lens,
                    block_size=getattr(metadata.attention, "storage_block_size", None),
                )
            _timing_exit(timing)
            if staged is not None:
                source_cache, cmp_block_table = staged

        operator_metadata = metadata.attention if has_compressed else metadata.swa
        op_metadata = operator_metadata.smla_metadata
        wait_for_device_metadata(
            DeviceMetadataStage.ATTENTION,
            id(op_metadata),
        )
        swa_cache = attn.dsa_attn.swa_cache_layer.kv_cache
        if metadata.swa.replay_block_table is not None:
            # Rebase only SWA: A3 CSA cannot mask its missing prefix with
            # sparse ori indices. Compressed KV retains its causal coordinates.
            timing = _timing_enter("swa_gather", self.pcp_config.enable_timing)
            swa_cache = swa_cache[metadata.swa.replay_cache_blocks, metadata.swa.replay_cache_offsets]
            _timing_exit(timing)
            swa_cache = swa_cache.reshape(-1, metadata.swa.storage_block_size, 1, swa_cache.shape[-1])
            ori_block_table = metadata.swa.replay_block_table
            seq_lens = metadata.swa.replay_seq_lens
        elif self.pcp_config.owner_stage_swa_chunk:
            # Each rank-local row is already one DualChunkSwap chunk, so the
            # kernel only needs ``[chunk_start - window, chunk_end)`` of the
            # owner slab. Stage that bounded range and rebase the operator
            # coordinates onto it; the sliding-window mask is relative, so it
            # is unchanged. Decodes, mixed batches, non-owner caches and sparse
            # ori indices keep the direct peer-read path.
            #
            # ``_start_owner_swa_stage_async`` already built the tables and
            # issued the copy on a side stream after the cache writes, so here
            # we only join that event before the kernel.  A missing stash
            # (decode/mixed, non-owner cache, overflow, packed-cache path) runs
            # the synchronous staging below exactly as before.
            async_stage = getattr(metadata.swa, "_pcp_swa_async", None)
            metadata.swa._pcp_swa_async = None
            staged = None
            plan = None
            rebuilt = None
            if async_stage is not None:
                done, staged_pair, async_plan, async_rebuilt, async_compressed = async_stage
                # Always join the side-stream copy, even when the compressed
                # geometry no longer matches: the fallback below copies into
                # the same scratch, so it must wait for the async write.
                timing = _timing_enter("stage_swa_chunk_wait", self.pcp_config.enable_timing)
                torch.npu.current_stream().wait_event(done)
                _timing_exit(timing)
                if async_compressed == has_compressed:
                    staged, plan, rebuilt = staged_pair, async_plan, async_rebuilt
            if staged is None:
                fallback_stage_cache = getattr(metadata, "swa_stage_cache", None)
                eligible = (
                    # A negative from the overlapped path means this step's
                    # plan/metadata/overflow already failed; retrying the same
                    # device work per layer would only add latency.
                    _pcp_stage_negative(fallback_stage_cache) is None
                    and metadata.swa.num_decodes == 0
                    and metadata.swa.num_prefills > 0
                    and metadata.swa.ori_sparse_indices is None
                    # A per-token visible-length floor (bounded replay) is tied
                    # to the unbounded coordinate system; keep that path untouched.
                    and metadata.swa.ori_topk_length is None
                    and metadata.swa.storage_block_size
                    and metadata.swa.ori_win_left >= 0
                )
                if eligible:
                    from vllm_ascend.attention.context_parallel.pcp_owner_stage import (
                        get_swa_chunk_stage_cache,
                    )
                    from vllm_ascend.attention.context_parallel.pcp_owner_vmm import (
                        owner_slab_for_tensor,
                    )

                    if owner_slab_for_tensor(swa_cache) is not None:
                        timing = _timing_enter("stage_swa_chunk", self.pcp_config.enable_timing)
                        stage_cache = getattr(metadata, "swa_stage_cache", None)
                        plan = _build_swa_chunk_plan(
                            metadata.swa,
                            ori_block_table,
                            num_reqs,
                            int(metadata.swa.storage_block_size),
                            int(metadata.swa.ori_win_left) + 1,
                            int(swa_cache.shape[0]),
                            stage_cache=stage_cache,
                        )
                        if plan is not None and plan.total_blocks > 0:
                            rebuilt = _build_swa_chunk_smla_metadata(
                                attn,
                                metadata.swa,
                                operator_metadata,
                                plan,
                                stage_cache=stage_cache,
                                has_compressed=has_compressed,
                                ratio=ratio,
                                cmp_seq_lens=cmp_seq_lens,
                                cmp_residual=cmp_residual,
                                cmp_topk=cmp_topk,
                                num_reqs=num_reqs,
                            )
                            if rebuilt is not None:
                                staged = get_swa_chunk_stage_cache(
                                    swa_cache.device,
                                    swa_cache,
                                    budget_bytes=self.pcp_config.owner_stage_swa_chunk_bytes,
                                ).stage(
                                    swa_cache,
                                    plan.window_block_table,
                                    counts=plan.counts,
                                    prepared=plan.prepared,
                                    enable_timing=self.pcp_config.enable_timing,
                                )
                        _timing_exit(timing)
            if staged is not None:
                swa_cache, ori_block_table = staged
                seq_lens = plan.replay_seq_lens
                op_metadata = rebuilt
        timing = _timing_enter("kernel", self.pcp_config.enable_timing)
        output, _ = torch.ops._C_ascend.npu_sparse_flash_mla(
            q,
            ori_kv=swa_cache,
            cmp_kv=source_cache,
            ori_sparse_indices=metadata.swa.ori_sparse_indices,
            ori_topk_length=metadata.swa.ori_topk_length,
            cmp_sparse_indices=cmp_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=query_start_loc,
            seqused_ori_kv=seq_lens,
            seqused_cmp_kv=cmp_seq_lens,
            cmp_residual_kv=cmp_residual,
            sinks=attn.attn_sink,
            metadata=op_metadata,
            softmax_scale=attn.softmax_scale,
            cmp_ratio=ratio,
            ori_mask_mode=metadata.swa.ori_mask_mode,
            cmp_mask_mode=3 if has_compressed else 0,
            ori_win_left=metadata.swa.ori_win_left,
            ori_win_right=metadata.swa.ori_win_right,
            layout_q="TND",
            layout_kv="PA_BBND",
            topk_value_mode=1,
            return_softmax_lse=False,
        )
        _timing_exit(timing)
        return output

    @staticmethod
    def update_graph_params(*args, **kwargs):
        """V4.1 owns stable metadata buffers; no backend pointer patch is needed."""
        return None

    def forward(self, attn, positions, hidden_states, output: torch.Tensor | None = None):
        # The custom-op caller provides a graph-stable output buffer.  Write
        # O-projection results into it directly instead of materializing a
        # second full hidden-state tensor and copying it at the graph boundary.
        if output is None:
            output = torch.empty_like(hidden_states)
        forward_context = get_forward_context()
        if forward_context.attn_metadata is None:
            projector = attn.dsa_attn.dsa_attn.impl
            output.zero_()
            if oproj_tp_enable():
                capacity = (
                    projector._oproj_send_buf.shape[1]
                    if hasattr(projector, "_oproj_send_buf")
                    else get_potential_max_tokens()
                )
                profile_tokens = min(hidden_states.shape[0], capacity)
                o_proj_input = hidden_states.new_zeros((profile_tokens, attn.n_local_heads, attn.head_dim))
                projector._forward_o_proj(o_proj_input, output[:profile_tokens])
            elif getattr(projector, "enable_pcp_o_proj_weight_sharding", False):
                # Match V4 profiling: exercise OTP communication buffers or
                # PCP resident weight shards before graph capture.
                o_proj_input = hidden_states.new_zeros((hidden_states.shape[0], attn.n_local_heads, attn.head_dim))
                projector._forward_o_proj(o_proj_input, output)
            return output
        metadata = self._get_layer_metadata(forward_context.attn_metadata)
        wait_for_kv_layer_from_connector(attn.v41_layer_name)
        q, prepared_indexer = self._prepare_inputs_and_caches(
            attn, hidden_states, metadata, forward_context.attn_metadata
        )
        num_tokens = metadata.swa.num_actual_tokens
        if num_tokens:
            positions = metadata.positions[:num_tokens]
            cos, sin = metadata.rope(attn.rotary_emb.layername, num_tokens)
            compressed_indices = self._select_sparse_indices(
                attn, positions, metadata, prepared_indexer=prepared_indexer
            )
            attention_output = self._forward_attention(attn, q, metadata, compressed_indices)
            AscendDSAV41Impl._apply_rotary(attn, attention_output, cos, sin, inverse=True)
        else:
            heads = attn.n_local_heads
            attention_output = hidden_states.new_empty((0, heads, attn.head_dim))
        self._project_output(attn, attention_output, hidden_states, projected=output)
        maybe_save_kv_layer_to_connector(attn.v41_layer_name, attn.dsa_attn.swa_cache_layer.kv_cache)
        return output


class AscendDSAV41MetadataBuilder(AttentionMetadataBuilder[AscendDSAV41Metadata]):
    @classmethod
    def get_varlen_cudagraph_max_query_len(cls, vllm_config: VllmConfig, kv_cache_spec) -> int | None:
        # Every proposed row is written to the C2 ring before rejection. Keep
        # the first accepted row available when the next step rolls back to it.
        # Wider blocks could overwrite that row even if only the anchor survives.
        return min(max_decode_query_len(vllm_config), STATE_RING_ROWS)

    def __init__(
        self,
        kv_cache_spec,
        layer_names,
        vllm_config,
        device,
        *,
        build_query_metadata=True,
        build_compressor_metadata=True,
    ):
        if (
            isinstance(kv_cache_spec, AscendMLAAttentionSpec)
            and getattr(kv_cache_spec, "model_version", None) == "deepseek_v41"
        ):
            logical_block_size = vllm_config.cache_config.block_size
            if kv_cache_spec.block_size != logical_block_size:
                updates = {"block_size": logical_block_size}
                if "storage_block_size" in kv_cache_spec.__dataclass_fields__:
                    updates["storage_block_size"] = logical_block_size // get_kv_cache_compression_ratio(kv_cache_spec)
                kv_cache_spec = replace(kv_cache_spec, **updates)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        max_tokens = getattr(vllm_config.scheduler_config, "max_num_batched_tokens", 4096)
        max_reqs = getattr(vllm_config.scheduler_config, "max_num_seqs", 256)
        self.common_v41_metadata: dict[str, Any] = {}
        self.common_v41_batch_metadata: dict[str, Any] = {}
        self.common_v41_smla_metadata: dict[str, Any] = {}
        self._request_counts_metadata: dict[str, int] = {}
        self._supports_device_ops = getattr(device, "type", "cpu") != "cpu"
        # Allocate both decode and prefill owners before graph capture.
        self._build_query_metadata = build_query_metadata
        self._build_compressor_metadata = build_compressor_metadata
        if isinstance(kv_cache_spec, CircularBufferSpec):
            self._cache_kind = "compressor_state"
        elif isinstance(kv_cache_spec, AscendSlidingWindowMLASpec):
            self._cache_kind = "swa"
        elif isinstance(kv_cache_spec, AscendMLAAttentionSpec):
            # make_folded_index_cache_spec packs eight index rows into one:
            # 8 * (128 / 2 FP4 bytes + 128 / 32 scale bytes) = 544 bytes.
            # Its scales are inline (scale_dim=0), like long KV, so identify
            # this layout first to avoid treating it as compressed long KV.
            if kv_cache_spec.tokens_per_state == 8 and kv_cache_spec.head_size == 544:
                self._cache_kind = "index_k_folded"
            else:
                self._cache_kind = "index_k" if kv_cache_spec.scale_dim else "long_kv"
        else:
            raise TypeError(f"Unsupported V4.1 cache spec: {type(kv_cache_spec).__name__}")
        self._device_backend = DeviceOperator.get_dsv41_packed_cache_ops()
        self._uses_a5_packed_cache = self._device_backend is not None
        text_config = vllm_config.model_config.hf_text_config
        window_size = int(_config_value(text_config, "sliding_window", 0))
        self._window_size = window_size
        if self._cache_kind == "swa":
            # Replay padding and query masking must use the same fixed window.
            spec_window = int(kv_cache_spec.sliding_window)
            assert spec_window == window_size, (
                "SWA bounded replay pads the spec's sliding_window "
                f"({spec_window}) while the mask floors on the text config's ({window_size})."
            )
        query_metadata_size = V41_METADATA_BUFFER_SIZE if build_query_metadata else 0
        compressor_tokens = max_tokens if build_compressor_metadata else 0
        compressor_reqs = max_reqs if build_compressor_metadata else 0
        self._slot_mapping = torch.full((max_tokens,), -1, dtype=torch.int64, device=device)
        self._slot_mapping_2d = torch.full((max_tokens, 2), -1, dtype=torch.int32, device=device)
        self._flat_slot_mapping = torch.full((max_tokens,), -1, dtype=torch.int64, device=device)
        self._seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._max_tokens = max_tokens
        self._dspark_swa_indices = None
        self._dspark_swa_lengths = None
        self._cache_seq_lens = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._cmp_residual = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self._smla_metadata = torch.zeros(query_metadata_size, dtype=torch.int32, device=device)
        self._qli_metadata = torch.zeros(query_metadata_size, dtype=torch.int32, device=device)
        if self._supports_device_ops and self._uses_a5_packed_cache and self._cache_kind == "swa":
            self._a5_causal_swa_indices = torch.empty((max_tokens, 1, window_size), dtype=torch.int32, device=device)
            self._a5_causal_swa_lengths = torch.empty((max_tokens, 1), dtype=torch.int32, device=device)
            self._a5_smla_metadata = torch.empty(V41_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device)
            self._a5_smla_length_rows = torch.empty((max_tokens, 1), dtype=torch.int32, device=device)
        else:
            self._a5_causal_swa_indices = None
            self._a5_causal_swa_lengths = None
            self._a5_smla_metadata = None
            self._a5_smla_length_rows = None
        self._a5_smla_plans: dict[int, torch.Tensor] = {}
        self._a5_smla_plan_lengths: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._c2_ring_metadata = torch.zeros(5 * compressor_reqs, dtype=torch.int32, device=device)
        self._c2_complete_mask = torch.zeros(compressor_tokens, dtype=torch.bool, device=device)
        self._c2_source_positions = torch.zeros(compressor_tokens, dtype=torch.int64, device=device)
        rope_dim = int(
            _config_value(
                text_config,
                "qk_rope_head_dim",
                _config_value(text_config, "head_dim"),
            )
        )
        c2_rope_rows = compressor_tokens if self._supports_device_ops and self._cache_kind == "compressor_state" else 0
        self._c2_source_cos = torch.ones(
            (c2_rope_rows, 1, 1, rope_dim),
            dtype=torch.float32,
            device=device,
        )
        self._c2_source_sin = torch.zeros_like(self._c2_source_cos)
        self._c2_rope_layer_names = tuple(
            name.removesuffix(".compressor.state_cache") + ".attn"
            for name in layer_names
            if name.endswith(".compressor.state_cache")
        )
        self._c2_full_source_rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self._device_metadata_enabled = False
        # Default to eager preparation. MRV2 temporarily sets this during
        # FULL warmup/capture so producers and consumer waits share one graph.
        self._device_metadata_in_graph = False
        self._device_metadata_tasks: tuple[DeviceMetadataTask, ...] = ()

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata,
        **kwargs,
    ) -> AscendDSAV41Metadata:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )

    def build_for_drafting(self, common_attn_metadata, draft_index, **kwargs):
        # DSpark issues one eager block per step. Group-local tables and slots
        # remain independent; the builder owns the operator metadata buffers.
        return self.build(0, common_attn_metadata)

    def prepare_source_rope(self) -> None:
        """Validate and cache V4.1 source RoPE tables without async tasks.

        MRV2 keeps metadata tasks synchronous (``_publish_task`` runs them
        inline), so this must NOT flip ``_device_metadata_enabled`` — only
        ``enable_device_metadata`` may turn the async task switch on.
        """
        if self._build_compressor_metadata and self._cache_kind == "compressor_state":
            source_rope = get_full_cos_and_sin_dsa_for_layer(self._c2_rope_layer_names[0])
            self._c2_full_source_rope = source_rope

    def enable_device_metadata(self) -> None:
        self._device_metadata_enabled = True
        self.prepare_source_rope()

    @contextmanager
    def defer_device_metadata(self, *, in_graph: bool = False):
        """Defer this target build without changing the drafter's builder mode."""
        was_enabled = self._device_metadata_enabled
        was_in_graph = self._device_metadata_in_graph
        if in_graph and not self._uses_a5_packed_cache:
            raise ValueError("Captured V4.1 device metadata currently requires the A5 packed-cache backend")
        self.enable_device_metadata()
        self._device_metadata_in_graph = in_graph
        try:
            yield
        finally:
            self._device_metadata_enabled = was_enabled
            self._device_metadata_in_graph = was_in_graph

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        tasks = self._device_metadata_tasks
        self._device_metadata_tasks = ()
        return tasks

    def _publish_task(
        self,
        shared: dict[str, Any],
        key: str,
        buffer: torch.Tensor,
        stage: DeviceMetadataStage,
        run,
    ) -> torch.Tensor:
        existing = shared.get(key)
        if existing is not None:
            return existing
        shared[key] = buffer
        if self._device_metadata_enabled and not (
            self._device_metadata_in_graph and stage == DeviceMetadataStage.COMPRESSOR
        ):
            self._device_metadata_tasks = (
                *self._device_metadata_tasks,
                DeviceMetadataTask(stage, run, id(buffer)),
            )
        else:
            run()
        return buffer

    def _build_batch_metadata(self, common, num_reqs, num_actual_reqs, num_input_tokens):
        self._seq_lens[:num_reqs].copy_(common.seq_lens[:num_reqs])
        if num_actual_reqs < num_reqs:
            self._seq_lens[num_actual_reqs:num_reqs].zero_()
        seq_lens_cpu = getattr(common, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = getattr(common, "_seq_lens_cpu", None)
        max_seq_len = int(getattr(common, "max_seq_len", 0))
        if seq_lens_cpu is not None:
            max_seq_len = int(seq_lens_cpu[:num_actual_reqs].max().item()) if num_actual_reqs else 0
        positions = common.positions
        if positions is not None:
            positions = positions[:num_input_tokens].long()
        return dict(
            query_start_loc=common.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=getattr(common, "query_start_loc_cpu", None),
            seq_lens=self._seq_lens[:num_reqs],
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            max_cache_seq_len=max_seq_len,
        )

    def _new_req_metadata(
        self,
        common,
        num_actual_reqs: int,
    ) -> AscendDSAV41Metadata:
        """Keep the existing metadata interface without preparing unused work"""
        num_reqs = common.num_reqs
        num_tokens = common.num_input_tokens
        is_state = self._cache_kind == "compressor_state"

        values = dict(self.common_v41_batch_metadata["batch"])
        values.update(self._request_counts_metadata)
        values["cache_seq_lens"] = values["seq_lens"]
        stage_cache = self.common_v41_batch_metadata.get("swa_chunk_stage")
        if not isinstance(stage_cache, _SwaChunkStageCache):
            stage_cache = _SwaChunkStageCache()
            self.common_v41_batch_metadata["swa_chunk_stage"] = stage_cache

        return AscendDSAV41Metadata(
            block_table=common.block_table_tensor[:num_reqs],
            slot_mapping=(self._slot_mapping[:num_tokens] if is_state else self._slot_mapping_2d[:0]),
            flat_slot_mapping=None,
            compress_ratio=get_kv_cache_compression_ratio(self.kv_cache_spec),
            storage_block_size=get_storage_block_size(self.kv_cache_spec),
            logical_block_size=self.kv_cache_spec.block_size,
            is_compressor_state=is_state,
            cache_kind=self._cache_kind,
            num_actual_tokens=common.num_actual_tokens,
            num_input_tokens=num_tokens,
            num_reqs=num_reqs,
            num_actual_reqs=num_actual_reqs,
            max_query_len=common.max_query_len,
            max_seq_len=common.max_seq_len,
            attn_state=getattr(common, "attn_state", None),
            is_prefilling=getattr(common, "is_prefilling", None),
            causal=getattr(common, "causal", True),
            swa_stage_cache=stage_cache,
            **values,
        )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
        self._device_metadata_tasks = ()
        common = common_attn_metadata
        num_reqs = int(getattr(common, "num_reqs", common.seq_lens.shape[0]))
        num_actual_reqs = min(int(kwargs.pop("num_actual_reqs", num_reqs)), num_reqs)
        num_input_tokens = int(getattr(common, "num_input_tokens", common.slot_mapping.shape[0]))
        shared = kwargs.get("common_v41_metadata")
        self.common_v41_metadata = {} if shared is None else shared
        batch_shared = kwargs.get("common_v41_batch_metadata")
        self.common_v41_batch_metadata = self.common_v41_metadata if batch_shared is None else batch_shared
        # Classification alone depends on is_prefilling. Lengths, positions and
        # compression calculations remain shared when that field is overridden.
        counts_cache = (
            self.common_v41_metadata if kwargs.get("v41_prefill_override", False) else self.common_v41_batch_metadata
        )
        counts = counts_cache.get("request_counts")
        if counts is None:
            counts = dict(
                zip(
                    ("num_decodes", "num_decode_tokens", "num_prefills", "num_prefill_tokens"),
                    _request_counts(common, num_reqs),
                )
            )
            counts_cache["request_counts"] = counts
        self._request_counts_metadata = counts
        separate_smla = (
            kwargs.get("v41_smla_override", False)
            or getattr(common, "causal", True) is not True
            or (getattr(common, "replay_start", None) is not None and not self._uses_a5_packed_cache)
            or kwargs.get("replay_start") is not None
            or kwargs.get("ori_sparse_indices") is not None
            or kwargs.get("num_query_heads") is not None
        )
        self.common_v41_smla_metadata = self.common_v41_metadata if separate_smla else self.common_v41_batch_metadata
        # Cache references are rebound for every build. Physical addressing is
        # group-local; compatible request coordinates span cache groups.
        if "batch" not in self.common_v41_batch_metadata:
            self.common_v41_batch_metadata["batch"] = self._build_batch_metadata(
                common, num_reqs, num_actual_reqs, num_input_tokens
            )
        return self.build_req_metadata(common, num_actual_reqs, **kwargs)

    def build_req_metadata(
        self, common, num_actual_reqs: int, *, build_query: bool = True, update_cache: bool = True, **kwargs
    ) -> AscendDSAV41Metadata:
        """Preserve the generic non-PCP and legacy CP metadata contract."""
        req = self._new_req_metadata(common, num_actual_reqs)
        self._prepare_cache_lengths(common, req, **kwargs)
        self._build_cache_slots(common, req, **kwargs)
        self._prepare_rope(common, req, **kwargs)
        max_ori_seq_len = self._prepare_query_inputs(common, req, **kwargs)
        if self._build_query_metadata:
            AscendDSAV41AttentionMetadataBuilder._build_smla_metadata(
                self, common, req, max_ori_seq_len=max_ori_seq_len, **kwargs
            )
            AscendDSAV41IndexerMetadataBuilder._build_qli_metadata(self, common, req, **kwargs)
        AscendDSAV41StateMetadataBuilder._build_compressor_cache(self, common, req, **kwargs)
        return req

    def _build_cache_slots(self, common, req: AscendDSAV41Metadata, **kwargs) -> None:
        # SWA uses original-token coordinates; circular state has no token slots.
        # Long KV and index K are addressed in completed compression groups.
        compressed = req.cache_kind in {"long_kv", "index_k"}
        if req.is_compressor_state:
            # State writes use ring ownership metadata; this buffer stays PAD.
            req.slot_mapping = self._slot_mapping[: req.num_input_tokens]
            req.flat_slot_mapping = None
        else:
            # Scope ``shared`` to one framework KV cache group in the model
            # runner. Long KV and Indexer builders with the same physical
            # layout then share one persistent [T, 2] mapping, while every SWA
            # group owns a distinct mapping buffer.
            slot_key = f"slot:coordinates:c{req.compress_ratio}:b{req.storage_block_size}"
            flat_slot_key = f"slot:flat:c{req.compress_ratio}:b{req.storage_block_size}"
            prepared_slots = self.common_v41_metadata.get(slot_key)
            prepared_flat_slots = self.common_v41_metadata.get(flat_slot_key)
            if prepared_slots is None or prepared_flat_slots is None:
                active_slots = common.slot_mapping[: req.num_input_tokens]
                if self._uses_a5_packed_cache and active_slots.device.type == "npu":
                    assert self._device_backend is not None
                    prepared_slots, prepared_flat_slots = self._device_backend.build_packed_cache_slot_mapping(
                        active_slots,
                        req.positions,
                        common.query_start_loc,
                        req.num_input_tokens,
                        req.num_actual_reqs,
                        req.num_actual_tokens,
                        req.storage_block_size,
                        req.compress_ratio if compressed else 1,
                        skip_update=kwargs.get("skip_ring_state_update", False),
                        coordinates_output=self._slot_mapping_2d[: req.num_input_tokens],
                        flat_output=self._flat_slot_mapping[: req.num_input_tokens],
                    )
                else:
                    if compressed and req.compress_ratio != 1:
                        active_slots = compressed_slot_mapping(active_slots, req.compress_ratio)
                    # The common slot-mapping producer already marks padding as PAD.
                    valid = active_slots >= 0
                    if compressed and req.compress_ratio == 2:
                        if kwargs.get("skip_ring_state_update", False):
                            valid.zero_()
                        else:
                            valid_end = common.query_start_loc[req.num_actual_reqs].clamp_max(req.num_actual_tokens)
                            valid &= torch.arange(req.num_input_tokens, device=active_slots.device) < valid_end
                            if req.positions is not None:
                                valid &= req.positions.remainder(2) == 1
                    physical = active_slots.clamp_min(0)
                    self._slot_mapping_2d[: req.num_input_tokens, 0].copy_(
                        torch.where(
                            valid,
                            torch.div(physical, req.storage_block_size, rounding_mode="floor"),
                            -1,
                        )
                    )
                    self._slot_mapping_2d[: req.num_input_tokens, 1].copy_(
                        torch.where(valid, physical.remainder(req.storage_block_size), -1)
                    )
                    self._flat_slot_mapping[: req.num_input_tokens].copy_(torch.where(valid, active_slots, -1))
                    prepared_slots = self._slot_mapping_2d[: req.num_input_tokens]
                    prepared_flat_slots = self._flat_slot_mapping[: req.num_input_tokens]
                self.common_v41_metadata[slot_key] = prepared_slots
                self.common_v41_metadata[flat_slot_key] = prepared_flat_slots
            req.slot_mapping = prepared_slots
            req.flat_slot_mapping = prepared_flat_slots

    def _prepare_cache_lengths(self, common, req: AscendDSAV41Metadata, **kwargs) -> None:
        compressed = req.cache_kind in {"long_kv", "index_k"}
        full_graph_mode = bool(kwargs.get("full_graph_mode", False))
        plane_ratio = req.compress_ratio if compressed else 1
        req.cache_seq_lens = req.seq_lens
        req.cmp_residual = None
        if compressed and req.compress_ratio == 2:
            compressed_lengths = self.common_v41_batch_metadata.get("lengths:c2")
            if compressed_lengths is None:
                torch.div(
                    req.seq_lens, req.compress_ratio, rounding_mode="floor", out=self._cache_seq_lens[: req.num_reqs]
                )
                torch.remainder(req.seq_lens, req.compress_ratio, out=self._cmp_residual[: req.num_reqs])
                compressed_lengths = (self._cache_seq_lens[: req.num_reqs], self._cmp_residual[: req.num_reqs])
                self.common_v41_batch_metadata["lengths:c2"] = compressed_lengths
            req.cache_seq_lens, req.cmp_residual = compressed_lengths
        # Capture the indexer path even when a dummy C2 decode has no
        # completed compression group. Device cache lengths remain exact.
        if full_graph_mode and compressed:
            req.max_cache_seq_len = max(plane_ratio, req.max_cache_seq_len)
        req.max_cache_seq_len //= plane_ratio

    def _prepare_rope(self, common, req: AscendDSAV41Metadata, **kwargs) -> None:
        full_graph_mode = bool(kwargs.get("full_graph_mode", False))
        req.cos = req.sin = None
        if req.cache_kind == "swa" and req.positions is not None:
            rope = kwargs.get("rope_views")
            if rope is None:
                rope = self.common_v41_batch_metadata.get("rope")
            if rope is None:
                rope = get_cos_and_sin_dsa(req.positions, use_cache=(req.num_prefills == 0 or full_graph_mode))
                self.common_v41_batch_metadata["rope"] = rope
            req.cos, req.sin = rope

    def _prepare_query_inputs(self, common, req: AscendDSAV41Metadata, **kwargs) -> int:
        window_size = self._window_size
        req.ori_sparse_indices = kwargs.get("ori_sparse_indices")
        req.ori_topk_length = None
        replay_start = kwargs.get("replay_start")
        req.replay_seq_lens = req.replay_block_table = req.replay_cache_blocks = req.replay_cache_offsets = None
        max_ori_seq_len = int(getattr(common, "max_seq_len", 0))
        if replay_start is not None:
            req.replay_seq_lens = (req.seq_lens - replay_start).clamp_min(0)
            max_ori_seq_len = kwargs["replay_max_seq_len"]
            if req.cache_kind == "swa":
                blocks_per_req = (max_ori_seq_len + req.storage_block_size - 1) // req.storage_block_size
                capacity = blocks_per_req * req.storage_block_size
                cache_positions = replay_start[:, None] + torch.arange(capacity, device=req.seq_lens.device)
                cache_positions = torch.minimum(cache_positions, kwargs["replay_seq_lens"][:, None] - 1).clamp_min(0)
                req.replay_cache_blocks = (
                    common.block_table_tensor[: req.num_reqs]
                    .gather(1, cache_positions.div(req.storage_block_size, rounding_mode="floor").long())
                    .long()
                )
                req.replay_cache_offsets = cache_positions.remainder(req.storage_block_size).long()
                req.replay_block_table = torch.arange(
                    req.num_reqs * blocks_per_req, dtype=torch.int32, device=req.seq_lens.device
                ).view(req.num_reqs, blocks_per_req)
        noncausal = not bool(getattr(common, "causal", True))
        if noncausal and req.ori_sparse_indices is None:
            req.ori_sparse_indices, req.ori_topk_length = build_dspark_swa_indices(
                common.block_table_tensor[: req.num_reqs],
                self.vllm_config.speculative_config.num_speculative_tokens,
                window_size,
                req.storage_block_size,
                common.query_start_loc[: req.num_reqs + 1],
                req.seq_lens,
                req.num_actual_tokens,
                use_logical_indices=True,
            )
            # Full draft graphs retain these addresses from capture. Rebuild
            # the values in persistent buffers rather than returning temporary
            # tensors whose capture-time contents never advance with decoding.
            if self._dspark_swa_indices is None:
                self._dspark_swa_indices = req.ori_sparse_indices.new_empty(
                    (self._max_tokens, *req.ori_sparse_indices.shape[1:])
                )
                self._dspark_swa_lengths = req.ori_topk_length.new_empty(
                    (self._max_tokens, *req.ori_topk_length.shape[1:])
                )
            indices_view = self._dspark_swa_indices[: req.num_actual_tokens]
            assert self._dspark_swa_lengths is not None
            lengths_view = self._dspark_swa_lengths[: req.num_actual_tokens]
            indices_view.copy_(req.ori_sparse_indices)
            lengths_view.copy_(req.ori_topk_length)
            req.ori_sparse_indices, req.ori_topk_length = indices_view, lengths_view
        replay_start = getattr(common, "replay_start", None)
        replay_visible_lens = None
        if replay_start is not None and req.cache_kind == "swa":
            # Physical cache groups can share query coordinates and a replay
            # floor. Cache only within this build; distinct floors retain
            # distinct masks and therefore distinct A5 execution plans.
            replay_key = (
                window_size,
                req.num_reqs,
                req.num_actual_tokens,
                common.query_start_loc.data_ptr(),
                req.seq_lens.data_ptr(),
                replay_start.data_ptr(),
            )
            replay_masks = self.common_v41_batch_metadata.setdefault("replay_swa_visible_lens", {})
            cached_replay = replay_masks.get(replay_key)
            if cached_replay is None:
                replay_visible_lens = build_replay_swa_visible_lens(
                    window_size,
                    common.query_start_loc[: req.num_reqs + 1],
                    req.seq_lens,
                    replay_start,
                    req.num_actual_tokens,
                )
                # Retain the inputs so allocator reuse cannot alias a later
                # group's pointer key within this batch.
                replay_masks[replay_key] = (replay_visible_lens, common.query_start_loc, req.seq_lens, replay_start)
            else:
                replay_visible_lens = cached_replay[0]
            if self._uses_a5_packed_cache and self.common_v41_smla_metadata is self.common_v41_batch_metadata:
                # Isolate different replay contracts independently of the
                # planner's own cache keys (row-only or per-ratio FD plans).
                replay_plans = self.common_v41_batch_metadata.setdefault("replay_smla_plans", {})
                self.common_v41_smla_metadata = replay_plans.setdefault(replay_key, {})
        if (
            not noncausal
            and req.ori_sparse_indices is None
            and self._a5_causal_swa_indices is not None
            and req.positions is not None
        ):
            cache_key = f"a5-causal-swa:w{window_size}"
            cached_window = self.common_v41_batch_metadata.get(cache_key) if replay_visible_lens is None else None
            if cached_window is None:
                assert self._device_backend is not None
                # A5 reads the first N sparse indices, so a replay floor changes both tensors.
                cached_window = self._device_backend.build_window_indices(
                    req.positions[: req.num_input_tokens],
                    window_size,
                    visible_lengths=replay_visible_lens,
                    indices_output=self._a5_causal_swa_indices[: req.num_input_tokens],
                    lengths_output=self._a5_causal_swa_lengths[: req.num_input_tokens],
                )
                if replay_visible_lens is None:
                    self.common_v41_batch_metadata[cache_key] = cached_window
            req.ori_sparse_indices, req.ori_topk_length = cached_window
        elif replay_visible_lens is not None:
            # Scalar-band backends consume a count from the right edge of the window.
            req.ori_topk_length = replay_visible_lens
        elif req.ori_topk_length is None and req.ori_sparse_indices is not None and noncausal:
            req.ori_topk_length = (req.ori_sparse_indices >= 0).sum(dim=-1, dtype=torch.int32)
        req.ori_mask_mode = 0 if noncausal else 4
        req.ori_win_left = max(0, window_size - 1)
        req.ori_win_right = 0
        return max_ori_seq_len


class AscendDSAV41StateMetadataBuilder(AscendDSAV41MetadataBuilder):
    """State cache: build C2 controls only for the cache-update owner."""

    def build_req_metadata(
        self, common, num_actual_reqs: int, *, build_query: bool = True, update_cache: bool = True, **kwargs
    ) -> AscendDSAV41Metadata:
        req = self._new_req_metadata(common, num_actual_reqs)
        if update_cache:
            self._build_compressor_cache(common, req, **kwargs)
        return req

    def _build_compressor_cache(self, common, req: AscendDSAV41Metadata, **kwargs) -> None:
        """Publish C2 ring updates and source RoPE metadata for the state cache."""
        req.c2_ring_metadata = None
        req.c2_complete_mask = None
        req.c2_source_positions = None
        req.c2_source_cos = None
        req.c2_source_sin = None
        req.c2_metadata_group_id = None
        if (
            self._build_compressor_metadata
            and kwargs.get("build_compressor_metadata", True)
            and req.cache_kind == "compressor_state"
            and req.positions is not None
        ):
            ring_meta = self._c2_ring_metadata[: 5 * req.num_reqs].view(5, req.num_reqs)
            input_positions = req.positions
            if self._supports_device_ops:
                assert (
                    self._c2_full_source_rope is not None
                ), "Enable device metadata before building compressor metadata"
                full_source_cos, full_source_sin = self._c2_full_source_rope
            else:
                full_source_cos = full_source_sin = None
            skip_ring_update = bool(kwargs.get("skip_ring_state_update", False))

            def build_c2_metadata() -> None:
                if self._uses_a5_packed_cache and full_source_cos is not None and full_source_sin is not None:
                    build_c2_ring_metadata(
                        common.query_start_loc,
                        req.seq_lens,
                        input_positions,
                        common.block_table_tensor,
                        full_source_cos,
                        full_source_sin,
                        req.num_reqs,
                        req.num_input_tokens,
                        req.num_actual_reqs,
                        req.num_actual_tokens,
                        skip_update=skip_ring_update,
                        ring_metadata_output=ring_meta,
                        complete_mask_output=self._c2_complete_mask[: req.num_input_tokens],
                        source_positions_output=self._c2_source_positions[: req.num_input_tokens],
                        cos_output=self._c2_source_cos[: req.num_input_tokens],
                        sin_output=self._c2_source_sin[: req.num_input_tokens],
                    )
                    return
                starts = common.query_start_loc[: req.num_reqs].int()
                ends = common.query_start_loc[1 : req.num_reqs + 1].int()
                query_lens = ends - starts
                live = torch.arange(req.num_reqs, device=starts.device) < req.num_actual_reqs
                used = (ends.clamp_max(req.num_actual_tokens) - starts).clamp_min(0)
                used = torch.where(live, used, 0)
                if skip_ring_update:
                    used = torch.zeros_like(used)
                ring_meta[0].copy_((req.seq_lens - query_lens).clamp_min(0))
                ring_meta[1].copy_(used)
                ring_meta[2].copy_(starts)
                ring_meta[3].copy_(starts)
                ring_meta[4].copy_(torch.where(used > 0, common.block_table_tensor[: req.num_reqs, 0], 0))
                valid_end = common.query_start_loc[req.num_actual_reqs].clamp_max(req.num_actual_tokens)
                valid = torch.arange(req.num_input_tokens, device=input_positions.device) < valid_end
                complete = (input_positions.remainder(2) == 1) & valid
                if skip_ring_update:
                    complete = torch.zeros_like(complete)
                self._c2_complete_mask[: req.num_input_tokens].copy_(complete)
                self._c2_source_positions[: req.num_input_tokens].copy_(
                    torch.where(
                        complete,
                        input_positions - 1,
                        torch.zeros_like(input_positions),
                    )
                )
                if full_source_cos is not None and full_source_sin is not None:
                    gather_idx = (
                        self._c2_source_positions[: req.num_input_tokens]
                        .reshape(-1, 1, 1, 1)
                        .expand(
                            req.num_input_tokens,
                            1,
                            1,
                            full_source_cos.shape[-1],
                        )
                    )
                    torch.gather(
                        full_source_cos,
                        0,
                        gather_idx,
                        out=self._c2_source_cos[: req.num_input_tokens],
                    )
                    torch.gather(
                        full_source_sin,
                        0,
                        gather_idx,
                        out=self._c2_source_sin[: req.num_input_tokens],
                    )

            self._publish_task(
                self.common_v41_metadata,
                "c2:compressor",
                self._c2_complete_mask,
                DeviceMetadataStage.COMPRESSOR,
                build_c2_metadata,
            )
            req.c2_complete_mask = self._c2_complete_mask[: req.num_input_tokens]
            req.c2_ring_metadata = ring_meta
            req.c2_source_positions = self._c2_source_positions[: req.num_input_tokens]
            if self._supports_device_ops:
                req.c2_source_cos = self._c2_source_cos[: req.num_input_tokens]
                req.c2_source_sin = self._c2_source_sin[: req.num_input_tokens]
            req.c2_metadata_group_id = None if self._device_metadata_in_graph else id(self._c2_complete_mask)


class AscendDSAV41AttentionMetadataBuilder(AscendDSAV41MetadataBuilder):
    """Compressed attention: SMLA queries and compressed cache writes.

    SWA reuses attention queries; Indexer reuses compressed cache addressing.
    Each concrete builder assembles only its consumer's request metadata.
    """

    def build_req_metadata(
        self, common, num_actual_reqs: int, *, build_query: bool = True, update_cache: bool = True, **kwargs
    ) -> AscendDSAV41Metadata:
        req = self._new_req_metadata(common, num_actual_reqs)
        self._prepare_cache_lengths(common, req, **kwargs)
        if self._build_query_metadata and build_query:
            self._build_attention_query(common, req, **kwargs)
        if update_cache:
            self._build_compressed_cache(common, req, **kwargs)
        return req

    def _build_attention_query(self, common, req: AscendDSAV41Metadata, **kwargs) -> None:
        max_ori_seq_len = self._prepare_query_inputs(common, req, **kwargs)
        self._build_smla_metadata(common, req, max_ori_seq_len=max_ori_seq_len, **kwargs)

    def _build_compressed_cache(self, common, req: AscendDSAV41Metadata, **kwargs) -> None:
        self._build_cache_slots(common, req, **kwargs)

    def _build_smla_metadata(
        self,
        common,
        req: AscendDSAV41Metadata,
        max_ori_seq_len: int,
        **kwargs,
    ) -> None:
        text_config = self.vllm_config.model_config.hf_text_config
        n_local_heads = int(
            kwargs.get(
                "num_query_heads",
                int(_config_value(text_config, "num_attention_heads"))
                // self.vllm_config.parallel_config.tensor_parallel_size,
            )
        )
        head_dim = int(_config_value(text_config, "head_dim"))
        index_topk = int(_config_value(text_config, "index_topk"))
        operator_ratio = 0 if req.cache_kind == "swa" else req.compress_ratio
        metadata_tokens = req.num_input_tokens if self._device_metadata_in_graph else req.num_actual_tokens
        if (
            self._uses_a5_packed_cache
            and req.cache_kind == "swa"
            and self._a5_smla_metadata is not None
            and self._a5_smla_length_rows is not None
        ):
            # One plan per distinct visibility contract, in first-use order.
            # Produce on DeviceMetadataExecutor's stream, never inside qsmla.
            # Index identities are score-dependent; their cardinality is not
            # for dense causal TopK (or a candidate pool with enough coverage).
            ratios = kwargs.get("smla_plan_ratios")
            if ratios is None:
                ratios = tuple(dict.fromkeys((0, *(_config_value(text_config, "compress_ratios", ()) or ()))))
            noncausal = not bool(getattr(common, "causal", True))
            if noncausal:
                ratios = (0,)  # DSpark uses noncausal SWA, no compressed KV.
            if any(r not in (0, 1, 2) for r in ratios):
                raise ValueError("A5 shared MLA plans support compression ratios 0/1/2")
            candidate_blocks = int(_config_value(text_config, "candidate_topk_blocks", 0))
            candidate_width = int(_config_value(text_config, "candidate_block_size", 0))
            if candidate_blocks and candidate_blocks * candidate_width - (candidate_width - 1) < index_topk:
                raise ValueError("A5 shared MLA planning requires candidate coverage >= index_topk")
            req.a5_smla_plans = {}
            # Noncausal/replay masks may differ even at equal Q shapes.
            mask_key = (
                id(req.ori_topk_length)
                if noncausal
                or (getattr(common, "replay_start", None) is not None and not self._uses_a5_packed_cache)
                or kwargs.get("replay_start") is not None
                else 0
            )
            visible = None
            needs_visible_lengths = True
            for plan_ratio in ratios:
                plan_key = f"smla:a5:w{self._window_size}:mask{mask_key}:c{plan_ratio}"
                shared_plan = self.common_v41_smla_metadata.get(plan_key)
                if shared_plan is not None:
                    req.a5_smla_plans[plan_ratio] = shared_plan
                    continue
                if plan_ratio not in self._a5_smla_plans:
                    self._a5_smla_plans[plan_ratio] = (
                        self._a5_smla_metadata if plan_ratio == 0 else torch.empty_like(self._a5_smla_metadata)
                    )
                    self._a5_smla_plan_lengths[plan_ratio] = (
                        torch.empty_like(self._a5_smla_length_rows),
                        torch.empty_like(self._a5_smla_length_rows),
                    )
                plan_buffer = self._a5_smla_plans[plan_ratio]
                plan_ori, plan_cmp = self._a5_smla_plan_lengths[plan_ratio]
                refresh_lengths = plan_ratio != 0 and needs_visible_lengths
                if refresh_lengths:
                    needs_visible_lengths = False

                def build_a5_smla_metadata(
                    plan_ratio=plan_ratio,
                    plan_buffer=plan_buffer,
                    plan_ori=plan_ori,
                    plan_cmp=plan_cmp,
                    refresh_lengths=refresh_lengths,
                ) -> None:
                    nonlocal visible
                    if metadata_tokens == 0 or req.num_reqs == 0:
                        plan_buffer.zero_()
                        return
                    positions = req.positions
                    ori = plan_ori[:metadata_tokens]
                    cmp = plan_cmp[:metadata_tokens]
                    if req.ori_topk_length is not None:
                        lengths = req.ori_topk_length[:metadata_tokens]
                        ori[: lengths.shape[0]].copy_(lengths.reshape(-1, 1))
                        # Full draft graphs can contain rows beyond the live mask.
                        if lengths.shape[0] < metadata_tokens:
                            ori[lengths.shape[0] :].zero_()
                    else:
                        assert positions is not None
                        ori.copy_((positions[:metadata_tokens] + 1).clamp(0, self._window_size).reshape(-1, 1))
                    if plan_ratio:
                        # Graph padding can carry position=0 with an empty KV
                        # sequence. Bound visibility by the request length too.
                        # The first compressed task recomputes lengths on every
                        # execution, including capture after eager warmup.
                        if refresh_lengths:
                            assert positions is not None
                            rows = torch.arange(metadata_tokens, device=positions.device)
                            ends = common.query_start_loc[1 : req.num_reqs + 1]
                            owners = (rows[:, None] >= ends[None, :]).sum(-1).clamp_max(req.num_reqs - 1)
                            visible = torch.minimum(positions[:metadata_tokens] + 1, req.seq_lens[owners]).clamp_min(0)
                            visible = torch.where(rows < ends[-1], visible, 0)
                        assert visible is not None
                        cmp.copy_(
                            torch.div(visible, plan_ratio, rounding_mode="floor").clamp(0, index_topk).reshape(-1, 1)
                        )
                    else:
                        cmp.zero_()
                    assert self._device_backend is not None
                    value = self._device_backend.mixed_quant_sparse_flash_mla_metadata(
                        ori,
                        cmp,
                        cu_seqlens_q=common.query_start_loc[: req.num_reqs + 1].int().contiguous(),
                        num_heads_q=64,
                        num_heads_kv=1,
                        head_dim=512,
                        quant_mode=1,
                        layout_q="TND",
                        layout_kv="PA_BBND",
                        has_ori_kv=True,
                        has_cmp_kv=bool(plan_ratio),
                    )
                    plan_buffer.copy_(value)

                req.a5_smla_plans[plan_ratio] = self._publish_task(
                    self.common_v41_smla_metadata,
                    plan_key,
                    plan_buffer,
                    DeviceMetadataStage.ATTENTION,
                    build_a5_smla_metadata,
                )
            req.smla_metadata = req.a5_smla_plans.get(0)

        if self._supports_device_ops and not self._uses_a5_packed_cache and req.cache_kind in {"swa", "long_kv"}:
            has_compressed = operator_ratio in (1, 2)
            cmp_seq_lens = req.cache_seq_lens if has_compressed else None
            cmp_residual = req.cmp_residual

            def build_smla_metadata() -> None:
                # Keep graph event frontiers stable even when this CP rank has no query.
                if req.num_actual_tokens == 0:
                    self._smla_metadata.zero_()
                    return
                value = torch.ops._C_ascend.npu_sparse_flash_mla_metadata(
                    n_local_heads,
                    1,
                    head_dim,
                    cu_seqlens_q=common.query_start_loc[: req.num_reqs + 1].int(),
                    seqused_ori_kv=req.replay_seq_lens if req.replay_seq_lens is not None else req.seq_lens,
                    seqused_cmp_kv=cmp_seq_lens,
                    cmp_residual_kv=cmp_residual,
                    batch_size=req.num_reqs,
                    max_seqlen_q=int(getattr(common, "max_query_len", 0)),
                    max_seqlen_ori_kv=max_ori_seq_len,
                    max_seqlen_cmp_kv=(req.max_cache_seq_len if has_compressed else 0),
                    ori_topk=req.ori_sparse_indices.shape[-1] if req.ori_sparse_indices is not None else 0,
                    ori_topk_length=req.ori_topk_length,
                    cmp_topk=index_topk if has_compressed else 0,
                    cmp_ratio=operator_ratio,
                    ori_mask_mode=req.ori_mask_mode,
                    cmp_mask_mode=3 if has_compressed else 0,
                    ori_win_left=req.ori_win_left,
                    ori_win_right=req.ori_win_right,
                    layout_q="TND",
                    layout_kv="PA_BBND",
                    has_ori_kv=True,
                    has_cmp_kv=has_compressed,
                )
                self._smla_metadata.copy_(value)

            req.smla_metadata = self._publish_task(
                self.common_v41_smla_metadata,
                f"smla:c{operator_ratio}",
                self._smla_metadata,
                DeviceMetadataStage.ATTENTION,
                build_smla_metadata,
            )


class AscendDSAV41SWAMetadataBuilder(AscendDSAV41AttentionMetadataBuilder):
    """SWA: local window queries and original-token cache writes."""

    def build_req_metadata(
        self, common, num_actual_reqs: int, *, build_query: bool = True, update_cache: bool = True, **kwargs
    ) -> AscendDSAV41Metadata:
        req = self._new_req_metadata(common, num_actual_reqs)
        build_query = self._build_query_metadata and bool(build_query)
        update_cache = bool(update_cache)
        if build_query or update_cache:
            self._prepare_rope(common, req, **kwargs)
        if build_query:
            self._build_attention_query(common, req, **kwargs)
        if update_cache:
            self._build_swa_cache(common, req, **kwargs)
        return req

    def _build_swa_cache(self, common, req: AscendDSAV41Metadata, **kwargs) -> None:
        self._build_cache_slots(common, req, **kwargs)


class AscendDSAV41IndexerMetadataBuilder(AscendDSAV41AttentionMetadataBuilder):
    """Indexer: QLI queries and compressed index-cache writes, without SMLA."""

    def build_req_metadata(
        self, common, num_actual_reqs: int, *, build_query: bool = True, update_cache: bool = True, **kwargs
    ) -> AscendDSAV41Metadata:
        req = self._new_req_metadata(common, num_actual_reqs)
        self._prepare_cache_lengths(common, req, **kwargs)
        if self._build_query_metadata and build_query:
            self._build_indexer_query(common, req, **kwargs)
        if update_cache:
            self._build_compressed_cache(common, req, **kwargs)
        return req

    def _build_indexer_query(self, common, req: AscendDSAV41Metadata, **kwargs) -> None:
        self._prepare_query_inputs(common, req, **kwargs)
        self._build_qli_metadata(common, req, **kwargs)

    def _build_qli_metadata(self, common, req: AscendDSAV41Metadata, **kwargs) -> None:
        text_config = self.vllm_config.model_config.hf_text_config
        index_topk = int(_config_value(text_config, "index_topk"))
        if self._supports_device_ops and req.cache_kind == "index_k":
            residual = req.cmp_residual

            if self._uses_a5_packed_cache:
                candidate_source = int(_config_value(text_config, "candidate_source_layer_id", -1))
                ratios = _config_value(text_config, "compress_ratios")
                emits_candidates = (
                    ratios is not None
                    and 0 <= candidate_source < len(ratios)
                    and int(ratios[candidate_source]) == req.compress_ratio
                )
                candidate_blocks = int(_config_value(text_config, "candidate_topk_blocks")) if emits_candidates else -1
                candidate_block_size = (
                    int(_config_value(text_config, "candidate_block_size")) if emits_candidates else -1
                )
                # Deferred tasks run after the build context restores the builder's mode.
                metadata_in_graph = self._device_metadata_in_graph

                def build_qli_metadata() -> None:
                    if req.num_actual_tokens == 0 and not metadata_in_graph:
                        self._qli_metadata.zero_()
                        return
                    assert self._device_backend is not None
                    value = self._device_backend.quant_lightning_indexer_metadata(
                        cu_seqlens_q=common.query_start_loc[: req.num_reqs + 1].int(),
                        seqused_k=req.cache_seq_lens,
                        cmp_residual_k=residual,
                        batch_size=req.num_reqs,
                        max_seqlen_q=-1,
                        max_seqlen_k=-1,
                        num_heads_q=int(_config_value(text_config, "index_n_heads")),
                        num_heads_k=1,
                        head_dim=int(_config_value(text_config, "index_head_dim")),
                        topk=index_topk,
                        mask_mode=3,
                        cmp_ratio=req.compress_ratio,
                        layout_q="TND",
                        layout_k="PA_BBND",
                        candidate_topk_blocks=candidate_blocks,
                        candidate_block_size=candidate_block_size,
                    )
                    self._qli_metadata.copy_(value)
            else:

                def build_qli_metadata() -> None:
                    value = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
                        int(_config_value(text_config, "index_n_heads")),
                        1,
                        int(_config_value(text_config, "index_head_dim")),
                        index_topk,
                        2,
                        cu_seqlens_q=common.query_start_loc[: req.num_reqs + 1].int(),
                        seqused_k=req.cache_seq_lens,
                        cmp_residual_k=residual,
                        batch_size=req.num_reqs,
                        max_seqlen_q=int(getattr(common, "max_query_len", 0)),
                        max_seqlen_k=req.max_cache_seq_len,
                        layout_q="TND",
                        layout_k="PA_BBND",
                        mask_mode=3,
                        cmp_ratio=req.compress_ratio,
                    )
                    self._qli_metadata.copy_(value)

            req.qli_metadata = self._publish_task(
                self.common_v41_batch_metadata,
                f"qli:c{req.compress_ratio}",
                self._qli_metadata,
                DeviceMetadataStage.INDEXER,
                build_qli_metadata,
            )


class AscendDSAV41FoldedIndexMetadataBuilder(AscendDSAV41MetadataBuilder):
    """The packed indexer accesses this resource without query or write metadata."""

    def build_req_metadata(
        self, common, num_actual_reqs: int, *, build_query: bool = True, update_cache: bool = True, **kwargs
    ) -> AscendDSAV41Metadata:
        req = self._new_req_metadata(common, num_actual_reqs)
        return req


def _get_v41_role_builder(role: str):
    from vllm_ascend.attention.context_parallel import dsa_v41_cp

    builder_cls, _ = dsa_v41_cp.get_v41_cp_classes()

    if builder_cls is AscendDSAV41MetadataBuilder:
        return {
            "State": AscendDSAV41StateMetadataBuilder,
            "SWA": AscendDSAV41SWAMetadataBuilder,
            "Attention": AscendDSAV41AttentionMetadataBuilder,
            "Indexer": AscendDSAV41IndexerMetadataBuilder,
            "FoldedIndex": AscendDSAV41FoldedIndexMetadataBuilder,
        }[role]

    return {
        "State": dsa_v41_cp.AscendDSAV41PCPStateMetadataBuilder,
        "SWA": dsa_v41_cp.AscendDSAV41PCPSWAMetadataBuilder,
        "Attention": dsa_v41_cp.AscendDSAV41PCPAttentionMetadataBuilder,
        "Indexer": dsa_v41_cp.AscendDSAV41PCPIndexerMetadataBuilder,
        "FoldedIndex": dsa_v41_cp.AscendDSAV41PCPFoldedIndexMetadataBuilder,
    }[role]


class DeepseekV41CacheBackend(AttentionBackend):
    """Cache-only backend: supplies layout and metadata, not an AttentionImpl."""

    @staticmethod
    def get_name():
        return "ASCEND_DSA_V41_CACHE"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        # vLLM #60271 added the batch-invariance support check in
        # `init_attn_backend` in both supported vLLM revisions.
        return True

    @staticmethod
    def get_impl_cls():
        from vllm_ascend.attention.context_parallel.dsa_v41_cp import get_v41_cp_classes

        return get_v41_cp_classes()[1]

    @staticmethod
    def get_builder_cls():
        from vllm_ascend.attention.context_parallel.dsa_v41_cp import get_v41_cp_classes

        return get_v41_cp_classes()[0]

    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
        return num_blocks, block_size, num_kv_heads, head_size

    @classmethod
    def get_runtime_shapes(
        cls,
        spec: AttentionSpec,
        num_blocks: int,
        layout: Any,
        kernel_block_size: int | None = None,
    ) -> KVCacheRuntimeShapes:
        """Expose NHD data/scale pages, including packed A5 byte widths."""
        return AscendDSABackend.get_runtime_shapes(spec, num_blocks, layout, kernel_block_size)


class _DeepseekV41RoleBackend(DeepseekV41CacheBackend):
    metadata_role = ""

    @classmethod
    def get_name(cls):
        return f"ASCEND_DSA_V41_{cls.metadata_role.upper()}"

    @classmethod
    def get_builder_cls(cls):
        return _get_v41_role_builder(cls.metadata_role)


class DeepseekV41StateBackend(_DeepseekV41RoleBackend):
    metadata_role = "State"


class DeepseekV41SWABackend(_DeepseekV41RoleBackend):
    metadata_role = "SWA"


class DeepseekV41AttentionBackend(_DeepseekV41RoleBackend):
    metadata_role = "Attention"


class DeepseekV41IndexerBackend(_DeepseekV41RoleBackend):
    metadata_role = "Indexer"


class DeepseekV41FoldedIndexBackend(_DeepseekV41RoleBackend):
    metadata_role = "FoldedIndex"


class DeepseekV41CacheLayer(nn.Module, AttentionLayerBase):
    supports_dcp = False

    def __init__(self, vllm_config, prefix, spec):
        super().__init__()
        self.prefix = prefix
        self.spec = spec
        self.kv_cache: torch.Tensor | tuple[torch.Tensor, ...] = torch.empty(0)
        context = vllm_config.compilation_config.static_forward_context
        context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
        """Bind a single cache view or a tuple of component views."""
        self.kv_cache = kv_cache

    def get_kv_cache_spec(self, vllm_config):
        return self.spec

    def get_attn_backend(self):
        spec = self.spec

        if isinstance(spec, CircularBufferSpec):
            return DeepseekV41StateBackend

        if isinstance(spec, AscendSlidingWindowMLASpec):
            return DeepseekV41SWABackend

        if isinstance(spec, AscendMLAAttentionSpec):
            if spec.tokens_per_state == 8 and spec.head_size == 544:
                return DeepseekV41FoldedIndexBackend
            if spec.scale_dim:
                return DeepseekV41IndexerBackend
            return DeepseekV41AttentionBackend
        raise TypeError(f"Unsupported V4.1 cache spec: {type(spec).__name__}")
