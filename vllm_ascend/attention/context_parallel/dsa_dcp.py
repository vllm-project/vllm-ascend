# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode Context Parallelism (DCP) for the DeepSeek-V4 DSA attention stack.

Architecture
------------
Under DCP the scheduler shards *full-attention* KV cache groups along the
sequence dimension (interleaved layout, ``cp_kv_cache_interleave_size`` raw
tokens per run) while *sliding-window / state* groups keep replicated
per-rank state (upstream ``dcp_world_size_for_kv_cache_spec``).  For a
DeepSeek-V4 layer this means:

======================  =====================  =============================
cache                   spec type              DCP behaviour
======================  =====================  =============================
SWA raw-token KV        SlidingWindowMLASpec   replicated (unchanged)
C4 / C128 compressed KV MLAAttentionSpec       **sharded** (this module)
C4 indexer K cache      MLAAttentionSpec       **sharded** (this module)
compressor state        SlidingWindowMLASpec   replicated (unchanged)
======================  =====================  =============================

Because the scheduler (spec) block of a compressed group spans
``storage_block_size * compress_ratio`` raw tokens and DCP scales it by
``dcp_size``, every rank owns exactly ``storage_block_size`` compressed
tokens per virtual scheduler block -- i.e. exactly one physical compressed
block per virtual block.  The rank-local block table therefore has one
column per virtual block and the local compressed sequence is the rank's
share of each request, in global order.

Execution model
---------------
* **Pure-decode batches** (the DCP hot path): the query is all-gathered on
  the head dim, the sparse attention op runs against the *local* compressed
  shard with DCP-local compressed lengths, and partial ``(out, LSE)`` pairs
  are merged with an exact exp-space all-reduce.  Because the SWA part is
  replicated (every rank attends the full raw window), the fused per-rank
  outputs each contain that term; a SWA-only pass recovers it exactly once
  so the merge can subtract the ``dcp_size - 1`` duplicates (same contract
  as MLA/SFA DCP, plus the replicated-SWA correction).
* **Prefill-bearing batches** (chunked prefill): causal masking cannot be
  expressed in local coordinates with per-request cumsums, so the compressed
  KV of the whole batch is gathered into a transient workspace laid out in
  *global* compressed order (identity block table) and the batch runs with
  global semantics.  SWA / state / indexer-selection inputs are replicated
  or index-based and need no gather.
* **Indexer top-k**: each rank selects top-k over its local index-K shard
  (the selection op returns scores with ``return_value=1``); candidates are
  all-gathered and merged into the exact global top-k by score.  Decode then
  remaps the global indices to local compressed coordinates; prefill uses
  them directly against the workspace.
* **Compressed-KV writes**: the compressor slot mapping is derived by the
  ``compressor_metadata`` operator from a block table.  Under DCP the builder
  feeds the operator an *identity* table (``compressor_block_table``) so the
  emitted slots are global compressed positions; the impl remaps them to
  rank-local slots (padding foreign-owned groups) before the scatter.

Constraints (validated in :mod:`vllm_ascend.ascend_config`)
-----------------------------------------------------------
* ``tp_size % dcp_size == 0`` (upstream).
* ``cp_kv_cache_interleave_size`` must be a multiple of every compress ratio
  (4 and 128) so compression groups never split ranks, and
  ``block_size % cp_kv_cache_interleave_size == 0`` (upstream).  In practice
  this pins ``--block-size 128`` with interleave 128 for DeepSeek-V4.
* DCP is mutually exclusive with the legacy ``enable_dsa_cp`` token-sharded
  path and with PCP.
"""

import os

import torch
import torch.distributed as dist
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

from vllm_ascend.attention.context_parallel.common_cp import (
    DCPImplMixin,
    DCPMetadataBuilderMixin,
)
from vllm_ascend.attention.dsa_attn_kv_plan import get_dsa_attn_kv_plan
from vllm_ascend.attention.dsa_v1 import (
    DSA_METADATA_BUFFER_SIZE,
    AscendDSAImpl,
    AscendDSAMetadata,
    AscendDSAMetadataBuilder,
    AscendDSAReqMetadata,
    DeviceOperator,
    _dsa_layout_kv,
    _dsa_swa_only_cmp_ratio,
    _require_req_metadata,
    attention_transfer_window,
    notify_kv_cache_written,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
    wait_for_device_metadata,
)

DSA_DCP_SUPPORTED_COMPRESS_RATIOS = (4, 128)
DSA_DCP_PAD_SLOT = -1

# Temporary debug instrumentation (eager mode only): set DSA_DCP_DEBUG=1 to
# dump a bounded number of intermediate-value snapshots per rank.
_DSA_DCP_DEBUG = os.environ.get("DSA_DCP_DEBUG") == "1"
_DSA_DCP_DEBUG_BUDGET = {"prefill": 4, "decode": 24, "probe": 12}


def _dsa_dcp_debug_on() -> bool:
    """Whether DCP debug printing is safe right now.

    Returns False while an ACL graph capture is in progress: every debug site
    reads device values (``.item()`` / ``.tolist()``), which forces a host
    sync -- at best it errors during capture, at worst it bakes the
    capture-time values into the graph and corrupts every replay.
    """
    if not _DSA_DCP_DEBUG:
        return False
    try:
        if torch.npu.is_current_stream_capturing():
            return False
    except (AttributeError, RuntimeError):
        return True
    return True


def _dsa_dcp_debug(kind: str, msg: str) -> None:
    if _dsa_dcp_debug_on() and _DSA_DCP_DEBUG_BUDGET.get(kind, 0) > 0:
        _DSA_DCP_DEBUG_BUDGET[kind] -= 1
        print(f"[dsa-dcp-debug {kind}]{msg}", flush=True)
# python -c "
# import vllm_ascend.patch.platform.patch_kv_cache_coordinator as m
# import vllm.v1.core.kv_cache_coordinator as c
# print(m.get_kv_cache_coordinator is c.get_kv_cache_coordinator)
# print(c.get_kv_cache_coordinator.__module__)
# "

# One-shot KV-usage logging: keyed by (rank, group type, request batch
# fingerprint) so each request logs its storage split / cache sizes exactly
# once, on the first compressed (C4/C128) layer that touches it.
_KV_USAGE_LOGGED_KEYS: set[tuple] = set()


def dsa_dcp_group_is_sharded(kv_cache_spec) -> bool:
    """Only MLA-spec (full-attention) groups are DCP-sharded.

    SWA and compressor-state groups use sliding-window specs and keep their
    replicated per-rank state under DCP, so their builders behave exactly
    like the base (non-DCP) builders.
    """
    return isinstance(kv_cache_spec, AscendMLAAttentionSpec)


def _compressed_stripe_len(interleave_size: int, compress_ratio: int) -> int:
    stripe = interleave_size // compress_ratio
    if interleave_size % compress_ratio != 0 or stripe <= 0:
        raise ValueError(
            "DSA DCP requires cp_kv_cache_interleave_size "
            f"({interleave_size}) to be a multiple of the compress ratio "
            f"({compress_ratio})."
        )
    return stripe


def dsa_dcp_global_cmp_to_local(
    global_positions: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
    compress_ratio: int,
    storage_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map per-request global compressed positions to (owner, local pos).

    Args:
        global_positions: integer tensor of global compressed positions
            (position within the request's compressed sequence).
    Returns:
        owner: owner rank per position (same shape as the input).
        local_positions: rank-local compressed positions (valid only where
            ``owner == dcp_rank``).
    """
    stripe = _compressed_stripe_len(interleave_size, compress_ratio)
    cmp_per_vblk = storage_block_size * dcp_size
    pos = global_positions.to(torch.int64)
    vblk = torch.div(pos, cmp_per_vblk, rounding_mode="floor")
    within = pos - vblk * cmp_per_vblk
    stripe_idx = torch.div(within, stripe, rounding_mode="floor")
    owner = torch.remainder(stripe_idx, dcp_size)
    local_within = torch.div(stripe_idx, dcp_size, rounding_mode="floor") * stripe + torch.remainder(within, stripe)
    local_positions = vblk * storage_block_size + local_within
    return owner, local_positions


def dsa_dcp_local_cmp_to_global(
    local_positions: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
    compress_ratio: int,
    storage_block_size: int,
) -> torch.Tensor:
    """Inverse of :func:`dsa_dcp_global_cmp_to_local` for this rank's shard."""
    stripe = _compressed_stripe_len(interleave_size, compress_ratio)
    cmp_per_vblk = storage_block_size * dcp_size
    lp = local_positions.to(torch.int64)
    vblk = torch.div(lp, storage_block_size, rounding_mode="floor")
    within = lp - vblk * storage_block_size
    local_stripe = torch.div(within, stripe, rounding_mode="floor")
    in_stripe = torch.remainder(within, stripe)
    global_stripe = local_stripe * dcp_size + dcp_rank
    return vblk * cmp_per_vblk + global_stripe * stripe + in_stripe


def get_dsa_dcp_local_compressed_seq_lens(
    seq_lens_cpu: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Per-request compressed token count stored on this DCP rank.

    A compression group is owned by exactly one rank (interleave is a
    multiple of the ratio), so the local compressed length is the local raw
    length of the group-aligned prefix divided by the ratio.
    """
    _compressed_stripe_len(interleave_size, compress_ratio)
    group_aligned = torch.div(seq_lens_cpu.to(torch.int64), compress_ratio, rounding_mode="floor") * compress_ratio
    local_raw = get_dcp_local_seq_lens(
        group_aligned,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=interleave_size,
    ).to(torch.int64)
    return torch.div(local_raw, compress_ratio, rounding_mode="floor").to(torch.int32)


def get_dsa_dcp_local_cmp_residual(
    seq_lens_cpu: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
    compress_ratio: int,
    storage_block_size: int,
) -> torch.Tensor:
    """Per-request trailing partial group owned by this rank (raw tokens)."""
    residual = torch.remainder(seq_lens_cpu.to(torch.int64), compress_ratio)
    last_pos = (seq_lens_cpu.to(torch.int64) - 1).clamp(min=0)
    owner, _ = dsa_dcp_global_cmp_to_local(
        torch.div(last_pos, compress_ratio, rounding_mode="floor"),
        dcp_size,
        dcp_rank,
        interleave_size,
        compress_ratio,
        storage_block_size,
    )
    owned = (owner == dcp_rank) & (seq_lens_cpu.to(torch.int64) > 0)
    return torch.where(owned, residual, torch.zeros_like(residual)).to(torch.int32)


def remap_dsa_dcp_topk_indices(
    topk_indices: torch.Tensor,
    *,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
    compress_ratio: int,
    storage_block_size: int,
    global_cmp_seq_lens: torch.Tensor,
    req_indices: torch.Tensor,
) -> torch.Tensor:
    """Remap global compressed top-k indices to local compressed coordinates.

    The merged indices are request-relative *global* compressed positions, so
    the padding-slot guard must bound them by the *global* compressed length
    (a local bound would drop real owned positions on every rank but the
    first).  Foreign-owned entries are compacted to the tail and replaced with
    ``DSA_DCP_PAD_SLOT`` so the attention kernel skips them; the per-rank LSE
    merge treats an empty local selection as a neutral (-inf) contribution.
    """
    if topk_indices.numel() == 0:
        return topk_indices
    topk = topk_indices.shape[-1]
    flat = topk_indices.reshape(-1, topk).to(torch.int64)
    reqs = req_indices.reshape(-1, 1).expand_as(flat).clamp(min=0)
    valid_seq = (flat >= 0) & (flat < global_cmp_seq_lens.to(torch.int64)[reqs])
    owner, local = dsa_dcp_global_cmp_to_local(
        flat, dcp_size, dcp_rank, interleave_size, compress_ratio, storage_block_size
    )
    owned = valid_seq & (owner == dcp_rank)
    local_idx = torch.where(owned, local, torch.full_like(local, DSA_DCP_PAD_SLOT))
    # Compact owned entries to the front, preserving top-k order.
    order = torch.arange(topk, dtype=torch.int64, device=flat.device)
    pack_keys = order.unsqueeze(0) + (~owned).to(torch.int64) * topk
    _, pack_order = torch.sort(pack_keys, dim=-1, stable=True)
    remapped = torch.gather(local_idx, dim=-1, index=pack_order)
    return remapped.reshape(topk_indices.shape).to(topk_indices.dtype)


def merge_dsa_dcp_indexer_topk(
    local_topk_indices: torch.Tensor,
    local_topk_values: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    dcp_group,
    dcp_rank: int,
    interleave_size: int,
    compress_ratio: int,
    storage_block_size: int,
) -> torch.Tensor:
    """Exact global top-k from per-rank (index, score) candidates.

    Each rank selected its local top-k with scores; both are all-gathered
    along a new rank dimension, the local indices are mapped to global
    compressed coordinates (the mapping is request-relative, so it applies
    uniformly to every query row), and the global top-k is taken by score
    (ties resolved deterministically by index order).

    ``valid_mask`` marks candidates whose local index is inside the request's
    rank-local compressed length.  Ranks owning no part of a short request
    still emit top-k slots (the selection kernel pads them), and without this
    mask their garbage candidates - scored against uninitialized cache - can
    displace the real ones.  Invalid entries are turned into -1 indices with
    -inf scores so they can never win the merge; surviving -1 slots are
    skipped by the attention kernel (same contract as foreign-owned slots).
    """
    topk = local_topk_indices.shape[-1]
    flat_idx = local_topk_indices.reshape(-1, topk).to(torch.int64)
    flat_val = local_topk_values.reshape(-1, topk).to(torch.float32)
    global_idx = dsa_dcp_local_cmp_to_global(
        flat_idx,
        dcp_group.world_size,
        dcp_rank,
        interleave_size,
        compress_ratio,
        storage_block_size,
    )
    sane = (flat_idx >= 0) & valid_mask.reshape(flat_idx.shape).to(torch.bool)
    global_idx = torch.where(sane, global_idx, torch.full_like(global_idx, -1))
    scores = torch.where(sane, flat_val, torch.full_like(flat_val, float("-inf")))
    packed = torch.stack([scores, global_idx.to(torch.float32)], dim=-1).unsqueeze(0)
    gathered = dcp_group.all_gather(packed.contiguous(), dim=0)
    # [dcp, rows, topk, 2] -> [rows, dcp * topk, 2]
    rows = packed.shape[1]
    gathered = gathered.permute(1, 0, 2, 3).reshape(rows, dcp_group.world_size * topk, 2)
    all_scores = gathered[..., 0]
    all_indices = gathered[..., 1].to(torch.int64)
    topk_order = torch.argsort(-all_scores, dim=-1, stable=True)[..., :topk]
    merged = torch.gather(all_indices, dim=-1, index=topk_order)
    selected_scores = torch.gather(all_scores, dim=-1, index=topk_order)
    # Garbage slots never carry a real index into the attention op.
    merged = torch.where(
        torch.isfinite(selected_scores), merged, torch.full_like(merged, -1)
    )
    return merged.reshape(local_topk_indices.shape).to(local_topk_indices.dtype)


class AscendDSADCPContext:
    """Per-(sharded group, batch) DCP state consumed by the impl."""

    def __init__(
        self,
        *,
        dcp_size: int,
        dcp_rank: int,
        interleave_size: int,
        compress_ratio: int,
        storage_block_size: int,
        has_prefill: bool,
        num_reqs: int,
    ) -> None:
        self.dcp_size = dcp_size
        self.dcp_rank = dcp_rank
        self.interleave_size = interleave_size
        self.compress_ratio = compress_ratio
        self.storage_block_size = storage_block_size
        self.has_prefill = has_prefill
        self.num_reqs = num_reqs
        # Per-request local compressed lengths and their exclusive cumsum.
        self.local_cmp_seq_lens: torch.Tensor | None = None
        self.local_cu_cmp_seqlens: torch.Tensor | None = None
        # Per-request trailing partial compression group owned by this rank.
        self.local_cmp_residual: torch.Tensor | None = None
        # Per-request global compressed lengths (prefill workspace path).
        self.global_cmp_seq_lens: torch.Tensor | None = None
        # Identity block table fed to the compressor metadata operator so
        # its emitted slots are global compressed positions per request.
        self.compressor_block_table: torch.Tensor | None = None
        # Request index per compressor output row (device int64).
        self.compress_row_req_indices: torch.Tensor | None = None
        # SWA-only (cmp disabled) sas metadata for the decode merge SWA
        # correction (built by the builder, consumed by the impl).
        self.swa_only_sas_metadata: torch.Tensor | None = None


class AscendDSADCPMetadata(AscendDSAMetadata):
    """DSA metadata carrying the DCP context for sharded groups."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dcp_context: AscendDSADCPContext | None = None


class AscendDSADCPMetadataBuilder(DCPMetadataBuilderMixin, AscendDSAMetadataBuilder):
    """Build DSA metadata for DCP.

    Sharded (MLA-spec) groups get DCP-local compressed lengths, an identity
    compressor block table and DCP-local sas/qli metadata; replicated
    (SWA/state) groups fall through to the base behaviour untouched.
    """

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
        metadata_cls: type[AscendDSAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls or AscendDSADCPMetadata,
            supports_dcp_with_varlen,
        )
        self.sharded_group = dsa_dcp_group_is_sharded(kv_cache_spec)
        if not self.sharded_group:
            return
        parallel_config = vllm_config.parallel_config
        self.cp_interleave_size = parallel_config.cp_kv_cache_interleave_size
        if self.compressor_ratio not in DSA_DCP_SUPPORTED_COMPRESS_RATIOS:
            raise ValueError(
                f"DSA DCP supports compress ratios {DSA_DCP_SUPPORTED_COMPRESS_RATIOS}, got {self.compressor_ratio}."
            )
        _compressed_stripe_len(self.cp_interleave_size, self.compressor_ratio)
        group_span = self.storage_block_size * self.compressor_ratio
        if group_span % self.cp_interleave_size != 0:
            raise ValueError(
                f"DSA DCP requires the group block span ({group_span}) to be a multiple "
                f"of cp_kv_cache_interleave_size ({self.cp_interleave_size})."
            )
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs * self._request_capacity_factor + 1
        self.dcp_local_cmp_buf = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.dcp_local_cu_cmp_buf = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
        self.dcp_local_cmp_residual_buf = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.dcp_global_cmp_buf = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # Identity table indexed by global compressed block column.
        # Materialized with repeat (not expand) so custom ops see a
        # contiguous tensor.
        max_global_cmp_blocks = (
            cdiv(vllm_config.model_config.max_model_len, self.compressor_ratio) // self.storage_block_size + 2
        )
        self.dcp_identity_block_table = (
            torch.arange(max_global_cmp_blocks, dtype=torch.int32, device=device).unsqueeze(0).repeat(max_num_reqs, 1)
        )
        max_compressed_rows = (
            vllm_config.scheduler_config.max_num_batched_tokens // self.compressor_ratio + max_num_reqs
        )
        self.dcp_compress_row_reqs_buf = torch.zeros(max_compressed_rows, dtype=torch.int64, device=device)
        # SWA-only sas metadata for the decode-merge SWA correction: the
        # fused per-rank outputs all contain the identical replicated SWA
        # term, which the plain LSE merge would count dcp_size times.
        self.swa_sas_metadata_buffer: torch.Tensor = torch.zeros(
            DSA_METADATA_BUFFER_SIZE, dtype=torch.int32, device=device
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSADCPMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build, **kwargs)
        if not isinstance(metadata, AscendDSADCPMetadata):
            metadata = AscendDSADCPMetadata(
                num_actual_tokens=metadata.num_actual_tokens,
                head_dim=metadata.head_dim,
                num_decodes=metadata.num_decodes,
                num_decode_tokens=metadata.num_decode_tokens,
                num_prefills=metadata.num_prefills,
                attn_state=metadata.attn_state,
                req_metadata=metadata.req_metadata,
                reshape_cache_event=metadata.reshape_cache_event,
                hadamard=metadata.hadamard,
            )
        if not self.sharded_group:
            return metadata
        metadata.dcp_context = self._build_dcp_context(common_attn_metadata, metadata)
        return metadata

    def _seq_lens_cpu(self, common_attn_metadata) -> torch.Tensor:
        if common_attn_metadata._seq_lens_cpu is not None:
            return common_attn_metadata._seq_lens_cpu
        if common_attn_metadata.seq_lens_cpu is not None:
            return common_attn_metadata.seq_lens_cpu
        return common_attn_metadata.seq_lens.cpu()

    def _build_dcp_context(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        metadata: AscendDSADCPMetadata,
    ) -> AscendDSADCPContext:
        num_reqs = common_attn_metadata.num_reqs
        req_metadata = metadata.req_metadata
        assert req_metadata is not None
        # Keep every DCP-local quantity device-resident: these buffers feed the
        # per-step metadata and slot remaps, so deriving them from CPU tensors
        # would freeze the first step's lengths under ACL graph capture (and
        # costs a host sync every eager step).
        seq_lens = self.seq_lens[:num_reqs]

        context = AscendDSADCPContext(
            dcp_size=self.dcp_size,
            dcp_rank=self.dcp_rank,
            interleave_size=self.cp_interleave_size,
            compress_ratio=self.compressor_ratio,
            storage_block_size=self.storage_block_size,
            has_prefill=self.num_prefills > 0,
            num_reqs=num_reqs,
        )

        local_cmp = get_dsa_dcp_local_compressed_seq_lens(
            seq_lens,
            self.dcp_size,
            self.dcp_rank,
            self.cp_interleave_size,
            self.compressor_ratio,
        )
        self.dcp_local_cmp_buf[:num_reqs].copy_(local_cmp.to(torch.int32))
        torch.cumsum(
            self.dcp_local_cmp_buf[:num_reqs],
            dim=0,
            out=self.dcp_local_cu_cmp_buf[1 : num_reqs + 1],
        )
        self.dcp_local_cu_cmp_buf[0].zero_()
        if num_reqs + 1 < self.dcp_local_cu_cmp_buf.shape[0]:
            self.dcp_local_cu_cmp_buf[num_reqs + 1 :].zero_()
        local_residual = get_dsa_dcp_local_cmp_residual(
            seq_lens,
            self.dcp_size,
            self.dcp_rank,
            self.cp_interleave_size,
            self.compressor_ratio,
            self.storage_block_size,
        )
        self.dcp_local_cmp_residual_buf[:num_reqs].copy_(local_residual.to(torch.int32))
        global_cmp = torch.div(
            seq_lens.to(torch.int64),
            self.compressor_ratio,
            rounding_mode="floor",
        ).to(torch.int32)
        self.dcp_global_cmp_buf[:num_reqs].copy_(global_cmp)

        context.local_cmp_seq_lens = self.dcp_local_cmp_buf[:num_reqs]
        context.local_cu_cmp_seqlens = self.dcp_local_cu_cmp_buf[: num_reqs + 1]
        context.local_cmp_residual = self.dcp_local_cmp_residual_buf[:num_reqs]
        context.global_cmp_seq_lens = self.dcp_global_cmp_buf[:num_reqs]
        context.compressor_block_table = self.dcp_identity_block_table[:num_reqs]
        # The compressor metadata operator must emit *global* compressed
        # positions (identity table); the remap callable installed on the
        # request metadata converts them to this rank's local slots inside
        # Compressor.forward, covering every compressed-cache write path
        # (main C4/C128 KV, indexer K and scale caches).
        req_metadata.compressor_block_table = context.compressor_block_table
        req_metadata.dcp_slot_remap = self._make_slot_remap(context)

        if self.num_prefills == 0:
            self._apply_decode_local_metadata(req_metadata, context, common_attn_metadata)
        else:
            self._apply_prefill_local_qli(req_metadata, context, common_attn_metadata)
        self._build_compress_row_req_indices(req_metadata, common_attn_metadata, context)
        return context

    def _apply_decode_local_metadata(
        self,
        req_metadata: AscendDSAReqMetadata,
        context: AscendDSADCPContext,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> None:
        """Switch the shared sas/qli metadata to DCP-local compressed lens."""
        num_reqs = context.num_reqs
        req_metadata.cu_cmp_seqlen_list = context.local_cu_cmp_seqlens
        # Replace the base async tasks with DCP-local ones.  The runner takes
        # and re-runs these tasks EVERY step, which is what keeps the metadata
        # buffer tracking the growing decode lengths -- a synchronous build()
        # alone is only valid where it runs (eager); under ACL graph capture it
        # would freeze the captured step's metadata for every replay, which
        # shows up as the first decode step being right and every later step
        # drifting.  The COMPRESSOR task keeps running (identity table).
        self._drop_device_metadata_tasks(
            DeviceMetadataStage.ATTENTION,
            DeviceMetadataStage.INDEXER,
        )
        dcp_tasks = (
            DeviceMetadataTask(
                DeviceMetadataStage.ATTENTION,
                lambda: self._build_local_sas_metadata(context, common_attn_metadata),
                id(self.sas_metadata_buffer),
            ),
            DeviceMetadataTask(
                DeviceMetadataStage.ATTENTION,
                lambda: self._build_local_swa_only_metadata(context, common_attn_metadata),
                id(self.swa_sas_metadata_buffer),
            ),
        )
        if self.compressor_ratio == 4:
            dcp_tasks += (
                DeviceMetadataTask(
                    DeviceMetadataStage.INDEXER,
                    lambda: self._refresh_local_qli(context, common_attn_metadata),
                    id(self.qli_metadata_buffer),
                ),
            )
        self._device_metadata_tasks = self._device_metadata_tasks + dcp_tasks
        # Also build once inline so the buffers are valid for the step being
        # constructed (eager) and at the capture step itself.
        self._build_local_sas_metadata(context, common_attn_metadata)
        req_metadata.sas_metadata = self.sas_metadata_buffer
        self._build_local_swa_only_metadata(context, common_attn_metadata)
        context.swa_only_sas_metadata = self.swa_sas_metadata_buffer
        if self.compressor_ratio == 4:
            self._refresh_local_qli(context, common_attn_metadata)
            req_metadata.qli_metadata = self.qli_metadata_buffer
            req_metadata.qli_seqused_k = self.qli_seqused_k[:num_reqs]
            req_metadata.qli_cmp_residual_k = self.qli_cmp_residual_k[:num_reqs]

    def _drop_device_metadata_tasks(self, *stages: DeviceMetadataStage) -> None:
        self._device_metadata_tasks = tuple(task for task in self._device_metadata_tasks if task.stage not in stages)

    def _build_local_sas_metadata(
        self,
        context: AscendDSADCPContext,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> torch.Tensor:
        """Rebuild the attention metadata with local compressed cumsums.

        Mirrors the base decode build exactly (global raw seq lens for the
        replicated SWA path, per-request decode cu_seqlens_ori_kv) and only
        swaps ``cu_seqlens_cmp_kv`` for the DCP-local variant.
        """
        num_reqs = context.num_reqs
        seq_lens = self.seq_lens[:num_reqs]
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        seq_lens_cpu = self._seq_lens_cpu(common_attn_metadata)[:num_reqs]
        max_seqlen_q = int(torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item())
        max_seqlen_kv = int(torch.max(seq_lens_cpu).item())
        cu_seqlens_ori_kv = DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
            self.common_ratio_to_sas_metadata,
            "cu_seqlens_ori_kv",
            seq_lens,
            num_reqs,
            self._zero_i32,
            self.cu_seqlens_ori_kv,
        )
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
        # Decode runs the attention op on the DCP-head-gathered query, and the
        # metadata op derives its GQA group size (and per-core block
        # scheduling) from num_heads_q -- it must describe the gathered head
        # count or only the first rank's head slice gets scheduled.
        n_gathered_heads = n_local_heads * self.dcp_size
        kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
        metadata_op = kv_plan.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = kv_plan.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
        sas_metadata = metadata_op(
            **metadata_kwargs,
            num_heads_q=n_gathered_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=query_start_loc,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=context.local_cu_cmp_seqlens,
            seqused_q=self.seqused_q,
            seqused_kv=seq_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=num_reqs,
            cmp_topk=self.model_config.hf_config.index_topk if self.compressor_ratio == 4 else 0,
            cmp_ratio=self.compressor_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=self.model_config.hf_config.sliding_window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv=_dsa_layout_kv(self.vllm_config),
            has_ori_kv=True,
            has_cmp_kv=True,
        )
        self.sas_metadata_buffer[:DSA_METADATA_BUFFER_SIZE] = sas_metadata[:DSA_METADATA_BUFFER_SIZE]
        return self.sas_metadata_buffer

    def _build_local_swa_only_metadata(
        self,
        context: AscendDSADCPContext,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> torch.Tensor:
        """Rebuild a cmp-disabled sas metadata for the decode merge correction.

        The fused per-rank decode outputs all contain the identical replicated
        SWA term; the plain LSE merge counts it ``dcp_size`` times.  A
        SWA-only pass provides that term exactly once so the impl can subtract
        the ``dcp_size - 1`` duplicates in exp space.  Rebuilt every step for
        the same frozen-metadata reason as :meth:`_build_local_sas_metadata`.
        """
        num_reqs = context.num_reqs
        seq_lens = self.seq_lens[:num_reqs]
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        seq_lens_cpu = self._seq_lens_cpu(common_attn_metadata)[:num_reqs]
        max_seqlen_q = int(torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item())
        max_seqlen_kv = int(torch.max(seq_lens_cpu).item())
        cu_seqlens_ori_kv = DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
            self.common_ratio_to_sas_metadata,
            "cu_seqlens_ori_kv",
            seq_lens,
            num_reqs,
            self._zero_i32,
            self.cu_seqlens_ori_kv,
        )
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
        # Decode runs the attention op on the DCP-head-gathered query, and the
        # metadata op derives its GQA group size (and per-core block
        # scheduling) from num_heads_q -- it must describe the gathered head
        # count or only the first rank's head slice gets scheduled.
        n_gathered_heads = n_local_heads * self.dcp_size
        kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
        metadata_op = kv_plan.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = kv_plan.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
        swa_metadata = metadata_op(
            **metadata_kwargs,
            num_heads_q=n_gathered_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=query_start_loc,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=None,
            seqused_q=self.seqused_q,
            seqused_kv=seq_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=num_reqs,
            cmp_topk=0,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=self.model_config.hf_config.sliding_window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv=_dsa_layout_kv(self.vllm_config),
            has_ori_kv=True,
            has_cmp_kv=False,
        )
        self.swa_sas_metadata_buffer[:DSA_METADATA_BUFFER_SIZE] = swa_metadata[:DSA_METADATA_BUFFER_SIZE]
        return self.swa_sas_metadata_buffer

    def _refresh_local_qli(
        self,
        context: AscendDSADCPContext,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> None:
        """Refresh the indexer metadata/buffers with local compressed lens.

        Always recomputes this builder's own ``qli_seqused_k`` /
        ``qli_cmp_residual_k`` buffers and re-runs the metadata operator every
        step: its output carries the current ``cu_seqlens_q`` / max lengths
        and the growing decode ``seqused_k``, so caching the first call's
        tensor would freeze the indexer at prefill-time lengths and corrupt
        every subsequent top-k selection.
        """
        num_reqs = context.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        max_seqlen_q = int(torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item())
        self.qli_seqused_k[:num_reqs].copy_(context.local_cmp_seq_lens)
        self.qli_cmp_residual_k[:num_reqs].copy_(self.dcp_local_cmp_residual_buf[:num_reqs])
        local_cmp = context.local_cmp_seq_lens
        max_local_cmp = int(local_cmp.max().item()) if num_reqs else 1
        # Drop the base async INDEXER task: it would rebuild with global
        # compressed lengths after this synchronous local refresh.
        self._drop_device_metadata_tasks(DeviceMetadataStage.INDEXER)
        qli_metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            num_heads_q=self.model_config.hf_config.index_n_heads,
            num_heads_k=1,
            head_dim=self.model_config.hf_config.index_head_dim,
            topk=self.model_config.hf_config.index_topk,
            quant_mode=DeviceOperator.get_dsa_indexer_quant_mode(),
            cu_seqlens_q=query_start_loc,
            seqused_k=self.qli_seqused_k[:num_reqs],
            cmp_residual_k=self.qli_cmp_residual_k[:num_reqs],
            batch_size=num_reqs,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_local_cmp,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=4,
            device=str(self.seqused_q.device),
        )
        self.qli_metadata_buffer[:DSA_METADATA_BUFFER_SIZE] = qli_metadata[:DSA_METADATA_BUFFER_SIZE]

    def _apply_prefill_local_qli(
        self,
        req_metadata: AscendDSAReqMetadata,
        context: AscendDSADCPContext,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> None:
        """Prefill keeps global attention semantics (workspace path) but the
        indexer still selects over the local shard, so qli goes local."""
        if self.compressor_ratio != 4:
            return
        self._refresh_local_qli(context, common_attn_metadata)
        req_metadata.qli_metadata = self.qli_metadata_buffer
        req_metadata.qli_seqused_k = self.qli_seqused_k[: context.num_reqs]
        req_metadata.qli_cmp_residual_k = self.qli_cmp_residual_k[: context.num_reqs]

    def _build_compress_row_req_indices(
        self,
        req_metadata: AscendDSAReqMetadata,
        common_attn_metadata: AscendCommonAttentionMetadata,
        context: AscendDSADCPContext,
    ) -> None:
        """Request index per compressor output row (global group order).

        The compressor emits one row per compression group completed by this
        step's queries, ordered by request, so the slot remap can address the
        correct request's block-table row.

        Computed entirely on device with a STATIC output shape:
        ``repeat_interleave`` over a data-dependent count would produce a
        data-dependent output shape, which forces a host sync per step and
        freezes the mapping under ACL graph capture.  The equivalent fixed
        length form is "row j belongs to the request whose cumulative group
        count first exceeds j".
        """
        num_reqs = context.num_reqs
        seq_lens = self.seq_lens[:num_reqs].to(torch.int64)
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1].to(torch.int64)
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        start_pos = (seq_lens - query_lens).clamp(min=0)
        ratio = self.compressor_ratio
        completed = (
            torch.div(start_pos + query_lens, ratio, rounding_mode="floor")
            - torch.div(start_pos, ratio, rounding_mode="floor")
        ).clamp(min=0)
        cumulative = torch.cumsum(completed, dim=0)
        buf = self.dcp_compress_row_reqs_buf
        rows = torch.arange(buf.shape[0], dtype=torch.int64, device=buf.device)
        row_reqs = torch.searchsorted(cumulative, rows, right=True)
        last_req = max(num_reqs - 1, 0)
        valid = rows < cumulative[-1]
        buf.copy_(torch.where(valid, row_reqs.clamp(max=last_req), torch.full_like(row_reqs, last_req)))
        context.compress_row_req_indices = buf

    def _make_slot_remap(self, context: AscendDSADCPContext):
        """Build the global->local compress slot remap for this group.

        The returned closure is installed on the request metadata and runs
        inside ``Compressor.forward``.  It converts the identity-table slots
        (global compressed positions) into rank-local physical slots using
        this group's local block table; foreign-owned groups are padded so
        the scatter kernels skip them.
        """

        def remap(slot_mapping: torch.Tensor) -> torch.Tensor:
            if slot_mapping is None or slot_mapping.numel() == 0:
                return slot_mapping
            storage = context.storage_block_size
            if slot_mapping.ndim == 2:
                blocks = slot_mapping[..., 0].to(torch.int64)
                offsets = slot_mapping[..., 1].to(torch.int64)
                global_pos = blocks * storage + offsets
            else:
                global_pos = slot_mapping.to(torch.int64).reshape(-1)
            row_reqs = context.compress_row_req_indices
            if row_reqs is None or row_reqs.numel() == 0:
                return slot_mapping
            row_reqs = row_reqs[: global_pos.shape[0]]
            owner, local_pos = dsa_dcp_global_cmp_to_local(
                global_pos,
                context.dcp_size,
                context.dcp_rank,
                context.interleave_size,
                context.compress_ratio,
                storage,
            )
            owned = (owner == context.dcp_rank) & (global_pos >= 0)
            local_blocks = torch.div(local_pos, storage, rounding_mode="floor")
            local_offsets = torch.remainder(local_pos, storage)
            # self.block_table holds this group's rank-local table for the
            # current step (set by the base build()).
            table = self.block_table.to(torch.int64)
            num_rows, num_cols = table.shape
            flat_index = row_reqs.clamp(max=num_rows - 1) * num_cols + local_blocks.clamp(max=num_cols - 1)
            block_ids = table.reshape(-1)[flat_index]
            local_slots = block_ids * storage + local_offsets
            local_slots = torch.where(owned, local_slots, torch.full_like(local_slots, DSA_DCP_PAD_SLOT))
            if slot_mapping.ndim == 2:
                remapped = torch.stack(
                    [
                        torch.div(local_slots, storage, rounding_mode="floor"),
                        torch.remainder(local_slots, storage),
                    ],
                    dim=-1,
                )
                return remapped.to(slot_mapping.dtype)
            return local_slots.reshape(slot_mapping.shape).to(slot_mapping.dtype)

        return remap


class AscendDSADCPImpl(DCPImplMixin, AscendDSAImpl):
    """DSA attention impl with DCP decode and gathered-prefill execution."""

    # Upstream check_attention_cp_compatibility requires decode LSE support.
    # AttentionImplBase.__new__ recomputes need_to_return_lse_for_decode as
    # ``dcp_world_size > 1 and can_return_lse_for_decode`` (the instance
    # attribute shadows the class attribute), so the capability flag is what
    # actually matters.
    can_return_lse_for_decode = True
    need_to_return_lse_for_decode = True

    def _get_dcp_context(self, layer_metadata) -> AscendDSADCPContext | None:
        metadata = layer_metadata.attention
        if metadata is None:
            metadata = layer_metadata.swa
        if not isinstance(metadata, AscendDSADCPMetadata):
            return None
        return metadata.dcp_context

    def _query_row_req_indices(
        self,
        query_start_loc: torch.Tensor,
        num_reqs: int,
        num_rows: int,
    ) -> torch.Tensor:
        """Request index per query row (graph-padding rows clamp to the last)."""
        qsl = query_start_loc[: num_reqs + 1].to(torch.int64).contiguous()
        rows = torch.arange(num_rows, dtype=torch.int64, device=qsl.device)
        req_ids = torch.searchsorted(qsl[1:], rows, right=True)
        return req_ids.clamp(max=max(num_reqs - 1, 0))

    def _merge_indexer_topk(
        self,
        topk_indices: torch.Tensor,
        topk_values: torch.Tensor,
        context: AscendDSADCPContext,
        num_rows: int,
        common_metadata,
    ) -> torch.Tensor:
        # The local selection emits a full top-k per row even when this rank
        # owns fewer (or zero) compressed groups for the request; candidates
        # beyond the rank-local compressed length read uninitialized cache and
        # must not enter the global merge.
        req_metadata = _require_req_metadata(common_metadata)
        # The indexer emits [T, 1, K] (singleton k-head dim from the
        # return_value=0 op output); flatten to [T, K] so the per-row mask
        # broadcasts row-wise instead of exploding over the singleton head
        # dim, and restore the original shape afterwards -- the attention op
        # consumes the same layout the base (non-DCP) path feeds it.
        out_shape = topk_indices[:num_rows].shape
        idx2d = topk_indices[:num_rows].reshape(num_rows, -1)
        val2d = topk_values[:num_rows].reshape(num_rows, -1)
        req_ids = self._query_row_req_indices(req_metadata.query_start_loc, context.num_reqs, num_rows)
        local_lens = context.local_cmp_seq_lens.to(torch.int64)[req_ids].unsqueeze(-1)
        valid_mask = idx2d < local_lens
        merged = merge_dsa_dcp_indexer_topk(
            idx2d,
            val2d,
            valid_mask=valid_mask,
            dcp_group=self.dcp_group,
            dcp_rank=self.dcp_rank,
            interleave_size=context.interleave_size,
            compress_ratio=context.compress_ratio,
            storage_block_size=context.storage_block_size,
        )
        # The f-string below reads device values, so the guard must wrap the
        # call itself -- gating inside _dsa_dcp_debug is too late (the
        # arguments, including .item()/.tolist(), are evaluated first).
        if num_rows > 0 and _dsa_dcp_debug_on():
            _dsa_dcp_debug(
                "prefill",
                f" rank={self.dcp_rank} rows={num_rows} reqs={context.num_reqs}"
                f" local_lens={context.local_cmp_seq_lens[: context.num_reqs].tolist()}"
                f" global_lens={context.global_cmp_seq_lens[: context.num_reqs].tolist()}"
                f" valid_last={int((merged[-1] != DSA_DCP_PAD_SLOT).sum())}"
                f" last_idx={merged[-1, :8].tolist()}"
                f" raw_last_idx={idx2d[-1, :8].tolist()}"
                f" raw_last_val={[f'{v:.3f}' for v in val2d[-1, :4].tolist()]}"
                f" bt_cols={req_metadata.block_table.shape[-1]}"
                f" bt_rows={req_metadata.block_table.shape[0]}",
            )
        return merged.reshape(out_shape)

    def _maybe_update_compressed_caches_and_select_topk(
        self,
        layer_name,
        hidden_states,
        qr,
        kv_cache,
        layer_metadata,
        qr_pertoken_scale,
        compress_kv_cache,
        state_cache,
        compressor_overlap_output=None,
        write_cache=True,
    ):
        """DCP variant: local indexer selection with scores, global merge.

        The indexer module is asked for top-k *values* alongside indices
        (``return_value=1``); candidates are merged into the exact global
        top-k across the DCP group.  The returned indices stay in *global*
        compressed coordinates; the decode path remaps them later, prefill
        consumes them against the workspace.
        """
        context = self._get_dcp_context(layer_metadata)
        if context is None or self.compress_ratio != 4 or getattr(self.indexer, "skip_topk", False):
            if context is not None and getattr(self.indexer, "skip_topk", False):
                raise NotImplementedError("DSA DCP does not support the cached-topk (skip_topk) path yet.")
            return super()._maybe_update_compressed_caches_and_select_topk(
                layer_name,
                hidden_states,
                qr,
                kv_cache,
                layer_metadata,
                qr_pertoken_scale,
                compress_kv_cache,
                state_cache,
                compressor_overlap_output,
                write_cache,
            )
        assert self.indexer is not None
        assert self.compressor is not None
        assert layer_metadata.compressor is not None
        common_metadata = layer_metadata.attention or layer_metadata.swa

        from vllm_ascend.models.deepseek_v4.indexer import IndexerOverlapPlan

        def compute_attention_compressed_kv():
            if compressor_overlap_output is not None:
                overlap_result, compressor_done = compressor_overlap_output
                torch.npu.current_stream().wait_event(compressor_done)
                return overlap_result
            return self.compressor(
                hidden_states=hidden_states,
                state_cache=state_cache,
                metadata=layer_metadata.compressor,
            )

        def scatter_attention_compressed_kv(compressed_kv, compress_slot_mapping):
            # Slots already arrive rank-local: Compressor.forward applied the
            # DCP remap installed by the metadata builder.
            if compressed_kv.shape[0] > 0:
                get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                    compress_kv_cache,
                    compressed_kv,
                    compress_slot_mapping,
                )

        overlap_plan = IndexerOverlapPlan(
            compute_attention_compressed_kv=compute_attention_compressed_kv,
            scatter_attention_compressed_kv=scatter_attention_compressed_kv,
            aux_stream=None,
        )
        local_topk_indices, local_topk_values = self.indexer(
            hidden_states=hidden_states,
            qr=qr,
            kv_cache=kv_cache,
            metadata=layer_metadata.indexer,
            overlap_plan=overlap_plan,
            layer_name=layer_name,
            qr_pertoken_scale=qr_pertoken_scale,
            write_cache=write_cache,
            return_topk_values=True,
        )
        # Keep EVERY row the indexer produced: `cmp_sparse_indices` must have
        # the same first dimension as the query the attention op sees.  Under
        # ACL graph capture the batch is padded to the capture size, so
        # `num_actual_tokens` is smaller than the query row count and clamping
        # to it would hand the op a short index tensor.  Padded rows fall back
        # to the last real request and their candidates resolve to -1, which
        # the op skips.
        num_rows = local_topk_indices.shape[0]
        return self._merge_indexer_topk(
            local_topk_indices,
            local_topk_values,
            context,
            num_rows,
            common_metadata,
        )

    def _forward_attention(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        layer_metadata,
        cache_is_prepared: bool = False,
    ) -> torch.Tensor:
        context = self._get_dcp_context(layer_metadata)
        if context is None or self.dcp_size == 1:
            return super()._forward_attention(layer_name, hidden_states, kv_cache, layer_metadata, cache_is_prepared)
        if cache_is_prepared:
            raise RuntimeError("DSA DCP does not support prepared caches (PCP).")
        if context.has_prefill:
            return self._forward_attention_prefill_gather(layer_name, hidden_states, kv_cache, layer_metadata, context)
        return self._forward_attention_decode_dcp(layer_name, hidden_states, kv_cache, layer_metadata, context)

    def _unpack_kv_caches(self, kv_cache):
        return DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)

    def _log_dsa_dcp_kv_usage(
        self,
        context: AscendDSADCPContext,
        compress_kv_cache: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        state_cache: torch.Tensor,
        indexer_k_cache: torch.Tensor | None = None,
        indexer_scale_cache: torch.Tensor | None = None,
    ) -> None:
        """Log per-request KV storage split + per-attention-type cache bytes.

        Prints once per (rank, group type, request batch):

        * ``req_global_cmp`` / ``req_local_cmp``: the request's total
          compressed token count vs how many are stored ON THIS card.  For a
          DCP-sharded group (C4/C128) the per-rank ``req_local_cmp`` values
          split the request across cards (each card holds only its stripe,
          and non-owner cards hold 0) -- proof of "each card keeps only its
          own KV, nothing is replicated".
        * Per-attention-type cache sizes: the bytes a single card allocates
          for each cache group.  The sharded groups scale down with dcp_size,
          so the sum across cards exceeds any one card's share.
        """
        g_cmp = context.global_cmp_seq_lens
        l_cmp = context.local_cmp_seq_lens
        key = (self.dcp_rank, int(self.compress_ratio), tuple(int(x) for x in g_cmp.tolist()))
        if key in _KV_USAGE_LOGGED_KEYS:
            return
        _KV_USAGE_LOGGED_KEYS.add(key)

        def _mb(t: torch.Tensor) -> str:
            nbytes = t.numel() * t.element_size()
            return f"{nbytes / 1024 / 1024:.1f}MiB {tuple(t.shape)}"

        lines = [
            f"[dsa-dcp-kv] rank={self.dcp_rank} group_ratio={self.compress_ratio}",
            f"  requests: n_req={g_cmp.shape[0]}",
            f"    req_global_cmp={g_cmp.tolist()}   (request total compressed tokens)",
            f"    req_local_cmp ={l_cmp.tolist()}   (tokens stored ON THIS card)",
            f"  per-card cache sizes:",
            f"    compress_kv_cache = {_mb(compress_kv_cache)}",
            f"    swa_kv_cache      = {_mb(swa_kv_cache)}",
            f"    state_cache       = {_mb(state_cache)}",
        ]
        if indexer_k_cache is not None:
            lines.append(f"    indexer_k_cache   = {_mb(indexer_k_cache)}")
        if indexer_scale_cache is not None:
            lines.append(f"    indexer_scale     = {_mb(indexer_scale_cache)}")
        print("\n".join(lines), flush=True)

    def _forward_attention_decode_dcp(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        layer_metadata,
        context: AscendDSADCPContext,
    ) -> torch.Tensor:
        """DCP decode: Q head-gather, local-shard attention, LSE merge."""
        common_attn_metadata = layer_metadata.attention or layer_metadata.swa
        swa_req_metadata = _require_req_metadata(layer_metadata.swa)
        common_metadata = _require_req_metadata(common_attn_metadata)
        num_tokens = hidden_states.shape[0]
        cos = common_metadata.cos[layer_name][:num_tokens]
        sin = common_metadata.sin[layer_name][:num_tokens]

        compress_kv_cache, swa_kv_cache, state_cache, _, _, _ = self._unpack_kv_caches(kv_cache)

        if _dsa_dcp_debug_on() and self.compress_ratio > 1 and num_tokens == 1:
            self._log_dsa_dcp_kv_usage(
                context,
                compress_kv_cache,
                swa_kv_cache,
                state_cache,
            )
        if _dsa_dcp_debug_on() and self.compress_ratio == 4 and num_tokens == 1:
            cid = int(common_metadata.block_table[0, 0].item())
            _dsa_dcp_debug(
                "probe",
                f" rank={self.dcp_rank} c4_blk={cid}"
                f" row0={[f'{v:.4f}' for v in compress_kv_cache[cid, 0, 0, :4].float().tolist()]}"
                f" absmax={compress_kv_cache[cid].abs().max().item():.4f}",
            )

        q, qr, qr_pertoken_scale = self._mla_prolog_single_stream(
            hidden_states,
            cos,
            sin,
            swa_kv_cache,
            swa_req_metadata.slot_mapping,
            write_swa_cache=True,
        )
        compress_topk_idxs = None
        if self.compress_ratio > 1:
            compress_topk_idxs = self._maybe_update_compressed_caches_and_select_topk(
                layer_name=layer_name,
                hidden_states=hidden_states,
                qr=qr,
                kv_cache=kv_cache,
                layer_metadata=layer_metadata,
                qr_pertoken_scale=qr_pertoken_scale,
                compress_kv_cache=compress_kv_cache,
                state_cache=state_cache,
                write_cache=True,
            )
            # C128 layers return None here (no indexer top-k); only the C4
            # indexer selection produces indices that need the local remap.
            if compress_topk_idxs is not None:
                # Remap the merged global top-k to this rank's local coordinates.
                req_ids = self._query_row_req_indices(common_metadata.query_start_loc, context.num_reqs, num_tokens)
                compress_topk_idxs = remap_dsa_dcp_topk_indices(
                    compress_topk_idxs,
                    dcp_size=self.dcp_size,
                    dcp_rank=self.dcp_rank,
                    interleave_size=context.interleave_size,
                    compress_ratio=context.compress_ratio,
                    storage_block_size=context.storage_block_size,
                    global_cmp_seq_lens=context.global_cmp_seq_lens,
                    req_indices=req_ids,
                )
                if _dsa_dcp_debug_on():
                    _dsa_dcp_debug(
                        "decode",
                        f" rank={self.dcp_rank} remap: tokens={num_tokens}"
                        f" local_lens={context.local_cmp_seq_lens[: context.num_reqs].tolist()}"
                        f" row0_idx={compress_topk_idxs.reshape(num_tokens, -1)[0, :24].tolist()}",
                    )

        notify_kv_cache_written(layer_name)
        wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(common_metadata.sas_metadata))

        # Gather Q on the head dim: [T, H, D] -> [T, H * dcp, D].
        q_local = q
        q = self._dcp_all_gather(q.contiguous(), dim=1)
        if _dsa_dcp_debug_on() and self.compress_ratio == 4:
            h0 = self.dcp_rank * self.n_local_heads
            _dsa_dcp_debug(
                "probe",
                f" rank={self.dcp_rank} head_slice_ok="
                f"{bool(torch.allclose(q[:, h0 : h0 + self.n_local_heads], q_local, atol=1e-3))}",
            )

        kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
        attn_op = kv_plan.get_dsa_sparse_attn_op()
        attn_kwargs = kv_plan.get_dsa_sparse_attn_base_kwargs()
        attn_kwargs.update(
            ori_kv=swa_kv_cache,
            ori_block_table=swa_req_metadata.block_table,
            cu_seqlens_q=common_metadata.query_start_loc,
            seqused_kv=common_metadata.seq_lens,
            sinks=self._dcp_gathered_sinks(),
            metadata=common_metadata.sas_metadata,
            softmax_scale=self.softmax_scale,
            cmp_ratio=_dsa_swa_only_cmp_ratio(self.compress_ratio, self.vllm_config),
            ori_mask_mode=4,
            ori_win_left=self.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv=_dsa_layout_kv(self.vllm_config),
            return_softmax_lse=True,
        )
        if self.compress_ratio > 1:
            kv_plan.add_dsa_sparse_attn_extra_kwargs(attn_kwargs, cu_seqlens_cmp_kv=context.local_cu_cmp_seqlens)
            # Compressed-length metadata is carried by the (DCP-local) sas
            # metadata buffer for the quantized sharedkv op; the SparseFlashMla
            # BF16 adapter derives them from seqused_kv via setdefault and
            # would need explicit overrides on that path only.
            attn_kwargs.update(
                cmp_kv=compress_kv_cache,
                cmp_block_table=common_metadata.block_table,
                cmp_mask_mode=3,
            )
            if self.compress_ratio == 4:
                assert compress_topk_idxs is not None
                attn_kwargs["cmp_sparse_indices"] = compress_topk_idxs
        if swa_req_metadata.dspark_swa_indices is not None:
            attn_kwargs["ori_sparse_indices"] = swa_req_metadata.dspark_swa_indices

        with attention_transfer_window():
            attn_output, softmax_lse = attn_op(q, **attn_kwargs)
        # SWA-only pass: every fused per-rank output contains the identical
        # replicated SWA term; the plain LSE merge would count it dcp_size
        # times.  Re-run the op with cmp disabled to recover that term exactly
        # once and subtract the duplicates in exp space.
        swa_kwargs = kv_plan.get_dsa_sparse_attn_base_kwargs()
        swa_kwargs.update(
            ori_kv=swa_kv_cache,
            ori_block_table=swa_req_metadata.block_table,
            cu_seqlens_q=common_metadata.query_start_loc,
            seqused_kv=common_metadata.seq_lens,
            sinks=self._dcp_gathered_sinks(),
            metadata=context.swa_only_sas_metadata,
            softmax_scale=self.softmax_scale,
            cmp_ratio=1,
            ori_mask_mode=4,
            ori_win_left=self.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv=_dsa_layout_kv(self.vllm_config),
            return_softmax_lse=True,
        )
        # The SWA-only pass consumes the second (merge-only) metadata buffer.
        # Both buffers are refreshed by ATTENTION-stage tasks and are keyed by
        # buffer identity, so this pass needs its own wait.
        if context.swa_only_sas_metadata is not None:
            wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(context.swa_only_sas_metadata))
        with attention_transfer_window():
            swa_output, swa_lse = attn_op(q, **swa_kwargs)
        return self._merge_dsa_dcp_decode_output(
            attn_output,
            softmax_lse,
            swa_output,
            swa_lse,
            q.shape[0],
            q.shape[1],
        )

    def _merge_dsa_dcp_decode_output(
        self,
        attn_output: torch.Tensor,
        softmax_lse: torch.Tensor,
        swa_output: torch.Tensor,
        swa_lse: torch.Tensor,
        num_tokens: int,
        num_gathered_heads: int,
    ) -> torch.Tensor:
        """Exact DCP merge for DSA decode with a replicated SWA part.

        Per rank r the fused op computes softmax over (full SWA ∪ cmp shard
        r), i.e. out_r = (S_swa + S_r) / (Z_swa + Z_r) with
        lse_r = log(Z_swa + Z_r).  All ranks share the same gathered query and
        replicated SWA cache, so an all-reduce of the exp-space sums gives
        Σ_r (S_swa + S_r) and Σ_r (Z_swa + Z_r); subtracting the (dcp_size-1)
        duplicated SWA terms yields the true softmax over SWA ∪ cmp_all.

        The merged output keeps the gathered head layout, so each rank finally
        slices its own TP-local heads: ``all_gather`` concatenated the head
        dim in group-rank order, so this rank's heads sit at the slice
        ``[dcp_rank * H_local, (dcp_rank + 1) * H_local)`` -- exactly the
        heads its o_proj weights cover (the same redistribution the shared
        all-to-all path performs, done as a post-merge slice here).
        """
        dcp = self.dcp_size
        if dcp == 1:
            return attn_output
        # Normalize TND outputs to [T, H_gathered, D] and LSEs to
        # [T, H_gathered, 1] (defensive: flattened [T*H, ...] layouts reshape
        # back to the same tensors).
        head_dim = attn_output.shape[-1]
        attn_output = attn_output.reshape(num_tokens, num_gathered_heads, head_dim)
        softmax_lse = softmax_lse.reshape(num_tokens, num_gathered_heads, -1)
        swa_output = swa_output.reshape(num_tokens, num_gathered_heads, head_dim)
        swa_lse = swa_lse.reshape(num_tokens, num_gathered_heads, -1)
        out = attn_output.to(torch.float32)
        lse = softmax_lse.to(torch.float32)
        z = torch.exp(lse)
        sums_out = z * out
        dist.all_reduce(sums_out, group=self.dcp_device_group)
        dist.all_reduce(z, group=self.dcp_device_group)
        swa_z = torch.exp(swa_lse.to(torch.float32))
        dup = dcp - 1
        num = sums_out - dup * swa_z * swa_output.to(torch.float32)
        den = z - dup * swa_z
        tiny = torch.finfo(torch.float32).tiny
        merged = torch.where(den > 0, num / den.clamp_min(tiny), torch.zeros_like(den))
        if _dsa_dcp_debug_on():
            _dsa_dcp_debug(
                "decode",
                f" rank={self.dcp_rank} merge: lse[{lse.min().item():.2f},{lse.max().item():.2f}]"
                f" swa_lse[{swa_lse.min().item():.2f},{swa_lse.max().item():.2f}]"
                f" den[{den.min().item():.3f},{den.max().item():.3f}]"
                f" nonpos={int((den <= 0).sum())}"
                f" out_absmax={out.abs().max().item():.3f}"
                f" swa_absmax={swa_output.to(torch.float32).abs().max().item():.3f}",
            )
        h0 = self.dcp_rank * self.n_local_heads
        own = merged[:, h0 : h0 + self.n_local_heads, :]
        return own.to(attn_output.dtype)

    def _forward_attention_prefill_gather(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        layer_metadata,
        context: AscendDSADCPContext,
    ) -> torch.Tensor:
        """Prefill: gather compressed KV into a global-order workspace."""
        common_attn_metadata = layer_metadata.attention or layer_metadata.swa
        swa_req_metadata = _require_req_metadata(layer_metadata.swa)
        common_metadata = _require_req_metadata(common_attn_metadata)
        num_tokens = hidden_states.shape[0]
        cos = common_metadata.cos[layer_name][:num_tokens]
        sin = common_metadata.sin[layer_name][:num_tokens]

        compress_kv_cache, swa_kv_cache, state_cache, _, _, _ = self._unpack_kv_caches(kv_cache)

        q, qr, qr_pertoken_scale = self._mla_prolog_single_stream(
            hidden_states,
            cos,
            sin,
            swa_kv_cache,
            swa_req_metadata.slot_mapping,
            write_swa_cache=True,
        )
        compress_topk_idxs = None
        if self.compress_ratio > 1:
            # Global top-k (no local remap): indices address the workspace.
            compress_topk_idxs = self._maybe_update_compressed_caches_and_select_topk(
                layer_name=layer_name,
                hidden_states=hidden_states,
                qr=qr,
                kv_cache=kv_cache,
                layer_metadata=layer_metadata,
                qr_pertoken_scale=qr_pertoken_scale,
                compress_kv_cache=compress_kv_cache,
                state_cache=state_cache,
                write_cache=True,
            )

        notify_kv_cache_written(layer_name)
        wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(common_metadata.sas_metadata))

        kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
        attn_op = kv_plan.get_dsa_sparse_attn_op()
        attn_kwargs = kv_plan.get_dsa_sparse_attn_base_kwargs()
        attn_kwargs.update(
            ori_kv=swa_kv_cache,
            ori_block_table=swa_req_metadata.block_table,
            cu_seqlens_q=common_metadata.query_start_loc,
            seqused_kv=common_metadata.seq_lens,
            sinks=self.attn_sink,
            metadata=common_metadata.sas_metadata,
            softmax_scale=self.softmax_scale,
            cmp_ratio=_dsa_swa_only_cmp_ratio(self.compress_ratio, self.vllm_config),
            ori_mask_mode=4,
            ori_win_left=self.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv=_dsa_layout_kv(self.vllm_config),
        )
        kv_plan.add_dsa_sparse_attn_extra_kwargs(attn_kwargs, cu_seqlens_ori_kv=common_metadata.query_start_loc)
        if self.compress_ratio > 1:
            cmp_kv, cmp_block_table = self._gather_prefill_cmp_workspace(
                compress_kv_cache,
                context,
                common_metadata.block_table,
            )
            attn_kwargs.update(
                cmp_kv=cmp_kv,
                cmp_block_table=cmp_block_table,
                cmp_mask_mode=3,
            )
            if self.compress_ratio == 4:
                assert compress_topk_idxs is not None
                attn_kwargs["cmp_sparse_indices"] = compress_topk_idxs
        if swa_req_metadata.vision_swa_indices is not None:
            attn_kwargs["ori_sparse_indices"] = swa_req_metadata.vision_swa_indices

        with attention_transfer_window():
            return attn_op(q, **attn_kwargs)[0]

    def _gather_prefill_cmp_workspace(
        self,
        compress_kv_cache: torch.Tensor,
        context: AscendDSADCPContext,
        local_block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-gather the compressed KV and lay it out in global order.

        Each rank contributes the compacted set of local blocks referenced
        by this batch (uniformly padded so a single all-gather works), plus
        its local block table.  Every global compressed position is then
        copied from the owning rank's segment into a workspace block via a
        token-granularity indexed copy, yielding a cache the attention op
        can read with global semantics through an identity block table.
        """
        storage = context.storage_block_size
        num_reqs = context.num_reqs
        device = compress_kv_cache.device
        global_cmp = context.global_cmp_seq_lens.to(torch.int64)
        blocks_per_req = cdiv(global_cmp, storage).clamp(min=0)
        total_ws_blocks = int(blocks_per_req.sum().item())
        if total_ws_blocks == 0:
            return compress_kv_cache, local_block_table[:num_reqs]

        table = local_block_table.to(torch.int64)
        num_cols = table.shape[1]
        # Compact this rank's referenced local blocks (sorted unique ids).
        valid_block_ids, _ = table[:num_reqs].flatten().unique(return_inverse=True)
        num_valid = int(valid_block_ids.shape[0])
        # Uniform segment size across ranks for the collective gathers.
        size_tensor = torch.tensor([num_valid], dtype=torch.int64, device=device)
        if self.dcp_device_group is not None:
            dist.all_reduce(size_tensor, op=dist.ReduceOp.MAX, group=self.dcp_device_group)
        max_valid = max(int(size_tensor.item()), 1)
        num_valid_tensor = torch.tensor([num_valid], dtype=torch.int64, device=device)
        if self.dcp_device_group is not None:
            gathered_counts = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(self.dcp_size)]
            dist.all_gather(gathered_counts, num_valid_tensor, group=self.dcp_device_group)
            counts_per_rank = torch.stack(gathered_counts).reshape(-1)
        else:
            counts_per_rank = num_valid_tensor.view(1)

        # Segment 1: per-rank sorted local block ids, tail-padded with a
        # large sentinel so every row stays sorted for the batched
        # searchsorted below (a -1 tail would break sortedness).
        pad_id = torch.iinfo(torch.int64).max
        local_ids = torch.full((max_valid,), pad_id, dtype=torch.int64, device=device)
        local_ids[:num_valid] = valid_block_ids
        if self.dcp_size > 1:
            gathered_ids = self.dcp_group.all_gather(local_ids, dim=0)
        else:
            gathered_ids = local_ids.unsqueeze(0)
        gathered_ids = gathered_ids.reshape(self.dcp_size, max_valid)

        # Segment 2: compacted KV blocks, zero-padded.
        block_shape = compress_kv_cache.shape[1:]
        local_blocks = compress_kv_cache.new_zeros((max_valid, *block_shape))
        local_blocks[:num_valid] = compress_kv_cache[valid_block_ids.clamp(max=compress_kv_cache.shape[0] - 1)]
        if self.dcp_size > 1:
            gathered_pool = self.dcp_group.all_gather(local_blocks, dim=0)
        else:
            gathered_pool = local_blocks.unsqueeze(0)
        gathered_pool = gathered_pool.reshape(self.dcp_size * max_valid, *block_shape)

        # Segment 3: per-rank local block tables (same shape on every rank).
        if self.dcp_size > 1:
            gathered_tables = self.dcp_group.all_gather(table[:num_reqs].contiguous(), dim=0)
        else:
            gathered_tables = table[:num_reqs].unsqueeze(0)
        gathered_tables = gathered_tables.reshape(self.dcp_size, num_reqs, num_cols)

        # Enumerate every (request, global compressed position) pair.
        # `pos` is expanded to [num_reqs, max_cmp] up front: ownership and the
        # local position depend only on the position, but every downstream
        # tensor (src_blocks via req_row, in_range, ws_block, src/dst_token)
        # is [num_reqs, max_cmp].  Leaving pos at [1, max_cmp] and relying on
        # implicit broadcasting makes the searchsorted below see boundaries
        # [1, max_cmp, max_valid] against values [num_reqs, max_cmp, 1], which
        # only happens to match when num_reqs == 1 (single-request batches).
        max_cmp = int(global_cmp.max().item())
        req_row = torch.arange(num_reqs, dtype=torch.int64, device=device).unsqueeze(1)
        pos = torch.arange(max_cmp, dtype=torch.int64, device=device).unsqueeze(0).expand(num_reqs, -1)
        owner, local_pos = dsa_dcp_global_cmp_to_local(
            pos,
            context.dcp_size,
            context.dcp_rank,
            context.interleave_size,
            context.compress_ratio,
            storage,
        )
        in_range = pos < global_cmp.unsqueeze(1)
        local_vblk = torch.div(local_pos, storage, rounding_mode="floor")
        local_off = torch.remainder(local_pos, storage)
        vblk_clamped = local_vblk.clamp(max=num_cols - 1)
        # Source block id on the owning rank.
        src_blocks = gathered_tables[
            owner.clamp(0, self.dcp_size - 1),
            req_row,
            vblk_clamped,
        ]
        # Compact index of the source block inside the owner's segment.
        # Batched searchsorted requires a value dim matching the N-D
        # boundaries (per-(req, pos) sorted owner block lists).
        owner_rows = gathered_ids[owner.clamp(0, self.dcp_size - 1)]
        compact_idx = torch.searchsorted(owner_rows, src_blocks.unsqueeze(-1).to(torch.int64)).squeeze(-1)
        compact_valid = compact_idx < counts_per_rank[owner.clamp(0, self.dcp_size - 1)]
        compact_idx = compact_idx.clamp(max=max_valid - 1)
        # Destination: global-order workspace blocks per request.
        req_block_base = torch.zeros(num_reqs + 1, dtype=torch.int64, device=device)
        torch.cumsum(blocks_per_req, dim=0, out=req_block_base[1:])
        ws_block = req_block_base[:-1].unsqueeze(1) + torch.div(pos, storage, rounding_mode="floor")
        ws_off = torch.remainder(pos, storage)

        take = in_range & compact_valid & (src_blocks >= 0)
        src_token = (owner.clamp(0, self.dcp_size - 1) * max_valid + compact_idx) * storage + local_off
        dst_token = ws_block * storage + ws_off
        ws = compress_kv_cache.new_zeros((total_ws_blocks, *block_shape))
        # Token-granularity views: one row per compressed token (block_shape
        # already includes the storage dim, so strip it back off here).
        ws2 = ws.reshape(total_ws_blocks * storage, *ws.shape[2:])
        pool2 = gathered_pool.reshape(self.dcp_size * max_valid * storage, *gathered_pool.shape[2:])
        dst_flat = dst_token[take].reshape(-1)
        src_flat = src_token[take].reshape(-1)
        keep = dst_flat < ws2.shape[0]
        ws2[dst_flat[keep]] = pool2[src_flat[keep]]

        # Identity block table into the workspace.
        max_blocks_per_req = int(blocks_per_req.max().item())
        block_cols = torch.arange(max_blocks_per_req, dtype=torch.int64, device=device).unsqueeze(0)
        ws_block_table = (req_block_base[:-1].unsqueeze(1) + block_cols).clamp(min=0)
        ws_block_table = torch.where(
            block_cols < blocks_per_req.unsqueeze(1),
            ws_block_table,
            torch.zeros_like(ws_block_table),
        ).to(torch.int32)
        return ws, ws_block_table

    def _dcp_gathered_sinks(self):
        """Attention sinks aligned with the DCP-gathered head dim."""
        sinks = self.attn_sink
        if not isinstance(sinks, torch.Tensor) or self.dcp_size == 1:
            return sinks
        if sinks.numel() == self.num_heads:
            # Replicated per-(global-)head sinks: the gathered head dim is in
            # global head order, so the values pass through unchanged.
            return sinks
        if sinks.numel() == self.n_local_heads:
            # TP-sharded per-head sinks: all-gather along the head axis.
            return self.dcp_group.all_gather(sinks.contiguous().flatten(), dim=0).reshape(-1)
        return sinks
