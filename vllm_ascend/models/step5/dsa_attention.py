# SPDX-License-Identifier: Apache-2.0
"""Ascend Step5 DSA (sparse attention) backend for vLLM 0.33 / MRV2.

Structure follows the CUDA implementation shipped in vLLM 0.33
(``vllm.models.step5.nvidia.sparse_attention``): CSA region state lives in
the upstream deterministic layout --

* one ``Step5CSATailRing`` per request inside its own KV block, addressed as
  ``block_table[req] * rows_per_block + position % capacity`` (the slot
  allocation has no atomics, no async assertions and no cross-step state);
* one FP8 summary per closed eight-token region in a
  ``Step5SparseSummaryCacheLayer`` page that shares the DSA layer's block
  ids, so block lifetime and prefix caching follow the normal KV manager.

The heavy compute stays on the operators proven on Ascend MRV1: region
scoring / causal top-k / ascending sort are fused into the aclnn
``IndexerRegionScore`` operator, and the exact sparse attention runs in the
aclnn ``GenericBlockSparseAttention`` operator reading the paged KV cache
directly (``PA_BBND``). The AICPU scheduling metadata of the attention
operator is built once per step inside the metadata builder -- outside the
model graph, like the FA3 scheduler tiling -- and shared across all layers.

The prefill plan is computed synchronously in the builder (one existing
D2H round trip for ``seq_lens``, host-side grouping, persistent device
buffers); nothing in the prefill path stages temporary pinned host tensors
or relies on cross-step event ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, KVCacheSpec

from vllm_ascend.models.step5.generic_block_sparse_attention import (
    gbsa_expected_counts,
    gbsa_sparse_attention,
    gbsa_sparse_attention_metadata,
)
from vllm.models.step5.nvidia.sparse_summary_cache import (
    Step5SparseSummaryBackend as UpstreamStep5SparseSummaryBackend,
    Step5SparseSummaryCacheLayer as UpstreamStep5SparseSummaryCacheLayer,
    Step5SparseSummaryMetadataBuilder as UpstreamStep5SparseSummaryMetadataBuilder,
)
from vllm_ascend.models.step5.indexer_region_score import indexer_score_topk
from vllm_ascend.models.step5.summary_update import (
    build_prefill_groups,
    update_decode,
    update_prefill,
    update_verify,
)

logger = init_logger(__name__)


class AscendStep5SummarySpec(FullAttentionSpec):
    """Unpadded 8-token/256-byte summary spec.

    Keeps the real 128-token geometry (16 states per 4,096-byte page) so the
    fused indexer sees S = block_size / region_size = 16 rows per page, the
    contract aclnnIndexerRegionScore enforces. The exact-type FullAttentionSpec
    filter in _align_hybrid_attention_page_sizes must not inflate this spec to
    the common 98,304-byte page: the inflated block size (3,072) widens the
    pages to S=384 and the aclnn parameter check rejects that.
    """


class AscendStep5SparseSummaryCacheLayer(UpstreamStep5SparseSummaryCacheLayer):
    def __init__(
        self,
        *,
        prefix: str,
        target_impl: Any,
        static_forward_context: dict[str, Any],
    ) -> None:
        super().__init__(
            prefix=prefix,
            target_impl=target_impl,
            static_forward_context=static_forward_context,
        )
        # The DSA impl reads this group's metadata from the forward context;
        # naming it here keeps the lookup independent of name construction.
        target_impl.summary_layer_name = self.layer_name

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        spec = super().get_kv_cache_spec(vllm_config)
        # The summary layer shares the DSA layer's block ids, so its physical
        # page stride must equal the DSA page: pre-pad it here and upstream's
        # unify_kv_cache_spec_page_size passes the spec through untouched
        # (page_size_bytes == max) instead of growing block_size 128 -> 3072,
        # which would desync the summary group from the 128-token block
        # tables and hand aclnnIndexerRegionScore S=384 pages instead of 16.
        impl = self._target_impl
        kv_dtype = impl.kv_cache_dtype
        if isinstance(kv_dtype, str):
            from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

            kv_dtype = kv_cache_dtype_str_to_dtype(kv_dtype, vllm_config.model_config)
        from vllm.utils.torch_utils import get_dtype_size

        dsa_page = (
            impl.block_size
            * impl.num_kv_heads
            * impl.head_size
            * 2
            * get_dtype_size(kv_dtype)
        )
        return AscendStep5SummarySpec(
            block_size=spec.block_size,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            head_size_v=spec.head_size_v,
            dtype=spec.dtype,
            tokens_per_state=spec.tokens_per_state,
            page_size_padded=int(dsa_page),
        )

    def get_attn_backend(self) -> type["AscendStep5SparseSummaryBackend"]:
        return AscendStep5SparseSummaryBackend


@dataclass
class AscendStep5SparseSummaryMetadata(AttentionMetadata):
    """Addresses from the summary KV group, in full-batch token/request order.

    The summary layer forms its own KV cache group on Ascend, so its manager
    block ids differ from the DSA layer's. Every summary read and write must
    go through these tensors -- the MRV2 runner swaps the common metadata to
    each group's own slot mapping / block table before calling this builder,
    and graph replay refreshes those persistent buffers in place.
    """

    slot_mapping: torch.Tensor
    block_table: torch.Tensor


class AscendStep5SparseSummaryMetadataBuilder(UpstreamStep5SparseSummaryMetadataBuilder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendStep5SparseSummaryMetadata:
        # Retain the runner's device buffers; graph replay refreshes them in place.
        return AscendStep5SparseSummaryMetadata(
            slot_mapping=common_attn_metadata.slot_mapping,
            block_table=common_attn_metadata.block_table_tensor,
        )


class AscendStep5SparseSummaryBackend(UpstreamStep5SparseSummaryBackend):
    @staticmethod
    def get_builder_cls() -> type[AscendStep5SparseSummaryMetadataBuilder]:
        return AscendStep5SparseSummaryMetadataBuilder


class AscendStep5DSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_STEP5_DSA"

    @staticmethod
    def get_impl_cls():
        return AscendStep5DSAImpl

    @staticmethod
    def get_builder_cls():
        return AscendStep5DSAMetadataBuilder

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128, 192]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # The kernels only need block_size % region_size == 0; keep the
        # platform default (128) so scheduler block == kernel block and the
        # ring page shares the main layer's page size.
        return [128]

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)


@dataclass
class AscendStep5DSAPrefillMetadata:
    seq_lens: torch.Tensor
    """(num_prefills,) full sequence lengths (cached prefix + this chunk)."""

    context_lens: torch.Tensor
    """(num_prefills,) tokens already cached before this chunk."""

    block_table: torch.Tensor
    """(num_prefills, max_pages) int32 rows for the prefill requests."""

    query_start_loc: torch.Tensor
    """(num_prefills + 1,) start of each prefill request within the query."""

    max_query_len: int

    num_query_tokens: int = 0
    """Total prefill query tokens, filled from the CPU start-loc copy so the
    step never pays a device sync for it."""

    seq_lens_host: list = None
    context_lens_host: list = None
    """CPU copies produced by the builder's existing seq_lens transfer; the
    impl reads these instead of calling ``.tolist()`` on device tensors."""

    positions: torch.Tensor = None
    """(num_query_tokens,) device positions of the prefill query tokens
    (context + in-chunk offset), built once in the builder."""

    owners: torch.Tensor = None
    """(num_query_tokens,) device local request index of each prefill token,
    for the device-side prefill summary grouping."""

    gbsa_metadata: torch.Tensor = None
    """Per-step AICPU GenericBlockSparseAttention scheduling tensor, built
    once in the builder from the query positions and shared across all
    layers (the scheduled counts are position-derived, layer-independent)."""

    @property
    def query_lens(self) -> torch.Tensor:
        return self.query_start_loc[1:] - self.query_start_loc[:-1]

    @property
    def num_query_tokens_padded(self) -> int:
        return int(self.query_start_loc[-1])


@dataclass
class AscendStep5DSADecodeMetadata:
    seq_lens: torch.Tensor
    """(num_decodes,) context length including the current token."""

    block_table: torch.Tensor
    """(num_decodes, max_pages) int32 rows for the decode requests."""

    query_len: int = 1
    """Uniform decode query length K (1 for plain decode, 1+num_spec for
    MTP verify batches)."""

    gbsa_metadata: torch.Tensor = None
    """Per-step AICPU scheduling tensor for the (possibly flattened verify)
    decode path, built once in the builder and shared across all layers."""


@dataclass
class AscendStep5DSAMetadata(AttentionMetadata):
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    """(num_reqs + 1,) int32 full-batch token start locations, including
    padding (the worker's persistent int32 start-loc buffer, so slices are
    zero-copy and directly usable as the fused indexer's cu_seq_lens)."""
    query_start_loc_int64: torch.Tensor
    """(num_reqs + 1,) int64 copy of ``query_start_loc``, refreshed in
    place into a persistent builder buffer each build: the aclnn
    GenericBlockSparseAttention operator requires ``cu_seqlens_q`` in
    int64 (the fused indexer consumes the int32 original)."""
    seq_lens: torch.Tensor
    """(num_reqs,) int32 full-batch sequence lengths, including padding
    rows (the worker's persistent int32 seq-len buffer)."""
    prefill: AscendStep5DSAPrefillMetadata | None = None
    decode: AscendStep5DSADecodeMetadata | None = None


def _active_decode_num_reqs(
    num_decodes: int,
    num_decode_tokens: int,
    decode_query_len: int,
) -> int:
    """Uniform decode rows lead the batch; count the requests that fill it."""
    if num_decodes <= 0 or decode_query_len <= 0:
        return 0
    return min(num_decodes, num_decode_tokens // decode_query_len)


def _active_prefill_num_reqs(
    num_prefills: int,
    num_prefill_tokens: int,
    qsl_cpu: torch.Tensor,
    num_decodes: int,
) -> int:
    """Trailing prefill requests may be pure padding; count real ones."""
    if num_prefills <= 0:
        return 0
    active = num_prefills
    while active > 0:
        start = int(qsl_cpu[num_decodes + active - 1])
        end = int(qsl_cpu[num_decodes + active])
        if end > start:
            break
        active -= 1
    return active


class AscendStep5DSAMetadataBuilder(
    AttentionMetadataBuilder[AscendStep5DSAMetadata]
):
    # Decode forward is fully tensor-driven: static shapes per capture
    # bucket, all data-dependent choices realized with device-side validity
    # masks, so uniform decode (and MTP verify) batches are ACL-graph
    # capturable under FULL_DECODE_ONLY; mixed prefill+decode batches still
    # run eager -- the contract of UNIFORM_BATCH.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        self.context_len_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )
        # DSA selector geometry, read from the layer impls (all layers share
        # one config) -- the GBSA AICPU metadata is scheduled once per step
        # here in the builder, outside the model graph (FA3 tiling pattern).
        impl = vllm_config.compilation_config.static_forward_context[
            layer_names[0]
        ].impl
        self.gbsa_num_heads = impl.num_heads
        self.gbsa_num_kv_groups = impl.num_kv_groups
        self.gbsa_head_size = impl.head_size
        self.gbsa_region_size = impl.region_size
        self.gbsa_topk = impl.topk_regions
        self.gbsa_block_size = kv_cache_spec.block_size
        self.capture_sizes = set(
            vllm_config.compilation_config.cudagraph_capture_sizes or []
        )
        # Graph-replay metadata buffers: on capture-sized steps the AICPU
        # tensor must live at a stable address (baked into the captured
        # graph) and be refreshed in place each step -- the FA3
        # scheduler_buffers pattern.
        self.gbsa_meta_buffers: dict[tuple, torch.Tensor] = {}
        # Persistent int64 staging for query_start_loc: the aclnn
        # GenericBlockSparseAttention operator requires cu_seqlens_q in
        # int64. build() refreshes it in place so captured verify graphs
        # observe the current step's start locations at a stable address.
        self.query_start_loc_int64_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens + 1,
            dtype=torch.int64,
            device=device,
        )

    def _build_gbsa_metadata(
        self,
        positions: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        block_table: torch.Tensor,
        max_query_len: int,
        *,
        path: str,
        graph_shape: bool,
    ) -> torch.Tensor:
        """One AICPU scheduling pass per step for these query positions.

        On capture-sized shapes the fresh tensor is copied into a persistent
        buffer keyed by the shape signature, so a captured graph replays
        against a stable address whose content is refreshed every step.
        """
        counts = gbsa_expected_counts(
            positions,
            groups=self.gbsa_num_kv_groups,
            region_size=self.gbsa_region_size,
            topk=self.gbsa_topk,
        )
        metadata = gbsa_sparse_attention_metadata(
            counts,
            num_heads=self.gbsa_num_heads,
            num_kv_groups=self.gbsa_num_kv_groups,
            head_dim=self.gbsa_head_size,
            region_size=self.gbsa_region_size,
            topk=self.gbsa_topk,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            max_query_len=max_query_len,
            max_pages=int(block_table.shape[1]),
            page_size=self.gbsa_block_size,
        )
        if not graph_shape:
            return metadata
        buffer_key = (
            path,
            tuple(counts.shape),
            int(cu_seqlens_q.shape[0]) - 1,
            max_query_len,
        )
        buffer = self.gbsa_meta_buffers.get(buffer_key)
        if buffer is None:
            self.gbsa_meta_buffers[buffer_key] = metadata
            return metadata
        buffer.copy_(metadata)
        return buffer

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendStep5DSAMetadata:
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_int64 = self.query_start_loc_int64_buffer[
            : query_start_loc.shape[0]
        ]
        query_start_loc_int64.copy_(query_start_loc)
        seq_lens = common_attn_metadata.seq_lens
        block_table = common_attn_metadata.block_table_tensor
        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        num_input_tokens = (
            getattr(common_attn_metadata, "num_input_tokens", 0)
            or common_attn_metadata.num_actual_tokens
        )
        graph_shape = num_input_tokens in self.capture_sizes

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        prefill_metadata: AscendStep5DSAPrefillMetadata | None = None
        active_prefills = 0
        if num_prefills > 0:
            active_prefills = _active_prefill_num_reqs(
                num_prefills, num_prefill_tokens, qsl_cpu, num_decodes
            )
            prefill_end = num_decodes + active_prefills
            prefill_seq_lens = seq_lens[num_decodes:prefill_end]
            prefill_query_lens = qsl_cpu[num_decodes + 1 : prefill_end + 1] - qsl_cpu[
                num_decodes:prefill_end
            ]
            # One CPU round trip that already existed for context_lens: keep
            # the host lists around so the prefill step never calls .tolist()
            # on a device tensor again (each of those stalls the NPU pipe).
            seq_lens_host_t = seq_lens[num_decodes:prefill_end].detach().cpu()
            context_lens_host_t = seq_lens_host_t - prefill_query_lens
            prefill_context_lens = self.context_len_buffer[num_decodes:prefill_end]
            prefill_context_lens.copy_(
                context_lens_host_t.to(
                    device=self.context_len_buffer.device,
                    dtype=torch.int32,
                )
            )
            prefill_metadata = AscendStep5DSAPrefillMetadata(
                seq_lens=prefill_seq_lens,
                context_lens=prefill_context_lens,
                block_table=block_table[num_decodes:prefill_end],
                query_start_loc=query_start_loc[num_decodes : prefill_end + 1]
                - num_decode_tokens,
                max_query_len=int(prefill_query_lens.max()),
                num_query_tokens=int(prefill_query_lens.sum()),
                seq_lens_host=seq_lens_host_t.tolist(),
                context_lens_host=context_lens_host_t.tolist(),
            )
            # GBSA AICPU scheduling plus the summary-grouping inputs, once
            # per step outside the model graph. Per-token positions:
            # context + in-chunk offset, straight from the (rebased)
            # start-loc; the scheduled counts depend on these positions
            # only, so every layer reuses these tensors.
            qsl = prefill_metadata.query_start_loc.to(torch.int64)
            num_query_tokens = prefill_metadata.num_query_tokens
            token_idx = torch.arange(
                num_query_tokens, device=seq_lens.device, dtype=torch.int64
            )
            reqs = torch.searchsorted(qsl, token_idx, right=True) - 1
            prefill_positions = (
                prefill_context_lens.to(torch.int64)[reqs] + token_idx - qsl[reqs]
            )
            prefill_metadata.positions = prefill_positions
            prefill_metadata.owners = reqs.to(torch.int64)
            prefill_metadata.gbsa_metadata = self._build_gbsa_metadata(
                prefill_positions,
                prefill_metadata.query_start_loc,
                prefill_seq_lens,
                block_table,
                prefill_metadata.max_query_len,
                path="prefill",
                graph_shape=graph_shape,
            )

        decode_metadata: AscendStep5DSADecodeMetadata | None = None
        active_decodes = 0
        if num_decodes > 0:
            query_lens_cpu = qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]
            decode_query_len = int(query_lens_cpu[0].item())
            active_decodes = _active_decode_num_reqs(
                num_decodes, num_decode_tokens, decode_query_len
            )
            gbsa_metadata = None
            if active_decodes > 0:
                decode_seq_lens = seq_lens[:active_decodes]
                if decode_query_len == 1:
                    # Plain decode: one query per request at its own position.
                    positions = decode_seq_lens.to(torch.int64) - 1
                    cu_seq_lens = torch.arange(
                        active_decodes + 1, device=seq_lens.device, dtype=torch.int64
                    )
                    seqused_kv = decode_seq_lens
                    max_query_len = 1
                else:
                    # MTP verify multi-token varlen: each request stays one
                    # query_len=K+1 request spanning kv positions ctx..ctx+K
                    # (ctx = seq_len - q_len). The per-token positions drive
                    # the scheduled counts; the request-level start-loc head
                    # (decode rows lead the batch) and the full seq lens
                    # anchor the right-aligned causal bound.
                    total = active_decodes * decode_query_len
                    row_base = (
                        decode_seq_lens.to(torch.int64) - decode_query_len
                    ).repeat_interleave(decode_query_len)
                    positions = row_base + (
                        torch.arange(total, device=seq_lens.device, dtype=torch.int64)
                        % decode_query_len
                    )
                    cu_seq_lens = query_start_loc[: active_decodes + 1]
                    seqused_kv = decode_seq_lens
                    max_query_len = decode_query_len
                gbsa_metadata = self._build_gbsa_metadata(
                    positions,
                    cu_seq_lens,
                    seqused_kv,
                    block_table,
                    max_query_len,
                    path="decode",
                    graph_shape=graph_shape,
                )
            decode_metadata = AscendStep5DSADecodeMetadata(
                seq_lens=seq_lens[:active_decodes],
                block_table=block_table[:active_decodes],
                query_len=decode_query_len,
                gbsa_metadata=gbsa_metadata,
            )

        return AscendStep5DSAMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            num_decodes=active_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=active_prefills,
            num_prefill_tokens=num_prefill_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            query_start_loc=query_start_loc,
            query_start_loc_int64=query_start_loc_int64,
            seq_lens=seq_lens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )


class AscendStep5DSAImpl(AttentionImplBase[AscendStep5DSAMetadata]):
    """Ascend Step5 DSA implementation over the upstream ring layout."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_regions: int,
        region_size: int,
        proxy_dim: int,
        max_num_seqs: int,
        max_model_len: int,
        block_size: int,
        decode_out_dtype: torch.dtype | None = None,
        num_speculative_tokens: int = 0,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.topk_regions = topk_regions
        self.region_size = region_size
        self.proxy_dim = proxy_dim
        self.block_size = block_size
        if block_size % region_size != 0:
            raise ValueError(
                "Step5 DSA requires block_size to be a multiple of the region "
                f"size, got block_size={block_size}, region_size={region_size}."
            )
        self.num_speculative_tokens = int(num_speculative_tokens)
        if self.num_speculative_tokens > 0 and (
            self.num_speculative_tokens + 1 > region_size
        ):
            raise ValueError(
                "Step5 DSA MTP requires num_speculative_tokens + 1 <= "
                f"region_size, got {self.num_speculative_tokens + 1} > "
                f"{region_size}."
            )
        self.num_kv_groups = self.num_kv_heads
        self.max_num_seqs = max(1, int(max_num_seqs))
        self.max_model_len = int(max_model_len)
        self._decode_out_dtype = decode_out_dtype or torch.bfloat16
        # Selector width: every region a max-length sequence can fill.
        self._max_regions = -(-self.max_model_len // self.region_size)
        # Injected by the model after construction (upstream wiring):
        # ``summary_cache`` is the Step5SparseSummaryCache bound to the
        # per-block FP8 summary page; ``csa_ring`` is the Step5CSATailRing;
        # ``summary_layer_name`` locates this group's own metadata.
        self.summary_cache: Any = None
        self.summary_layer_name: str | None = None
        self.csa_ring: Any = None

    def _dsa_virtual_engine(self) -> int:
        if is_forward_context_available():
            return int(getattr(get_forward_context(), "virtual_engine", 0) or 0)
        return 0

    # -- summary state (upstream ring layout) ------------------------------

    def _summary_metadata(self) -> "AscendStep5SparseSummaryMetadata":
        """This DSA layer's summary-group addresses from the forward context.

        The summary layer forms its own KV cache group whose manager block
        ids are distinct from the DSA layer's; under the shared hybrid
        backing, addressing the summary pages with DSA ids would land on the
        DSA group's live K/V bytes.
        """
        attn_metadata = get_forward_context().attn_metadata
        summary_md = attn_metadata.get(self.summary_layer_name)
        if not isinstance(summary_md, AscendStep5SparseSummaryMetadata):
            raise RuntimeError(
                "Step5 summary group metadata is missing or invalid for "
                f"layer {self.summary_layer_name!r}."
            )
        return summary_md

    def _update_summary_state(
        self,
        md: AscendStep5DSAMetadata,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
    ) -> None:
        """Fold this step's indexer K/Z rows into the ring and the FP8 page
        summaries.

        Plain decode stores its row (or closes its region when the row is
        the eighth); MTP verify stores K+1 rows in order, re-closing any
        region a rejected draft closed (the next step overwrites the row);
        prefill first replays the regions this chunk closes -- including
        carry-over rows stored by an earlier chunk -- then stores the open
        rows. Mixed batches run the decode rows first (they lead the batch).

        Summary pages are addressed with the summary group's own slot
        mapping; ring rows keep the ring group's block-table addressing.
        """
        num_tokens = int(md.num_actual_tokens)
        if num_tokens <= 0:
            return
        summary = self.summary_cache
        ring = self.csa_ring
        if ring is None or summary is None:
            raise RuntimeError(
                "Step5 DSA requires the CSA ring and summary cache to be "
                "bound before the first forward."
            )
        summary_md = self._summary_metadata()
        mean = summary.mean
        slots = summary_md.slot_mapping[:num_tokens]
        ring_md = ring.metadata()
        index_k = index_k[:num_tokens].contiguous()
        index_z = index_z[:num_tokens].contiguous()

        verify_batch = (
            self.num_speculative_tokens > 0
            and md.decode is not None
            and md.decode.query_len > 1
        )
        if verify_batch:
            d = md.decode
            total = md.num_decodes * int(d.query_len)
            update_verify(
                index_k[:total],
                index_z[:total],
                slots[:total],
                md.query_start_loc[: md.num_decodes + 1],
                d.seq_lens[: md.num_decodes],
                ring_md.base[:total],
                mean,
                ring.rows,
                ring.capacity,
            )
            return

        if md.num_decodes > 0 and md.decode is not None:
            d = md.decode
            n = md.num_decodes
            seq_lens_i32 = d.seq_lens[:n]
            update_decode(
                index_k[:n],
                index_z[:n],
                slots[:n],
                seq_lens_i32.to(torch.int64) - 1,
                ring_md.base[:n],
                mean,
                ring.rows,
                ring.capacity,
            )

        p = md.prefill
        if p is not None and p.num_query_tokens > 0:
            start = md.num_decode_tokens
            rows = slice(start, start + p.num_query_tokens)
            # Device-side grouping, one synchronous launch: which rows share
            # a region, and which region each closed group materializes.
            summary_plan = build_prefill_groups(
                slots[rows],
                p.positions,
                p.owners,
                p.query_start_loc,
            )
            update_prefill(
                index_k[rows],
                index_z[rows],
                summary_plan,
                p.positions,
                ring_md.base[rows],
                ring_md.slot_mapping[rows],
                mean,
                ring.rows,
                ring.capacity,
            )


    # -- sparse GQA forward paths (aclnn operators) -------------------------

    def _forward_sparse_gqa_decode(
        self,
        *,
        md: AscendStep5DSAMetadata,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        d = md.decode
        assert d is not None and d.query_len == 1
        n = md.num_decodes
        seq_lens_i32 = d.seq_lens.to(torch.int32)
        summary_md = self._summary_metadata()
        # Fused score + causal topk + ascending sort via aclnn; the exact
        # attention is the aclnn GenericBlockSparseAttention operator, which
        # appends the current region (floor(pos / region_size)) and applies
        # the per-token causal truncation itself. actual_kv_lengths is the
        # effective KV length including the current query (seq_len); the
        # query token's own 0-based position (seq_len - 1) feeds the
        # attention operator's current-region append. The summary table is
        # the summary group's own block table; the ids it returns are
        # request-relative, so GBSA keeps translating them through the DSA
        # layer's table.
        cu_seq_lens = torch.arange(n + 1, device=index_q.device, dtype=torch.int32)
        positions = seq_lens_i32.to(torch.int64) - 1
        actual_kv_lengths = seq_lens_i32
        _scores, ids = indexer_score_topk(
            index_q[:n],
            weights[:n],
            self.summary_cache.mean.view(torch.float8_e4m3fn).unsqueeze(2),
            summary_md.block_table[:n],
            cu_seq_lens,
            actual_kv_lengths,
            max_region_len=self._max_regions,
            topk=self.topk_regions,
            region_size=self.region_size,
        )
        out = gbsa_sparse_attention(
            query,
            key_cache,
            value_cache,
            ids,
            positions,
            d.block_table,
            cu_seq_lens.to(torch.int64),
            seq_lens_i32,
            d.gbsa_metadata,
            num_kv_groups=self.num_kv_groups,
            region_size=self.region_size,
            softmax_scale=self.scale,
        )
        output[:n] = out.to(output.dtype)

    def _forward_sparse_gqa_verify(
        self,
        *,
        md: AscendStep5DSAMetadata,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Sparse attention for an MTP verify batch (K+1 rows per request).

        Each verify request stays one multi-token varlen request of
        ``query_len = K+1`` rows spanning kv positions ``ctx..ctx+K`` with
        ``ctx = seq_len - K - 1``. The exact attention goes through the aclnn
        ``GenericBlockSparseAttention`` operator: its right-aligned causal
        bound (``seqused_kv = seq_len``, ``q_len = K+1``) places query row
        ``i`` at kv position ``ctx + i`` and truncates it to its own
        position, and the operator appends each row's current region with
        causal truncation. The fused indexer's per-token mask3 bound keeps
        the row's current region out of top-k competition -- the same
        contract the single-token decode path guarantees, evaluated once per
        request instead of once per row. Everything is device-driven on
        static per-bucket shapes, so uniform verify batches stay ACL-graph
        capturable.
        """
        decode_md = md.decode
        num_decodes = md.num_decodes
        query_len = int(decode_md.query_len)
        total_tokens = num_decodes * query_len
        seq_lens = decode_md.seq_lens[:num_decodes]
        cu_seq_lens = md.query_start_loc[: num_decodes + 1]
        cu_seqlens_q = md.query_start_loc_int64[: num_decodes + 1]
        actual_kv_lengths = seq_lens
        summary_md = self._summary_metadata()
        _scores, ids = indexer_score_topk(
            index_q[:total_tokens],
            weights[:total_tokens],
            self.summary_cache.mean.view(torch.float8_e4m3fn).unsqueeze(2),
            summary_md.block_table[:num_decodes],
            cu_seq_lens,
            actual_kv_lengths,
            max_region_len=self._max_regions,
            topk=self.topk_regions,
            region_size=self.region_size,
        )
        out = gbsa_sparse_attention(
            query,
            key_cache,
            value_cache,
            ids,
            positions[:total_tokens],
            decode_md.block_table[:num_decodes],
            cu_seqlens_q,
            seq_lens,
            decode_md.gbsa_metadata,
            num_kv_groups=self.num_kv_groups,
            region_size=self.region_size,
            softmax_scale=self.scale,
        )
        output[:total_tokens] = out.to(output.dtype)

    def _forward_sparse_gqa_prefill(
        self,
        *,
        md: AscendStep5DSAMetadata,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        index_q: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
        token_offset: int,
    ) -> None:
        p = md.prefill
        assert p is not None
        num_reqs = md.num_prefills
        num_query_tokens = p.num_query_tokens

        # Live history width in logical regions (the furthest last-position
        # of any request in this chunk), bounding what the fused operator
        # needs to score.
        query_lens_list = [
            int(p.seq_lens_host[row]) - int(p.context_lens_host[row])
            for row in range(num_reqs)
        ]
        width = max(
            (
                (int(p.seq_lens_host[row]) - 1) // self.region_size
                for row in range(num_reqs)
                if query_lens_list[row] > 0
            ),
            default=0,
        )

        q_positions = p.positions[:num_query_tokens].to(torch.int32)

        # Fused score + causal topk + ascending sort via aclnn. The
        # operator's mask3 bound floor((context_lens[n] + j)/region_size) is
        # each token's own position // region_size, so the current region
        # never competes for a slot; the attention operator appends it and
        # causally truncates it to the token's position.
        cu_seq_lens = torch.cat(
            [
                torch.zeros((1,), device=index_q.device, dtype=torch.int32),
                p.query_start_loc[1 : num_reqs + 1].to(torch.int32),
            ]
        )
        actual_kv_lengths = p.seq_lens[:num_reqs].to(torch.int32)
        summary_md = self._summary_metadata()
        _scores, ids = indexer_score_topk(
            index_q[:num_query_tokens],
            weights[:num_query_tokens],
            self.summary_cache.mean.view(torch.float8_e4m3fn).unsqueeze(2),
            summary_md.block_table[md.num_decodes : md.num_decodes + num_reqs],
            cu_seq_lens,
            actual_kv_lengths,
            max_region_len=width,
            topk=self.topk_regions,
            region_size=self.region_size,
        )
        out = gbsa_sparse_attention(
            query,
            key_cache,
            value_cache,
            ids,
            q_positions,
            p.block_table[:num_reqs],
            cu_seq_lens.to(torch.int64),
            p.seq_lens[:num_reqs].to(torch.int32),
            p.gbsa_metadata,
            num_kv_groups=self.num_kv_groups,
            region_size=self.region_size,
            softmax_scale=self.scale,
        )
        output[token_offset : token_offset + num_query_tokens] = out.to(output.dtype)

    def forward(
        self,
        layer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        index_z: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output.zero_()
        md = attn_metadata[layer.layer_name]
        if not isinstance(md, AscendStep5DSAMetadata):
            return output.zero_()

        num_tokens = md.num_actual_tokens
        self._update_summary_state(md, index_k, index_z)

        key_cache, value_cache = kv_cache[0], kv_cache[1]

        if md.num_decodes > 0:
            n = md.num_decodes
            decode_query_len = int(md.decode.query_len) if md.decode else 1
            if decode_query_len > 1:
                # MTP verify batch: every decode request contributes K+1
                # consecutive rows (positions ctx..ctx+K).
                total = n * decode_query_len
                self._forward_sparse_gqa_verify(
                    md=md,
                    query=query[:total].contiguous(),
                    key_cache=key_cache,
                    value_cache=value_cache,
                    index_q=index_q[:total],
                    weights=weights[:total],
                    positions=positions,
                    output=output,
                )
            else:
                self._forward_sparse_gqa_decode(
                    md=md,
                    query=query[:n].contiguous(),
                    key_cache=key_cache,
                    value_cache=value_cache,
                    index_q=index_q[:n],
                    weights=weights[:n],
                    output=output,
                )

        if md.num_prefills > 0:
            start = md.num_decode_tokens
            self._forward_sparse_gqa_prefill(
                md=md,
                query=query[start:num_tokens].contiguous(),
                key_cache=key_cache,
                value_cache=value_cache,
                index_q=index_q[start:num_tokens],
                weights=weights[start:num_tokens],
                output=output,
                token_offset=start,
            )
        return output
