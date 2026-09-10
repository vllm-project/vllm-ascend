from dataclasses import dataclass
from typing import Any

import scipy  # type: ignore
import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_tp_group
from vllm.triton_utils import HAS_TRITON
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
from vllm_ascend.utils import (
    enable_dsa_cp,
    enable_sfa_dcp_replicated_indexer,
    enable_sfa_dcp_sharded_indexer,
    vllm_version_is,
)

if vllm_version_is("0.28.0"):
    from vllm.model_executor.layers.attention.pcp import _gather_prefill_cache_inputs  # type: ignore[import-not-found]
else:
    from vllm.v1.attention.ops.pcp import _gather_prefill_cache_inputs  # type: ignore[import-not-found]

# Slots of the k / scale caches inside an indexer's own ``k_cache.kv_cache``
# tuple (the scale slot exists only when LI C8 is enabled).
INDEXER_K_CACHE_SLOT = 0
INDEXER_SCALE_CACHE_SLOT = 1


def _get_int_config_attr(config: Any, name: str, default: int) -> int:
    value = getattr(config, name, default)
    return value if type(value) is int else default


def _get_dcp_metadata_max_lens_from_cpu(
    common_attn_metadata: CommonAttentionMetadata,
    num_reqs: int,
) -> tuple[int, int] | None:
    query_start_loc_cpu = getattr(common_attn_metadata, "query_start_loc_cpu", None)
    seq_lens_cpu_upper_bound = getattr(common_attn_metadata, "seq_lens_cpu_upper_bound", None)
    if query_start_loc_cpu is None or seq_lens_cpu_upper_bound is None:
        return None
    query_lens_cpu = query_start_loc_cpu[1 : num_reqs + 1] - query_start_loc_cpu[:num_reqs]
    return int(query_lens_cpu.max()), int(seq_lens_cpu_upper_bound[:num_reqs].max())


@dataclass
class AscendSFAIndexerMetadata:
    """Engine-side metadata owned by an SFA indexer cache layer.

    Carries everything the indexer kernels need from the engine: the paged
    cache view (block table, slot mapping), the rope tables, and the LI C8
    reshape-optim fields. The sequence lengths and rope tables reflect the
    unsharded batch; the parallel-layout sequence lengths the top-k kernel
    consumes are injected per forward by SFA (see
    ``actual_seq_lengths_query`` / ``actual_seq_lengths_key``).
    """

    num_actual_tokens: int
    # Write-ready slot mapping for the indexer's own cache layout, already
    # resolved for the active parallel mode: under PCP it is the full
    # (gather-region) mapping, otherwise the input-token slice. Under DSA-CP
    # the input-token count is padded to the TP-aligned size, so the slice
    # equals the full padded mapping the gathered write needs.
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    cum_query_lens: torch.Tensor
    block_table: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor
    block_size: int = 0
    group_len: torch.Tensor | None = None
    group_key_idx: torch.Tensor | None = None
    group_key_cache_idx: torch.Tensor | None = None
    # Parallel-layout sequence lengths consumed by the top-k kernel, injected
    # by SFA per forward: base/PCP modes use the unsharded ``cum_query_lens``
    # / ``seq_lens`` equivalents, DSA-CP shards them per rank. Transported on
    # this metadata so the indexer forward interface stays layout-agnostic.
    actual_seq_lengths_query: torch.Tensor | None = None
    actual_seq_lengths_key: torch.Tensor | None = None
    # Decode-token count injected by SFA per forward; the PCP cache-write
    # gather splits the local prefill region on it (all-decode batches skip
    # the gather).
    num_decode_tokens: int = 0
    dcp_sharded_indexer_enabled: bool = False
    dcp_rank: int = 0
    dcp_world_size: int = 1
    dcp_interleave_size: int = 128
    dcp_local_block_table: torch.Tensor | None = None
    dcp_local_slot_mapping: torch.Tensor | None = None
    dcp_local_token_mask: torch.Tensor | None = None
    request_query_lens: torch.Tensor | None = None
    request_context_lens: torch.Tensor | None = None
    local_visible_by_query: torch.Tensor | None = None
    li_cum_query_lens: torch.Tensor | None = None
    # True only when CPU-authoritative prefill metadata proves every DCP rank
    # has at least one visible local key for every query row.  This permits the
    # expensive empty-row publication mask to be skipped without weakening the
    # short-context fallback semantics.
    all_local_rows_active: bool = False


def dcp_local_visible_counts(
    global_visible: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    interleave_size: int,
) -> torch.Tensor:
    base = global_visible // interleave_size // dcp_world_size * interleave_size
    remainder = global_visible - base * dcp_world_size
    return base + torch.clamp(
        remainder - dcp_rank * interleave_size,
        min=0,
        max=interleave_size,
    )


def dcp_local_to_global_indices(
    local_indices: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    interleave_size: int,
) -> torch.Tensor:
    valid = local_indices >= 0
    local = torch.clamp(local_indices, min=0)
    # Reuse the quotient instead of issuing both integer division and modulo.
    # For local = q*interleave + r, the original mapping
    # q*world*interleave + rank*interleave + r equals
    # local + q*(world-1)*interleave + rank*interleave.
    local_block = local // interleave_size
    global_indices = local + local_block * ((dcp_world_size - 1) * interleave_size) + dcp_rank * interleave_size
    return torch.where(valid, global_indices.to(local_indices.dtype), local_indices)


def mask_dcp_inactive_local_candidates(
    local_indices: torch.Tensor,
    local_scores: torch.Tensor,
    local_visible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask pseudo-rows whose rank-local K-domain is empty.

    The sharded cache allocation is already physical factor=1, so an empty
    local row cannot fall back to the replicated indexer.  The native LI call
    is issued with a one-token dummy visibility for such rows and its result
    is discarded here before publication.
    """
    active = (local_visible > 0).view(-1, *([1] * (local_indices.dim() - 1)))
    masked_indices = torch.where(active, local_indices, torch.full_like(local_indices, -1))
    masked_scores = torch.where(active, local_scores, torch.full_like(local_scores, float("-inf")))
    return masked_indices, masked_scores


def merge_dcp_indexer_candidates(
    candidate_indices: torch.Tensor,
    candidate_scores: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Merge gathered DCP records by score.

    LightningIndexer returns ``[row, 1, candidate]`` for the single indexer
    head. A real DCP all-gather therefore produces ``[rank,row,1,candidate]``.
    Preserve the singleton head dimension for the existing SFA consumer ABI.

    Unique-cutoff rows must match incumbent full-K LI. At a cutoff tie the
    supported contract is strict-threshold equivalence plus deterministic
    repeat; choosing the same tied subset by global-index order is not required.
    """
    preserve_head_dim = False
    if candidate_indices.dim() == 4:
        if candidate_indices.shape[2] != 1 or candidate_scores.shape[2] != 1:
            raise ValueError(
                "DCP sharded indexer expects exactly one LI head, got "
                f"indices={tuple(candidate_indices.shape)} scores={tuple(candidate_scores.shape)}"
            )
        candidate_indices = candidate_indices.squeeze(2)
        candidate_scores = candidate_scores.squeeze(2)
        preserve_head_dim = True
    elif candidate_indices.dim() == 2:
        candidate_indices = candidate_indices.unsqueeze(0)
        candidate_scores = candidate_scores.unsqueeze(0)
    if candidate_indices.dim() != 3 or candidate_scores.dim() != 3:
        raise ValueError(
            "DCP sharded indexer candidates must be [rank,row,K] or [rank,row,1,K], got "
            f"indices={tuple(candidate_indices.shape)} scores={tuple(candidate_scores.shape)}"
        )
    if candidate_indices.shape != candidate_scores.shape:
        raise ValueError(
            "DCP sharded indexer index/score shapes must match, got "
            f"indices={tuple(candidate_indices.shape)} scores={tuple(candidate_scores.shape)}"
        )
    flat_indices = candidate_indices.permute(1, 0, 2).reshape(candidate_indices.shape[1], -1)
    flat_scores = candidate_scores.permute(1, 0, 2).reshape(candidate_scores.shape[1], -1)
    valid = flat_indices >= 0
    scores = torch.where(valid, flat_scores, torch.full_like(flat_scores, float("-inf")))
    k = min(topk, scores.shape[-1])
    selected_scores, selected_pos = torch.topk(scores, k=k, dim=-1, largest=True, sorted=True)
    selected = torch.gather(flat_indices, -1, selected_pos)
    selected = torch.where(selected_scores > float("-inf"), selected, torch.full_like(selected, -1))
    if selected.shape[-1] < topk:
        selected = torch.nn.functional.pad(selected, (0, topk - selected.shape[-1]), value=-1)
    return selected.unsqueeze(1) if preserve_head_dim else selected


class AscendSFAIndexerBackend(nn.Module, AttentionBackend):
    """Backend and impl for split SFA indexer cache layers - one class per
    indexer family, two interfaces:

    - Engine side (class interface): the vLLM AttentionBackend contract
      (builder selection, KV-cache shape, kernel block sizes), consumed
      through static/class methods; the engine never instantiates it.
    - Model side (instance interface): the per-layer indexer impl
      (an ``nn.Module``) instantiated by IndexerWrapper, owning the compute
      (k path, top-k selection) and cache persistence.

    The SFA indexer cache is represented as its own AttentionLayerBase so the
    KV-cache planner can assign an independent physical tensor while sharing
    block ids with the main MLA cache group. Its builder constructs the
    metadata the indexer forward consumes (paged cache view, rope tables,
    LI C8 reshape-optim fields); SFA only injects the parallel-layout values
    (sequence lengths, decode count) onto that metadata per forward.

    Do not reuse AscendSFAMetadataBuilder here. It inherits vLLM's
    MLACommonMetadataBuilder, whose initializer assumes layer_names[0] points to
    a real MLAAttention object with ``prefill_backend`` in static_forward_context.
    The indexer cache layer points to DeepseekV32IndexerCache instead, which has
    no ``prefill_backend``.

    The forward path is re-implemented with NPU kernels because the upstream
    Indexer hardcodes the CUDA fp8 path.
    TODO: Will be removed once original Indexer supports different quantization methods.
    """

    accept_output_buffer: bool = True

    # q_hadamard and k_hadamard tensor shared when dsa c8 enabled
    q_hadamard: torch.Tensor | None = None
    k_hadamard: torch.Tensor | None = None

    @staticmethod
    def get_impl_cls():
        return None

    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_SFA_INDEXER"

    @staticmethod
    def get_builder_cls():
        return AscendSFAIndexerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]

    # ---- model-side impl interface (per-layer instance) ----

    def __init__(self, vllm_indexer: nn.Module, qk_rope_head_dim: int) -> None:
        super().__init__()

        self.n_head: int = vllm_indexer.n_head  # 64
        self.head_dim: int = vllm_indexer.head_dim  # 128
        self.topk_tokens: int = vllm_indexer.topk_tokens  # 2048
        self.q_lora_rank: int = vllm_indexer.q_lora_rank  # 1536
        self.wq_b = vllm_indexer.wq_b
        self.wk_weights_proj = vllm_indexer.wk_weights_proj
        self.k_norm = vllm_indexer.k_norm
        self.softmax_scale = vllm_indexer.softmax_scale
        self.k_cache: Any = getattr(vllm_indexer, "k_cache", None)
        if self.k_cache is None:
            raise RuntimeError(
                "Indexer backend requires the vLLM indexer module to expose "
                "its k_cache (registered by the attention layer); got None."
            )
        self.qk_rope_head_dim = qk_rope_head_dim
        vllm_indexer.topk_indices_buffer = None  # delete topk_indices_buffer

        self.enable_sparse_li_c8 = get_ascend_config().is_sparse_li_c8_layer(self.k_cache.prefix)
        if self.enable_sparse_li_c8:
            if get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION):
                self.c8_k_cache_dtype = torch.float8_e4m3fn
                self.c8_k_scale_cache_dtype = torch.float32
            else:
                self.c8_k_cache_dtype = torch.int8
                self.c8_k_scale_cache_dtype = torch.float16

        model_type = get_current_vllm_config().model_config.hf_config.model_type
        self.is_rope_neox_style = model_type not in ["glm_moe_dsa"]
        self.use_torch_npu_lightning_indexer = model_type in ["glm_moe_dsa"]

        # Cache-write gathers for parallel layouts: PCP all-gathers the
        # prefill region across the CP group, DSA-CP all-gathers the indexer
        # k across the TP group. Both are no-ops in the base layout.
        parallel_config = get_current_vllm_config().parallel_config
        self._pcp_active = parallel_config.prefill_context_parallel_size > 1
        self._dsa_cp_active = enable_dsa_cp()

    def _select_dcp_sharded_topk(
        self,
        q_li: torch.Tensor,
        q_li_scale: torch.Tensor | None,
        q_li_shape_ori: tuple[Any, ...] | None,
        weights: torch.Tensor,
        indexer_metadata: AscendSFAIndexerMetadata,
    ) -> torch.Tensor | None:
        if (
            not getattr(indexer_metadata, "dcp_sharded_indexer_enabled", False)
            or self.enable_sparse_li_c8
            or indexer_metadata.dcp_local_block_table is None
            or indexer_metadata.local_visible_by_query is None
            or indexer_metadata.li_cum_query_lens is None
        ):
            return None
        local_visible = indexer_metadata.local_visible_by_query
        # PA_BSND LI does not need to observe an empty physical K-domain.
        # Use one dummy visible key for inactive rows, then discard that row's
        # native output before DCP publication.  This keeps factor=1 cache
        # semantics valid without a late replicated-path fallback.
        native_local_visible = (
            local_visible if indexer_metadata.all_local_rows_active else torch.clamp(local_visible, min=1)
        )

        selected = DeviceOperator.indexer_select_post_process(
            q_li,
            q_li_scale,
            q_li_shape_ori,
            weights,
            self.k_cache.kv_cache,
            INDEXER_K_CACHE_SLOT,
            INDEXER_SCALE_CACHE_SLOT,
            indexer_metadata,
            indexer_metadata.li_cum_query_lens,
            native_local_visible,
            self.enable_sparse_li_c8,
            self.use_torch_npu_lightning_indexer,
            return_selected_scores=True,
            sparse_mode=0,
            block_table=indexer_metadata.dcp_local_block_table,
        )
        local_indices, local_scores = selected
        if not indexer_metadata.all_local_rows_active:
            local_indices, local_scores = mask_dcp_inactive_local_candidates(local_indices, local_scores, local_visible)
        global_indices = dcp_local_to_global_indices(
            local_indices,
            indexer_metadata.dcp_rank,
            indexer_metadata.dcp_world_size,
            indexer_metadata.dcp_interleave_size,
        )
        if indexer_metadata.dcp_world_size == 1:
            return global_indices

        dcp_group = get_dcp_group()
        if dcp_group.world_size != 16 or indexer_metadata.dcp_world_size != 16:
            raise RuntimeError("DCP sharded indexer butterfly merge is restricted to DCP16.")
        rank_in_group = dcp_group.rank_in_group
        indices = global_indices.contiguous()
        scores = local_scores.contiguous()
        # Four pairwise merge rounds reduce communication from publishing all
        # 16 rank-local K lists to every rank at once to O(K log DCP).  Both
        # peers order lower-rank subgroup first, making cutoff-tie selection
        # repeat-deterministic while preserving the established threshold
        # equivalence contract.
        for level, step in enumerate((1, 2, 4, 8)):
            peer_in_group = rank_in_group ^ step
            peer = dcp_group.ranks[peer_in_group]
            other_scores = torch.empty_like(scores)
            other_indices = torch.empty_like(indices)
            p2p_ops = [
                dist.P2POp(
                    dist.isend,
                    scores,
                    peer,
                    group=dcp_group.device_group,
                    tag=320 + level * 2,
                ),
                dist.P2POp(
                    dist.irecv,
                    other_scores,
                    peer,
                    group=dcp_group.device_group,
                    tag=320 + level * 2,
                ),
                dist.P2POp(
                    dist.isend,
                    indices,
                    peer,
                    group=dcp_group.device_group,
                    tag=321 + level * 2,
                ),
                dist.P2POp(
                    dist.irecv,
                    other_indices,
                    peer,
                    group=dcp_group.device_group,
                    tag=321 + level * 2,
                ),
            ]
            for request in dist.batch_isend_irecv(p2p_ops):
                request.wait()
            if rank_in_group & step:
                candidate_scores = torch.cat((other_scores, scores), dim=-1)
                candidate_indices = torch.cat((other_indices, indices), dim=-1)
            else:
                candidate_scores = torch.cat((scores, other_scores), dim=-1)
                candidate_indices = torch.cat((indices, other_indices), dim=-1)
            scores, selected_pos = torch.topk(
                candidate_scores,
                k=self.topk_tokens,
                dim=-1,
                largest=True,
                sorted=True,
            )
            indices = torch.gather(candidate_indices, -1, selected_pos)
        return indices

    def process_weights_after_loading(self) -> None:
        if self.enable_sparse_li_c8 and AscendSFAIndexerBackend.q_hadamard is None:
            hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu")
            AscendSFAIndexerBackend.q_hadamard = hadamard / (128**0.5)
        if self.enable_sparse_li_c8 and AscendSFAIndexerBackend.k_hadamard is None:
            hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu")
            AscendSFAIndexerBackend.k_hadamard = hadamard / (128**0.5)

    @property
    def num_cache_tensors(self) -> int:
        """Number of tensors this indexer's cache occupies in the composed
        ``kv_cache`` tuple (k cache only, or k cache plus scale cache)."""
        return 2 if self.enable_sparse_li_c8 else 1

    def write_cache(
        self,
        k_li: torch.Tensor,
        k_li_scale: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        indexer_attn_metadata: Any | None = None,
    ) -> None:
        """Persist ``k_li`` (and ``k_li_scale`` when LI C8 is enabled) into
        this indexer's own cache tensors: slot 0 of ``self.k_cache.kv_cache``
        is the k cache, slot 1 (present only for LI C8) is the scale cache.

        ``forward`` calls this after ``_gather_cache_inputs`` has resolved
        the parallel layout of the tensors and the slot mapping; variants
        with a different cache layout should override it.
        ``indexer_attn_metadata`` is this indexer's own layer metadata; the
        LI C8 reshape-optim path reads its group fields.
        """
        indexer_k_cache = self.k_cache.kv_cache[INDEXER_K_CACHE_SLOT]
        use_reshape_optim = self._use_c8_reshape_optim()
        if use_reshape_optim:
            assert indexer_attn_metadata is not None
            torch.ops._C_ascend.store_kv_block(
                k_li,
                indexer_k_cache,
                indexer_attn_metadata.group_len,
                indexer_attn_metadata.group_key_idx,
                indexer_attn_metadata.group_key_cache_idx,
                indexer_attn_metadata.block_size,
            )
        else:
            torch_npu.npu_scatter_nd_update_(
                indexer_k_cache.view(-1, k_li.shape[-1]),
                slot_mapping.view(-1, 1),
                k_li.view(-1, k_li.shape[-1]),
            )
        if self.enable_sparse_li_c8:
            assert k_li_scale is not None
            indexer_scale_cache = self.k_cache.kv_cache[INDEXER_SCALE_CACHE_SLOT]
            if use_reshape_optim:
                assert indexer_attn_metadata is not None
                torch.ops._C_ascend.store_kv_block(
                    k_li_scale,
                    indexer_scale_cache,
                    indexer_attn_metadata.group_len,
                    indexer_attn_metadata.group_key_idx,
                    indexer_attn_metadata.group_key_cache_idx,
                    indexer_attn_metadata.block_size,
                )
            else:
                torch_npu.npu_scatter_nd_update_(
                    indexer_scale_cache.view(-1, k_li_scale.shape[-1]),
                    slot_mapping.view(-1, 1),
                    k_li_scale.view(-1, k_li_scale.shape[-1]),
                )

    def _use_c8_reshape_optim(self) -> bool:
        """Whether this indexer can use the LI C8 cache-write operator."""
        return self.enable_sparse_li_c8 and get_ascend_config().c8_reshape_optim_enabled

    def forward_k(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """k path: compute ``k_li`` (and ``k_li_scale`` when LI C8 is
        enabled) from the hidden-states stage SFA hands in (raw states on
        fused preprocess paths, prepared states on native paths). SFA then
        persists the result through ``write_cache`` before the top-k stage
        runs, since the top-k kernel reads the freshly written cache."""
        assert self.wk_weights_proj is not None
        assert self.k_norm is not None

        kw, _ = self.wk_weights_proj(hidden_states)
        k_li = kw[:, : self.head_dim]
        k_li = self.k_norm(k_li).unsqueeze(1)
        k_li = k_li.view(-1, 1, self.head_dim)

        if HAS_TRITON:
            cos = cos.view(-1, self.qk_rope_head_dim)
            sin = sin.view(-1, self.qk_rope_head_dim)
            k_li = rope_forward_triton_siso(
                k_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            k_li_pe, k_li_nope = torch.split(
                k_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )

            cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
            sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

            k_li_pe = k_li_pe.unsqueeze(2)
            k_li_pe = torch_npu.npu_rotary_mul(k_li_pe, cos, sin)
            k_li_pe = k_li_pe.squeeze(2)

            k_li = torch.cat([k_li_pe, k_li_nope], dim=-1)  # [b*s,128]

        if self.enable_sparse_li_c8:
            k_li = k_li @ AscendSFAIndexerBackend.k_hadamard
            k_li, k_li_scale = torch_npu.npu_dynamic_quant(k_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
            k_li_scale = k_li_scale.to(self.c8_k_scale_cache_dtype)  # [b*s,]
            k_li_scale = k_li_scale.unsqueeze(-1)  # [b*s,1]
        else:
            k_li_scale = None

        return k_li, k_li_scale

    def _gather_cache_inputs(
        self,
        k_li: torch.Tensor,
        k_li_scale: torch.Tensor | None,
        indexer_metadata: AscendSFAIndexerMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Parallel-layout transforms applied to the k path output between
        ``forward_k`` and the cache write. Identity in the base layout; PCP
        all-gathers the prefill region across the CP group (reordering the
        slot mapping to the gathered layout), DSA-CP all-gathers the indexer
        k across the TP group (its padded slot mapping already covers the
        gathered layout)."""
        slot_mapping = indexer_metadata.slot_mapping
        if getattr(indexer_metadata, "dcp_sharded_indexer_enabled", False):
            if indexer_metadata.dcp_local_token_mask is None or indexer_metadata.dcp_local_slot_mapping is None:
                raise RuntimeError("DCP sharded indexer cache write requires rank-local token metadata.")
            local_mask = indexer_metadata.dcp_local_token_mask.to(device=k_li.device)
            k_li = k_li[local_mask]
            if k_li_scale is not None:
                k_li_scale = k_li_scale[local_mask]
            slot_mapping = indexer_metadata.dcp_local_slot_mapping
            assert slot_mapping.numel() == k_li.shape[0], (
                "DCP sharded indexer cache write requires one rank-local slot per rank-local K token: "
                f"tokens={k_li.shape[0]}, slots={slot_mapping.numel()}."
            )
        elif self._pcp_active:
            tensors = (k_li,) if k_li_scale is None else (k_li, k_li_scale)
            gathered_tensors, slot_mapping = _gather_prefill_cache_inputs(
                tensors, slot_mapping, indexer_metadata.num_decode_tokens
            )
            k_li = gathered_tensors[0]
            assert slot_mapping.numel() == k_li.shape[0], (
                "PCP indexer cache write requires one slot per gathered token: "
                f"tokens={k_li.shape[0]}, slots={slot_mapping.numel()}."
            )
            if k_li_scale is not None:
                k_li_scale = gathered_tensors[1]
        elif self._dsa_cp_active:
            # Serialized with respect to the main KV all-gather on purpose:
            # the indexer owns its cache-write pipeline, so it cannot join
            # SFA's fused collective the way the pre-refactor inline flow
            # did. The k and scale gathers are launched back-to-back and
            # waited together so they at least overlap each other.
            # TODO: re-fuse with the main KV all-gather (e.g. pass a
            # collective plan through the indexer metadata) if DSA-CP
            # throughput becomes a concern.
            k_li, k_handle = all_gather_async(k_li, get_tp_group(), async_op=True)
            scale_handle = None
            if self.enable_sparse_li_c8:
                assert k_li_scale is not None
                k_li_scale, scale_handle = all_gather_async(k_li_scale, get_tp_group(), async_op=True)
            if k_handle is not None:
                k_handle.wait()
            if scale_handle is not None:
                scale_handle.wait()
        return k_li, k_li_scale, slot_mapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_hidden_states: torch.Tensor,
        indexer_metadata: AscendSFAIndexerMetadata,
        compute_topk: bool = True,
    ) -> torch.Tensor | None:
        """Full indexer pipeline: k path -> cache write -> top-k selection.

        The k path output is persisted first because the selection kernel
        reads the freshly written cache. ``compute_topk=False`` (SFA layers
        sharing top-k indices) still runs the k path and the write so the
        cache stays up to date, and returns None.

        ``cos`` / ``sin`` come from SFA's metadata, not the indexer's own:
        DSA-CP shards them to the local token shard, matching the sharded
        inputs. ``indexer_metadata`` carries the indexer's own cache view
        plus the parallel-layout values injected by SFA."""
        k_li, k_li_scale = self.forward_k(k_hidden_states, cos, sin)
        k_li, k_li_scale, slot_mapping = self._gather_cache_inputs(k_li, k_li_scale, indexer_metadata)
        self.write_cache(k_li, k_li_scale, slot_mapping, indexer_attn_metadata=indexer_metadata)
        if not compute_topk:
            return None

        assert self.wk_weights_proj is not None
        assert self.wq_b is not None
        assert indexer_metadata.actual_seq_lengths_query is not None
        assert indexer_metadata.actual_seq_lengths_key is not None

        kw, _ = self.wk_weights_proj(hidden_states)
        weights = kw[:, self.head_dim :]
        if isinstance(q_c, tuple):
            q_c_tensor, q_c_scale = q_c
            q_c_tensor = q_c_tensor.view(-1, q_c_tensor.shape[-1])
            quant_matmul_kwargs = dict(
                bias=None,
                output_dtype=hidden_states.dtype,
            )
            if q_c_tensor.dtype == torch.float8_e4m3fn:
                if q_c_scale.dim() == 2:
                    q_c_scale = q_c_scale.view(q_c_scale.shape[0], -1, 2)
                quant_matmul_kwargs.update(
                    scale_dtype=torch_npu.float8_e8m0fnu,
                    pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                    group_sizes=[1, 1, getattr(self.wq_b.quant_method.quant_method, "group_size", 32)],
                )
            elif q_c_scale.dim() > 1 and q_c_scale.shape[-1] == 1:
                q_c_scale = q_c_scale.squeeze(dim=-1)
            q_li = torch_npu.npu_quant_matmul(
                q_c_tensor,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=q_c_scale,
                **quant_matmul_kwargs,
            )
        else:
            q_li, _ = self.wq_b(q_c)
        q_li = q_li.view(-1, self.n_head, self.head_dim)
        if HAS_TRITON:
            q_li = rope_forward_triton_siso(
                q_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            q_li_pe, q_li_nope = torch.split(
                q_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )

            q_li_pe = q_li_pe.unsqueeze(2)
            q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
            q_li_pe = q_li_pe.squeeze(2)
            q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)

        q_li_scale = None
        q_li_shape_ori = None
        if self.enable_sparse_li_c8:
            q_li_shape_ori = q_li.shape
            q_li = q_li @ AscendSFAIndexerBackend.q_hadamard
            q_li, q_li_scale = torch_npu.npu_dynamic_quant(q_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
            q_li_scale = q_li_scale.to(self.c8_k_scale_cache_dtype)  # [b*s,]

        sharded_topk = self._select_dcp_sharded_topk(
            q_li,
            q_li_scale,
            q_li_shape_ori,
            weights,
            indexer_metadata,
        )
        if sharded_topk is not None:
            return sharded_topk

        return DeviceOperator.indexer_select_post_process(
            q_li,
            q_li_scale,
            q_li_shape_ori,
            weights,
            self.k_cache.kv_cache,
            INDEXER_K_CACHE_SLOT,
            INDEXER_SCALE_CACHE_SLOT,
            indexer_metadata,
            indexer_metadata.actual_seq_lengths_query,
            indexer_metadata.actual_seq_lengths_key,
            self.enable_sparse_li_c8,
            self.use_torch_npu_lightning_indexer,
        )


class AscendSFAIndexerMetadataBuilder(AttentionMetadataBuilder[AscendSFAIndexerMetadata]):
    """Builds the metadata consumed by SFA indexer forwards.

    The indexer cache layer shares block ids with the main SFA cache group,
    so the slot mapping and block table mirror the ``*.attn`` layer's; the
    rope tables are rebuilt from the same positions via the shared helper.
    The slot mapping is emitted write-ready for the active parallel mode
    (full gather mapping under PCP). Variants with their own cache geometry
    override this construction to supply their layout's equivalents.
    """

    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Match the logical block size selected for BlockTable.
        self.kernel_block_size = select_common_block_size(kv_cache_spec.block_size, [AscendSFAIndexerBackend])
        parallel_config = vllm_config.parallel_config
        self._pcp_active = _get_int_config_attr(parallel_config, "prefill_context_parallel_size", 1) > 1
        self._dcp_sharded_indexer = enable_sfa_dcp_sharded_indexer(vllm_config)
        self._dcp_world_size = _get_int_config_attr(parallel_config, "decode_context_parallel_size", 1)
        self._dcp_interleave_size = _get_int_config_attr(parallel_config, "cp_kv_cache_interleave_size", 128)
        try:
            self._dcp_rank = get_dcp_group().rank_in_group
        except Exception:
            self._dcp_rank = 0

        # The incumbent DCP indexer cache is physically replicated. Before the
        # indexer became an independent backend, AscendSFADCPMetadataBuilder
        # temporarily replaced the DCP-local BlockTable/slot mapping with a
        # replicated indexer view. The independent builder must preserve that
        # contract itself; otherwise the native LI sees DCP-local/padded block
        # metadata while actual_seq_lengths_key remains global.
        #
        # PCP+DCP has an additional gathered-token ordering and is deliberately
        # left on its existing path here. The current key-domain candidate is
        # PCP1-only, and this repair targets the ordinary incumbent PCP1+DCP
        # contract without broadening the candidate surface.
        self._dcp_replicated_indexer = (
            self._dcp_world_size > 1
            and not self._dcp_sharded_indexer
            and not self._pcp_active
            and enable_sfa_dcp_replicated_indexer(vllm_config)
        )
        self._dcp_replicated_view_block_size = self.kernel_block_size
        self._dcp_blocks_per_phys_block = 1
        self._dcp_max_local_block_table_cols = 0
        self._dcp_replicated_block_table_buf: torch.Tensor | None = None
        self._dcp_replicated_col_idx: torch.Tensor | None = None
        self._dcp_replicated_slot_mapping_buf: torch.Tensor | None = None
        if self._dcp_replicated_indexer:
            if kv_cache_spec.block_size % self._dcp_replicated_view_block_size != 0:
                raise RuntimeError(
                    "SFA indexer replicated view requires the KV cache block size "
                    f"({kv_cache_spec.block_size}) to be divisible by "
                    f"{self._dcp_replicated_view_block_size}."
                )
            self._dcp_blocks_per_phys_block = kv_cache_spec.block_size // self._dcp_replicated_view_block_size
            self._dcp_max_local_block_table_cols = (
                cdiv(
                    vllm_config.model_config.max_model_len,
                    kv_cache_spec.block_size * self._dcp_world_size,
                )
                * self._dcp_blocks_per_phys_block
            )
            max_replicated_cols = self._dcp_max_local_block_table_cols * self._dcp_world_size
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            max_num_input_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self._dcp_replicated_block_table_buf = torch.empty(
                (max_num_reqs, max_replicated_cols),
                dtype=torch.int32,
                device=device,
            )
            self._dcp_replicated_col_idx = torch.arange(max_replicated_cols, dtype=torch.int32, device=device)
            self._dcp_replicated_slot_mapping_buf = torch.empty(max_num_input_tokens, dtype=torch.int32, device=device)

    def _build_dcp_replicated_block_table(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        if self._dcp_replicated_block_table_buf is None or self._dcp_replicated_col_idx is None:
            raise RuntimeError("DCP replicated indexer buffers are not initialized.")
        local_cols = min(block_table.shape[1], self._dcp_max_local_block_table_cols)
        replicated_cols = local_cols * self._dcp_world_size
        out = self._dcp_replicated_block_table_buf[:num_reqs, :replicated_cols]
        col_idx = self._dcp_replicated_col_idx[:replicated_cols]
        blocks_per_phys = self._dcp_blocks_per_phys_block
        local_col_idx = (
            col_idx // (self._dcp_world_size * blocks_per_phys) * blocks_per_phys + col_idx % blocks_per_phys
        )
        rank_in_replicated_view = (col_idx // blocks_per_phys) % self._dcp_world_size
        local_logical_blocks = torch.index_select(block_table[:num_reqs, :local_cols], 1, local_col_idx)
        if blocks_per_phys == 1:
            replicated_blocks = local_logical_blocks * self._dcp_world_size + rank_in_replicated_view
        else:
            local_sub_blocks = local_logical_blocks % blocks_per_phys
            local_phys_blocks = local_logical_blocks // blocks_per_phys
            replicated_blocks = (
                local_phys_blocks * self._dcp_world_size + rank_in_replicated_view
            ) * blocks_per_phys + local_sub_blocks
        valid_req_mask = (seq_lens[:num_reqs].to(device=self.device) > 0).to(replicated_blocks.dtype).view(-1, 1)
        out.copy_(replicated_blocks * valid_req_mask)
        return out

    def _build_dcp_replicated_slot_mapping(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        if self._dcp_replicated_slot_mapping_buf is None:
            raise RuntimeError("DCP replicated indexer slot buffer is not initialized.")
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        num_actual_tokens = min(common_attn_metadata.num_actual_tokens, num_input_tokens)
        out = self._dcp_replicated_slot_mapping_buf[:num_input_tokens]
        out.fill_(-1)
        if num_actual_tokens == 0:
            return out
        query_lens = (
            common_attn_metadata.query_start_loc[1 : num_reqs + 1] - common_attn_metadata.query_start_loc[:num_reqs]
        )
        req_indices = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int32, device=self.device),
            query_lens.to(device=self.device),
            output_size=num_input_tokens,
        )[:num_actual_tokens]
        num_actual_tokens = min(num_actual_tokens, req_indices.shape[0])
        req_indices = req_indices[:num_actual_tokens]
        positions = common_attn_metadata.positions[:num_actual_tokens].to(device=self.device, dtype=torch.int32)
        logical_block_idx = positions // self._dcp_replicated_view_block_size
        block_offsets = positions % self._dcp_replicated_view_block_size
        table_indices = req_indices * block_table.shape[1] + logical_block_idx
        block_numbers = block_table.flatten()[table_indices]
        out[:num_actual_tokens] = block_numbers * self._dcp_replicated_view_block_size + block_offsets
        return out

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendSFAIndexerMetadata:
        # common_prefix_len / fast_build are unused; kept for API compatibility.
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        if self._pcp_active:
            # PCP writes cover the gathered prefill region too, which
            # requires the full slot mapping.
            slot_mapping = common_attn_metadata.slot_mapping
        else:
            slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        if self._dcp_replicated_indexer:
            block_table = self._build_dcp_replicated_block_table(
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.seq_lens,
                num_reqs,
            )
            slot_mapping = self._build_dcp_replicated_slot_mapping(common_attn_metadata, block_table)
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        block_size = self.kernel_block_size

        cos, sin = get_cos_and_sin_mla(input_positions, use_cache=True)

        if get_ascend_config().c8_reshape_optim_enabled:
            torch.ops._C_ascend.store_kv_block_metadata(
                slot_mapping,
                common_attn_metadata.group_len,
                common_attn_metadata.group_key_idx,
                common_attn_metadata.group_key_cache_idx,
                block_size,
            )

        query_start = common_attn_metadata.query_start_loc[: num_reqs + 1]
        query_lens = query_start[1:] - query_start[:-1]
        context_lens = common_attn_metadata.seq_lens[:num_reqs] - query_lens
        li_cum_query_lens = None
        local_visible_by_query = None
        dcp_local_block_table = None
        dcp_local_slot_mapping = None
        dcp_local_token_mask = None
        all_local_rows_active = False
        if self._dcp_sharded_indexer:
            max_lens_cpu = _get_dcp_metadata_max_lens_from_cpu(common_attn_metadata, num_reqs)
            if max_lens_cpu is None:
                max_query_len = int(query_lens.max().item())
                max_seq_len = int(common_attn_metadata.seq_lens[:num_reqs].max().item())
            else:
                max_query_len, max_seq_len = max_lens_cpu
            row_offsets = torch.arange(
                1,
                max_query_len + 1,
                dtype=context_lens.dtype,
                device=context_lens.device,
            )
            global_visible = context_lens.unsqueeze(1) + row_offsets.unsqueeze(0)
            row_mask = row_offsets.unsqueeze(0) <= query_lens.unsqueeze(1)
            global_visible = global_visible[row_mask]
            local_visible_by_query = dcp_local_visible_counts(
                global_visible,
                self._dcp_rank,
                self._dcp_world_size,
                self._dcp_interleave_size,
            ).to(torch.int32)
            # For a single prefill request, both phase and sequence length are
            # available on CPU without a device synchronization.  Once the
            # cached context spans at least one full DCP interleave cycle, every
            # rank has non-empty local visibility for every newly scheduled row.
            # Other/mixed/decode geometries retain the sentinel mask below.
            is_prefilling_cpu = getattr(common_attn_metadata, "is_prefilling", None)
            seq_lens_cpu_upper_bound = getattr(common_attn_metadata, "seq_lens_cpu_upper_bound", None)
            query_start_loc_cpu = getattr(common_attn_metadata, "query_start_loc_cpu", None)
            if (
                num_reqs == 1
                and is_prefilling_cpu is not None
                and seq_lens_cpu_upper_bound is not None
                and query_start_loc_cpu is not None
                and bool(is_prefilling_cpu[0])
            ):
                query_len_cpu = int(query_start_loc_cpu[1] - query_start_loc_cpu[0])
                context_len_cpu = int(seq_lens_cpu_upper_bound[0]) - query_len_cpu
                all_local_rows_active = context_len_cpu >= (self._dcp_world_size * self._dcp_interleave_size)
            li_cum_query_lens = torch.arange(
                1,
                local_visible_by_query.numel() + 1,
                dtype=torch.int32,
                device=local_visible_by_query.device,
            )
            dcp_local_token_mask = (
                (input_positions // self._dcp_interleave_size) % self._dcp_world_size
            ) == self._dcp_rank
            actual_token_mask = torch.arange(num_input_tokens, device=input_positions.device) < max(
                0, min(common_attn_metadata.num_actual_tokens, num_input_tokens)
            )
            dcp_local_token_mask = dcp_local_token_mask & actual_token_mask
            local_positions = input_positions[dcp_local_token_mask]
            dcp_block_cols_divisor = self.kernel_block_size * self._dcp_world_size
            local_block_cols = max(
                1,
                (max_seq_len + dcp_block_cols_divisor - 1) // dcp_block_cols_divisor,
            )
            dcp_local_request_block_table = common_attn_metadata.block_table_tensor[:num_reqs, :local_block_cols]
            token_req_indices = torch.repeat_interleave(
                torch.arange(num_reqs, dtype=torch.int64, device=input_positions.device),
                query_lens.to(device=input_positions.device),
                output_size=num_input_tokens,
            )
            # sparse_mode=0 represents every query token as its own LI
            # pseudo-row.  PA_BSND requires block_table.dim(0) to match that
            # pseudo-row batch, while all rows of one request share the same
            # rank-local physical pages (the geometry proven by R5).
            dcp_local_block_table = dcp_local_request_block_table[token_req_indices]
            local_req_indices = token_req_indices[dcp_local_token_mask]
            local_indices = (
                local_positions // (self._dcp_interleave_size * self._dcp_world_size) * self._dcp_interleave_size
            )
            local_indices = local_indices + local_positions % self._dcp_interleave_size
            local_block_idx = local_indices // self.kernel_block_size
            local_block_offsets = local_indices % self.kernel_block_size
            dcp_local_slot_mapping = (
                dcp_local_request_block_table[local_req_indices, local_block_idx.to(torch.long)]
                * self.kernel_block_size
                + local_block_offsets
            ).to(slot_mapping.dtype)

        return AscendSFAIndexerMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=common_attn_metadata.seq_lens[:num_reqs],
            cum_query_lens=common_attn_metadata.query_start_loc[1 : num_reqs + 1],
            block_table=block_table,
            sin=sin[:num_input_tokens],
            cos=cos[:num_input_tokens],
            block_size=block_size,
            group_len=common_attn_metadata.group_len,
            group_key_idx=common_attn_metadata.group_key_idx,
            group_key_cache_idx=common_attn_metadata.group_key_cache_idx,
            dcp_sharded_indexer_enabled=self._dcp_sharded_indexer,
            dcp_rank=self._dcp_rank,
            dcp_world_size=self._dcp_world_size,
            dcp_interleave_size=self._dcp_interleave_size,
            dcp_local_block_table=dcp_local_block_table,
            dcp_local_slot_mapping=dcp_local_slot_mapping,
            dcp_local_token_mask=dcp_local_token_mask,
            request_query_lens=query_lens.to(torch.int32),
            request_context_lens=context_lens.to(torch.int32),
            local_visible_by_query=local_visible_by_query,
            li_cum_query_lens=li_cum_query_lens,
            all_local_rows_active=all_local_rows_active,
        )
