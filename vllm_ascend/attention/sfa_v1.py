from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, TypeVar

import torch
import torch_npu
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_dcp_group, get_pcp_group, get_tensor_model_parallel_world_size, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.model_executor.utils import replace_parameter
from vllm.triton_utils import HAS_TRITON
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import (
    AttentionBackend,  # type: ignore
    AttentionCGSupport,
    MLAAttentionImpl,
)
from vllm.v1.attention.ops.pcp import _gather_prefill_cache_inputs  # type: ignore[import-not-found]
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.common_cp import (
    get_pcp_num_replicated_tokens,
    is_pcp_decode_sharding_enabled,
)
from vllm_ascend.attention.context_parallel.sfa_dcp_utils import _index_select_kv_blocks, as_byte_view
from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from vllm_ascend.attention.utils import (
    MLAPO_MAX_SUPPORTED_TOKENS,
    SFA_QSFA_TILE_SIZE,
    AscendCommonAttentionMetadata,
    PreprocessType,
    ascend_chunked_prefill_workspace_size,
    get_sfa_qsfa_packed_head_dim,
    get_tq_fused_slot_bytes,
    get_tq_packed_bytes,
    maybe_save_kv_layer_to_connector,
    notify_kv_cache_written,
    split_decodes_and_prefills,
    trans_rope_weight,
    transdata,
    uses_pcp_sharded_layout,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
    attention_transfer_window,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    OFFLOAD_K_CACHE_NPU_INDEX,
    OFFLOAD_KV_CACHE_TUPLE_LEN,
    OFFLOAD_V_CACHE_NPU_INDEX,
)
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.triton.dcp.sfa_dcp_query import prepare_query_head_major, unpack_query
from vllm_ascend.ops.triton.pcp_kv_cache import copy_pcp_kv_cache
from vllm_ascend.quantization.methods import (
    AscendW8A8DynamicLinearMethod,
    AscendW8A8LinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.quantization.methods.kv_cache.turboquant.latent import TurboQuantLatent
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_ND,
    ACL_FORMAT_FRACTAL_NZ,
    dispose_layer,
    enable_pcp_o_proj_weight_sharding,
    enable_sfa_dcp_force_tmajor_restore,
    enable_sfa_dcp_replicated_indexer,
    enable_sp,
    is_mtp_layer,
    is_rl_weight_update_enabled,
    maybe_trans_nz,
    vllm_version_is,
)
from vllm_ascend.weight_switch import ParallelOProjection, WeightSwitchConfig

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_ascend.worker.npu_input_batch import NPUInputBatch


# NoPE sparse MLA operator helpers.
SMLA_METADATA_SIZE = 1024
# GLM5Next's SFA path carries no learnable attention sink, but SparseFlashMla
# still requires a per-head float32 sinks tensor. This is the placeholder value
# the path has always used; whether -inf is the correct "no sink" value is a
# separate question, tracked outside this change.
SMLA_DEFAULT_SINK_VALUE = 1.0
# TQ4 is a quantization type of the KV-quant sparse flash attention operator
# (key_quant_mode=value_quant_mode=3, 386-byte KV slot), not a separate operator.
TQ_QUANT_MODE = 3
TQ_TILE_SIZE = 128


def generate_smla_plan(metadata, num_heads, head_dim, topk, cu_seqlens_q, topk_length):
    """Derive the operator's core-split plan for one specific set of inputs.

    ``cu_seqlens_q`` and ``topk_length`` have to be the very tensors the
    operator call will receive: the plan tells the kernel how many query rows to
    walk and how far into each row's indices to go, so a plan built over
    different ones makes it index past what it was handed. The metadata
    interface enforces the second half of that itself - with ori_mask_mode 0 and
    a non-zero ori_topk it rejects an absent ori_topk_length.
    """
    return sparse_flash_mla_metadata(
        num_heads_q=num_heads,
        num_heads_kv=1,
        head_dim=head_dim,
        cu_seqlens_q=cu_seqlens_q,
        seqused_ori_kv=metadata.seq_lens,
        ori_topk_length=topk_length,
        batch_size=metadata.seq_lens.numel(),
        # These are scheduler CPU scalars. Passing device max() results here
        # would force a host synchronization for each metadata build.
        max_seqlen_q=metadata.max_query_len,
        max_seqlen_ori_kv=metadata.max_seq_len,
        max_seqlen_cmp_kv=0,
        ori_topk=topk,
        cmp_topk=0,
        cmp_ratio=1,
        # No Mask, which is what the sparse ori_kv scenario asks for: the
        # selected indices are themselves the mask, since the indexer only ever
        # offers causally valid tokens. Anything other than 0 also makes the
        # kernel skip its softmax initialisation whenever no sequence has a
        # query longer than its KV - which is every ordinary decode step - and
        # a sparse gather does not write every accumulator slot, so the skipped
        # rows keep whatever the previous step left in them.
        ori_mask_mode=0,
        # Required to be 0 while cmp_kv is absent.
        cmp_mask_mode=0,
        # Only ori_mask_mode 4 may carry a bounded window on this product line.
        ori_win_left=-1,
        ori_win_right=-1,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=False,
        device=str(metadata.seq_lens.device),
    )


def build_smla_metadata(metadata, buffer, num_heads, head_dim, topk, cu_seqlens_q, topk_length):
    generated = generate_smla_plan(metadata, num_heads, head_dim, topk, cu_seqlens_q, topk_length)
    if generated.numel() != buffer.numel():
        # The persistent buffer is sized by the operator's fixed [1024] contract.
        # A generated plan of any other size means this build does not match the
        # operator this buffer was allocated for; say so here rather than letting
        # copy_ decide.
        raise ValueError(f"Sparse MLA plan must contain {buffer.numel()} int32 values, got {generated.numel()}.")
    buffer.copy_(generated)
    metadata.smla_metadata = buffer


def _view_cache_as_operator_pages(cache: torch.Tensor, block_size: int) -> torch.Tensor:
    """Expose oversized contiguous storage pages at operator block granularity."""
    storage_block_size = cache.shape[1]
    if storage_block_size == block_size:
        return cache
    if storage_block_size % block_size:
        raise ValueError(
            f"Sparse MLA storage block size {storage_block_size} is not divisible by operator block size {block_size}."
        )
    try:
        return cache.view(-1, block_size, *cache.shape[2:])
    except RuntimeError as err:
        raise ValueError("Sparse MLA oversized storage pages must support a zero-copy operator-page view.") from err


def sparse_mla(query, cache, indices, metadata, scale):
    """Attend to original latent KV, using the platform's NoPE operator."""
    cache = _view_cache_as_operator_pages(cache, metadata.block_size)
    if metadata.smla_metadata is not None:
        # The A5 DMA merges adjacent columns. Preserve the selected set while
        # sorting token positions and moving invalid padding to the end.
        sentinel = torch.iinfo(torch.int32).max
        sorted_indices = torch.where(indices >= 0, indices, sentinel).sort(dim=-1).values
        sorted_indices = torch.where(sorted_indices == sentinel, -1, sorted_indices)
        if metadata.smla_sinks is None:
            raise RuntimeError("Sparse MLA requires persistent sinks owned by SparseMLAMetadataState.")
        # MTP and shared indexers can reuse selections while positions advance.
        # Count the actual left-aligned prefix on every call, including target
        # FULL graphs whose query shape still matches the metadata capacity.
        topk_length = metadata.smla_topk_length[: query.shape[0]]
        metadata.smla_topk_length.zero_()
        # The builder refreshes this persistent mask outside graph capture.
        # A Python token count used here would freeze at the captured batch size.
        valid = metadata.smla_valid_mask[: query.shape[0]]
        topk_length.copy_((sorted_indices >= 0).sum(dim=-1, dtype=torch.int32).masked_fill(~valid, 0))
        cu_seqlens_q = metadata.query_start_loc
        if query.shape[0] != metadata.smla_topk_length.shape[0]:
            # Eager and piecewise execution trim graph-padding query rows.
            cu_seqlens_q = cu_seqlens_q.clamp(max=query.shape[0])
        build_smla_metadata(
            metadata,
            metadata.smla_metadata,
            query.shape[1],
            query.shape[2],
            sorted_indices.shape[-1],
            cu_seqlens_q,
            topk_length,
        )
        result = sparse_flash_mla(
            query.contiguous(),
            ori_kv=cache,
            ori_sparse_indices=sorted_indices,
            ori_block_table=metadata.block_table,
            cu_seqlens_q=cu_seqlens_q,
            seqused_ori_kv=metadata.seq_lens,
            ori_topk_length=topk_length,
            sinks=metadata.smla_sinks,
            metadata=metadata.smla_metadata,
            softmax_scale=scale,
            cmp_ratio=1,
            # Must match the mode the plan above was generated with.
            ori_mask_mode=0,
            cmp_mask_mode=0,
            ori_win_left=-1,
            ori_win_right=-1,
            layout_q="TND",
            layout_kv="PA_BBND",
            topk_value_mode=1,
            return_softmax_lse=False,
        )
    else:
        result = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=query.contiguous(),
            key=cache,
            value=cache,
            sparse_indices=indices,
            scale_value=scale,
            sparse_block_size=1,
            block_table=metadata.block_table,
            actual_seq_lengths_query=metadata.query_start_loc[1:].to(torch.int32),
            actual_seq_lengths_kv=metadata.seq_lens.to(torch.int32),
            query_rope=None,
            key_rope=None,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )
    return result[0]


class SparseMLAMetadataState:
    """Persistent operator buffers for NoPE within the shared SFA builder.

    Sparse MLA fills lengths and the plan after receiving selected indices.
    Pool construction and scoring remain outside operator dispatch.
    """

    def __init__(self, kv_cache_spec, vllm_config, device, kernel_block_size=128, num_heads=None):
        block_size = kv_cache_spec.block_size
        if block_size <= 0 or block_size % kernel_block_size:
            raise ValueError("Sparse MLA block size must be a positive multiple of the SFA kernel block size.")
        self.split = block_size // kernel_block_size
        self.use_smla = get_current_hardware_profile().device_adaptor_family == DeviceAdaptorFamily.FP8_OPTIMIZED
        if self.use_smla:
            config = get_ascend_config()
            self.use_smla = not (config.enable_sparse_sfa_c8 or config.enable_sparse_sfa_turboquant)
        self.block_size = kernel_block_size
        self.table_stride = 1
        cache_block_size = vllm_config.cache_config.block_size
        expand_factor = max(cache_block_size // kernel_block_size, 1)
        table_width = cdiv(vllm_config.model_config.max_model_len, cache_block_size) * expand_factor
        capture_sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
        table_rows = max([vllm_config.scheduler_config.max_num_seqs + 1, *capture_sizes])
        self.block_table_buffer = torch.empty(table_rows, table_width, dtype=torch.int32, device=device)
        if self.use_smla:
            config = vllm_config.model_config.hf_text_config
            self.num_heads = (
                config.num_attention_heads // vllm_config.parallel_config.tensor_parallel_size
                if num_heads is None
                else num_heads
            )
            self.metadata_buffer = torch.empty(SMLA_METADATA_SIZE, dtype=torch.int32, device=device)
            self.length_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                1,
                dtype=torch.int32,
                device=device,
            )
            self.valid_mask_buffer = torch.empty_like(self.length_buffer, dtype=torch.bool)
            # ACL Graph replay reads the addresses captured on the first run,
            # so every tensor the operator consumes has to outlive the capture.
            # Allocate the sinks once here, next to the other persistent
            # operator buffers, and keep it for this state's lifetime.
            self.sinks = torch.full(
                (self.num_heads,),
                SMLA_DEFAULT_SINK_VALUE,
                dtype=torch.float32,
                device=device,
            )

    def prepare(self, metadata):
        expanded = metadata.block_table
        if expanded.shape[1] % self.split:
            raise ValueError("Sparse MLA received a partially expanded SFA block table.")
        width = expanded.shape[1] // self.table_stride
        if width > self.block_table_buffer.shape[1]:
            raise ValueError("Sparse MLA block table exceeds its persistent buffer.")
        table = self.block_table_buffer[: expanded.shape[0], :width]
        if self.table_stride == 1:
            table.copy_(expanded)
        else:
            torch.div(expanded[:, :: self.table_stride], self.table_stride, rounding_mode="floor", out=table)
        metadata.block_table = table
        metadata.block_size = self.block_size
        if self.use_smla:
            positions = metadata.positions
            if positions.numel() > self.length_buffer.shape[0]:
                raise ValueError("Sparse MLA token count exceeds its persistent top-k buffer.")
            mask = self.valid_mask_buffer[: positions.numel()]
            mask.fill_(True)
            if metadata.num_actual_tokens < positions.numel():
                mask[metadata.num_actual_tokens :].zero_()
            metadata.smla_valid_mask = mask
            metadata.smla_topk_length = self.length_buffer[: positions.numel()]
            metadata.smla_metadata = self.metadata_buffer
            metadata.smla_sinks = self.sinks
        return metadata


# npu_transpose_batchmatmul rejects operand dimensions >= 65536
TRANSPOSE_BMM_MAX_SUPPORTED_DIM = 65536


def _get_indexer_types(configs: tuple[Any, ...]) -> Any | None:
    for config in configs:
        if config is None:
            continue
        indexer_types = getattr(config, "indexer_types", None)
        if indexer_types is not None:
            return indexer_types
    return None


def _has_shared_indexer_layers(configs: tuple[Any, ...]) -> bool:
    indexer_types = _get_indexer_types(configs)
    if indexer_types is None:
        return False
    return any(isinstance(indexer_type, str) and indexer_type.lower() == "shared" for indexer_type in indexer_types)


def _get_config_bool(configs: tuple[Any, ...], attr: str) -> bool:
    for config in configs:
        if config is not None and hasattr(config, attr):
            return bool(getattr(config, attr))
    return False


class AscendSFABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_SFA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        # vLLM #60271 added the batch-invariance support check in
        # `init_attn_backend` in both supported vLLM revisions.
        return True

    @staticmethod
    def get_builder_cls():
        if get_ascend_config().sparse_kv_offload_config.enabled:
            from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadMetadataBuilder

            return AscendSFAKVOffloadMetadataBuilder
        vllm_config = get_current_vllm_config()
        if enable_sfa_dcp_replicated_indexer(vllm_config):
            from vllm_ascend.attention.context_parallel.sfa_cp import (
                AscendSFADCPMetadataBuilder,
                AscendSFAPCPDCPMetadataBuilder,
            )

            if vllm_config.parallel_config.prefill_context_parallel_size > 1:
                return AscendSFAPCPDCPMetadataBuilder
            return AscendSFADCPMetadataBuilder
        return AscendSFAMetadataBuilder

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
    def get_impl_cls() -> type["AscendSFAImpl"]:
        if get_ascend_config().sparse_kv_offload_config.enabled:
            from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl

            return AscendSFAKVOffloadImpl
        return AscendSFAImpl

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
        return [128]


@dataclass
class AscendSFAMetadata:
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|
    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor | None
    cum_query_lens: torch.Tensor
    block_table: torch.Tensor
    sin: torch.Tensor | None
    cos: torch.Tensor | None

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.
    pcp_slot_mapping: torch.Tensor | None = None
    # Resolved by the builder; execution must not infer layout from tensor sizes.
    uses_pcp_sharded_layout: bool = False
    pcp_prolog_local_slots: torch.Tensor | None = None
    pcp_prolog_global_slots: torch.Tensor | None = None
    # All PCP ranks must join prefill KV gathers even when a rank has only padding.
    pcp_has_global_prefill: bool = False
    # The dimension of the attention heads
    head_dim: int | None = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill
    reshape_cache_event: torch.npu.Event = None
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    block_size: int = 0
    # Request identity for the Sparse KV offload resident LRU; only populated
    # by AscendSFAKVOffloadMetadataBuilder.
    req_ids_tensor: torch.Tensor | None = None
    token_to_req: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    max_query_len: int = 0
    max_seq_len: int = 0
    smla_metadata: torch.Tensor | None = None
    smla_topk_length: torch.Tensor | None = None
    smla_valid_mask: torch.Tensor | None = None
    smla_sinks: torch.Tensor | None = None


class DCPGatherContext(NamedTuple):
    """State needed to finish an async DCP query or KV all-gather."""

    gathered: torch.Tensor
    handle: torch.distributed.Work | None
    restore_perm: tuple[int, ...] | None
    split_sizes: tuple[int, ...]
    query_prepared: bool = False


@dataclass
class DCPContext:
    # Keep the complete main-cache mapping. Consumers that operate on the
    # rank-local model input take a view of the required prefix.
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    kv_gather_block_ids: torch.Tensor | None = None
    kv_gather_block_table: torch.Tensor | None = None
    gather_context: DCPGatherContext | None = None


@dataclass
class AscendSFADCPMetadata(AscendSFAMetadata):
    """SFA metadata fields used only by the DCP execution path."""

    dcp_context: DCPContext | None = None


M = TypeVar("M", bound=AscendSFAMetadata)


def needs_pcp_kv_gather(attn_metadata: AscendSFAMetadata, is_decode_sharded: bool) -> bool:
    """Whether PCP ranks must exchange this step's KV.

    Only replicated decode-only steps already hold identical KV on every
    rank, including graph padding. Ranks without local prefill still join the
    gather when another rank has one. A step built without a PCP context
    (e.g. a replicated draft decode) keeps an unexpanded slot mapping.
    """
    if not attn_metadata.uses_pcp_sharded_layout:
        return False
    return is_decode_sharded or attn_metadata.num_prefills > 0 or attn_metadata.pcp_has_global_prefill


def _int64_kv_slots(slots: torch.Tensor, attn_metadata: M) -> torch.Tensor:
    """Convert the KV slot mapping to int64 once per scheduling step.

    ``npu_kv_rmsnorm_rope_cache`` requires int64 cache indices while the SFA
    metadata carries int32 slots. Every layer of a step shares the same slot
    tensor, so cache the converted copy on the metadata object instead of
    re-casting it inside each layer's ``exec_kv`` (one Cast kernel per step
    instead of one per layer).
    """
    if slots.dtype == torch.int64:
        return slots
    cached = getattr(attn_metadata, "kv_slots_i64", None)
    if cached is None or cached[0] is not slots:
        cached = (slots, slots.to(torch.int64))
        attn_metadata.kv_slots_i64 = cached  # type: ignore[attr-defined]
    return cached[1]


class AscendSFAMetadataBuilder(MLACommonMetadataBuilder[AscendSFAMetadata]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendSFAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls if metadata_cls is not None else AscendSFAMetadata,
            supports_dcp_with_varlen,
        )

        # Match the logical block size selected for BlockTable.
        self.kernel_block_size = select_common_block_size(kv_cache_spec.block_size, [AscendSFABackend])

        layer = vllm_config.compilation_config.static_forward_context[layer_names[0]]
        self.nope = layer.qk_rope_head_dim == 0
        self.nope_states: dict[int | None, SparseMLAMetadataState] = {}

        self.use_pcp = vllm_config.parallel_config.prefill_context_parallel_size > 1
        self.is_pcp_decode_sharded = is_pcp_decode_sharding_enabled(vllm_config)
        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )
        self.reorder_batch_threshold = self.decode_threshold
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
        return ascend_chunked_prefill_workspace_size(vllm_config)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendSFAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        speculative_config = vllm_config.speculative_config
        if (
            speculative_config is not None
            and speculative_config.method == "dspark"
            and getattr(speculative_config, "enable_adaptive_verification", False)
        ):
            return AttentionCGSupport.ALWAYS
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch", scheduler_output: "SchedulerOutput") -> bool:
        # No need to reorder for Ascend SFA
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendSFAMetadata:
        # common_prefix_len / fast_build are unused; kept for API compatibility.
        metadata = self._build_with_metadata_view(
            common_attn_metadata,
            lambda: self._build(common_attn_metadata, draft_index=None, pcp_context=common_attn_metadata.pcp_context),
        )
        if (
            self.use_pcp
            and common_attn_metadata.pcp_context is not None
            and needs_pcp_kv_gather(metadata, self.is_pcp_decode_sharded)
        ):
            assert metadata.pcp_slot_mapping is not None
            group = get_pcp_group()
            num_tokens = metadata.num_input_tokens
            rank_slots = metadata.pcp_slot_mapping[: group.world_size * num_tokens].view(group.world_size, num_tokens)
            num_replicated_tokens = get_pcp_num_replicated_tokens(
                metadata.num_decode_tokens, self.is_pcp_decode_sharded
            )
            local_slots = rank_slots[group.rank_in_group].contiguous()
            if num_replicated_tokens and group.rank_in_group != 0:
                # Replicated decode slots are masked outside rank 0, but each
                # rank still writes its locally computed decode KV.
                local_slots = torch.cat((rank_slots[0, :num_replicated_tokens], local_slots[num_replicated_tokens:]))
            metadata.pcp_prolog_local_slots = local_slots
            metadata.pcp_prolog_global_slots = rank_slots[:, num_replicated_tokens:].reshape(-1)
        return metadata

    def build_for_drafting(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
        **kwargs,
    ) -> AscendSFAMetadata:
        return self._build_with_metadata_view(
            common_attn_metadata,
            lambda: self._build(
                common_attn_metadata,
                draft_index=draft_index,
            ),
        )

    def _build_with_metadata_view(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        build_metadata: Callable[[], AscendSFAMetadata],
    ) -> AscendSFAMetadata:
        """Build against the default KV-cache view.

        Distributed layouts can override this hook to expose a temporary view
        while reusing the complete SFA metadata construction flow.
        """
        return build_metadata()

    def _build(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int | None = None,
        pcp_context: Any | None = None,
    ) -> AscendSFAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_input_tokens = common_attn_metadata.num_input_tokens
        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        pcp_slot_mapping = common_attn_metadata.slot_mapping
        slot_mapping = pcp_slot_mapping[:num_input_tokens]
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()

        block_size = self.kernel_block_size

        cum_query_lens = common_attn_metadata.query_start_loc[1 : num_reqs + 1]
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]

        # Prefer _seq_lens_cpu (always available, updated during draft
        # iterations) over seq_lens_cpu (None in async spec decode mode).
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]
        elif self.nope:
            # MTP accepted counts are device-resident. NoPE operators use
            # device lengths and scheduler upper bounds, so need no CPU copy.
            seq_lens_cpu = None
        else:
            seq_lens_cpu = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        if self.nope:
            cos, sin = None, None
        else:
            cos, sin = get_cos_and_sin_mla(input_positions, use_cache=(draft_index is None))

        metadata = self.metadata_cls(  # type: ignore
            num_input_tokens=num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            cum_query_lens=cum_query_lens,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            slot_mapping=slot_mapping,
            pcp_slot_mapping=pcp_slot_mapping,
            uses_pcp_sharded_layout=(
                pcp_context is not None and uses_pcp_sharded_layout(common_attn_metadata, self.use_pcp)
            ),
            pcp_has_global_prefill=bool(pcp_context is not None and pcp_context.global_batch.is_prefilling_np.any()),
            head_dim=self.model_config.get_head_size(),
            attn_mask=self.attn_mask_builder.get_attention_mask(common_attn_metadata.causal, self.model_config),
            attn_state=common_attn_metadata.attn_state,
            block_table=block_table,
            sin=None if sin is None else sin[:num_input_tokens],
            cos=None if cos is None else cos[:num_input_tokens],
            positions=input_positions,
            query_start_loc=common_attn_metadata.query_start_loc[: num_reqs + 1],
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            block_size=block_size,
        )
        (
            metadata.num_decodes,
            metadata.num_prefills,
            metadata.num_decode_tokens,
            _,
        ) = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            treat_short_extends_as_decodes=not self.use_pcp,
        )
        if self.nope:
            self._prepare_nope_metadata(metadata, draft_index)
        return metadata

    def _prepare_nope_metadata(self, metadata: AscendSFAMetadata, draft_index: int | None) -> None:
        if draft_index not in self.nope_states:
            self.nope_states[draft_index] = SparseMLAMetadataState(
                self.kv_cache_spec, self.vllm_config, self.device, self.kernel_block_size
            )
        self.nope_states[draft_index].prepare(metadata)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        **kwargs: Any,
    ) -> AscendSFAMetadata:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in {AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding}:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError("Currently we only support building dummy metadata for DecodeOnly state")

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendSFAImpl(MLAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # ``W_UV``/``W_UK_T`` are injected during ``process_weights_after_loading``
    # through ``replace_parameter``, which is ``setattr``-based and therefore
    # invisible to static analysis. These declarations are what let mypy resolve
    # the attributes at every use site; without them ``pre-commit`` fails with
    # `Cannot determine type of "W_UK_T"  [has-type]`.
    W_UV: torch.Tensor
    W_UK_T: torch.Tensor

    # A replicated MTP draft may inherit a PCP target's non-trivial interleave
    # value. With DCP disabled it does not change the draft KV-cache layout.
    supports_mtp_with_cp_non_trivial_interleave_size: bool = True

    supports_pcp: ClassVar[bool] = True
    supports_dcp: bool = True
    can_return_lse_for_decode: bool = True
    o_proj_full_pools: ClassVar[dict[Any, torch.Tensor]] = {}
    o_proj_weight_switch_pool_key = "sfa_pcp_o_proj"

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # MLA Args
        self.q_lora_rank = kwargs["q_lora_rank"]
        self.kv_lora_rank = kwargs["kv_lora_rank"]
        self.qk_nope_head_dim = kwargs["qk_nope_head_dim"]
        self.qk_rope_head_dim = kwargs["qk_rope_head_dim"]
        self.qk_head_dim = kwargs["qk_head_dim"]
        self.v_head_dim = kwargs["v_head_dim"]
        self.q_proj = kwargs["q_proj"] if self.q_lora_rank is None else kwargs["q_b_proj"]
        self.fused_qkv_a_proj = kwargs.get("fused_qkv_a_proj")
        self.kv_b_proj = kwargs["kv_b_proj"]
        self.o_proj = kwargs["o_proj"]
        self.indexer = kwargs["indexer"]
        self.g_proj = kwargs.get("g_proj")
        # NoPE sparse layers use the same SFA template and native prefill;
        # they do not need an upstream dense-MHA prefill backend.
        if self.qk_rope_head_dim == 0:
            self.supports_dense_mha_prefill = False
        self.kv_a_proj_with_mqa = kwargs.get("kv_a_proj_with_mqa")
        self.kv_a_layernorm = kwargs.get("kv_a_layernorm")
        self.q_a_layernorm = kwargs.get("q_a_layernorm")
        self.tp_size = get_tensor_model_parallel_world_size()
        self._skip_topk = bool(kwargs.get("skip_topk", False))
        self.topk_indices_buffer = kwargs.get("topk_indices_buffer")
        # Optional platform service injected by the model runner. Attention
        # stays independent of KVPP scheduling and the concrete transport.
        self.layerwise_kv_cache_hook: Any = None

        ascend_config = get_ascend_config()
        self.vllm_config = get_current_vllm_config()
        self.pcp_enabled = self.vllm_config.parallel_config.prefill_context_parallel_size > 1
        self.is_pcp_decode_sharded = is_pcp_decode_sharding_enabled(self.vllm_config)
        # SFA absorbs kv_b_proj (and, for KV consumers on PROLOG_V3, the fused
        # qkv/q projections) and disposes the source parameters. A disposed
        # parameter is no longer a valid destination for the in-place weight
        # updates that RL pushes through vLLM's layerwise reload, so those
        # sources must survive whenever such updates are possible.
        self.rl_weight_update_enabled = is_rl_weight_update_enabled(self.vllm_config)
        kv_transfer_config = self.vllm_config.kv_transfer_config
        self.is_kv_producer = kv_transfer_config is not None and kv_transfer_config.is_kv_producer
        self.is_kv_consumer = kv_transfer_config is not None and kv_transfer_config.is_kv_consumer

        self.sfa_qsfa_tile_size = SFA_QSFA_TILE_SIZE
        self.sfa_qsfa_packed_kv_head_dim = 0
        self.sfa_qsfa_k_nope_clip_alpha: torch.Tensor | None = None
        self.sfa_qsfa_kr_cache_dummy: torch.Tensor | None = None

        self.local_num_heads = self.num_heads
        self.layer_name = kwargs.get("layer_name")
        hf_config = self.vllm_config.model_config.hf_config
        hf_text_config = getattr(self.vllm_config.model_config, "hf_text_config", None)
        config_candidates = (hf_config, hf_text_config)
        index_cache_enabled = _get_config_bool(
            config_candidates,
            "use_index_cache",
        ) or _has_shared_indexer_layers(config_candidates)
        self.use_index_cache = self.skip_topk or index_cache_enabled
        self._is_mtp_layer = is_mtp_layer(hf_config, self.layer_name)
        self.skip_indexer_pre_process = self.skip_topk and not self._is_mtp_layer
        self.has_indexer = self.indexer is not None
        if not self.has_indexer and not self.skip_topk:
            raise ValueError(
                "Indexer is required for DSA unless skip_topk is enabled. "
                f"Got indexer=None, skip_topk={self.skip_topk}, "
                f"layer_name={self.layer_name}."
            )
        if not self.has_indexer and self.topk_indices_buffer is None:
            raise ValueError(
                "topk_indices_buffer is required when indexer is None and "
                f"skip_topk is enabled. layer_name={self.layer_name}."
            )
        # The indexer module owns its compute rules; SFA only keeps the
        # indexer head_dim for cache-layout composition on CP paths.
        if self.has_indexer:
            self.head_dim: int = self.indexer.head_dim  # 128
        else:
            self.head_dim = getattr(hf_config, "index_head_dim", 0)

        # Sparse C8 has two independent meanings in SFA:
        # - SFA packed KV cache for npu_kv_quant_sparse_flash_attention_vllm.
        # - C8 indexer cache for lightning indexer.
        # The user-facing switches control these layouts independently. LI C8
        # applies only to layers that own an indexer cache.
        self.enable_sparse_sfa_turboquant = ascend_config.enable_sparse_sfa_turboquant
        self.enable_sparse_sfa_c8 = ascend_config.enable_sparse_sfa_c8
        if self.enable_sparse_sfa_c8:
            self.c8_cache_dtype = kv_cache_dtype_str_to_dtype(
                self.vllm_config.cache_config.cache_dtype, self.vllm_config.model_config
            )

        if self.enable_sparse_sfa_turboquant:
            self.sfa_qsfa_packed_kv_head_dim = get_tq_fused_slot_bytes(self.kv_lora_rank, self.qk_rope_head_dim)
        elif self.enable_sparse_sfa_c8:
            self.sfa_qsfa_packed_kv_head_dim = get_sfa_qsfa_packed_head_dim(
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self.sfa_qsfa_tile_size,
            )
        # Shared signed-Hadamard transform plus the ops-nn quantizer.
        self.turboquant = TurboQuantLatent() if self.enable_sparse_sfa_turboquant else None
        self.preprocess_type = PreprocessType.NATIVE

        self.enable_mlapo = bool(get_ascend_config().enable_mlapo)

        self.enable_sp = enable_sp()
        self._initialize_pcp_o_proj()
        self._initialize_dcp()

    def _initialize_dcp(self) -> None:
        self.dcp_enabled = enable_sfa_dcp_replicated_indexer(self.vllm_config)
        self.dcp_group = None
        self.dcp_size = 1
        self.dcp_rank = 0
        if not self.dcp_enabled:
            return
        self.dcp_group = get_dcp_group()
        self.dcp_size = self.dcp_group.world_size
        self.dcp_rank = self.dcp_group.rank_in_group
        # DCP shards only the SFA KV cache. MLAPO writes the SFA KV cache
        # internally, so keep DCP on the native path where we pass the DCP
        # slot mapping explicitly.
        self.enable_mlapo = False
        self._dcp_interleave_size = self.vllm_config.parallel_config.cp_kv_cache_interleave_size
        if self._dcp_interleave_size <= 0:
            raise RuntimeError(f"Invalid cp_kv_cache_interleave_size: {self._dcp_interleave_size}")
        self._dcp_index_topk = 0
        for config in (
            getattr(self.vllm_config.model_config, "hf_text_config", None),
            getattr(self.vllm_config.model_config, "hf_config", None),
        ):
            index_topk = getattr(config, "index_topk", None)
            if isinstance(index_topk, int) and index_topk > 0:
                self._dcp_index_topk = index_topk
                break
        if self._dcp_index_topk <= 0:
            raise RuntimeError("index_topk must be set in the model config for DCP SFA.")
        device = self.q_proj.weight.device
        self._remap_order = torch.arange(self._dcp_index_topk, dtype=torch.float32, device=device)
        self._remap_invalid_index = torch.tensor(-1.0, dtype=torch.float32, device=device)

    def _initialize_pcp_o_proj(self) -> None:
        self._parallel_o_proj: ParallelOProjection | None = None
        self.enable_pcp_o_proj_weight_sharding = False
        if not self.pcp_enabled:
            return
        self.enable_pcp_o_proj_weight_sharding = enable_pcp_o_proj_weight_sharding()
        self._parallel_o_proj = ParallelOProjection(
            self.o_proj.input_size_per_partition,
            self.o_proj,
            config=WeightSwitchConfig.from_group(get_pcp_group(), shard_axis="input"),
            tp_group=get_tp_group(),
            pool=self.o_proj_full_pools,
            pool_key=self.o_proj_weight_switch_pool_key,
            project_full=self._project_o_proj_full,
            decode_sharded=self.is_pcp_decode_sharded,
        )
        if self.enable_pcp_o_proj_weight_sharding:
            try:
                self._parallel_o_proj.prepare_weight_load(self.vllm_config)
            except (RuntimeError, ValueError) as exc:
                logger.warning_once(
                    "SFA-PCP O-projection weight sharding is unavailable (%s); using unsharded O-projection weights.",
                    exc,
                )
                self.enable_pcp_o_proj_weight_sharding = False

    @property
    def uses_packed_sfa_main_cache(self) -> bool:
        return bool(self.enable_sparse_sfa_c8 or self.enable_sparse_sfa_turboquant)

    @property
    def skip_topk(self) -> bool:
        return self._skip_topk

    @skip_topk.setter
    def skip_topk(self, value: bool) -> None:
        self._skip_topk = bool(value)
        if hasattr(self, "_is_mtp_layer"):
            self.skip_indexer_pre_process = self._skip_topk and not self._is_mtp_layer

    @property
    def runtime_has_indexer(self) -> bool:
        return self.has_indexer and not getattr(self, "skip_indexer_pre_process", False)

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config=None,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        # sfa does not need to update graph params
        pass

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # kv_b_proj is absorbed into W_UK/W_UV below and then disposed, so it never runs a
        # matmul. What matters is that whichever quant method owns it left a dense weight
        # behind; a quantized one would have replaced it with a layout we cannot split.
        assert self.kv_b_proj.weight.dtype == act_dtype, (
            f"SFA absorbs kv_b_proj and needs it dense in {act_dtype}, "
            f"got {self.kv_b_proj.weight.dtype} from {type(self.kv_b_proj.quant_method).__name__}"
        )
        # NOTE: Weight will be reshaped next, we need to revert and transpose it.
        kv_b_proj_weight = torch_npu.npu_format_cast(self.kv_b_proj.weight.data, ACL_FORMAT_FRACTAL_ND).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.local_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.local_num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.local_num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # NOTE: `W_UK`/`W_UV` must live at a stable address: in graph + RL
        # scenario the graph is captured once, so every later weight update
        # reload has to refresh the stored tensors in place instead of
        # rebinding the attributes to freshly allocated tensors.
        # `replace_parameter(..., prefer_copy=True)` does exactly that on
        # v0.32.0: it copies into the existing storage while the new value stays
        # compatible, and rebinds only when shape/dtype/device actually
        # changes. vLLM main (#62326) dropped `prefer_copy` and keeps the
        # parameter object by default. This mirrors how upstream vLLM refreshes
        # `MLAAttention.W_UV`/`W_UK_T`, and unlike a plain `copy_` it cannot
        # raise on a shape mismatch.
        w_uv = W_UV.transpose(0, 1).contiguous()  # (L, N, V) -> (N, L, V)
        w_uk_t = W_UK.permute(1, 2, 0).contiguous()  # (L, N, P) -> (N, P, L)
        if vllm_version_is("0.32.0"):
            replace_parameter(self, "W_UV", w_uv, prefer_copy=True)
            replace_parameter(self, "W_UK_T", w_uk_t, prefer_copy=True)
        else:
            replace_parameter(self, "W_UV", w_uv)
            replace_parameter(self, "W_UK_T", w_uk_t)

        # Dispose kv_b_proj since it is replaced by W_UV and W_UK_T to save memory.
        # RL keeps it: it is the only source of W_UV/W_UK_T, so every weight
        # update re-derives them from this parameter and the parameter must stay
        # loadable (#15463).
        if not self.rl_weight_update_enabled:
            dispose_layer(self.kv_b_proj)
        self.preprocess_type = self._resolve_preprocess_type(act_dtype)

        if self.preprocess_type == PreprocessType.NATIVE:
            self.W_UK_T = maybe_trans_nz(self.W_UK_T)

        if self.preprocess_type == PreprocessType.PROLOG_V3 and self.enable_sparse_sfa_c8:
            if self.sfa_qsfa_kr_cache_dummy is None:
                self.sfa_qsfa_kr_cache_dummy = torch.empty(
                    0,
                    dtype=torch.bfloat16,
                    device=self.weight_dq.device,
                )

        if self.has_indexer:
            self.indexer.process_weights_after_loading()

        if self._parallel_o_proj is not None:
            self._parallel_o_proj.enable()
        if self.dcp_enabled:
            self._register_remap_buffers()

    @staticmethod
    def _get_layer_quant_method(layer: torch.nn.Module | None):
        return getattr(getattr(layer, "quant_method", None), "quant_method", None)

    def _resolve_preprocess_type(self, act_dtype: torch.dtype) -> PreprocessType:
        quant_method = self._get_layer_quant_method(self.fused_qkv_a_proj)
        self._quant_type = type(quant_method) if quant_method is not None else None

        pp_type = self._fused_preprocess_type()
        if pp_type is not None and self._try_enable_type(pp_type, act_dtype):
            return pp_type
        return PreprocessType.NATIVE

    def _fused_preprocess_type(self) -> PreprocessType | None:
        """Return the enabled fused preprocess type, or None if it cannot run."""
        quant_method = self._get_layer_quant_method(self.fused_qkv_a_proj)
        qt = type(quant_method) if quant_method is not None else None

        # PROLOG_V3 takes precedence over MLAPO and is the default fused
        # preprocessing for quantized SFA layers in every deployment (plain
        # serving, PD KV producers and KV consumers). The C8 switches select the
        # KV cache layout and are orthogonal to this choice. Unquantized
        # layers keep the NATIVE chain outside KV consumers because the
        # unquantized weight preparation transposes fused_qkv_a_proj.weight
        # in place, which the NATIVE fallback still consumes.
        if getattr(self, "dcp_group", None) is None:
            prolog_v3_eligible = qt is not None or self.is_kv_consumer
            if prolog_v3_eligible and (
                qt is AscendW8A8DynamicLinearMethod or qt is AscendW8A8MXFP8DynamicLinearMethod or qt is None
            ):
                if not self._get_fused_type_unsupported_reasons(PreprocessType.PROLOG_V3):
                    return PreprocessType.PROLOG_V3

        eligible = qt is AscendW8A8LinearMethod and self.enable_mlapo
        if eligible and not self._get_fused_type_unsupported_reasons(PreprocessType.MLAPO):
            return PreprocessType.MLAPO

        return None

    def _try_enable_type(self, pp_type: PreprocessType, act_dtype: torch.dtype) -> bool:
        reasons = self._get_fused_type_unsupported_reasons(pp_type)
        if reasons:
            for msg in reasons:
                logger.warning_once(msg)
            return False
        if pp_type is PreprocessType.PROLOG_V3:
            self._process_weights_for_fused_prolog_v3()
        else:
            self._process_weights_for_fused_mlapo(act_dtype)
        return True

    def _get_fused_type_unsupported_reasons(self, pp_type: PreprocessType) -> list[str]:
        reasons = []
        if self.qk_rope_head_dim == 0:
            reasons.append("NoPE SFA currently uses native preprocessing; fused NoPE contracts are not enabled.")
        if self.kv_a_layernorm is None or self.q_a_layernorm is None:
            reasons.append("Fused preprocessing requires q_a_layernorm and kv_a_layernorm.")
        if self.fused_qkv_a_proj is None:
            reasons.append("fused_qkv_a_proj is None, mlapo is disabled.")

        quant_method = self._get_layer_quant_method(self.fused_qkv_a_proj)
        qt = type(quant_method) if quant_method is not None else None
        if pp_type is PreprocessType.PROLOG_V3:
            # Prolog's indexer query can carry quantization scales independently
            # of the MLA weight type. Preserve the native query for an
            # unquantized indexer, including layers that later toggle skip_topk.
            if self.has_indexer:
                wq_b = getattr(self.indexer, "wq_b", None)
                if not hasattr(wq_b, "weight_scale"):
                    reasons.append("PROLOG_V3 requires a quantized indexer query projection.")
            if self.enable_sparse_sfa_turboquant:
                reasons.append("PROLOG_V3 does not support the TurboQuant 4-bit latent cache.")
            if qt is None and self.enable_sparse_sfa_c8:
                reasons.append("PROLOG_V3: C8 sparse requires quantized MLAPO.")
            if getattr(self.q_proj, "_chunk_size", 0):
                reasons.append("PROLOG_V3 does not support chunked q_proj weights yet.")
        elif pp_type is PreprocessType.MLAPO:
            if self.enable_sparse_sfa_turboquant:
                reasons.append("MLAPO does not support the TurboQuant 4-bit latent cache.")
            elif self.enable_sparse_sfa_c8:
                reasons.append("MLAPO does not support sparse C8; use PROLOG_V3 instead.")

        return reasons

    def _process_weights_for_fused_prolog_v3(self) -> None:
        assert self.fused_qkv_a_proj is not None
        assert self.q_proj is not None

        qt = self._quant_type

        if qt is None:
            self.fused_qkv_a_proj.weight.data = self.fused_qkv_a_proj.weight.data.T

        fused_weight = self.fused_qkv_a_proj.weight.data
        weight_dq = fused_weight[..., : self.q_lora_rank].contiguous()
        weight_dkv_kr = fused_weight[..., self.q_lora_rank :].contiguous()
        if qt is not None:
            weight_uq_qr = self.q_proj.weight.data.contiguous()
        else:
            weight_uq_qr = self.q_proj.weight.data.T.contiguous()

        self.weight_dq = torch_npu.npu_format_cast(weight_dq, ACL_FORMAT_FRACTAL_NZ)
        self.weight_dkv_kr = torch_npu.npu_format_cast(weight_dkv_kr, ACL_FORMAT_FRACTAL_NZ)
        self.weight_uq_qr = torch_npu.npu_format_cast(weight_uq_qr, ACL_FORMAT_FRACTAL_NZ)

        if qt is AscendW8A8DynamicLinearMethod:
            q_scl = self.fused_qkv_a_proj.weight_scale[: self.q_lora_rank].contiguous()
            kv_scl = self.fused_qkv_a_proj.weight_scale[self.q_lora_rank :].contiguous()
            self.dequant_scale_w_dq = q_scl.view(1, -1).to(torch.float)
            self.dequant_scale_w_dkv_kr = kv_scl.view(1, -1).to(torch.float)
            self.dequant_scale_w_uq_qr = self.q_proj.weight_scale.data.view(1, -1).to(torch.float)
            if self.enable_sparse_sfa_c8:
                self.sfa_qsfa_k_nope_clip_alpha = torch.ones(
                    1,
                    dtype=torch.float32,
                    device=self.weight_dq.device,
                )
        elif qt is AscendW8A8MXFP8DynamicLinearMethod:
            w_scale = self.fused_qkv_a_proj.weight_scale
            w_scale = w_scale.transpose(0, 1)
            w_scale = w_scale.reshape(-1, w_scale.shape[1] * w_scale.shape[2])
            self.weight_dq_scale = w_scale[: self.q_lora_rank, ...]
            self.weight_dkv_kr_scale = w_scale[self.q_lora_rank :, ...]

            uq_scale = self.q_proj.weight_scale.data.transpose(0, 1)
            self.weight_uq_qr_scale = uq_scale.reshape(-1, uq_scale.shape[1] * uq_scale.shape[2])

        # Same reasoning as kv_b_proj: once the fused projections are consumed by
        # PROLOG_V3 they are pure load sources, but discarding their storage
        # breaks the next layerwise reload, so RL keeps them.
        if self.is_kv_consumer and not self.rl_weight_update_enabled:
            dispose_layer(self.fused_qkv_a_proj)
            dispose_layer(self.q_proj)
            torch.npu.empty_cache()

    # Processing the input parameters for MLAPO by reordering and transposing
    # QKV(and part of Q) weight, applying RoPE-related dimension transformations,
    # and handling quantization parameters.
    def _process_weights_for_fused_mlapo(self, act_dtype: torch.dtype):
        assert self.kv_a_proj_with_mqa is None
        assert self.fused_qkv_a_proj is not None

        kv_a_proj_wt = self.fused_qkv_a_proj.weight.data[..., self.q_lora_rank :].contiguous()
        q_a_proj_wt = self.fused_qkv_a_proj.weight.data[..., : self.q_lora_rank].contiguous()

        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        kv_a_proj_wt = trans_rope_weight(kv_a_proj_wt, self.qk_rope_head_dim)
        kv_a_proj_wt = kv_a_proj_wt.t().contiguous()
        wd_qkv = torch.cat((kv_a_proj_wt, q_a_proj_wt), dim=-1)
        wd_qkv = wd_qkv.t().contiguous()
        wd_qkv = transdata(wd_qkv, block_size=(16, 32)).unsqueeze(0).contiguous()
        self.wd_qkv = torch_npu.npu_format_cast(wd_qkv, ACL_FORMAT_FRACTAL_NZ)

        kv_a_proj_deq_scl = self.fused_qkv_a_proj.deq_scale[self.q_lora_rank :].contiguous()
        q_a_proj_deq_scl = self.fused_qkv_a_proj.deq_scale[: self.q_lora_rank].contiguous()
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.reshape(self.kv_lora_rank + self.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_deq_scl = trans_rope_weight(kv_a_proj_deq_scl, self.qk_rope_head_dim)
        kv_a_proj_deq_scl = kv_a_proj_deq_scl.view(self.kv_lora_rank + self.qk_rope_head_dim).contiguous()
        self.deq_scale_qkv = torch.cat((kv_a_proj_deq_scl, q_a_proj_deq_scl), dim=-1).contiguous()

        kv_a_proj_qt_bias = self.fused_qkv_a_proj.quant_bias[self.q_lora_rank :].contiguous()
        q_a_proj_qt_bias = self.fused_qkv_a_proj.quant_bias[: self.q_lora_rank].contiguous()

        kv_a_proj_qt_bias = kv_a_proj_qt_bias.reshape(self.kv_lora_rank + self.qk_rope_head_dim, -1).contiguous()
        kv_a_proj_qt_bias = trans_rope_weight(kv_a_proj_qt_bias, self.qk_rope_head_dim)
        kv_a_proj_qt_bias = kv_a_proj_qt_bias.view(self.kv_lora_rank + self.qk_rope_head_dim).contiguous()
        self.quant_bias_qkv = torch.cat((kv_a_proj_qt_bias, q_a_proj_qt_bias), dim=-1).contiguous()

        wu_q = self.q_proj.weight.data
        wu_q = wu_q.t().reshape(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        wu_q = trans_rope_weight(wu_q, self.qk_rope_head_dim)
        wu_q = wu_q.reshape(self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), -1)
        wu_q = transdata(wu_q, block_size=(16, 32)).unsqueeze(0).contiguous()
        self.wu_q = torch_npu.npu_format_cast(wu_q, ACL_FORMAT_FRACTAL_NZ)

        qb_deq_scl = self.q_proj.deq_scale.data
        qb_deq_scl = qb_deq_scl.reshape(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        qb_deq_scl = trans_rope_weight(qb_deq_scl, self.qk_rope_head_dim)
        self.qb_deq_scl = qb_deq_scl.reshape(self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim))

        qb_qt_bias = self.q_proj.quant_bias.data
        qb_qt_bias = qb_qt_bias.reshape(self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1)
        qb_qt_bias = trans_rope_weight(qb_qt_bias, self.qk_rope_head_dim)
        self.qb_qt_bias = qb_qt_bias.reshape(self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim))

        device = self.q_proj.weight.device
        self.gamma1 = self.q_a_layernorm.weight.data  # type: ignore[union-attr]
        self.beta1 = self.q_a_layernorm.bias.data  # type: ignore[union-attr]
        self.gamma2 = self.kv_a_layernorm.weight.data  # type: ignore[union-attr]
        self.quant_scale0 = self.fused_qkv_a_proj.input_scale.data
        self.quant_offset0 = self.fused_qkv_a_proj.input_offset.data
        self.quant_scale1 = self.q_proj.input_scale.data
        self.quant_offset1 = self.q_proj.input_offset.data
        self.ctkv_scale = torch.tensor([1], dtype=act_dtype, device=device)
        self.q_nope_scale = torch.tensor([1], dtype=act_dtype, device=device)

        # On KV consumers (decode-only) MLAPO uses the transformed weights built above;
        # the original fused_qkv_a_proj/q_proj weights and quant params are no longer
        # referenced, so drop them to save memory.
        if (
            self.vllm_config.kv_transfer_config is not None
            and self.vllm_config.kv_transfer_config.is_kv_consumer
            and self.vllm_config.scheduler_config.max_num_batched_tokens <= MLAPO_MAX_SUPPORTED_TOKENS
        ):
            self.fused_qkv_a_proj.weight = None
            self.fused_qkv_a_proj.deq_scale = None
            self.fused_qkv_a_proj.quant_bias = None
            self.q_proj.weight = None
            self.q_proj.deq_scale = None
            self.q_proj.quant_bias = None
            torch.npu.empty_cache()

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        raise NotImplementedError("forward_mha is not supported for SFA attention. Use forward() instead.")

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("forward_mqa is not supported for SFA attention. Use forward() instead.")

    def rope_single(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        S = 1
        x = x.view(B, N, S, D)
        x = torch_npu.npu_interleave_rope(x, cos, sin)
        return x.view(B, N, D)

    def exec_kv(
        self,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: M,
    ):
        if self.dcp_enabled and self.pcp_enabled and self._has_prefill(attn_metadata):
            assert isinstance(attn_metadata, AscendSFADCPMetadata)
            assert attn_metadata.dcp_context is not None
            slots = attn_metadata.dcp_context.slot_mapping
        if self.pcp_enabled:
            if needs_pcp_kv_gather(attn_metadata, self.is_pcp_decode_sharded):
                num_replicated_tokens = get_pcp_num_replicated_tokens(
                    attn_metadata.num_decode_tokens or 0, self.is_pcp_decode_sharded
                )
                if self.qk_rope_head_dim == 0:
                    # NoPE keeps the builder's padded rank stride for KV gathers.
                    pad_tokens = attn_metadata.num_input_tokens - kv_no_split.shape[0]
                    if pad_tokens < 0:
                        raise ValueError("SFA PCP KV exceeds the rank-local padded token count.")
                    if pad_tokens:
                        kv_no_split = nn.functional.pad(kv_no_split, (0, 0, 0, pad_tokens))
                    (kv_no_split,), slots = _gather_prefill_cache_inputs((kv_no_split,), slots, num_replicated_tokens)
                else:
                    (kv_no_split, cos, sin), slots = _gather_prefill_cache_inputs(
                        (kv_no_split, cos, sin), slots, num_replicated_tokens
                    )
                assert slots.numel() == kv_no_split.shape[0], (
                    "SFA PCP cache write requires one slot per gathered token: "
                    f"tokens={kv_no_split.shape[0]}, slots={slots.numel()}."
                )
            else:
                slots = slots[: kv_no_split.shape[0]]
        if self.qk_rope_head_dim == 0 and not self.enable_sparse_sfa_c8:
            assert self.kv_a_layernorm is not None
            values = self.kv_a_layernorm(kv_no_split.reshape(-1, self.kv_lora_rank))
            cache = kv_cache[0]
            # The hybrid cache configuration keeps NoPE main KV pages packed.
            torch_npu.npu_scatter_nd_update_(
                cache.view(-1, self.kv_lora_rank),
                slots[: values.shape[0]].view(-1, 1),
                values.to(cache.dtype),
            )
            return None, None
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA"
        # TQ4 must be checked before C8 because both return packed byte components.
        if self.enable_sparse_sfa_turboquant:
            assert self.kv_a_layernorm is not None
            assert self.turboquant is not None
            return turboquant_kv_rmsnorm_rope(
                kv_no_split,
                self.kv_a_layernorm.weight,
                cos,
                sin,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                turboquant=self.turboquant,
                epsilon=self.kv_a_layernorm.variance_epsilon,
            )

        # npu_kv_rmsnorm_rope_cache doesn't support C8 fp8 block quant;
        # all sparse-C8-SFA layers use custom_kv_rmsnorm_rope instead.
        if self.enable_sparse_sfa_c8:
            assert self.kv_a_layernorm is not None
            return custom_kv_rmsnorm_rope(
                kv_no_split,
                self.kv_a_layernorm.weight,
                cos,
                sin,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                dst_type=self.c8_cache_dtype,
                tile_size=self.sfa_qsfa_tile_size,
            )

        torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,  # type: ignore[union-attr]
            cos,
            sin,
            _int64_kv_slots(slots, attn_metadata),
            kv_cache[1],
            kv_cache[0],
            epsilon=self.kv_a_layernorm.variance_epsilon,  # type: ignore[union-attr]
            cache_mode=cache_mode,
        )
        return None, None

    # Return `ql_nope`, `q_pe`
    def _q_proj_and_k_up_proj(self, x):
        q_nope, q_pe = (
            self.q_proj(x)[0]
            .view(-1, self.local_num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )

        if (
            q_nope.dtype in [torch.float16, torch.bfloat16]
            and hasattr(torch_npu, "npu_transpose_batchmatmul")
            and q_nope.shape[0] < TRANSPOSE_BMM_MAX_SUPPORTED_DIM
        ):
            # Convert from (B, N, P) to (N, B, P) and multiply
            # (N, B, P) x (N, P, L) -> (B, N, L)
            ql_nope = torch_npu.npu_transpose_batchmatmul(
                q_nope,
                self.W_UK_T,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            )
        else:
            # Fallback for torch_npu builds without the fused op, unsupported
            # dtypes, or a token dim beyond the operand limit.
            # Convert from (B, N, P) to (N, B, P)
            q_nope = q_nope.transpose(0, 1)
            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            ql_nope = torch.bmm(q_nope, self.W_UK_T)
            # Convert from (N, B, L) to (B, N, L)
            ql_nope = ql_nope.transpose(0, 1)
        return ql_nope, q_pe

    def _v_up_proj(self, x):
        if hasattr(torch_npu, "npu_transpose_batchmatmul"):
            # (B, N, L) -> operator transposes internally -> (B, N, V)
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank)
            x = torch_npu.npu_transpose_batchmatmul(x, self.W_UV, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
            # (B, N, V) -> (B, N * V)
            x = x.reshape(-1, self.local_num_heads * self.v_head_dim)
        else:
            # (B, N, L) -> (N, B, L)
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank).transpose(0, 1)
            # (N, B, L) x (N, L, V) -> (N, B, V)
            x = torch.bmm(x, self.W_UV)
            # (N, B, V) -> (B, N * V)
            x = x.transpose(0, 1).reshape(-1, self.local_num_heads * self.v_head_dim)
        return x

    def _sfa_preprocess_prolog_v3(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        attn_metadata: M | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        assert slot_mapping.numel() == hidden_states.shape[0], (
            "SFA Prolog V3 requires one cache index per input token, "
            f"got token_x={hidden_states.shape[0]} and cache_index={slot_mapping.numel()}."
        )
        assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized for PROLOG_V3"
        assert self.kv_a_layernorm is not None, "kv_a_layernorm must be initialized for PROLOG_V3"

        qt = self._quant_type
        use_c8 = self.enable_sparse_sfa_c8

        common: dict[str, Any] = dict(
            weight_dq=self.weight_dq,
            weight_uq_qr=self.weight_uq_qr,
            weight_uk=self.W_UK_T,
            weight_dkv_kr=self.weight_dkv_kr,
            rmsnorm_gamma_cq=self.q_a_layernorm.weight.data,
            rmsnorm_gamma_ckv=self.kv_a_layernorm.weight.data,
            rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
            rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
            query_norm_flag=self.has_indexer,
            qc_qr_scale=1.0,
            kc_scale=1.0,
            cache_mode="PA_BSND",
            query_quant_mode=0,
        )
        kv_cache_nope = kv_cache[0]
        extra_kwargs: dict[str, Any] = {}
        if use_c8:
            extra_kwargs.update(
                ckvkr_repo_mode=1,
                quant_scale_repo_mode=1,
                tile_size=self.sfa_qsfa_tile_size,
                k_nope_clip_alpha=self.sfa_qsfa_k_nope_clip_alpha,
            )
            kr_cache = self.sfa_qsfa_kr_cache_dummy
        else:
            kr_cache = kv_cache[1]
        rope_cos_ = cos.view(cos.shape[0], cos.shape[-1])
        rope_sin_ = sin.view(sin.shape[0], sin.shape[-1])
        # The caller forwards the per-step cached int64 slots, so the .to()
        # below is a no-op in production; kept for direct-call safety.
        cache_index = slot_mapping.view(-1).to(torch.int64)

        if qt is not None:
            if qt is AscendW8A8MXFP8DynamicLinearMethod:
                token_x, ds = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
                branch = dict(
                    dequant_scale_x=ds.reshape(token_x.shape[0], -1).view(torch.float8_e8m0fnu),
                    dequant_scale_w_dq=self.weight_dq_scale.view(torch.float8_e8m0fnu),
                    dequant_scale_w_uq_qr=self.weight_uq_qr_scale.view(torch.float8_e8m0fnu),
                    dequant_scale_w_dkv_kr=self.weight_dkv_kr_scale.view(torch.float8_e8m0fnu),
                    weight_quant_mode=3,
                )
            else:
                assert qt is AscendW8A8DynamicLinearMethod, (
                    f"PROLOG_V3 only supports W8A8Dynamic or W8A8MXFP8 quant, "
                    f"got {qt}. Did _resolve_preprocess_type allow a new quant type?"
                )
                token_x, dequant_x = torch_npu.npu_dynamic_quant(hidden_states.contiguous())
                branch = dict(
                    dequant_scale_x=dequant_x.view(-1, 1),
                    dequant_scale_w_dq=self.dequant_scale_w_dq,
                    dequant_scale_w_uq_qr=self.dequant_scale_w_uq_qr,
                    dequant_scale_w_dkv_kr=self.dequant_scale_w_dkv_kr,
                    weight_quant_mode=2,
                )
        else:
            token_x = hidden_states
            branch = dict(
                dequant_scale_x=None,
                dequant_scale_w_dq=None,
                dequant_scale_w_uq_qr=None,
                dequant_scale_w_dkv_kr=None,
                weight_quant_mode=0,
            )

        ql_nope, q_pe, _, q_c, q_c_scale = torch_npu.npu_mla_prolog_v3(
            token_x=token_x,
            rope_sin=rope_sin_,
            rope_cos=rope_cos_,
            kv_cache=kv_cache_nope,
            kr_cache=kr_cache,
            cache_index=cache_index,
            kv_cache_quant_mode=3 if use_c8 else 0,
            **common,
            **branch,
            **extra_kwargs,
        )
        num_h = self.local_num_heads
        ql_nope = ql_nope.view(-1, num_h, self.kv_lora_rank)
        q_pe = q_pe.view(-1, num_h, self.qk_rope_head_dim)

        if self.has_indexer:
            if q_c is None:
                raise RuntimeError("npu_mla_prolog_v3 did not return query_norm for SFA indexer.")
            q_c = q_c.view(-1, self.q_lora_rank)
            if q_c_scale is not None:
                if qt is AscendW8A8MXFP8DynamicLinearMethod:
                    q_c_scale = q_c_scale.view(-1, q_c_scale.shape[-1])
                    q_c = (q_c, q_c_scale)
                else:
                    q_c = (q_c, q_c_scale.view(-1))
        elif self._quant_type is None:
            q_c = None

        return hidden_states, ql_nope, q_pe, q_c

    def _sfa_preprocess_mlapo(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        num_input_tokens: int = 0,
        attn_metadata: M | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        """A3 MLAPO via ``torch.ops._C_ascend.mla_preprocess`` (W8A8, ≤ 1024 tokens)."""
        k_nope, k_pe = kv_cache[0], kv_cache[1]
        ql_nope = torch.empty(
            (num_input_tokens, self.W_UK_T.shape[0], k_nope.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_pe = torch.empty(
            (num_input_tokens, self.W_UK_T.shape[0], k_pe.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        q_c = torch.empty(
            (num_input_tokens, self.q_lora_rank),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops._C_ascend.mla_preprocess(
            hidden_states,
            self.wd_qkv,
            self.deq_scale_qkv,
            self.gamma1,
            self.beta1,
            self.wu_q,
            self.qb_deq_scl,
            self.gamma2,
            cos,
            sin,
            self.W_UK_T,
            k_nope,
            k_pe,
            slot_mapping,
            quant_scale0=self.quant_scale0,
            quant_offset0=self.quant_offset0,
            bias0=self.quant_bias_qkv,
            quant_scale1=self.quant_scale1,
            quant_offset1=self.quant_offset1,
            bias1=self.qb_qt_bias,
            ctkv_scale=self.ctkv_scale,
            q_nope_scale=self.q_nope_scale,
            cache_mode="krope_ctkv",
            quant_mode="per_tensor_quant_asymm",
            enable_inner_out=True,
            q_out0=ql_nope,
            kv_cache_out0=k_nope,
            q_out1=q_pe,
            kv_cache_out1=k_pe,
            inner_out=q_c,
        )
        return hidden_states, ql_nope, q_pe, q_c

    def _prepare_indexer_metadata(self, indexer_metadata, attn_metadata) -> None:
        """Allow an attention backend to supply indexer selection metadata."""
        return

    def _get_indexcache_topk_indices(self, num_tokens: int) -> torch.Tensor:
        if self.topk_indices_buffer is None:
            raise RuntimeError("IndexCache requires topk_indices_buffer when skip_topk is enabled.")
        topk_indices = self.topk_indices_buffer[:num_tokens]
        if topk_indices.dim() == 2:
            topk_indices = topk_indices.unsqueeze(1)
        return topk_indices

    def _update_indexcache_topk_indices(self, topk_indices: torch.Tensor) -> None:
        if self.topk_indices_buffer is None:
            return
        num_tokens = topk_indices.shape[0]
        topk_tokens = topk_indices.shape[-1]
        topk_indices_to_cache = topk_indices
        topk_indices_buffer = self.topk_indices_buffer[:num_tokens, :topk_tokens]
        if topk_indices_to_cache.dim() == 3 and topk_indices_buffer.dim() == 2:
            assert topk_indices_to_cache.shape[1] == 1
            topk_indices_to_cache = topk_indices_to_cache.squeeze(1)
        topk_indices_buffer.copy_(topk_indices_to_cache)

    def _tq_rotate_query(self, ql_nope: torch.Tensor, q_pe: torch.Tensor) -> torch.Tensor:
        assert self.turboquant is not None
        rotated = self.turboquant.forward(ql_nope)
        return torch.cat([rotated, q_pe], dim=-1).contiguous()

    def _turboquant_sfa(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        topk_indices: torch.Tensor,
        block_table: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        sparse_mode: int,
        return_softmax_lse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # TQ4 is a quantization type of this operator rather than a separate one.
        # Call it through torch_npu, i.e. the operator that ships with CANN,
        # instead of this repo's csrc binding of the same name.
        kwargs = {
            "key_dequant_scale": None,
            "value_dequant_scale": None,
            "block_table": block_table,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "scale_value": self.scale,
            "key_quant_mode": TQ_QUANT_MODE,
            "value_quant_mode": TQ_QUANT_MODE,
            "sparse_block_size": 1,
            "layout_query": "TND",
            "layout_kv": "PA_BSND",
            "sparse_mode": sparse_mode,
            "attention_mode": 2,
            "quant_scale_repo_mode": 1,
            "tile_size": TQ_TILE_SIZE,
            "rope_head_dim": self.qk_rope_head_dim,
        }
        if return_softmax_lse:
            kwargs["return_softmax_lse"] = True
        result = torch_npu.npu_kv_quant_sparse_flash_attention(query, kv, kv, topk_indices, **kwargs)
        return result if isinstance(result, tuple) else (result, None, None)

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table=None,
    ):
        if self.dcp_enabled:
            return self._execute_dcp_sparse_flash_attention_process(
                ql_nope,
                q_pe,
                kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
            )
        if self.enable_sparse_sfa_turboquant:
            attn_out, _, _ = self._turboquant_sfa(
                self._tq_rotate_query(ql_nope, q_pe),
                kv_cache[0],
                topk_indices,
                attn_metadata.block_table if block_table is None else block_table,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                sparse_mode=3,
            )
            assert self.turboquant is not None
            return self.turboquant.inverse(attn_out)
        if self.qk_rope_head_dim == 0 and not self.enable_sparse_sfa_c8:
            return sparse_mla(ql_nope, kv_cache[0], topk_indices, attn_metadata, self.scale)
        if self.qk_rope_head_dim == 0:
            # Present packed NoPE cache with the logical page size used by its metadata.
            cache = kv_cache[0]
            kv_cache = (cache.view(-1, attn_metadata.block_size, *cache.shape[2:]), *kv_cache[1:])
        output = DeviceOperator.execute_sparse_flash_attention_process(
            self,
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            block_table=block_table,
        )
        if self.qk_rope_head_dim == 0:
            # As in floating NoPE, graph-capacity rows may be unwritten by the kernel.
            assert isinstance(output, torch.Tensor)
            valid = torch.arange(ql_nope.shape[0], device=ql_nope.device) < actual_seq_lengths_query[-1]
            output = output.masked_fill(~valid[:, None, None], 0)
        return output

    def _store_packed_kv_cache(
        self,
        k_pe: torch.Tensor | None,
        k_nope: torch.Tensor | None,
        knope_scale: torch.Tensor | None,
        kv_cache: tuple[torch.Tensor, ...],
        slot_mapping_sfa: torch.Tensor,
    ) -> None:
        """Store the packed C8 or TurboQuant KV produced by native preprocessing."""
        assert k_pe is not None
        assert k_nope is not None
        assert knope_scale is not None
        packed_kv = torch.cat(
            [
                k_nope.view(-1, k_nope.shape[-1]),
                # Flatten leading dimensions even when the RoPE feature dimension is zero.
                k_pe.flatten(0, -2),
                knope_scale.view(-1, knope_scale.shape[-1]),
            ],
            dim=-1,
        )
        packed_head_dim = self.sfa_qsfa_packed_kv_head_dim
        assert packed_kv.shape[-1] == packed_head_dim
        packed_kv = packed_kv.view(-1, packed_head_dim)
        DeviceOperator.scatter_cache(
            kv_cache[0].view(-1, packed_head_dim),
            slot_mapping_sfa[: packed_kv.shape[0]].view(-1, 1),
            packed_kv,
        )

    def _sync_pcp_kv_cache(self, kv_cache: tuple[torch.Tensor, ...], attn_metadata: M) -> None:
        local_slots = attn_metadata.pcp_prolog_local_slots
        global_slots = attn_metadata.pcp_prolog_global_slots
        assert local_slots is not None and global_slots is not None
        if not global_slots.numel():
            return
        num_replicated_tokens = get_pcp_num_replicated_tokens(
            attn_metadata.num_decode_tokens or 0, self.is_pcp_decode_sharded
        )
        # Same stream orders the fused cache write, pack, collective and scatter.
        # C8 packs quantized K, BF16 RoPE and scales into the first cache.
        # The remaining tensors belong to the indexer and synchronize separately.
        main_cache = kv_cache[:1] if self.enable_sparse_sfa_c8 else kv_cache[:2]
        packed = copy_pcp_kv_cache(main_cache, local_slots[num_replicated_tokens:])
        group = get_pcp_group()
        gathered = group.all_gather(packed, dim=0)
        if self.enable_sparse_sfa_c8:
            # Preserve packed K, RoPE and scale bytes, including FP8 storage.
            # The paired scatter writes the same payload to the same cache.
            packed_kv = gathered.unsqueeze(1)
            cache_bytes = main_cache[0].view(torch.int8)
            key, value = packed_kv, packed_kv
            key_cache, value_cache = cache_bytes, cache_bytes
        else:
            k_nope, k_pe = gathered.split([kv_cache[0].shape[-1], kv_cache[1].shape[-1]], dim=-1)
            key, value = k_nope.unsqueeze(1), k_pe.unsqueeze(1)
            key_cache, value_cache = kv_cache[0], kv_cache[1]
        DeviceOperator.reshape_and_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=global_slots,
        )

    def _register_remap_buffers(self) -> None:
        """Expose the DCP remap constants to the level-2 sleep restore path.

        ``_remap_order``/``_remap_invalid_index`` are created while the
        sleep-mode weights mem-pool is active and are read by the non-Triton
        remap fallback in ``_remap_sparse_indices``. Attention impls are not
        ``nn.Module`` instances, so without an owning buffer a level-2 wake
        would leave them pointing at remapped, discarded storage. Own them as
        non-persistent buffers of this layer's attention module, the same owner
        pattern the DSA/SFA Hadamard buffers use.
        """
        static_forward_context = self.vllm_config.compilation_config.static_forward_context
        owner = static_forward_context.get(self.layer_name)
        if not isinstance(owner, torch.nn.Module):
            # Integrations that do not forward ``layer_name`` to the impl still
            # keep the projections in this impl's model tree.
            owner = self.o_proj
        for name, tensor in (
            ("_dcp_sfa_remap_order", self._remap_order),
            ("_dcp_sfa_remap_invalid_index", self._remap_invalid_index),
        ):
            existing = getattr(owner, name, None)
            if existing is None:
                owner.register_buffer(name, tensor, persistent=False)
            elif existing is not tensor:
                raise RuntimeError(f"DCP SFA remap buffer {name!r} is already registered with a different tensor.")

    @staticmethod
    def _has_prefill(attn_metadata: M) -> bool:
        # PCP ranks with no local prefill still join the DCP KV gather and use
        # the same attention path as ranks that received prefill tokens.
        return attn_metadata.num_prefills > 0 or attn_metadata.pcp_has_global_prefill

    def _record_dcp_kv_gather_context(
        self,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
    ) -> None:
        """Start the compact KV all-gather used by prefill/mixed DCP batches."""
        if not self._has_prefill(attn_metadata):
            return
        assert isinstance(attn_metadata, AscendSFADCPMetadata)
        assert attn_metadata.dcp_context is not None, "DCP SFA requires attn_metadata.dcp_context."
        assert self.dcp_group is not None, "DCP SFA requires dcp_group when dcp_size > 1."

        valid_block_ids = attn_metadata.dcp_context.kv_gather_block_ids
        block_table = attn_metadata.dcp_context.kv_gather_block_table
        assert valid_block_ids is not None and block_table is not None
        cache = kv_cache[0]
        if self.qk_rope_head_dim == 0:
            cache = _view_cache_as_operator_pages(cache, attn_metadata.block_size)
        kv = _index_select_kv_blocks(cache, valid_block_ids)
        split_sizes: tuple[int, ...]
        if self.uses_packed_sfa_main_cache or self.qk_rope_head_dim == 0:
            # Packed SFA formats store nope, rope, and quantization data in one
            # SFA KV cache. The remaining cache entries belong to the indexer
            # and must not participate in the DCP SFA KV all-gather.
            gather_input = kv.contiguous()
            split_sizes = (kv.shape[-1],)
        else:
            if len(kv_cache) < 2:
                raise RuntimeError("DCP SFA KV all-gather requires nope and rope KV caches.")
            key_rope = _index_select_kv_blocks(kv_cache[1], valid_block_ids)
            if kv.shape[:-1] != key_rope.shape[:-1] or kv.dtype != key_rope.dtype:
                raise RuntimeError(
                    "Cannot fuse DCP KV gather for KV/nope and KV/rope caches with "
                    f"shapes {tuple(kv.shape)} / {tuple(key_rope.shape)} and dtypes {kv.dtype} / {key_rope.dtype}."
                )
            gather_input = torch.cat([kv, key_rope], dim=-1).contiguous()
            split_sizes = (kv.shape[-1], key_rope.shape[-1])
        attn_metadata.dcp_context.gather_context = self._start_dcp_gather(
            gather_input,
            dim=0,
            split_sizes=split_sizes,
        )

    def _start_dcp_gather(
        self,
        x: torch.Tensor,
        dim: int,
        split_sizes: tuple[int, ...],
    ) -> DCPGatherContext:
        # HCCL does not transport FP8 directly; gather the packed cache as bytes.
        transport = as_byte_view(x)
        gathered, handle, restore_perm = self._all_gather_dim_async(transport, dim)
        if transport.dtype != x.dtype:
            gathered = gathered.view(x.dtype)
        return DCPGatherContext(
            gathered=gathered,
            handle=handle,
            restore_perm=restore_perm,
            split_sizes=split_sizes,
        )

    @staticmethod
    def _finish_dcp_gather(
        context: DCPGatherContext,
        keep_view: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if context.handle is not None:
            context.handle.wait()
        if context.query_prepared and not keep_view:
            return unpack_query(context.gathered, context.split_sizes)
        gathered = context.gathered
        if context.restore_perm is not None:
            gathered = gathered.permute(context.restore_perm)
            if not keep_view:
                gathered = gathered.contiguous()
        return torch.split(gathered, context.split_sizes, dim=-1)

    def _finish_dcp_kv_gather(self, kv_cache, attn_metadata: AscendSFADCPMetadata) -> tuple[torch.Tensor, ...]:
        dcp_context = attn_metadata.dcp_context
        assert dcp_context is not None
        gather_context = dcp_context.gather_context
        dcp_context.gather_context = None
        if gather_context is None:
            # Normal forward starts this after cache writes to overlap indexer
            # work. Direct attention callers still need to start it here.
            self._record_dcp_kv_gather_context(kv_cache, attn_metadata)
            gather_context = dcp_context.gather_context
            dcp_context.gather_context = None
        assert gather_context is not None
        return self._finish_dcp_gather(gather_context)

    def _all_gather_dim_async(
        self,
        x: torch.Tensor,
        dim: int,
    ) -> tuple[torch.Tensor, torch.distributed.Work | None, tuple[int, ...] | None]:
        assert self.dcp_group is not None
        if dim == 0:
            gathered, handle = all_gather_async(x.contiguous(), self.dcp_group)
            return gathered, handle, None

        perm = (dim, *[i for i in range(x.dim()) if i != dim])
        restore_perm = tuple(perm.index(i) for i in range(x.dim()))
        gathered, handle = all_gather_async(x.permute(perm).contiguous(), self.dcp_group)
        return gathered, handle, restore_perm

    def _remap_sparse_indices(self, topk_indices: torch.Tensor) -> torch.Tensor:
        if self.dcp_size <= 1:
            return topk_indices

        topk_count = topk_indices.shape[-1]
        if topk_count > self._dcp_index_topk:
            raise RuntimeError(
                f"topk_indices last dimension ({topk_count}) exceeds configured index_topk ({self._dcp_index_topk})."
            )
        if topk_indices.numel() == 0:
            return topk_indices

        if HAS_TRITON and topk_indices.is_npu:
            from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton

            return remap_sparse_indices_triton(
                topk_indices,
                self.dcp_size,
                self.dcp_rank,
                self._dcp_interleave_size,
            )

        # Fallback for environments without Triton: remap the topk indices from
        # the replicated view to the DCP-local KV cache view. We use float32 for
        # better performance on Ascend.
        topk_indices_fp32 = topk_indices.to(torch.float32)
        interleave_size = self._dcp_interleave_size
        local_block_indices = torch.floor(topk_indices_fp32 / interleave_size)
        local_owner_base = torch.floor(local_block_indices / self.dcp_size) * self.dcp_size
        local_owner = local_block_indices - local_owner_base
        local_owner_mask = (topk_indices_fp32 >= 0) & (local_owner == self.dcp_rank)
        if interleave_size == 1:
            remapped_indices_fp32 = torch.floor(topk_indices_fp32 / self.dcp_size)
        else:
            local_offsets = topk_indices_fp32 - local_block_indices * interleave_size
            remapped_indices_fp32 = torch.floor(topk_indices_fp32 / (self.dcp_size * interleave_size))
            remapped_indices_fp32 = remapped_indices_fp32 * interleave_size + local_offsets
        remapped_indices = torch.where(
            local_owner_mask,
            remapped_indices_fp32,
            self._remap_invalid_index,
        ).to(topk_indices.dtype)

        # Compact local indices to the front without changing their top-k order.
        original_order = self._remap_order[:topk_count].expand_as(topk_indices)
        pack_keys = original_order + (~local_owner_mask).to(torch.float32) * topk_count
        _, pack_order = torch.sort(pack_keys, dim=-1)
        return torch.gather(remapped_indices, dim=-1, index=pack_order.to(torch.int32))

    def _merge_dcp_outputs(
        self,
        sfa_output: torch.Tensor,
        softmax_lse: torch.Tensor,
    ) -> torch.Tensor:
        # The replicated MTP draft disables logical PCP, but still shares the
        # target's physical PCP ranks and DCP-sharded KV cache.
        if get_pcp_group().world_size > 1:
            pcp_group = get_pcp_group()
            tp_group = get_tp_group()
            tp_size = tp_group.world_size if self.dcp_size > pcp_group.world_size else 1
            return torch.ops.vllm.dcp_a2a_fused(
                sfa_output, softmax_lse, tp_size, 1, tp_group.unique_name, pcp_group.unique_name
            )

        assert self.dcp_group is not None, "DCP output All2All requires dcp_group when dcp_size > 1."
        return torch.ops.vllm.dcp_a2a_fused(
            sfa_output,
            softmax_lse,
            self.dcp_size,
            1,
            self.dcp_group.unique_name,
        )

    def _start_dcp_query_gather(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor | None,
    ) -> DCPGatherContext:
        if q_pe is None:
            q_pe = ql_nope[..., :0]
        assert self.dcp_group is not None, "DCP query gather requires dcp_group when dcp_size > 1."
        if ql_nope.shape[:-1] != q_pe.shape[:-1] or ql_nope.dtype != q_pe.dtype:
            raise RuntimeError(
                "Cannot fuse DCP query gather for ql_nope/q_pe with "
                f"shapes {tuple(ql_nope.shape)} / {tuple(q_pe.shape)} "
                f"and dtypes {ql_nope.dtype} / {q_pe.dtype}."
            )

        # Query heads are replicated across physical PCP ranks, including
        # when the MTP draft has logical PCP disabled. Gather only distinct
        # TP head shards; DCP == PCP already owns one complete TP shard.
        if get_pcp_group().world_size > 1:
            fused_q = torch.cat([ql_nope, q_pe], dim=-1).transpose(0, 1).contiguous()
            if self.dcp_size == get_pcp_group().world_size:
                gathered, handle = fused_q, None
            else:
                gathered, handle = all_gather_async(fused_q, get_tp_group())
            return DCPGatherContext(
                gathered=gathered,
                handle=handle,
                restore_perm=(1, 0, 2),
                split_sizes=(ql_nope.shape[-1], q_pe.shape[-1]),
            )

        # Decode-only callers reach this method; PCP+DCP keeps
        # its separate TP-group override.
        nope_dim, rope_dim = ql_nope.shape[-1], q_pe.shape[-1]
        if (
            ql_nope.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and ql_nope.shape[0] > 0
            and nope_dim > 0
            and rope_dim > 0
            and nope_dim & (nope_dim - 1) == 0  # Each fragment uses a full tl.arange vector.
            and rope_dim & (rope_dim - 1) == 0
        ):
            prepared = prepare_query_head_major(ql_nope, q_pe)
            gathered, handle = all_gather_async(prepared, self.dcp_group)
            return DCPGatherContext(
                gathered=gathered,
                handle=handle,
                restore_perm=(1, 0, 2),
                split_sizes=(nope_dim, rope_dim),
                query_prepared=True,
            )

        # Avoid back-to-back DCP all_gather calls for the two SFA query
        # fragments. On Ascend the separate gathers can leave SFA with an
        # incomplete stream dependency on the first prefill. Native DCP
        # restores query shards on dim 1.
        if getattr(ql_nope, "is_npu", False):
            try:
                from vllm_ascend.ops.triton.query_gather_prep import prep_query_head_major

                head_major = prep_query_head_major(ql_nope, q_pe)
            except Exception:
                # Hard kernel failure: route to the torch assembly below.
                head_major = None
            if head_major is not None:
                gathered, handle = all_gather_async(head_major, self.dcp_group)
                return DCPGatherContext(
                    gathered=gathered,
                    handle=handle,
                    restore_perm=(1, 0, 2),
                    split_sizes=(ql_nope.shape[-1], q_pe.shape[-1]),
                )
            # prep_query_head_major returned None (shape not provably legal
            # for the fast kernel): fall back to the torch assembly below.
        fused_q = torch.cat([ql_nope, q_pe], dim=-1).contiguous()
        return self._start_dcp_gather(
            fused_q,
            dim=1,
            split_sizes=(ql_nope.shape[-1], q_pe.shape[-1]),
        )

    def _mask_nope_dcp_padding(self, output, query_lens, softmax_max=None, softmax_sum=None):
        if self.qk_rope_head_dim == 0:
            valid = torch.arange(output.shape[0], device=output.device) < query_lens[-1]
            output = output.masked_fill(~valid[:, None, None], 0)
            if softmax_max is not None:
                softmax_max = softmax_max.masked_fill(~valid[None, :, None], 0)
                softmax_sum = softmax_sum.masked_fill(~valid[None, :, None], 1)
        return output, softmax_max, softmax_sum

    def _execute_tq_dcp_sfa(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
    ):
        # Mirrors AscendSFAImpl for the non-DCP case, and additionally asks for
        # softmax LSE so decode can merge partial results across CP ranks.
        dcp_context = attn_metadata.dcp_context

        if self._has_prefill(attn_metadata):
            # Prefill keeps Q local and attends over the gathered KV, so no LSE.
            gathered_kv_cache = self._finish_dcp_kv_gather(kv_cache, attn_metadata)
            block_table = dcp_context.kv_gather_block_table
            assert block_table is not None
            attn_out, _, _ = self._turboquant_sfa(
                self._tq_rotate_query(ql_nope, q_pe),
                gathered_kv_cache[0],
                topk_indices,
                block_table,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                sparse_mode=3,
            )
            assert self.turboquant is not None
            output = self.turboquant.inverse(attn_out)
            return self._mask_nope_dcp_padding(output, actual_seq_lengths_query)[0]

        # Decode attends over this rank's KV shard only, so the partial results
        # are recombined by _merge_dcp_outputs using the LSE.
        gather_context = dcp_context.gather_context
        dcp_context.gather_context = None
        if gather_context is None:
            gather_context = self._start_dcp_query_gather(ql_nope, q_pe)
        topk_indices = self._remap_sparse_indices(topk_indices)
        ql_nope, q_pe = self._finish_dcp_gather(gather_context)
        attn_out, softmax_max, softmax_sum = self._turboquant_sfa(
            self._tq_rotate_query(ql_nope, q_pe),
            kv_cache[0],
            topk_indices,
            dcp_context.block_table,
            actual_seq_lengths_query,
            dcp_context.seq_lens,
            sparse_mode=0,
            return_softmax_lse=True,
        )
        assert softmax_max is not None
        assert softmax_sum is not None
        attn_out, softmax_max, softmax_sum = self._mask_nope_dcp_padding(
            attn_out, actual_seq_lengths_query, softmax_max, softmax_sum
        )
        softmax_lse = softmax_max.to(torch.float32) + torch.log(softmax_sum.to(torch.float32))
        softmax_lse = softmax_lse.permute(1, 0, 2).reshape(softmax_lse.shape[1], -1, 1)
        output_dtype = attn_out.dtype
        output = self._merge_dcp_outputs(attn_out, softmax_lse)
        # The merge runs in Hadamard space, so rotate back once at the end.
        assert self.turboquant is not None
        output = self.turboquant.inverse(output)
        return output.to(output_dtype)

    def _execute_dcp_sparse_flash_attention_process(
        self,
        ql_nope,
        q_pe,
        kv_cache,
        topk_indices,
        attn_metadata,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
    ):
        assert attn_metadata.dcp_context is not None, "DCP SFA requires attn_metadata.dcp_context."
        assert self.dcp_group is not None, "DCP SFA requires dcp_group when dcp_size > 1."
        dcp_context = attn_metadata.dcp_context
        if self.qk_rope_head_dim == 0:
            kv_cache = (_view_cache_as_operator_pages(kv_cache[0], attn_metadata.block_size), *kv_cache[1:])
            if q_pe is None and self.enable_sparse_sfa_turboquant:
                q_pe = ql_nope[..., :0]
        if self.enable_sparse_sfa_turboquant:
            return self._execute_tq_dcp_sfa(
                ql_nope,
                q_pe,
                kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
            )
        if self._has_prefill(attn_metadata):
            # Splitting the packed KV buffer leaves non-contiguous inner strides.
            gathered_kv_cache = tuple(t.contiguous() for t in self._finish_dcp_kv_gather(kv_cache, attn_metadata))
            block_table = dcp_context.kv_gather_block_table
            assert block_table is not None
            # The gathered KV cache is complete, so each rank can attend with
            # its local Q heads/tokens directly; no Q all-gather, sparse-index
            # remap, LSE, or output all-to-all merge is required.
            attn_output = DeviceOperator.execute_sparse_flash_attention_process(
                self,
                ql_nope,
                q_pe,
                gathered_kv_cache,
                topk_indices,
                attn_metadata,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                block_table=block_table,
                sparse_mode=3,
                return_lse=False,
            )
            return self._mask_nope_dcp_padding(attn_output, actual_seq_lengths_query)[0]

        gather_context = dcp_context.gather_context
        dcp_context.gather_context = None
        if gather_context is None:
            gather_context = self._start_dcp_query_gather(ql_nope, q_pe)
        topk_indices = self._remap_sparse_indices(topk_indices)
        # Keep the gathered query as a strided view of the head-major gather
        # output. The split query fragments handed to npu_sparse_flash_
        # attention have always been strided views (torch.split never copies),
        # so the old permute+contiguous call only materialized the t-major
        # [T, H, D] storage behind the views;
        # additional-config sfa_dcp_force_tmajor_restore=true keeps the
        # t-major materialization selectable for comparison runs.
        ql_nope, q_pe = self._finish_dcp_gather(
            gather_context,
            keep_view=not enable_sfa_dcp_force_tmajor_restore(),
        )
        sfa_output, softmax_max, softmax_sum = DeviceOperator.execute_sparse_flash_attention_process(
            self,
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            dcp_context.seq_lens,
            block_table=dcp_context.block_table,
            # The replicated-view indexer already applies the causal visibility rule.
            # After DCP remaps topk indices to local KV positions, local KV
            # length no longer shares the same coordinate system as global
            # query length, so SFA must not apply its right-down causal crop.
            sparse_mode=0,
            return_lse=True,
        )
        sfa_output, softmax_max, softmax_sum = self._mask_nope_dcp_padding(
            sfa_output, actual_seq_lengths_query, softmax_max, softmax_sum
        )
        softmax_lse = softmax_max + torch.log(softmax_sum)
        softmax_lse = softmax_lse.permute(1, 0, 2).reshape(softmax_lse.shape[1], -1, 1)
        output_dtype = sfa_output.dtype
        output = self._merge_dcp_outputs(
            sfa_output,
            softmax_lse,
        )
        return output.to(output_dtype)

    def _finalize_o_proj(
        self,
        attn_output: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self._parallel_o_proj is not None:
            return self._parallel_o_proj.forward(attn_output, output)
        return self._project_o_proj_full(attn_output, output)

    def _project_o_proj_full(self, attn_output: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        output[...] = self.o_proj(attn_output)[0]
        return output

    def _get_sfa_kv_slot_mapping(
        self,
        attn_metadata: M,
    ) -> torch.Tensor:
        if self.dcp_enabled:
            assert isinstance(attn_metadata, AscendSFADCPMetadata)
            assert attn_metadata.dcp_context is not None
            slots = attn_metadata.dcp_context.slot_mapping
            # PCP stores gathered KV with the mapping prepared by the builder.
            if self.pcp_enabled and self._has_prefill(attn_metadata):
                return attn_metadata.slot_mapping
            return slots[: attn_metadata.num_input_tokens]
        if self.pcp_enabled:
            assert attn_metadata.pcp_slot_mapping is not None
            return attn_metadata.pcp_slot_mapping
        return attn_metadata.slot_mapping

    def _compose_sfa_kv_cache(self, kv_cache) -> tuple[torch.Tensor, ...] | None:
        """Compose split cache handles into the tuple expected by SFA kernels.

        ``kv_cache`` contains only the main MLA cache owned by the attention
        layer, while ``self.indexer.k_cache.kv_cache`` contains the cache owned
        by the indexer layer. Their possible layouts are:

        - neither cache uses C8:
          main ``(k_cache, v_cache)`` + indexer ``(indexer_k_cache,)``
          -> ``(k_cache, v_cache, indexer_k_cache)``
        - packed main cache only (C8 or TQ4):
          main ``(packed_kv_cache,)`` + indexer ``(indexer_k_cache,)``
          -> ``(packed_kv_cache, indexer_k_cache)``
        - LI C8 only:
          main ``(k_cache, v_cache)`` +
          indexer ``(indexer_k_cache, indexer_scale_cache)``
          -> ``(k_cache, v_cache, indexer_k_cache, indexer_scale_cache)``
        - packed main cache and LI C8:
          main ``(packed_kv_cache,)`` +
          indexer ``(indexer_k_cache, indexer_scale_cache)``
          -> ``(packed_kv_cache, indexer_k_cache, indexer_scale_cache)``

        Static shared-index layers have no runtime indexer cache;
        for those layers, the main cache tuple is returned unchanged.
        """
        # TODO: Remove this recomposition once SFA kernels accept split
        # main/indexer cache handles directly. The allocator now owns them as
        # separate cache specs, while the current kernel path still expects the
        # legacy combined tuple layout.
        main_cache = kv_cache
        if main_cache is None:
            return None
        if self.qk_rope_head_dim == 0:
            # The page-strided allocator may retain an empty RoPE view for
            # the common MLA layout. It owns no storage and is not an input
            # to the NoPE operator.
            if len(main_cache) == 2 and main_cache[1].numel() == 0:
                return (main_cache[0],)
            if len(main_cache) != 1:
                raise RuntimeError("NoPE SFA requires one latent KV cache tensor.")
            # The indexer owns and consumes its own cache. No LightningIndexer
            # cache layout is imposed on this attention operator.
            return main_cache
        if not self.runtime_has_indexer:
            return main_cache

        # Sparse KV offload registers the main MLA cache as a 6-tuple
        # (k_npu, v_npu, k_cpu, v_cpu, topk_buffer_k, topk_buffer_v); the
        # attention kernels only consume the leading NPU pair.
        if len(main_cache) == OFFLOAD_KV_CACHE_TUPLE_LEN:
            main_cache = (main_cache[OFFLOAD_K_CACHE_NPU_INDEX], main_cache[OFFLOAD_V_CACHE_NPU_INDEX])

        indexer_cache = self.indexer.k_cache.kv_cache
        if indexer_cache is None:
            raise RuntimeError(f"SFA indexer cache is not initialized or bound. layer_name={self.layer_name}.")

        expected_main_tensors = 1 if self.uses_packed_sfa_main_cache else 2
        if len(main_cache) != expected_main_tensors:
            raise RuntimeError(
                f"SFA main cache expects {expected_main_tensors} tensor(s), "
                f"got {len(main_cache)} for layer_name={self.layer_name}."
            )

        expected_indexer_tensors = self.indexer.num_cache_tensors
        if len(indexer_cache) != expected_indexer_tensors:
            raise RuntimeError(
                f"SFA indexer cache expects {expected_indexer_tensors} tensor(s), "
                f"got {len(indexer_cache)} for layer_name={self.layer_name}."
            )
        return (*main_cache, *indexer_cache)

    def _get_indexer_attn_metadata(self) -> Any | None:
        """Fetch the indexer cache layer's own metadata, built by the indexer
        backend's builder; ``None`` when this layer has no runtime indexer."""
        if not self.runtime_has_indexer:
            return None
        own_prefix = self.indexer.k_cache.prefix
        prefixes = [own_prefix]
        kv_sharing_target = getattr(self, "kv_sharing_target_layer_name", None)
        if kv_sharing_target is not None:
            # Prefer the sharing target's cache view, but keep the draft
            # indexer's own independently built metadata as a valid fallback.
            # Some proposers register the draft indexer as its own metadata
            # dependency even when the physical cache is shared.
            target_base = kv_sharing_target.removesuffix(".attn")
            prefixes = [f"{target_base}.indexer.k_cache", own_prefix]
        forward_metadata = get_forward_context().attn_metadata
        indexer_metadata = None
        if isinstance(forward_metadata, dict):
            for prefix in prefixes:
                indexer_metadata = forward_metadata.get(prefix)
                if indexer_metadata is not None:
                    break
        if indexer_metadata is None:
            raise RuntimeError(
                f"No metadata was built for the indexer cache layer prefixes={prefixes}. layer_name={self.layer_name}."
            )
        return indexer_metadata

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if self._parallel_o_proj is None:
            return self._forward_impl(layer_name, hidden_states, kv_cache, attn_metadata, output)
        # A short prefill shard can look like decode locally. Preserve the
        # global prefill decision so every PCP rank selects the same mode.
        has_prefill = attn_metadata is not None and (
            attn_metadata.pcp_has_global_prefill
            or attn_metadata.attn_state
            not in {
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            }
        )
        needs_global_metadata = attn_metadata is not None and (has_prefill or self.is_pcp_decode_sharded)
        with self._parallel_o_proj.batch(needs_global_metadata=needs_global_metadata, has_prefill=has_prefill):
            return self._forward_impl(layer_name, hidden_states, kv_cache, attn_metadata, output)

    def _forward_impl(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if self.enable_sparse_sfa_turboquant:
            assert self.turboquant is not None
            # Build the transform tensors outside graph capture.
            self.turboquant._initialize(hidden_states.device)
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        if self.qk_rope_head_dim == 0:
            num_tokens = min(hidden_states.shape[0], attn_metadata.slot_mapping.shape[0])
            if get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.FULL:
                if self.vllm_config.parallel_config.prefill_context_parallel_size > 1 and needs_pcp_kv_gather(
                    attn_metadata, self.is_pcp_decode_sharded
                ):
                    # Empty and uneven PCP ranks must keep the same padded
                    # rows for KV gathers and O-projection collectives.
                    num_tokens = min(num_tokens, attn_metadata.num_input_tokens)
                else:
                    num_tokens = min(num_tokens, attn_metadata.num_actual_tokens)
            if num_tokens == 0:
                return output.zero_()
            hidden_states = hidden_states[:num_tokens]
        gate_hidden_states = hidden_states if self.g_proj is not None else None

        composed_kv_cache = self._compose_sfa_kv_cache(kv_cache)
        assert composed_kv_cache is not None
        kv_cache = composed_kv_cache

        cos = attn_metadata.cos
        sin = attn_metadata.sin
        slot_mapping_sfa = self._get_sfa_kv_slot_mapping(attn_metadata)
        indexer_attn_metadata = self._get_indexer_attn_metadata()

        # Inputs and outputs may be padded for CUDA graphs
        num_input_tokens = hidden_states.shape[0] if self.qk_rope_head_dim == 0 else attn_metadata.num_input_tokens
        actual_seq_lengths_query = attn_metadata.cum_query_lens
        actual_seq_lengths_key = attn_metadata.seq_lens
        topk_num_tokens = num_input_tokens or hidden_states.shape[0]
        if self._parallel_o_proj is not None:
            self._parallel_o_proj.all_gather_weights()

        fused_type: PreprocessType = self.preprocess_type
        # PROLOG_V3 serves every attention state (decode, spec decoding and
        # prefill); only MLAPO carries a per-call token-count limit.
        if self.preprocess_type == PreprocessType.MLAPO and num_input_tokens > MLAPO_MAX_SUPPORTED_TOKENS:
            fused_type = PreprocessType.NATIVE

        # IMPORTANT!
        # NOTE: With PIECEWISE ACL graphs, this method is executed in
        # eager-mode PyTorch. Even view and slice operations can add CPU
        # overhead without launching NPU kernels. Minimize PyTorch operations
        # in this method and benchmark changes to avoid regressions.
        # Inputs and outputs may contain DP or graph padding. Keep the
        # preallocated output for the caller while running attention only on
        # the token rows described by metadata, matching upstream backends.
        hidden_states = hidden_states[:num_input_tokens]
        if gate_hidden_states is not None:
            gate_hidden_states = gate_hidden_states[:num_input_tokens]

        if fused_type == PreprocessType.NATIVE and gate_hidden_states is not None:
            gate_hidden_states = hidden_states

        if fused_type != PreprocessType.NATIVE:
            # Keep the raw hidden states for the indexer's k path: the fused
            # preprocess below returns new tensors and does not modify this
            # one in place.
            k_hidden_states = hidden_states if self.runtime_has_indexer else None
            wait_for_kv_layer_from_connector(layer_name)
            if self.layerwise_kv_cache_hook is not None:
                # The fused preprocess is the first operation that may read or
                # update the paged SFA and LI caches.
                self.layerwise_kv_cache_hook.wait_for_layer(layer_name)

            gather_pcp_kv = self.pcp_enabled and needs_pcp_kv_gather(attn_metadata, self.is_pcp_decode_sharded)
            fused_slots = slot_mapping_sfa
            if gather_pcp_kv:
                fused_slots = attn_metadata.pcp_prolog_local_slots
                assert fused_slots is not None and attn_metadata.pcp_prolog_global_slots is not None, (
                    "PCP fused preprocessing requires slots prepared by the metadata builder."
                )
            if fused_type == PreprocessType.PROLOG_V3:
                if self.pcp_enabled and not gather_pcp_kv:
                    fused_slots = fused_slots[: hidden_states.shape[0]]
                hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_prolog_v3(
                    hidden_states=hidden_states,
                    kv_cache=kv_cache,
                    cos=cos,
                    sin=sin,
                    # npu_mla_prolog_v3 requires int64 cache indices; reuse
                    # the per-step conversion so all layers of a step share
                    # one Cast kernel (the .to() inside is a no-op on int64).
                    slot_mapping=_int64_kv_slots(fused_slots, attn_metadata),
                    attn_metadata=attn_metadata,
                )
            else:
                hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_mlapo(
                    hidden_states=hidden_states,
                    kv_cache=kv_cache,
                    cos=cos,
                    sin=sin,
                    slot_mapping=fused_slots,
                    num_input_tokens=num_input_tokens,
                    attn_metadata=attn_metadata,
                )
            if gather_pcp_kv:
                self._sync_pcp_kv_cache(kv_cache, attn_metadata)
        # native
        else:
            assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_no_split = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized"
            q_c = self.q_a_layernorm(q_c)

            # The prepared hidden states feed the indexer's k path (same stage
            # as the weights path input).
            k_hidden_states = hidden_states if self.runtime_has_indexer else None

            wait_for_kv_layer_from_connector(layer_name)
            if self.layerwise_kv_cache_hook is not None:
                # Q/KV projections above overlap the full-layer broadcast.
                # Wait before the first main or indexer cache access.
                self.layerwise_kv_cache_hook.wait_for_layer(layer_name)

            kv_outputs = self.exec_kv(
                kv_no_split,
                cos,
                sin,
                kv_cache,
                slot_mapping_sfa,
                attn_metadata,
            )
            k_pe, k_nope = kv_outputs[:2]
            knope_scale = kv_outputs[2] if len(kv_outputs) == 3 else None
            ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
            if self.qk_rope_head_dim:
                q_pe = self.rope_single(q_pe, cos, sin)
            if self.dcp_enabled and not self._has_prefill(attn_metadata):
                assert isinstance(attn_metadata, AscendSFADCPMetadata)
                assert attn_metadata.dcp_context is not None, "DCP SFA requires attn_metadata.dcp_context."
                attn_metadata.dcp_context.gather_context = self._start_dcp_query_gather(ql_nope, q_pe)

            if self.uses_packed_sfa_main_cache:
                self._store_packed_kv_cache(k_pe, k_nope, knope_scale, kv_cache, slot_mapping_sfa)
            if self.dcp_enabled and self._has_prefill(attn_metadata):
                # Gather the newly written KV while indexer selection runs.
                self._record_dcp_kv_gather_context(kv_cache, attn_metadata)

        # A sharded-decode PCP rank can own no request but must still write
        # and gather KV with the other ranks.
        has_local_query = attn_metadata.num_actual_tokens > 0
        if self.runtime_has_indexer:
            # One unified indexer call: k path -> cache write -> top-k
            # selection (the selection kernel reads the freshly written
            # cache). MTP skip_topk layers still run the k path and the write
            # (compute_topk=False) so their cache stays up to date, then
            # reuse the shared top-k indices. Cache layout, RoPE, parallel
            # sequence lengths, and decode count all come from the indexer's
            # independently built metadata.
            assert k_hidden_states is not None
            assert indexer_attn_metadata is not None
            self._prepare_indexer_metadata(indexer_attn_metadata, attn_metadata)
            topk_indices = self.indexer(
                hidden_states,
                q_c,
                k_hidden_states,
                indexer_attn_metadata,
                compute_topk=not self.skip_topk and has_local_query,
            )
            if self.skip_topk:
                topk_indices = self._get_indexcache_topk_indices(topk_num_tokens)
            elif self.use_index_cache and has_local_query:
                self._update_indexcache_topk_indices(topk_indices)
        elif self.skip_topk:
            # Static shared-index layers keep no runtime indexer cache and
            # only reuse the shared top-k indices.
            topk_indices = self._get_indexcache_topk_indices(topk_num_tokens)
        else:
            raise RuntimeError(f"skip_topk is False but indexer is None. layer_name={self.layer_name}.")

        # Notify for every layer that wrote the cache, not just indexer layers:
        # by this point all of the layer's KV (main + indexer) has been
        # scattered - indexer layers persisted it inside indexer.forward
        # above - so the connector can dispatch the PD pull.
        notify_kv_cache_written(self.layer_name or "")

        # Open the prefetch gate for every SFA layer. Some GLM-5.2 layers
        # reuse cached top-k indices and have no indexer, so recording this
        # inside the indexer's forward would leave their gate closed.
        with attention_transfer_window():
            if has_local_query:
                attn_output = self._execute_sparse_flash_attention_process(
                    ql_nope,
                    q_pe,
                    kv_cache,
                    topk_indices,
                    attn_metadata,
                    actual_seq_lengths_query,
                    actual_seq_lengths_key,
                )

        if has_local_query:
            attn_output = self._v_up_proj(attn_output)
        else:
            # The O-proj below still joins its PCP/TP collectives.
            attn_output = hidden_states.new_zeros((ql_nope.shape[0], self.local_num_heads * self.v_head_dim))
        if gate_hidden_states is not None:
            assert self.g_proj is not None
            attn_output.mul_(torch.sigmoid(self.g_proj(gate_hidden_states.contiguous())[0]))
        self._finalize_o_proj(
            attn_output,
            output[:num_input_tokens],
        )

        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

        return output


def turboquant_kv_rmsnorm_rope(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    *,
    turboquant: TurboQuantLatent,
    epsilon: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TurboQuant counterpart of custom_kv_rmsnorm_rope.

    Returns the same (k_rope, k_nope, scale) byte views, so the caller packs and
    stores a TQ4 slot exactly the way it packs and stores a C8 one.
    """
    rms_in, rope_in = kv.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
    k_nope, _ = torch_npu.npu_rms_norm(rms_in.reshape(-1, kv_lora_rank), gamma, epsilon=epsilon)
    k_rope = torch_npu.npu_interleave_rope(rope_in, cos, sin).reshape(-1, qk_rope_head_dim)

    # compress() returns one row per token: the packed nibbles followed by the two
    # fp16 bytes of s_t = ||latent|| / ||dequantized unit vector||, the factor
    # attention applies to rescale the whole slot.
    packed_bytes = get_tq_packed_bytes(kv_lora_rank)
    row = turboquant.compress(k_nope).reshape(k_nope.shape[0], -1)
    nibbles = row[:, :packed_bytes].view(torch.int8)
    scale = row[:, packed_bytes:].contiguous().view(torch.float16).reshape(-1, 1)

    # Pre-divide the rope half by s_t so attention rescales the whole slot with one
    # per-column multiply instead of a separate rope pass.
    k_rope = (k_rope.float() / (scale.float() + 1e-20)).to(torch.bfloat16).contiguous()
    return (
        k_rope.view(torch.int8),
        nibbles,
        scale.contiguous().view(torch.int8),
    )


def custom_kv_rmsnorm_rope(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    *,
    epsilon: float = 1e-5,
    dst_type: torch.dtype | int = torch.float8_e4m3fn,
    tile_size: int = SFA_QSFA_TILE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rms_in, rope_in = kv.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
    k_nope, _ = torch_npu.npu_rms_norm(rms_in, gamma, epsilon=epsilon)
    k_rope = torch_npu.npu_interleave_rope(rope_in, cos, sin) if qk_rope_head_dim else rope_in

    prefix_shape = k_nope.shape[:-1]
    # npu_rms_norm returns a contiguous tensor, so the explicit
    # .contiguous() copy before the view is redundant.
    k_nope, knope_scale = torch_npu.npu_dynamic_block_quant(
        k_nope.view(-1, 1, kv_lora_rank),
        dst_type=dst_type,
        row_block_size=1,
        col_block_size=tile_size,
    )
    if dst_type == torch.int8:
        # Return byte views so the caller can concatenate all three components.
        return (
            k_rope.contiguous().view(torch.int8) if qk_rope_head_dim else k_nope.new_empty((*prefix_shape, 0)),
            k_nope.view(*prefix_shape, kv_lora_rank),
            knope_scale.to(torch.float32).view(*prefix_shape, -1).contiguous().view(torch.int8),
        )

    # A5 transports the BF16 rope and scale bytes through FP8-typed tensors.
    return (
        k_rope.view(torch.float8_e4m3fn),
        k_nope,
        knope_scale.view(knope_scale.shape[0], -1).view(torch.float8_e4m3fn),
    )
