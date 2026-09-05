import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
import torch.distributed as dist
import torch_npu
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_attn_kv_plan import (
    get_dsa_attn_kv_plan,
    is_a5_bf16_kv_enabled,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_pcp,
    get_or_register_attention_buffer,
    maybe_save_kv_layer_to_connector,
    notify_kv_cache_written,
    split_decodes_and_prefills,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.distributed.parallel_state import get_otp_group
from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorMetadata
from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerMetadata, IndexerOverlapPlan
from vllm_ascend.ops.cv_linear import CVLinearWrapper
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa, get_full_cos_and_sin_dsa
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod
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
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_ascend.ops.triton.rms_norm import triton_q_rms

if HAS_TRITON:
    from vllm_ascend.ops.triton.rms_norm import triton_q_rms  # noqa: F811
else:
    triton_q_rms = None  # type: ignore


# The SAS and QLI metadata operators use a fixed 1024-element int32 layout.
DSA_METADATA_BUFFER_SIZE = 1024
CompressorMetadataOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

_DSV4_DSA_OVERLAP_STREAM = None
CompressorForwardOutput = tuple[torch.Tensor, torch.Tensor]
CompressorOverlapOutput = tuple[CompressorForwardOutput, torch.npu.Event]


def build_compressor_metadata_out(
    metadata: Any,
    compress_ratio: int,
    outputs: CompressorMetadataOutput,
    vllm_config: VllmConfig,
) -> None:
    assert metadata.full_compress_cos is not None
    assert metadata.full_compress_sin is not None
    assert metadata.start_pos is not None
    assert metadata.num_actual_reqs is not None
    full_compress_cos = metadata.full_compress_cos.view(
        metadata.full_compress_cos.shape[0],
        metadata.full_compress_cos.shape[-1],
    )
    full_compress_sin = metadata.full_compress_sin.view(
        metadata.full_compress_sin.shape[0],
        metadata.full_compress_sin.shape[-1],
    )
    torch.ops._C_ascend.compressor_metadata_out(
        full_compress_cos,
        full_compress_sin,
        metadata.query_start_loc,
        metadata.start_pos,
        metadata.block_table,
        metadata.storage_block_size,
        get_dsa_attn_kv_plan(vllm_config).get_dsa_compressor_slot_mapping_format(),
        compress_ratio,
        metadata.num_actual_reqs,
        *outputs,
    )


def dsv4_dsa_overlap_stream() -> torch.npu.Stream:
    global _DSV4_DSA_OVERLAP_STREAM
    if _DSV4_DSA_OVERLAP_STREAM is None:
        _DSV4_DSA_OVERLAP_STREAM = torch_npu.npu.Stream()
    return _DSV4_DSA_OVERLAP_STREAM


def _is_w8a8_dynamic(linear) -> bool:
    """True iff ``linear`` is wired up with ``AscendW8A8DynamicLinearMethod``."""
    quant_method = getattr(linear, "quant_method", None)
    if quant_method is None or isinstance(quant_method, AscendUnquantizedLinearMethod):
        return False
    inner_method = getattr(quant_method, "quant_method", None)
    return isinstance(inner_method, AscendW8A8DynamicLinearMethod)


def _has_weight_scale(linear) -> bool:
    return getattr(linear, "weight_scale", None) is not None


def _dsa_layout_kv(vllm_config: VllmConfig) -> str:
    return get_dsa_attn_kv_plan(vllm_config).layout_kv


def _dsa_swa_only_cmp_ratio(compress_ratio: int, vllm_config: VllmConfig) -> int:
    """BF16 SWA-only attention takes no compressed stream; otherwise keep main's value."""
    if is_a5_bf16_kv_enabled(vllm_config) and compress_ratio <= 1:
        return 0
    return max(compress_ratio, 1)


class AscendDSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA"

    @staticmethod
    def get_builder_cls():
        from vllm_ascend.utils import enable_dsa_cp

        use_dsa_cp = enable_dsa_cp()
        use_pcp = enable_pcp()
        if use_dsa_cp and use_pcp:
            raise ValueError("Legacy DSACP and PCP cannot be enabled at the same time.")
        if use_dsa_cp:
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder

            return AscendDSACPMetadataBuilder
        if use_pcp:
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSAPCPMetadataBuilder

            return AscendDSAPCPMetadataBuilder
        return AscendDSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        return num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def get_scale_shape(num_blocks: int, block_size: int, scale_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, scale_size

    @staticmethod
    def get_impl_cls() -> type[AttentionImplBase[Any]]:
        from vllm_ascend.utils import enable_dsa_cp

        use_dsa_cp = enable_dsa_cp()
        use_pcp = enable_pcp()
        if use_dsa_cp and use_pcp:
            raise ValueError("Legacy DSACP and PCP cannot be enabled at the same time.")
        if use_dsa_cp:
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPImpl

            return AscendDSACPImpl
        if use_pcp:
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSAPCPImpl

            return AscendDSAPCPImpl
        return AscendDSAImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [2, 4, 8, 16, 32, 64, 128]


class AscendDSAC4Backend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_C4"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # Align with upstream's logical block-size contract: Ascend's physical
        # 32/64/128-token C4 pages represent 128/256/512 raw scheduler tokens.
        return [128, 256, 512]


class AscendDSAC128Backend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_C128"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # Align with upstream's logical block-size contract: Ascend's physical
        # 32/64/128-token C128 pages represent 4096/8192/16384 raw scheduler tokens.
        return [4096, 8192, 16384]


class AscendDSASWABackend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_SWA"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [32, 64, 128]


class AscendDSAC4StateBackend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_C4_STATE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [2, 4, 8]


class AscendDSAC128StateBackend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_C128_STATE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        if get_current_hardware_profile().supports(HardwareCapability.DSA_C128_STATE_SMALL_BLOCK_SIZES):
            return [4, 8, 16]
        return [8, 16, 32]


@dataclass
class AscendDSAReqMetadata:
    """Unified metadata for all requests in one attention invocation."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor | None
    storage_block_size: int
    query_start_loc: torch.Tensor

    num_compressed_tokens: int | None = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    full_compress_sin: torch.Tensor = None
    full_compress_cos: torch.Tensor = None
    start_pos: torch.Tensor | None = None
    num_actual_reqs: int | None = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None
    compressor_metadata: CompressorMetadataOutput | None = None
    compressor_metadata_group_id: int | None = None
    attn_mask: torch.Tensor | None = None
    cu_cmp_seqlen_list: torch.Tensor = None
    ori_win_left: int | None = None
    ori_win_right: int | None = None
    dspark_swa_indices: torch.Tensor | None = None
    vision_swa_indices: torch.Tensor | None = None


@dataclass
class AscendDSAMetadata:
    """Metadata for MLACommon.
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    num_actual_tokens: int  # Number of tokens excluding padding.
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # The dimension of the attention heads
    head_dim: int | None = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    req_metadata: AscendDSAReqMetadata | None = None
    reshape_cache_event: torch.npu.Event = None

    # metadata for dsv4 indexer

    hadamard: torch.Tensor | None = None


DSAMetadataDict: TypeAlias = dict[str, AscendDSAMetadata]


@dataclass(frozen=True)
class AscendDSALayerMetadata:
    attention: AscendDSAMetadata | None
    swa: AscendDSAMetadata
    compressor: AscendCompressorMetadata | None = None
    indexer: AscendIndexerMetadata | None = None


def _require_req_metadata(metadata: AscendDSAMetadata) -> AscendDSAReqMetadata:
    assert metadata.req_metadata is not None
    return metadata.req_metadata


def get_dspark_sparse_sas_window(vllm_config: Any) -> tuple[int, int]:
    hf_config = vllm_config.model_config.hf_config
    window_size = int(hf_config.sliding_window)
    block_size = vllm_config.speculative_config.num_speculative_tokens
    return window_size + block_size - 1, 0


def _aligned_dspark_index_width(window_size: int, block_size: int, alignment: int = 128) -> int:
    min_width = int(window_size) + int(block_size)
    return ((min_width + alignment - 1) // alignment) * alignment


def build_dspark_swa_indices(
    block_table: torch.Tensor,
    num_speculative_tokens: int,
    window_size: int,
    block_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_decode_tokens: int | None = None,
    index_width: int | None = None,
    indices_output: torch.Tensor | None = None,
    buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build DSpark non-causal visible slot ids for a paged SWA cache.

    Each token in a draft block sees the trailing context window plus the
    whole current draft block. Invalid/padded rows get lens=0 and -1 slots.

    When ``buffer`` is given, the per-token slots are copied into its leading
    rows and the returned tensor is a slice view of ``buffer``. This keeps the
    address stable across async ACL-graph replays, where the DSA operator
    captures ``ori_sparse_indices``'s data pointer at capture time.
    """
    if index_width is None:
        index_width = _aligned_dspark_index_width(window_size, num_speculative_tokens)
    min_width = int(window_size) + int(num_speculative_tokens)
    if index_width < min_width:
        raise ValueError(
            "DSpark SWA index_width must cover window_size + block_size: "
            f"index_width={index_width}, required={min_width}"
        )
    if query_start_loc is None or seq_lens is None:
        raise ValueError("DSpark SWA query_start_loc and seq_lens must both be provided")

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    prefix_lens = seq_lens - query_lens
    start_pos = (prefix_lens - int(window_size)).clamp(min=0)
    visible_lens = seq_lens - start_pos

    # Per-request visible-position grid [req_count, index_width]. Columns
    # j >= visible_len are out of range and masked to -1 below.
    cols = torch.arange(index_width, device=start_pos.device)
    col_mask = cols[None, :] < visible_lens[:, None]
    pos = start_pos[:, None] + cols[None, :]
    block_nums = pos // block_size
    # Clamp to valid block-table columns so gather never goes OOB on the
    # out-of-range columns (their results are discarded by col_mask anyway).
    safe_nums = block_nums.clamp(min=0, max=int(block_table.shape[1]) - 1)
    block_offsets = pos % block_size
    block_ids = torch.gather(block_table, 1, safe_nums)
    slot_ids = (block_ids * block_size + block_offsets).to(torch.int32)
    slot_ids = slot_ids.where(col_mask, torch.full_like(slot_ids, -1))

    per_token_slots = torch.repeat_interleave(slot_ids, query_lens, dim=0, output_size=num_decode_tokens).unsqueeze(1)
    per_token_lens = torch.repeat_interleave(visible_lens, query_lens, dim=0, output_size=num_decode_tokens)

    if indices_output is not None:
        if indices_output.shape != per_token_slots.shape:
            raise ValueError(
                "DSpark SWA indices output shape does not match active metadata: "
                f"output={tuple(indices_output.shape)}, active={tuple(per_token_slots.shape)}"
            )
        indices_output.copy_(per_token_slots)
        per_token_slots = indices_output

    if buffer is not None:
        # Copy the freshly built indices into the caller-provided buffer and hand
        # back a zero-copy view of it: ACL graph capture freezes tensor addresses,
        # so the DSA operator must read from the stable buffer at replay instead of
        # a freshly allocated tensor.
        num_rows = per_token_slots.shape[0]
        assert num_rows <= buffer.shape[0], (
            f"dspark_swa_indices needs {num_rows} rows but `buffer` only has {buffer.shape[0]}"
        )
        buffer[:num_rows].copy_(per_token_slots)
        per_token_slots = buffer[:num_rows]

    return per_token_slots, per_token_lens


def build_vision_bidirectional_swa_indices(
    block_table: torch.Tensor,
    window_size: int,
    max_image_tokens: int,
    block_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    mm_prefix_ranges: dict[int, list[tuple[int, int]]],
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build paged original-KV indices with bidirectional image spans.

    Ranges are inclusive absolute token positions. Tokens outside an image
    keep the normal causal sliding window. A token inside an image sees the
    union of that causal window and its complete image span. The fixed output
    width is ``window_size + max_image_tokens`` so it is graph- and
    operator-workspace friendly.
    """
    if max_image_tokens <= 0:
        raise ValueError("max_image_tokens must be positive for vision SWA")

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    req_ids = torch.repeat_interleave(
        torch.arange(
            query_lens.shape[0],
            device=query_start_loc.device,
            dtype=torch.long,
        ),
        query_lens,
        output_size=num_tokens,
    )
    token_offsets = torch.arange(num_tokens, device=query_start_loc.device) - query_start_loc[req_ids]
    positions = seq_lens[req_ids] - query_lens[req_ids] + token_offsets
    start_positions = (positions - int(window_size) + 1).clamp_min(0)
    end_positions = positions.clone()

    for req_idx, ranges in mm_prefix_ranges.items():
        if req_idx >= query_lens.shape[0]:
            continue
        for span_start, span_end in ranges:
            if span_end < span_start:
                raise ValueError(f"Invalid image span [{span_start}, {span_end}]")
            if span_end - span_start + 1 > max_image_tokens:
                raise ValueError(
                    f"Image span exceeds vision_max_n_token: span=[{span_start}, {span_end}], max={max_image_tokens}"
                )
            in_span = (req_ids == req_idx) & (positions >= span_start) & (positions <= span_end)
            start_positions = torch.where(
                in_span,
                torch.minimum(
                    start_positions,
                    start_positions.new_full((), span_start),
                ),
                start_positions,
            )
            end_positions = torch.where(
                in_span,
                torch.maximum(
                    end_positions,
                    end_positions.new_full((), span_end),
                ),
                end_positions,
            )

    visible_lens = end_positions - start_positions + 1
    index_width = int(window_size) + int(max_image_tokens)
    columns = torch.arange(index_width, device=block_table.device)
    visible = columns.unsqueeze(0) < visible_lens.unsqueeze(1)
    visible_positions = start_positions.unsqueeze(1) + columns.unsqueeze(0)
    block_numbers = visible_positions // int(block_size)
    safe_block_numbers = block_numbers.clamp(
        min=0,
        max=block_table.shape[1] - 1,
    )
    request_block_tables = block_table[req_ids]
    block_ids = torch.gather(
        request_block_tables,
        1,
        safe_block_numbers,
    )
    slot_ids = (block_ids * int(block_size) + visible_positions % int(block_size)).to(torch.int32)
    slot_ids = slot_ids.where(visible, torch.full_like(slot_ids, -1))
    return slot_ids.unsqueeze(1), visible_lens.to(torch.int32)


class AscendDSAMetadataBuilder(AttentionMetadataBuilder[AscendDSAMetadata]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec: AscendMLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendDSAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.metadata_cls = metadata_cls if metadata_cls is not None else AscendDSAMetadata
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.logical_block_size = kv_cache_spec.block_size
        self.storage_block_size = kv_cache_spec.storage_block_size
        scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        self.spec_slot_mapping = None
        self.dspark_swa_indices_buffer: torch.Tensor | None = None
        if get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION) and not is_a5_bf16_kv_enabled(
            vllm_config
        ):
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens,)  # type: ignore
        else:
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens, 2)  # type: ignore
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.spec_slot_mapping = [
                torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)
                for _ in range(spec_token_num)
            ]
            self.spec_sas_metadata = [
                torch.zeros(
                    DSA_METADATA_BUFFER_SIZE,
                    dtype=torch.int32,
                    device=self.device,
                )
                for _ in range(spec_token_num)
            ]
            # Shared static buffer for dspark_swa_indices, so its address
            # stays stable across async ACL-graph replays.
            _dspark_index_width = _aligned_dspark_index_width(
                self.model_config.hf_config.sliding_window, spec_token_num
            )
            max_dspark_rows = max(
                scheduler_config.max_num_batched_tokens,
                scheduler_config.max_num_seqs * (self.speculative_config.num_speculative_tokens + 1),
            )
            self.dspark_swa_indices_buffer = torch.zeros(
                (max_dspark_rows, 1, _dspark_index_width),
                dtype=torch.int32,
                device=self.device,
            )
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold
        self.num_decodes = 0
        self.num_prefills = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.num_actual_tokens: int | None = None
        self.block_table: torch.Tensor = None
        self.common_ratio_to_sas_metadata: dict | None = None
        self.seq_lens: torch.Tensor = None

        self.compressor_ratio = getattr(kv_cache_spec, "compress_ratio", 0)
        self.hadamard = None
        self._init_hadamard(layer_names)
        self.start_pos_prefill: torch.Tensor = torch.zeros(
            scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device
        )
        self.sas_metadata_buffer: torch.Tensor = torch.zeros(
            DSA_METADATA_BUFFER_SIZE, dtype=torch.int32, device=self.device
        )
        self.qli_metadata_buffer: torch.Tensor = torch.zeros(
            DSA_METADATA_BUFFER_SIZE, dtype=torch.int32, device=self.device
        )
        self._device_metadata_enabled = False
        self._device_metadata_tasks: tuple[DeviceMetadataTask, ...] = ()
        self.cu_seqlens_ori_kv = torch.tensor([], device=self.device)
        self.cu_seqlens_cmp_kv = torch.tensor([], device=self.device)
        self.seqused_q = torch.tensor([], device=self.device)
        self._zero_i32 = torch.tensor([0], device=self.device, dtype=torch.int32)
        # Note(qcs): we use two dimension slot_mapping for kvcache with shape
        # [block_nums, block_size, head_num, head_dim]
        self.slot_mapping = torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)
        self.compressor_metadata_buffers: CompressorMetadataOutput | None = None

    def _init_hadamard(self, layer_names: list[str]) -> None:
        hf_config = self.model_config.hf_config
        if hf_config.model_type != "deepseek_v4":
            return

        indexer_head_dim = hf_config.index_head_dim
        try:
            from scipy.linalg import hadamard  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("Please install scipy") from e
        log_dim = math.ceil(math.log2(indexer_head_dim))
        dim_padded = 2**log_dim
        self.hadamard = get_or_register_attention_buffer(
            self.vllm_config,
            layer_names,
            "_dsa_hadamard",
            lambda: torch.tensor(hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device).to(
                torch.bfloat16
            ),
        )

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendDSAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch", scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if num_tokens <= self.decode_threshold:
                decodes.append(i)
            else:
                prefills.append(i)

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill], decodes[num_decodes - i])
                first_prefill += 1
                modified_batch = True
            else:
                break

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        return modified_batch

    def set_num_actual_tokens(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens

    def _num_compressor_metadata_rows(self, num_reqs: int) -> int:
        assert self.num_actual_tokens is not None
        num_tokens = self.num_actual_tokens
        return min(num_tokens, num_tokens // self.compressor_ratio + num_reqs)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        **kwargs,
    ) -> AscendDSAMetadata:
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            **kwargs,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_reqs = kwargs.get("num_actual_reqs")
        self.common_ratio_to_sas_metadata = kwargs.get("common_ratio_to_sas_metadata")
        assert self.common_ratio_to_sas_metadata is not None
        self.set_num_actual_tokens(common_attn_metadata)
        num_input_tokens = common_attn_metadata.num_input_tokens

        if self.common_ratio_to_sas_metadata.get("num_decodes") is None:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.decode_threshold,
                )
            )
            self.common_ratio_to_sas_metadata["num_decodes"] = self.num_decodes
            self.common_ratio_to_sas_metadata["num_prefills"] = self.num_prefills
            self.common_ratio_to_sas_metadata["num_decode_tokens"] = self.num_decode_tokens
            self.common_ratio_to_sas_metadata["num_prefill_tokens"] = self.num_prefill_tokens
            assert self.num_decodes + self.num_prefills == num_reqs
            assert self.num_decode_tokens + self.num_prefill_tokens == common_attn_metadata.num_actual_tokens
            self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
            self.common_ratio_to_sas_metadata["seq_lens"] = self.seq_lens
            # Prefer _seq_lens_cpu (always available, updated during draft
            # iterations) over seq_lens_cpu (None in async spec decode mode).
            if common_attn_metadata._seq_lens_cpu is not None:
                seq_lens_cpu = common_attn_metadata._seq_lens_cpu
            elif common_attn_metadata.seq_lens_cpu is not None:
                seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            else:
                seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
            self.common_ratio_to_sas_metadata["seq_lens_cpu"] = seq_lens_cpu
            input_positions = common_attn_metadata.positions[:num_input_tokens].long()
            cos, sin = get_cos_and_sin_dsa(
                input_positions,
                use_cache=self.num_prefills == 0,
            )
            self.common_ratio_to_sas_metadata["cos"] = cos
            self.common_ratio_to_sas_metadata["sin"] = sin
        else:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                self.common_ratio_to_sas_metadata["num_decodes"],
                self.common_ratio_to_sas_metadata["num_prefills"],
                self.common_ratio_to_sas_metadata["num_decode_tokens"],
                self.common_ratio_to_sas_metadata["num_prefill_tokens"],
            )
            self.seq_lens = self.common_ratio_to_sas_metadata["seq_lens"]
            seq_lens_cpu = self.common_ratio_to_sas_metadata["seq_lens_cpu"]
            cos = self.common_ratio_to_sas_metadata["cos"]
            sin = self.common_ratio_to_sas_metadata["sin"]

        # CommonAttentionMetadata uses logical raw-token slots. They directly
        # describe only uncompressed SWA/state caches; C4/C128 physical slots
        # are generated later from the logical block table by compressor_metadata.
        if self.compressor_ratio <= 1:
            slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
            self.slot_mapping[:num_input_tokens] = get_dsa_attn_kv_plan(self.vllm_config).format_dsa_slot_mapping(
                slot_mapping, self.storage_block_size
            )

        self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        req_metadata = self.build_req_metadata(
            common_attn_metadata=common_attn_metadata,
            seq_lens_cpu=seq_lens_cpu,
            num_actual_reqs=num_actual_reqs,
            cos=cos,
            sin=sin,
            full_graph_mode=kwargs.get("full_graph_mode", False),
        )

        return self.metadata_cls(  # type: ignore
            num_actual_tokens=self.num_actual_tokens,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_state=common_attn_metadata.attn_state,
            req_metadata=req_metadata,
            hadamard=self.hadamard,
        )

    def _build_sas_metadata(
        self,
        metadata_cache: dict,
        layer_name: str,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seqlen_q: int | torch.Tensor,
        max_seqlen_kv: int | torch.Tensor,
        cu_seqlens_ori_kv: torch.Tensor | None,
        cu_seqlens_cmp_kv: torch.Tensor | None,
    ) -> torch.Tensor:
        sas_metadata = metadata_cache.get(layer_name)
        if sas_metadata is None:
            tp_size = get_tensor_model_parallel_world_size()
            n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
            index_topk = self.model_config.hf_config.index_topk
            cmp_ratio = (
                _dsa_swa_only_cmp_ratio(self.compressor_ratio, self.vllm_config)
                if self.compressor_ratio <= 1
                else 4
                if self.compressor_ratio == 4
                else 128
            )
            kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
            metadata_op = kv_plan.get_dsa_sparse_attn_metadata_op()
            metadata_kwargs = kv_plan.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
            sas_metadata = metadata_op(
                **metadata_kwargs,
                num_heads_q=n_local_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=query_start_loc,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=self.seqused_q,
                seqused_kv=seq_lens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=len(seq_lens),
                cmp_topk=index_topk if self.compressor_ratio == 4 else 0,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=4,  # 4:sliding window
                cmp_mask_mode=3,  # 3:causal
                ori_win_left=self.model_config.hf_config.sliding_window - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv=_dsa_layout_kv(self.vllm_config),
                has_ori_kv=True,
                has_cmp_kv=self.compressor_ratio > 1,
            )
            metadata_cache[layer_name] = sas_metadata

        self.sas_metadata_buffer[:DSA_METADATA_BUFFER_SIZE] = sas_metadata
        return self.sas_metadata_buffer

    def _build_qli_metadata(
        self,
        metadata_cache: dict,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
    ) -> torch.Tensor:
        qli_metadata = metadata_cache.get("qli")
        if qli_metadata is None:
            qli_metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=query_start_loc[1:].clone(),
                actual_seq_lengths_key=seq_lens.clone(),
                num_heads_q=self.model_config.hf_config.index_n_heads,  # 64
                num_heads_k=1,
                head_dim=self.model_config.hf_config.index_head_dim,  # 128
                query_quant_mode=0,
                key_quant_mode=0,
                batch_size=len(seq_lens),
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_kv,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.model_config.hf_config.index_topk,  # 512
                sparse_mode=3,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                cmp_ratio=4,
                device=str(self.seqused_q.device),
            )
            metadata_cache["qli"] = qli_metadata

        self.qli_metadata_buffer[:DSA_METADATA_BUFFER_SIZE] = qli_metadata
        return self.qli_metadata_buffer

    def enable_device_metadata(self) -> None:
        self._device_metadata_enabled = True
        if self.compressor_ratio > 1 and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL:
            max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            output_shape = (max_tokens, 1, 1, self.model_config.hf_config.qk_rope_head_dim)
            self.compressor_metadata_buffers = (
                torch.empty(output_shape, dtype=torch.float32, device=self.device),
                torch.empty(output_shape, dtype=torch.float32, device=self.device),
                self.slot_mapping,
            )

    def enable_dspark_device_metadata(self, max_num_tokens: int) -> None:
        self.enable_device_metadata()
        assert self.speculative_config is not None
        index_width = _aligned_dspark_index_width(
            self.model_config.hf_config.sliding_window,
            self.speculative_config.num_speculative_tokens,
        )
        self.dspark_swa_indices_buffer = torch.empty(
            (max_num_tokens, 1, index_width),
            dtype=torch.int32,
            device=self.device,
        )

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        tasks = self._device_metadata_tasks
        self._device_metadata_tasks = ()
        return tasks

    def build_req_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        seq_lens_cpu: torch.Tensor,
        num_actual_reqs: int | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        full_graph_mode: bool = False,
    ) -> AscendDSAReqMetadata:
        assert self.common_ratio_to_sas_metadata is not None
        metadata_cache = self.common_ratio_to_sas_metadata
        assert self.num_actual_tokens is not None
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        seq_lens = self.seq_lens[:num_reqs]
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()
        max_seqlen_kv = torch.max(seq_lens_cpu[:num_reqs]).item()
        has_prefill = self.num_prefills > 0

        self.start_pos_prefill.fill_(0)
        self.start_pos_prefill[:num_reqs] = seq_lens - seq_lens_q
        if num_actual_reqs is None:
            num_actual_reqs = num_reqs
        else:
            num_actual_reqs = min(num_actual_reqs, num_reqs)
            if num_actual_reqs < num_reqs:
                self.start_pos_prefill[num_actual_reqs:num_reqs].fill_(0)
                self.block_table[num_actual_reqs:num_reqs, ...].fill_(0)
        layer_name = f"c{self.compressor_ratio}"
        cu_seqlens_ori_kv = None
        cu_seqlens_cmp_kv = None
        dspark_swa_indices = None
        vision_swa_indices = None
        ori_win_left, ori_win_right = self.model_config.hf_config.sliding_window - 1, 0
        if not has_prefill and not common_attn_metadata.causal:
            # DSpark non-causal parallel drafting: every draft query attends to
            # the trailing context window plus the whole current draft block.
            # Not gated on the SAS metadata cache: the indices depend on the
            # current step's block table / sequence lengths, so they must be
            # rebuilt whenever a DSpark draft step runs.
            assert self.speculative_config is not None
            dspark_swa_indices, _ = build_dspark_swa_indices(
                self.block_table[: self.num_decodes],
                self.speculative_config.num_speculative_tokens,
                self.model_config.hf_config.sliding_window,
                self.storage_block_size,
                query_start_loc[: self.num_decodes + 1],
                self.seq_lens[: self.num_decodes],
                self.num_decode_tokens,
                buffer=self.dspark_swa_indices_buffer,
            )
            ori_win_left, ori_win_right = get_dspark_sparse_sas_window(self.vllm_config)
        # Text-only requests and lightweight metadata fixtures do not carry
        # multimodal document ranges. Treat those as having no vision spans.
        mm_ranges = getattr(common_attn_metadata, "mm_req_doc_ranges", None)
        max_image_tokens = (
            getattr(
                self.model_config.hf_config,
                "vision_max_n_token",
                0,
            )
            if getattr(
                self.model_config.hf_config,
                "vision_n_layers",
                0,
            )
            > 0
            else 0
        )
        if has_prefill and max_image_tokens > 0 and mm_ranges:
            actual_reqs = num_reqs if num_actual_reqs is None else num_actual_reqs
            vision_swa_indices, _ = build_vision_bidirectional_swa_indices(
                block_table=self.block_table[:actual_reqs],
                window_size=self.model_config.hf_config.sliding_window,
                max_image_tokens=max_image_tokens,
                block_size=self.storage_block_size,
                query_start_loc=query_start_loc[: actual_reqs + 1],
                seq_lens=seq_lens[:actual_reqs],
                mm_prefix_ranges=mm_ranges,
                num_tokens=self.num_actual_tokens,
            )
        if not has_prefill and self.common_ratio_to_sas_metadata.get(layer_name) is None:
            cu_seqlens_ori_kv = DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
                self.common_ratio_to_sas_metadata,
                "cu_seqlens_ori_kv",
                seq_lens,
                num_reqs,
                self._zero_i32,
                self.cu_seqlens_ori_kv,
            )
            cu_seqlens_cmp_kv = DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)
        elif has_prefill:
            cu_seqlens_ori_kv = query_start_loc

        if self._device_metadata_enabled:

            def build_sas_metadata() -> None:
                self._build_sas_metadata(
                    metadata_cache=metadata_cache,
                    layer_name=layer_name,
                    query_start_loc=query_start_loc,
                    seq_lens=seq_lens,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                )

            def build_qli_metadata() -> None:
                self._build_qli_metadata(
                    metadata_cache=metadata_cache,
                    query_start_loc=query_start_loc,
                    seq_lens=seq_lens,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                )

            if self.compressor_ratio == 4:
                self._device_metadata_tasks = (
                    DeviceMetadataTask(DeviceMetadataStage.INDEXER, build_qli_metadata, id(self.qli_metadata_buffer)),
                    DeviceMetadataTask(DeviceMetadataStage.ATTENTION, build_sas_metadata, id(self.sas_metadata_buffer)),
                )
            else:
                self._device_metadata_tasks = (
                    DeviceMetadataTask(DeviceMetadataStage.ATTENTION, build_sas_metadata, id(self.sas_metadata_buffer)),
                )
            sas_metadata = self.sas_metadata_buffer
            qli_metadata = self.qli_metadata_buffer if self.compressor_ratio == 4 else None
        else:
            self._device_metadata_tasks = ()
            sas_metadata = self._build_sas_metadata(
                metadata_cache=metadata_cache,
                layer_name=layer_name,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            )
            qli_metadata = self._build_qli_metadata(
                metadata_cache=metadata_cache,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
            )

        full_compress_cos, full_compress_sin = None, None
        if self.compressor_ratio > 1:
            if full_graph_mode:
                num_tokens = common_attn_metadata.num_input_tokens
                num_compressed_tokens = min(num_tokens, num_tokens // self.compressor_ratio + num_reqs)
                num_actual_reqs = num_reqs
            else:
                num_compressed_tokens = self._num_compressor_metadata_rows(num_reqs)
            full_compress_cos, full_compress_sin = get_full_cos_and_sin_dsa(layer_name)
            slot_mapping = None
        else:
            num_compressed_tokens = self.num_actual_tokens
            slot_mapping = self.slot_mapping[: self.num_actual_tokens]

        req_metadata = AscendDSAReqMetadata(
            block_table=self.block_table[:num_reqs, ...],
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            storage_block_size=self.storage_block_size,
            query_start_loc=query_start_loc,
            num_compressed_tokens=num_compressed_tokens,
            sin=sin,
            cos=cos,
            full_compress_sin=full_compress_sin,
            full_compress_cos=full_compress_cos,
            start_pos=self.start_pos_prefill[:num_reqs],
            num_actual_reqs=num_actual_reqs,
            sas_metadata=sas_metadata,
            qli_metadata=qli_metadata,
            attn_mask=None,
            cu_cmp_seqlen_list=cu_seqlens_cmp_kv,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            dspark_swa_indices=dspark_swa_indices,
            vision_swa_indices=vision_swa_indices,
        )
        if self._device_metadata_enabled and self.compressor_metadata_buffers is not None:
            assert num_compressed_tokens is not None
            buffers = self.compressor_metadata_buffers
            outputs = (
                buffers[0][:num_compressed_tokens],
                buffers[1][:num_compressed_tokens],
                buffers[2][:num_compressed_tokens],
            )
            group_id = id(buffers[0])
            req_metadata.compressor_metadata = outputs
            req_metadata.compressor_metadata_group_id = group_id
            self._device_metadata_tasks = (
                DeviceMetadataTask(
                    DeviceMetadataStage.COMPRESSOR,
                    lambda: build_compressor_metadata_out(
                        req_metadata, self.compressor_ratio, outputs, self.vllm_config
                    ),
                    group_id,
                ),
                *self._device_metadata_tasks,
            )
        return req_metadata

    def build_for_drafting(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        assert self.compressor_ratio <= 1, "vLLM-Ascend only support SWA-layer for Deepseek-V4 now."
        self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.decode_threshold,
            )
        )
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens
        self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]

        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        if self.num_prefills:
            cos, sin = get_cos_and_sin_dsa(input_positions)
        else:
            cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True, draft_index=draft_index)
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        assert self.spec_slot_mapping is not None
        self.spec_slot_mapping[draft_index - 1][:num_input_tokens] = get_dsa_attn_kv_plan(
            self.vllm_config
        ).format_dsa_slot_mapping(slot_mapping, self.storage_block_size)
        req_metadata = self.build_req_metadata_for_drafting(
            draft_index=draft_index,
            common_attn_metadata=common_attn_metadata,
            cos=cos,
            sin=sin,
        )

        return self.metadata_cls(  # type: ignore
            num_actual_tokens=self.num_actual_tokens,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_state=common_attn_metadata.attn_state,
            req_metadata=req_metadata,
            hadamard=None,
        )

    def build_req_metadata_for_drafting(
        self,
        draft_index: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        cos,
        sin,
    ) -> AscendDSAReqMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        seq_lens = self.seq_lens[:num_reqs]
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        else:
            seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        max_seqlen_kv = torch.max(seq_lens_cpu[:num_reqs]).item()
        has_prefill = self.num_prefills > 0

        dspark_swa_indices = None
        build_dspark_swa = None
        ori_win_left = self.model_config.hf_config.sliding_window - 1
        ori_win_right = 0
        if not common_attn_metadata.causal:
            assert self.speculative_config is not None
            dspark_swa_args = (
                self.block_table[:num_reqs],
                self.speculative_config.num_speculative_tokens,
                self.model_config.hf_config.sliding_window,
                self.storage_block_size,
                query_start_loc,
                seq_lens,
                self.num_actual_tokens,
            )
            if self._device_metadata_enabled and not has_prefill:
                if self.dspark_swa_indices_buffer is None:
                    raise RuntimeError(
                        "DSpark device metadata buffers must be initialized before building draft attention metadata"
                    )
                if self.num_actual_tokens > self.dspark_swa_indices_buffer.shape[0]:
                    raise ValueError(
                        "DSpark SWA metadata rows exceed the persistent buffer capacity: "
                        f"active={self.num_actual_tokens}, capacity={self.dspark_swa_indices_buffer.shape[0]}"
                    )
                dspark_swa_indices = self.dspark_swa_indices_buffer[: self.num_actual_tokens]
                build_dspark_swa = lambda: build_dspark_swa_indices(
                    *dspark_swa_args,
                    indices_output=dspark_swa_indices,
                )
            else:
                dspark_swa_indices, _ = build_dspark_swa_indices(*dspark_swa_args)
                dspark_swa_indices = dspark_swa_indices[: self.num_actual_tokens]
            ori_win_left, ori_win_right = get_dspark_sparse_sas_window(self.vllm_config)

        cu_seqlens_ori_kv = (
            query_start_loc
            if has_prefill
            else DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
                None,
                "draft_cu_seqlens_ori_kv",
                seq_lens,
                num_reqs,
                self._zero_i32,
                self.cu_seqlens_ori_kv,
            )
        )
        cu_seqlens_cmp_kv = (
            None if has_prefill else DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)
        )
        kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
        metadata_op = kv_plan.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = kv_plan.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size

        def build_attention_metadata() -> torch.Tensor:
            if build_dspark_swa is not None:
                build_dspark_swa()
            result = metadata_op(
                **metadata_kwargs,
                num_heads_q=n_local_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=query_start_loc,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=self.seqused_q,
                seqused_kv=seq_lens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=num_reqs,
                cmp_ratio=1,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q="TND",
                layout_kv=_dsa_layout_kv(self.vllm_config),
                has_ori_kv=True,
                has_cmp_kv=False,
            )
            if not has_prefill:
                assert self.spec_sas_metadata is not None
                self.spec_sas_metadata[draft_index - 1][:DSA_METADATA_BUFFER_SIZE].copy_(
                    result[:DSA_METADATA_BUFFER_SIZE]
                )
                return self.spec_sas_metadata[draft_index - 1]
            return result

        if build_dspark_swa is not None:
            sas_metadata = self.spec_sas_metadata[draft_index - 1]

            def run_attention_metadata() -> None:
                build_attention_metadata()

            self._device_metadata_tasks = (
                DeviceMetadataTask(DeviceMetadataStage.ATTENTION, run_attention_metadata, id(sas_metadata)),
            )
        else:
            self._device_metadata_tasks = ()
            sas_metadata = build_attention_metadata()

        assert self.spec_slot_mapping is not None
        slot_mapping = self.spec_slot_mapping[draft_index - 1][: self.num_actual_tokens]
        return AscendDSAReqMetadata(
            block_table=self.block_table[:num_reqs, ...],
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            storage_block_size=self.storage_block_size,
            query_start_loc=query_start_loc,
            num_compressed_tokens=self.num_actual_tokens,
            sin=sin,
            cos=cos,
            start_pos=self.seq_lens[:num_reqs] - seq_lens_q,
            num_actual_reqs=num_reqs,
            sas_metadata=sas_metadata,
            qli_metadata=None,
            attn_mask=None,
            cu_cmp_seqlen_list=cu_seqlens_cmp_kv,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            dspark_swa_indices=dspark_swa_indices,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        **kwargs,
    ):
        if attn_state in {AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding}:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and SpecDecoding state"
            )

        assert attn_metadata is not None
        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendDSAImpl(AttentionImplBase[Any]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        **kwargs,
    ):
        self.vllm_config = kwargs["vllm_config"]
        self.num_heads = n_heads
        self.n_local_heads = n_local_heads
        self.scale = scale
        self.o_lora_rank = o_lora_rank
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.head_dim = head_dim
        self.n_group = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim**-0.5
        self.support_fp8_attention = get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION)

        # MLA Args
        self.wq_a = kwargs["wq_a"]
        self.wq_b = kwargs["wq_b"]
        self.wkv = kwargs["wkv"]
        self.q_norm = kwargs["q_norm"]
        self.q_norm_without_weight = kwargs["q_norm_without_weight"]
        self.kv_norm = kwargs["kv_norm"]

        # CV wrapper: split wq_a/wkv/wq_b into quantize(Vector) + matmul(Cube)
        self.cv_wq_a = CVLinearWrapper(self.wq_a)
        self.cv_wkv = CVLinearWrapper(self.wkv)
        self.cv_wq_b = CVLinearWrapper(self.wq_b)

        self.indexer = kwargs.get("indexer")
        self.compressor = kwargs.get("compressor")
        self.swa_cache_layer = kwargs.get("swa_cache_layer")
        assert self.swa_cache_layer is not None

        self.wo_a = kwargs["wo_a"]
        self.wo_b = kwargs["wo_b"]

        self.eps = kwargs["eps"]

        self.attn_sink = kwargs["attn_sink"]

        ascend_config = get_ascend_config()
        self.multistream_dsv4_dsa_overlap = ascend_config.multistream_dsv4_dsa_overlap
        if self.multistream_dsv4_dsa_overlap and is_a5_bf16_kv_enabled(self.vllm_config):
            self.multistream_dsv4_dsa_overlap = False

    def _get_layer_metadata(
        self,
        attn_layer_name: str,
        attn_metadata: DSAMetadataDict,
    ) -> AscendDSALayerMetadata:
        assert self.swa_cache_layer is not None
        swa_metadata = attn_metadata[self.swa_cache_layer.prefix]
        attention_metadata = None
        compressor_metadata = None
        indexer_metadata = None

        if self.compress_ratio > 1:
            assert self.compressor is not None
            attention_metadata = attn_metadata[attn_layer_name]
            compressor_metadata = AscendCompressorMetadata(
                cache=attention_metadata,
                state=attn_metadata[self.compressor.state_cache.prefix],
            )
            if self.compress_ratio == 4:
                assert self.indexer is not None
                assert self.indexer.compressor is not None
                indexer_cache_metadata = attn_metadata[self.indexer.k_cache.prefix]
                indexer_metadata = AscendIndexerMetadata(
                    compressor=AscendCompressorMetadata(
                        cache=indexer_cache_metadata,
                        state=attn_metadata[self.indexer.compressor.state_cache.prefix],
                    ),
                )

        return AscendDSALayerMetadata(
            attention=attention_metadata,
            swa=swa_metadata,
            compressor=compressor_metadata,
            indexer=indexer_metadata,
        )

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config=None,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        # dsa does not need to update graph params
        pass

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # Attention impls are not walked by vllm's process_weights_after_loading
        # dispatcher (only LinearMethodBase subclasses are). OTP buffers are
        # allocated lazily on the first _forward_o_proj call, which always runs
        # before ACL graph capture (profiling run triggers it).
        pass

    def _forward_o_proj(self, o_proj_input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        num_tokens = o_proj_input.shape[0]
        group_hidden_dim = o_proj_input.shape[1] * o_proj_input.shape[2] // self.n_local_groups
        o_proj_input = o_proj_input.view(num_tokens, self.n_local_groups, group_hidden_dim)
        # A5 (Ascend950) uses an FP8-quantized o_proj path (dynamic MX quant
        # + quantized batch matmul). Preserve it as-is: it predates and is
        # orthogonal to the OTP path below, so it must win first.
        use_a5_quant_o_proj = self.support_fp8_attention and _has_weight_scale(self.wo_a)
        if use_a5_quant_o_proj:
            o = o_proj_input
            o, swiglu_out_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
            o = torch_npu.npu_transpose_quant_batchmatmul(
                o,
                self.wo_a.weight,
                dtype=torch.bfloat16,
                bias=None,
                group_sizes=(0, 0, 32),
                x1_scale=swiglu_out_scale.view(torch.float8_e8m0fnu),
                x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            )
            o = o.reshape(num_tokens, -1)
            output[...] = self.wo_b(o)
        elif oproj_tp_enable():
            oproj_group = get_otp_group()
            oproj_tp_size = oproj_group.world_size
            if self.n_local_groups % oproj_tp_size != 0:
                raise ValueError(
                    "n_local_groups must be divisible by "
                    f"oproj_tensor_parallel_size, got {self.n_local_groups} "
                    f"and {oproj_tp_size}."
                )

            groups_per_rank = self.n_local_groups // oproj_tp_size

            o_proj_input = o_proj_input.view(num_tokens, oproj_tp_size, groups_per_rank, group_hidden_dim)
            # Pad to a static exchange size so the all_to_all / reduce_scatter
            # shapes are identical across all ACL graph buckets — variable
            # shapes desync the HCCL communicator during graph replay.
            # potential_max_tokens is computed once in the model runner __init__,
            # so reading it here is a cheap global lookup.
            exchange_num_tokens = get_potential_max_tokens()
            if exchange_num_tokens < num_tokens:
                raise ValueError(
                    "oproj static exchange capacity must cover local tokens, "
                    f"got {exchange_num_tokens} and {num_tokens}."
                )
            # Lazily allocate static send/recv buffers on first call. The
            # profiling run hits this path before ACL graph capture, so the
            # buffers exist and keep a stable device address across all later
            # capture/replay cycles (graph replay requires the same address
            # that was recorded at capture; new_zeros per call would desync
            # the HCCL operator).
            if not hasattr(self, "_oproj_send_buf"):
                buf_shape = (oproj_tp_size, exchange_num_tokens, groups_per_rank, group_hidden_dim)
                self._oproj_send_buf = torch.zeros(buf_shape, dtype=o_proj_input.dtype, device=o_proj_input.device)
                self._oproj_recv_buf = torch.empty_like(self._oproj_send_buf)
            send = self._oproj_send_buf
            recv = self._oproj_recv_buf
            # In-place fill into the address-stable buffer: zero the padding
            # tail, then copy the real tokens.
            send.zero_()
            send[:, :num_tokens].copy_(o_proj_input.transpose(1, 0))
            dist.all_to_all_single(recv.view(-1), send.view(-1), group=oproj_group.device_group)
            o_proj_input = recv.view(oproj_tp_size * exchange_num_tokens, groups_per_rank, group_hidden_dim)
            o_proj_input = torch_npu.npu_transpose_batchmatmul(
                o_proj_input,
                self.wo_a.weight,
                bias=None,
                scale=None,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
                batch_split_factor=1,
            )
            o_proj_input = o_proj_input.reshape(oproj_tp_size * exchange_num_tokens, -1)
            o_proj_output = self.wo_b(o_proj_input)
            # reduce_scatter via a raw dist collective into an address-stable
            # static buffer. oproj_group.reduce_scatter is a list-based wrapper
            # that allocates per call, which desyncs the HCCL operator recorded
            # at capture during ACL graph replay — the same reason all_to_all
            # and the embedding TP path use raw dist + static buffers.
            if not hasattr(self, "_oproj_rs_out_buf"):
                self._oproj_rs_out_buf = torch.empty(
                    (exchange_num_tokens, o_proj_output.shape[-1]),
                    dtype=o_proj_output.dtype,
                    device=o_proj_output.device,
                )
            dist.reduce_scatter_tensor(self._oproj_rs_out_buf, o_proj_output, group=oproj_group.device_group)
            output[...] = self._oproj_rs_out_buf[:num_tokens]
        else:
            # A5 BF16 wo_a is reshaped to [groups, hidden, rank] at load time,
            # matching the A3 layout expected by npu_transpose_batchmatmul.
            o_proj_input = torch_npu.npu_transpose_batchmatmul(
                o_proj_input,
                self.wo_a.weight,
                bias=None,
                scale=None,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
                batch_split_factor=1,
            )
            o_proj_input = o_proj_input.reshape(num_tokens, -1)
            output[...] = self.wo_b(o_proj_input)
        return output

    def _prepare_caches_before_attention(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataDict,
    ) -> bool:
        """Prepare cache updates and report whether local writes can be skipped."""
        return False

    def _get_o_proj_input_shape(
        self,
        attn_metadata: DSAMetadataDict | None,
    ) -> tuple[int, int, int]:
        return (
            _EXTRA_CTX.num_tokens,
            self.n_local_heads,
            self.head_dim,
        )

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, ...] | None,
        attn_metadata: DSAMetadataDict,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        output_padded = output
        o_proj_input_shape = self._get_o_proj_input_shape(attn_metadata)
        if attn_metadata is None:
            # Profiling run: run o_proj on zero input so HCCL collectives are
            # captured by the ACL graph.  Non-OTP just zeros the output.
            if oproj_tp_enable():
                o_proj_input = hidden_states.new_zeros(o_proj_input_shape)
                self._forward_o_proj(o_proj_input, output)
            else:
                output.fill_(0)
            return output
        layer_metadata = self._get_layer_metadata(layer_name, attn_metadata)
        common_attn_metadata = layer_metadata.attention
        if common_attn_metadata is None:
            common_attn_metadata = layer_metadata.swa
        actual_tokens = common_attn_metadata.num_actual_tokens

        o_proj_input = hidden_states.new_zeros(o_proj_input_shape)
        assert kv_cache is not None, "kv_cache tensor tuple must be provided."
        wait_for_kv_layer_from_connector(layer_name)
        cache_is_prepared = self._prepare_caches_before_attention(
            layer_name,
            hidden_states,
            kv_cache,
            attn_metadata,
        )
        if actual_tokens == 0:
            output.zero_()
            notify_kv_cache_written(layer_name)
            maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
            return output

        req_metadata = _require_req_metadata(common_attn_metadata)
        o_proj_input[:actual_tokens] = self._forward_attention(
            layer_name,
            hidden_states[:actual_tokens],
            kv_cache,
            layer_metadata,
            cache_is_prepared,
        )
        cos = req_metadata.cos[layer_name]
        sin = req_metadata.sin[layer_name]

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            o_proj_input[:actual_tokens].unsqueeze(1),
            cos[:actual_tokens],
            -sin[:actual_tokens],
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # o
        self._forward_o_proj(o_proj_input, output)
        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

        return output_padded

    def _mla_prolog_single_stream(
        self,
        hidden_states,
        cos,
        sin,
        swa_kv_cache,
        slot_mapping,
        write_swa_cache=True,
    ):
        """Run the MLA prolog on the current stream."""
        share_hs_quant = _is_w8a8_dynamic(self.wq_a) and _is_w8a8_dynamic(self.wkv)
        if share_hs_quant:
            hs_int8, hs_pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
            q_a = torch_npu.npu_quant_matmul(
                hs_int8,
                self.wq_a.weight,
                self.wq_a.weight_scale,
                pertoken_scale=hs_pertoken_scale,
                bias=self.wq_a.bias,
                output_dtype=hidden_states.dtype,
            )
        else:
            q_a = self.wq_a(hidden_states)

        # q
        if _is_w8a8_dynamic(self.wq_b):
            qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                q_a, self.q_norm.weight, epsilon=self.eps
            )
            q = torch_npu.npu_quant_matmul(
                qr,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            ).unflatten(-1, (self.n_local_heads, self.head_dim))
        else:
            qr = self.q_norm(q_a)
            q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
            qr_pertoken_scale = None
        q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # win kv & tok_dis
        if write_swa_cache:
            if share_hs_quant:
                kv = torch_npu.npu_quant_matmul(
                    hs_int8,
                    self.wkv.weight,
                    self.wkv.weight_scale,
                    pertoken_scale=hs_pertoken_scale,
                    bias=self.wkv.bias,
                    output_dtype=hidden_states.dtype,
                )
            else:
                kv = self.wkv(hidden_states)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)

            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )

            # swa exec kv
            get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                swa_kv_cache,
                kv,
                slot_mapping,
            )

        return q, qr, qr_pertoken_scale

    def _mla_prolog_multistream(
        self,
        hidden_states,
        cos,
        sin,
        swa_kv_cache,
        slot_mapping,
        is_prefill=False,
        tail_overlap_fn: Callable[[], CompressorForwardOutput] | None = None,
    ):
        """3-block multi-stream: 3-stage CV parallel + serial tail

        Block partition (V: Vector, C: Cube, AIV: AI Vector):
          Part1: q_quant[V] -> q_a_down[C]  ||  kv_quant[V]
          Part2: q_norm[V] + q_b_quant[V]  ||  kv_matmul[C]
          Part3: q_b_matmul[C]             ||  kv_norm[V] + rope[V] + scatter[AIV]
          Tail:  q_rms[V] + rope[V]  ||  optional compressor metadata + compressor

        Each stream's data is self-contained; no cross-stream sync is needed between blocks.
        Only the tail wait_stream ensures scatter is complete.
        """
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()

        is_w8a8 = _is_w8a8_dynamic(self.wq_b)

        # Part1: q_quant[V] -> q_a_down[C]  ||  kv_quant[V]
        # When wq_a and wkv have the same quant_method and the same
        # communication status, their quantize() outputs are equivalent.
        # Share the result instead of calling quantize() twice on the same input.
        # - W8A8 no-comm: saves one npu_dynamic_quant (full-tensor read + absmax).
        # - W4A8 no-comm: saves one no-op pass-through (kernel launch + ref).
        # - TP comm: both return (hidden_states, None); shareable when custom_op
        #   types match (same communication path).
        share_quant = (
            type(self.cv_wq_a._quant_method) is type(self.cv_wkv._quant_method)
            and self.cv_wq_a._has_communication == self.cv_wkv._has_communication
        )
        e_kv_quant_done = None
        if share_quant:
            q_quant, q_pertoken_scale = self.cv_wq_a.quantize(hidden_states)
            kv_quant, kv_pertoken_scale = q_quant, q_pertoken_scale
        else:
            q_quant, q_pertoken_scale = self.cv_wq_a.quantize(hidden_states)
            e_q_quant_done = main_stream.record_event()
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_q_quant_done)
                kv_quant, kv_pertoken_scale = self.cv_wkv.quantize(hidden_states)
                e_kv_quant_done = torch.npu.current_stream().record_event()

        wq_a_result = self.cv_wq_a.matmul(q_quant, q_pertoken_scale)

        # Part2: q_norm[V] + q_b_quant[V]  ||  kv_matmul[C]
        e_part2_start = main_stream.record_event()
        if e_kv_quant_done is not None:
            main_stream.wait_event(e_kv_quant_done)

        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(e_part2_start)
            kv = self.cv_wkv.matmul(kv_quant, kv_pertoken_scale)
            e_kv_matmul_done = torch.npu.current_stream().record_event()

        if is_prefill:
            qr = self.q_norm(wq_a_result)
            q_b_quant, q_b_scale = self.cv_wq_b.quantize(qr)
            qr_pertoken_scale = None
        elif is_w8a8:
            qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                wq_a_result, self.q_norm.weight, epsilon=self.eps
            )
            q_b_quant, q_b_scale = qr, qr_pertoken_scale
        else:
            qr = self.q_norm(wq_a_result)
            q_b_quant, q_b_scale = qr, None
            qr_pertoken_scale = None

        # Part3: q_b_matmul[C]  ||  kv_norm[V] + rope[V] + scatter[AIV]
        e_part3_start = main_stream.record_event()
        # kv_matmul and q_b_matmul are both Cube ops. Ensure kv_matmul (launched on
        # aux_stream) completes before q_b_matmul starts so they do not contend for
        # the Cube units. kv_norm (Vector) follows kv_matmul on aux_stream and is
        # unaffected as it overlaps with q_b_matmul.
        main_stream.wait_event(e_kv_matmul_done)

        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(e_part3_start)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )
            get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)

        if is_prefill:
            q = self.cv_wq_b.matmul(q_b_quant, q_b_scale).unflatten(-1, (self.n_local_heads, self.head_dim))
        elif is_w8a8:
            q = torch_npu.npu_quant_matmul(
                q_b_quant,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=q_b_scale,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            ).unflatten(-1, (self.n_local_heads, self.head_dim))
        else:
            q = self.cv_wq_b.matmul(q_b_quant, q_b_scale).unflatten(-1, (self.n_local_heads, self.head_dim))

        # Join the Q and SWA-KV branches, then reuse the auxiliary stream for
        # independent tail work while q_rms[V] + rope[V] run on the main stream.
        main_stream.wait_stream(aux_stream)

        tail_overlap_output: CompressorOverlapOutput | None = None
        if tail_overlap_fn is not None:
            e_tail_start = main_stream.record_event()
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_tail_start)
                overlap_result = tail_overlap_fn()
                e_tail_overlap_done = torch.npu.current_stream().record_event()
            tail_overlap_output = overlap_result, e_tail_overlap_done

        q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        return q, qr, qr_pertoken_scale, tail_overlap_output

    def _maybe_update_compressed_caches_and_select_topk(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        layer_metadata: AscendDSALayerMetadata,
        qr_pertoken_scale: torch.Tensor | None,
        compress_kv_cache: torch.Tensor,
        state_cache: torch.Tensor,
        compressor_overlap_output: CompressorOverlapOutput | None = None,
        write_cache: bool = True,
    ) -> torch.Tensor | None:
        """Update compressed caches and return Indexer top-k indices."""
        compressor = self.compressor
        assert compressor is not None
        assert layer_metadata.compressor is not None

        if self.compress_ratio == 4:
            assert self.indexer is not None
            assert layer_metadata.indexer is not None

            def compute_attention_compressed_kv() -> tuple[torch.Tensor, torch.Tensor]:
                if compressor_overlap_output is not None:
                    overlap_result, compressor_done = compressor_overlap_output
                    torch.npu.current_stream().wait_event(compressor_done)
                    return overlap_result
                return compressor(
                    hidden_states=hidden_states,
                    state_cache=state_cache,
                    metadata=layer_metadata.compressor,
                )

            def scatter_attention_compressed_kv(
                compressed_kv: torch.Tensor,
                compress_slot_mapping: torch.Tensor,
            ) -> None:
                if compressed_kv.shape[0] > 0:
                    get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                        compress_kv_cache,
                        compressed_kv,
                        compress_slot_mapping,
                    )

            overlap_plan = IndexerOverlapPlan(
                compute_attention_compressed_kv=compute_attention_compressed_kv,
                scatter_attention_compressed_kv=scatter_attention_compressed_kv,
                aux_stream=dsv4_dsa_overlap_stream() if self.multistream_dsv4_dsa_overlap else None,
            )
            return self.indexer(
                hidden_states=hidden_states,
                qr=qr,
                kv_cache=kv_cache,
                metadata=layer_metadata.indexer,
                overlap_plan=overlap_plan,
                layer_name=layer_name,
                qr_pertoken_scale=qr_pertoken_scale,
                write_cache=write_cache,
            )

        if not write_cache:
            return None
        if compressor_overlap_output is not None:
            (compressed_kv, compress_slot_mapping), compressor_done = compressor_overlap_output
            torch.npu.current_stream().wait_event(compressor_done)
        else:
            compressed_kv, compress_slot_mapping = compressor(
                hidden_states=hidden_states,
                state_cache=state_cache,
                metadata=layer_metadata.compressor,
            )
        if compressed_kv.shape[0] > 0:
            get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                compress_kv_cache,
                compressed_kv,
                compress_slot_mapping,
            )
        return None

    def _forward_attention(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        layer_metadata: AscendDSALayerMetadata,
        cache_is_prepared: bool = False,
    ) -> torch.Tensor:
        # DSA PCP sets cache_is_prepared after global cache updates and forces
        # single-stream attention because there is no local KV update to overlap.
        if cache_is_prepared and self.multistream_dsv4_dsa_overlap:
            raise RuntimeError("Prepared DSA caches require single-stream attention.")

        (
            compress_kv_cache,
            swa_kv_cache,
            state_cache,
            _,
            _,
            _,
        ) = DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)

        common_attn_metadata = layer_metadata.attention
        if common_attn_metadata is None:
            common_attn_metadata = layer_metadata.swa
        swa_metadata = layer_metadata.swa

        common_metadata = _require_req_metadata(common_attn_metadata)
        swa_req_metadata = _require_req_metadata(swa_metadata)
        has_prefill = common_attn_metadata.num_prefills > 0
        num_tokens = hidden_states.shape[0]
        cos = common_metadata.cos[layer_name][:num_tokens]
        sin = common_metadata.sin[layer_name][:num_tokens]
        actual_seq_lengths_query = common_metadata.query_start_loc
        actual_seq_lengths_key = common_metadata.seq_lens
        ori_win_left = self.window_size - 1 if swa_req_metadata.ori_win_left is None else swa_req_metadata.ori_win_left
        ori_win_right = 0 if swa_req_metadata.ori_win_right is None else swa_req_metadata.ori_win_right

        compressor_tail_fn = None
        if self.multistream_dsv4_dsa_overlap and self.compress_ratio > 1:
            compressor = self.compressor
            tail_compressor_metadata = layer_metadata.compressor
            assert compressor is not None
            assert tail_compressor_metadata is not None

            def compressor_tail_fn() -> CompressorForwardOutput:
                return compressor(
                    hidden_states=hidden_states,
                    state_cache=state_cache,
                    metadata=tail_compressor_metadata,
                )

        if self.multistream_dsv4_dsa_overlap:
            q, qr, qr_pertoken_scale, compressor_overlap_output = self._mla_prolog_multistream(
                hidden_states,
                cos,
                sin,
                swa_kv_cache,
                swa_req_metadata.slot_mapping,
                is_prefill=has_prefill,
                tail_overlap_fn=compressor_tail_fn,
            )
        else:
            compressor_overlap_output = None
            q, qr, qr_pertoken_scale = self._mla_prolog_single_stream(
                hidden_states,
                cos,
                sin,
                swa_kv_cache,
                swa_req_metadata.slot_mapping,
                write_swa_cache=not cache_is_prepared,
            )

        compress_topk_idxs = None
        compressor_metadata = None
        if self.compress_ratio > 1:
            compressor_metadata = layer_metadata.compressor
            assert compressor_metadata is not None
            compress_topk_idxs = self._maybe_update_compressed_caches_and_select_topk(
                layer_name=layer_name,
                hidden_states=hidden_states,
                qr=qr,
                kv_cache=kv_cache,
                layer_metadata=layer_metadata,
                qr_pertoken_scale=qr_pertoken_scale,
                compress_kv_cache=compress_kv_cache,
                state_cache=state_cache,
                compressor_overlap_output=compressor_overlap_output,
                write_cache=not cache_is_prepared,
            )

        notify_kv_cache_written(layer_name)
        wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(common_metadata.sas_metadata))
        record_attention_compute_start()
        kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
        attn_op = kv_plan.get_dsa_sparse_attn_op()
        attn_kwargs = kv_plan.get_dsa_sparse_attn_base_kwargs()
        if has_prefill:
            kv_plan.add_dsa_sparse_attn_extra_kwargs(attn_kwargs, cu_seqlens_ori_kv=actual_seq_lengths_query)
        if self.compress_ratio > 1:
            kv_plan.add_dsa_sparse_attn_extra_kwargs(attn_kwargs, cu_seqlens_cmp_kv=common_metadata.cu_cmp_seqlen_list)

        attn_kwargs.update(
            ori_kv=swa_kv_cache,
            ori_block_table=swa_req_metadata.block_table,
            cu_seqlens_q=actual_seq_lengths_query,
            seqused_kv=actual_seq_lengths_key,
            sinks=self.attn_sink,
            metadata=common_metadata.sas_metadata,
            softmax_scale=self.softmax_scale,
            cmp_ratio=_dsa_swa_only_cmp_ratio(self.compress_ratio, self.vllm_config),
            ori_mask_mode=4,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q="TND",
            layout_kv=_dsa_layout_kv(self.vllm_config),
        )

        # Vision prefill uses explicit original-KV indices so tokens inside an
        # image span can see the complete span bidirectionally. Compressed KV
        # selection remains active and is supplied independently below.
        if swa_req_metadata.vision_swa_indices is not None:
            attn_kwargs["ori_sparse_indices"] = swa_req_metadata.vision_swa_indices

        if self.compress_ratio <= 1:
            if swa_req_metadata.dspark_swa_indices is not None:
                attn_kwargs["ori_sparse_indices"] = swa_req_metadata.dspark_swa_indices
        else:
            assert compressor_metadata is not None
            attn_kwargs.update(
                cmp_kv=compress_kv_cache,
                cmp_block_table=common_metadata.block_table,
                cmp_mask_mode=3,
            )
            if self.compress_ratio == 4:
                assert compress_topk_idxs is not None
                attn_kwargs["cmp_sparse_indices"] = compress_topk_idxs

        return attn_op(q, **attn_kwargs)[0]
