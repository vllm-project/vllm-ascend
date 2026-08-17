import logging
import math
import time
from dataclasses import dataclass
from typing import ClassVar, TypeVar

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.triton_utils import HAS_TRITON, triton
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend import envs as ascend_envs
from vllm_ascend.attention.abstract import DSAAttentionImpl
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.compressor_sp import (
    CompressorSPPlan,
    all_ranks_have_compressor_sp_rows,
    build_compressor_sp_plan,
    build_padded_destination_for_scatter,
    collect_state_row_indices,
    run_compressor_op,
    sync_boundary_state_blocks,
)
from vllm_ascend.attention.dsa_v1 import (
    build_dspark_swa_indices,
    get_dspark_sparse_sas_window,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    maybe_save_kv_layer_to_connector,
    notify_kv_cache_written,
    split_decodes_and_prefills,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.memcache_comm_fence import record_attention_compute_start
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.rope_dsv4 import RopeDataProxy, get_cos_and_sin_dsa, get_full_cos_and_sin_dsa
from vllm_ascend.ops.triton.dsa_cp import build_local_metadata_triton
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (
    AscendDeviceType,
    enable_dsa_cp_with_o_proj_tp,
    get_ascend_device_type,
    olora_tp_enable,
)

logger = logging.getLogger(__name__)


def hadamard_transform_ref(
    x: torch.Tensor,
    hadamard: torch.Tensor,
    scale: float = 1.0,  # type: ignore[assignment]
):
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, hadamard)
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x, hadamard=hadamard, scale=hidden_size**-0.5)


@dataclass
class DSACPMetadata:
    """Context-parallel metadata for sequence-sharded DSA execution."""

    local_query_start_loc: torch.Tensor
    local_seq_lens: torch.Tensor
    local_start: int
    local_end: int
    tokens_per_rank: int
    num_tokens_pad: int
    local_sin: torch.Tensor = None
    local_cos: torch.Tensor = None


@dataclass
class CompressorSPMetadata:
    enabled: bool
    reason: str
    ratio: int = 0
    path: str = ""
    coff: int = 0
    cache_mode: int = 1
    is_chunked_prefill: bool = False
    state_block_table_rows: int = 0
    start_pos_zero: bool | None = None
    seq_len_aligned: bool | None = None
    requires_history_state: bool = False
    requires_tail_state_update: bool = False
    requires_boundary_state_sync: bool = False
    global_compressed_row_count: int = 0
    boundary_req_indices: torch.Tensor = None
    boundary_positions: torch.Tensor = None
    boundary_owner_mask: torch.Tensor = None
    supports_boundary_state_replay: bool = False
    boundary_replay_token_ranges: tuple[tuple[int, int], ...] = ()
    boundary_replay_token_indices: torch.Tensor = None
    boundary_replay_token_slice: tuple[int, int] | None = None
    boundary_replay_req_indices: torch.Tensor = None
    boundary_replay_req_slice: tuple[int, int] | None = None
    boundary_replay_cu_seqlens: torch.Tensor = None
    boundary_replay_start_pos: torch.Tensor = None
    boundary_replay_compressed_row_indices: torch.Tensor = None
    boundary_replay_compressed_row_slice: tuple[int, int] | None = None
    boundary_replay_rope_row_indices: torch.Tensor = None
    boundary_replay_rope_row_slice: tuple[int, int] | None = None
    history_start_positions: torch.Tensor = None
    request_start_positions: torch.Tensor = None
    token_indices: torch.Tensor = None
    token_slice: tuple[int, int] | None = None
    req_indices: torch.Tensor = None
    req_slice: tuple[int, int] | None = None
    cu_seqlens: torch.Tensor = None
    start_pos: torch.Tensor = None
    compressed_row_indices: torch.Tensor = None
    compressed_row_slice: tuple[int, int] | None = None
    rope_row_indices: torch.Tensor = None
    rope_row_slice: tuple[int, int] | None = None
    valid_row_indices: torch.Tensor = None
    valid_row_slice: tuple[int, int] | None = None
    output_keep_indices: torch.Tensor = None
    output_keep_slice: tuple[int, int] | None = None
    slot_mapping_indices: torch.Tensor = None
    slot_mapping_slice: tuple[int, int] | None = None
    local_keep_to_full_row_indices: torch.Tensor = None
    local_keep_to_slot_row_indices: torch.Tensor = None
    tail_token_ranges: tuple[tuple[int, int], ...] = ()
    padding_row_indices: torch.Tensor = None
    padding_row_slice: tuple[int, int] | None = None
    gather_compact_indices: torch.Tensor = None
    gather_compact_slice: tuple[int, int] | None = None
    sp_row_counts_per_rank: tuple[int, ...] = ()
    tp_rank: int = 0
    tp_size: int = 1


@dataclass
class AscendDSAReqMetadata:
    """Unified per-request metadata — combines fields formerly split into
    prefill and decode sub-structures.

    All methods (builder, forward) operate on this single metadata,
    without distinguishing prefill vs decode request types.
    """

    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor | None
    block_size: int
    query_start_loc: torch.Tensor
    cp_metadata: DSACPMetadata
    num_compressed_tokens: int | None = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    full_compress_sin: torch.Tensor = None
    full_compress_cos: torch.Tensor = None
    start_pos: torch.Tensor = None
    num_reqs_actual: int | None = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None
    cu_cmp_seqlen_list: torch.Tensor = None
    attn_mask: torch.Tensor | None = None
    ori_win_left: int | None = None
    ori_win_right: int = 0
    dspark_swa_indices: torch.Tensor | None = None
    compressor_sp: CompressorSPMetadata | None = None


@dataclass
class AscendDSAMetadata:
    """Metadata for MLACommon.
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # The dimension of the attention heads
    head_dim: int | None = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    req_metadata: AscendDSAReqMetadata | None = None
    reshape_cache_event: torch.npu.Event = None

    # metadata for dsv4 indexer

    hadamard: torch.Tensor | None = None

    start_pos: torch.Tensor | None = None


M = TypeVar("M", bound=AscendDSAMetadata)


class AscendDSACPMetadataBuilder(AttentionMetadataBuilder[AscendDSAMetadata]):
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    hadamard = None
    start_pos_prefill: torch.Tensor
    req_sas_metadata: torch.Tensor
    req_qli_metadata: torch.Tensor
    block_size: int = 128
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
        scheduler_config = vllm_config.scheduler_config

        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim

        self.num_decodes = 0
        self.num_prefills = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.num_actual_tokens: int | None = None
        self.block_table: torch.Tensor = None
        self.slot_mapping: torch.Tensor = None
        self.seq_lens: torch.Tensor = None
        self.seq_lens_cpu: torch.Tensor = None

        self.compressor_ratio = getattr(kv_cache_spec, "compress_ratio", 0)
        hf_config = self.model_config.hf_config

        if AscendDSACPMetadataBuilder.hadamard is None:
            if hf_config.model_type == "deepseek_v4":
                indexer_head_dim = hf_config.index_head_dim
                try:
                    from scipy.linalg import hadamard  # type: ignore[import-untyped]
                except ImportError as e:
                    raise ImportError(
                        "DeepSeek-V4 indexer attention requires SciPy for Hadamard transform. Please install scipy."
                    ) from e
                log_dim = math.ceil(math.log2(indexer_head_dim))
                dim_padded = 2**log_dim
                if self.vllm_config.model_config.enable_sleep_mode:
                    # Sleep mode allocates KV inside CaMemAllocator; tag Hadamard so
                    # sleep/wake does not treat it as KV cache.
                    from vllm_ascend.device_allocator.camem import CaMemAllocator

                    allocator = CaMemAllocator.get_instance()
                    with allocator.use_allocation_tag(CaMemAllocator.sleep_persistent_tag):
                        AscendDSACPMetadataBuilder.hadamard = torch.tensor(
                            hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device
                        ).to(torch.bfloat16)
                else:
                    AscendDSACPMetadataBuilder.hadamard = torch.tensor(
                        hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device
                    ).to(torch.bfloat16)
        self.start_pos_prefill = torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)
        self.req_sas_metadata = torch.zeros(1024, dtype=torch.int32, device=self.device)
        self.req_qli_metadata = torch.zeros(1024, dtype=torch.int32, device=self.device)
        self.cu_seqlens_ori_kv = torch.tensor([], device=self.device)
        self.cu_seqlens_cmp_kv = torch.tensor([], device=self.device)
        self.seqused_q = torch.tensor([], device=self.device)
        self._zero_i32 = torch.tensor([0], device=self.device, dtype=torch.int32)
        self.local_query_start_loc = torch.zeros(
            scheduler_config.max_num_seqs + 1, dtype=torch.int32, device=self.device
        )
        self.local_seq_lens = torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        self.spec_slot_mapping = None
        if get_ascend_device_type() in {AscendDeviceType.A5}:
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens,)  # type: ignore
        else:
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens, 2)  # type: ignore
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.spec_slot_mapping = [
                torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)
                for _ in range(spec_token_num)
            ]
            self.spec_local_query_start_loc = [
                torch.zeros(scheduler_config.max_num_seqs + 1, dtype=torch.int32, device=self.device)
                for _ in range(spec_token_num)
            ]
            self.spec_local_seq_lens = [
                torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)
                for _ in range(spec_token_num)
            ]
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold
        # Note(qcs): we use two dimension slot_mapping for kvcache with shape
        # [block_nums, block_size, head_num, head_dim]
        self.slot_mapping = torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendDSACPMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        num_reqs_actual = kwargs.get("num_reqs_actual")
        self.block_size = kwargs.get("block_size", 128)

        common_ratio_to_sas_metadata = kwargs.get("common_ratio_to_sas_metadata")
        assert common_ratio_to_sas_metadata is not None
        self.common_ratio_to_sas_metadata = common_ratio_to_sas_metadata
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens
        attn_state = kwargs.get("attn_state", common_attn_metadata.attn_state)

        num_input_tokens = common_attn_metadata.num_input_tokens
        if self.common_ratio_to_sas_metadata.get("input_positions", None) is None:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.decode_threshold,
                    treat_short_extends_as_decodes=False,
                )
            )
            self.common_ratio_to_sas_metadata["num_decodes"] = self.num_decodes
            self.common_ratio_to_sas_metadata["num_prefills"] = self.num_prefills
            self.common_ratio_to_sas_metadata["num_decode_tokens"] = self.num_decode_tokens
            self.common_ratio_to_sas_metadata["num_prefill_tokens"] = self.num_prefill_tokens
            input_positions = common_attn_metadata.positions[:num_input_tokens].long()
            self.common_ratio_to_sas_metadata["input_positions"] = input_positions
            has_prefill = self.num_prefills > 0
            cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=not has_prefill)
            self.common_ratio_to_sas_metadata["cos"] = cos
            self.common_ratio_to_sas_metadata["sin"] = sin
            self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
            self.common_ratio_to_sas_metadata["seq_lens"] = self.seq_lens
            # Prefer _seq_lens_cpu (always available, updated during draft
            # iterations) over seq_lens_cpu (None in async spec decode mode).
            if common_attn_metadata._seq_lens_cpu is not None:
                _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
            elif common_attn_metadata.seq_lens_cpu is not None:
                _seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            else:
                _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
            self.seq_lens_cpu = _seq_lens_cpu
            self.common_ratio_to_sas_metadata["seq_lens_cpu"] = self.seq_lens_cpu
        else:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                self.common_ratio_to_sas_metadata["num_decodes"],
                self.common_ratio_to_sas_metadata["num_prefills"],
                self.common_ratio_to_sas_metadata["num_decode_tokens"],
                self.common_ratio_to_sas_metadata["num_prefill_tokens"],
            )
            input_positions = self.common_ratio_to_sas_metadata["input_positions"]
            cos, sin = self.common_ratio_to_sas_metadata["cos"], self.common_ratio_to_sas_metadata["sin"]
            self.seq_lens = self.common_ratio_to_sas_metadata["seq_lens"]
            self.seq_lens_cpu = self.common_ratio_to_sas_metadata["seq_lens_cpu"]

        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        self.slot_mapping[:num_input_tokens] = DeviceOperator.format_dsa_slot_mapping(slot_mapping, self.block_size)

        self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]

        req_metadata = self.build_req_metadata(
            common_attn_metadata,
            input_positions,
            num_input_tokens,
            num_reqs_actual,
            attn_state,
            cos=cos,
            sin=sin,
        )

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            head_dim=self.model_config.get_head_size(),
            attn_mask=None,
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_state=attn_state,
            req_metadata=req_metadata,
            query_start_loc=query_start_loc,
            block_tables=None,
            seq_lens=self.seq_lens,
            cos=cos,
            sin=sin,
            hadamard=AscendDSACPMetadataBuilder.hadamard,
        )

    def build_for_drafting(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        assert self.compressor_ratio <= 1, "vLLM-Ascend only support SWA-layer for Deepseek-V4 now."
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            treat_short_extends_as_decodes=False,
        )

        self.num_decodes = num_decodes
        self.num_prefills = num_prefills
        self.num_decode_tokens = num_decode_tokens
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens
        self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        if common_attn_metadata._seq_lens_cpu is not None:
            self.seq_lens_cpu = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            self.seq_lens_cpu = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            self.seq_lens_cpu = self.seq_lens.cpu()
        self.block_size = kwargs.get("block_size", 128)

        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        # Draft steps update positions independently. Reusing the global RoPE
        # cache can let later draft steps overwrite step-0 metadata.
        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=False)

        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]

        assert self.spec_slot_mapping is not None
        self.spec_slot_mapping[draft_index - 1][:num_input_tokens] = DeviceOperator.format_dsa_slot_mapping(
            slot_mapping, self.block_size
        )

        self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        req_metadata = self.build_req_metadata_for_drafting(
            draft_index=draft_index,
            common_attn_metadata=common_attn_metadata,
            input_positions=input_positions,
            num_input_tokens=num_input_tokens,
            cos=cos,
            sin=sin,
        )

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            head_dim=self.model_config.get_head_size(),
            attn_mask=None,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            attn_state=common_attn_metadata.attn_state,
            req_metadata=req_metadata,
            query_start_loc=common_attn_metadata.query_start_loc,
            block_tables=None,
            seq_lens=self.seq_lens,
            cos=cos,
            sin=sin,
            hadamard=None,
        )

    def build_req_metadata_for_drafting(
        self,
        draft_index: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        input_positions: torch.Tensor,
        num_input_tokens: int,
        cos: RopeDataProxy,
        sin: RopeDataProxy,
    ) -> AscendDSAReqMetadata:
        """Build DSA-CP metadata for one draft step."""
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        is_noncausal = not common_attn_metadata.causal
        has_prefill = self.num_prefills > 0

        (
            local_start,
            local_end_with_pad,
            tokens_per_rank,
            num_tokens_pad,
            local_query_start_loc,
            local_seq_lens,
        ) = self._build_local_token_metadata(
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            query_start_loc=query_start_loc,
            seq_lens=self.seq_lens[:num_reqs],
            local_query_start_loc=self.spec_local_query_start_loc[draft_index - 1],
            local_seq_lens=self.spec_local_seq_lens[draft_index - 1],
            is_noncausal=is_noncausal,
        )
        local_query_start_loc = local_query_start_loc.clone()
        local_seq_lens = local_seq_lens.clone()
        local_cos = cos.pad_to(num_tokens_pad)[local_start:local_end_with_pad]
        local_sin = sin.pad_to(num_tokens_pad)[local_start:local_end_with_pad]

        _, _, _, _, local_query_start_loc_cpu, local_seq_lens_cpu = self._build_local_token_metadata(
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            query_start_loc=query_start_loc_cpu,
            seq_lens=self.seq_lens_cpu[:num_reqs],
            is_noncausal=is_noncausal,
        )
        local_seq_lens_q_cpu = local_query_start_loc_cpu[1 : num_reqs + 1] - local_query_start_loc_cpu[:num_reqs]
        max_local_query_len = max(1, int(local_seq_lens_q_cpu.max().item()))
        max_local_seq_lens = max(1, int(local_seq_lens_cpu.max().item()))

        start_pos = self.seq_lens[:num_reqs] - seq_lens_q

        dspark_swa_indices = None
        ori_win_left, ori_win_right = self.model_config.hf_config.sliding_window - 1, 0
        if is_noncausal:
            assert self.speculative_config is not None
            global_dspark_indices, _ = build_dspark_swa_indices(
                self.block_table[:num_reqs],
                self.speculative_config.num_speculative_tokens,
                self.model_config.hf_config.sliding_window,
                self.block_size,
                query_start_loc[: num_reqs + 1],
                self.seq_lens[:num_reqs],
                self.num_actual_tokens,
            )
            pad_rows = num_tokens_pad - global_dspark_indices.shape[0]
            if pad_rows < 0:
                raise ValueError(
                    "DSpark CP metadata has fewer padded query rows than actual rows: "
                    f"num_tokens_pad={num_tokens_pad}, actual={global_dspark_indices.shape[0]}"
                )
            if pad_rows:
                global_dspark_indices = F.pad(global_dspark_indices, (0, 0, 0, 0, 0, pad_rows), value=-1)
            dspark_swa_indices = global_dspark_indices[local_start:local_end_with_pad].contiguous()
            ori_win_left, ori_win_right = get_dspark_sparse_sas_window(self.vllm_config)

        assert self.spec_slot_mapping is not None
        slot_mapping = self.spec_slot_mapping[draft_index - 1][: self.num_actual_tokens]

        num_heads = self.model_config.hf_config.num_attention_heads
        metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
        metadata_kwargs.setdefault("device", str(self.seqused_q.device))
        cu_seqlens_ori_kv = (
            local_query_start_loc
            if has_prefill
            else DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
                None,
                "draft_cu_seqlens_ori_kv",
                local_seq_lens,
                num_reqs,
                self._zero_i32,
                self.cu_seqlens_ori_kv,
            )
        )
        cu_seqlens_cmp_kv = (
            None if has_prefill else DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)
        )
        sas_metadata = metadata_op(
            **metadata_kwargs,
            num_heads_q=num_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=local_query_start_loc,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=self.seqused_q,
            seqused_kv=local_seq_lens,
            max_seqlen_q=max_local_query_len,
            max_seqlen_kv=max_local_seq_lens,
            batch_size=num_reqs,
            cmp_ratio=1,
            ori_mask_mode=4,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
        )

        cp_metadata = DSACPMetadata(
            local_query_start_loc=local_query_start_loc,
            local_seq_lens=local_seq_lens,
            local_start=local_start,
            local_end=local_end_with_pad,
            tokens_per_rank=tokens_per_rank,
            num_tokens_pad=num_tokens_pad,
            local_sin=local_sin,
            local_cos=local_cos,
        )

        coff = 2 if self.compressor_ratio == 4 else 1
        compressor_sp_metadata = self._build_compressor_sp_metadata(
            common_attn_metadata=common_attn_metadata,
            input_positions_cpu=input_positions,
            num_reqs=num_reqs,
            has_prefill=has_prefill,
            local_start=local_start,
            local_end=local_end_with_pad,
            path="metadata",
            coff=coff,
            cache_mode=1,
            state_block_table=self.block_table[:num_reqs, ...],
            attn_state=common_attn_metadata.attn_state,
        )
        return AscendDSAReqMetadata(
            input_positions=input_positions,
            block_table=self.block_table[:num_reqs, ...],
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            seq_lens=self.seq_lens[:num_reqs],
            query_start_loc=query_start_loc,
            cp_metadata=cp_metadata,
            sin=sin,
            cos=cos,
            start_pos=start_pos,
            sas_metadata=sas_metadata,
            qli_metadata=None,
            cu_cmp_seqlen_list=None,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            dspark_swa_indices=dspark_swa_indices,
            compressor_sp=compressor_sp_metadata,
        )

    def _num_compressor_metadata_rows(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> int:
        assert self.num_actual_tokens is not None
        num_tokens = self.num_actual_tokens
        return min(num_tokens, num_tokens // self.compressor_ratio + common_attn_metadata.num_reqs)

    def _ensure_device_local_metadata(
        self,
        num_reqs: int,
        num_input_tokens: int,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        """Return device local metadata, cached across kv-cache groups.

        The computation (clamp + cumsum + offset + mask) is identical for
        all attention groups, so we compute once and cache the results.
        """
        cache = self.common_ratio_to_sas_metadata.get("_device_local")
        if cache is None:
            # Calc and cache device tensor results
            (
                local_start,
                local_end_with_pad,
                tokens_per_rank,
                num_tokens_pad,
                local_query_start_loc,
                local_seq_lens,
            ) = self._build_local_token_metadata(
                num_reqs=num_reqs,
                num_input_tokens=num_input_tokens,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                local_query_start_loc=self.local_query_start_loc,
                local_seq_lens=self.local_seq_lens,
                start_pos_out=self.start_pos_prefill,
            )
            self.common_ratio_to_sas_metadata["_device_local"] = {
                "local_start": local_start,
                "local_end": local_end_with_pad,
                "tokens_per_rank": tokens_per_rank,
                "num_tokens_pad": num_tokens_pad,
                "qsl": self.local_query_start_loc[: num_reqs + 1].clone(),
                "sl": self.local_seq_lens[:num_reqs].clone(),
                "sp": self.start_pos_prefill[:num_reqs].clone(),
            }
        else:
            # copy from cache
            assert cache is not None
            local_start = cache["local_start"]
            local_end_with_pad = cache["local_end"]
            tokens_per_rank = cache["tokens_per_rank"]
            num_tokens_pad = cache["num_tokens_pad"]
            self.local_query_start_loc[: num_reqs + 1].copy_(cache["qsl"])
            self.local_seq_lens[:num_reqs].copy_(cache["sl"])
            self.start_pos_prefill[:num_reqs].copy_(cache["sp"])
            local_query_start_loc = self.local_query_start_loc[: num_reqs + 1]
            local_seq_lens = self.local_seq_lens[:num_reqs]

        return (
            local_start,
            local_end_with_pad,
            tokens_per_rank,
            num_tokens_pad,
            local_query_start_loc,
            local_seq_lens,
        )

    def build_req_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        input_positions: torch.Tensor | None,
        num_input_tokens: int,
        num_reqs_actual: int | None,
        attn_state: AscendAttentionState,
        cos: RopeDataProxy,
        sin: RopeDataProxy,
    ) -> AscendDSAReqMetadata:
        """Build a single unified metadata for all requests (prefill + decode)."""
        num_reqs = common_attn_metadata.num_reqs
        has_prefill = self.num_prefills > 0
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

        # ── GPU local metadata (cached across kv-cache groups) ──
        (
            local_start,
            local_end_with_pad,
            tokens_per_rank,
            num_tokens_pad,
            local_query_start_loc,
            local_seq_lens,
        ) = self._ensure_device_local_metadata(
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            query_start_loc=query_start_loc,
            seq_lens=self.seq_lens[:num_reqs],
        )

        # RoPE local slices (cached across kv-cache groups: same cos/sin,
        # num_tokens_pad, local_start, local_end_with_pad for all groups)
        if input_positions is not None:
            rope_local = self.common_ratio_to_sas_metadata.get("_rope_local")
            if rope_local is None:
                local_cos = cos.pad_to(num_tokens_pad)[local_start:local_end_with_pad]
                local_sin = sin.pad_to(num_tokens_pad)[local_start:local_end_with_pad]
                self.common_ratio_to_sas_metadata["_rope_local"] = (local_cos, local_sin)
            else:
                assert rope_local is not None
                local_cos, local_sin = rope_local
        else:
            local_cos = None
            local_sin = None

        # ── CPU local metadata (cached) ──
        cpu_cache = self.common_ratio_to_sas_metadata.get("_cpu_local")
        if cpu_cache is None:
            _, _, _, _, local_query_start_loc_cpu, local_seq_lens_cpu = self._build_local_token_metadata(
                num_reqs=num_reqs,
                num_input_tokens=num_input_tokens,
                query_start_loc=query_start_loc_cpu,
                seq_lens=self.seq_lens_cpu[:num_reqs],
            )
            self.common_ratio_to_sas_metadata["_cpu_local"] = {
                "qsl_cpu": local_query_start_loc_cpu.clone(),
                "sl_cpu": local_seq_lens_cpu.clone(),
            }
        else:
            assert cpu_cache is not None
            local_query_start_loc_cpu = cpu_cache["qsl_cpu"]
            local_seq_lens_cpu = cpu_cache["sl_cpu"]
        local_seq_lens_q = local_query_start_loc[1 : num_reqs + 1] - local_query_start_loc[:num_reqs]
        local_seq_lens_q_cpu = local_query_start_loc_cpu[1 : num_reqs + 1] - local_query_start_loc_cpu[:num_reqs]
        max_local_query_len = max(1, int(local_seq_lens_q_cpu.max().item()))
        max_local_seq_lens = max(1, int(local_seq_lens_cpu.max().item()))

        if num_reqs_actual is None:
            num_reqs_actual = num_reqs
        else:
            num_reqs_actual = min(num_reqs_actual, num_reqs)
            if num_reqs_actual < num_reqs:
                self.start_pos_prefill[num_reqs_actual:].fill_(0)
                self.block_table[num_reqs_actual:num_reqs, ...].fill_(0)

        # --- Compressed positions ---
        full_compress_cos, full_compress_sin = None, None
        cu_cmp_seqlens = self._get_cmp_seqlens_for_metadata(has_prefill)

        if self.compressor_ratio > 1:
            layer_name = f"c{self.compressor_ratio}"
            # Keep only graph inputs here. The compressor metadata op itself is
            # launched in forward at the real compressor consumer.
            num_compressed_tokens = self._num_compressor_metadata_rows(common_attn_metadata)
            full_compress_cos, full_compress_sin = get_full_cos_and_sin_dsa(layer_name)
            slot_mapping = None
        else:
            num_compressed_tokens = None
            slot_mapping = self.slot_mapping[: self.num_actual_tokens]

        planner_positions = getattr(common_attn_metadata, "positions_cpu", None)
        if planner_positions is None:
            assert input_positions is not None
            planner_positions = input_positions.cpu()
        planner_positions = planner_positions[:num_input_tokens].long()
        coff = 2 if self.compressor_ratio == 4 else 1
        compressor_sp_metadata = self._build_compressor_sp_metadata(
            common_attn_metadata=common_attn_metadata,
            input_positions_cpu=planner_positions,
            num_reqs=num_reqs,
            has_prefill=has_prefill,
            local_start=local_start,
            local_end=local_end_with_pad,
            path="metadata",
            coff=coff,
            cache_mode=1,
            state_block_table=self.block_table[:num_reqs, ...],
            attn_state=attn_state,
        )

        # --- SAS metadata (all requests combined) ---
        num_heads = self.model_config.hf_config.num_attention_heads
        index_topk = self.model_config.hf_config.index_topk

        sas_metadata = self._build_sas_metadata(
            num_heads=num_heads,
            query_start_loc=local_query_start_loc,
            seq_lens=local_seq_lens,
            seq_lens_q=local_seq_lens_q,
            max_query_len=max_local_query_len,
            max_seq_lens=max_local_seq_lens,
            index_topk=index_topk,
            num_reqs=num_reqs,
            has_prefill=has_prefill,
            cu_cmp_seqlen_list=cu_cmp_seqlens,
        )

        # --- QLI metadata (all requests combined) ---
        qli_metadata = self._build_qli_metadata(
            query_start_loc=local_query_start_loc,
            seq_lens=local_seq_lens,
            seq_lens_q=local_seq_lens_q,
            num_reqs=num_reqs,
        )

        cp_metadata = DSACPMetadata(
            local_query_start_loc=local_query_start_loc,
            local_seq_lens=local_seq_lens,
            local_start=local_start,
            local_end=local_end_with_pad,
            tokens_per_rank=tokens_per_rank,
            num_tokens_pad=num_tokens_pad,
            local_sin=local_sin,
            local_cos=local_cos,
        )

        return AscendDSAReqMetadata(
            input_positions=input_positions,
            block_table=self.block_table[:num_reqs, ...],
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            seq_lens=self.seq_lens[:num_reqs],
            query_start_loc=query_start_loc,
            cp_metadata=cp_metadata,
            sin=sin,
            cos=cos,
            full_compress_sin=full_compress_sin,
            full_compress_cos=full_compress_cos,
            start_pos=self.start_pos_prefill[:num_reqs],
            num_compressed_tokens=num_compressed_tokens,
            num_reqs_actual=num_reqs_actual,
            sas_metadata=sas_metadata,
            qli_metadata=qli_metadata,
            cu_cmp_seqlen_list=cu_cmp_seqlens,
            compressor_sp=compressor_sp_metadata,
        )

    def _build_compressor_sp_metadata(
        self,
        *,
        common_attn_metadata: AscendCommonAttentionMetadata,
        input_positions_cpu: torch.Tensor,
        num_reqs: int,
        has_prefill: bool,
        local_start: int,
        local_end: int,
        path: str,
        coff: int,
        cache_mode: int,
        state_block_table: torch.Tensor | None = None,
        attn_state: AscendAttentionState | None = None,
    ) -> CompressorSPMetadata:
        if self.compressor_ratio <= 1:
            return CompressorSPMetadata(False, "no_compressor")
        if attn_state is None:
            attn_state = getattr(common_attn_metadata, "attn_state", None)
        is_chunked_prefill = attn_state == AscendAttentionState.ChunkedPrefill
        if attn_state == AscendAttentionState.SpecDecoding:
            return CompressorSPMetadata(
                False,
                "unsupported_spec_or_mtp",
                ratio=self.compressor_ratio,
                path=path,
                coff=coff,
                cache_mode=cache_mode,
            )
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        if query_start_loc_cpu is None or seq_lens_cpu is None:
            return CompressorSPMetadata(False, "missing_cpu_metadata")

        tp_size = get_tp_group().world_size
        tp_rank = get_tp_group().rank_in_group
        query_start_loc_key = tuple(query_start_loc_cpu[: num_reqs + 1].tolist())
        seq_lens_key = tuple(seq_lens_cpu[:num_reqs].tolist())
        input_positions_key = tuple(input_positions_cpu.tolist())
        cache_key = (
            "compressor_sp",
            self.compressor_ratio,
            path,
            int(ascend_envs.VLLM_ASCEND_ENABLE_COMPRESSOR_SP),
            int(ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_C4_NONALIGNED),
            int(ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_C128_NONALIGNED),
            int(ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_DUAL_RUN),
            int(has_prefill),
            tp_size,
            num_reqs,
            local_start,
            local_end,
            coff,
            cache_mode,
            int(is_chunked_prefill),
            state_block_table.shape[0] if state_block_table is not None else 0,
            query_start_loc_key,
            seq_lens_key,
            input_positions_key,
        )
        metadata = self.common_ratio_to_sas_metadata.get(cache_key)
        if metadata is not None:
            return metadata

        plan = build_compressor_sp_plan(
            enabled=ascend_envs.VLLM_ASCEND_ENABLE_COMPRESSOR_SP,
            has_prefill=has_prefill,
            need_gather_q_kv=True,
            tp_size=tp_size,
            compress_ratio=self.compressor_ratio,
            path=path,
            coff=coff,
            cache_mode=cache_mode,
            is_chunked_prefill=is_chunked_prefill,
            state_block_table_rows=state_block_table.shape[0] if state_block_table is not None else 0,
            allow_c4_non_aligned=(
                ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_C4_NONALIGNED
                or ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_DUAL_RUN
            ),
            allow_c128_non_aligned=(
                ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_C128_NONALIGNED
                or ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_DUAL_RUN
            ),
            input_positions=input_positions_cpu.tolist(),
            query_start_loc=query_start_loc_cpu[: num_reqs + 1].tolist(),
            seq_lens=seq_lens_cpu[:num_reqs].tolist(),
            local_start=local_start,
            local_end=local_end,
            tp_rank=tp_rank,
        )
        metadata = self._to_compressor_sp_metadata(plan)
        metadata.path = path
        metadata.coff = coff
        metadata.cache_mode = cache_mode
        self.common_ratio_to_sas_metadata[cache_key] = metadata
        return metadata

    def _to_compressor_sp_metadata(self, plan: CompressorSPPlan) -> CompressorSPMetadata:
        if not plan.enabled:
            return CompressorSPMetadata(
                False,
                plan.reason,
                ratio=plan.ratio,
                path=plan.path,
                coff=plan.coff,
                cache_mode=plan.cache_mode,
                is_chunked_prefill=plan.is_chunked_prefill,
                state_block_table_rows=plan.state_block_table_rows,
                start_pos_zero=plan.start_pos_zero,
                seq_len_aligned=plan.seq_len_aligned,
                requires_history_state=plan.requires_history_state,
                requires_tail_state_update=plan.requires_tail_state_update,
                requires_boundary_state_sync=plan.requires_boundary_state_sync,
                global_compressed_row_count=plan.global_compressed_row_count,
                boundary_req_indices=torch.tensor(plan.boundary_req_indices, dtype=torch.long, device=self.device)
                if plan.boundary_req_indices
                else None,
                boundary_positions=torch.tensor(plan.boundary_positions, dtype=torch.int32, device=self.device)
                if plan.boundary_positions
                else None,
                boundary_owner_mask=torch.tensor(plan.boundary_owner_mask, dtype=torch.bool, device=self.device)
                if plan.boundary_owner_mask
                else None,
                history_start_positions=torch.tensor(
                    plan.history_start_positions, dtype=torch.int32, device=self.device
                )
                if plan.history_start_positions
                else None,
                request_start_positions=torch.tensor(
                    plan.request_start_positions, dtype=torch.int32, device=self.device
                )
                if plan.request_start_positions
                else None,
                tail_token_ranges=plan.tail_token_ranges,
                sp_row_counts_per_rank=plan.sp_row_counts_per_rank,
                tp_rank=plan.tp_rank,
                tp_size=plan.tp_size,
            )

        def indices(values, index_slice):
            if index_slice is not None:
                return None
            return torch.tensor(values, dtype=torch.long, device=self.device)

        return CompressorSPMetadata(
            enabled=True,
            reason=plan.reason,
            ratio=plan.ratio,
            path=plan.path,
            coff=plan.coff,
            cache_mode=plan.cache_mode,
            is_chunked_prefill=plan.is_chunked_prefill,
            state_block_table_rows=plan.state_block_table_rows,
            start_pos_zero=plan.start_pos_zero,
            seq_len_aligned=plan.seq_len_aligned,
            requires_history_state=plan.requires_history_state,
            requires_tail_state_update=plan.requires_tail_state_update,
            requires_boundary_state_sync=plan.requires_boundary_state_sync,
            global_compressed_row_count=plan.global_compressed_row_count,
            boundary_req_indices=torch.tensor(plan.boundary_req_indices, dtype=torch.long, device=self.device),
            boundary_positions=torch.tensor(plan.boundary_positions, dtype=torch.int32, device=self.device),
            boundary_owner_mask=torch.tensor(plan.boundary_owner_mask, dtype=torch.bool, device=self.device),
            supports_boundary_state_replay=plan.supports_boundary_state_replay,
            boundary_replay_token_ranges=plan.boundary_replay_token_ranges,
            boundary_replay_token_indices=indices(
                plan.boundary_replay_token_indices,
                plan.boundary_replay_token_slice,
            ),
            boundary_replay_token_slice=plan.boundary_replay_token_slice,
            boundary_replay_req_indices=indices(
                plan.boundary_replay_req_indices,
                plan.boundary_replay_req_slice,
            ),
            boundary_replay_req_slice=plan.boundary_replay_req_slice,
            boundary_replay_cu_seqlens=torch.tensor(
                plan.boundary_replay_cu_seqlens,
                dtype=torch.int32,
                device=self.device,
            ),
            boundary_replay_start_pos=torch.tensor(
                plan.boundary_replay_start_pos,
                dtype=torch.int32,
                device=self.device,
            ),
            boundary_replay_compressed_row_indices=indices(
                plan.boundary_replay_compressed_row_indices,
                plan.boundary_replay_compressed_row_slice,
            ),
            boundary_replay_compressed_row_slice=(plan.boundary_replay_compressed_row_slice),
            boundary_replay_rope_row_indices=indices(
                plan.boundary_replay_rope_row_indices,
                plan.boundary_replay_rope_row_slice,
            ),
            boundary_replay_rope_row_slice=(plan.boundary_replay_rope_row_slice),
            history_start_positions=torch.tensor(plan.history_start_positions, dtype=torch.int32, device=self.device),
            request_start_positions=torch.tensor(plan.request_start_positions, dtype=torch.int32, device=self.device),
            token_indices=indices(plan.token_indices, plan.token_slice),
            token_slice=plan.token_slice,
            req_indices=indices(plan.req_indices, plan.req_slice),
            req_slice=plan.req_slice,
            cu_seqlens=torch.tensor(plan.cu_seqlens, dtype=torch.int32, device=self.device),
            start_pos=torch.tensor(plan.start_pos, dtype=torch.int32, device=self.device),
            compressed_row_indices=indices(plan.compressed_row_indices, plan.compressed_row_slice),
            compressed_row_slice=plan.compressed_row_slice,
            rope_row_indices=indices(plan.rope_row_indices, plan.rope_row_slice),
            rope_row_slice=plan.rope_row_slice,
            valid_row_indices=indices(plan.valid_row_indices, plan.valid_row_slice),
            valid_row_slice=plan.valid_row_slice,
            output_keep_indices=indices(plan.output_keep_indices, plan.output_keep_slice),
            output_keep_slice=plan.output_keep_slice,
            slot_mapping_indices=indices(plan.slot_mapping_indices, plan.slot_mapping_slice),
            slot_mapping_slice=plan.slot_mapping_slice,
            local_keep_to_full_row_indices=torch.tensor(
                plan.local_keep_to_full_row_indices, dtype=torch.long, device=self.device
            ),
            local_keep_to_slot_row_indices=torch.tensor(
                plan.local_keep_to_slot_row_indices, dtype=torch.long, device=self.device
            ),
            tail_token_ranges=plan.tail_token_ranges,
            padding_row_indices=indices(plan.padding_row_indices, plan.padding_row_slice),
            padding_row_slice=plan.padding_row_slice,
            gather_compact_indices=indices(plan.gather_compact_indices, plan.gather_compact_slice),
            gather_compact_slice=plan.gather_compact_slice,
            sp_row_counts_per_rank=plan.sp_row_counts_per_rank,
            tp_rank=plan.tp_rank,
            tp_size=plan.tp_size,
        )

    def _build_local_token_metadata(
        self,
        num_reqs,
        num_input_tokens,
        query_start_loc,
        seq_lens,
        local_query_start_loc=None,
        local_seq_lens=None,
        start_pos_out=None,
        is_noncausal=False,
    ):
        """
        For example:
        If we have TP size 3, num_input_tokens=45, and
        query_start_loc = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45].
        That means we have 9 requests with seq lens [1, 2, 3, 4, 5, 6, 7, 8, 9].
        For tp_rank 1, local_start=15, local_end=30, tokens_per_rank=15.
        local_query_start=[15, 15, 15, 15, 15, 15, 21, 28, 30]
        local_query_end = [15, 15, 15, 15, 15, 21, 28, 30, 30]
        local_query_lens = [0, 0, 0, 0, 0, 6, 7, 2, 0]
        self.local_query_start_loc = [0, 0, 0, 0, 0, 0, 6, 13, 15]
        offset = [-14, -12, -9, -5, 0, 0, 0, 6, 15]
        seq_lens-offset=[15, 14, 12, 9, 5, 6, 7, 2, -6]
        local_reqs_mask = [0, 0, 0, 0, 0, 1, 1, 1, 0]
        local_seq_lens = [0, 0, 0, 0, 0, 6, 7, 2, 0]
        """
        tp_group = get_tp_group()
        tp_size = tp_group.world_size
        tp_rank = tp_group.rank_in_group
        # Split the flattened token stream evenly across TP ranks. Padding keeps
        # every rank's local slice the same length, which simplifies CP kernels.
        num_tokens_pad = ((num_input_tokens + tp_size - 1) // tp_size) * tp_size
        tokens_per_rank = num_tokens_pad // tp_size
        local_start = tp_rank * tokens_per_rank
        local_end = local_start + tokens_per_rank

        if local_query_start_loc is not None:
            local_query_start_loc.fill_(0)
            local_seq_lens.fill_(0)

        if query_start_loc.device.type != "cpu" and HAS_TRITON:
            assert local_query_start_loc is not None and local_seq_lens is not None
            # Use next-power-of-2 block size to avoid wasted compute.
            build_local_metadata_triton[(1,)](
                query_start_loc,
                seq_lens,
                local_query_start_loc,
                local_seq_lens,
                local_start,
                local_end,
                num_reqs,
                start_pos_out if start_pos_out is not None else self._zero_i32,
                BLOCK_NUM_REQS=triton.next_power_of_2(num_reqs),
                COMPUTE_START_POS=start_pos_out is not None,
            )
        else:
            # torch fallback.
            # Intersect each request's global token interval with this rank's local
            # token interval, then build the per-rank query_start_loc from lengths.
            local_query_start = torch.clamp(query_start_loc[:-1], min=local_start, max=local_end)
            local_query_end = torch.clamp(query_start_loc[1:], min=local_start, max=local_end)
            local_query_lens = local_query_end - local_query_start
            if local_query_start_loc is not None:
                local_query_start_loc[1 : num_reqs + 1] = torch.cumsum(local_query_lens, dim=0)
            else:
                local_query_start_loc = torch.cat(
                    [
                        torch.tensor([0], dtype=local_query_lens.dtype, device=local_query_lens.device),
                        torch.cumsum(local_query_lens, dim=0),
                    ],
                    0,
                )

            # For requests that cross the local slice boundary, offset removes the
            # tokens that live on later ranks so local_seq_lens matches local queries.
            offset = query_start_loc[1:] - local_query_end
            valid_local_req = (local_query_lens > 0) & (seq_lens > 0)
            safe_local_seq_lens = torch.clamp_min(seq_lens - offset, 0)
            safe_local_seq_lens = torch.where(
                valid_local_req,
                safe_local_seq_lens,
                torch.zeros_like(safe_local_seq_lens),
            )
            if local_seq_lens is not None:
                local_seq_lens[:num_reqs] = safe_local_seq_lens
            else:
                local_seq_lens = safe_local_seq_lens

            if start_pos_out is not None:
                seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
                start_pos_out[:num_reqs] = seq_lens[:num_reqs] - seq_lens_q

        if is_noncausal:
            local_query_lens = local_query_start_loc[1 : num_reqs + 1] - local_query_start_loc[:num_reqs]
            local_seq_lens[:num_reqs].copy_(torch.where(local_query_lens > 0, seq_lens[:num_reqs], 0))
        return (
            local_start,
            local_end,
            tokens_per_rank,
            num_tokens_pad,
            local_query_start_loc[: num_reqs + 1],
            local_seq_lens[:num_reqs],
        )

    def _get_cmp_seqlens_for_metadata(self, has_prefill):
        if self.compressor_ratio <= 1:
            return None
        if has_prefill:
            return None
        return DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)

    def _build_sas_metadata(
        self,
        num_heads,
        query_start_loc,
        seq_lens,
        seq_lens_q,
        max_query_len,
        max_seq_lens,
        index_topk,
        num_reqs,
        has_prefill,
        cu_cmp_seqlen_list,
    ):
        cmp_ratio = self.compressor_ratio if self.compressor_ratio > 1 else 1
        cache_key = f"cp_sas_c{cmp_ratio}"
        metadata = self.common_ratio_to_sas_metadata.get(cache_key)
        if metadata is None:
            cu_seqlens_ori_kv = (
                query_start_loc
                if has_prefill
                else DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
                    self.common_ratio_to_sas_metadata,
                    f"{cache_key}_cu_seqlens_ori_kv",
                    seq_lens,
                    num_reqs,
                    self._zero_i32,
                    self.cu_seqlens_ori_kv,
                )
            )
            cu_seqlens_cmp_kv = (
                None if has_prefill else DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)
            )
            metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
            metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
            metadata_kwargs.setdefault("device", str(self.seqused_q.device))
            kw = dict(
                **metadata_kwargs,
                num_heads_q=num_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=query_start_loc,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=self.seqused_q,
                seqused_kv=seq_lens,
                max_seqlen_q=max_query_len,
                max_seqlen_kv=max_seq_lens,
                batch_size=num_reqs,
                ori_mask_mode=4,
                ori_win_left=self.model_config.hf_config.sliding_window - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
            )

            if self.compressor_ratio > 1:
                kw["has_cmp_kv"] = True
                if self.compressor_ratio == 4:
                    kw["cmp_mask_mode"] = 3
                    kw["cmp_topk"] = index_topk
                else:
                    kw["cmp_mask_mode"] = 3
                kw["cmp_ratio"] = cmp_ratio
                kw["cu_seqlens_cmp_kv"] = cu_cmp_seqlen_list
            else:
                kw["cmp_ratio"] = cmp_ratio
                kw["has_cmp_kv"] = False

            metadata = metadata_op(**kw)
        self.common_ratio_to_sas_metadata[cache_key] = metadata
        self.req_sas_metadata[:1024] = metadata
        return self.req_sas_metadata[:1024]

    def _build_qli_metadata(self, query_start_loc, seq_lens, seq_lens_q, num_reqs):
        if self.compressor_ratio != 4:
            return None

        cache_key = "cp_qli"
        metadata = self.common_ratio_to_sas_metadata.get(cache_key)

        if metadata is None:
            max_seqlen_q = max(1, int(seq_lens_q.max().item()))
            max_seqlen_k = max(1, int(seq_lens.max().item()))
            metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=query_start_loc[1:].clone(),
                actual_seq_lengths_key=seq_lens.clone(),
                num_heads_q=self.model_config.hf_config.index_n_heads,
                num_heads_k=1,
                head_dim=self.model_config.hf_config.index_head_dim,
                query_quant_mode=0,
                key_quant_mode=0,
                batch_size=num_reqs,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.model_config.hf_config.index_topk,
                sparse_mode=3,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                cmp_ratio=4,
                device=str(self.seqused_q.device),
            )
        self.common_ratio_to_sas_metadata[cache_key] = metadata
        self.req_qli_metadata[:1024] = metadata
        return self.req_qli_metadata[:1024]

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
                attn_state=attn_state,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                f"Graph capture only supports DecodeOnly and SpecDecoding attn states, got {attn_state}."
            )

        assert attn_metadata is not None
        return attn_metadata


class AscendDSACPImpl(DSAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    wo_a_full_pool: ClassVar[torch.Tensor | None] = None
    wo_a_full_weight_scale_pool: ClassVar[torch.Tensor | None] = None
    wo_b_full_pool: ClassVar[torch.Tensor | None] = None
    wo_b_full_weight_scale_pool: ClassVar[torch.Tensor | None] = None

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
        self.tp_group = get_tp_group()
        self.tp_size = self.tp_group.world_size
        self.tp_rank = self.tp_group.rank_in_group

        # MLA Args
        self.wq_a = kwargs["wq_a"]
        self.wq_b = kwargs["wq_b"]
        self.wkv = kwargs["wkv"]
        self.q_norm = kwargs["q_norm"]
        self.q_norm_without_weight = kwargs.get("q_norm_without_weight")
        self.kv_norm = kwargs["kv_norm"]

        self.indexer = kwargs.get("indexer")
        self.compressor = kwargs.get("compressor")

        self.wo_a = kwargs["wo_a"]
        self.wo_b = kwargs["wo_b"]

        self.enable_dsa_cp_with_o_proj_tp = enable_dsa_cp_with_o_proj_tp() and (
            get_ascend_device_type() == AscendDeviceType.A5
        )
        self._wo_a_dynamic_quant = False
        self._wo_b_dynamic_quant = False

        self.eps = kwargs["eps"]

        self.attn_sink = kwargs["attn_sink"]

        self.vllm_config = get_current_vllm_config()

        # indexer param
        if self.indexer is not None:
            self.indexer_heads: int = self.indexer.n_heads
            self.inderxer_dim: int = self.indexer.head_dim
            self.inderxer_wq_b = self.indexer.wq_b
            self.weights_proj = self.indexer.weights_proj
            self.indexer_softmax_scale = self.inderxer_dim**-0.5

            self.indexer_compress = self.indexer.compressor

            # indexer_compressor
            self.indexcom_ape = self.indexer.compressor.ape
            self.indexcom_wkv = self.indexer.compressor.wkv
            self.indexcom_wgate = self.indexer.compressor.wgate
            self.indexcom_norm = self.indexer.compressor.norm

            self.indexcom_head_dim = self.indexer.compressor.head_dim
            self.indexcom_rotate = self.indexer.compressor.rotate
            self.index_topk = self.indexer.index_topk

        # compress param
        if self.compressor is not None:
            self.compressor_head_dim = self.compressor.head_dim
            self.compressor_overlap = self.compressor.overlap
            self.compressor_rotate = self.compressor.rotate

            self.compressor_ape = self.compressor.ape
            self.compressor_wkv = self.compressor.wkv
            self.compressor_wgate = self.compressor.wgate
            self.compressor_norm = self.compressor.norm
            self.compressor_norm_eps = self.compressor.norm_eps
        self.enable_compressor_sp = ascend_envs.VLLM_ASCEND_ENABLE_COMPRESSOR_SP
        self.compressor_sp_debug_interval = ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_DEBUG_INTERVAL
        self.compressor_sp_debug_sync = ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_DEBUG_SYNC
        self.compressor_sp_dual_run = ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_DUAL_RUN
        self._compressor_sp_debug_events = 0
        self._compressor_sp_debug_stats: dict = {}
        # wkv is a required MLA param and always present, unlike the optional
        # compressor (some layers have compressor=None).
        self.device = self.wkv.weight.device

    def _compute_compressor_metadata(
        self,
        metadata: AscendDSAReqMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert metadata.full_compress_cos is not None
        assert metadata.full_compress_sin is not None
        assert metadata.num_compressed_tokens is not None
        assert metadata.start_pos is not None
        assert metadata.num_reqs_actual is not None
        full_compress_cos = metadata.full_compress_cos.view(
            metadata.full_compress_cos.shape[0],
            metadata.full_compress_cos.shape[-1],
        )
        full_compress_sin = metadata.full_compress_sin.view(
            metadata.full_compress_sin.shape[0],
            metadata.full_compress_sin.shape[-1],
        )
        return torch.ops._C_ascend.compressor_metadata(
            full_compress_cos,
            full_compress_sin,
            metadata.query_start_loc,
            metadata.start_pos,
            metadata.block_table,
            metadata.block_size,
            DeviceOperator.get_dsa_compressor_slot_mapping_format(),
            self.compress_ratio,
            metadata.num_compressed_tokens,
            metadata.num_reqs_actual,
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if self.attn_sink.numel() != self.num_heads:
            raise RuntimeError(
                "DSA-CP expects full-head attn_sink loaded on every TP rank, "
                f"got {self.attn_sink.numel()} heads, expected {self.num_heads}."
            )
        if self.enable_dsa_cp_with_o_proj_tp:
            self._maybe_init_o_proj_tp_full_params()

    @staticmethod
    def _check_dynamic_quant(layer: torch.nn.Module) -> bool:
        return get_ascend_device_type() in {AscendDeviceType.A5} and hasattr(layer, "weight_scale")

    def _maybe_init_o_proj_tp_full_params(self) -> None:
        self._wo_a_dynamic_quant = type(self)._check_dynamic_quant(self.wo_a)
        self._wo_b_dynamic_quant = type(self)._check_dynamic_quant(self.wo_b)
        if AscendDSACPImpl.wo_a_full_pool is None:
            sample = self.wo_a.weight
            AscendDSACPImpl.wo_a_full_pool = torch.empty(
                (sample.shape[0] * self.tp_size, *sample.shape[1:]),
                dtype=sample.dtype,
                device=sample.device,
            )
        self.wo_a_tp_weight = self.wo_a.weight.clone().detach().contiguous()
        self.wo_a.weight.set_(self.wo_a_tp_weight)
        if AscendDSACPImpl.wo_b_full_pool is None:
            sample = self.wo_b.weight
            AscendDSACPImpl.wo_b_full_pool = torch.empty(
                (sample.shape[0] * self.tp_size, *sample.shape[1:]),
                dtype=sample.dtype,
                device=sample.device,
            )
        self.wo_b_tp_weight = self.wo_b.weight.clone().detach().contiguous()
        self.wo_b.weight.set_(self.wo_b_tp_weight)

        if self._wo_a_dynamic_quant:
            if AscendDSACPImpl.wo_a_full_weight_scale_pool is None:
                sample = self.wo_a.weight_scale
                AscendDSACPImpl.wo_a_full_weight_scale_pool = torch.empty(
                    (sample.shape[0] * self.tp_size, *sample.shape[1:]),
                    dtype=sample.dtype,
                    device=sample.device,
                )
            self.wo_a_tp_weight_scale = self.wo_a.weight_scale.clone().detach().contiguous()
            self.wo_a.weight_scale.set_(self.wo_a_tp_weight_scale)
        if self._wo_b_dynamic_quant:
            if AscendDSACPImpl.wo_b_full_weight_scale_pool is None:
                sample = self.wo_b.weight_scale
                AscendDSACPImpl.wo_b_full_weight_scale_pool = torch.empty(
                    (sample.shape[0] * self.tp_size, *sample.shape[1:]),
                    dtype=sample.dtype,
                    device=sample.device,
                )
            self.wo_b_tp_weight_scale = self.wo_b.weight_scale.clone().detach().contiguous()
            self.wo_b.weight_scale.set_(self.wo_b_tp_weight_scale)

    def _maybe_all_gather_o_proj_full_weight(
        self,
        enabled: bool,
    ) -> list[torch.distributed.Work]:
        if not enabled:
            return []
        handles = []
        assert AscendDSACPImpl.wo_a_full_pool is not None
        _, weight_handle = all_gather_async(
            self.wo_a_tp_weight,
            self.tp_group,
            output=AscendDSACPImpl.wo_a_full_pool,
        )
        if weight_handle is not None:
            handles.append(weight_handle)
        assert AscendDSACPImpl.wo_b_full_pool is not None
        _, wo_b_weight_handle = all_gather_async(
            self.wo_b_tp_weight,
            self.tp_group,
            output=AscendDSACPImpl.wo_b_full_pool,
        )
        if wo_b_weight_handle is not None:
            handles.append(wo_b_weight_handle)
        if self._wo_a_dynamic_quant:
            assert AscendDSACPImpl.wo_a_full_weight_scale_pool is not None
            _, weight_scale_handle = all_gather_async(
                self.wo_a_tp_weight_scale,
                self.tp_group,
                output=AscendDSACPImpl.wo_a_full_weight_scale_pool,
            )
            if weight_scale_handle is not None:
                handles.append(weight_scale_handle)
        if self._wo_b_dynamic_quant:
            assert AscendDSACPImpl.wo_b_full_weight_scale_pool is not None
            _, wo_b_weight_scale_handle = all_gather_async(
                self.wo_b_tp_weight_scale,
                self.tp_group,
                output=AscendDSACPImpl.wo_b_full_weight_scale_pool,
            )
            if wo_b_weight_scale_handle is not None:
                handles.append(wo_b_weight_scale_handle)
        return handles

    def _switch_o_proj_to_full_weight(
        self,
        handles: list[torch.distributed.Work],
    ) -> None:
        for handle in handles:
            handle.wait()
        assert AscendDSACPImpl.wo_a_full_pool is not None
        self.wo_a.weight.set_(AscendDSACPImpl.wo_a_full_pool)
        if self._wo_a_dynamic_quant:
            assert AscendDSACPImpl.wo_a_full_weight_scale_pool is not None
            self.wo_a.weight_scale.set_(AscendDSACPImpl.wo_a_full_weight_scale_pool)
        assert AscendDSACPImpl.wo_b_full_pool is not None
        self.wo_b.weight.set_(AscendDSACPImpl.wo_b_full_pool)
        if self._wo_b_dynamic_quant:
            assert AscendDSACPImpl.wo_b_full_weight_scale_pool is not None
            self.wo_b.weight_scale.set_(AscendDSACPImpl.wo_b_full_weight_scale_pool)

    def _switch_o_proj_to_tp_weight(self) -> None:
        self.wo_a.weight.set_(self.wo_a_tp_weight)
        if self._wo_a_dynamic_quant:
            self.wo_a.weight_scale.set_(self.wo_a_tp_weight_scale)
        self.wo_b.weight.set_(self.wo_b_tp_weight)
        if self._wo_b_dynamic_quant:
            self.wo_b.weight_scale.set_(self.wo_b_tp_weight_scale)

    def _apply_wo_b(
        self,
        o_proj_input: torch.Tensor,
        full_weight: bool,
    ) -> torch.Tensor:
        if not full_weight:
            return self.wo_b(o_proj_input)
        return self.wo_b.quant_method.apply(self.wo_b, o_proj_input, bias=None)

    def _has_compressor_sp_selector(self, plan: CompressorSPMetadata, name: str) -> bool:
        return getattr(plan, f"{name}_slice") is not None or getattr(plan, f"{name}_indices") is not None

    def _compressor_sp_selector_len(self, plan: CompressorSPMetadata, name: str) -> int:
        index_slice = getattr(plan, f"{name}_slice")
        if index_slice is not None:
            return int(index_slice[1])
        indices = getattr(plan, f"{name}_indices")
        return int(indices.numel()) if indices is not None else 0

    def _select_compressor_sp_dim0(self, tensor: torch.Tensor, plan: CompressorSPMetadata, name: str) -> torch.Tensor:
        index_slice = getattr(plan, f"{name}_slice")
        if index_slice is not None:
            start, length = index_slice
            return tensor.narrow(0, int(start), int(length))
        indices = getattr(plan, f"{name}_indices")
        assert indices is not None
        return tensor.index_select(0, indices)

    def _compressor_sp_layout_assert_enabled(self) -> bool:
        return bool(ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_LAYOUT_ASSERT)

    def _compressor_sp_selector_indices(
        self,
        plan: CompressorSPMetadata,
        name: str,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        index_slice = getattr(plan, f"{name}_slice")
        if index_slice is not None:
            start, length = index_slice
            return torch.arange(
                int(start),
                int(start) + int(length),
                dtype=torch.long,
                device=device,
            )
        indices = getattr(plan, f"{name}_indices")
        if indices is None:
            return torch.empty(0, dtype=torch.long, device=device)
        return indices.to(device=device, dtype=torch.long)

    def _assert_compressor_sp_equal_tensor(
        self,
        *,
        label: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        plan: CompressorSPMetadata,
    ) -> None:
        if actual.shape != expected.shape:
            raise RuntimeError(
                "CompressorSP layout assert failed: "
                f"{label} shape mismatch, actual={tuple(actual.shape)}, "
                f"expected={tuple(expected.shape)}, path={plan.path}, "
                f"ratio={plan.ratio}, coff={plan.coff}, rank={plan.tp_rank}/{plan.tp_size}"
            )
        if actual.numel() == 0:
            return
        if not torch.equal(actual, expected):
            raise RuntimeError(
                "CompressorSP layout assert failed: "
                f"{label} value mismatch, shape={tuple(actual.shape)}, "
                f"path={plan.path}, ratio={plan.ratio}, coff={plan.coff}, "
                f"rank={plan.tp_rank}/{plan.tp_size}"
            )

    def _assert_compressor_sp_keep_layout(
        self,
        *,
        local_compressed_kv: torch.Tensor,
        kept_rows: torch.Tensor,
        plan: CompressorSPMetadata,
    ) -> None:
        if not self._compressor_sp_layout_assert_enabled():
            return

        keep_indices = self._compressor_sp_selector_indices(plan, "output_keep", device=local_compressed_kv.device)
        expected_keep_rows = (
            local_compressed_kv.index_select(0, keep_indices) if keep_indices.numel() > 0 else local_compressed_kv[:0]
        )
        self._assert_compressor_sp_equal_tensor(
            label="output_keep rows",
            actual=kept_rows,
            expected=expected_keep_rows,
            plan=plan,
        )

        if keep_indices.numel() != plan.local_keep_to_full_row_indices.numel():
            raise RuntimeError(
                "CompressorSP layout assert failed: output_keep length "
                f"{int(keep_indices.numel())} != local_keep_to_full_row length "
                f"{int(plan.local_keep_to_full_row_indices.numel())}"
            )
        if keep_indices.numel() != plan.local_keep_to_slot_row_indices.numel():
            raise RuntimeError(
                "CompressorSP layout assert failed: output_keep length "
                f"{int(keep_indices.numel())} != local_keep_to_slot_row length "
                f"{int(plan.local_keep_to_slot_row_indices.numel())}"
            )

        compressed_row_indices = self._compressor_sp_selector_indices(
            plan, "compressed_row", device=local_compressed_kv.device
        )
        if keep_indices.numel() > 0:
            mapped_full_rows = compressed_row_indices.index_select(0, keep_indices)
        else:
            mapped_full_rows = compressed_row_indices[:0]
        self._assert_compressor_sp_equal_tensor(
            label="output_keep -> full_row mapping",
            actual=plan.local_keep_to_full_row_indices.to(device=local_compressed_kv.device, dtype=torch.long),
            expected=mapped_full_rows,
            plan=plan,
        )

        slot_indices = self._compressor_sp_selector_indices(plan, "slot_mapping", device=local_compressed_kv.device)
        self._assert_compressor_sp_equal_tensor(
            label="slot_mapping -> slot_row mapping",
            actual=plan.local_keep_to_slot_row_indices.to(device=local_compressed_kv.device, dtype=torch.long),
            expected=slot_indices,
            plan=plan,
        )

    def _assert_compressor_sp_token_layout(
        self,
        *,
        full_x: torch.Tensor,
        selected_x: torch.Tensor,
        plan: CompressorSPMetadata,
        selector_name: str = "token",
    ) -> None:
        if not self._compressor_sp_layout_assert_enabled():
            return
        token_indices = self._compressor_sp_selector_indices(plan, selector_name, device=full_x.device)
        expected_x = full_x.index_select(0, token_indices) if token_indices.numel() > 0 else full_x[:0]
        self._assert_compressor_sp_equal_tensor(
            label=f"{selector_name} layout",
            actual=selected_x,
            expected=expected_x,
            plan=plan,
        )

    def _assert_compressor_sp_req_layout(
        self,
        *,
        full_state_block_table: torch.Tensor,
        selected_state_block_table: torch.Tensor,
        plan: CompressorSPMetadata,
        selector_name: str = "req",
    ) -> None:
        if not self._compressor_sp_layout_assert_enabled():
            return
        req_indices = self._compressor_sp_selector_indices(plan, selector_name, device=full_state_block_table.device)
        expected_state_block_table = (
            full_state_block_table.index_select(0, req_indices)
            if req_indices.numel() > 0
            else full_state_block_table[:0]
        )
        self._assert_compressor_sp_equal_tensor(
            label=f"{selector_name} state_block_table layout",
            actual=selected_state_block_table,
            expected=expected_state_block_table,
            plan=plan,
        )

    def _assert_compressor_sp_slot_layout(
        self,
        *,
        full_slot_mapping: torch.Tensor,
        selected_slot_mapping: torch.Tensor,
        plan: CompressorSPMetadata,
    ) -> None:
        if not self._compressor_sp_layout_assert_enabled():
            return
        slot_row_indices = plan.local_keep_to_slot_row_indices.to(device=full_slot_mapping.device, dtype=torch.long)
        expected_slot_mapping = (
            full_slot_mapping.index_select(0, slot_row_indices)
            if slot_row_indices.numel() > 0
            else full_slot_mapping[:0]
        )
        self._assert_compressor_sp_equal_tensor(
            label="slot mapping selector",
            actual=selected_slot_mapping,
            expected=expected_slot_mapping,
            plan=plan,
        )

    def _assert_compressor_sp_rope_layout(
        self,
        *,
        rope: torch.Tensor,
        selected_rope: torch.Tensor,
        plan: CompressorSPMetadata,
        num_tokens: int,
        rope_row_selector: str,
        req_selector: str,
    ) -> None:
        if not self._compressor_sp_layout_assert_enabled():
            return

        layout_indices = self._compressor_sp_selector_indices(plan, rope_row_selector, device=rope.device)
        target_rows = min(
            num_tokens,
            num_tokens // self.compress_ratio + self._compressor_sp_selector_len(plan, req_selector),
        )
        if int(layout_indices.numel()) != target_rows:
            raise RuntimeError(
                "CompressorSP layout assert failed: RoPE selector length "
                f"{int(layout_indices.numel())} != target {target_rows}, "
                f"path={plan.path}, ratio={plan.ratio}"
            )
        expected_rope = rope.index_select(0, layout_indices) if layout_indices.numel() > 0 else rope[:0]
        self._assert_compressor_sp_equal_tensor(
            label=f"{rope_row_selector} RoPE layout",
            actual=selected_rope,
            expected=expected_rope,
            plan=plan,
        )

    def _assert_compressor_sp_gather_layout(
        self,
        *,
        gathered_rows: torch.Tensor,
        global_rows: torch.Tensor,
        plan: CompressorSPMetadata,
    ) -> None:
        if not self._compressor_sp_layout_assert_enabled():
            return
        sp_row_counts = plan.sp_row_counts_per_rank
        if not sp_row_counts:
            return
        total_rows = sum(sp_row_counts)
        if plan.global_compressed_row_count > 0 and total_rows != plan.global_compressed_row_count:
            raise RuntimeError(
                "CompressorSP layout assert failed: gathered row count "
                f"{total_rows} != global_compressed_row_count "
                f"{plan.global_compressed_row_count}, path={plan.path}, "
                f"ratio={plan.ratio}, coff={plan.coff}"
            )

        compact_index_tensor = self._compressor_sp_selector_indices(plan, "gather_compact", device=global_rows.device)
        if compact_index_tensor.numel() > 0:
            compact_rows = gathered_rows.index_select(0, compact_index_tensor)
        else:
            compact_rows = global_rows[:0]

        self._assert_compressor_sp_equal_tensor(
            label="all_gather compact_indices",
            actual=global_rows,
            expected=compact_rows,
            plan=plan,
        )

    def _gather_compressor_sp_buffer(self, local_rows, plan):
        """AllGather padded rank rows; return the raw rank-major buffer.

        Returns (gathered_flat, max_rows). Unlike ``_gather_compressor_sp_rows``
        this does not compact the buffer with the gather_compact selector; the
        padded-scatter path feeds the raw buffer to the selected scatter with a
        padded destination, skipping the ``global_compressed_kv`` materialization.
        """
        sp = plan.sp_row_counts_per_rank
        tp_sz, tp_rk = plan.tp_size, plan.tp_rank
        if not sp or tp_sz <= 1:
            return local_rows, local_rows.shape[0]
        if sum(sp) == 0:
            return local_rows[:0], 0

        expected_local = sp[tp_rk]
        if local_rows.shape[0] != expected_local:
            raise RuntimeError(
                f"CompressorSP gather: local_rows has {local_rows.shape[0]} rows "
                f"but plan expects {expected_local} for rank {tp_rk}"
            )

        max_rows = max(sp)
        head_dim = local_rows.shape[1] if local_rows.dim() >= 2 else 0
        if head_dim == 0:
            raise RuntimeError(
                "CompressorSP gather: local_rows must be at least 2D "
                f"[num_rows, head_dim], got shape {local_rows.shape}"
            )

        if local_rows.shape[0] < max_rows:
            padded_local = local_rows.new_zeros(max_rows, head_dim)
            padded_local[: local_rows.shape[0]].copy_(local_rows)
        else:
            padded_local = local_rows

        gathered_flat = local_rows.new_empty((tp_sz * max_rows, *local_rows.shape[1:]))
        dist.all_gather_into_tensor(gathered_flat, padded_local, group=self.tp_group.device_group)
        return gathered_flat, max_rows

    def _gather_compressor_sp_rows(
        self,
        local_rows: torch.Tensor,
        plan: CompressorSPMetadata,
    ) -> torch.Tensor:
        """Gather padded rank rows and select valid rows in rank order.

        Every rank contributes a fixed [max_rows, head_dim] tensor. The
        planner-provided gather_compact selector removes padded rows from the
        contiguous rank-major all-gather output.
        """
        sp_row_counts = plan.sp_row_counts_per_rank
        tp_size = plan.tp_size
        tp_rank = plan.tp_rank

        if not sp_row_counts or tp_size <= 1:
            return local_rows

        total_rows = sum(sp_row_counts)
        if total_rows == 0:
            return local_rows[:0]

        # Validate local row count matches plan expectation
        expected_local = sp_row_counts[tp_rank]
        if local_rows.shape[0] != expected_local:
            raise RuntimeError(
                f"CompressorSP gather: local_rows has {local_rows.shape[0]} rows "
                f"but plan expects {expected_local} for rank {tp_rank}"
            )

        max_rows = max(sp_row_counts)
        head_dim = local_rows.shape[1] if local_rows.dim() >= 2 else 0
        if head_dim == 0:
            raise RuntimeError(
                "CompressorSP gather: local_rows must be at least 2D "
                f"[num_rows, head_dim], got shape {local_rows.shape}"
            )

        # Pad without a ConcatD. HCCL still receives one fixed-shape
        # contiguous tensor from every rank.
        if local_rows.shape[0] < max_rows:
            padded_local = local_rows.new_zeros(max_rows, head_dim)
            padded_local[: local_rows.shape[0]].copy_(local_rows)
        else:
            padded_local = local_rows

        # Gather directly into one rank-major contiguous buffer so compact
        # selection does not require a runtime list-to-stack reconstruction.
        gathered_flat = local_rows.new_empty((tp_size * max_rows, *local_rows.shape[1:]))
        dist.all_gather_into_tensor(
            gathered_flat,
            padded_local,
            group=self.tp_group.device_group,
        )

        # Compact the fixed-shape rank-major buffer with the metadata
        # selector instead of rebuilding a dynamic parts list and ConcatD.
        global_rows = self._select_compressor_sp_dim0(gathered_flat, plan, "gather_compact")
        assert global_rows.shape[0] == total_rows, (
            f"CompressorSP gather: expected {total_rows} global rows, got {global_rows.shape[0]}"
        )
        self._assert_compressor_sp_gather_layout(
            gathered_rows=gathered_flat,
            global_rows=global_rows,
            plan=plan,
        )
        return global_rows

    def _compressor_sp_uses_boundary_replay(self, plan: CompressorSPMetadata | None) -> bool:
        return bool(
            plan is not None
            and plan.enabled
            and plan.ratio == 4
            and plan.is_chunked_prefill
            and plan.requires_boundary_state_sync
            and plan.supports_boundary_state_replay
            and (ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_REPLAY_C4_BOUNDARY_STATE or self.compressor_sp_dual_run)
        )

    def _compressor_sp_boundary_replay_metadata_complete(self, plan: CompressorSPMetadata) -> bool:
        return bool(
            self._has_compressor_sp_selector(plan, "boundary_replay_token")
            and self._has_compressor_sp_selector(plan, "boundary_replay_req")
            and self._has_compressor_sp_selector(plan, "boundary_replay_compressed_row")
            and self._has_compressor_sp_selector(plan, "boundary_replay_rope_row")
            and plan.boundary_replay_cu_seqlens is not None
            and plan.boundary_replay_start_pos is not None
            and self._compressor_sp_selector_len(plan, "boundary_replay_token") > 0
            and self._compressor_sp_selector_len(plan, "boundary_replay_req") > 0
        )

    def _compressor_sp_unavailable_reason(
        self,
        plan: CompressorSPMetadata | None,
        *,
        path: str,
        need_gather_q_kv: bool,
        has_prefill: bool,
        coff: int,
        cache_mode: int = 1,
    ) -> str | None:
        if path not in {"main", "indexer", "metadata", "spec", "mtp"}:
            return "unknown_path"
        if path in {"spec", "mtp"}:
            return "unsupported_spec_or_mtp"
        if not self.enable_compressor_sp:
            return "runtime_env_disabled"
        if not has_prefill:
            return "not_prefill"
        if not need_gather_q_kv:
            return "no_need_gather"
        if self.compress_ratio not in (4, 128):
            return "unsupported_ratio"
        if self.compress_ratio == 4 and coff != 2:
            return "unsupported_coff"
        if self.compress_ratio == 128 and coff != 1:
            return "unsupported_coff"
        if cache_mode != 1:
            return "unsupported_cache_mode"
        if plan is None:
            return "missing_compressor_sp_plan"
        if not plan.enabled:
            return plan.reason
        if plan.is_chunked_prefill and not (
            ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_CHUNKED_PREFILL or self.compressor_sp_dual_run
        ):
            return "unsupported_chunked_prefill"
        uses_boundary_replay = self._compressor_sp_uses_boundary_replay(plan)
        if plan.is_chunked_prefill and plan.requires_boundary_state_sync:
            if uses_boundary_replay:
                if not self._compressor_sp_boundary_replay_metadata_complete(plan):
                    return "missing_boundary_replay_metadata"
            elif not ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_SYNC_CHUNKED_BOUNDARY_STATE:
                return "missing_chunked_boundary_state_sync"
        if len(plan.sp_row_counts_per_rank) != plan.tp_size or not self._has_compressor_sp_selector(
            plan, "gather_compact"
        ):
            return "missing_gather_layout_metadata"
        if not uses_boundary_replay and not all_ranks_have_compressor_sp_rows(plan.sp_row_counts_per_rank):
            # Every rank must make the same collective-entry decision. The
            # replay path is the only validated path where a rank may
            # contribute an empty padded tensor.
            return "zero_row_rank_collective_unsupported"
        if self._compressor_sp_selector_len(plan, "token") == 0 and not uses_boundary_replay:
            return "no_local_compressed_rows"
        if plan.state_block_table_rows <= 0:
            return "missing_history_state" if plan.requires_history_state else "missing_metadata"
        if (
            self.compress_ratio == 4
            and plan.start_pos_zero is False
            and not (ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_C4_NONZERO_START or self.compressor_sp_dual_run)
        ):
            return "start_pos_nonzero_unverified"
        if (
            self.compress_ratio == 4
            and not plan.seq_len_aligned
            and not (ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_C4_NONALIGNED or self.compressor_sp_dual_run)
        ):
            return "seq_len_not_aligned_c4"
        if (
            self.compress_ratio == 128
            and not plan.seq_len_aligned
            and not (ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_C128_NONALIGNED or self.compressor_sp_dual_run)
        ):
            return "seq_len_not_aligned_c128"
        allow_tail_state_update = (
            ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_C4_NONALIGNED
            if self.compress_ratio == 4
            else ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_C128_NONALIGNED
        )
        if plan.requires_tail_state_update and not (allow_tail_state_update or self.compressor_sp_dual_run):
            return "tail_state_update_unverified"
        for name in (
            "token",
            "req",
            "compressed_row",
            "rope_row",
            "output_keep",
            "slot_mapping",
            "gather_compact",
        ):
            if not self._has_compressor_sp_selector(plan, name):
                return "missing_metadata"
        if plan.cu_seqlens is None or plan.start_pos is None:
            return "missing_metadata"
        if plan.local_keep_to_full_row_indices is None or plan.local_keep_to_slot_row_indices is None:
            return "missing_metadata"
        if self._compressor_sp_selector_len(plan, "output_keep") != int(plan.local_keep_to_full_row_indices.numel()):
            return "local_rows_keep_rows_mismatch"
        if self._compressor_sp_selector_len(plan, "gather_compact") != sum(plan.sp_row_counts_per_rank):
            return "gather_compact_rows_mismatch"
        return None

    def _compressor_sp_debug_start(self) -> float | None:
        if self.compressor_sp_debug_interval <= 0 or not self.compressor_sp_debug_sync:
            return None
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        return time.perf_counter()

    def _compressor_sp_debug_elapsed_ms(self, start: float | None) -> float:
        if start is None:
            return 0.0
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        return (time.perf_counter() - start) * 1000

    def _record_compressor_sp_debug(
        self,
        *,
        path: str,
        status: str,
        reason: str,
        layer_name: str,
        plan: CompressorSPMetadata | None,
        coff: int = 0,
        full_shape_call: bool = False,
        local_shape_call: bool = False,
        elapsed_ms: float = 0.0,
    ) -> None:
        interval = self.compressor_sp_debug_interval
        if interval <= 0:
            return

        token_count = self._compressor_sp_selector_len(plan, "token") if plan and plan.enabled else 0
        compressed_rows = self._compressor_sp_selector_len(plan, "compressed_row") if plan and plan.enabled else 0
        keep_rows = self._compressor_sp_selector_len(plan, "output_keep") if plan and plan.enabled else 0
        slot_rows = self._compressor_sp_selector_len(plan, "slot_mapping") if plan and plan.enabled else 0
        discarded_rows = max(0, compressed_rows - keep_rows)
        chunked_prefill = "unknown"
        start_pos_zero = "unknown"
        seq_len_aligned = "unknown"
        if plan is not None:
            chunked_prefill = str(bool(plan.is_chunked_prefill)).lower()
            if plan.start_pos_zero is not None:
                start_pos_zero = str(bool(plan.start_pos_zero)).lower()
            if plan.seq_len_aligned is not None:
                seq_len_aligned = str(bool(plan.seq_len_aligned)).lower()

        key = (path, status, self.compress_ratio, coff, chunked_prefill, start_pos_zero, seq_len_aligned, reason)
        entry = self._compressor_sp_debug_stats.setdefault(
            key,
            {
                "count": 0,
                "full_shape_calls": 0,
                "local_shape_calls": 0,
                "tokens": 0,
                "compressed_rows": 0,
                "keep_rows": 0,
                "discarded_rows": 0,
                "slot_rows": 0,
                "elapsed_ms": 0.0,
                "layers": set(),
            },
        )
        entry["count"] = int(entry["count"]) + 1
        entry["full_shape_calls"] = int(entry["full_shape_calls"]) + int(full_shape_call)
        entry["local_shape_calls"] = int(entry["local_shape_calls"]) + int(local_shape_call)
        entry["tokens"] = int(entry["tokens"]) + token_count
        entry["compressed_rows"] = int(entry["compressed_rows"]) + compressed_rows
        entry["keep_rows"] = int(entry["keep_rows"]) + keep_rows
        entry["discarded_rows"] = int(entry["discarded_rows"]) + discarded_rows
        entry["slot_rows"] = int(entry["slot_rows"]) + slot_rows
        entry["elapsed_ms"] = float(entry["elapsed_ms"]) + elapsed_ms
        layers = entry["layers"]
        assert isinstance(layers, set)
        layers.add(layer_name)

        self._compressor_sp_debug_events += 1
        if interval == 1:
            logger.warning(
                (
                    "Compressor SP decision event=%d path=%s status=%s ratio=%d coff=%d "
                    "chunked_prefill=%s start_pos_zero=%s seq_len_aligned=%s reason=%s full_shape_calls=%d "
                    "local_shape_calls=%d layer=%s tokens=%d compressed_rows=%d keep_rows=%d "
                    "discarded_rows=%d slot_rows=%d elapsed_ms=%.3f"
                ),
                self._compressor_sp_debug_events,
                path,
                status,
                self.compress_ratio,
                coff,
                chunked_prefill,
                start_pos_zero,
                seq_len_aligned,
                reason,
                int(full_shape_call),
                int(local_shape_call),
                layer_name,
                token_count,
                compressed_rows,
                keep_rows,
                discarded_rows,
                slot_rows,
                elapsed_ms,
            )
            return
        if self._compressor_sp_debug_events % interval == 0:
            self._log_compressor_sp_debug_stats()

    def _log_compressor_sp_debug_stats(self) -> None:
        parts = []
        for (path, status, ratio, coff, chunked_prefill, start_pos_zero, seq_len_aligned, reason), entry in sorted(
            self._compressor_sp_debug_stats.items()
        ):
            layers = entry["layers"]
            assert isinstance(layers, set)
            parts.append(
                f"path={path} status={status} ratio={ratio} coff={coff} "
                f"chunked_prefill={chunked_prefill} start_pos_zero={start_pos_zero} "
                f"seq_len_aligned={seq_len_aligned} reason={reason} "
                f"count={entry['count']} full_shape_calls={entry['full_shape_calls']} "
                f"local_shape_calls={entry['local_shape_calls']} layers={len(layers)} tokens={entry['tokens']} "
                f"compressed_rows={entry['compressed_rows']} keep_rows={entry['keep_rows']} "
                f"discarded_rows={entry['discarded_rows']} slot_rows={entry['slot_rows']} "
                f"elapsed_ms={float(entry['elapsed_ms']):.3f}"
            )
        logger.warning(
            "Compressor SP debug stats after %d events: %s",
            self._compressor_sp_debug_events,
            " | ".join(parts),
        )

    def _compressor_sp_status_for_reason(self, reason: str) -> str:
        if reason == "missing_compressor_sp_plan":
            return "full_without_plan"
        if reason in {
            "not_prefill",
            "no_need_gather",
            "unsupported_spec_or_mtp",
            "unsupported_chunked_prefill",
            "unknown_path",
            "runtime_env_disabled",
        }:
            return "unsupported_path"
        return "fallback_with_plan"

    def _compressor_sp_should_dual_run(self) -> bool:
        return self.compressor_sp_dual_run

    def _compressor_sp_state_rows(
        self,
        *,
        input_positions: torch.Tensor,
        state_block_table: torch.Tensor,
        plan: CompressorSPMetadata,
        state_block_size: int,
    ) -> torch.Tensor:
        token_positions = self._select_compressor_sp_dim0(input_positions, plan, "token")
        req_block_table = self._select_compressor_sp_dim0(state_block_table, plan, "req")
        row_ids = set(
            collect_state_row_indices(
                token_positions=token_positions,
                req_block_table=req_block_table,
                cu_seqlens=plan.cu_seqlens,
                state_block_size=state_block_size,
            )
        )
        if self._compressor_sp_uses_boundary_replay(plan):
            replay_positions = self._select_compressor_sp_dim0(input_positions, plan, "boundary_replay_token")
            replay_block_table = self._select_compressor_sp_dim0(state_block_table, plan, "boundary_replay_req")
            row_ids.update(
                collect_state_row_indices(
                    token_positions=replay_positions,
                    req_block_table=replay_block_table,
                    cu_seqlens=plan.boundary_replay_cu_seqlens,
                    state_block_size=state_block_size,
                )
            )
        if not row_ids:
            return token_positions[:0]
        return torch.tensor(sorted(row_ids), dtype=torch.long, device=token_positions.device)

    def _sync_compressor_sp_boundary_state(
        self,
        *,
        state_cache: torch.Tensor,
        state_block_table: torch.Tensor,
        plan: CompressorSPMetadata | None,
    ) -> None:
        if self._compressor_sp_uses_boundary_replay(plan):
            return
        if (
            plan is None
            or not plan.is_chunked_prefill
            or not plan.requires_boundary_state_sync
            or not (ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_CHUNKED_PREFILL or self.compressor_sp_dual_run)
            or not ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_SYNC_CHUNKED_BOUNDARY_STATE
        ):
            return
        if plan.boundary_req_indices is None or plan.boundary_positions is None or plan.boundary_owner_mask is None:
            raise RuntimeError("Chunked Compressor SP is missing boundary-state metadata")

        sync_boundary_state_blocks(
            state_cache=state_cache,
            state_block_table=state_block_table,
            boundary_req_indices=plan.boundary_req_indices,
            boundary_positions=plan.boundary_positions,
            boundary_owner_mask=plan.boundary_owner_mask,
            all_reduce=get_tp_group().all_reduce,
        )

    def _compressor_sp_compare_rows(
        self,
        *,
        label: str,
        local_rows: torch.Tensor,
        full_rows: torch.Tensor,
        layer_name: str,
        path: str,
        coff: int,
        plan: CompressorSPMetadata | None,
    ) -> str | None:
        if local_rows.shape != full_rows.shape:
            return "local_rows_keep_rows_mismatch"
        if local_rows.numel() == 0:
            return None
        local_f = local_rows.float()
        full_f = full_rows.float()
        diff = (local_f - full_f).abs()
        max_err = float(diff.max().item()) if diff.numel() else 0.0
        mean_err = float(diff.mean().item()) if diff.numel() else 0.0
        if not torch.allclose(local_f, full_f, rtol=1e-2, atol=1e-2):
            logger.warning(
                "Compressor SP dual-run %s mismatch: path=%s layer=%s coff=%s max_err=%.6f mean_err=%.6f plan=%s",
                label,
                path,
                layer_name,
                coff,
                max_err,
                mean_err,
                plan.reason if plan is not None else "missing_plan",
            )
            return "local_rows_keep_rows_mismatch"
        return None

    def _compressor_sp_compare_state_rows(
        self,
        *,
        local_state: torch.Tensor,
        full_state: torch.Tensor,
        state_rows: torch.Tensor,
        layer_name: str,
        path: str,
        coff: int,
        plan: CompressorSPMetadata | None,
    ) -> str | None:
        if state_rows.numel() == 0:
            return "missing_history_state" if plan is not None and plan.requires_history_state else None
        local_rows = local_state.index_select(0, state_rows.long())
        full_rows = full_state.index_select(0, state_rows.long())
        if local_rows.shape != full_rows.shape:
            return "missing_history_state"
        local_f = local_rows.float()
        full_f = full_rows.float()
        diff = (local_f - full_f).abs()
        max_err = float(diff.max().item()) if diff.numel() else 0.0
        mean_err = float(diff.mean().item()) if diff.numel() else 0.0
        if not torch.allclose(local_f, full_f, rtol=1e-2, atol=1e-2):
            logger.warning(
                "Compressor SP dual-run state mismatch: path=%s layer=%s coff=%s "
                "max_err=%.6f mean_err=%.6f plan=%s rows=%d",
                path,
                layer_name,
                coff,
                max_err,
                mean_err,
                plan.reason if plan is not None else "missing_plan",
                int(state_rows.numel()),
            )
            return (
                "tail_state_update_unverified"
                if plan is not None and plan.requires_tail_state_update
                else "missing_history_state"
            )
        return None

    def _compressor_sp_compare_cache_rows(
        self,
        *,
        local_cache: torch.Tensor,
        full_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_name: str,
        path: str,
        coff: int,
        plan: CompressorSPMetadata | None,
    ) -> str | None:
        if slot_mapping.numel() == 0:
            return None
        local_rows = self._gather_cache_rows(local_cache, slot_mapping)
        full_rows = self._gather_cache_rows(full_cache, slot_mapping)
        return self._compressor_sp_compare_rows(
            label="cache",
            local_rows=local_rows,
            full_rows=full_rows,
            layer_name=layer_name,
            path=path,
            coff=coff,
            plan=plan,
        )

    def _gather_cache_rows(self, cache: torch.Tensor, slot_mapping: torch.Tensor) -> torch.Tensor:
        block_idx = slot_mapping[:, 0].long()
        offset = slot_mapping[:, 1].long()
        return cache[block_idx, offset]

    def _can_use_compressor_sp(
        self,
        plan: CompressorSPMetadata | None,
        *,
        need_gather_q_kv: bool,
        has_prefill: bool,
        coff: int,
    ) -> bool:
        return (
            self._compressor_sp_unavailable_reason(
                plan, path="main", need_gather_q_kv=need_gather_q_kv, has_prefill=has_prefill, coff=coff
            )
            is None
        )

    def _run_compressor_sp(
        self,
        *,
        x: torch.Tensor,
        plan: CompressorSPMetadata,
        wkv: torch.Tensor,
        wgate: torch.Tensor,
        state_cache: torch.Tensor,
        ape: torch.Tensor,
        norm_weight: torch.Tensor,
        compressed_sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        state_block_table: torch.Tensor,
        coff: int,
    ) -> torch.Tensor | None:
        full_x = x
        full_compressed_sin = compressed_sin.view(-1, compressed_sin.shape[-1])
        full_compressed_cos = compressed_cos.view(-1, compressed_cos.shape[-1])
        if (
            self._compressor_sp_selector_len(plan, "token") == 0
            or self._compressor_sp_selector_len(plan, "compressed_row") == 0
        ):
            if not self._compressor_sp_uses_boundary_replay(plan):
                return None
            self._run_compressor_sp_boundary_replay(
                x=full_x,
                plan=plan,
                wkv=wkv,
                wgate=wgate,
                state_cache=state_cache,
                ape=ape,
                norm_weight=norm_weight,
                compressed_sin=full_compressed_sin,
                compressed_cos=full_compressed_cos,
                state_block_table=state_block_table,
                coff=coff,
            )
            return full_x.new_empty((0, wkv.shape[0] // coff))

        x = self._select_compressor_sp_dim0(full_x, plan, "token")
        self._assert_compressor_sp_token_layout(
            full_x=full_x,
            selected_x=x,
            plan=plan,
            selector_name="token",
        )
        compressed_sin = self._select_compressor_sp_rope(full_compressed_sin, plan, x.shape[0])
        compressed_cos = self._select_compressor_sp_rope(full_compressed_cos, plan, x.shape[0])
        full_state_block_table = state_block_table
        state_block_table = self._select_compressor_sp_dim0(full_state_block_table, plan, "req")
        self._assert_compressor_sp_req_layout(
            full_state_block_table=full_state_block_table,
            selected_state_block_table=state_block_table,
            plan=plan,
            selector_name="req",
        )

        compressed_kv = run_compressor_op(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            norm_weight,
            compressed_sin,
            compressed_cos,
            state_block_table=state_block_table,
            cu_seqlens=plan.cu_seqlens,
            seqused=None,
            start_pos=plan.start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=coff,
            norm_eps=self.compressor_norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )
        self._run_compressor_sp_boundary_replay(
            x=full_x,
            plan=plan,
            wkv=wkv,
            wgate=wgate,
            state_cache=state_cache,
            ape=ape,
            norm_weight=norm_weight,
            compressed_sin=full_compressed_sin,
            compressed_cos=full_compressed_cos,
            state_block_table=full_state_block_table,
            coff=coff,
        )
        if compressed_kv.numel() == 0:
            return compressed_kv
        if self._compressor_sp_selector_len(plan, "output_keep") == 0:
            kept_rows = compressed_kv[:0]
            self._assert_compressor_sp_keep_layout(
                local_compressed_kv=compressed_kv,
                kept_rows=kept_rows,
                plan=plan,
            )
            return kept_rows
        kept_rows = self._select_compressor_sp_dim0(compressed_kv, plan, "output_keep")
        self._assert_compressor_sp_keep_layout(
            local_compressed_kv=compressed_kv,
            kept_rows=kept_rows,
            plan=plan,
        )
        return kept_rows

    def _run_compressor_sp_boundary_replay(
        self,
        *,
        x: torch.Tensor,
        plan: CompressorSPMetadata,
        wkv: torch.Tensor,
        wgate: torch.Tensor,
        state_cache: torch.Tensor,
        ape: torch.Tensor,
        norm_weight: torch.Tensor,
        compressed_sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        state_block_table: torch.Tensor,
        coff: int,
    ) -> None:
        if not self._compressor_sp_uses_boundary_replay(plan):
            return

        replay_x = self._select_compressor_sp_dim0(x, plan, "boundary_replay_token")
        self._assert_compressor_sp_token_layout(
            full_x=x,
            selected_x=replay_x,
            plan=plan,
            selector_name="boundary_replay_token",
        )
        replay_sin = self._select_compressor_sp_rope(
            compressed_sin,
            plan,
            replay_x.shape[0],
            compressed_row_selector="boundary_replay_compressed_row",
            req_selector="boundary_replay_req",
        )
        replay_cos = self._select_compressor_sp_rope(
            compressed_cos,
            plan,
            replay_x.shape[0],
            compressed_row_selector="boundary_replay_compressed_row",
            req_selector="boundary_replay_req",
        )
        replay_state_block_table = self._select_compressor_sp_dim0(state_block_table, plan, "boundary_replay_req")
        self._assert_compressor_sp_req_layout(
            full_state_block_table=state_block_table,
            selected_state_block_table=replay_state_block_table,
            plan=plan,
            selector_name="boundary_replay_req",
        )
        run_compressor_op(
            replay_x,
            wkv,
            wgate,
            state_cache,
            ape,
            norm_weight,
            replay_sin,
            replay_cos,
            state_block_table=replay_state_block_table,
            cu_seqlens=plan.boundary_replay_cu_seqlens,
            seqused=None,
            start_pos=plan.boundary_replay_start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=coff,
            norm_eps=self.compressor_norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )

    def _select_compressor_sp_rope(
        self,
        rope: torch.Tensor,
        plan: CompressorSPMetadata,
        num_tokens: int,
        *,
        compressed_row_selector: str = "compressed_row",
        req_selector: str = "req",
    ) -> torch.Tensor:
        rope_row_selector = "rope_row" if compressed_row_selector == "compressed_row" else "boundary_replay_rope_row"
        selected = self._select_compressor_sp_dim0(rope, plan, rope_row_selector)
        target_rows = min(
            num_tokens,
            num_tokens // self.compress_ratio + self._compressor_sp_selector_len(plan, req_selector),
        )
        if selected.shape[0] != target_rows:
            raise RuntimeError(
                "CompressorSP RoPE selector length mismatch: "
                f"{selected.shape[0]} != {target_rows}, path={plan.path}, "
                f"ratio={plan.ratio}, selector={rope_row_selector}"
            )
        self._assert_compressor_sp_rope_layout(
            rope=rope,
            selected_rope=selected,
            plan=plan,
            num_tokens=num_tokens,
            rope_row_selector=rope_row_selector,
            req_selector=req_selector,
        )
        return selected

    def _try_update_compressor_cache_sp(
        self,
        *,
        x: torch.Tensor,
        kv_cache: torch.Tensor,
        state_cache: torch.Tensor,
        attn_metadata: AscendDSAMetadata,
        state_metadata: AscendDSAMetadata,
        compressed_sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        coff: int,
        need_gather_q_kv: bool,
        has_prefill: bool,
        layer_name: str,
        full_slot_mapping: torch.Tensor,
    ) -> tuple[bool, str]:
        assert attn_metadata.req_metadata is not None
        assert state_metadata.req_metadata is not None
        plan = attn_metadata.req_metadata.compressor_sp
        fallback_reason = self._compressor_sp_unavailable_reason(
            plan,
            path="main",
            need_gather_q_kv=need_gather_q_kv,
            has_prefill=has_prefill,
            coff=coff,
        )
        if fallback_reason is not None:
            return False, fallback_reason

        timer_start = self._compressor_sp_debug_start()
        # full_slot_mapping is the compressor full slot mapping passed by forward
        slot_mapping = self._select_compressor_sp_dim0(full_slot_mapping, plan, "slot_mapping")
        self._assert_compressor_sp_slot_layout(
            full_slot_mapping=full_slot_mapping,
            selected_slot_mapping=slot_mapping,
            plan=plan,
        )
        expected_rows = self._compressor_sp_selector_len(plan, "output_keep")
        if slot_mapping.shape[0] != expected_rows:
            self._record_compressor_sp_debug(
                path="main",
                status="fallback_with_plan",
                reason="slot_mapping_mismatch",
                layer_name=layer_name,
                plan=plan,
                coff=coff,
                local_shape_call=True,
                elapsed_ms=self._compressor_sp_debug_elapsed_ms(timer_start),
            )
            return False, "slot_mapping_mismatch"

        state_cache_view = state_cache.squeeze(-2)
        if self._compressor_sp_should_dual_run():
            base_state_cache = state_cache_view.clone()
            local_state_cache = base_state_cache.clone()
            full_state_cache = base_state_cache.clone()

            local_compressed_kv = self._run_compressor_sp(
                x=x,
                plan=plan,
                wkv=self.compressor_wkv.weight,
                wgate=self.compressor_wgate.weight,
                state_cache=local_state_cache,
                ape=self.compressor_ape,
                norm_weight=self.compressor_norm.weight,
                compressed_sin=compressed_sin,
                compressed_cos=compressed_cos,
                state_block_table=state_metadata.req_metadata.block_table,
                coff=coff,
            )
            full_compressed_kv = run_compressor_op(
                x,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                full_state_cache,
                self.compressor_ape,
                self.compressor_norm.weight,
                compressed_sin.view(-1, compressed_sin.shape[-1]),
                compressed_cos.view(-1, compressed_cos.shape[-1]),
                state_block_table=state_metadata.req_metadata.block_table,
                cu_seqlens=attn_metadata.req_metadata.query_start_loc,
                seqused=None,
                start_pos=attn_metadata.req_metadata.start_pos,
                rope_head_dim=self.rope_head_dim,
                cmp_ratio=self.compress_ratio,
                coff=coff,
                norm_eps=self.compressor_norm_eps,
                rotary_mode=2,
                cache_mode=1,
            )

            full_rows = (
                full_compressed_kv.index_select(0, plan.local_keep_to_full_row_indices)
                if plan.local_keep_to_full_row_indices.numel() > 0
                else full_compressed_kv[:0]
            )
            row_reason = self._compressor_sp_compare_rows(
                label="rows",
                local_rows=local_compressed_kv,
                full_rows=full_rows,
                layer_name=layer_name,
                path="main",
                coff=coff,
                plan=plan,
            )
            if row_reason is not None:
                self._record_compressor_sp_debug(
                    path="main",
                    status="fallback_with_plan",
                    reason=row_reason,
                    layer_name=layer_name,
                    plan=plan,
                    coff=coff,
                    full_shape_call=True,
                    local_shape_call=True,
                    elapsed_ms=self._compressor_sp_debug_elapsed_ms(timer_start),
                )
                return False, row_reason

            state_rows = self._compressor_sp_state_rows(
                input_positions=attn_metadata.req_metadata.input_positions,
                state_block_table=state_metadata.req_metadata.block_table,
                plan=plan,
                state_block_size=state_cache_view.shape[1],
            )
            state_reason = self._compressor_sp_compare_state_rows(
                local_state=local_state_cache,
                full_state=full_state_cache,
                state_rows=state_rows,
                layer_name=layer_name,
                path="main",
                coff=coff,
                plan=plan,
            )
            if state_reason is not None:
                self._record_compressor_sp_debug(
                    path="main",
                    status="fallback_with_plan",
                    reason=state_reason,
                    layer_name=layer_name,
                    plan=plan,
                    coff=coff,
                    full_shape_call=True,
                    local_shape_call=True,
                    elapsed_ms=self._compressor_sp_debug_elapsed_ms(timer_start),
                )
                return False, state_reason

            # Compare only the cache rows touched by this request. Cloning the
            # full cache twice costs multiple GiB and can make debug dual-run
            # OOM even though production SP fits. Padding destinations are
            # invalid slots filtered by the scatter kernel, so the in-place
            # local scatter only mutates rows in full_slot_mapping.
            _du_padded = (
                ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_SCATTER_FUSION
                and plan.gather_compact_indices is not None
                and full_slot_mapping.ndim == 2
                and full_slot_mapping.shape[1] == 2
            )
            if _du_padded:
                global_local_compressed_kv, max_rows = self._gather_compressor_sp_buffer(local_compressed_kv, plan)
                if global_local_compressed_kv.numel() > 0:
                    padded_slot_mapping, self._psmb = build_padded_destination_for_scatter(
                        full_slot_mapping,
                        plan.gather_compact_indices,
                        plan.gather_compact_slice,
                        max_rows,
                        plan.tp_size,
                        int(self.block_size),
                        getattr(self, "_psmb", None),
                    )
                    DeviceOperator.dsa_kv_compress_scatter(kv_cache, global_local_compressed_kv, padded_slot_mapping)
            else:
                global_local_compressed_kv = self._gather_compressor_sp_rows(local_compressed_kv, plan)
                DeviceOperator.dsa_kv_compress_scatter(kv_cache, global_local_compressed_kv, full_slot_mapping)
            local_cache_rows = self._gather_cache_rows(kv_cache, full_slot_mapping).clone()
            DeviceOperator.dsa_kv_compress_scatter(kv_cache, full_compressed_kv, full_slot_mapping)
            full_cache_rows = self._gather_cache_rows(kv_cache, full_slot_mapping).clone()
            cache_reason = self._compressor_sp_compare_rows(
                label="cache",
                local_rows=local_cache_rows,
                full_rows=full_cache_rows,
                layer_name=layer_name,
                path="main",
                coff=coff,
                plan=plan,
            )
            if cache_reason is not None:
                self._record_compressor_sp_debug(
                    path="main",
                    status="fallback_with_plan",
                    reason=cache_reason,
                    layer_name=layer_name,
                    plan=plan,
                    coff=coff,
                    full_shape_call=True,
                    local_shape_call=True,
                    elapsed_ms=self._compressor_sp_debug_elapsed_ms(timer_start),
                )
                return False, cache_reason

            state_cache_view.copy_(local_state_cache)
            DeviceOperator.dsa_kv_compress_scatter(kv_cache, local_cache_rows, full_slot_mapping)
            self._record_compressor_sp_debug(
                path="main",
                status="local_hit",
                reason="enabled",
                layer_name=layer_name,
                plan=plan,
                coff=coff,
                full_shape_call=True,
                local_shape_call=True,
                elapsed_ms=self._compressor_sp_debug_elapsed_ms(timer_start),
            )
            return True, "enabled"

        compressed_kv = self._run_compressor_sp(
            x=x,
            plan=plan,
            wkv=self.compressor_wkv.weight,
            wgate=self.compressor_wgate.weight,
            state_cache=state_cache_view,
            ape=self.compressor_ape,
            norm_weight=self.compressor_norm.weight,
            compressed_sin=compressed_sin,
            compressed_cos=compressed_cos,
            state_block_table=state_metadata.req_metadata.block_table,
            coff=coff,
        )
        if compressed_kv is None:
            compressed_kv = state_cache_view[:0]
        if slot_mapping.shape[0] != compressed_kv.shape[0]:
            self._record_compressor_sp_debug(
                path="main",
                status="fallback_with_plan",
                reason="local_rows_keep_rows_mismatch",
                layer_name=layer_name,
                plan=plan,
                coff=coff,
                local_shape_call=True,
                elapsed_ms=self._compressor_sp_debug_elapsed_ms(timer_start),
            )
            return False, "local_rows_keep_rows_mismatch"

        # All-gather compressed rows from all TP ranks to reconstruct the
        # full global compressed KV, then scatter with the full slot_mapping
        # so every rank's local compress_kv_cache is a complete replica.
        # With SCATTER_FUSION and a ragged gather_compact selector, scatter the
        # padded buffer directly with a padded destination, skipping the
        # global_compressed_kv materialization.
        _use_padded_scatter = (
            ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_SCATTER_FUSION
            and plan.gather_compact_indices is not None
            and full_slot_mapping.ndim == 2
            and full_slot_mapping.shape[1] == 2
        )
        if _use_padded_scatter:
            padded_updates, max_rows = self._gather_compressor_sp_buffer(compressed_kv, plan)
            if padded_updates.numel() > 0:
                padded_slot_mapping, self._psmb = build_padded_destination_for_scatter(
                    full_slot_mapping,
                    plan.gather_compact_indices,
                    plan.gather_compact_slice,
                    max_rows,
                    plan.tp_size,
                    int(self.block_size),
                    getattr(self, "_psmb", None),
                )
                DeviceOperator.dsa_kv_compress_scatter(kv_cache, padded_updates, padded_slot_mapping)
        else:
            global_compressed_kv = self._gather_compressor_sp_rows(compressed_kv, plan)
            if global_compressed_kv.numel() > 0:
                DeviceOperator.dsa_kv_compress_scatter(kv_cache, global_compressed_kv, full_slot_mapping)
        self._record_compressor_sp_debug(
            path="main",
            status="local_hit",
            reason="enabled",
            layer_name=layer_name,
            plan=plan,
            coff=coff,
            local_shape_call=True,
            elapsed_ms=self._compressor_sp_debug_elapsed_ms(timer_start),
        )
        return True, "enabled"

    def forward(  # type: ignore[override]
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor],
        attn_metadata: list[M],
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        if not isinstance(attn_metadata, list):
            attn_metadata = [attn_metadata]
        wait_for_kv_layer_from_connector(layer_name)
        full_gather_wo_a_enabled = (
            self.tp_size > 1
            and self.enable_dsa_cp_with_o_proj_tp
            and attn_metadata[0].attn_state
            not in {
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            }
        )
        local_attn_output, o_proj_full_handles = self._forward(
            layer_name,
            hidden_states,
            kv_cache,
            attn_metadata,
            need_gather_q_kv,
            full_gather_wo_a_enabled,
        )
        o_proj_input = self._restore_tp_head_layout(
            local_attn_output,
            layer_name,
            attn_metadata[0],
            skip_all_to_all=full_gather_wo_a_enabled,
        )
        num_tokens = o_proj_input.shape[0]

        # o
        if full_gather_wo_a_enabled:
            self._switch_o_proj_to_full_weight(o_proj_full_handles)
        o_proj_groups = self.n_group if full_gather_wo_a_enabled else self.n_local_groups
        try:
            if get_ascend_device_type() in {AscendDeviceType.A5}:
                o = o_proj_input.view(num_tokens, o_proj_groups, -1)
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
                output[...] = self._apply_wo_b(o, full_gather_wo_a_enabled)
            else:
                o_proj_input = o_proj_input.view(num_tokens, o_proj_groups, -1)
                if olora_tp_enable():
                    o_proj_input = self.wo_a(o_proj_input)
                else:
                    # wo_a = self.wo_a.weight.view(o_proj_groups, self.o_lora_rank, -1)
                    # o = torch.einsum("tgd,grd->tgr", o, wo_a)
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
                output[...] = self._apply_wo_b(o_proj_input, full_gather_wo_a_enabled)
        finally:
            if full_gather_wo_a_enabled:
                self._switch_o_proj_to_tp_weight()

        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

        return output

    def _forward(
        self,
        layer_name,
        hidden_states_local: torch.Tensor,
        kv_cache: tuple,
        attn_metadata: list[M],
        need_gather_q_kv: bool = False,
        full_gather_wo_a_enabled: bool = False,
    ):
        """Run full-sequence KV cache updates and local-token attention."""
        (compress_kv_cache, swa_kv_cache, state_cache, _, _, _) = DeviceOperator.unpack_dsa_forward_kv_cache(
            kv_cache, self.compress_ratio
        )
        if self.compress_ratio == 4:
            (compressor_attn_metadata, compressor_kv_state_metadata, _, _, swa_metadata) = attn_metadata
        elif self.compress_ratio == 128:
            (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata
        else:
            (swa_metadata,) = attn_metadata
        common_attn_metadata = attn_metadata[0]

        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states_local, need_gather_q_kv)

        assert common_attn_metadata.req_metadata is not None
        assert swa_metadata.req_metadata is not None
        req_metadata = common_attn_metadata.req_metadata
        cp_metadata = req_metadata.cp_metadata
        cos = req_metadata.cos[layer_name]
        sin = req_metadata.sin[layer_name]
        local_cos = cp_metadata.local_cos[layer_name]
        local_sin = cp_metadata.local_sin[layer_name]
        actual_seq_lengths_query = req_metadata.query_start_loc
        local_seq_lengths_query = cp_metadata.local_query_start_loc
        local_seq_lengths_key = cp_metadata.local_seq_lens
        has_prefill = common_attn_metadata.num_prefills > 0
        swa_req_metadata = swa_metadata.req_metadata
        hidden_states_cache = hidden_states[: common_attn_metadata.num_actual_tokens]

        if (not isinstance(self.wq_b.quant_method, AscendUnquantizedLinearMethod)) and isinstance(
            self.wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod
        ):
            q_a = self.wq_a(hidden_states_local)
            qr_local, qr_pertoken_scale_local = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                q_a, self.q_norm.weight, epsilon=self.eps
            )
            if getattr(self.wq_b, "_chunk_size", 0):
                bias = self.wq_b.bias
                chunk_size = self.wq_b._chunk_size
                bias_1 = bias[:chunk_size] if bias is not None else None
                bias_2 = bias[chunk_size:] if bias is not None else None
                q = torch.cat(
                    [
                        torch_npu.npu_quant_matmul(
                            qr_local,
                            self.wq_b.weight_1,
                            self.wq_b.weight_1_scale,
                            pertoken_scale=qr_pertoken_scale_local,
                            bias=bias_1,
                            output_dtype=hidden_states_local.dtype,
                        ),
                        torch_npu.npu_quant_matmul(
                            qr_local,
                            self.wq_b.weight_2,
                            self.wq_b.weight_2_scale,
                            pertoken_scale=qr_pertoken_scale_local,
                            bias=bias_2,
                            output_dtype=hidden_states_local.dtype,
                        ),
                    ],
                    dim=-1,
                )
            else:
                q = torch_npu.npu_quant_matmul(
                    qr_local,
                    self.wq_b.weight,
                    self.wq_b.weight_scale,
                    pertoken_scale=qr_pertoken_scale_local,
                    bias=self.wq_b.bias,
                    output_dtype=hidden_states_local.dtype,
                )
        else:
            qr_local = self.q_norm(self.wq_a(hidden_states_local))
            q = self.wq_b(qr_local)
            qr_pertoken_scale_local = None

        q = q.unflatten(-1, (self.num_heads, self.head_dim))

        q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            local_cos,
            local_sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        o_proj_full_handles = self._maybe_all_gather_o_proj_full_weight(full_gather_wo_a_enabled)

        kv = self.wkv(hidden_states_cache)
        kv = self.kv_norm(kv)
        assert self.rope_head_dim is not None
        kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            cos[: kv.shape[0]],
            sin[: kv.shape[0]],
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )
        DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, swa_metadata.req_metadata.slot_mapping)

        compress_topk_idxs = None
        if self.compress_ratio > 1:
            assert compressor_attn_metadata.req_metadata is not None
            assert compressor_kv_state_metadata.req_metadata is not None
            if self.compress_ratio == 4:
                self._update_indexer_cache(
                    x=hidden_states_cache,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    actual_seq_lengths_query=actual_seq_lengths_query,
                )
                compress_topk_idxs = self._indexer_select_topk(
                    x=hidden_states_local,
                    qr=qr_local,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    cos=local_cos,
                    sin=local_sin,
                    actual_seq_lengths_query=local_seq_lengths_query,
                    actual_seq_lengths_key=local_seq_lengths_key,
                    qr_pertoken_scale=qr_pertoken_scale_local,
                )

            coff = 2 if self.compressor_overlap else 1
            compress_cos, compress_sin, compress_slot_mapping = self._compute_compressor_metadata(
                compressor_attn_metadata.req_metadata,
            )
            sp_ok, _sp_reason = self._try_update_compressor_cache_sp(
                x=hidden_states_cache,
                kv_cache=compress_kv_cache,
                state_cache=state_cache,
                attn_metadata=compressor_attn_metadata,
                state_metadata=compressor_kv_state_metadata,
                compressed_sin=compress_sin,
                compressed_cos=compress_cos,
                coff=coff,
                need_gather_q_kv=need_gather_q_kv,
                has_prefill=has_prefill,
                layer_name=layer_name,
                full_slot_mapping=compress_slot_mapping,
            )
            if not sp_ok:
                compressed_kv = run_compressor_op(
                    hidden_states_cache,
                    self.compressor_wkv.weight,
                    self.compressor_wgate.weight,
                    state_cache.squeeze(-2),
                    self.compressor_ape,
                    self.compressor_norm.weight,
                    compress_sin.view(-1, compress_sin.shape[-1]),
                    compress_cos.view(-1, compress_cos.shape[-1]),
                    state_block_table=compressor_kv_state_metadata.req_metadata.block_table,
                    cu_seqlens=actual_seq_lengths_query,
                    seqused=None,
                    start_pos=req_metadata.start_pos,
                    rope_head_dim=self.rope_head_dim,
                    cmp_ratio=self.compress_ratio,
                    coff=coff,
                    norm_eps=self.compressor_norm_eps,
                    rotary_mode=2,
                    cache_mode=1,
                )
                if compressed_kv.numel() == 0:
                    compressed_kv = None
                DeviceOperator.dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, compress_slot_mapping)

        notify_kv_cache_written(layer_name)
        record_attention_compute_start()
        attn_op = DeviceOperator.get_dsa_sparse_attn_op()
        extra_attn_kwargs: dict = DeviceOperator.get_dsa_sparse_attn_base_kwargs()
        if has_prefill:
            DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
                extra_attn_kwargs, cu_seqlens_ori_kv=local_seq_lengths_query
            )
        if swa_req_metadata.dspark_swa_indices is not None:
            extra_attn_kwargs["ori_sparse_indices"] = swa_req_metadata.dspark_swa_indices

        ori_win_left = self.window_size - 1 if swa_req_metadata.ori_win_left is None else swa_req_metadata.ori_win_left
        ori_win_right = 0 if swa_req_metadata.ori_win_right is None else swa_req_metadata.ori_win_right

        common_attn_kwargs = dict(
            cu_seqlens_q=local_seq_lengths_query,
            seqused_kv=local_seq_lengths_key,
            sinks=self.attn_sink,
            softmax_scale=self.softmax_scale,
            cmp_ratio=max(self.compress_ratio, 1),
            ori_mask_mode=4,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q="TND",
            layout_kv="PA_ND",
            **extra_attn_kwargs,
        )

        if self.compress_ratio <= 1:
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                ori_block_table=swa_metadata.req_metadata.block_table,
                metadata=swa_metadata.req_metadata.sas_metadata,
                **common_attn_kwargs,
            )[0]
        elif self.compress_ratio == 4:
            assert compressor_attn_metadata.req_metadata is not None
            DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
                common_attn_kwargs, cu_seqlens_cmp_kv=req_metadata.cu_cmp_seqlen_list
            )
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_metadata.req_metadata.block_table,
                cmp_block_table=compressor_attn_metadata.req_metadata.block_table,
                metadata=req_metadata.sas_metadata,
                cmp_mask_mode=3,
                **common_attn_kwargs,
            )[0]
        else:
            assert compressor_attn_metadata.req_metadata is not None
            DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
                common_attn_kwargs, cu_seqlens_cmp_kv=req_metadata.cu_cmp_seqlen_list
            )
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                ori_block_table=swa_metadata.req_metadata.block_table,
                cmp_block_table=compressor_attn_metadata.req_metadata.block_table,
                metadata=compressor_attn_metadata.req_metadata.sas_metadata,
                cmp_mask_mode=3,
                **common_attn_kwargs,
            )[0]
        return attn_output, o_proj_full_handles

    def _restore_tp_head_layout(
        self,
        local_attn_output: torch.Tensor,
        layer_name: str,
        attn_metadata: M,
        skip_all_to_all: bool = False,
    ) -> torch.Tensor:
        assert attn_metadata.req_metadata is not None
        req_metadata = attn_metadata.req_metadata
        cp_metadata = req_metadata.cp_metadata
        num_tokens = local_attn_output.shape[0]
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            local_attn_output.unsqueeze(1),
            cp_metadata.local_cos[layer_name],
            -cp_metadata.local_sin[layer_name],
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        if self.tp_size == 1 or skip_all_to_all:
            return local_attn_output

        send = (
            local_attn_output.view(num_tokens, self.tp_size, self.n_local_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(-1, self.n_local_heads, self.head_dim)
        )
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.tp_group.device_group)
        return recv

    def _update_indexer_cache(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: list[M],
        actual_seq_lengths_query: torch.Tensor,
    ) -> None:
        (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        )
        (_, _, indexer_kv_state_metadata, indexer_kv_scale_metadata, _) = attn_metadata
        coff = 2 if self.compressor_overlap else 1
        assert indexer_kv_scale_metadata is not None
        assert indexer_kv_state_metadata is not None
        assert indexer_kv_scale_metadata.req_metadata is not None
        assert indexer_kv_state_metadata.req_metadata is not None
        assert self.indexer is not None
        compressed_cos, compressed_sin, indexer_slot_mapping = self._compute_compressor_metadata(
            indexer_kv_scale_metadata.req_metadata,
        )
        kv = run_compressor_op(
            x,
            self.indexcom_wkv.weight,
            self.indexcom_wgate.weight,
            indexer_state_cache.squeeze(-2),
            self.indexcom_ape,
            self.indexcom_norm.weight,
            compressed_sin.view(-1, compressed_sin.shape[-1]),
            compressed_cos.view(-1, compressed_cos.shape[-1]),
            state_block_table=indexer_kv_state_metadata.req_metadata.block_table,
            cu_seqlens=actual_seq_lengths_query,
            seqused=None,
            start_pos=indexer_kv_scale_metadata.req_metadata.start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=coff,
            norm_eps=self.compressor_norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )

        if kv.numel() == 0:
            return
        if self.indexer.compressor.rotate:
            kv = rotate_activation(kv, indexer_kv_scale_metadata.hadamard)

        _, kv_scale = DeviceOperator.indexer_quant_scatter_part1(
            kv,
            indexer_k_cache,
            indexer_full_cache,
            indexer_slot_mapping,
        )
        if kv_scale is not None:
            DeviceOperator.dsa_indexer_scatter_scale_part3(
                kv_scale,
                indexer_scale_cache,
                indexer_slot_mapping,
            )

    def _indexer_select_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: list[M],
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        (_, indexer_k_cache, indexer_scale_cache, _) = DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        (_, _, _, indexer_kv_scale_metadata, _) = attn_metadata
        assert indexer_kv_scale_metadata is not None

        if (
            (not isinstance(self.inderxer_wq_b.quant_method, AscendUnquantizedLinearMethod))
            and isinstance(self.inderxer_wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod)
            and qr_pertoken_scale is not None
            and get_ascend_device_type() not in {AscendDeviceType.A5}
        ):
            q = torch_npu.npu_quant_matmul(
                qr,
                self.inderxer_wq_b.weight,
                self.inderxer_wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.inderxer_wq_b.bias,
                output_dtype=x.dtype,
            )
        else:
            q = self.inderxer_wq_b(qr)
        q = q.view(-1, self.indexer_heads, self.indexcom_head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.indexcom_head_dim - self.rope_head_dim, self.indexcom_head_dim],
        )
        q = rotate_activation(q, indexer_kv_scale_metadata.hadamard)
        weights = self.weights_proj(x) * (self.indexer_softmax_scale * self.indexer_heads**-0.5)

        q, q_scale = DeviceOperator.indexer_quantize_query(q)

        assert indexer_kv_scale_metadata.req_metadata is not None
        qli_metadata = indexer_kv_scale_metadata.req_metadata.qli_metadata
        block_table = indexer_kv_scale_metadata.req_metadata.block_table
        topk_idxs, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
            query=q,
            key=indexer_k_cache,
            weights=DeviceOperator.prepare_dsa_indexer_weights(weights),
            query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale(q_scale),
            key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(indexer_scale_cache),
            actual_seq_lengths_query=actual_seq_lengths_query[1:],
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=block_table,
            metadata=qli_metadata,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            return_value=False,
        )
        return topk_idxs

    def dsa_warmup_with_multistream(self, hidden_states: torch.Tensor):
        pass
