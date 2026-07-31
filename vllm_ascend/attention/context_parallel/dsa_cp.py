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
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

from vllm_ascend import envs as ascend_envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.abstract import DSAAttentionImpl
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.compressor_sp import (
    CompressorSPPlan,
    build_compressor_sp_plan,
    collect_state_row_indices,
    run_compressor_op,
    sync_boundary_state_blocks,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, split_decodes_and_prefills
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (
    AscendDeviceType,
    attention_calculation_stream,
    get_ascend_device_type,
    npu_stream_switch,
    olora_tp_enable,
)

if HAS_TRITON:
    from vllm_ascend.ops.triton.rms_norm import triton_q_rms  # noqa: F811
else:
    triton_q_rms = None  # type: ignore


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


def _has_prefill(attn_state: AscendAttentionState) -> bool:
    return attn_state not in {
        AscendAttentionState.DecodeOnly,
        AscendAttentionState.SpecDecoding,
    }


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
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    cp_metadata: DSACPMetadata
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    compress_sin: torch.Tensor = None
    compress_cos: torch.Tensor = None
    start_pos: torch.Tensor = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None
    cu_cmp_seqlen_list: torch.Tensor = None
    attn_mask: torch.Tensor | None = None
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
    start_pos_prefill: torch.Tensor | None = None
    req_sas_metadata: torch.Tensor
    req_qli_metadata: torch.Tensor
    block_size: int | None = 128
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec: MLAAttentionSpec,
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

        self.compressor_ratio = getattr(kv_cache_spec, "compress_ratio", 0)
        self.compressor_sp_dual_run = ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_DUAL_RUN
        hf_config = self.model_config.hf_config

        if AscendDSACPMetadataBuilder.hadamard is None:
            if hf_config.model_type == "deepseek_v4":
                indexer_head_dim = hf_config.index_head_dim
                try:
                    from scipy.linalg import hadamard  # type: ignore[import-untyped]
                except ImportError as e:
                    raise ImportError("Please install scipy") from e
                log_dim = math.ceil(math.log2(indexer_head_dim))
                dim_padded = 2**log_dim
                AscendDSACPMetadataBuilder.hadamard = torch.tensor(
                    hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device
                ).to(torch.bfloat16)
        self.start_pos_prefill = torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)
        self.req_sas_metadata = torch.zeros(1024, dtype=torch.int32, device=self.device)
        self.req_qli_metadata = torch.zeros(1024, dtype=torch.int32, device=self.device)
        self.cu_seqlens_ori_kv = torch.tensor([], device=self.device)
        self.cu_seqlens_cmp_kv = torch.tensor([], device=self.device)
        self.seqused_q = torch.tensor([], device=self.device)
        self.local_query_start_loc = torch.zeros(
            scheduler_config.max_num_seqs + 1, dtype=torch.int32, device=self.device
        )
        self.local_seq_lens = torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)
        # Note(qcs): we use two dimension slot_mapping for kvcache
        # with shape [block_nums, block_size, head_num, head_dim]
        self.slot_mapping = torch.zeros(
            (vllm_config.scheduler_config.max_num_batched_tokens, 2), dtype=torch.int32, device=self.device
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        self.spec_slot_mapping = None
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.spec_slot_mapping = [
                torch.zeros(
                    (vllm_config.scheduler_config.max_num_batched_tokens, 2), dtype=torch.int32, device=self.device
                )
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

        common_ratio = kwargs.get("common_ratio_to_sas_metadata")
        if common_ratio is None:
            common_ratio = {}
        self.common_ratio_to_sas_metadata = common_ratio
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens
        attn_state = kwargs.get("attn_state", common_attn_metadata.attn_state)
        has_prefill = _has_prefill(attn_state)

        num_input_tokens = common_attn_metadata.num_input_tokens
        if self.common_ratio_to_sas_metadata.get("input_positions", None) is None:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
            )
            self.common_ratio_to_sas_metadata["num_decodes"] = self.num_decodes
            self.common_ratio_to_sas_metadata["num_prefills"] = self.num_prefills
            self.common_ratio_to_sas_metadata["num_decode_tokens"] = self.num_decode_tokens
            self.common_ratio_to_sas_metadata["num_prefill_tokens"] = self.num_prefill_tokens
            input_positions = common_attn_metadata.positions[:num_input_tokens].long()
            input_positions_cpu = common_attn_metadata.positions_cpu[:num_input_tokens].long()
            self.common_ratio_to_sas_metadata["input_positions"] = input_positions
            self.common_ratio_to_sas_metadata["input_positions_cpu"] = input_positions_cpu
            cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=not has_prefill)
            self.common_ratio_to_sas_metadata["cos"] = cos
            self.common_ratio_to_sas_metadata["sin"] = sin
            self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
            self.common_ratio_to_sas_metadata["seq_lens"] = self.seq_lens
        else:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                self.common_ratio_to_sas_metadata["num_decodes"],
                self.common_ratio_to_sas_metadata["num_prefills"],
                self.common_ratio_to_sas_metadata["num_decode_tokens"],
                self.common_ratio_to_sas_metadata["num_prefill_tokens"],
            )
            input_positions = self.common_ratio_to_sas_metadata["input_positions"]
            input_positions_cpu = self.common_ratio_to_sas_metadata["input_positions_cpu"]
            cos, sin = self.common_ratio_to_sas_metadata["cos"], self.common_ratio_to_sas_metadata["sin"]
            self.seq_lens = self.common_ratio_to_sas_metadata["seq_lens"]

        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        self.slot_mapping[:num_input_tokens] = torch.stack(
            [slot_mapping // self.block_size, slot_mapping % self.block_size], dim=-1
        )

        self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]

        req_metadata = self.build_req_metadata(
            common_attn_metadata, input_positions, input_positions_cpu, num_input_tokens, num_reqs_actual, attn_state
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
        draft_step: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        assert self.compressor_ratio <= 1, "vLLM-Ascend only support SWA-layer for Deepseek-V4 now."
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )

        self.num_decodes = num_decodes
        self.num_prefills = num_prefills
        self.num_decode_tokens = num_decode_tokens
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens
        self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        self.block_size = kwargs.get("block_size", 128)

        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        # Draft steps update positions independently. Reusing the global RoPE
        # cache can let later draft steps overwrite step-0 metadata.
        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=False)

        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]

        assert self.spec_slot_mapping is not None
        self.spec_slot_mapping[draft_step - 1][:num_input_tokens] = torch.stack(
            [slot_mapping // self.block_size, slot_mapping % self.block_size], dim=-1
        )

        self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        req_metadata = self.build_req_metadata_for_drafting(
            draft_step=draft_step,
            common_attn_metadata=common_attn_metadata,
            input_positions=input_positions,
            input_positions_cpu=common_attn_metadata.positions_cpu[:num_input_tokens].long(),
            num_input_tokens=num_input_tokens,
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
        draft_step: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        input_positions: torch.Tensor,
        input_positions_cpu: torch.Tensor,
        num_input_tokens: int,
    ) -> AscendDSAReqMetadata:
        """Build DSA-CP metadata for one draft step."""
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        has_prefill = _has_prefill(common_attn_metadata.attn_state)

        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=False)
        (
            local_start,
            local_end_with_pad,
            tokens_per_rank,
            num_tokens_pad,
            local_query_start_loc,
            local_seq_lens,
            local_cos,
            local_sin,
        ) = self._build_local_token_metadata(
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            input_positions=input_positions,
            query_start_loc=query_start_loc,
            seq_lens=self.seq_lens[:num_reqs],
            use_cache=False,
            local_query_start_loc=self.spec_local_query_start_loc[draft_step - 1],
            local_seq_lens=self.spec_local_seq_lens[draft_step - 1],
        )
        local_query_start_loc = local_query_start_loc.clone()
        local_seq_lens = local_seq_lens.clone()

        local_seq_lens_q = local_query_start_loc[1 : num_reqs + 1] - local_query_start_loc[:num_reqs]
        max_local_query_len = max(1, int(local_seq_lens_q.max().item()))
        max_local_seq_lens = max(1, int(local_seq_lens.max().item()))
        coff = 2 if getattr(self, "compressor_overlap", False) else 1

        start_pos = self.seq_lens[:num_reqs] - seq_lens_q

        assert self.spec_slot_mapping is not None
        slot_mapping = self.spec_slot_mapping[draft_step - 1][: self.num_actual_tokens]

        num_heads = self.model_config.hf_config.num_attention_heads
        sas_metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(
            num_heads_q=num_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=local_query_start_loc,
            cu_seqlens_ori_kv=local_query_start_loc if has_prefill else self.cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=None,
            seqused_q=self.seqused_q,
            seqused_kv=local_seq_lens,
            max_seqlen_q=max_local_query_len,
            max_seqlen_kv=max_local_seq_lens,
            batch_size=num_reqs,
            cmp_ratio=1,
            ori_mask_mode=4,
            ori_win_left=self.model_config.hf_config.sliding_window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
            device=str(self.seqused_q.device),
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
        compressor_sp = self._build_compressor_sp_metadata(
            common_attn_metadata=common_attn_metadata,
            input_positions_cpu=input_positions_cpu,
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
            seq_lens=self.seq_lens[:num_reqs],
            query_start_loc=query_start_loc,
            cp_metadata=cp_metadata,
            sin=sin,
            cos=cos,
            compress_sin=None,
            compress_cos=None,
            start_pos=start_pos,
            sas_metadata=sas_metadata,
            qli_metadata=None,
            cu_cmp_seqlen_list=None,
            compressor_sp=compressor_sp,
        )

    def build_req_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        input_positions: torch.Tensor,
        input_positions_cpu: torch.Tensor,
        num_input_tokens: int,
        num_reqs_actual: int | None,
        attn_state: AscendAttentionState,
    ) -> AscendDSAReqMetadata:
        """Build a single unified metadata for all requests (prefill + decode)."""
        num_reqs = common_attn_metadata.num_reqs
        has_prefill = _has_prefill(attn_state)
        query_start_loc = common_attn_metadata.query_start_loc

        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]

        # cos/sin for all tokens
        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=not has_prefill)

        (
            local_start,
            local_end_with_pad,
            tokens_per_rank,
            num_tokens_pad,
            local_query_start_loc,
            local_seq_lens,
            local_cos,
            local_sin,
        ) = self._build_local_token_metadata(
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            input_positions=input_positions,
            query_start_loc=query_start_loc,
            seq_lens=self.seq_lens[:num_reqs],
            use_cache=not has_prefill,
            local_query_start_loc=self.local_query_start_loc,
            local_seq_lens=self.local_seq_lens,
        )
        local_seq_lens_q = local_query_start_loc[1 : num_reqs + 1] - local_query_start_loc[:num_reqs]
        # TODO(qcs): remove this .item() to avoid D2H synchronization.
        max_local_query_len = max(1, int(local_seq_lens_q.max().item()))
        max_local_seq_lens = max(1, int(local_seq_lens.max().item()))
        coff = 2 if getattr(self, "compressor_overlap", False) else 1

        # start_pos: context length before current query
        start_pos = self.seq_lens[:num_reqs] - seq_lens_q

        assert self.start_pos_prefill is not None
        self.start_pos_prefill.fill_(0)
        self.start_pos_prefill[:num_reqs] = start_pos

        if num_reqs_actual is not None and num_reqs_actual < num_reqs:
            self.start_pos_prefill[num_reqs_actual:].fill_(0)
            self.block_table[num_reqs_actual:num_reqs, ...].fill_(0)

        # --- Compressed positions ---
        compress_cos, compress_sin = None, None
        cu_cmp_seqlens = self._get_cmp_seqlens_for_metadata(has_prefill)

        if self.compressor_ratio > 1:
            layer_name = f"c{self.compressor_ratio}"
            compressed_input_positions = self._get_padded_compressed_position(
                input_positions_cpu, self.compressor_ratio, num_reqs, num_input_tokens
            )
            compress_cos, compress_sin = get_cos_and_sin_dsa(
                {layer_name: compressed_input_positions}, use_cache=not has_prefill
            )

        slot_mapping_size = self._get_slot_mapping_size(input_positions_cpu, self.compressor_ratio)
        slot_mapping = self.slot_mapping[:slot_mapping_size]

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
        compressor_sp = self._build_compressor_sp_metadata(
            common_attn_metadata=common_attn_metadata,
            input_positions_cpu=input_positions_cpu,
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

        return AscendDSAReqMetadata(
            input_positions=input_positions,
            block_table=self.block_table[:num_reqs, ...],
            slot_mapping=slot_mapping,
            seq_lens=self.seq_lens[:num_reqs],
            query_start_loc=query_start_loc,
            cp_metadata=cp_metadata,
            sin=sin,
            cos=cos,
            compress_sin=compress_sin,
            compress_cos=compress_cos,
            start_pos=self.start_pos_prefill[:num_reqs],
            sas_metadata=sas_metadata,
            qli_metadata=qli_metadata,
            cu_cmp_seqlen_list=cu_cmp_seqlens,
            compressor_sp=compressor_sp,
        )

    def _build_local_token_metadata(
        self,
        num_reqs,
        num_input_tokens,
        input_positions,
        query_start_loc,
        seq_lens,
        use_cache,
        local_query_start_loc,
        local_seq_lens,
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

        local_query_start_loc.fill_(0)
        local_seq_lens.fill_(0)

        # Intersect each request's global token interval with this rank's local
        # token interval, then build the per-rank query_start_loc from lengths.
        local_query_start = torch.clamp(query_start_loc[:-1], min=local_start, max=local_end)
        local_query_end = torch.clamp(query_start_loc[1:], min=local_start, max=local_end)
        local_query_lens = local_query_end - local_query_start
        local_query_start_loc[1 : num_reqs + 1] = torch.cumsum(local_query_lens, dim=0)

        # For requests that cross the local slice boundary, offset removes the
        # tokens that live on later ranks so local_seq_lens matches local queries.
        offset = query_start_loc[1:] - local_query_end
        local_seq_lens[:num_reqs] = (local_query_lens > 0) * (seq_lens - offset)

        # RoPE tables are generated on the padded global positions first, then
        # sliced to this rank so local tokens keep their original positions.
        pad_tokens = num_tokens_pad - input_positions.shape[0]
        if pad_tokens > 0:
            input_positions = F.pad(input_positions, (0, pad_tokens), value=0)
        local_cos, local_sin = get_cos_and_sin_dsa(input_positions, use_cache=use_cache)
        local_cos = local_cos[local_start:local_end]
        local_sin = local_sin[local_start:local_end]
        return (
            local_start,
            local_end,
            tokens_per_rank,
            num_tokens_pad,
            local_query_start_loc[: num_reqs + 1],
            local_seq_lens[:num_reqs],
            local_cos,
            local_sin,
        )

    # --- helper: padded compressed positions ---
    def _get_padded_compressed_position(self, input_positions, compress_ratio, num_reqs, num_input_tokens):
        if compress_ratio <= 1:
            return input_positions
        mask = ((input_positions + 1) % compress_ratio) == 0
        pos = input_positions[mask]
        pos = (pos + 1) - compress_ratio
        target_shape = (min(num_input_tokens, num_input_tokens // compress_ratio + num_reqs),)
        pad_right = target_shape[0] - pos.shape[0]
        return F.pad(pos, (0, pad_right), value=0.0)

    def _get_cmp_seqlens_for_metadata(self, has_prefill):
        if self.compressor_ratio <= 1:
            return None
        if has_prefill:
            return None
        return self.cu_seqlens_cmp_kv

    def _get_slot_mapping_size(self, input_positions, compress_ratio):
        if compress_ratio <= 1:
            return self.num_actual_tokens
        mask = ((input_positions + 1) % compress_ratio) == 0
        return mask.sum()

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
            int(self.compressor_sp_dual_run),
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
                ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_C4_NONALIGNED or self.compressor_sp_dual_run
            ),
            allow_c128_non_aligned=(
                ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_C128_NONALIGNED or self.compressor_sp_dual_run
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
            boundary_replay_compressed_row_slice=(
                plan.boundary_replay_compressed_row_slice
            ),
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
            sp_row_counts_per_rank=plan.sp_row_counts_per_rank,
            tp_rank=plan.tp_rank,
            tp_size=plan.tp_size,
        )

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
            cu_seqlens_ori_kv = query_start_loc if has_prefill else self.cu_seqlens_ori_kv
            cu_seqlens_cmp_kv = None if has_prefill else self.cu_seqlens_cmp_kv
            kw = dict(
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
                device=str(self.seqused_q.device),
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

            metadata = torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata(**kw)
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
            metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_metadata(
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
                "Currently we only support building dummy metadata for DecodeOnly and SpecDecoding state"
            )

        assert attn_metadata is not None
        return attn_metadata


class AscendDSACPImpl(DSAAttentionImpl):
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
        self.kv_norm = kwargs["kv_norm"]

        self.indexer = kwargs.get("indexer")
        self.compressor = kwargs.get("compressor")

        self.wo_a = kwargs["wo_a"]
        self.wo_b = kwargs["wo_b"]

        self.eps = kwargs["eps"]

        self.attn_sink = kwargs["attn_sink"]

        ascend_config = get_ascend_config()
        self.multistream_dsa_preprocess = ascend_config.multistream_dsa_preprocess
        self.enable_compressor_sp = ascend_envs.VLLM_ASCEND_ENABLE_COMPRESSOR_SP
        self.compressor_sp_debug_interval = ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_DEBUG_INTERVAL
        self.compressor_sp_debug_sync = ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_DEBUG_SYNC
        self.compressor_sp_dual_run = ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_DUAL_RUN
        self.compressor_sp_allow_c128_non_aligned = ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_C128_NONALIGNED
        self.compressor_sp_allow_c4_non_aligned = ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_C4_NONALIGNED
        self._compressor_sp_debug_events = 0
        self._compressor_sp_debug_stats: dict[tuple[str, str, int, int, str, str, str, str], dict[str, object]] = {}

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
        self.compressor_overlap = False
        if self.compressor is not None:
            self.compressor_head_dim = self.compressor.head_dim
            self.compressor_overlap = self.compressor.overlap
            self.compressor_rotate = self.compressor.rotate

            self.compressor_ape = self.compressor.ape
            self.compressor_wkv = self.compressor.wkv
            self.compressor_wgate = self.compressor.wgate
            self.compressor_norm = self.compressor.norm
            self.compressor_norm_eps = self.compressor.norm_eps

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if self.attn_sink.numel() != self.num_heads:
            raise RuntimeError(
                "DSA-CP expects full-head attn_sink loaded on every TP rank, "
                f"got {self.attn_sink.numel()} heads, expected {self.num_heads}."
            )

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
        local_attn_output = self._forward(layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv)
        o_proj_input = self._restore_tp_head_layout(local_attn_output, layer_name, attn_metadata[0])
        num_tokens = o_proj_input.shape[0]

        # o
        o_proj_input = o_proj_input.view(num_tokens, self.n_local_groups, -1)
        if olora_tp_enable():
            o_proj_tmp = self.wo_a(o_proj_input)
        else:
            # wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            # o = torch.einsum("tgd,grd->tgr", o, wo_a)
            o_proj_tmp = torch_npu.npu_transpose_batchmatmul(
                o_proj_input,
                self.wo_a.weight,
                bias=None,
                scale=None,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
                batch_split_factor=1,
            ).view(num_tokens, -1)
        output[...] = self.wo_b(o_proj_tmp)

        return output

    def _forward(
        self,
        layer_name,
        hidden_states_local: torch.Tensor,
        kv_cache: tuple,
        attn_metadata: list[M],
        need_gather_q_kv: bool = False,
    ):
        """Run full-sequence KV cache updates and local-token attention."""
        if self.compress_ratio == 4:
            (compress_kv_cache, swa_kv_cache, state_cache, _, _, _) = kv_cache
            (compressor_attn_metadata, compressor_kv_state_metadata, _, _, swa_metadata) = attn_metadata
        elif self.compress_ratio == 128:
            (compress_kv_cache, swa_kv_cache, state_cache, _, _, _) = kv_cache
            (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata
        else:
            (_, swa_kv_cache, _, _, _, _) = kv_cache
            (swa_metadata,) = attn_metadata
        common_attn_metadata = attn_metadata[0]

        overlap_hidden_states_allgather = self.multistream_dsa_preprocess and need_gather_q_kv
        wait_hidden_states_local_event = (
            torch.npu.current_stream().record_event() if overlap_hidden_states_allgather else None
        )
        with npu_stream_switch(attention_calculation_stream(), enabled=overlap_hidden_states_allgather):
            if wait_hidden_states_local_event:
                torch.npu.current_stream().wait_event(wait_hidden_states_local_event)
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states_local, need_gather_q_kv)
            wait_hidden_states_allgather_event = (
                torch.npu.current_stream().record_event() if overlap_hidden_states_allgather else None
            )

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
        has_prefill = _has_prefill(common_attn_metadata.attn_state)

        if (not isinstance(self.wq_b.quant_method, AscendUnquantizedLinearMethod)) and isinstance(
            self.wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod
        ):
            q_a = self.wq_a(hidden_states_local)
            qr_local, qr_pertoken_scale_local = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                q_a, self.q_norm.weight, epsilon=self.eps
            )
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

        q = triton_q_rms(q, self.eps)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            local_cos,
            local_sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        if wait_hidden_states_allgather_event:
            torch.npu.current_stream().wait_event(wait_hidden_states_allgather_event)

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
        torch.ops._C_ascend.npu_scatter_nd_update_v2(swa_kv_cache, swa_metadata.req_metadata.slot_mapping, kv)

        compress_topk_idxs = None
        if self.compress_ratio > 1:
            assert compressor_attn_metadata.req_metadata is not None
            assert compressor_kv_state_metadata.req_metadata is not None
            compress_cos = req_metadata.compress_cos[layer_name]
            compress_sin = req_metadata.compress_sin[layer_name]
            if self.compress_ratio == 4:
                self._update_indexer_cache(
                    x=hidden_states,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    compressed_cos=compress_cos,
                    compressed_sin=compress_sin,
                    actual_seq_lengths_query=actual_seq_lengths_query,
                    need_gather_q_kv=need_gather_q_kv,
                    has_prefill=has_prefill,
                    layer_name=layer_name,
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

            coff = 2 if getattr(self, "compressor_overlap", False) else 1
            compressor_sp_done, compressor_sp_reason = self._try_update_compressor_cache_sp(
                x=hidden_states,
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
            )
            if not compressor_sp_done:
                compressed_kv = run_compressor_op(
                    hidden_states,
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
                self._record_compressor_sp_debug(
                    path="main",
                    status=self._compressor_sp_status_for_reason(compressor_sp_reason),
                    reason=compressor_sp_reason,
                    layer_name=layer_name,
                    plan=compressor_attn_metadata.req_metadata.compressor_sp,
                    coff=coff,
                    full_shape_call=True,
                )

                if compressed_kv.numel() == 0:
                    compressed_kv = None
                torch.ops._C_ascend.npu_scatter_nd_update_v2(
                    compress_kv_cache, compressor_attn_metadata.req_metadata.slot_mapping, compressed_kv
                )

            self._sync_compressor_sp_boundary_state(
                state_cache=state_cache.squeeze(-2),
                state_block_table=compressor_kv_state_metadata.req_metadata.block_table,
                plan=compressor_attn_metadata.req_metadata.compressor_sp,
            )

        common_attn_kwargs = dict(
            cu_seqlens_q=local_seq_lengths_query,
            seqused_kv=local_seq_lengths_key,
            sinks=self.attn_sink,
            softmax_scale=self.softmax_scale,
            cmp_ratio=self.compress_ratio,
            ori_mask_mode=4,
            ori_win_left=self.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
        )
        if has_prefill:
            common_attn_kwargs["cu_seqlens_ori_kv"] = local_seq_lengths_query

        if self.compress_ratio <= 1:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_kv_cache,
                ori_block_table=swa_metadata.req_metadata.block_table,
                metadata=swa_metadata.req_metadata.sas_metadata,
                **common_attn_kwargs,
            )[0]
        elif self.compress_ratio == 4:
            assert compressor_attn_metadata.req_metadata is not None
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_metadata.req_metadata.block_table,
                cmp_block_table=compressor_attn_metadata.req_metadata.block_table,
                cu_seqlens_cmp_kv=req_metadata.cu_cmp_seqlen_list,
                metadata=req_metadata.sas_metadata,
                cmp_mask_mode=3,
                **common_attn_kwargs,
            )[0]
        else:
            assert compressor_attn_metadata.req_metadata is not None
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                ori_block_table=swa_metadata.req_metadata.block_table,
                cmp_block_table=compressor_attn_metadata.req_metadata.block_table,
                cu_seqlens_cmp_kv=req_metadata.cu_cmp_seqlen_list,
                metadata=compressor_attn_metadata.req_metadata.sas_metadata,
                cmp_mask_mode=3,
                **common_attn_kwargs,
            )[0]
        return attn_output

    def _restore_tp_head_layout(
        self,
        local_attn_output: torch.Tensor,
        layer_name: str,
        attn_metadata: M,
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

        if self.tp_size == 1:
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

    def _gather_compressor_sp_rows(
        self,
        local_rows: torch.Tensor,
        plan: CompressorSPMetadata,
    ) -> torch.Tensor:
        """All-gather compressed rows from all TP ranks and concatenate in
        rank order to reconstruct the full global compressed rows.

        Uses padded fixed-shape all_gather for HCCL compatibility: each rank
        pads its local rows to max_rows (the maximum across all ranks), the
        all_gather collects fixed-shape tensors, then each rank's contribution
        is sliced to its true count and concatenated.
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

        # Pad local rows to [max_rows, head_dim]
        if local_rows.shape[0] < max_rows:
            pad = local_rows.new_zeros(max_rows - local_rows.shape[0], head_dim)
            padded_local = torch.cat([local_rows, pad], dim=0)
        else:
            padded_local = local_rows

        # All-gather: each rank contributes [max_rows, head_dim]
        gather_list = [
            torch.empty_like(padded_local) for _ in range(tp_size)
        ]
        dist.all_gather(
            gather_list, padded_local, group=self.tp_group.device_group
        )

        # Slice each rank's contribution to its true count and concatenate
        parts = []
        for rank_idx in range(tp_size):
            count = sp_row_counts[rank_idx]
            if count > 0:
                parts.append(gather_list[rank_idx][:count])

        if not parts:
            return local_rows[:0]
        global_rows = torch.cat(parts, dim=0)
        assert global_rows.shape[0] == total_rows, (
            f"CompressorSP gather: expected {total_rows} global rows, "
            f"got {global_rows.shape[0]}"
        )
        return global_rows

    def _compressor_sp_uses_boundary_replay(
        self, plan: CompressorSPMetadata | None
    ) -> bool:
        return bool(
            plan is not None
            and plan.enabled
            and plan.ratio == 4
            and plan.is_chunked_prefill
            and plan.requires_boundary_state_sync
            and plan.supports_boundary_state_replay
            and (
                ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_REPLAY_C4_BOUNDARY_STATE
                or self.compressor_sp_dual_run
            )
        )

    def _compressor_sp_boundary_replay_metadata_complete(
        self, plan: CompressorSPMetadata
    ) -> bool:
        return bool(
            self._has_compressor_sp_selector(plan, "boundary_replay_token")
            and self._has_compressor_sp_selector(plan, "boundary_replay_req")
            and self._has_compressor_sp_selector(
                plan, "boundary_replay_compressed_row"
            )
            and plan.boundary_replay_cu_seqlens is not None
            and plan.boundary_replay_start_pos is not None
            and self._compressor_sp_selector_len(
                plan, "boundary_replay_token"
            )
            > 0
            and self._compressor_sp_selector_len(plan, "boundary_replay_req")
            > 0
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
        if plan.is_chunked_prefill and plan.requires_boundary_state_sync:
            if self._compressor_sp_uses_boundary_replay(plan):
                if not self._compressor_sp_boundary_replay_metadata_complete(
                    plan
                ):
                    return "missing_boundary_replay_metadata"
            elif not ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_SYNC_CHUNKED_BOUNDARY_STATE:
                return "missing_chunked_boundary_state_sync"
        if (
            self._compressor_sp_selector_len(plan, "token") == 0
            and not self._compressor_sp_uses_boundary_replay(plan)
        ):
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
        for name in ("token", "req", "compressed_row", "output_keep", "slot_mapping"):
            if not self._has_compressor_sp_selector(plan, name):
                return "missing_metadata"
        if plan.cu_seqlens is None or plan.start_pos is None:
            return "missing_metadata"
        if plan.local_keep_to_full_row_indices is None or plan.local_keep_to_slot_row_indices is None:
            return "missing_metadata"
        if self._compressor_sp_selector_len(plan, "output_keep") != int(plan.local_keep_to_full_row_indices.numel()):
            return "local_rows_keep_rows_mismatch"
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
        row_ids = set(collect_state_row_indices(
            token_positions=token_positions,
            req_block_table=req_block_table,
            cu_seqlens=plan.cu_seqlens,
            state_block_size=state_block_size,
        ))
        if self._compressor_sp_uses_boundary_replay(plan):
            replay_positions = self._select_compressor_sp_dim0(
                input_positions, plan, "boundary_replay_token"
            )
            replay_block_table = self._select_compressor_sp_dim0(
                state_block_table, plan, "boundary_replay_req"
            )
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
        return torch.tensor(
            sorted(row_ids), dtype=torch.long, device=token_positions.device
        )

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
            or not (
                ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_ALLOW_CHUNKED_PREFILL
                or self.compressor_sp_dual_run
            )
            or not ascend_envs.VLLM_ASCEND_COMPRESSOR_SP_SYNC_CHUNKED_BOUNDARY_STATE
        ):
            return
        if (
            plan.boundary_req_indices is None
            or plan.boundary_positions is None
            or plan.boundary_owner_mask is None
        ):
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
        return self._compressor_sp_unavailable_reason(
            plan, path="main", need_gather_q_kv=need_gather_q_kv, has_prefill=has_prefill, coff=coff
        ) is None

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
        full_compressed_sin = compressed_sin.view(
            -1, compressed_sin.shape[-1]
        )
        full_compressed_cos = compressed_cos.view(
            -1, compressed_cos.shape[-1]
        )
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
        compressed_sin = self._select_compressor_sp_rope(
            full_compressed_sin, plan, x.shape[0]
        )
        compressed_cos = self._select_compressor_sp_rope(
            full_compressed_cos, plan, x.shape[0]
        )
        full_state_block_table = state_block_table
        state_block_table = self._select_compressor_sp_dim0(
            full_state_block_table, plan, "req"
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
            return compressed_kv[:0]
        return self._select_compressor_sp_dim0(compressed_kv, plan, "output_keep")

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

        replay_x = self._select_compressor_sp_dim0(
            x, plan, "boundary_replay_token"
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
        replay_state_block_table = self._select_compressor_sp_dim0(
            state_block_table, plan, "boundary_replay_req"
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
        selected = self._select_compressor_sp_dim0(
            rope, plan, compressed_row_selector
        )
        target_rows = min(
            num_tokens,
            num_tokens // self.compress_ratio
            + self._compressor_sp_selector_len(plan, req_selector),
        )
        pad_rows = target_rows - selected.shape[0]
        if pad_rows <= 0:
            return selected
        # Full-path metadata pads compressed RoPE positions with position 0.
        padding = rope[:1].expand(pad_rows, -1)
        return torch.cat((selected, padding), dim=0)

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
        slot_mapping = self._select_compressor_sp_dim0(attn_metadata.req_metadata.slot_mapping, plan, "slot_mapping")
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

            local_cache = kv_cache.clone()
            full_cache = kv_cache.clone()
            if local_compressed_kv.numel() > 0:
                # Gather local SP rows to reconstruct global compressed KV,
                # then compare against the full compressor's global output and
                # scatter with the full slot_mapping for complete cache replica.
                global_local_compressed_kv = self._gather_compressor_sp_rows(
                    local_compressed_kv, plan
                )
                full_slot_mapping = attn_metadata.req_metadata.slot_mapping
                torch.ops._C_ascend.npu_scatter_nd_update_v2(
                    local_cache, full_slot_mapping, global_local_compressed_kv
                )
                torch.ops._C_ascend.npu_scatter_nd_update_v2(
                    full_cache, full_slot_mapping, full_compressed_kv
                )
            cache_reason = self._compressor_sp_compare_cache_rows(
                local_cache=local_cache,
                full_cache=full_cache,
                slot_mapping=attn_metadata.req_metadata.slot_mapping,
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
            kv_cache.copy_(local_cache)
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
        global_compressed_kv = self._gather_compressor_sp_rows(compressed_kv, plan)
        full_slot_mapping = attn_metadata.req_metadata.slot_mapping
        if global_compressed_kv.numel() > 0:
            torch.ops._C_ascend.npu_scatter_nd_update_v2(
                kv_cache, full_slot_mapping, global_compressed_kv
            )
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

    def _update_indexer_cache(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: list[M],
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        need_gather_q_kv: bool = False,
        has_prefill: bool = False,
        layer_name: str = "",
    ) -> None:
        (_, _, _, indexer_state_cache, indexer_k_cache, indexer_scale_cache) = kv_cache
        (_, _, indexer_kv_state_metadata, indexer_kv_scale_metadata, _) = attn_metadata
        coff = 2 if getattr(self, "compressor_overlap", False) else 1
        assert indexer_kv_scale_metadata is not None
        assert indexer_kv_state_metadata is not None
        assert indexer_kv_scale_metadata.req_metadata is not None
        assert indexer_kv_state_metadata.req_metadata is not None
        assert self.indexer is not None
        plan = indexer_kv_scale_metadata.req_metadata.compressor_sp
        full_slot_mapping = indexer_kv_scale_metadata.req_metadata.slot_mapping
        slot_mapping = full_slot_mapping
        state_cache_view = indexer_state_cache.squeeze(-2)
        kv = None
        timer_start = None
        fallback_reason = self._compressor_sp_unavailable_reason(
            plan,
            path="indexer",
            need_gather_q_kv=need_gather_q_kv,
            has_prefill=has_prefill,
            coff=coff,
        )
        if fallback_reason is None:
            timer_start = self._compressor_sp_debug_start()
            slot_mapping = self._select_compressor_sp_dim0(slot_mapping, plan, "slot_mapping")
            expected_rows = self._compressor_sp_selector_len(plan, "output_keep")
            if slot_mapping.shape[0] != expected_rows:
                fallback_reason = "slot_mapping_mismatch"
                used_local_sp = False
            elif self._compressor_sp_should_dual_run():
                base_state_cache = state_cache_view.clone()
                local_state_cache = base_state_cache.clone()
                full_state_cache = base_state_cache.clone()
                local_kv = self._run_compressor_sp(
                    x=x,
                    plan=plan,
                    wkv=self.indexcom_wkv.weight,
                    wgate=self.indexcom_wgate.weight,
                    state_cache=local_state_cache,
                    ape=self.indexcom_ape,
                    norm_weight=self.indexcom_norm.weight,
                    compressed_sin=compressed_sin,
                    compressed_cos=compressed_cos,
                    state_block_table=indexer_kv_state_metadata.req_metadata.block_table,
                    coff=coff,
                )
                local_kv_unrotated = local_kv
                full_kv = run_compressor_op(
                    x,
                    self.indexcom_wkv.weight,
                    self.indexcom_wgate.weight,
                    full_state_cache,
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
                if self.indexcom_rotate:
                    local_kv = rotate_activation(local_kv, indexer_kv_scale_metadata.hadamard)
                    full_kv = rotate_activation(full_kv, indexer_kv_scale_metadata.hadamard)
                full_rows = (
                    full_kv.index_select(0, plan.local_keep_to_full_row_indices)
                    if plan.local_keep_to_full_row_indices.numel() > 0
                    else full_kv[:0]
                )
                row_reason = self._compressor_sp_compare_rows(
                    label="rows",
                    local_rows=local_kv,
                    full_rows=full_rows,
                    layer_name=layer_name,
                    path="indexer",
                    coff=coff,
                    plan=plan,
                )
                if row_reason is None:
                    state_rows = self._compressor_sp_state_rows(
                        input_positions=indexer_kv_state_metadata.req_metadata.input_positions,
                        state_block_table=indexer_kv_state_metadata.req_metadata.block_table,
                        plan=plan,
                        state_block_size=local_state_cache.shape[1],
                    )
                    state_reason = self._compressor_sp_compare_state_rows(
                        local_state=local_state_cache,
                        full_state=full_state_cache,
                        state_rows=state_rows,
                        layer_name=layer_name,
                        path="indexer",
                        coff=coff,
                        plan=plan,
                    )
                else:
                    state_reason = row_reason

                if state_reason is None:
                    soc_version = get_ascend_device_type()
                    dst_type = torch.float8_e4m3fn if soc_version in {AscendDeviceType.A5} else torch.int8
                    local_kv_q, local_kv_scale = torch_npu.npu_dynamic_quant(local_kv, dst_type=dst_type)
                    full_kv_q, full_kv_scale = torch_npu.npu_dynamic_quant(full_rows, dst_type=dst_type)
                    local_kv_scale = local_kv_scale.unsqueeze(-1)
                    full_kv_scale = full_kv_scale.unsqueeze(-1)
                    if soc_version not in {AscendDeviceType.A5}:
                        local_kv_scale = local_kv_scale.to(torch.float16).unsqueeze(-1)
                        full_kv_scale = full_kv_scale.to(torch.float16).unsqueeze(-1)
                    local_k_cache = indexer_k_cache.clone()
                    local_scale_cache = indexer_scale_cache.clone()
                    full_k_cache = indexer_k_cache.clone()
                    full_scale_cache = indexer_scale_cache.clone()
                    if local_kv_q.numel() > 0:
                        torch.ops._C_ascend.npu_scatter_nd_update_v2(local_k_cache, slot_mapping, local_kv_q)
                        torch.ops._C_ascend.npu_scatter_nd_update_v2(local_scale_cache, slot_mapping, local_kv_scale)
                    if full_kv_q.numel() > 0:
                        torch.ops._C_ascend.npu_scatter_nd_update_v2(full_k_cache, slot_mapping, full_kv_q)
                        torch.ops._C_ascend.npu_scatter_nd_update_v2(full_scale_cache, slot_mapping, full_kv_scale)
                    cache_reason = self._compressor_sp_compare_cache_rows(
                        local_cache=local_k_cache,
                        full_cache=full_k_cache,
                        slot_mapping=slot_mapping,
                        layer_name=layer_name,
                        path="indexer",
                        coff=coff,
                        plan=plan,
                    )
                    if cache_reason is None:
                        scale_reason = self._compressor_sp_compare_cache_rows(
                            local_cache=local_scale_cache,
                            full_cache=full_scale_cache,
                            slot_mapping=slot_mapping,
                            layer_name=layer_name,
                            path="indexer",
                            coff=coff,
                            plan=plan,
                        )
                    else:
                        scale_reason = cache_reason

                    if scale_reason is None:
                        state_cache_view.copy_(local_state_cache)
                        # The common update below applies the configured
                        # rotation once before quantization and cache write.
                        kv = local_kv_unrotated
                        used_local_sp = True
                    else:
                        fallback_reason = scale_reason
                        used_local_sp = False
                        kv = None
                else:
                    fallback_reason = state_reason
                    used_local_sp = False
                    kv = None
            else:
                kv = self._run_compressor_sp(
                    x=x,
                    plan=plan,
                    wkv=self.indexcom_wkv.weight,
                    wgate=self.indexcom_wgate.weight,
                    state_cache=state_cache_view,
                    ape=self.indexcom_ape,
                    norm_weight=self.indexcom_norm.weight,
                    compressed_sin=compressed_sin,
                    compressed_cos=compressed_cos,
                    state_block_table=indexer_kv_state_metadata.req_metadata.block_table,
                    coff=coff,
                )
                if kv is not None:
                    # Gather BF16 compressed rows from all ranks before
                    # rotation and quantization so every rank gets a complete
                    # and consistent view.
                    kv = self._gather_compressor_sp_rows(kv, plan)
                    slot_mapping = full_slot_mapping
                    used_local_sp = True
                else:
                    used_local_sp = False
        else:
            used_local_sp = False

        if kv is None:
            if fallback_reason is None:
                fallback_reason = "empty_local_output"
                self._record_compressor_sp_debug(
                    path="indexer",
                    status="fallback_with_plan",
                    reason=fallback_reason,
                    layer_name=layer_name,
                    plan=plan,
                    coff=coff,
                    local_shape_call=True,
                    elapsed_ms=self._compressor_sp_debug_elapsed_ms(timer_start),
                )
            slot_mapping = full_slot_mapping
            kv = run_compressor_op(
                x,
                self.indexcom_wkv.weight,
                self.indexcom_wgate.weight,
                state_cache_view,
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
            self._record_compressor_sp_debug(
                path="indexer",
                status=self._compressor_sp_status_for_reason(fallback_reason),
                reason=fallback_reason,
                layer_name=layer_name,
                plan=plan,
                coff=coff,
                full_shape_call=True,
            )

        self._sync_compressor_sp_boundary_state(
            state_cache=state_cache_view,
            state_block_table=indexer_kv_state_metadata.req_metadata.block_table,
            plan=plan,
        )

        if kv.numel() == 0:
            if used_local_sp:
                self._record_compressor_sp_debug(
                    path="indexer",
                    status="local_hit",
                    reason="enabled",
                    layer_name=layer_name,
                    plan=plan,
                    coff=coff,
                    local_shape_call=True,
                    elapsed_ms=self._compressor_sp_debug_elapsed_ms(timer_start),
                )
            return
        if self.indexer.compressor.rotate:
            kv = rotate_activation(kv, indexer_kv_scale_metadata.hadamard)

        soc_version = get_ascend_device_type()
        dst_type = torch.float8_e4m3fn if soc_version in {AscendDeviceType.A5} else torch.int8
        kv, kv_scale = torch_npu.npu_dynamic_quant(kv, dst_type=dst_type)
        kv_scale = kv_scale.unsqueeze(-1)
        if soc_version not in {AscendDeviceType.A5}:
            kv_scale = kv_scale.to(torch.float16).unsqueeze(-1)

        torch.ops._C_ascend.npu_scatter_nd_update_v2(indexer_k_cache, slot_mapping, kv)
        torch.ops._C_ascend.npu_scatter_nd_update_v2(indexer_scale_cache, slot_mapping, kv_scale)
        if used_local_sp:
            self._record_compressor_sp_debug(
                path="indexer",
                status="local_hit",
                reason="enabled",
                layer_name=layer_name,
                plan=plan,
                coff=coff,
                local_shape_call=True,
                elapsed_ms=self._compressor_sp_debug_elapsed_ms(timer_start),
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
        (_, _, _, _, indexer_k_cache, indexer_scale_cache) = kv_cache
        (_, _, _, indexer_kv_scale_metadata, _) = attn_metadata
        assert indexer_kv_scale_metadata is not None

        if (
            (not isinstance(self.inderxer_wq_b.quant_method, AscendUnquantizedLinearMethod))
            and isinstance(self.inderxer_wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod)
            and qr_pertoken_scale is not None
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

        soc_version = get_ascend_device_type()
        dst_type = torch.float8_e4m3fn if soc_version in {AscendDeviceType.A5} else torch.int8
        q, q_scale = torch_npu.npu_dynamic_quant(q, dst_type=dst_type)
        if soc_version not in {AscendDeviceType.A5}:
            q_scale = q_scale.to(torch.float16)

        assert indexer_kv_scale_metadata.req_metadata is not None
        qli_metadata = indexer_kv_scale_metadata.req_metadata.qli_metadata
        block_table = indexer_kv_scale_metadata.req_metadata.block_table
        topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer(
            query=q,
            key=indexer_k_cache,
            weights=weights.to(torch.float16),
            query_dequant_scale=q_scale,
            key_dequant_scale=indexer_scale_cache.squeeze(-2),
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
