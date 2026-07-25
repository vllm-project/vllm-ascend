import math
from dataclasses import dataclass
from typing import ClassVar, TypeVar

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.v1.attention.backend import AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention.abstract import DSAAttentionImpl
from vllm_ascend.attention.attention_v1 import AscendAttentionState
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
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa, get_full_cos_and_sin_dsa
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (
    AscendDeviceType,
    enable_dsa_cp_with_o_proj_tp,
    get_ascend_device_type,
    olora_tp_enable,
)


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
        has_prefill = _has_prefill(attn_state)

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
            input_positions_cpu = common_attn_metadata.positions_cpu[:num_input_tokens].long()
            self.common_ratio_to_sas_metadata["input_positions"] = input_positions
            self.common_ratio_to_sas_metadata["input_positions_cpu"] = input_positions_cpu
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
            input_positions_cpu = self.common_ratio_to_sas_metadata["input_positions_cpu"]
            cos, sin = self.common_ratio_to_sas_metadata["cos"], self.common_ratio_to_sas_metadata["sin"]
            self.seq_lens = self.common_ratio_to_sas_metadata["seq_lens"]
            self.seq_lens_cpu = self.common_ratio_to_sas_metadata["seq_lens_cpu"]

        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        self.slot_mapping[:num_input_tokens] = DeviceOperator.format_dsa_slot_mapping(slot_mapping, self.block_size)

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
    ) -> AscendDSAReqMetadata:
        """Build DSA-CP metadata for one draft step."""
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
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
            local_query_start_loc=self.spec_local_query_start_loc[draft_index - 1],
            local_seq_lens=self.spec_local_seq_lens[draft_index - 1],
        )
        local_query_start_loc = local_query_start_loc.clone()
        local_seq_lens = local_seq_lens.clone()

        _, _, _, _, local_query_start_loc_cpu, local_seq_lens_cpu, _, _ = self._build_local_token_metadata(
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            input_positions=None,
            query_start_loc=query_start_loc_cpu,
            seq_lens=self.seq_lens_cpu[:num_reqs],
            use_cache=False,
        )
        local_seq_lens_q_cpu = local_query_start_loc_cpu[1 : num_reqs + 1] - local_query_start_loc_cpu[:num_reqs]
        max_local_query_len = max(1, int(local_seq_lens_q_cpu.max().item()))
        max_local_seq_lens = max(1, int(local_seq_lens_cpu.max().item()))

        start_pos = self.seq_lens[:num_reqs] - seq_lens_q

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
            ori_win_left=self.model_config.hf_config.sliding_window - 1,
            ori_win_right=0,
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
        )

    def _num_compressor_metadata_rows(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> int:
        assert self.num_actual_tokens is not None
        num_tokens = self.num_actual_tokens
        return min(num_tokens, num_tokens // self.compressor_ratio + common_attn_metadata.num_reqs)

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
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu

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

        _, _, _, _, local_query_start_loc_cpu, local_seq_lens_cpu, _, _ = self._build_local_token_metadata(
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            input_positions=None,
            query_start_loc=query_start_loc_cpu,
            seq_lens=self.seq_lens_cpu[:num_reqs],
            use_cache=False,
        )
        local_seq_lens_q_cpu = local_query_start_loc_cpu[1 : num_reqs + 1] - local_query_start_loc_cpu[:num_reqs]
        max_local_query_len = max(1, int(local_seq_lens_q_cpu.max().item()))
        max_local_seqlen = max(1, int(local_seq_lens_cpu.max().item()))

        # start_pos: context length before current query
        start_pos = self.seq_lens[:num_reqs] - seq_lens_q

        assert self.start_pos_prefill is not None
        self.start_pos_prefill.fill_(0)
        self.start_pos_prefill[:num_reqs] = start_pos

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

        # --- SAS metadata (all requests combined) ---
        num_heads = self.model_config.hf_config.num_attention_heads
        index_topk = self.model_config.hf_config.index_topk

        sas_metadata = self._build_sas_metadata(
            num_heads=num_heads,
            query_start_loc=local_query_start_loc,
            seq_lens=local_seq_lens,
            seq_lens_q=local_seq_lens_q,
            max_seqlen=max_local_seqlen,
            max_seqlen_q=max_local_query_len,
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
            max_seqlen=max_local_seqlen,
            max_seqlen_q=max_local_query_len,
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
        )

    def _build_local_token_metadata(
        self,
        num_reqs,
        num_input_tokens,
        input_positions,
        query_start_loc,
        seq_lens,
        use_cache,
        local_query_start_loc=None,
        local_seq_lens=None,
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
        if local_seq_lens is not None:
            local_seq_lens[:num_reqs] = (local_query_lens > 0) * (seq_lens - offset)
        else:
            local_seq_lens = (local_query_lens > 0) * (seq_lens - offset)

        # RoPE tables are generated on the padded global positions first, then
        # sliced to this rank so local tokens keep their original positions.
        if input_positions is not None:
            pad_tokens = num_tokens_pad - input_positions.shape[0]
            if pad_tokens > 0:
                input_positions = F.pad(input_positions, (0, pad_tokens), value=0)
            local_cos, local_sin = get_cos_and_sin_dsa(input_positions, use_cache=use_cache)
            local_cos = local_cos[local_start:local_end]
            local_sin = local_sin[local_start:local_end]
        else:
            local_cos = None
            local_sin = None
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
        max_seqlen,
        max_seqlen_q,
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
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen,
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

    def _build_qli_metadata(self, query_start_loc, seq_lens, seq_lens_q, max_seqlen, max_seqlen_q, num_reqs):
        if self.compressor_ratio != 4:
            return None

        cache_key = "cp_qli"
        metadata = self.common_ratio_to_sas_metadata.get(cache_key)

        if metadata is None:
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
                max_seqlen_k=max_seqlen,
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
        has_prefill = _has_prefill(common_attn_metadata.attn_state)
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
            compressed_kv = torch.ops._C_ascend.compressor(
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

        common_attn_kwargs = dict(
            cu_seqlens_q=local_seq_lengths_query,
            seqused_kv=local_seq_lengths_key,
            sinks=self.attn_sink,
            softmax_scale=self.softmax_scale,
            cmp_ratio=max(self.compress_ratio, 1),
            ori_mask_mode=4,
            ori_win_left=self.window_size - 1,
            ori_win_right=0,
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
        kv = torch.ops._C_ascend.compressor(
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
