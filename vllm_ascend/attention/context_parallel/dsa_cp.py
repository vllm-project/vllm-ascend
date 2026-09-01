import math
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

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.abstract import DSAAttentionImpl
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_compressor import (
    CompressorExecutor,
    CompressorSPMetadata,
    CompressorSPMetadataBuilder,
    CompressorSPPending,
    IndexerCompressorExecutor,
    rotate_activation,
)
from vllm_ascend.attention.dsa_v1 import (
    build_dspark_swa_indices,
    get_dspark_sparse_sas_window,
    get_or_compute_compressor_metadata,
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
    npu_stream_switch,
    olora_tp_enable,
)

_DSV4_OVERLAP_STREAMS = dict()

def dsv4_overlap_stream(name: str) -> torch.npu.Stream:
    global _DSV4_OVERLAP_STREAMS
    if name not in _DSV4_OVERLAP_STREAMS:
        _DSV4_OVERLAP_STREAMS[name] = torch_npu.npu.Stream()
    return _DSV4_OVERLAP_STREAMS[name]


_DSV4_SWA_OVERLAP_STREAM = None
_DSV4_QUERY_OVERLAP_STREAM = None

_COMPRESSOR_SP_METADATA_KEY = "_compressor_sp_metadata"
_COMPRESSOR_SP_STATE_KEY = "_compressor_sp_state"
_MAIN_COMPRESSOR = "main"
_INDEXER_COMPRESSOR = "indexer"


def dsv4_swa_overlap_stream() -> torch.npu.Stream:
    global _DSV4_SWA_OVERLAP_STREAM
    if _DSV4_SWA_OVERLAP_STREAM is None:
        _DSV4_SWA_OVERLAP_STREAM = torch_npu.npu.Stream()
    return _DSV4_SWA_OVERLAP_STREAM


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
    compressor_sp: CompressorSPMetadata | None = None


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
    cache_group_key: str = ""
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
    req_sas_metadata: torch.Tensor | None
    req_qli_metadata: torch.Tensor | None
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
        if not layer_names:
            raise ValueError("DSA-CP metadata builder requires at least one layer name")
        # vLLM assigns one builder result to every layer in an attention group.
        self.cache_group_key = layer_names[0]

        # Output cache specs retain their compressor ratio, while state caches
        # use a sliding-window spec that does not. Identify state groups first
        # so they are not mistaken for ordinary attention/SWA groups.
        is_indexer_compressor_state = (
            ".indexer.compressor.state_cache" in self.cache_group_key and kv_cache_spec.dtype == torch.float32
        )
        is_main_compressor_state = (
            ".compressor.state_cache" in self.cache_group_key
            and not is_indexer_compressor_state
            and kv_cache_spec.dtype == torch.float32
        )
        self.is_compressor_state = is_indexer_compressor_state or is_main_compressor_state
        self.is_compressor_output = self.compressor_ratio in (4, 128)
        self.is_indexer_compressor_output = (
            self.is_compressor_output and ".indexer.k_cache" in self.cache_group_key
        )
        # Main attention consumes SAS metadata, LI consumes QLI metadata, and
        # compressor state groups consume neither.
        self.needs_sas_metadata = not self.is_compressor_state and not self.is_indexer_compressor_output
        self.needs_qli_metadata = self.is_indexer_compressor_output
        # Output and state cache groups are built independently. This key lets
        # their SP metadata rendezvous through common_ratio_to_sas_metadata.
        self.compressor_sp_output_key: tuple[str, int] | None = None
        if self.is_compressor_output:
            owner = _INDEXER_COMPRESSOR if self.is_indexer_compressor_output else _MAIN_COMPRESSOR
            self.compressor_sp_output_key = (owner, self.compressor_ratio)

        # Allocate the fixed SP workspaces only on configurations that can
        # actually enter the Compressor SP execution path.
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        ascend_device_type = get_ascend_device_type()
        compressor_sp_enabled = (
            bool(additional_config.get("enable_compressor_sp", False))
            and vllm_config.parallel_config.tensor_parallel_size > 1
            and ascend_device_type == AscendDeviceType.A3
        )
        self.compressor_sp_metadata_builder: CompressorSPMetadataBuilder | None = None
        if compressor_sp_enabled and self.compressor_sp_output_key is not None:
            coff = 2 if self.compressor_ratio == 4 else 1
            output_dim = (
                self.model_config.hf_text_config.index_head_dim
                if self.is_indexer_compressor_output
                else self.model_config.hf_text_config.head_dim
            )
            self.compressor_sp_metadata_builder = CompressorSPMetadataBuilder(
                max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
                max_num_seqs=scheduler_config.max_num_seqs,
                tp_size=vllm_config.parallel_config.tensor_parallel_size,
                compress_ratio=self.compressor_ratio,
                coff=coff,
                hidden_dim=self.model_config.get_hidden_size(),
                output_dim=output_dim,
                dtype=self.model_config.dtype,
                device=self.device,
            )
        self.compressor_sp_state_key: tuple[str, int] | None = None
        state_dim = getattr(kv_cache_spec, "head_size", 0)
        if compressor_sp_enabled:
            # State cache specs do not retain the compressor ratio, so derive
            # the rendezvous key from the state identity and current layout.
            # LI currently exists only for C4. Main state stores KV and score
            # state with shape 2 * coff * head_dim: C4 has coff=2 (4D), while
            # C128 has coff=1 (2D).
            if is_indexer_compressor_state:
                self.compressor_sp_state_key = (_INDEXER_COMPRESSOR, 4)
            elif is_main_compressor_state:
                if state_dim == 4 * self.model_config.hf_text_config.head_dim:
                    self.compressor_sp_state_key = (_MAIN_COMPRESSOR, 4)
                elif state_dim == 2 * self.model_config.hf_text_config.head_dim:
                    self.compressor_sp_state_key = (_MAIN_COMPRESSOR, 128)

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
        self.req_sas_metadata = (
            torch.zeros(1024, dtype=torch.int32, device=self.device) if self.needs_sas_metadata else None
        )
        self.req_qli_metadata = (
            torch.zeros(1024, dtype=torch.int32, device=self.device) if self.needs_qli_metadata else None
        )
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
        if ascend_device_type in {AscendDeviceType.A5}:
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
            assert self.decode_threshold <= 16, f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"

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

        # Compressor metadata generates its own compressed-cache slots. Only
        # state and ordinary/SWA groups need the original-token slot mapping.
        if not self.is_compressor_output:
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

        return AscendDSAReqMetadata(
            input_positions=input_positions,
            block_table=self.block_table[:num_reqs, ...],
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            seq_lens=self.seq_lens[:num_reqs],
            query_start_loc=query_start_loc,
            cp_metadata=cp_metadata,
            cache_group_key=self.cache_group_key,
            sin=sin,
            cos=cos,
            start_pos=start_pos,
            sas_metadata=sas_metadata,
            qli_metadata=None,
            cu_cmp_seqlen_list=None,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            dspark_swa_indices=dspark_swa_indices,
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
            # The first cache group computes the shared partition once.
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
            # Each builder owns fixed output buffers, so later groups copy the
            # shared values into their own stable tensor addresses.
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
            local_seq_lens_q_cpu = (
                local_query_start_loc_cpu[1 : num_reqs + 1] - local_query_start_loc_cpu[:num_reqs]
            )
            max_local_query_len = max(1, int(local_seq_lens_q_cpu.max().item()))
            max_local_seq_lens = max(1, int(local_seq_lens_cpu.max().item()))
            self.common_ratio_to_sas_metadata["_cpu_local"] = {
                "qsl_cpu": local_query_start_loc_cpu.clone(),
                "sl_cpu": local_seq_lens_cpu.clone(),
                "max_query_len": max_local_query_len,
                "max_seq_lens": max_local_seq_lens,
            }
        else:
            assert cpu_cache is not None
            local_query_start_loc_cpu = cpu_cache["qsl_cpu"]
            local_seq_lens_cpu = cpu_cache["sl_cpu"]
            max_local_query_len = cpu_cache["max_query_len"]
            max_local_seq_lens = cpu_cache["max_seq_lens"]

        if num_reqs_actual is None:
            num_reqs_actual = num_reqs
        else:
            num_reqs_actual = min(num_reqs_actual, num_reqs)
            if num_reqs_actual < num_reqs:
                self.start_pos_prefill[num_reqs_actual:].fill_(0)
                self.block_table[num_reqs_actual:num_reqs, ...].fill_(0)

        # --- Compressed positions ---
        full_compress_cos, full_compress_sin = None, None
        cu_cmp_seqlens = self._get_cmp_seqlens_for_metadata(has_prefill) if self.needs_sas_metadata else None

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

        # Main attention and ordinary/SWA groups consume SAS metadata.
        sas_metadata = None
        if self.needs_sas_metadata:
            local_seq_lens_q = local_query_start_loc[1 : num_reqs + 1] - local_query_start_loc[:num_reqs]
            sas_metadata = self._build_sas_metadata(
                num_heads=self.model_config.hf_config.num_attention_heads,
                query_start_loc=local_query_start_loc,
                seq_lens=local_seq_lens,
                seq_lens_q=local_seq_lens_q,
                max_query_len=max_local_query_len,
                max_seq_lens=max_local_seq_lens,
                index_topk=self.model_config.hf_config.index_topk,
                num_reqs=num_reqs,
                has_prefill=has_prefill,
                cu_cmp_seqlen_list=cu_cmp_seqlens,
            )

        # Only the LI compressed-output group consumes QLI metadata.
        qli_metadata = None
        if self.needs_qli_metadata:
            qli_metadata = self._build_qli_metadata(
                query_start_loc=local_query_start_loc,
                seq_lens=local_seq_lens,
                max_seqlen_q=max_local_query_len,
                max_seqlen_k=max_local_seq_lens,
                num_reqs=num_reqs,
            )

        # SP is a pure-prefill optimization. Initialization has already gated
        # this builder by device type, TP size, and enable_compressor_sp.
        is_pure_prefill = has_prefill and self.num_decodes == 0
        compressor_sp_metadata = None
        if is_pure_prefill and self.compressor_sp_metadata_builder is not None:
            assert self.compressor_sp_output_key is not None
            assert self.num_actual_tokens is not None
            tp_group = get_tp_group()
            compressor_sp_metadata = self.compressor_sp_metadata_builder.build_sp(
                query_start_loc=query_start_loc_cpu[: num_reqs + 1].tolist(),
                seq_lens=self.seq_lens_cpu[:num_reqs].tolist(),
                num_actual_tokens=self.num_actual_tokens,
                num_input_tokens=num_input_tokens,
                tp_rank=tp_group.rank_in_group,
                num_reqs_actual=num_reqs_actual,
            )
            metadata_key = (_COMPRESSOR_SP_METADATA_KEY, *self.compressor_sp_output_key)
            state_key = (_COMPRESSOR_SP_STATE_KEY, *self.compressor_sp_output_key)
            # The output group owns the SP plan/workspaces. If its state group
            # ran first, complete that pending state-slot binding now.
            self.common_ratio_to_sas_metadata[metadata_key] = compressor_sp_metadata
            pending_state = self.common_ratio_to_sas_metadata.pop(state_key, None)
            if pending_state is not None:
                CompressorSPMetadataBuilder.bind_state_slots(
                    metadata=compressor_sp_metadata,
                    state_slot_mapping=pending_state[0],
                    local_token_start=pending_state[1],
                    tokens_per_rank=pending_state[2],
                    num_tokens_pad=pending_state[3],
                )

        if is_pure_prefill and self.compressor_sp_state_key is not None:
            assert slot_mapping is not None
            metadata_key = (_COMPRESSOR_SP_METADATA_KEY, *self.compressor_sp_state_key)
            state_key = (_COMPRESSOR_SP_STATE_KEY, *self.compressor_sp_state_key)
            target_sp_metadata = self.common_ratio_to_sas_metadata.get(metadata_key)
            # Cache-group order is not part of the contract: bind immediately
            # when the output plan exists, otherwise leave slots pending.
            if target_sp_metadata is None:
                self.common_ratio_to_sas_metadata[state_key] = (
                    slot_mapping,
                    local_start,
                    tokens_per_rank,
                    num_tokens_pad,
                )
            else:
                CompressorSPMetadataBuilder.bind_state_slots(
                    metadata=target_sp_metadata,
                    state_slot_mapping=slot_mapping,
                    local_token_start=local_start,
                    tokens_per_rank=tokens_per_rank,
                    num_tokens_pad=num_tokens_pad,
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
            compressor_sp=compressor_sp_metadata,
        )

        return AscendDSAReqMetadata(
            input_positions=input_positions,
            block_table=self.block_table[:num_reqs, ...],
            slot_mapping=slot_mapping,
            block_size=self.block_size,
            seq_lens=self.seq_lens[:num_reqs],
            query_start_loc=query_start_loc,
            cp_metadata=cp_metadata,
            cache_group_key=self.cache_group_key,
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
        assert self.req_sas_metadata is not None
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

    def _build_qli_metadata(self, query_start_loc, seq_lens, max_seqlen_q, max_seqlen_k, num_reqs):
        assert self.compressor_ratio == 4
        assert self.req_qli_metadata is not None

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
        self.wq_a_kv = kwargs["wq_a_kv"]
        self.q_norm = kwargs["q_norm"]
        self.q_norm_without_weight = kwargs.get("q_norm_without_weight")
        self.kv_norm = kwargs["kv_norm"]

        self.indexer = kwargs.get("indexer")
        self.compressor = kwargs.get("compressor")
        self.compressor_executor: CompressorExecutor | None = None
        self.indexer_compressor_executor: IndexerCompressorExecutor | None = None

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

        ascend_config = get_ascend_config()
        self.multistream_dsv4_dsa_overlap = ascend_config.multistream_dsv4_dsa_overlap
        self.eliminate_dsa_cp_comm = ascend_config.eliminate_dsa_cp_comm and (
            get_ascend_device_type() != AscendDeviceType.A5
        )

        # indexer param
        if self.indexer is not None:
            self.indexer_heads: int = self.indexer.n_heads
            self.inderxer_dim: int = self.indexer.head_dim
            self.inderxer_wq_b = self.indexer.wq_b
            self.weights_proj = self.indexer.weights_proj
            self.indexer_softmax_scale = self.inderxer_dim**-0.5

            self.indexcom_head_dim = self.indexer.compressor.head_dim
            self.index_topk = self.indexer.index_topk
            self.indexer_compressor_executor = IndexerCompressorExecutor(
                self.indexer.compressor,
                self.rope_head_dim,
                self.tp_group,
            )

        # compress param
        if self.compressor is not None:
            self.compressor_executor = CompressorExecutor(
                self.compressor,
                self.rope_head_dim,
                self.tp_group,
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
            skip_all_to_all=full_gather_wo_a_enabled or self.eliminate_dsa_cp_comm,
        )
        num_tokens = o_proj_input.shape[0]

        # o
        if full_gather_wo_a_enabled:
            self._switch_o_proj_to_full_weight(o_proj_full_handles)
        o_proj_groups = self.n_group if full_gather_wo_a_enabled or self.eliminate_dsa_cp_comm else self.n_local_groups
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
                    if full_gather_wo_a_enabled or self.eliminate_dsa_cp_comm:
                        o_proj_input = torch.bmm(
                            o_proj_input.permute(1, 0, 2).contiguous(),
                            self.wo_a.weight,
                        )
                        o_proj_input = o_proj_input.permute(1, 0, 2).contiguous()
                    else:
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

        if self.compress_ratio > 1 and self.multistream_dsv4_dsa_overlap:
            state_stream = dsv4_overlap_stream("state")
            torch.npu.current_stream().wait_stream(state_stream)
        notify_kv_cache_written(layer_name)
        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

        return output

    def _forward_query(self, hidden_states_local, local_cos, local_sin):
        if self.wq_a_kv is not None:
            q_a_kv = self.wq_a_kv(hidden_states_local)
            q_a, kv = torch.split(q_a_kv, [self.q_lora_rank, self.head_dim], dim=-1)
        else:
            q_a = self.wq_a(hidden_states_local)
            kv = None

        if (not isinstance(self.wq_b.quant_method, AscendUnquantizedLinearMethod)) and isinstance(
            self.wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod
        ):
            qr_local, qr_pertoken_scale_local = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                q_a, self.q_norm.weight, epsilon=self.eps
            )
            qr_kv_ready_evt = torch.npu.current_stream().record_event()
            q = torch_npu.npu_quant_matmul(
                qr_local,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale_local,
                bias=self.wq_b.bias,
                output_dtype=hidden_states_local.dtype,
            )
        else:
            qr_local = self.q_norm(q_a)
            qr_kv_ready_evt = torch.npu.current_stream().record_event()
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

        return q, qr_local, qr_pertoken_scale_local, kv, qr_kv_ready_evt

    def _forward_swa_kv(
        self,
        attn_metadata,
        kv_cache,
        layer_name,
        hidden_states_local,
        kv: torch.Tensor | None = None,
        need_gather_q_kv: bool = False,
    ):
        _, swa_kv_cache, _, _, _, _ = DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)
        swa_metadata = attn_metadata[-1] if self.compress_ratio > 1 else attn_metadata[0]
        common_attn_metadata = attn_metadata[0]
        cp_metadata = common_attn_metadata.req_metadata.cp_metadata

        local_cos = cp_metadata.local_cos[layer_name]
        local_sin = cp_metadata.local_sin[layer_name]

        if kv is None:
            assert self.wkv is not None
            kv = self.wkv(hidden_states_local)

        kv = self.kv_norm(kv)
        assert self.rope_head_dim is not None
        kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            local_cos[: kv.shape[0]],
            local_sin[: kv.shape[0]],
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )
        kv = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(kv, need_gather_q_kv)[
            : common_attn_metadata.num_actual_tokens
        ]
        DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, swa_metadata.req_metadata.slot_mapping)

    def _forward_compressor_kv(
        self,
        kv_cache,
        compressor_input: torch.Tensor,
        compressor_output_metadata: M,
        compressor_state_metadata: M,
        sp_metadata: CompressorSPMetadata | None,
        comm_stream: torch.npu.Stream | None = None,
    ) -> CompressorSPPending | None:
        """Run the main Compressor with separately built output/state metadata.

        With ``comm_stream`` the SP row all-gather is only launched here and the
        caller must finalize the returned tail before attention reads the cache;
        otherwise the whole update completes inline as before.
        """
        compress_kv_cache, _, state_cache, _, _, _ = DeviceOperator.unpack_dsa_forward_kv_cache(
            kv_cache, self.compress_ratio
        )
        assert self.compressor_executor is not None
        assert compressor_output_metadata.req_metadata is not None
        assert compressor_state_metadata.req_metadata is not None
        if comm_stream is not None:
            assert sp_metadata is not None
            return self.compressor_executor.launch_sp(
                compressor_input,
                state_cache,
                compress_kv_cache,
                metadata=compressor_output_metadata.req_metadata,
                state_block_table=compressor_state_metadata.req_metadata.block_table,
                sp_metadata=sp_metadata,
                comm_stream=comm_stream,
            )
        self.compressor_executor.run(
            compressor_input,
            state_cache,
            compress_kv_cache,
            metadata=compressor_output_metadata.req_metadata,
            state_block_table=compressor_state_metadata.req_metadata.block_table,
            sp_metadata=sp_metadata,
            delay_sync_sp_state=self.multistream_dsv4_dsa_overlap,
        )
        return None

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
        compress_kv_cache, swa_kv_cache, _, _, _, _ = DeviceOperator.unpack_dsa_forward_kv_cache(
            kv_cache, self.compress_ratio
        )

        # Cache-group order is fixed by the hybrid KV-cache manager. Give each
        # role an explicit name before the execution path starts.
        if self.compress_ratio == 4:
            (
                main_compressor_output_metadata,
                main_compressor_state_metadata,
                indexer_compressor_state_metadata,
                indexer_compressor_output_metadata,
                swa_metadata,
            ) = attn_metadata
        elif self.compress_ratio == 128:
            main_compressor_output_metadata, main_compressor_state_metadata, swa_metadata = attn_metadata
            indexer_compressor_state_metadata = None
            indexer_compressor_output_metadata = None
        else:
            (swa_metadata,) = attn_metadata
            indexer_compressor_state_metadata = None
            indexer_compressor_output_metadata = None
        common_attn_metadata = attn_metadata[0]

        assert common_attn_metadata.req_metadata is not None
        assert swa_metadata.req_metadata is not None
        req_metadata = common_attn_metadata.req_metadata
        cp_metadata = req_metadata.cp_metadata
        local_cos = cp_metadata.local_cos[layer_name]
        local_sin = cp_metadata.local_sin[layer_name]
        local_seq_lengths_query = cp_metadata.local_query_start_loc
        local_seq_lengths_key = cp_metadata.local_seq_lens
        has_prefill = common_attn_metadata.num_prefills > 0
        swa_req_metadata = swa_metadata.req_metadata

        main_compressor_sp_metadata = None
        indexer_compressor_sp_metadata = None
        if self.compress_ratio > 1:
            assert main_compressor_output_metadata.req_metadata is not None
            main_compressor_sp_metadata = (
                main_compressor_output_metadata.req_metadata.cp_metadata.compressor_sp
            )
            if self.compress_ratio == 4:
                assert indexer_compressor_output_metadata is not None
                assert indexer_compressor_output_metadata.req_metadata is not None
                indexer_compressor_sp_metadata = (
                    indexer_compressor_output_metadata.req_metadata.cp_metadata.compressor_sp
                )
                if (main_compressor_sp_metadata is None) != (indexer_compressor_sp_metadata is None):
                    raise ValueError("Main and indexer compressors must use the same SP execution mode")

        # One FIFO stream carries every Compressor SP collective, so the
        # all-gathers keep a single deterministic order across TP ranks and stay
        # off the compute streams. Graph capture cannot host-wait on a handle, so
        # it keeps the inline collectives; the results are byte-identical either
        # way, only the join point moves.
        compressor_sp_comm_stream = None
        if (
            main_compressor_sp_metadata is not None
            and self.multistream_dsv4_dsa_overlap
            and not torch.npu.is_current_stream_capturing()
        ):
            compressor_sp_comm_stream = dsv4_overlap_stream("compressor_sp_comm")

        hs_local_ready_evt = torch.npu.current_stream().record_event()

        query_aux_stream = dsv4_overlap_stream("query")
        with npu_stream_switch(query_aux_stream, enabled=self.multistream_dsv4_dsa_overlap):
            torch.npu.current_stream().wait_event(hs_local_ready_evt)
            q, qr_local, qr_pertoken_scale_local, kv, qr_kv_ready_evt = self._forward_query(hidden_states_local, local_cos, local_sin)

        # The existing weight all-gather can overlap with subsequent computation.
        o_proj_full_handles = self._maybe_all_gather_o_proj_full_weight(full_gather_wo_a_enabled)

        compress_topk_idxs = None
        if self.compress_ratio > 1:
            assert main_compressor_output_metadata.req_metadata is not None
            assert main_compressor_state_metadata.req_metadata is not None
            assert self.compressor_executor is not None

            # Main and LI share one prepared input. Non-SP restores the global
            # hidden batch; SP gathers only boundary suffixes and packs locally.
            if main_compressor_sp_metadata is None:
                compressor_input = self.compressor_executor.prepare_non_sp_input(
                    hidden_states_local,
                    common_attn_metadata.num_actual_tokens,
                    need_gather_q_kv,
                )
            else:
                # Every TP rank must enter this collective, including ranks whose
                # packed compressor input is empty.
                compressor_input = self.compressor_executor.prepare_sp_input(
                    hidden_states_local,
                    main_compressor_sp_metadata,
                )
            compressor_input_ready_evt = torch.npu.current_stream().record_event()

            # Launch the main Compressor first so its all-gather is already in
            # flight while the Indexer computes. Both ranks reach the collective
            # on shape-balanced local work, so no rank waits at the rendezvous.
            main_compressor_aux_stream = dsv4_overlap_stream("main_compressor_aux_stream")
            with npu_stream_switch(main_compressor_aux_stream, enabled=self.multistream_dsv4_dsa_overlap):
                torch.npu.current_stream().wait_event(compressor_input_ready_evt)
                main_compressor_pending = self._forward_compressor_kv(
                    kv_cache,
                    compressor_input,
                    main_compressor_output_metadata,
                    main_compressor_state_metadata,
                    main_compressor_sp_metadata,
                    comm_stream=compressor_sp_comm_stream,
                )

            if self.compress_ratio == 4:
                assert indexer_compressor_state_metadata is not None
                assert indexer_compressor_output_metadata is not None
                indexer_aux_stream = dsv4_overlap_stream("indexer")
                with npu_stream_switch(indexer_aux_stream, enabled=self.multistream_dsv4_dsa_overlap):
                    torch.npu.current_stream().wait_event(compressor_input_ready_evt)
                    indexer_compressor_pending = self._update_indexer_cache(
                        compressor_input=compressor_input,
                        kv_cache=kv_cache,
                        indexer_state_metadata=indexer_compressor_state_metadata,
                        indexer_output_metadata=indexer_compressor_output_metadata,
                        sp_metadata=indexer_compressor_sp_metadata,
                        comm_stream=compressor_sp_comm_stream,
                    )
                    if indexer_compressor_pending is not None:
                        # TopK below reads the Indexer caches, so the gathered
                        # rows must be written before it runs.
                        self.indexer_compressor_executor.finalize_sp(indexer_compressor_pending)
                    indexer_compressor_done_evt = torch.npu.current_stream().record_event()

                    torch.npu.current_stream().wait_event(qr_kv_ready_evt)
                    qr_local.record_stream(torch.npu.current_stream())
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

            if main_compressor_pending is not None:
                # Only attention reads these rows, so the join comes last: the
                # Indexer local Compressor, its finalize and TopK have already
                # covered the main collective.
                with npu_stream_switch(main_compressor_aux_stream, enabled=self.multistream_dsv4_dsa_overlap):
                    self.compressor_executor.finalize_sp(main_compressor_pending)

        # SWA does not depend on preprocessing communication, so it is scheduled later.
        # Launch SWA all-gather last to avoid blocking compressor communication.
        swa_aux_stream = dsv4_overlap_stream("swa")
        with npu_stream_switch(swa_aux_stream, enabled=self.multistream_dsv4_dsa_overlap):
            if kv is not None:
                torch.npu.current_stream().wait_event(qr_kv_ready_evt)
                kv.record_stream(torch.npu.current_stream())
            self._forward_swa_kv(
                attn_metadata,
                kv_cache,
                layer_name,
                hidden_states_local,
                kv,
                need_gather_q_kv,
            )

        if self.multistream_dsv4_dsa_overlap:
            # main stream wait for aux streams to finish before running attention kernel
            torch.npu.current_stream().wait_stream(query_aux_stream)
            torch.npu.current_stream().wait_stream(swa_aux_stream)
            if self.compress_ratio > 1:
                if self.compress_ratio == 4:
                    torch.npu.current_stream().wait_stream(indexer_aux_stream)
                torch.npu.current_stream().wait_stream(main_compressor_aux_stream)
            q.record_stream(torch.npu.current_stream())

            # State cache is not required by attention, so it can be written asynchronously on
            # a separate stream. To avoid impacting compressor execution, perform the state-cache
            # write only after the compressor completion event.
            if self.compress_ratio > 1 and main_compressor_sp_metadata is not None:
                state_stream = dsv4_overlap_stream("state")
                with npu_stream_switch(state_stream, enabled=True):
                    torch.npu.current_stream().wait_stream(main_compressor_aux_stream)
                    if self.compress_ratio == 4:
                        torch.npu.current_stream().wait_event(indexer_compressor_done_evt)

                    main_state_cache = DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)[2]
                    self.compressor_executor._sync_sp_state(
                        main_state_cache,
                        main_compressor_sp_metadata,
                        compressor_sp_comm_stream,
                    )
                    if self.compress_ratio == 4:
                        indexer_state_cache = DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)[0]
                        self.indexer_compressor_executor._sync_sp_state(
                            indexer_state_cache,
                            indexer_compressor_sp_metadata,
                            compressor_sp_comm_stream,
                        )

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
            assert main_compressor_output_metadata.req_metadata is not None
            DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
                common_attn_kwargs, cu_seqlens_cmp_kv=req_metadata.cu_cmp_seqlen_list
            )
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_metadata.req_metadata.block_table,
                cmp_block_table=main_compressor_output_metadata.req_metadata.block_table,
                metadata=req_metadata.sas_metadata,
                cmp_mask_mode=3,
                **common_attn_kwargs,
            )[0]
        else:
            assert main_compressor_output_metadata.req_metadata is not None
            DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
                common_attn_kwargs, cu_seqlens_cmp_kv=req_metadata.cu_cmp_seqlen_list
            )
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                ori_block_table=swa_metadata.req_metadata.block_table,
                cmp_block_table=main_compressor_output_metadata.req_metadata.block_table,
                metadata=main_compressor_output_metadata.req_metadata.sas_metadata,
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
        compressor_input: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        indexer_state_metadata: M,
        indexer_output_metadata: M,
        sp_metadata: CompressorSPMetadata | None,
        comm_stream: torch.npu.Stream | None = None,
    ) -> CompressorSPPending | None:
        """Run LI Compressor and update its K, scale, and full-value caches.

        With ``comm_stream`` only the local Compressor and its row all-gather are
        issued here; the caller must finalize the returned tail before TopK reads
        the Indexer caches.
        """
        indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache = (
            DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        )
        assert indexer_output_metadata.req_metadata is not None
        assert indexer_state_metadata.req_metadata is not None
        assert self.indexer_compressor_executor is not None
        if comm_stream is not None:
            assert sp_metadata is not None
            return self.indexer_compressor_executor.launch_sp(
                compressor_input,
                indexer_state_cache,
                (indexer_k_cache, indexer_scale_cache, indexer_full_cache),
                metadata=indexer_output_metadata.req_metadata,
                state_block_table=indexer_state_metadata.req_metadata.block_table,
                sp_metadata=sp_metadata,
                hadamard=indexer_output_metadata.hadamard,
                comm_stream=comm_stream,
            )
        self.indexer_compressor_executor.run(
            compressor_input,
            indexer_state_cache,
            (indexer_k_cache, indexer_scale_cache, indexer_full_cache),
            metadata=indexer_output_metadata.req_metadata,
            state_block_table=indexer_state_metadata.req_metadata.block_table,
            sp_metadata=sp_metadata,
            hadamard=indexer_output_metadata.hadamard,
            # The deferred state stream owns the replication when overlap is on,
            # so it must not also run inline here.
            delay_sync_sp_state=self.multistream_dsv4_dsa_overlap,
        )
        return None

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
        _, indexer_k_cache, indexer_scale_cache, _ = DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        _, _, _, indexer_output_metadata, _ = attn_metadata
        assert indexer_output_metadata is not None

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
        q = rotate_activation(q, indexer_output_metadata.hadamard)
        weights = self.weights_proj(x) * (self.indexer_softmax_scale * self.indexer_heads**-0.5)

        q, q_scale = DeviceOperator.indexer_quantize_query(q)

        assert indexer_output_metadata.req_metadata is not None
        qli_metadata = indexer_output_metadata.req_metadata.qli_metadata
        block_table = indexer_output_metadata.req_metadata.block_table
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
