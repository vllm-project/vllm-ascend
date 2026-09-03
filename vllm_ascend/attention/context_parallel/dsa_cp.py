import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_pcp_group, get_tp_group
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON, triton
from vllm.v1.attention.backend import AttentionCGSupport, AttentionImplBase, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.attention import dsa_v1
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_attn_kv_plan import (
    get_dsa_attn_kv_plan,
    is_a5_bf16_kv_enabled,
)
from vllm_ascend.attention.dsa_v1 import (
    _dsa_layout_kv,
    _dsa_swa_only_cmp_ratio,
    _has_weight_scale,
    build_dspark_swa_indices,
    get_dspark_sparse_sas_window,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
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
from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorMetadata
from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerMetadata
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.rope_dsv4 import RopeDataProxy, get_cos_and_sin_dsa, get_full_cos_and_sin_dsa
from vllm_ascend.ops.triton.dsa_cp import build_local_metadata_triton
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod
from vllm_ascend.quantization.tp_weight_switch import TPWeightSwitchMixin, TPWeightSwitchState
from vllm_ascend.utils import (
    enable_dsa_cp_with_o_proj_tp,
    olora_tp_enable,
)

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext


# AscendC fused op (aclnnDsaLocalMetadata) that replaces build_local_metadata_triton.
# Lazy resolution: the _C_ascend namespace may not be populated at import time.
# Note: module-level variables must not carry type annotations (annotations are
# evaluated eagerly at import time and may raise NameError if typing imports
# are not guaranteed).
_dsa_local_metadata_op = None
_dsa_local_metadata_op_resolved = False


def _get_dsa_local_metadata_op():
    global _dsa_local_metadata_op, _dsa_local_metadata_op_resolved
    if not _dsa_local_metadata_op_resolved:
        _dsa_local_metadata_op_resolved = True
        try:
            _dsa_local_metadata_op = torch.ops._C_ascend.npu_dsa_local_metadata
            # The op mutates its output tensors in place; mark it as having
            # side effects so FX/Inductor does not DCE or reorder it.
            torch.fx.node.has_side_effect(_dsa_local_metadata_op)
        except (AttributeError, RuntimeError):
            _dsa_local_metadata_op = None
            logger.debug_once(
                "npu_dsa_local_metadata op is unavailable; "
                "falling back to triton/torch implementation of "
                "_build_local_token_metadata."
            )
    return _dsa_local_metadata_op


# =============================================================================
# Legacy DSA-CP implementation (TP/SP group)
# =============================================================================


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
    storage_block_size: int
    query_start_loc: torch.Tensor
    cp_metadata: DSACPMetadata
    num_compressed_tokens: int | None = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    full_compress_sin: torch.Tensor = None
    full_compress_cos: torch.Tensor = None
    start_pos: torch.Tensor = None
    num_actual_reqs: int | None = None
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


DSACPMetadataDict: TypeAlias = dict[str, AscendDSAMetadata]


@dataclass(frozen=True)
class AscendDSACPLayerMetadata:
    swa: AscendDSAMetadata
    compressor_cache: AscendDSAMetadata | None = None
    compressor_state: AscendDSAMetadata | None = None
    indexer_cache: AscendDSAMetadata | None = None
    indexer_state: AscendDSAMetadata | None = None


class AscendDSACPMetadataBuilder(AttentionMetadataBuilder[AscendDSAMetadata]):
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

        self.hadamard = None
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
            self.hadamard = get_or_register_attention_buffer(
                self.vllm_config,
                layer_names,
                "_dsa_cp_hadamard",
                lambda: torch.tensor(hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device).to(
                    torch.bfloat16
                ),
            )
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
        num_actual_reqs = kwargs.get("num_actual_reqs")
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
            common_attn_metadata,
            input_positions,
            num_input_tokens,
            num_actual_reqs,
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
            hadamard=self.hadamard,
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
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        # Draft steps update positions independently. Reusing the global RoPE
        # cache can let later draft steps overwrite step-0 metadata.
        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=False)

        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]

        assert self.spec_slot_mapping is not None
        self.spec_slot_mapping[draft_index - 1][:num_input_tokens] = get_dsa_attn_kv_plan(
            self.vllm_config
        ).format_dsa_slot_mapping(slot_mapping, self.storage_block_size)

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
                self.storage_block_size,
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
        kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
        metadata_op = kv_plan.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = kv_plan.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
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
            layout_kv=_dsa_layout_kv(self.vllm_config),
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
            storage_block_size=self.storage_block_size,
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
        num_actual_reqs: int | None,
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

        if num_actual_reqs is None:
            num_actual_reqs = num_reqs
        else:
            num_actual_reqs = min(num_actual_reqs, num_reqs)
            if num_actual_reqs < num_reqs:
                self.start_pos_prefill[num_actual_reqs:].fill_(0)
                self.block_table[num_actual_reqs:num_reqs, ...].fill_(0)

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
            storage_block_size=self.storage_block_size,
            seq_lens=self.seq_lens[:num_reqs],
            query_start_loc=query_start_loc,
            cp_metadata=cp_metadata,
            sin=sin,
            cos=cos,
            full_compress_sin=full_compress_sin,
            full_compress_cos=full_compress_cos,
            start_pos=self.start_pos_prefill[:num_reqs],
            num_compressed_tokens=num_compressed_tokens,
            num_actual_reqs=num_actual_reqs,
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

        if query_start_loc.device.type != "cpu":
            dsa_local_metadata_op = _get_dsa_local_metadata_op()
        else:
            dsa_local_metadata_op = None

        if dsa_local_metadata_op is not None:
            assert local_query_start_loc is not None and local_seq_lens is not None
            # num_reqs is passed as a runtime attribute (not a constexpr shape),
            # so no re-specialization/recompilation is triggered across steps.
            dsa_local_metadata_op(
                query_start_loc,
                seq_lens,
                local_query_start_loc,
                local_seq_lens,
                start_pos_out if start_pos_out is not None else self._zero_i32,
                local_start,
                local_end,
                num_reqs,
                start_pos_out is not None,
            )
        elif query_start_loc.device.type != "cpu" and HAS_TRITON:
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
            kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
            metadata_op = kv_plan.get_dsa_sparse_attn_metadata_op()
            metadata_kwargs = kv_plan.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
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
                layout_kv=_dsa_layout_kv(self.vllm_config),
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


class AscendDSACPImpl(AttentionImplBase[Any]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    o_proj_full_pools: ClassVar[dict[Any, torch.Tensor]] = {}

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
        self.support_fp8_attention = get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION)
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
        self.swa_cache_layer = kwargs.get("swa_cache_layer")
        assert self.swa_cache_layer is not None

        self.wo_a = kwargs["wo_a"]
        self.wo_b = kwargs["wo_b"]

        self.enable_dsa_cp_with_o_proj_tp = enable_dsa_cp_with_o_proj_tp() and get_current_hardware_profile().supports(
            HardwareCapability.DSA_O_PROJ_TP
        )
        self._o_proj_tp_weight_switch_enabled = False

        self.eps = kwargs["eps"]

        self.attn_sink = kwargs["attn_sink"]

        self.vllm_config = kwargs.get("vllm_config", get_current_vllm_config())

        # indexer param
        if self.indexer is not None:
            self.indexer_heads: int = self.indexer.n_heads
            self.inderxer_dim: int = self.indexer.head_dim
            self.inderxer_wq_b = self.indexer.wq_b
            self.weights_proj = self.indexer.weights_proj
            self.indexer_softmax_scale = self.inderxer_dim**-0.5

            # indexer_compressor
            self.indexcom_ape = self.indexer.compressor.ape
            self.indexcom_wkv = self.indexer.compressor.wkv
            self.indexcom_wgate = self.indexer.compressor.wgate
            self.indexcom_norm = self.indexer.compressor.norm

            self.indexcom_head_dim = self.indexer.compressor.head_dim
            self.index_topk = self.indexer.index_topk

        # compress param
        if self.compressor is not None:
            self.compressor_overlap = self.compressor.overlap

            self.compressor_ape = self.compressor.ape
            self.compressor_wkv = self.compressor.wkv
            self.compressor_wgate = self.compressor.wgate
            self.compressor_norm = self.compressor.norm
            self.compressor_norm_eps = self.compressor.norm_eps

    def _get_layer_metadata(
        self,
        attn_layer_name: str,
        attn_metadata: DSACPMetadataDict,
    ) -> AscendDSACPLayerMetadata:
        assert self.swa_cache_layer is not None
        swa_metadata = attn_metadata[self.swa_cache_layer.prefix]
        compressor_cache_metadata = None
        compressor_state_metadata = None
        indexer_cache_metadata = None
        indexer_state_metadata = None

        if self.compress_ratio > 1:
            assert self.compressor is not None
            compressor_cache_metadata = attn_metadata[attn_layer_name]
            compressor_state_metadata = attn_metadata[self.compressor.state_cache.prefix]
            if self.compress_ratio == 4:
                assert self.indexer is not None
                assert self.indexer.compressor is not None
                indexer_cache_metadata = attn_metadata[self.indexer.k_cache.prefix]
                indexer_state_metadata = attn_metadata[self.indexer.compressor.state_cache.prefix]

        return AscendDSACPLayerMetadata(
            swa=swa_metadata,
            compressor_cache=compressor_cache_metadata,
            compressor_state=compressor_state_metadata,
            indexer_cache=indexer_cache_metadata,
            indexer_state=indexer_state_metadata,
        )

    def _compute_compressor_metadata(
        self,
        metadata: AscendDSAReqMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert metadata.full_compress_cos is not None
        assert metadata.full_compress_sin is not None
        assert metadata.num_compressed_tokens is not None
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
        return torch.ops._C_ascend.compressor_metadata(
            full_compress_cos,
            full_compress_sin,
            metadata.query_start_loc,
            metadata.start_pos,
            metadata.block_table,
            metadata.storage_block_size,
            get_dsa_attn_kv_plan(self.vllm_config).get_dsa_compressor_slot_mapping_format(),
            self.compress_ratio,
            metadata.num_compressed_tokens,
            metadata.num_actual_reqs,
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if self.attn_sink.numel() != self.num_heads:
            raise RuntimeError(
                "DSA-CP expects full-head attn_sink loaded on every TP rank, "
                f"got {self.attn_sink.numel()} heads, expected {self.num_heads}."
            )
        if self.enable_dsa_cp_with_o_proj_tp:
            self._enable_o_proj_tp_full_weight_switch()

    @staticmethod
    def _get_tp_weight_switch_method(layer: torch.nn.Module) -> TPWeightSwitchMixin:
        quant_method = layer.quant_method
        linear_method = getattr(quant_method, "quant_method", quant_method)
        if not isinstance(linear_method, TPWeightSwitchMixin) or not linear_method.supports_tp_weight_switch:
            raise RuntimeError(
                "DSA-CP o_proj TP full-weight switching requires a TP weight-switch capable method, "
                f"got {type(linear_method).__name__}."
            )
        return linear_method

    def _enable_linear_tp_weight_switch(
        self,
        layer: torch.nn.Module,
        name: str,
    ) -> tuple[TPWeightSwitchMixin, TPWeightSwitchState]:
        linear_method = self._get_tp_weight_switch_method(layer)
        state = linear_method.enable_tp_weight_switch(
            layer,
            self.tp_size,
            pool=AscendDSACPImpl.o_proj_full_pools,
            pool_key_prefix=(type(linear_method).__qualname__, name, "dsa_cp_o_proj"),
            clone_tp_tensors=True,
        )
        return linear_method, state

    def _enable_o_proj_tp_full_weight_switch(self) -> None:
        """Allocate o_proj TP/full buffers when the DSA-CP backend is enabled."""
        if self._o_proj_tp_weight_switch_enabled:
            return
        self.wo_a_tp_weight_method, self.wo_a_tp_weight_state = self._enable_linear_tp_weight_switch(
            self.wo_a,
            "wo_a",
        )
        self.wo_b_tp_weight_method, self.wo_b_tp_weight_state = self._enable_linear_tp_weight_switch(
            self.wo_b,
            "wo_b",
        )
        self._o_proj_tp_weight_switch_enabled = True

    def _maybe_all_gather_o_proj_full_weight(
        self,
        enabled: bool,
    ) -> None:
        if not enabled:
            return
        self._enable_o_proj_tp_full_weight_switch()
        self.wo_a_tp_weight_method.all_gather_tp_weight(
            self.wo_a_tp_weight_state,
            self.tp_group,
        )
        self.wo_b_tp_weight_method.all_gather_tp_weight(
            self.wo_b_tp_weight_state,
            self.tp_group,
        )

    def _switch_o_proj_to_full_weight(self) -> None:
        self.wo_a_tp_weight_method.wait_tp_weight_all_gather(self.wo_a_tp_weight_state)
        self.wo_b_tp_weight_method.wait_tp_weight_all_gather(self.wo_b_tp_weight_state)
        self.wo_a_tp_weight_method.switch_tp_weight(
            self.wo_a,
            self.wo_a_tp_weight_state,
            use_full_weight=True,
        )
        self.wo_b_tp_weight_method.switch_tp_weight(
            self.wo_b,
            self.wo_b_tp_weight_state,
            use_full_weight=True,
        )

    def _switch_o_proj_to_tp_weight(self) -> None:
        self.wo_a_tp_weight_method.switch_tp_weight(
            self.wo_a,
            self.wo_a_tp_weight_state,
            use_full_weight=False,
        )
        self.wo_b_tp_weight_method.switch_tp_weight(
            self.wo_b,
            self.wo_b_tp_weight_state,
            use_full_weight=False,
        )

    def _apply_wo_b(
        self,
        o_proj_input: torch.Tensor,
        full_weight: bool,
    ) -> torch.Tensor:
        if not full_weight:
            return self.wo_b(o_proj_input)
        return self.wo_b.quant_method.apply(self.wo_b, o_proj_input, bias=None)

    def _get_batched_wo_a_weight(self, num_groups: int) -> torch.Tensor:
        """Return wo_a in the DSA batched-matmul layout [group, input, rank]."""
        weight = self.wo_a.weight
        if weight.ndim == 3:
            if weight.shape[0] == num_groups:
                return weight
            if weight.shape[1] == num_groups:
                return weight.permute(1, 0, 2)
            raise RuntimeError(
                "DSA-CP wo_a weight has no group axis matching the o_proj input: "
                f"weight_shape={tuple(weight.shape)}, num_groups={num_groups}."
            )

        linear_method = getattr(self.wo_a.quant_method, "quant_method", self.wo_a.quant_method)
        if isinstance(linear_method, AscendUnquantizedLinearMethod):
            return weight.reshape(num_groups, -1, weight.shape[-1]).transpose(1, 2)
        return weight.reshape(weight.shape[0], num_groups, -1).permute(1, 0, 2)

    def _get_batched_wo_a_scale(self, num_groups: int) -> torch.Tensor:
        """Move the output-sharded wo_a scale's group axis to the front."""
        scale = self.wo_a.weight_scale
        if scale.ndim == 1:
            return scale.reshape(num_groups, -1)
        if scale.shape[0] == num_groups:
            return scale
        if scale.shape[1] % num_groups != 0:
            raise RuntimeError(
                "DSA-CP wo_a scale cannot be reshaped by o_proj group: "
                f"scale_shape={tuple(scale.shape)}, num_groups={num_groups}."
            )
        scale = scale.reshape(scale.shape[0], num_groups, -1, *scale.shape[2:])
        return scale.permute(1, 0, 2, *range(3, scale.ndim))

    @staticmethod
    def _split_full_hidden_states_for_cp(
        hidden_states: torch.Tensor,
        cp_metadata: DSACPMetadata,
    ) -> torch.Tensor:
        """Return this TP rank's token shard from the replicated model state.

        FlashComm used to reduce-scatter every row-parallel output, so DSA-CP
        received an already sharded tensor and gathered a second copy for KV
        updates. Without FlashComm, normal TP keeps the model state replicated:
        DSA-CP must slice Q locally while continuing to use the full tensor for
        KV cache updates.
        """
        expected_tokens = cp_metadata.num_tokens_pad
        actual_tokens = hidden_states.shape[0]
        if actual_tokens > expected_tokens:
            raise RuntimeError(
                "DSA-CP input exceeds its TP-aligned metadata, "
                f"got {actual_tokens} tokens and num_tokens_pad={expected_tokens}."
            )
        if actual_tokens < expected_tokens:
            hidden_states = F.pad(hidden_states, (0, 0, 0, expected_tokens - actual_tokens))

        local_hidden_states = hidden_states[cp_metadata.local_start : cp_metadata.local_end]
        if local_hidden_states.shape[0] != cp_metadata.tokens_per_rank:
            raise RuntimeError(
                "DSA-CP local token slice does not match tokens_per_rank, "
                f"got {local_hidden_states.shape[0]} and expected "
                f"{cp_metadata.tokens_per_rank}."
            )
        return local_hidden_states

    def _gather_cp_output(
        self,
        local_output: torch.Tensor,
        cp_metadata: DSACPMetadata,
        num_output_tokens: int | None = None,
    ) -> torch.Tensor:
        """Restore the replicated model-state layout after DSA-CP.

        DSA-CP computes a contiguous token shard on every physical TP rank.
        FlashComm used to keep that sharded layout between transformer layers;
        normal TP does not, so the local attention output must be gathered
        before it is returned to the residual path. Decode and speculative
        decoding take the regular TP all-to-all path, which has already
        restored the full token layout before the output projection.
        """
        if local_output.shape[0] == cp_metadata.num_tokens_pad:
            full_output = local_output
        elif local_output.shape[0] == cp_metadata.tokens_per_rank:
            full_output = local_output
            if self.tp_size > 1:
                full_output = self.tp_group.all_gather(local_output.contiguous(), dim=0)
        else:
            raise RuntimeError(
                "DSA-CP local output does not match tokens_per_rank, "
                f"got {local_output.shape[0]} and expected "
                f"{cp_metadata.tokens_per_rank}."
            )

        if full_output.shape[0] != cp_metadata.num_tokens_pad:
            raise RuntimeError(
                "DSA-CP gathered output does not match num_tokens_pad, "
                f"got {full_output.shape[0]} and expected "
                f"{cp_metadata.num_tokens_pad}."
            )
        if num_output_tokens is None:
            num_output_tokens = cp_metadata.num_tokens_pad
        if not 0 <= num_output_tokens <= cp_metadata.num_tokens_pad:
            raise RuntimeError(
                "DSA-CP output token count must fit the TP-aligned state, "
                f"got {num_output_tokens} and num_tokens_pad={cp_metadata.num_tokens_pad}."
            )
        if num_output_tokens == cp_metadata.num_tokens_pad:
            return full_output
        return full_output[:num_output_tokens]

    def forward(  # type: ignore[override]
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor],
        attn_metadata: DSACPMetadataDict,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        layer_metadata = self._get_layer_metadata(layer_name, attn_metadata)
        common_attn_metadata = layer_metadata.compressor_cache
        if common_attn_metadata is None:
            common_attn_metadata = layer_metadata.swa
        wait_for_kv_layer_from_connector(layer_name)
        full_gather_wo_a_enabled = (
            self.tp_size > 1
            and self.enable_dsa_cp_with_o_proj_tp
            and common_attn_metadata.attn_state
            not in {
                AscendAttentionState.DecodeOnly,
                AscendAttentionState.SpecDecoding,
            }
        )
        local_attn_output = self._forward(
            layer_name,
            hidden_states,
            kv_cache,
            layer_metadata,
            full_gather_wo_a_enabled,
        )
        o_proj_input = self._restore_tp_head_layout(
            local_attn_output,
            layer_name,
            common_attn_metadata,
            skip_all_to_all=full_gather_wo_a_enabled,
        )
        num_tokens = o_proj_input.shape[0]

        # o
        if full_gather_wo_a_enabled:
            self._switch_o_proj_to_full_weight()
        o_proj_groups = self.n_group if full_gather_wo_a_enabled else self.n_local_groups
        try:
            use_a5_quant_o_proj = self.support_fp8_attention and _has_weight_scale(self.wo_a)
            if use_a5_quant_o_proj:
                o = o_proj_input.view(num_tokens, o_proj_groups, -1)
                wo_a_method = getattr(self.wo_a.quant_method, "quant_method", self.wo_a.quant_method)
                if isinstance(wo_a_method, AscendUnquantizedLinearMethod):
                    o = torch.bmm(o.transpose(0, 1), self._get_batched_wo_a_weight(o_proj_groups)).transpose(0, 1)
                else:
                    o, swiglu_out_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
                    o = torch_npu.npu_transpose_quant_batchmatmul(
                        o,
                        self._get_batched_wo_a_weight(o_proj_groups),
                        dtype=torch.bfloat16,
                        bias=None,
                        group_sizes=(0, 0, 32),
                        x1_scale=swiglu_out_scale.view(torch.float8_e8m0fnu),
                        x2_scale=self._get_batched_wo_a_scale(o_proj_groups).view(torch.float8_e8m0fnu),
                        perm_x1=(1, 0, 2),
                        perm_x2=(0, 1, 2),
                        perm_y=(1, 0, 2),
                    )
                o = o.reshape(num_tokens, -1)
                local_output = self._apply_wo_b(o, full_gather_wo_a_enabled)
            else:
                o_proj_input = o_proj_input.view(num_tokens, o_proj_groups, -1)
                if olora_tp_enable():
                    o_proj_input = self.wo_a(o_proj_input)
                else:
                    # wo_a = self.wo_a.weight.view(o_proj_groups, self.o_lora_rank, -1)
                    # o = torch.einsum("tgd,grd->tgr", o, wo_a)
                    # A5 BF16 uses the same 3D [groups, hidden, rank] layout.
                    o_proj_input = torch_npu.npu_transpose_batchmatmul(
                        o_proj_input,
                        self._get_batched_wo_a_weight(o_proj_groups),
                        bias=None,
                        scale=None,
                        perm_x1=(1, 0, 2),
                        perm_x2=(0, 1, 2),
                        perm_y=(1, 0, 2),
                        batch_split_factor=1,
                    )
                o_proj_input = o_proj_input.reshape(num_tokens, -1)
                local_output = self._apply_wo_b(o_proj_input, full_gather_wo_a_enabled)

            req_metadata = common_attn_metadata.req_metadata
            assert req_metadata is not None
            cp_metadata = req_metadata.cp_metadata
            output[...] = self._gather_cp_output(local_output, cp_metadata, output.shape[0])
        finally:
            if full_gather_wo_a_enabled:
                self._switch_o_proj_to_tp_weight()

        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

        return output

    def _forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple,
        layer_metadata: AscendDSACPLayerMetadata,
        full_gather_wo_a_enabled: bool = False,
    ):
        """Run full-sequence KV cache updates and local-token attention."""
        (compress_kv_cache, swa_kv_cache, state_cache, _, _, _) = DeviceOperator.unpack_dsa_forward_kv_cache(
            kv_cache, self.compress_ratio
        )
        swa_metadata = layer_metadata.swa
        common_attn_metadata = layer_metadata.compressor_cache
        if common_attn_metadata is None:
            common_attn_metadata = swa_metadata

        assert common_attn_metadata.req_metadata is not None
        assert swa_metadata.req_metadata is not None
        req_metadata = common_attn_metadata.req_metadata
        cp_metadata = req_metadata.cp_metadata
        hidden_states_local = self._split_full_hidden_states_for_cp(hidden_states, cp_metadata)
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

        self._maybe_all_gather_o_proj_full_weight(full_gather_wo_a_enabled)

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
        get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
            swa_kv_cache, kv, swa_metadata.req_metadata.slot_mapping
        )

        compress_topk_idxs = None
        if self.compress_ratio > 1:
            compressor_attn_metadata = layer_metadata.compressor_cache
            compressor_kv_state_metadata = layer_metadata.compressor_state
            assert compressor_attn_metadata is not None
            assert compressor_kv_state_metadata is not None
            assert compressor_attn_metadata.req_metadata is not None
            assert compressor_kv_state_metadata.req_metadata is not None
            if self.compress_ratio == 4:
                assert self.indexer is not None
                if self.indexer.skip_topk:
                    compress_topk_idxs = self.indexer._get_cached_topk_indices(
                        num_tokens=hidden_states_local.shape[0],
                    )
                else:
                    assert layer_metadata.indexer_cache is not None
                    assert layer_metadata.indexer_state is not None
                    self._update_indexer_cache(
                        x=hidden_states_cache,
                        kv_cache=kv_cache,
                        metadata=layer_metadata,
                        actual_seq_lengths_query=actual_seq_lengths_query,
                    )
                    compress_topk_idxs = self._indexer_select_topk(
                        x=hidden_states_local,
                        qr=qr_local,
                        kv_cache=kv_cache,
                        metadata=layer_metadata,
                        cos=local_cos,
                        sin=local_sin,
                        actual_seq_lengths_query=local_seq_lengths_query,
                        actual_seq_lengths_key=local_seq_lengths_key,
                        qr_pertoken_scale=qr_pertoken_scale_local,
                    )

                    if self.indexer.use_index_cache:
                        self.indexer._update_cached_topk_indices(compress_topk_idxs)

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
            get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                compress_kv_cache, compressed_kv, compress_slot_mapping
            )

        notify_kv_cache_written(layer_name)
        record_attention_compute_start()
        kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
        attn_op = kv_plan.get_dsa_sparse_attn_op()
        extra_attn_kwargs = kv_plan.get_dsa_sparse_attn_base_kwargs()
        if has_prefill:
            kv_plan.add_dsa_sparse_attn_extra_kwargs(extra_attn_kwargs, cu_seqlens_ori_kv=local_seq_lengths_query)
        if swa_req_metadata.dspark_swa_indices is not None:
            extra_attn_kwargs["ori_sparse_indices"] = swa_req_metadata.dspark_swa_indices

        ori_win_left = self.window_size - 1 if swa_req_metadata.ori_win_left is None else swa_req_metadata.ori_win_left
        ori_win_right = 0 if swa_req_metadata.ori_win_right is None else swa_req_metadata.ori_win_right

        common_attn_kwargs = dict(
            cu_seqlens_q=local_seq_lengths_query,
            seqused_kv=local_seq_lengths_key,
            sinks=self.attn_sink,
            softmax_scale=self.softmax_scale,
            cmp_ratio=_dsa_swa_only_cmp_ratio(self.compress_ratio, self.vllm_config),
            ori_mask_mode=4,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q="TND",
            layout_kv=_dsa_layout_kv(self.vllm_config),
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
            assert compressor_attn_metadata is not None
            compressor_req_metadata = compressor_attn_metadata.req_metadata
            assert compressor_req_metadata is not None
            kv_plan.add_dsa_sparse_attn_extra_kwargs(
                common_attn_kwargs, cu_seqlens_cmp_kv=req_metadata.cu_cmp_seqlen_list
            )
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_metadata.req_metadata.block_table,
                cmp_block_table=compressor_req_metadata.block_table,
                metadata=req_metadata.sas_metadata,
                cmp_mask_mode=3,
                **common_attn_kwargs,
            )[0]
        else:
            assert compressor_attn_metadata is not None
            compressor_req_metadata = compressor_attn_metadata.req_metadata
            assert compressor_req_metadata is not None
            kv_plan.add_dsa_sparse_attn_extra_kwargs(
                common_attn_kwargs, cu_seqlens_cmp_kv=req_metadata.cu_cmp_seqlen_list
            )
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                ori_block_table=swa_metadata.req_metadata.block_table,
                cmp_block_table=compressor_req_metadata.block_table,
                metadata=compressor_req_metadata.sas_metadata,
                cmp_mask_mode=3,
                **common_attn_kwargs,
            )[0]
        return attn_output

    def _restore_tp_head_layout(
        self,
        local_attn_output: torch.Tensor,
        layer_name: str,
        attn_metadata: AscendDSAMetadata,
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
        metadata: AscendDSACPLayerMetadata,
        actual_seq_lengths_query: torch.Tensor,
    ) -> None:
        (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        )
        indexer_kv_state_metadata = metadata.indexer_state
        indexer_kv_scale_metadata = metadata.indexer_cache
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
        metadata: AscendDSACPLayerMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        (_, indexer_k_cache, indexer_scale_cache, _) = DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        indexer_kv_scale_metadata = metadata.indexer_cache
        assert indexer_kv_scale_metadata is not None

        if (
            (not isinstance(self.inderxer_wq_b.quant_method, AscendUnquantizedLinearMethod))
            and isinstance(self.inderxer_wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod)
            and qr_pertoken_scale is not None
            and not self.support_fp8_attention
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


# =============================================================================
# MRV2 DSA-PCP implementation
# =============================================================================


@dataclass(kw_only=True)
class AscendDSAPCPMetadata(dsa_v1.AscendDSAMetadata):
    """Rank-local DSA metadata with its canonical cache-update view."""

    local_num_tokens_after_padding: int
    hidden_restore_idx: torch.Tensor
    global_dsa_metadata: dsa_v1.AscendDSAMetadata

    @classmethod
    def from_local_metadata(
        cls,
        local_metadata: dsa_v1.AscendDSAMetadata,
        local_num_tokens_after_padding: int,
        hidden_restore_idx: torch.Tensor,
        global_dsa_metadata: dsa_v1.AscendDSAMetadata,
    ) -> "AscendDSAPCPMetadata":
        return cls(
            num_actual_tokens=local_metadata.num_actual_tokens,
            num_decodes=local_metadata.num_decodes,
            num_decode_tokens=local_metadata.num_decode_tokens,
            num_prefills=local_metadata.num_prefills,
            head_dim=local_metadata.head_dim,
            attn_state=local_metadata.attn_state,
            req_metadata=local_metadata.req_metadata,
            reshape_cache_event=local_metadata.reshape_cache_event,
            hadamard=local_metadata.hadamard,
            local_num_tokens_after_padding=local_num_tokens_after_padding,
            hidden_restore_idx=hidden_restore_idx,
            global_dsa_metadata=global_dsa_metadata,
        )


class AscendDSAPCPMetadataBuilder(dsa_v1.AscendDSAMetadataBuilder):
    """Build rank-local attention and canonical global cache metadata."""

    def __init__(
        self,
        kv_cache_spec: AscendMLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
        )
        # DualChunkSwap can expand each scheduler-global prefill request into
        # two rank-local rows. Keep this in sync with PCPManager's local input
        # buffers, which are sized to ``2 * max_num_seqs``. The canonical
        # global builder below must retain the scheduler-global capacity.
        max_num_local_reqs = 2 * vllm_config.scheduler_config.max_num_seqs
        self.start_pos_prefill = self.start_pos_prefill.new_zeros(
            max_num_local_reqs,
        )
        self._global_metadata_builder = dsa_v1.AscendDSAMetadataBuilder(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls=dsa_v1.AscendDSAMetadata,
        )
        self._pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
        self._pcp_rank = get_pcp_group().rank_in_group

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendDSAPCPMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.NEVER

    @staticmethod
    def _build_global_common_attn_metadata(
        pcp_context: "AscendPCPAttentionContext",
        cache_group_idx: int,
        local_common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendCommonAttentionMetadata:
        global_batch = pcp_context.global_batch
        num_reqs = global_batch.num_reqs
        return AscendCommonAttentionMetadata(
            query_start_loc=global_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(global_batch.query_start_loc_np),
            seq_lens=global_batch.seq_lens[:num_reqs],
            seq_lens_cpu=torch.from_numpy(global_batch.seq_lens_np)[:num_reqs],
            seq_lens_cpu_upper_bound=global_batch.seq_lens_cpu_upper_bound[:num_reqs],
            num_computed_tokens_cpu=torch.from_numpy(global_batch.num_computed_tokens_np),
            num_reqs=num_reqs,
            num_actual_tokens=global_batch.num_tokens,
            max_query_len=int(global_batch.num_scheduled_tokens.max()),
            max_seq_len=local_common_attn_metadata.max_seq_len,
            block_table_tensor=pcp_context.global_block_tables[cache_group_idx],
            slot_mapping=pcp_context.global_slot_mappings[cache_group_idx],
            causal=local_common_attn_metadata.causal,
            dcp_local_seq_lens=global_batch.dcp_local_seq_lens,
            positions=global_batch.positions,
            attn_state=global_batch.attn_state,
            num_input_tokens=global_batch.num_tokens,
            is_prefilling=torch.from_numpy(global_batch.is_prefilling_np),
        )

    def _build_local_common_attn_metadata(
        self,
        pcp_context: "AscendPCPAttentionContext",
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendCommonAttentionMetadata:
        num_local_padded_tokens = pcp_context.local_num_tokens_after_padding
        gathered_slot_mapping = common_attn_metadata.slot_mapping
        local_slot_mapping = gathered_slot_mapping.view(
            self._pcp_world_size,
            num_local_padded_tokens,
        )[self._pcp_rank]
        return common_attn_metadata.replace(
            slot_mapping=local_slot_mapping,
            num_input_tokens=num_local_padded_tokens,
        )

    def _build_local_dsa_metadata(
        self,
        common_prefix_len: int,
        local_common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool,
        **kwargs,
    ) -> dsa_v1.AscendDSAMetadata:
        if local_common_attn_metadata.num_actual_tokens > 0:
            return super().build(
                common_prefix_len,
                local_common_attn_metadata,
                fast_build,
                **kwargs,
            )

        # Empty ranks still participate in the global cache update collectives.
        self.common_ratio_to_sas_metadata = kwargs.get(
            "common_ratio_to_sas_metadata",
        )
        return self.metadata_cls(  # type: ignore[call-arg]
            num_actual_tokens=0,
            head_dim=self.model_config.get_head_size(),
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=0,
            attn_state=local_common_attn_metadata.attn_state,
            req_metadata=None,
            hadamard=dsa_v1.AscendDSAMetadataBuilder.hadamard,
        )

    def _build_global_dsa_metadata(
        self,
        common_prefix_len: int,
        global_common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool,
        **kwargs,
    ) -> dsa_v1.AscendDSAMetadata:
        global_build_kwargs = {
            **kwargs,
            "common_ratio_to_sas_metadata": {},
            "num_actual_reqs": global_common_attn_metadata.num_reqs,
        }
        return self._global_metadata_builder.build(
            common_prefix_len,
            global_common_attn_metadata,
            fast_build,
            **global_build_kwargs,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        pcp_context: "AscendPCPAttentionContext | None" = None,
        pcp_cache_group_idx: int | None = None,
        **kwargs,
    ) -> AscendDSAPCPMetadata:
        assert pcp_context is not None
        assert pcp_cache_group_idx is not None
        global_common_attn_metadata = self._build_global_common_attn_metadata(
            pcp_context,
            pcp_cache_group_idx,
            common_attn_metadata,
        )
        global_dsa_metadata = self._build_global_dsa_metadata(
            common_prefix_len,
            global_common_attn_metadata,
            fast_build,
            **kwargs,
        )
        local_common_attn_metadata = self._build_local_common_attn_metadata(
            pcp_context,
            common_attn_metadata,
        )
        local_dsa_metadata = self._build_local_dsa_metadata(
            common_prefix_len,
            local_common_attn_metadata,
            fast_build,
            **kwargs,
        )
        return AscendDSAPCPMetadata.from_local_metadata(
            local_dsa_metadata,
            pcp_context.local_num_tokens_after_padding,
            pcp_context.hidden_restore_idx,
            global_dsa_metadata,
        )


class AscendDSAPCPImpl(dsa_v1.AscendDSAImpl):
    """Run batched global DSA cache updates before rank-local PCP attention."""

    supports_pcp: ClassVar[bool] = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # PCP prepares replicated caches before local attention, leaving no
        # cache-update work for the auxiliary stream to overlap.
        self.multistream_dsv4_dsa_overlap = False

    def _gather_and_restore_hidden_states(
        self,
        hidden_states: torch.Tensor,
        metadata: AscendDSAPCPMetadata,
    ) -> torch.Tensor:
        """All-gather one padded rank slice and restore scheduler token order."""
        gathered_hidden_states = get_pcp_group().all_gather(
            hidden_states.contiguous(),
            dim=0,
        )
        return torch.index_select(
            gathered_hidden_states,
            0,
            metadata.hidden_restore_idx,
        )

    def _update_global_swa_cache(
        self,
        layer_name: str,
        global_hidden_states: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        swa_metadata: dsa_v1.AscendDSAMetadata,
    ) -> None:
        """Update the replicated SWA cache from the canonical global batch."""
        req_metadata = dsa_v1._require_req_metadata(swa_metadata)
        assert req_metadata.slot_mapping is not None

        kv = self.kv_norm(self.wkv(global_hidden_states))
        assert self.rope_head_dim is not None
        kv = kv.view(
            -1,
            1,
            self.nope_head_dim + self.rope_head_dim,
        )
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            req_metadata.cos[layer_name],
            req_metadata.sin[layer_name],
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )
        get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
            swa_kv_cache,
            kv,
            req_metadata.slot_mapping,
        )

    def _update_global_compressor_cache(
        self,
        global_hidden_states: torch.Tensor,
        metadata: AscendCompressorMetadata,
        compress_kv_cache: torch.Tensor,
        state_cache: torch.Tensor,
    ) -> None:
        """Update the DSA compressor cache from the canonical global batch."""
        assert self.compressor is not None
        compressed_kv, compress_slot_mapping = self.compressor(
            hidden_states=global_hidden_states,
            state_cache=state_cache,
            metadata=metadata,
        )
        if compressed_kv.shape[0] > 0:
            get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                compress_kv_cache,
                compressed_kv,
                compress_slot_mapping,
            )

    def _update_global_indexer_cache(
        self,
        global_hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        metadata: AscendIndexerMetadata,
    ) -> None:
        """Update the Indexer cache from the canonical global batch."""
        indexer = self.indexer
        assert indexer is not None
        if indexer.skip_topk:
            return
        indexer.update_cache(
            hidden_states=global_hidden_states,
            kv_cache=kv_cache,
            metadata=metadata,
        )

    def _prepare_caches_before_attention(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: dsa_v1.DSAMetadataDict,
    ) -> bool:
        """Restore one global batch and update each replicated cache once."""
        pcp_metadata = next(iter(attn_metadata.values()))
        assert isinstance(pcp_metadata, AscendDSAPCPMetadata)
        global_hidden_states = self._gather_and_restore_hidden_states(
            hidden_states,
            pcp_metadata,
        )

        global_dsa_metadata_by_prefix = {}
        for cache_prefix, metadata in attn_metadata.items():
            assert isinstance(metadata, AscendDSAPCPMetadata)
            global_dsa_metadata_by_prefix[cache_prefix] = metadata.global_dsa_metadata
        global_layer_metadata = self._get_layer_metadata(
            layer_name,
            global_dsa_metadata_by_prefix,
        )

        cmp_kv, swa_kv, state_cache, _, _, _ = DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)

        self._update_global_swa_cache(
            layer_name,
            global_hidden_states,
            swa_kv,
            global_layer_metadata.swa,
        )
        if self.compress_ratio > 1:
            compressor_metadata = global_layer_metadata.compressor
            assert compressor_metadata is not None
            assert cmp_kv is not None
            assert state_cache is not None
            self._update_global_compressor_cache(
                global_hidden_states,
                compressor_metadata,
                cmp_kv,
                state_cache,
            )

            if self.compress_ratio == 4:
                indexer_metadata = global_layer_metadata.indexer
                assert indexer_metadata is not None
                self._update_global_indexer_cache(
                    global_hidden_states,
                    kv_cache,
                    indexer_metadata,
                )
        return True

    def _get_o_proj_input_shape(
        self,
        attn_metadata: dsa_v1.DSAMetadataDict | None,
    ) -> tuple[int, int, int]:
        if attn_metadata is None:
            return super()._get_o_proj_input_shape(attn_metadata)
        pcp_metadata = next(iter(attn_metadata.values()))
        assert isinstance(pcp_metadata, AscendDSAPCPMetadata)
        return (
            pcp_metadata.local_num_tokens_after_padding,
            self.n_local_heads,
            self.head_dim,
        )
