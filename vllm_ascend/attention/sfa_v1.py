import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import scipy  # type: ignore
import torch
import torch_npu
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import (
    AttentionBackend,  # type: ignore
    AttentionCGSupport,
    MLAAttentionImpl,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import MLAPO_MAX_SUPPORTED_TOKENS
from vllm_ascend.attention.utils import (
    SFA_QSFA_TILE_SIZE,
    AscendCommonAttentionMetadata,
    ascend_chunked_prefill_workspace_size,
    get_sfa_qsfa_packed_head_dim,
    maybe_save_kv_layer_to_connector,
    notify_kv_cache_written,
    trans_rope_weight,
    transdata,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
    record_attention_compute_start,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    OFFLOAD_K_CACHE_NPU_INDEX,
    OFFLOAD_KV_CACHE_TUPLE_LEN,
    OFFLOAD_V_CACHE_NPU_INDEX,
)
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
from vllm_ascend.quantization.methods import (
    AscendW8A8DynamicLinearMethod,
    AscendW8A8LinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.utils import (
    ACL_FORMAT_FRACTAL_ND,
    ACL_FORMAT_FRACTAL_NZ,
    AscendDeviceType,
    dispose_layer,
    enable_sp,
    get_ascend_device_type,
    maybe_trans_nz,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_ascend.worker.npu_input_batch import NPUInputBatch

# token count limits within bmm_transpose operator
BMM_TRANS_MAX_SUPPORTED_TOKENS = 1024


class PreprocessType(enum.Enum):
    NATIVE = "native"
    PROLOG_V3 = "prolog_v3"
    MLAPO = "mlapo"


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

    @staticmethod
    def get_builder_cls():
        if get_ascend_config().sparse_kv_offload_config.enabled:
            from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadMetadataBuilder

            return AscendSFAKVOffloadMetadataBuilder
        from vllm_ascend.attention.context_parallel.sfa_cp import resolve_sfa_metadata_builder

        return resolve_sfa_metadata_builder()

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_impl_cls() -> type["AscendSFAImpl"]:
        if get_ascend_config().sparse_kv_offload_config.enabled:
            from vllm_ascend.attention.sfa_kv_offload import AscendSFAKVOffloadImpl

            return AscendSFAKVOffloadImpl
        from vllm_ascend.attention.context_parallel.sfa_cp import resolve_sfa_impl

        return resolve_sfa_impl(get_current_vllm_config())

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
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
    seq_lens_cpu: torch.Tensor
    cum_query_lens: torch.Tensor
    block_table: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.
    pcp_slot_mapping: torch.Tensor | None = None
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
    group_len: torch.Tensor | None = None
    group_key_idx: torch.Tensor | None = None
    group_key_cache_idx: torch.Tensor | None = None
    # Request identity for the Sparse KV offload resident LRU; only populated
    # by AscendSFAKVOffloadMetadataBuilder.
    req_ids_tensor: torch.Tensor | None = None
    token_to_req: torch.Tensor | None = None


M = TypeVar("M", bound=AscendSFAMetadata)


@dataclass
class SFAForwardContext:
    """Parallel-layout inputs consumed by the shared SFA forward template."""

    actual_seq_lengths_query: torch.Tensor
    actual_seq_lengths_key: torch.Tensor
    kv_slot_mapping: torch.Tensor
    topk_num_tokens: int
    gather_full_o_proj: bool = False


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

    def _prepare_parallel_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        cum_query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        draft_index: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Customize metadata tensors for a parallel SFA layout."""
        return cos, sin, slot_mapping, {}

    def _update_parallel_slot_mapping(
        self,
        metadata: AscendSFAMetadata,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
    ) -> None:
        """Update optional parallel metadata after an outer layout wrapper."""
        return

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
        return self._build_with_metadata_view(
            common_attn_metadata,
            lambda: self._build(common_attn_metadata, draft_index=None),
        )

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
        else:
            seq_lens_cpu = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        cos, sin = get_cos_and_sin_mla(input_positions, use_cache=(draft_index is None))

        cos, sin, slot_mapping, parallel_metadata = self._prepare_parallel_metadata(
            common_attn_metadata,
            cos,
            sin,
            slot_mapping,
            cum_query_lens,
            seq_lens,
            draft_index,
        )

        if get_ascend_config().c8_enable_reshape_optim:
            torch.ops._C_ascend.store_kv_block_metadata(
                slot_mapping,
                common_attn_metadata.group_len,
                common_attn_metadata.group_key_idx,
                common_attn_metadata.group_key_cache_idx,
                block_size,
            )

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            cum_query_lens=cum_query_lens,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            slot_mapping=slot_mapping,
            pcp_slot_mapping=pcp_slot_mapping,
            head_dim=self.model_config.get_head_size(),
            attn_mask=self.attn_mask_builder.get_attention_mask(common_attn_metadata.causal, self.model_config),
            attn_state=common_attn_metadata.attn_state,
            block_table=block_table,
            sin=sin[:num_input_tokens],
            cos=cos[:num_input_tokens],
            block_size=block_size,
            group_len=common_attn_metadata.group_len,
            group_key_idx=common_attn_metadata.group_key_idx,
            group_key_cache_idx=common_attn_metadata.group_key_cache_idx,
            **parallel_metadata,
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

    # q_hadamard and k_hadamard tensor shared when dsa c8 enabled
    q_hadamard: torch.Tensor | None = None
    k_hadamard: torch.Tensor | None = None

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
        self.kv_a_proj_with_mqa = kwargs.get("kv_a_proj_with_mqa")
        self.kv_a_layernorm = kwargs.get("kv_a_layernorm")
        self.q_a_layernorm = kwargs.get("q_a_layernorm")
        self.tp_size = get_tensor_model_parallel_world_size()
        self.skip_topk = kwargs.get("skip_topk", False)
        self.topk_indices_buffer = kwargs.get("topk_indices_buffer")

        ascend_config = get_ascend_config()
        self.vllm_config = get_current_vllm_config()
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
        # indexer param
        if self.has_indexer:
            self.n_head: int = self.indexer.n_head  # 64
            self.head_dim: int = self.indexer.head_dim  # 128
            self.wq_b = self.indexer.wq_b
            self.wk_weights_proj = self.indexer.wk_weights_proj
            self.k_norm = self.indexer.k_norm
        else:
            self.n_head = getattr(hf_config, "index_n_heads", 0)
            self.head_dim = getattr(hf_config, "index_head_dim", 0)
            self.wq_b = None
            self.wk_weights_proj = None
            self.k_norm = None
        self.is_rope_neox_style = True
        self.use_torch_npu_lightning_indexer = False
        if self.vllm_config.model_config.hf_config.model_type in ["glm_moe_dsa"]:
            self.is_rope_neox_style = False
            self.use_torch_npu_lightning_indexer = True

        # Sparse C8 has two independent meanings in SFA:
        # - SFA packed KV cache for npu_kv_quant_sparse_flash_attention.
        # - C8 indexer cache for lightning indexer.
        # The user-facing switches control these layouts independently. LI C8
        # applies only to layers that own an indexer cache.
        self.enable_sparse_sfa_c8 = ascend_config.enable_sparse_sfa_c8
        self.enable_sparse_li_c8 = self.has_indexer and ascend_config.is_sparse_li_c8_layer(self.indexer.k_cache.prefix)
        if self.enable_sparse_sfa_c8 or self.enable_sparse_li_c8:
            if get_ascend_device_type() == AscendDeviceType.A5:
                self.c8_k_cache_dtype = torch.float8_e4m3fn
                self.c8_k_scale_cache_dtype = torch.float32
            else:
                self.c8_k_cache_dtype = torch.int8
                self.c8_k_scale_cache_dtype = torch.float16

        if self.enable_sparse_sfa_c8:
            self.sfa_qsfa_packed_kv_head_dim = get_sfa_qsfa_packed_head_dim(
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                self.sfa_qsfa_tile_size,
            )
        self.preprocess_type = PreprocessType.NATIVE

        self.enable_mlapo = bool(get_ascend_config().enable_mlapo)

        self.enable_sp = enable_sp()

    @property
    def kv_cache_indexer_k_idx(self) -> int:
        """Index of the indexer key cache in the KV cache tuple.

        When sparse C8 packs the SFA KV cache into a single tensor, the indexer
        key cache moves from slot 2 to slot 1:

        ================  =========  =========  =============  ==============
        Layout            kv_cache[0]  kv_cache[1]  kv_cache[2]  kv_cache[3]
        ================  =========  =========  =============  ==============
        Default           k_nope     k_pe       indexer_k      indexer_scale
        Sparse C8         packed_kv  indexer_k  indexer_scale  (unused)
        ================  =========  =========  =============  ==============
        """
        return 1 if self.enable_sparse_sfa_c8 else 2

    @property
    def kv_cache_indexer_scale_idx(self) -> int:
        """Index of the indexer scale cache in the KV cache tuple."""
        return 2 if self.enable_sparse_sfa_c8 else 3

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

        # NOTE: When we make a incontiguous weight contiguous, a new address will be allocated for the weight,
        # in graph + RL scenario, we only capture the graph once, and the weight address is expected to be the same
        # across iterations, so we need to copy the weight to the original address after making it contiguous.
        if not hasattr(self, "W_UV"):
            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1).contiguous()
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0).contiguous()
        else:
            self.W_UV.copy_(W_UV.transpose(0, 1).contiguous())
            self.W_UK_T.copy_(W_UK.permute(1, 2, 0).contiguous())

        # TODO(zzzzwwjj): Currently, torch.ops._C_ascend.batch_matmul_transpose cannot support weight nz
        # self.W_UV = maybe_trans_nz(self.W_UV)

        # Dispose kv_b_proj since it is replaced by W_UV and W_UK_T to save memory
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

        if self.has_indexer and self.enable_sparse_li_c8 and AscendSFAImpl.q_hadamard is None:
            AscendSFAImpl.q_hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu") / (
                128**0.5
            )
        if self.has_indexer and self.enable_sparse_li_c8 and AscendSFAImpl.k_hadamard is None:
            AscendSFAImpl.k_hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu") / (
                128**0.5
            )

    @staticmethod
    def _get_layer_quant_method(layer: torch.nn.Module | None):
        return getattr(getattr(layer, "quant_method", None), "quant_method", None)

    def _resolve_preprocess_type(self, act_dtype: torch.dtype) -> PreprocessType:
        quant_method = self._get_layer_quant_method(self.fused_qkv_a_proj)
        self._quant_type = type(quant_method) if quant_method is not None else None
        qt = self._quant_type

        if self.is_kv_consumer and (
            (qt is AscendW8A8DynamicLinearMethod and self.enable_sparse_sfa_c8)
            or qt is AscendW8A8MXFP8DynamicLinearMethod
            or qt is None
        ):
            if self._try_enable_type(PreprocessType.PROLOG_V3, act_dtype):
                return PreprocessType.PROLOG_V3

        if qt is AscendW8A8LinearMethod and self.enable_mlapo:
            if self._try_enable_type(PreprocessType.MLAPO, act_dtype):
                return PreprocessType.MLAPO

        return PreprocessType.NATIVE

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
        if self.kv_a_layernorm is None or self.q_a_layernorm is None:
            reasons.append("Fused preprocessing requires q_a_layernorm and kv_a_layernorm.")
        if self.fused_qkv_a_proj is None:
            reasons.append("fused_qkv_a_proj is None, mlapo is disabled.")

        if pp_type is PreprocessType.PROLOG_V3:
            if self.is_kv_producer:
                reasons.append("PROLOG_V3 is disabled on KV producer workers.")
            if self._quant_type is None and self.enable_sparse_sfa_c8:
                reasons.append("PROLOG_V3: C8 sparse requires quantized MLAPO.")
            if getattr(self.q_proj, "_chunk_size", 0):
                reasons.append("PROLOG_V3 does not support chunked q_proj weights yet.")
        elif pp_type is PreprocessType.MLAPO:
            if self.enable_sparse_sfa_c8:
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

        if self.is_kv_consumer:
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
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple,
        slots: torch.Tensor,
        attn_metadata: M,
    ):
        B = kv_no_split.shape[0]
        N = self.num_kv_heads
        S = 1
        # npu_kv_rmsnorm_rope_cache needs [B, N, S, D]
        kv_no_split = kv_no_split.view(B, N, S, self.kv_lora_rank + self.qk_rope_head_dim)
        cache_mode = "PA"

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
                dst_type=self.c8_k_cache_dtype,
                tile_size=self.sfa_qsfa_tile_size,
            )

        torch_npu.npu_kv_rmsnorm_rope_cache(
            kv_no_split,
            self.kv_a_layernorm.weight,  # type: ignore[union-attr]
            cos,
            sin,
            slots.to(torch.int64),
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

        # Convert from (B, N, P) to (N, B, P)
        q_nope = q_nope.transpose(0, 1)
        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        ql_nope = torch.bmm(q_nope, self.W_UK_T)
        # Convert from (N, B, L) to (B, N, L)
        return ql_nope.transpose(0, 1), q_pe

    def _v_up_proj(self, x):
        num_input_tokens, _, _ = x.shape
        if (
            x.dtype in [torch.float16, torch.bfloat16]
            and hasattr(torch.ops._C_ascend, "batch_matmul_transpose")
            and num_input_tokens <= BMM_TRANS_MAX_SUPPORTED_TOKENS
        ):
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank)
            res = torch.empty((num_input_tokens, self.local_num_heads, self.v_head_dim), dtype=x.dtype, device=x.device)
            torch.ops._C_ascend.batch_matmul_transpose(x, self.W_UV, res)
            x = res.reshape(-1, self.local_num_heads * self.v_head_dim)
        elif hasattr(torch_npu, "npu_transpose_batchmatmul"):
            # Convert from (N, B, L)/(N, B, 1, L) to (N, B, L)
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank)
            # Multiply (N, B, L) x (N, L, V) -> (B, N, V)
            x = torch_npu.npu_transpose_batchmatmul(x, self.W_UV, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
            # Convert from (N, B, V) to (B, N * V)
            x = x.reshape(-1, self.local_num_heads * self.v_head_dim)
        else:
            # Convert from (B, N, L) to (N, B, L)
            x = x.view(-1, self.local_num_heads, self.kv_lora_rank).transpose(0, 1)
            # # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            x = torch.bmm(x, self.W_UV)
            # # Convert from (N, B, V) to (B, N * V)
            x = x.transpose(0, 1).reshape(-1, self.local_num_heads * self.v_head_dim)
        return x

    def _sfa_preprocess_prolog_v3(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ]:
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

    def indexer_select_pre_process(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        if not self.has_indexer:
            raise RuntimeError(
                f"indexer_select_pre_process should not be called when indexer is None. layer_name={self.layer_name}."
            )

        assert self.wk_weights_proj is not None
        assert self.k_norm is not None

        kw, _ = self.wk_weights_proj(x)
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
            k_li = k_li @ AscendSFAImpl.k_hadamard
            k_li, k_li_scale = torch_npu.npu_dynamic_quant(k_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
            k_li_scale = k_li_scale.to(self.c8_k_scale_cache_dtype)  # [b*s,]
            k_li_scale = k_li_scale.unsqueeze(-1)  # [b*s,1]
        else:
            k_li_scale = None

        return k_li, k_li_scale

    def indexer_select_post_process(
        self,
        x: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        cos: torch.Tensor,
        sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ):
        if not self.has_indexer:
            raise RuntimeError(
                f"indexer_select_post_process should not be called when indexer is None. layer_name={self.layer_name}."
            )

        assert self.wk_weights_proj is not None
        assert self.wq_b is not None

        kw, _ = self.wk_weights_proj(x)
        weights = kw[:, self.head_dim :]
        if isinstance(q_c, tuple):
            q_c_tensor, q_c_scale = q_c
            q_c_tensor = q_c_tensor.view(-1, q_c_tensor.shape[-1])
            quant_matmul_kwargs = dict(
                bias=None,
                output_dtype=x.dtype,
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
            q_li = q_li @ AscendSFAImpl.q_hadamard
            q_li, q_li_scale = torch_npu.npu_dynamic_quant(q_li.view(-1, self.head_dim), dst_type=self.c8_k_cache_dtype)
            q_li_scale = q_li_scale.to(self.c8_k_scale_cache_dtype)  # [b*s,]

        return DeviceOperator.indexer_select_post_process(
            self,
            q_li,
            q_li_scale,
            q_li_shape_ori,
            weights,
            kv_cache,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
            self.enable_sparse_li_c8,
            self.use_torch_npu_lightning_indexer,
        )

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

    def _use_li_c8_reshape_optim(self) -> bool:
        """Whether this layer can use the LI C8 cache-write operator."""
        return self.enable_sparse_li_c8 and get_ascend_config().c8_enable_reshape_optim

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
        return DeviceOperator.execute_sparse_flash_attention_process(
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

    def _record_query_gather_context(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        attn_metadata: M,
    ) -> None:
        return

    def _parallel_query_gather_dim(self) -> int:
        """Dimension restored by an outer DCP query gather."""
        return 1

    def _prepare_kv_for_parallel(
        self,
        k_pe: torch.Tensor | None,
        k_nope: torch.Tensor | None,
        knope_scale: torch.Tensor | None,
        k_li: torch.Tensor | None,
        k_li_scale: torch.Tensor | None,
        full_gather_o_proj_enabled: bool,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        list[torch.distributed.Work],
    ]:
        """Prepare native KV tensors for an optional parallel layout."""
        return k_li, k_li_scale, None, []

    def _store_parallel_kv(
        self,
        k_pe: torch.Tensor | None,
        k_nope: torch.Tensor | None,
        knope_scale: torch.Tensor | None,
        k_li: torch.Tensor | None,
        fused_kv_no_split: torch.Tensor | None,
        kv_ag_handles: list[torch.distributed.Work],
        kv_cache: tuple[torch.Tensor, ...] | None,
        slot_mapping_sfa: torch.Tensor,
        attn_metadata: M,
        full_gather_o_proj_enabled: bool,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Store KV produced by native preprocessing."""
        if self.enable_sparse_sfa_c8:
            assert k_pe is not None
            assert k_nope is not None
            assert knope_scale is not None
            packed_kv = torch.cat(
                [
                    k_nope.view(-1, k_nope.shape[-1]),
                    k_pe.view(-1, k_pe.shape[-1]),
                    knope_scale.view(-1, knope_scale.shape[-1]),
                ],
                dim=-1,
            )
            packed_head_dim = self.sfa_qsfa_packed_kv_head_dim
            assert packed_kv.shape[-1] == packed_head_dim
            assert kv_cache is not None
            torch_npu.npu_scatter_nd_update_(
                kv_cache[0].view(-1, packed_head_dim),
                slot_mapping_sfa.view(-1, 1),
                packed_kv.view(-1, packed_head_dim),
            )

        return k_pe, k_nope, k_li

    def _get_parallel_forward_context(
        self,
        attn_metadata: M,
        num_input_tokens: int,
        hidden_states: torch.Tensor,
    ) -> SFAForwardContext:
        return SFAForwardContext(
            actual_seq_lengths_query=attn_metadata.cum_query_lens,
            actual_seq_lengths_key=attn_metadata.seq_lens,
            kv_slot_mapping=self._get_sfa_kv_slot_mapping(attn_metadata),
            topk_num_tokens=num_input_tokens or hidden_states.shape[0],
        )

    def _prepare_native_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: M,
    ) -> torch.Tensor:
        return hidden_states

    def _finalize_o_proj(
        self,
        attn_output: torch.Tensor,
        output: torch.Tensor,
        gather_full_o_proj: bool,
    ) -> torch.Tensor:
        output[...] = self.o_proj(attn_output)[0]
        return output

    def _get_sfa_kv_slot_mapping(
        self,
        attn_metadata: M,
    ) -> torch.Tensor:
        return attn_metadata.slot_mapping

    def _compose_sfa_kv_cache(self, kv_cache) -> tuple[torch.Tensor, ...] | None:
        """Compose split cache handles into the tuple expected by SFA kernels.

        ``kv_cache`` contains only the main MLA cache owned by the attention
        layer, while ``self.indexer.k_cache.kv_cache`` contains the cache owned
        by the indexer layer. Their possible layouts are:

        - neither cache uses C8:
          main ``(k_cache, v_cache)`` + indexer ``(indexer_k_cache,)``
          -> ``(k_cache, v_cache, indexer_k_cache)``
        - SFA C8 only:
          main ``(packed_kv_cache,)`` + indexer ``(indexer_k_cache,)``
          -> ``(packed_kv_cache, indexer_k_cache)``
        - LI C8 only:
          main ``(k_cache, v_cache)`` +
          indexer ``(indexer_k_cache, indexer_scale_cache)``
          -> ``(k_cache, v_cache, indexer_k_cache, indexer_scale_cache)``
        - both caches use C8:
          main ``(packed_kv_cache,)`` +
          indexer ``(indexer_k_cache, indexer_scale_cache)``
          -> ``(packed_kv_cache, indexer_k_cache, indexer_scale_cache)``

        Layers that reuse another layer's top-k indices have no local indexer;
        for those layers, the main cache tuple is returned unchanged.
        """
        # TODO: Remove this recomposition once SFA kernels accept split
        # main/indexer cache handles directly. The allocator now owns them as
        # separate cache specs, while the current kernel path still expects the
        # legacy combined tuple layout.
        main_cache = kv_cache
        if main_cache is None or not self.has_indexer:
            return main_cache

        # Sparse KV offload registers the main MLA cache as a 6-tuple
        # (k_npu, v_npu, k_cpu, v_cpu, topk_buffer_k, topk_buffer_v); the
        # attention kernels only consume the leading NPU pair.
        if len(main_cache) == OFFLOAD_KV_CACHE_TUPLE_LEN:
            main_cache = (main_cache[OFFLOAD_K_CACHE_NPU_INDEX], main_cache[OFFLOAD_V_CACHE_NPU_INDEX])

        indexer_cache = self.indexer.k_cache.kv_cache
        if indexer_cache is None:
            raise RuntimeError(f"SFA indexer cache is not initialized or bound. layer_name={self.layer_name}.")

        expected_main_tensors = 1 if self.enable_sparse_sfa_c8 else 2
        if len(main_cache) != expected_main_tensors:
            raise RuntimeError(
                f"SFA main cache expects {expected_main_tensors} tensor(s), "
                f"got {len(main_cache)} for layer_name={self.layer_name}."
            )

        expected_indexer_tensors = 2 if self.enable_sparse_li_c8 else 1
        if len(indexer_cache) != expected_indexer_tensors:
            raise RuntimeError(
                f"SFA indexer cache expects {expected_indexer_tensors} tensor(s), "
                f"got {len(indexer_cache)} for layer_name={self.layer_name}."
            )
        return (*main_cache, *indexer_cache)

    def _write_indexer_cache(
        self,
        k_li: torch.Tensor,
        k_li_scale: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        kv_cache: tuple,
        attn_metadata: M,
    ) -> None:
        dsa_k_cache_idx = self.kv_cache_indexer_k_idx
        dsa_k_scale_cache_idx = self.kv_cache_indexer_scale_idx

        use_li_c8_reshape_optim = self._use_li_c8_reshape_optim()
        if use_li_c8_reshape_optim:
            torch.ops._C_ascend.store_kv_block(
                k_li,
                kv_cache[dsa_k_cache_idx],
                attn_metadata.group_len,
                attn_metadata.group_key_idx,
                attn_metadata.group_key_cache_idx,
                attn_metadata.block_size,
            )
        else:
            torch_npu.npu_scatter_nd_update_(
                kv_cache[dsa_k_cache_idx].view(-1, k_li.shape[-1]),
                slot_mapping.view(-1, 1),
                k_li.view(-1, k_li.shape[-1]),
            )  # b, s, n, d
        if self.enable_sparse_li_c8:
            assert len(kv_cache) == (3 if self.enable_sparse_sfa_c8 else 4)
            assert k_li_scale is not None
            if use_li_c8_reshape_optim:
                torch.ops._C_ascend.store_kv_block(
                    k_li_scale,
                    kv_cache[dsa_k_scale_cache_idx],
                    attn_metadata.group_len,
                    attn_metadata.group_key_idx,
                    attn_metadata.group_key_cache_idx,
                    attn_metadata.block_size,
                )
            else:
                torch_npu.npu_scatter_nd_update_(
                    kv_cache[dsa_k_scale_cache_idx].view(-1, k_li_scale.shape[-1]),
                    slot_mapping.view(-1, 1),
                    k_li_scale.view(-1, k_li_scale.shape[-1]),
                )

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: M,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        composed_kv_cache = self._compose_sfa_kv_cache(kv_cache)
        assert composed_kv_cache is not None
        kv_cache = composed_kv_cache

        cos = attn_metadata.cos
        sin = attn_metadata.sin
        slot_mapping_li = attn_metadata.slot_mapping
        slot_mapping_sfa = self._get_sfa_kv_slot_mapping(attn_metadata)

        # Inputs and outputs may be padded for CUDA graphs
        num_input_tokens = attn_metadata.num_input_tokens
        parallel_context = self._get_parallel_forward_context(
            attn_metadata,
            num_input_tokens,
            hidden_states,
        )
        actual_seq_lengths_query = parallel_context.actual_seq_lengths_query
        actual_seq_lengths_key = parallel_context.actual_seq_lengths_key

        fused_type: PreprocessType = self.preprocess_type
        if (
            attn_metadata.attn_state not in (AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding)
            or self.preprocess_type == PreprocessType.MLAPO
            and num_input_tokens > MLAPO_MAX_SUPPORTED_TOKENS
        ):
            fused_type = PreprocessType.NATIVE

        if fused_type != PreprocessType.NATIVE:
            if fused_type == PreprocessType.PROLOG_V3:
                assert slot_mapping_sfa.numel() == hidden_states.shape[0], (
                    "SFA Prolog V3 requires one cache index per input token, "
                    f"got token_x={hidden_states.shape[0]} and cache_index={slot_mapping_sfa.numel()}."
                )
            if self.has_indexer:
                k_li, k_li_scale = self.indexer_select_pre_process(x=hidden_states, cos=cos, sin=sin)
            else:
                k_li, k_li_scale = None, None
            wait_for_kv_layer_from_connector(layer_name)

            if fused_type == PreprocessType.PROLOG_V3:
                hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_prolog_v3(
                    hidden_states=hidden_states,
                    kv_cache=kv_cache,
                    cos=cos,
                    sin=sin,
                    slot_mapping=slot_mapping_sfa,
                )
            else:
                hidden_states, ql_nope, q_pe, q_c = self._sfa_preprocess_mlapo(
                    hidden_states=hidden_states,
                    kv_cache=kv_cache,
                    cos=cos,
                    sin=sin,
                    slot_mapping=slot_mapping_sfa,
                    num_input_tokens=num_input_tokens,
                )
        # native
        else:
            assert self.fused_qkv_a_proj is not None, "q lora is required for DSA."
            hidden_states = self._prepare_native_hidden_states(hidden_states, attn_metadata)
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_no_split = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized"
            q_c = self.q_a_layernorm(q_c)

            if self.has_indexer:
                k_li, k_li_scale = self.indexer_select_pre_process(
                    x=hidden_states,
                    cos=cos,
                    sin=sin,
                )
            else:
                k_li, k_li_scale = None, None

            wait_for_kv_layer_from_connector(layer_name)

            kv_outputs = self.exec_kv(
                kv_no_split,
                cos,
                sin,
                kv_cache,
                parallel_context.kv_slot_mapping,
                attn_metadata,
            )
            k_pe, k_nope = kv_outputs[:2]
            knope_scale = kv_outputs[2] if len(kv_outputs) == 3 else None
            k_li, k_li_scale, fused_kv_no_split, kv_ag_handles = self._prepare_kv_for_parallel(
                k_pe,
                k_nope,
                knope_scale,
                k_li,
                k_li_scale,
                parallel_context.gather_full_o_proj,
            )

            ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
            q_pe = self.rope_single(q_pe, cos, sin)
            self._record_query_gather_context(
                ql_nope,
                q_pe,
                attn_metadata,
            )

            (
                k_pe,
                k_nope,
                k_li,
            ) = self._store_parallel_kv(
                k_pe,
                k_nope,
                knope_scale,
                k_li,
                fused_kv_no_split,
                kv_ag_handles,
                kv_cache,
                self._get_sfa_kv_slot_mapping(attn_metadata),
                attn_metadata,
                parallel_context.gather_full_o_proj,
            )

        if self.has_indexer:
            assert k_li is not None
            if self.vllm_config.parallel_config.prefill_context_parallel_size > 1:
                assert attn_metadata.pcp_slot_mapping is not None
                indexer_cache_slot_mapping = attn_metadata.pcp_slot_mapping
            else:
                indexer_cache_slot_mapping = slot_mapping_li
            self._write_indexer_cache(
                k_li,
                k_li_scale,
                indexer_cache_slot_mapping,
                kv_cache,
                attn_metadata,
            )

        # Notify for every layer that wrote the cache, not just indexer layers:
        # by this point all of the layer's KV (main + indexer) has been
        # scattered, so the connector can dispatch the PD pull immediately.
        notify_kv_cache_written(self.layer_name or "")

        # Open the prefetch gate for every SFA layer. Some GLM-5.2 layers
        # reuse cached top-k indices and have no indexer, so recording this
        # inside indexer_select_post_process would leave their gate closed.
        record_attention_compute_start()

        if self.skip_topk:
            topk_indices = self._get_indexcache_topk_indices(parallel_context.topk_num_tokens)
        else:
            if not self.has_indexer:
                raise RuntimeError(f"skip_topk is False but indexer is None. layer_name={self.layer_name}.")
            assert q_c is not None
            topk_indices = self.indexer_select_post_process(
                x=hidden_states,
                q_c=q_c,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                cos=cos,
                sin=sin,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
            )
            if self.use_index_cache:
                self._update_indexcache_topk_indices(topk_indices)

        attn_output = self._execute_sparse_flash_attention_process(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            attn_metadata,
            actual_seq_lengths_query,
            actual_seq_lengths_key,
        )

        attn_output = self._v_up_proj(attn_output)

        output = self._finalize_o_proj(
            attn_output,
            output,
            parallel_context.gather_full_o_proj,
        )

        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

        return output


def custom_kv_rmsnorm_rope(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    *,
    epsilon: float = 1e-5,
    dst_type: torch.dtype | int = torch.float8_e4m3fn,
    tile_size: int = SFA_QSFA_TILE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rms_in, rope_in = kv.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
    k_nope, _ = torch_npu.npu_rms_norm(rms_in, gamma, epsilon=epsilon)
    k_rope = torch_npu.npu_interleave_rope(rope_in, cos, sin)

    prefix_shape = k_nope.shape[:-1]
    k_nope, knope_scale = torch_npu.npu_dynamic_block_quant(
        k_nope.contiguous().view(-1, 1, kv_lora_rank),
        dst_type=dst_type,
        row_block_size=1,
        col_block_size=tile_size,
    )
    if dst_type == torch.int8:
        # Return byte views so the caller can concatenate all three components.
        return (
            k_rope.contiguous().view(torch.int8),
            k_nope.view(*prefix_shape, kv_lora_rank),
            knope_scale.to(torch.float32).view(*prefix_shape, -1).contiguous().view(torch.int8),
        )

    # A5 transports the BF16 rope and scale bytes through FP8-typed tensors.
    return (
        k_rope.view(torch.float8_e4m3fn),
        k_nope,
        knope_scale.view(knope_scale.shape[0], -1).view(torch.float8_e4m3fn),
    )
