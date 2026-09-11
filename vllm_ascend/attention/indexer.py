from dataclasses import dataclass
from typing import Any

import scipy  # type: ignore
import torch
import torch_npu
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tp_group
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.context_parallel.sfa_dcp_utils import (
    build_sfa_dcp_replicated_block_table,
    build_sfa_dcp_replicated_slot_mapping,
    get_sfa_dcp_local_block_table,
    get_sfa_dcp_max_local_block_table_cols,
)
from vllm_ascend.attention.utils import split_decodes_and_prefills
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
from vllm_ascend.utils import _round_up, enable_dsa_cp, enable_sfa_dcp_replicated_indexer, vllm_version_is

if vllm_version_is("0.28.0"):
    from vllm.model_executor.layers.attention.pcp import _gather_prefill_cache_inputs  # type: ignore[import-not-found]
else:
    from vllm.v1.attention.ops.pcp import _gather_prefill_cache_inputs  # type: ignore[import-not-found]

# Slots of the k / scale caches inside an indexer's own ``k_cache.kv_cache``
# tuple (the scale slot exists only when LI C8 is enabled).
INDEXER_K_CACHE_SLOT = 0
INDEXER_SCALE_CACHE_SLOT = 1


@dataclass
class AscendSFAIndexerMetadata:
    """Engine-side metadata owned by an SFA indexer cache layer.

    Carries everything the indexer kernels need from the engine: the paged
    cache view (block table, slot mapping), the rope tables, the parallel
    sequence lengths, and the LI C8 reshape-optim fields. All fields are
    derived from common attention metadata by the indexer's own builder.
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
    # Parallel-layout sequence lengths consumed by the top-k kernel. Base/PCP
    # modes use the unsharded values; DSA-CP uses rank-local lengths.
    actual_seq_lengths_query: torch.Tensor | None = None
    actual_seq_lengths_key: torch.Tensor | None = None
    # The PCP cache-write gather splits the local prefill region on this
    # independently computed decode-token count.
    num_decode_tokens: int = 0


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
    LI C8 reshape-optim fields, RoPE, and parallel-layout values). SFA only
    consumes the completed indexer metadata during its forward.

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
        if self._pcp_active:
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
        k_hidden_states: torch.Tensor,
        indexer_metadata: AscendSFAIndexerMetadata,
        compute_topk: bool = True,
    ) -> torch.Tensor | None:
        """Full indexer pipeline: k path -> cache write -> top-k selection.

        The k path output is persisted first because the selection kernel
        reads the freshly written cache. ``compute_topk=False`` (SFA layers
        sharing top-k indices) still runs the k path and the write so the
        cache stays up to date, and returns None.

        The indexer metadata owns its RoPE tables and parallel-layout values;
        DSA-CP metadata contains the local token shard that matches these
        inputs."""
        cos = indexer_metadata.cos
        sin = indexer_metadata.sin
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
    but owns its physical cache and constructs its metadata independently.
    Under DCP it expands the local common block table and slot mapping into
    the indexer's replicated address space directly; it never reads metadata
    built for the SFA attention layer. The slot mapping is emitted write-ready
    for the active parallel mode (full gather mapping under PCP).
    """

    reorder_batch_threshold = None
    consumes_pcp_context = True

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
        scheduler_config = vllm_config.scheduler_config
        self.decode_threshold = 1
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None:
            self.decode_threshold += speculative_config.num_speculative_tokens

        self._pcp_active = vllm_config.parallel_config.prefill_context_parallel_size > 1
        self._dsa_cp_active = enable_dsa_cp()
        self._group_metadata_buffers: dict[
            object,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._rope_buffers: dict[object, tuple[torch.Tensor, torch.Tensor]] = {}
        self._dsa_cp_rope_buffers: dict[object, tuple[torch.Tensor, torch.Tensor]] = {}
        self._dsa_cp_slot_mapping_buffers: dict[object, torch.Tensor] = {}
        self._dsa_cp_seq_buffers: dict[object, tuple[torch.Tensor, torch.Tensor]] = {}
        self._dcp_block_table_buffers: dict[object, torch.Tensor] = {}
        self._dcp_slot_mapping_buffers: dict[object, torch.Tensor] = {}
        self._pcp_indexer_slot_mapping_buffers: dict[object, torch.Tensor] = {}
        max_num_input_tokens = scheduler_config.max_num_batched_tokens
        self._rope_capacity = max_num_input_tokens
        pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self._dcp_slot_capacity = max_num_input_tokens * pcp_size
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        self._slot_capacity = max(
            max_num_input_tokens * pcp_size,
            _round_up(max_num_input_tokens, tp_size),
        )

        if self._dsa_cp_active and not self._pcp_active:
            self.dsa_cp_world_size = tp_size
            self.dsa_cp_slot_capacity = _round_up(
                max_num_input_tokens,
                self.dsa_cp_world_size,
            )
            max_num_reqs = scheduler_config.max_num_seqs
            self.dsa_cp_seq_capacity = max_num_reqs + 1
            if speculative_config is not None:
                spec_tokens = speculative_config.num_speculative_tokens
                self.dsa_cp_seq_capacity = max(
                    self.dsa_cp_seq_capacity,
                    max_num_reqs * (spec_tokens + 1) + 1,
                )
        self._dcp_active = enable_sfa_dcp_replicated_indexer(vllm_config)
        if not self._dcp_active:
            return

        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.replicated_view_block_size = self.kernel_block_size
        if kv_cache_spec.block_size % self.replicated_view_block_size != 0:
            raise RuntimeError(
                "SFA replicated indexer metadata requires the physical block "
                f"size ({kv_cache_spec.block_size}) to be divisible by the "
                f"kernel block size ({self.replicated_view_block_size})."
            )
        self.blocks_per_phys_block = kv_cache_spec.block_size // self.replicated_view_block_size

        max_num_reqs = scheduler_config.max_num_seqs
        if self._pcp_active:
            max_num_reqs *= 2
        max_num_reqs += 1
        self.max_local_block_table_cols = get_sfa_dcp_max_local_block_table_cols(
            vllm_config.model_config.max_model_len,
            kv_cache_spec.block_size,
            self.dcp_size,
            self.blocks_per_phys_block,
        )
        max_replicated_block_table_cols = self.max_local_block_table_cols * self.dcp_size
        self._dcp_block_table_shape = (max_num_reqs, max_replicated_block_table_cols)
        self.replicated_col_idx_buf = torch.arange(
            max_replicated_block_table_cols,
            dtype=torch.int32,
            device=device,
        )
        if self._pcp_active:
            self._pcp_indexer_slot_capacity = (
                max_num_input_tokens * vllm_config.parallel_config.prefill_context_parallel_size
            )

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def _ensure_replicated_view_buffers(
        self,
        num_reqs: int,
        num_input_tokens: int,
        local_block_table_cols: int,
        buffer_key: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_table_cols = local_block_table_cols * self.dcp_size
        block_table_buffer = self._dcp_block_table_buffers.get(buffer_key)
        if block_table_buffer is None:
            block_table_buffer = torch.empty(
                self._dcp_block_table_shape,
                dtype=torch.int32,
                device=self.device,
            )
            self._dcp_block_table_buffers[buffer_key] = block_table_buffer
        slot_mapping_buffer = self._dcp_slot_mapping_buffers.get(buffer_key)
        if slot_mapping_buffer is None:
            slot_mapping_buffer = torch.empty(
                self._dcp_slot_capacity,
                dtype=torch.int32,
                device=self.device,
            )
            self._dcp_slot_mapping_buffers[buffer_key] = slot_mapping_buffer
        if (
            block_table_buffer.shape[0] < num_reqs
            or block_table_buffer.shape[1] < block_table_cols
        ):
            raise RuntimeError(
                "Replicated indexer metadata buffer is too small: "
                f"block_table_shape={block_table_buffer.shape}, "
                f"num_reqs={num_reqs}, block_table_cols={block_table_cols}."
            )
        if slot_mapping_buffer.shape[0] < num_input_tokens:
            raise RuntimeError(
                "Replicated indexer metadata buffer is too small: "
                f"slot_mapping_shape={slot_mapping_buffer.shape}, "
                f"num_input_tokens={num_input_tokens}."
            )
        return (
            block_table_buffer[:num_reqs, :block_table_cols],
            self.replicated_col_idx_buf[:block_table_cols],
            slot_mapping_buffer[:num_input_tokens],
        )

    def _build_block_table_replicated_view(
        self,
        dcp_block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        buffer_key: object,
    ) -> torch.Tensor:
        num_reqs, local_block_table_cols = dcp_block_table.shape
        block_table, replicated_col_idx, _ = self._ensure_replicated_view_buffers(
            num_reqs,
            0,
            local_block_table_cols,
            buffer_key,
        )
        return build_sfa_dcp_replicated_block_table(
            dcp_block_table,
            seq_lens,
            block_table,
            replicated_col_idx,
            self.dcp_size,
            self.blocks_per_phys_block,
        )

    def _build_slot_mapping_replicated_view(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        block_table_replicated_view: torch.Tensor,
        buffer_key: object,
    ) -> torch.Tensor:
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        local_block_table_cols = block_table_replicated_view.shape[1] // self.dcp_size
        _, _, slot_mapping = self._ensure_replicated_view_buffers(
            num_reqs,
            num_input_tokens,
            local_block_table_cols,
            buffer_key,
        )
        return build_sfa_dcp_replicated_slot_mapping(
            common_attn_metadata,
            block_table_replicated_view,
            slot_mapping,
            self.replicated_view_block_size,
            self.device,
        )

    def _build_pcp_ordered_slot_mapping(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        pcp_context: Any,
        pcp_cache_group_idx: int,
        buffer_key: object,
    ) -> torch.Tensor:
        global_batch = pcp_context.global_batch
        num_reqs = global_batch.num_reqs
        global_block_table = pcp_context.global_block_tables[pcp_cache_group_idx]
        global_common_attn_metadata = common_attn_metadata.replace(
            query_start_loc=global_batch.query_start_loc,
            seq_lens=global_batch.seq_lens[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=global_batch.num_tokens,
            num_input_tokens=global_batch.num_tokens,
            positions=global_batch.positions,
            block_table_tensor=global_block_table,
        )
        dcp_block_table = get_sfa_dcp_local_block_table(
            global_block_table,
            num_reqs,
            self.max_local_block_table_cols,
        )
        replicated_block_table = self._build_block_table_replicated_view(
            dcp_block_table,
            global_common_attn_metadata.seq_lens,
            buffer_key,
        )
        global_slot_mapping = self._build_slot_mapping_replicated_view(
            global_common_attn_metadata,
            replicated_block_table,
            buffer_key,
        )

        gather_idx = pcp_context.padded_gather_idx
        write_mask = pcp_context.gathered_kv_write_mask
        if gather_idx is None or write_mask is None:
            raise RuntimeError("PCP+DCP indexer metadata requires the PCP gathered-token layout.")
        num_pcp_ordered_tokens = gather_idx.numel()
        pcp_slot_mapping_buffer = self._pcp_indexer_slot_mapping_buffers.get(buffer_key)
        if pcp_slot_mapping_buffer is None:
            pcp_slot_mapping_buffer = torch.empty(
                self._pcp_indexer_slot_capacity,
                dtype=torch.int32,
                device=self.device,
            )
            self._pcp_indexer_slot_mapping_buffers[buffer_key] = pcp_slot_mapping_buffer
        if pcp_slot_mapping_buffer.shape[0] < num_pcp_ordered_tokens:
            raise RuntimeError(
                "PCP+DCP indexer slot buffer is too small: "
                f"capacity={pcp_slot_mapping_buffer.shape[0]}, "
                f"required={num_pcp_ordered_tokens}."
            )
        pcp_slot_mapping = pcp_slot_mapping_buffer[:num_pcp_ordered_tokens]
        torch.index_select(global_slot_mapping, 0, gather_idx, out=pcp_slot_mapping)
        pcp_slot_mapping.masked_fill_(~write_mask, -1)
        return pcp_slot_mapping

    def _build_dcp_cache_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        pcp_context: Any | None,
        pcp_cache_group_idx: int | None,
        buffer_key: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pcp_slot_mapping = None
        if self._pcp_active and pcp_context is not None and bool(pcp_context.global_batch.is_prefilling_np.any()):
            if pcp_cache_group_idx is None:
                raise RuntimeError("PCP+DCP indexer metadata requires the PCP cache-group index.")
            pcp_slot_mapping = self._build_pcp_ordered_slot_mapping(
                common_attn_metadata,
                pcp_context,
                pcp_cache_group_idx,
                buffer_key,
            )

        num_reqs = common_attn_metadata.num_reqs
        dcp_block_table = get_sfa_dcp_local_block_table(
            common_attn_metadata.block_table_tensor,
            num_reqs,
            self.max_local_block_table_cols,
        )
        block_table = self._build_block_table_replicated_view(
            dcp_block_table,
            common_attn_metadata.seq_lens,
            buffer_key,
        )
        slot_mapping = self._build_slot_mapping_replicated_view(
            common_attn_metadata,
            block_table,
            buffer_key,
        )
        if pcp_slot_mapping is not None:
            slot_mapping = pcp_slot_mapping
        return block_table, slot_mapping

    def _build_dsa_cp_slot_mapping(
        self,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
        buffer_key: object,
    ) -> torch.Tensor:
        num_tokens_pad = _round_up(num_input_tokens, self.dsa_cp_world_size)
        slot_mapping_buffer = self._dsa_cp_slot_mapping_buffers.get(buffer_key)
        if slot_mapping_buffer is None:
            slot_mapping_buffer = torch.empty(
                self.dsa_cp_slot_capacity,
                dtype=torch.int32,
                device=self.device,
            )
            self._dsa_cp_slot_mapping_buffers[buffer_key] = slot_mapping_buffer
        if slot_mapping_buffer.shape[0] < num_tokens_pad:
            raise RuntimeError(
                "DSA-CP indexer slot buffer is too small: "
                f"capacity={slot_mapping_buffer.shape[0]}, "
                f"required={num_tokens_pad}."
            )
        padded_slot_mapping = slot_mapping_buffer[:num_tokens_pad]
        padded_slot_mapping.fill_(-1)
        padded_slot_mapping[: slot_mapping.shape[0]].copy_(slot_mapping)
        return padded_slot_mapping

    def _get_dsa_cp_seq_buffers(
        self,
        buffer_key: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        buffers = self._dsa_cp_seq_buffers.get(buffer_key)
        if buffers is None:
            query = torch.empty(
                self.dsa_cp_seq_capacity,
                dtype=torch.int32,
                device=self.device,
            )
            buffers = (query, torch.empty_like(query))
            self._dsa_cp_seq_buffers[buffer_key] = buffers
        return buffers

    def _build_dsa_cp_parallel_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        cos: torch.Tensor,
        sin: torch.Tensor,
        buffer_key: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = common_attn_metadata.num_input_tokens
        num_tokens_pad = _round_up(num_tokens, self.dsa_cp_world_size)
        num_tokens_per_device = num_tokens_pad // self.dsa_cp_world_size
        local_start = get_tp_group().rank_in_group * num_tokens_per_device
        local_end = local_start + num_tokens_per_device

        rope_buffers = self._dsa_cp_rope_buffers.get(buffer_key)
        if rope_buffers is None:
            local_capacity = self.dsa_cp_slot_capacity // self.dsa_cp_world_size
            rope_shape = (local_capacity, *cos.shape[1:])
            rope_buffers = (
                torch.empty(rope_shape, dtype=cos.dtype, device=cos.device),
                torch.empty(rope_shape, dtype=sin.dtype, device=sin.device),
            )
            self._dsa_cp_rope_buffers[buffer_key] = rope_buffers
        local_cos_buf, local_sin_buf = rope_buffers
        if local_cos_buf.shape[0] < num_tokens_per_device:
            raise RuntimeError(
                "DSA-CP indexer RoPE buffer is too small: "
                f"capacity={local_cos_buf.shape[0]}, required={num_tokens_per_device}."
            )
        local_cos = local_cos_buf[:num_tokens_per_device]
        local_sin = local_sin_buf[:num_tokens_per_device]
        local_cos.zero_()
        local_sin.zero_()
        source_end = min(local_end, cos.shape[0])
        if source_end > local_start:
            source_slice = slice(local_start, source_end)
            target_slice = slice(0, source_end - local_start)
            local_cos[target_slice].copy_(cos[source_slice])
            local_sin[target_slice].copy_(sin[source_slice])

        cum_query_lens = common_attn_metadata.query_start_loc[1 : common_attn_metadata.num_reqs + 1]
        seq_lens = common_attn_metadata.seq_lens[: common_attn_metadata.num_reqs]
        actual_seq_lengths_query, actual_seq_lengths_key = self._get_dsa_cp_seq_buffers(buffer_key)
        num_segs = cum_query_lens.shape[0]
        if actual_seq_lengths_query.shape[0] < num_segs:
            raise RuntimeError(
                "DSA-CP indexer sequence buffer is too small: "
                f"capacity={actual_seq_lengths_query.shape[0]}, required={num_segs}."
            )
        global_start = common_attn_metadata.query_start_loc[:num_segs]
        req_local_start = global_start.clamp(min=local_start)
        req_local_end = cum_query_lens.clamp(max=local_end)
        num_local_tokens = req_local_end - req_local_start
        actual_seq_lengths_query[:num_segs] = torch.cumsum(
            num_local_tokens.clamp(min=0),
            dim=0,
        )
        offset = cum_query_lens - req_local_end
        actual_seq_lengths_key[:num_segs] = torch.where(
            num_local_tokens > 0,
            torch.clamp_min(seq_lens - offset, 0),
            0,
        )
        return (
            local_cos,
            local_sin,
            actual_seq_lengths_query[:num_segs],
            actual_seq_lengths_key[:num_segs],
        )

    def _copy_rope_to_metadata_buffers(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        buffer_key: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = cos.shape[0]
        if num_tokens > self._rope_capacity:
            raise RuntimeError(
                "Indexer RoPE metadata buffer is too small: "
                f"capacity={self._rope_capacity}, required={num_tokens}."
            )
        buffers = self._rope_buffers.get(buffer_key)
        if buffers is None:
            shape = (self._rope_capacity, *cos.shape[1:])
            buffers = (
                torch.empty(shape, dtype=cos.dtype, device=cos.device),
                torch.empty(shape, dtype=sin.dtype, device=sin.device),
            )
            self._rope_buffers[buffer_key] = buffers
        cos_buffer, sin_buffer = buffers
        if cos_buffer.shape[1:] != cos.shape[1:] or sin_buffer.shape[1:] != sin.shape[1:]:
            raise RuntimeError(
                "Indexer RoPE metadata shape changed after initialization: "
                f"cos={tuple(cos.shape)}, buffer={tuple(cos_buffer.shape)}."
            )
        cos_view = cos_buffer[:num_tokens]
        sin_view = sin_buffer[:num_tokens]
        cos_view.copy_(cos)
        sin_view.copy_(sin)
        return cos_view, sin_view

    def _get_group_metadata_buffers(
        self,
        draft_index: object,
        num_slots: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_slots > self._slot_capacity:
            raise RuntimeError(
                "Indexer C8 group metadata buffer is too small: "
                f"capacity={self._slot_capacity}, required={num_slots}."
            )
        buffers = self._group_metadata_buffers.get(draft_index)
        if buffers is None:
            buffers = (
                torch.empty(self._slot_capacity, dtype=torch.int32, device=self.device),
                torch.empty(self._slot_capacity, dtype=torch.int32, device=self.device),
                torch.empty(self._slot_capacity, dtype=torch.int32, device=self.device),
            )
            self._group_metadata_buffers[draft_index] = buffers
        return (
            buffers[0][:num_slots],
            buffers[1][:num_slots],
            buffers[2][:num_slots],
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendSFAIndexerMetadata:
        # common_prefix_len / fast_build are unused; kept for API compatibility.
        return self._build(
            common_attn_metadata,
            buffer_key=self._metadata_buffer_key(common_attn_metadata),
            use_cached_rope=True,
            **kwargs,
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendSFAIndexerMetadata:
        return self._build(
            common_attn_metadata,
            buffer_key=self._metadata_buffer_key(common_attn_metadata),
            use_cached_rope=False,
            **kwargs,
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        attn_state: Any = None,
        **kwargs,
    ) -> AscendSFAIndexerMetadata:
        return self._build(
            common_attn_metadata,
            buffer_key=self._metadata_buffer_key(common_attn_metadata),
            # Match the normal first-step build: the cached RoPE tables have
            # persistent addresses that graph replay updates in place.
            use_cached_rope=True,
            **kwargs,
        )

    @staticmethod
    def _metadata_buffer_key(common_attn_metadata: CommonAttentionMetadata) -> tuple[str, int]:
        # The proposer uses one persistent slot-mapping tensor per logical
        # draft step. Key every mutable derived buffer by that address so
        # graph capture and runtime rebuild update the exact same storage.
        return ("slot_mapping", common_attn_metadata.slot_mapping.data_ptr())

    def _build(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        buffer_key: object,
        use_cached_rope: bool,
        **kwargs,
    ) -> AscendSFAIndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        if self._dcp_active:
            block_table, slot_mapping = self._build_dcp_cache_metadata(
                common_attn_metadata,
                kwargs.get("pcp_context"),
                kwargs.get("pcp_cache_group_idx"),
                buffer_key,
            )
        elif self._pcp_active:
            # PCP writes cover the gathered prefill region too, which
            # requires the full slot mapping.
            slot_mapping = common_attn_metadata.slot_mapping
            block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        else:
            slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
            block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        if self._dsa_cp_active and not self._pcp_active:
            slot_mapping = self._build_dsa_cp_slot_mapping(
                slot_mapping,
                num_input_tokens,
                buffer_key,
            )
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        block_size = self.kernel_block_size

        cos, sin = get_cos_and_sin_mla(input_positions, use_cache=use_cached_rope)
        cos = cos[:num_input_tokens]
        sin = sin[:num_input_tokens]
        cum_query_lens = common_attn_metadata.query_start_loc[1 : num_reqs + 1]
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        actual_seq_lengths_query = cum_query_lens
        actual_seq_lengths_key = seq_lens
        if self._dsa_cp_active and not self._pcp_active:
            (
                cos,
                sin,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
            ) = self._build_dsa_cp_parallel_metadata(
                common_attn_metadata,
                cos,
                sin,
                buffer_key,
            )
        else:
            cos, sin = self._copy_rope_to_metadata_buffers(
                cos,
                sin,
                buffer_key,
            )

        num_decode_tokens = 0
        if self._pcp_active:
            # Only PCP's cache-write gather consumes this boundary. Dummy
            # decode metadata does not always carry is_prefilling, so treat
            # its short queries as decode instead of requiring that field.
            _, _, num_decode_tokens, _ = split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.decode_threshold,
                treat_short_extends_as_decodes=(
                    getattr(common_attn_metadata, "is_prefilling", None) is None
                ),
            )

        group_len = None
        group_key_idx = None
        group_key_cache_idx = None
        if get_ascend_config().c8_reshape_optim_enabled:
            group_len, group_key_idx, group_key_cache_idx = self._get_group_metadata_buffers(
                buffer_key,
                slot_mapping.numel(),
            )
            torch.ops._C_ascend.store_kv_block_metadata(
                slot_mapping,
                group_len,
                group_key_idx,
                group_key_cache_idx,
                block_size,
            )

        return AscendSFAIndexerMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            cum_query_lens=cum_query_lens,
            block_table=block_table,
            sin=sin,
            cos=cos,
            block_size=block_size,
            group_len=group_len,
            group_key_idx=group_key_idx,
            group_key_cache_idx=group_key_cache_idx,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            num_decode_tokens=num_decode_tokens,
        )
