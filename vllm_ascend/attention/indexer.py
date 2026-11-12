from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import numpy as np
import scipy.linalg  # type: ignore
import torch
import torch_npu
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.triton_utils import HAS_TRITON
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.ops.pcp import _gather_prefill_cache_inputs  # type: ignore[import-not-found]
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.utils import select_common_block_size

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.context_parallel.common_cp import (
    build_pcp_ordered_slot_mapping,
    get_pcp_num_replicated_tokens,
    is_pcp_decode_sharding_enabled,
)
from vllm_ascend.attention.context_parallel.sfa_dcp_utils import (
    build_sfa_dcp_replicated_block_table,
    build_sfa_dcp_replicated_slot_mapping,
    get_sfa_dcp_local_block_table,
    get_sfa_dcp_max_local_block_table_cols,
    get_sfa_max_num_reqs,
    get_sfa_pcp_global_metadata,
)
from vllm_ascend.attention.context_parallel.sharded_indexer import use_local_topk_output
from vllm_ascend.attention.utils import split_decodes_and_prefills, uses_pcp_sharded_layout
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
from vllm_ascend.ops.triton.sfa_indexer_store import store_indexer_key_scale
from vllm_ascend.utils import (
    _round_up,
    enable_sfa_dcp_replicated_indexer,
    enable_sfa_dcp_sharded_indexer,
    is_pd_decode_recompute_scheduler_enabled,
)
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, DeviceMetadataTask, wait_for_device_metadata

if HAS_TRITON:
    from vllm_ascend.ops.triton.indexer_quant_cache_int8 import (
        indexer_quant_cache_int8,
        register_indexer_quant_cache_int8_warmup,
        supports_indexer_quant_cache_int8,
    )

_INDEXER_QK_STREAM: torch.npu.Stream | None = None


def _indexer_qk_stream() -> torch.npu.Stream:
    """Reuse one worker-local stream across sequential indexer layers."""
    global _INDEXER_QK_STREAM
    if _INDEXER_QK_STREAM is None:
        _INDEXER_QK_STREAM = torch_npu.npu.Stream()
    return _INDEXER_QK_STREAM


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
    # (gather-region) mapping, otherwise the input-token slice.
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    cum_query_lens: torch.Tensor
    block_table: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor
    block_size: int = 0
    # Block/token coordinates in the final cache-write order (after PCP
    # gather), shared by K and scale views of the same kernel pages.
    cache_scatter_indices: torch.Tensor | None = None
    # Parallel-layout sequence lengths consumed by the top-k kernel. Base/PCP
    # modes use the unsharded values.
    actual_seq_lengths_query: torch.Tensor | None = None
    actual_seq_lengths_key: torch.Tensor | None = None
    # The PCP cache-write gather splits the local prefill region on this
    # independently computed decode-token count.
    num_decode_tokens: int = 0
    # Resolved by the builder; execution must not infer layout from tensor sizes.
    uses_pcp_sharded_layout: bool = False
    # Optional selection supplied by the owning attention implementation.
    # Projection, rope and cache writes continue to use this backend.
    # The selector receives ``(q_li, weights, indexer, metadata, q_li_scale)``;
    # ``q_li_scale`` is set only when LI C8 has already quantized ``q_li``.
    topk_selector: Any | None = None
    local_visible_lengths: torch.Tensor | None = None
    # Shared verbatim by the A5 metadata producer and QLI consumer.
    qli_dcp_args: dict[str, Any] | None = None
    # Includes the leading zero; shared by every layer in this metadata group.
    query_start_loc: torch.Tensor | None = None
    # V2's deferred producer fills this before the first indexer consumes it.
    qli_metadata: torch.Tensor | None = None
    qli_metadata_group_id: int | None = None


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

    # Bare annotations only: the buffers themselves are registered per instance
    # in ``__init__`` and stay ``None`` until ``process_weights_after_loading``
    # fills them. A class-level ``= None`` default would shadow the registered
    # buffer on attribute lookup and make the matrices process-global again,
    # which is exactly the sleep-mode bug these annotations document for mypy.
    q_hadamard: torch.Tensor | None
    k_hadamard: torch.Tensor | None

    @property
    def topk_output_width(self) -> int:
        return self.topk_tokens

    def get_topk_lengths(self, positions: torch.Tensor) -> torch.Tensor:
        return (positions + 1).clamp(min=0, max=self.topk_tokens)

    @staticmethod
    def get_impl_cls():
        return None

    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        # vLLM #60271 added the batch-invariance support check in
        # `init_attn_backend` in both supported vLLM revisions.
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
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int]:
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
        # Indexer RoPE is independent of the main MLA projection dimensions.
        self.qk_rope_head_dim = getattr(getattr(vllm_indexer, "config", None), "index_rope_dim", qk_rope_head_dim)
        vllm_indexer.topk_indices_buffer = None  # delete topk_indices_buffer

        self.enable_sparse_li_c8 = get_ascend_config().is_sparse_li_c8_layer(self.k_cache.prefix)
        if self.enable_sparse_li_c8:
            self.c8_k_cache_dtype = kv_cache_dtype_str_to_dtype(
                get_current_vllm_config().attention_config.indexer_kv_dtype,
                get_current_vllm_config().model_config,
            )
            if self.c8_k_cache_dtype == torch.float8_e4m3fn:
                self.c8_k_scale_cache_dtype = torch.float32
            elif self.c8_k_cache_dtype == torch.int8:
                self.c8_k_scale_cache_dtype = torch.float16

        self.enable_sparse_li_c4 = get_ascend_config().is_sparse_li_c4_layer(self.k_cache.prefix)
        if self.enable_sparse_li_c4:
            self.c4_k_cache_dtype, self.c4_k_scale_cache_dtype = torch.uint8, torch.float8_e8m0fnu
            self.c4_k_op_dtype = torch_npu.float4_e2m1fn_x2

        model_type = get_current_vllm_config().model_config.hf_config.model_type
        self.is_rope_neox_style = model_type not in ["glm_moe_dsa"]
        self.use_torch_npu_lightning_indexer = model_type in ["glm_moe_dsa", "kimi_k31"]

        # Cache-write gathers for parallel layouts: PCP all-gathers every
        # non-replicated token across the CP group. Base layout is a no-op.
        parallel_config = get_current_vllm_config().parallel_config
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        self._is_pcp_decode_sharded = is_pcp_decode_sharding_enabled(get_current_vllm_config())
        self._use_fused_cache_store = (
            type(self) is AscendSFAIndexerBackend
            and not self.use_pcp  # Keep PCP cache gathering and slot remapping on the existing path.
            and self.enable_sparse_li_c8
            and self.c8_k_cache_dtype == torch.int8  # INT8 config uses FP16 scales; C8 may also be FP8.
            and self.head_dim & (self.head_dim - 1) == 0  # Full-row tl.arange requires a power-of-two width.
        )
        self._pcp_active = self.use_pcp
        self._sharded_dcp = enable_sfa_dcp_sharded_indexer()
        self._local_topk = False
        self._dcp_interleave = parallel_config.cp_kv_cache_interleave_size
        if self._sharded_dcp:
            from vllm_ascend.attention.context_parallel.sharded_indexer import MAX_EXACT_FLOAT32_INTEGER

            self._local_topk = use_local_topk_output(
                self._sharded_dcp,
                get_pcp_group().world_size,
                False,
                turboquant=get_ascend_config().enable_sparse_sfa_turboquant,
            )
            if get_current_vllm_config().model_config.max_model_len > MAX_EXACT_FLOAT32_INTEGER:
                raise ValueError("Sharded QLI's AiCore ID comparator supports at most 2**24 tokens.")
            text_config = get_current_vllm_config().model_config.hf_text_config
            expected_heads = getattr(text_config, "index_n_heads", getattr(text_config, "index_num_heads", self.n_head))
            if self.n_head != expected_heads:
                raise ValueError("Sharded QLI requires complete replicated indexer heads on every DCP rank.")

        # The LI C8 Hadamard matrices are created while the sleep-mode weights
        # mem-pool is active and are read by every forward, so they must survive
        # a level-2 sleep. Keep them as non-persistent buffers of this module so
        # the worker's buffer backup path (``model.named_buffers()``) restores
        # them; a plain attribute keeps its Python reference across sleep while
        # its device storage is discarded and remapped empty.
        self.register_buffer("q_hadamard", None, persistent=False)
        self.register_buffer("k_hadamard", None, persistent=False)
        hf_config = get_current_vllm_config().model_config.hf_config
        self._is_kimi_k31 = hf_config.model_type == "kimi_k31" or getattr(type(hf_config), "is_kimi_k31", False)
        self._q_stream: torch.npu.Stream | None = None
        self._fused_k_cache: Callable[..., None] | None = None

    @property
    def enable_sparse_li_quant(self) -> bool:
        return self.enable_sparse_li_c8 or self.enable_sparse_li_c4

    def process_weights_after_loading(self) -> None:
        if not self.enable_sparse_li_quant:
            return
        if self.q_hadamard is None:
            hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu")
            self.q_hadamard = hadamard / (128**0.5)
        if self.k_hadamard is None:
            hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu")
            self.k_hadamard = hadamard / (128**0.5)
        if (
            type(self) is AscendSFAIndexerBackend
            and self._is_kimi_k31
            and self.enable_sparse_li_c8
            and not self.enable_sparse_li_c4
            and self.head_dim == 128
            and self.qk_rope_head_dim == 0
            and get_current_vllm_config().lora_config is None
        ):
            if self._q_stream is None:
                self._q_stream = _indexer_qk_stream()
            if (
                get_current_hardware_profile().supports(HardwareCapability.INDEXER_QUANT_CACHE)
                and self.c8_k_cache_dtype == torch.float8_e4m3fn
                and find_spec("cann_ops_transformer") is not None
            ):
                # Load the existing CANN entry before graph capture. It accepts
                # separate page-strided K/scale views, unlike a packed epilogue.
                from cann_ops_transformer import ops

                self._fused_k_cache = getattr(ops, "indexer_quant_cache", None)
            elif (
                HAS_TRITON
                and get_current_hardware_profile().supports(HardwareCapability.NATIVE_INT8_SPARSE)
                and self.c8_k_cache_dtype == torch.int8
                and self.c8_k_scale_cache_dtype == torch.float16
            ):
                self._fused_k_cache = indexer_quant_cache_int8

    def register_k_cache_warmup(self) -> None:
        """Register the INT8 epilogue after the worker binds the real cache."""
        if (
            not HAS_TRITON
            or self._fused_k_cache is not indexer_quant_cache_int8
            or self._pcp_active
            or len(self.k_cache.kv_cache) != 2
        ):
            return
        register_indexer_quant_cache_int8_warmup(*self.k_cache.kv_cache)

    def _can_fuse_k_cache(self, hidden_states: torch.Tensor) -> bool:
        if (
            getattr(self, "_fused_k_cache", None) is None
            or self._pcp_active
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.shape[0] == 0
            or len(self.k_cache.kv_cache) != 2
        ):
            return False
        key, scale = self.k_cache.kv_cache
        if self.c8_k_cache_dtype == torch.int8:
            return HAS_TRITON and supports_indexer_quant_cache_int8(key, scale, hidden_states.shape[0])
        return (
            key.dtype == torch.float8_e4m3fn
            and scale.dtype == torch.float32
            and key.ndim == 4
            and scale.ndim == 4
            and key.shape[:3] == scale.shape[:3]
            and key.shape[2:] == (1, self.head_dim)
            and scale.shape[2:] == (1, 1)
            and key.stride(-1) == scale.stride(-1) == 1
        )

    @property
    def num_cache_tensors(self) -> int:
        """Number of tensors this indexer's cache occupies in the composed
        ``kv_cache`` tuple (k cache only, or k cache plus scale cache)."""
        return 2 if self.enable_sparse_li_quant else 1

    def _quantize_li_tensor(
        self, x: torch.Tensor, *, defer_scale_cast: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Hadamard transform and quantize for LI C8 or C4 path."""
        assert self.q_hadamard is not None
        return self._quantize_transformed_li_tensor(x @ self.q_hadamard, defer_scale_cast=defer_scale_cast)

    def _quantize_transformed_li_tensor(
        self, x: torch.Tensor, *, defer_scale_cast: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize the BF16 Hadamard result without repeating its Cube work."""
        shape_ori = x.shape
        x = x.view(-1, self.head_dim)
        if self.enable_sparse_li_c4:
            x, scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=self.c4_k_op_dtype)
            scale = scale.view(*shape_ori[:-1], *scale.shape[-2:])
            return x, scale.view(self.c4_k_scale_cache_dtype)
        x, scale = torch_npu.npu_dynamic_quant(x, dst_type=self.c8_k_cache_dtype)
        return x, scale if defer_scale_cast else scale.to(self.c8_k_scale_cache_dtype)

    @staticmethod
    def _write_cache_tensor(
        cache: torch.Tensor,
        updates: torch.Tensor,
        slots: torch.Tensor,
        cache_scatter_indices: torch.Tensor | None = None,
    ) -> None:
        if cache.is_contiguous():
            DeviceOperator.scatter_cache(cache.view(-1, *updates.shape[1:]), slots.view(-1, 1), updates)
        else:
            # A hybrid page contains K and scales, so flattening its component
            # views would either fail or copy the write destination. Address the
            # block/token axes explicitly to retain the original shared storage.
            assert cache_scatter_indices is not None
            DeviceOperator.scatter_cache(cache, cache_scatter_indices, updates.view(-1, *cache.shape[2:]))

    def write_cache(
        self,
        k_li: torch.Tensor,
        k_li_scale: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        cache_scatter_indices: torch.Tensor | None = None,
    ) -> None:
        """Persist ``k_li`` (and ``k_li_scale`` when LI quant is enabled) into
        this indexer's own cache tensors: slot 0 of ``self.k_cache.kv_cache``
        is the k cache, slot 1 (present only for LI quant) is the scale cache.

        ``forward`` calls this after ``_gather_cache_inputs`` has resolved
        the parallel layout of the tensors and the slot mapping; variants
        with a different cache layout should override it. Page-strided views
        require the builder's matching ``cache_scatter_indices``.
        """
        indexer_k_cache = self.k_cache.kv_cache[INDEXER_K_CACHE_SLOT]
        # float4_e2m1fn_x2 / float8_e8m0fnu are not supported by the scatter
        # kernel; view as uint8 (same 1-byte layout) for the cache write.
        if k_li.dtype == torch_npu.float4_e2m1fn_x2:
            indexer_k_cache = indexer_k_cache.view(torch.uint8)
            k_li = k_li.view(torch.uint8)
        self._write_cache_tensor(
            indexer_k_cache,
            k_li.view(-1, k_li.shape[-1]),
            slot_mapping,
            cache_scatter_indices,
        )
        if self.enable_sparse_li_quant:
            assert k_li_scale is not None
            # C4 scale is multi-dimensional (b*s, d/64, 2); C8 is (b*s, 1).
            # Keep all trailing dims beyond the token axis for the scatter.
            scale_per_token = k_li_scale.shape[1:]
            indexer_scale_cache = self.k_cache.kv_cache[INDEXER_SCALE_CACHE_SLOT]
            if indexer_scale_cache.dtype == torch.float8_e8m0fnu:
                indexer_scale_cache = indexer_scale_cache.view(torch.uint8)
                k_li_scale = k_li_scale.view(torch.uint8)
            self._write_cache_tensor(
                indexer_scale_cache,
                k_li_scale.view(-1, *scale_per_token),
                slot_mapping,
                cache_scatter_indices,
            )

    def _project_k(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        projected_kw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project K/weights, then normalize and rotate K before its Hadamard."""
        assert self.wk_weights_proj is not None
        assert self.k_norm is not None

        if projected_kw is None:
            projected_kw, _ = self.wk_weights_proj(hidden_states)
        kw = projected_kw
        k_li = kw[:, : self.head_dim]
        indexer_weights = kw[:, self.head_dim :]
        k_li = self.k_norm(k_li).unsqueeze(1)
        k_li = k_li.view(-1, 1, self.head_dim)

        if self.qk_rope_head_dim and HAS_TRITON:
            cos = cos.view(-1, self.qk_rope_head_dim)
            sin = sin.view(-1, self.qk_rope_head_dim)
            k_li = rope_forward_triton_siso(
                k_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        elif self.qk_rope_head_dim:
            k_li_pe, k_li_nope = torch.split(
                k_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )

            cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
            sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

            k_li_pe = k_li_pe.unsqueeze(2)
            k_li_pe = torch_npu.npu_rotary_mul(k_li_pe, cos, sin)
            k_li_pe = k_li_pe.squeeze(2)

            k_li = torch.cat([k_li_pe, k_li_nope], dim=-1)  # [b*s,128]

        return k_li, indexer_weights

    def forward_k(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        projected_kw: torch.Tensor | None = None,
        quantize: bool = True,
        defer_scale_cast: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        k_li, indexer_weights = self._project_k(hidden_states, cos, sin, projected_kw=projected_kw)
        if self.enable_sparse_li_quant:
            if quantize:
                k_li, k_li_scale = self._quantize_li_tensor(k_li, defer_scale_cast=defer_scale_cast)
                if self.enable_sparse_li_c8:
                    k_li_scale = k_li_scale.unsqueeze(-1)
            else:
                # Keep the exact existing Hadamard and BF16 boundary. Only
                # quantization and the two cache writes move into the epilogue.
                assert self.q_hadamard is not None
                k_li = (k_li @ self.q_hadamard).view(-1, self.head_dim)
                k_li_scale = None
        else:
            k_li_scale = None

        return k_li, k_li_scale, indexer_weights

    def _gather_cache_inputs(
        self,
        k_li: torch.Tensor,
        k_li_scale: torch.Tensor | None,
        indexer_metadata: AscendSFAIndexerMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Parallel-layout transforms applied to the k path output between
        ``forward_k`` and the cache write. Identity in the base layout; PCP
        all-gathers the prefill region across the CP group (reordering the
        slot mapping to the gathered layout)."""
        slot_mapping = indexer_metadata.slot_mapping
        if self.use_pcp:
            tensors = (k_li,) if k_li_scale is None else (k_li, k_li_scale)
            num_replicated_tokens = get_pcp_num_replicated_tokens(
                indexer_metadata.num_decode_tokens, self._is_pcp_decode_sharded
            )
            if not indexer_metadata.uses_pcp_sharded_layout:
                # An unexpanded slot mapping (e.g. a replicated draft decode)
                # means every rank holds the same tokens: write them locally.
                num_replicated_tokens = indexer_metadata.num_actual_tokens
            gathered_tensors, slot_mapping = _gather_prefill_cache_inputs(tensors, slot_mapping, num_replicated_tokens)
            k_li = gathered_tensors[0]
            assert slot_mapping.numel() == k_li.shape[0], (
                "PCP indexer cache write requires one slot per gathered token: "
                f"tokens={k_li.shape[0]}, slots={slot_mapping.numel()}."
            )
            if k_li_scale is not None:
                k_li_scale = gathered_tensors[1]
        return k_li, k_li_scale, slot_mapping

    def _get_local_qli_inputs(
        self, indexer_metadata: AscendSFAIndexerMetadata
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Consume V2's deferred result, or share one inline result per forward.

        V2 joins its producer only at the first QLI consumer in each group.
        Without a deferred producer, the first consuming layer rebuilds on
        every visit, including warmup and capture using the same context, so
        generation stays in the graph and refreshes from lengths on replay.
        """
        query_start_loc = indexer_metadata.query_start_loc
        seq_lens = indexer_metadata.actual_seq_lengths_key
        assert query_start_loc is not None and seq_lens is not None
        if indexer_metadata.qli_metadata_group_id is not None:
            assert indexer_metadata.qli_metadata is not None
            wait_for_device_metadata(DeviceMetadataStage.INDEXER, indexer_metadata.qli_metadata_group_id)
            return query_start_loc, seq_lens, indexer_metadata.qli_metadata
        forward_context = get_forward_context()
        cache = getattr(forward_context, "sfa_qli_metadata_cache", None)
        if cache is None:
            cache = {}
            forward_context.sfa_qli_metadata_cache = cache
        # Quant/mask modes and TND/PA_BBND layouts are fixed by the device
        # adapter. A3 metadata also consumes global bounds and DCP geometry. The
        # metadata identity and input identities keep cache groups/draft steps
        # separate, even when their sizes or physical buffers happen to match.
        key = (
            id(indexer_metadata),
            id(query_start_loc),
            id(seq_lens),
            id(indexer_metadata.qli_dcp_args),
            id(indexer_metadata.seq_lens),
            self.n_head,
            self.head_dim,
            self.topk_tokens,
        )
        owner = self.k_cache.prefix
        cached = cache.get(key)
        if cached is None or cached[0] == owner:
            cu_seqlens_q = query_start_loc.to(torch.int32)
            seqused_k = seq_lens.to(torch.int32)
            dcp_kwargs = indexer_metadata.qli_dcp_args or {}
            if get_current_hardware_profile().supports(HardwareCapability.NATIVE_INT8_SPARSE):
                group = get_dcp_group()
                dcp_kwargs = dict(
                    global_seq_lens=indexer_metadata.seq_lens,
                    cp_world=group.world_size,
                    cp_rank=group.rank_in_group,
                    cp_interleave=self._dcp_interleave,
                )
            metadata = DeviceOperator.build_indexer_local_metadata(
                cu_seqlens_q,
                seqused_k,
                num_heads_q=self.n_head,
                head_dim=self.head_dim,
                topk=self.topk_tokens,
                **dcp_kwargs,
            )
            cached = (owner, cu_seqlens_q, seqused_k, metadata)
            cache[key] = cached
        return cached[1:]

    def _project_q(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Project and rotate Q without reading the K cache."""
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
        if self.qk_rope_head_dim and HAS_TRITON:
            q_li = rope_forward_triton_siso(
                q_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        elif self.qk_rope_head_dim:
            q_li_pe, q_li_nope = torch.split(
                q_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )

            q_li_pe = q_li_pe.unsqueeze(2)
            q_li_pe = torch_npu.npu_rotary_mul(q_li_pe, cos, sin)
            q_li_pe = q_li_pe.squeeze(2)
            q_li = torch.cat([q_li_pe, q_li_nope], dim=-1)

        return q_li

    def _forward_q(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Size | None]:
        q_li = self._project_q(hidden_states, q_c, cos, sin)
        q_li_scale = None
        q_li_shape_ori = None
        if self.enable_sparse_li_quant:
            q_li_shape_ori = q_li.shape
            q_li, q_li_scale = self._quantize_li_tensor(q_li)

        return q_li, q_li_scale, q_li_shape_ori

    def _forward_qk(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k_hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        fuse_cache: bool,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Size], torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Pair Q Cube/K Vector, then Q Vector/K Cube; join before cache writes."""
        main_stream = torch_npu.npu.current_stream()
        q_stream = self._q_stream
        assert q_stream is not None
        kw, _ = self.wk_weights_proj(k_hidden_states)
        q_stream.wait_stream(main_stream)
        with torch_npu.npu.stream(q_stream):
            q_li = self._project_q(hidden_states, q_c, cos, sin)
            q_shape = q_li.shape
            q_hadamard = q_li @ self.q_hadamard
            for value in q_c if isinstance(q_c, tuple) else (q_c,):
                value.record_stream(q_stream)
        k_li, indexer_weights = self._project_k(k_hidden_states, cos, sin, projected_kw=kw)

        # These two event edges form a stage boundary: both Q Hadamard and
        # K LayerNorm must finish before Q quantization / K Hadamard begin.
        main_stream.wait_stream(q_stream)
        q_hadamard.record_stream(main_stream)
        q_stream.wait_stream(main_stream)
        with torch_npu.npu.stream(q_stream):
            k_hadamard = k_li @ self.q_hadamard
            k_li.record_stream(q_stream)
        q_li, q_scale = self._quantize_transformed_li_tensor(q_hadamard)
        main_stream.wait_stream(q_stream)
        k_hadamard.record_stream(main_stream)

        # Q is now ready on main; K Hadamard has joined before the cache
        # epilogue. No Q-stream join is needed after IndexerQuantCache.
        if fuse_cache:
            k_li, k_scale = k_hadamard.view(-1, self.head_dim), None
        else:
            k_li, k_scale = self._quantize_transformed_li_tensor(k_hadamard)
            k_scale = k_scale.unsqueeze(-1)
        return (q_li, q_scale, q_shape), k_li, k_scale, indexer_weights

    def _select_execution_paths(
        self,
        k_hidden_states: torch.Tensor,
        indexer_metadata: AscendSFAIndexerMetadata,
        compute_topk: bool,
    ) -> tuple[torch.npu.Stream | None, bool, bool]:
        # Quant-cache fuses K quantization with persistence. Q/K overlap can
        # accompany it, but needs a nonempty top-k query path; shared-top-k
        # layers still project and persist K without launching Q work.
        fuse_cache = self._can_fuse_k_cache(k_hidden_states)
        q_stream = getattr(self, "_q_stream", None) if compute_topk and k_hidden_states.shape[0] > 0 else None
        # The decode-only store combines already-quantized K and scale writes.
        # It defers the scale cast, so keep it mutually exclusive with both
        # quant-cache and the Q/K overlap path, which already casts its scales.
        fuse_store = (
            not fuse_cache
            and q_stream is None
            and getattr(self, "_use_fused_cache_store", False)
            and indexer_metadata.num_actual_tokens > 0
            and indexer_metadata.num_decode_tokens == indexer_metadata.num_actual_tokens
        )
        return q_stream, fuse_cache, fuse_store

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
        q_stream, fuse_cache, fuse_store = self._select_execution_paths(k_hidden_states, indexer_metadata, compute_topk)
        q_result = None
        if q_stream is not None:
            q_result, k_li, k_li_scale, indexer_weights = self._forward_qk(
                hidden_states, q_c, k_hidden_states, cos, sin, fuse_cache
            )
        elif fuse_cache:
            k_li, k_li_scale, indexer_weights = self.forward_k(k_hidden_states, cos, sin, quantize=False)
        elif fuse_store:
            k_li, k_li_scale, indexer_weights = self.forward_k(k_hidden_states, cos, sin, defer_scale_cast=True)
        else:
            k_li, k_li_scale, indexer_weights = self.forward_k(k_hidden_states, cos, sin)
        k_li, k_li_scale, slot_mapping = self._gather_cache_inputs(k_li, k_li_scale, indexer_metadata)
        if fuse_cache:
            int8_cache = self.c8_k_cache_dtype == torch.int8
            slots = slot_mapping.view(-1)
            assert self._fused_k_cache is not None
            self._fused_k_cache(
                self.k_cache.kv_cache[INDEXER_K_CACHE_SLOT],
                self.k_cache.kv_cache[INDEXER_SCALE_CACHE_SLOT],
                k_li,
                slots if int8_cache else slots.to(torch.int32),
                quant_mode="int8" if int8_cache else "fp8",
                round_scale=False,  # Match npu_dynamic_quant's per-token FP32 scale.
            )
        elif fuse_store:
            assert k_li_scale is not None
            store_indexer_key_scale(
                k_li,
                k_li_scale,
                slot_mapping,
                self.k_cache.kv_cache[INDEXER_K_CACHE_SLOT],
                self.k_cache.kv_cache[INDEXER_SCALE_CACHE_SLOT],
            )
        else:
            self.write_cache(k_li, k_li_scale, slot_mapping, indexer_metadata.cache_scatter_indices)
        if not compute_topk:
            return None

        assert self.wk_weights_proj is not None
        assert self.wq_b is not None
        assert indexer_metadata.actual_seq_lengths_query is not None
        assert indexer_metadata.actual_seq_lengths_key is not None

        if hidden_states is k_hidden_states:
            # The k path already ran wk_weights_proj on these hidden states
            # (SFA hands both stages the same tensor); reuse its weights tail
            # instead of duplicating the GEMM.
            weights = indexer_weights
        else:
            kw, _ = self.wk_weights_proj(hidden_states)
            weights = kw[:, self.head_dim :]

        if q_result is None:
            q_result = self._forward_q(hidden_states, q_c, cos, sin)
        q_li, q_li_scale, q_li_shape_ori = q_result

        topk_selector = getattr(indexer_metadata, "topk_selector", None)
        if topk_selector is not None:
            return topk_selector(q_li, weights, self, indexer_metadata, q_li_scale)
        if self._sharded_dcp:
            from vllm_ascend.attention.context_parallel.sharded_indexer import distributed_topk

            if not self.enable_sparse_li_c8 or self.enable_sparse_li_c4:
                raise NotImplementedError("Sharded Kimi indexer requires the native C8 QLI path.")
            cu_seqlens_q, seqused_k, qli_metadata = self._get_local_qli_inputs(indexer_metadata)
            if indexer_metadata.qli_dcp_args is not None:
                indices, scores = DeviceOperator.indexer_select_dcp_candidates(
                    q_li.view(*q_li_shape_ori),
                    weights,
                    self.k_cache.kv_cache[INDEXER_K_CACHE_SLOT],
                    self.k_cache.kv_cache[INDEXER_SCALE_CACHE_SLOT],
                    indexer_metadata.block_table,
                    cu_seqlens_q,
                    seqused_k,
                    q_li_scale,
                    qli_metadata=qli_metadata,
                    topk=self.topk_tokens,
                    **indexer_metadata.qli_dcp_args,
                )
            elif get_current_hardware_profile().supports(HardwareCapability.NATIVE_INT8_SPARSE):
                indices, scores = DeviceOperator.indexer_select_dcp_candidates(
                    q_li.view(*q_li_shape_ori),
                    weights,
                    self.k_cache.kv_cache[INDEXER_K_CACHE_SLOT],
                    self.k_cache.kv_cache[INDEXER_SCALE_CACHE_SLOT],
                    indexer_metadata.block_table,
                    cu_seqlens_q,
                    seqused_k,
                    q_li_scale,
                    global_seq_lens=indexer_metadata.seq_lens,
                    cp_world=get_dcp_group().world_size,
                    cp_rank=get_dcp_group().rank_in_group,
                    cp_interleave=self._dcp_interleave,
                    qli_metadata=qli_metadata,
                    topk=self.topk_tokens,
                )
            else:
                assert indexer_metadata.local_visible_lengths is not None
                indices, scores = DeviceOperator.indexer_select_local_candidates(
                    q_li.view(*q_li_shape_ori),
                    weights,
                    self.k_cache.kv_cache[INDEXER_K_CACHE_SLOT],
                    self.k_cache.kv_cache[INDEXER_SCALE_CACHE_SLOT],
                    indexer_metadata.block_table,
                    cu_seqlens_q,
                    seqused_k,
                    q_li_scale,
                    indexer_metadata.local_visible_lengths,
                    qli_metadata=qli_metadata,
                    topk=self.topk_tokens,
                )
            return distributed_topk(
                indices,
                scores,
                get_dcp_group(),
                self.topk_tokens,
                self._dcp_interleave,
                return_local=self._local_topk,
            )
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
            self.enable_sparse_li_c4,
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
        self.use_rope = getattr(vllm_config.model_config.hf_text_config, "index_rope_dim", None) != 0
        scheduler_config = vllm_config.scheduler_config
        self.decode_threshold = 1
        self.speculative_config = vllm_config.speculative_config
        speculative_config = self.speculative_config
        if speculative_config is not None:
            self.decode_threshold += speculative_config.num_speculative_tokens

        self.use_pcp = vllm_config.parallel_config.prefill_context_parallel_size > 1
        self._is_pcp_decode_sharded = is_pcp_decode_sharding_enabled(vllm_config)
        self._group_metadata_buffers: dict[
            object,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._rope_buffers: dict[object, tuple[torch.Tensor, torch.Tensor]] = {}
        self._dcp_block_table_buffers: dict[object, torch.Tensor] = {}
        self._dcp_slot_mapping_buffers: dict[object, torch.Tensor] = {}
        self._sharded_seq_buffers: dict[object, tuple[torch.Tensor, torch.Tensor]] = {}
        self._pcp_indexer_slot_mapping_buffers: dict[object, torch.Tensor] = {}
        self._lim_token_masks: dict[object, CpuGpuBuffer] = {}
        self._cache_scatter_indices_buffers: dict[object, torch.Tensor] = {}
        self._device_metadata_enabled = False
        self._device_metadata_tasks: tuple[DeviceMetadataTask, ...] = ()
        self._max_num_reqs = get_sfa_max_num_reqs(vllm_config)
        max_num_input_tokens = scheduler_config.max_num_batched_tokens
        self._rope_capacity = max_num_input_tokens
        pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self._dcp_slot_capacity = max_num_input_tokens * pcp_size
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        self._slot_capacity = max(
            max_num_input_tokens * pcp_size,
            _round_up(max_num_input_tokens, tp_size),
        )

        self.use_sharded_dcp = enable_sfa_dcp_sharded_indexer(vllm_config)
        self.use_native_dcp_qli = self.use_sharded_dcp and DeviceOperator.native_dcp_indexer
        self.use_dcp = enable_sfa_dcp_replicated_indexer(vllm_config) and not self.use_sharded_dcp
        if not self.use_dcp:
            return

        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.replicated_view_block_size = self.kernel_block_size
        physical_block_size = getattr(kv_cache_spec, "block_table_block_size", None) or kv_cache_spec.block_size
        if physical_block_size % self.replicated_view_block_size != 0:
            raise RuntimeError(
                "SFA replicated indexer metadata requires the physical block "
                f"size ({physical_block_size}) to be divisible by the "
                f"kernel block size ({self.replicated_view_block_size})."
            )
        self.blocks_per_phys_block = physical_block_size // self.replicated_view_block_size

        self.max_local_block_table_cols = get_sfa_dcp_max_local_block_table_cols(
            vllm_config.model_config.max_model_len,
            physical_block_size,
            self.dcp_size,
            self.blocks_per_phys_block,
        )
        max_replicated_block_table_cols = self.max_local_block_table_cols * self.dcp_size
        self._dcp_block_table_shape = (self._max_num_reqs, max_replicated_block_table_cols)
        self.replicated_col_idx_buf = torch.arange(
            max_replicated_block_table_cols,
            dtype=torch.int32,
            device=device,
        )
        if self.use_pcp:
            self._pcp_indexer_slot_capacity = (
                max_num_input_tokens * vllm_config.parallel_config.prefill_context_parallel_size
            )

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        speculative_config = vllm_config.speculative_config
        if (
            speculative_config is not None
            and speculative_config.method == "dspark"
            and getattr(speculative_config, "enable_adaptive_verification", False)
        ):
            return AttentionCGSupport.ALWAYS
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
        if block_table_buffer.shape[0] < num_reqs or block_table_buffer.shape[1] < block_table_cols:
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
        global_common_attn_metadata = get_sfa_pcp_global_metadata(
            common_attn_metadata,
            pcp_context,
            pcp_cache_group_idx,
        )
        dcp_block_table = get_sfa_dcp_local_block_table(
            global_common_attn_metadata.block_table_tensor,
            global_common_attn_metadata.num_reqs,
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

        pcp_slot_mapping_buffer = self._pcp_indexer_slot_mapping_buffers.get(buffer_key)
        if pcp_slot_mapping_buffer is None:
            pcp_slot_mapping_buffer = torch.empty(
                self._pcp_indexer_slot_capacity,
                dtype=torch.int32,
                device=self.device,
            )
            self._pcp_indexer_slot_mapping_buffers[buffer_key] = pcp_slot_mapping_buffer
        return build_pcp_ordered_slot_mapping(global_slot_mapping, pcp_context, pcp_slot_mapping_buffer)

    def _build_dcp_cache_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        pcp_context: Any | None,
        pcp_cache_group_idx: int | None,
        buffer_key: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pcp_slot_mapping = None
        if self.use_pcp and pcp_context is not None and bool(pcp_context.global_batch.is_prefilling_np.any()):
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

    def _copy_rope_to_metadata_buffers(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        buffer_key: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = cos.shape[0]
        if num_tokens > self._rope_capacity:
            raise RuntimeError(
                f"Indexer RoPE metadata buffer is too small: capacity={self._rope_capacity}, required={num_tokens}."
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

    @contextmanager
    def defer_device_metadata(self, *, in_graph: bool = False):
        """Opt in only inside V2's target build scope, never its draft builders."""
        was_enabled = self._device_metadata_enabled
        self._device_metadata_enabled = self.use_sharded_dcp
        try:
            yield
        finally:
            self._device_metadata_enabled = was_enabled

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        tasks = self._device_metadata_tasks
        self._device_metadata_tasks = ()
        return tasks

    def _defer_local_qli_metadata(self, metadata: AscendSFAIndexerMetadata) -> None:
        config = self.vllm_config.model_config.hf_text_config
        # Kimi's indexer heads are replicated across DCP ranks. These are the
        # same config fields used to construct every indexer in this group.
        num_heads_q, head_dim, topk = config.index_n_heads, config.index_head_dim, config.index_topk
        query_start_loc = metadata.query_start_loc
        actual_seq_lengths_key = metadata.actual_seq_lengths_key
        assert query_start_loc is not None and actual_seq_lengths_key is not None

        def build_qli_metadata() -> None:
            metadata.query_start_loc = query_start_loc.to(torch.int32)
            metadata.actual_seq_lengths_key = actual_seq_lengths_key.to(torch.int32)
            dcp_kwargs = metadata.qli_dcp_args or {}
            if get_current_hardware_profile().supports(HardwareCapability.NATIVE_INT8_SPARSE):
                group = get_dcp_group()
                dcp_kwargs = dict(
                    global_seq_lens=metadata.seq_lens,
                    cp_world=group.world_size,
                    cp_rank=group.rank_in_group,
                    cp_interleave=self.vllm_config.parallel_config.cp_kv_cache_interleave_size,
                )
            # Keep the operator output itself: no extra copy into a persistent
            # output buffer. FULL capture retains this allocation and replays
            # the producer before its consumers; eager keeps it in metadata.
            metadata.qli_metadata = DeviceOperator.build_indexer_local_metadata(
                metadata.query_start_loc,
                metadata.actual_seq_lengths_key,
                num_heads_q=num_heads_q,
                head_dim=head_dim,
                topk=topk,
                **dcp_kwargs,
            )

        # The builder is long-lived, so event keys do not grow with each new
        # metadata object. All layers sharing this builder share one wait.
        metadata.qli_metadata_group_id = id(self)
        self._device_metadata_tasks = (
            *self._device_metadata_tasks,
            DeviceMetadataTask(DeviceMetadataStage.INDEXER, build_qli_metadata, id(self)),
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
            # Speculative decoding prepares multiple step metadata objects
            # up front, so each step still needs independent RoPE storage.
            copy_rope=self.speculative_config is not None,
            pcp_context=common_attn_metadata.pcp_context,
            pcp_cache_group_idx=common_attn_metadata.pcp_cache_group_idx,
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
            copy_rope=True,
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
            # Draft graph capture starts from the cached values, then copies
            # them into the per-step storage used by runtime drafting.
            use_cached_rope=True,
            copy_rope=True,
            **kwargs,
        )

    @staticmethod
    def _metadata_buffer_key(common_attn_metadata: CommonAttentionMetadata) -> tuple[str, int]:
        # The proposer uses one persistent slot-mapping tensor per logical
        # draft step. Key every mutable derived buffer by that address so
        # graph capture and runtime rebuild update the exact same storage.
        return ("slot_mapping", common_attn_metadata.slot_mapping.data_ptr())

    def _mask_lim_slot_mapping(self, common_attn_metadata, slot_mapping, buffer_key) -> None:
        active = getattr(common_attn_metadata, "req_topk_buffer_active", None)
        if active is None or not get_ascend_config().sparse_kv_offload_config.use_fused_copy_sfa:
            return
        # Only request ownership and query layout are needed; exact device
        # positions in this group's slot mapping must remain unchanged.
        count = common_attn_metadata.num_reqs
        ends = common_attn_metadata.query_start_loc_cpu[1 : count + 1].numpy()
        positions = np.arange(slot_mapping.numel())
        rows = np.searchsorted(ends, positions, side="right").clip(max=count - 1)
        mask = self._lim_token_masks.get(buffer_key)
        if mask is None:
            mask = CpuGpuBuffer(
                self._slot_capacity, dtype=torch.bool, device=slot_mapping.device, pin_memory=is_pin_memory_available()
            )
            self._lim_token_masks[buffer_key] = mask
        size = slot_mapping.numel()
        mask.np[:size] = (~active.numpy()[rows]) | (positions >= ends[-1])
        slot_mapping.masked_fill_(mask.copy_to_gpu(size), -1)

    def _build_cache_scatter_indices(
        self,
        slot_mapping: torch.Tensor,
        num_input_tokens: int,
        num_actual_tokens: int,
        num_decode_tokens: int,
        is_pcp_sharded: bool,
        buffer_key: object,
    ) -> torch.Tensor:
        slots = slot_mapping
        if self.use_pcp:
            # Match _gather_cache_inputs / _gather_prefill_cache_inputs without
            # gathering data: replicated decodes occur once, followed by each
            # rank's prefill rows. PCP-only sharded decode gathers all rows.
            replicated = get_pcp_num_replicated_tokens(num_decode_tokens, self._is_pcp_decode_sharded)
            if not is_pcp_sharded:
                replicated = num_actual_tokens
            if slots.shape[0] <= num_input_tokens or replicated == num_input_tokens:
                slots = slots[:replicated]
            else:
                pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
                slots = slots[: pcp_size * num_input_tokens]
                if replicated:
                    rank_slots = slots.view(pcp_size, num_input_tokens)
                    slots = torch.cat((rank_slots[0, :replicated], rank_slots[:, replicated:].flatten()))
        slots = slots.view(-1)
        if slots.numel() > self._slot_capacity:
            raise ValueError(f"Indexer cache-write rows {slots.numel()} exceed capacity {self._slot_capacity}")
        buffer = self._cache_scatter_indices_buffers.get(buffer_key)
        if buffer is None:
            buffer = torch.empty((self._slot_capacity, 2), dtype=slots.dtype, device=slots.device)
            self._cache_scatter_indices_buffers[buffer_key] = buffer
        indices = buffer[: slots.numel()]
        # reshape_kv_cache exposes kernel-sized K/scale pages even when one
        # physical allocation spans several pages. Use that same row width.
        # Refresh the captured storage on every build; draft steps retain
        # independent buffers under the existing slot-mapping address key.
        torch.stack((slots // self.kernel_block_size, slots % self.kernel_block_size), dim=-1, out=indices)
        # A captured replicated PCP draft can retain a longer view than the
        # next step's live prefix. Match PAD_SLOT_ID=-1 for those padded rows.
        buffer[slots.numel() :, 0].fill_(-1)
        buffer[slots.numel() :, 1].fill_(self.kernel_block_size - 1)
        return indices

    def _build(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        buffer_key: object,
        use_cached_rope: bool,
        copy_rope: bool,
        **kwargs,
    ) -> AscendSFAIndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        is_pcp_sharded = uses_pcp_sharded_layout(common_attn_metadata, self.use_pcp)
        pcp_context = kwargs.get("pcp_context")
        if self.use_dcp:
            block_table, slot_mapping = self._build_dcp_cache_metadata(
                common_attn_metadata,
                pcp_context,
                kwargs.get("pcp_cache_group_idx"),
                buffer_key,
            )
        elif is_pcp_sharded:
            # PCP writes cover the gathered prefill region too, which
            # requires the full slot mapping.
            slot_mapping = common_attn_metadata.slot_mapping
            block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        else:
            slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
            block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        self._mask_lim_slot_mapping(common_attn_metadata, slot_mapping, buffer_key)
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        block_size = self.kernel_block_size

        cos, sin = get_cos_and_sin_mla(input_positions, use_cache=use_cached_rope, use_rope=self.use_rope)
        cos = cos[:num_input_tokens]
        sin = sin[:num_input_tokens]
        cum_query_lens = common_attn_metadata.query_start_loc[1 : num_reqs + 1]
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        actual_seq_lengths_query = cum_query_lens
        actual_seq_lengths_key = seq_lens
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        local_visible = None
        qli_dcp_args = None
        if self.use_sharded_dcp:
            from vllm_ascend.ops.triton.sharded_indexer import build_local_key_lengths, build_local_lengths

            group = get_dcp_group()
            interleave = self.vllm_config.parallel_config.cp_kv_cache_interleave_size
            buffers = self._sharded_seq_buffers.get(buffer_key)
            if buffers is None:
                buffers = (
                    torch.empty(self._max_num_reqs, dtype=torch.int32, device=self.device),
                    torch.empty(
                        (0 if self.use_native_dcp_qli else self._slot_capacity, 1),
                        dtype=torch.int32,
                        device=self.device,
                    ),
                )
                self._sharded_seq_buffers[buffer_key] = buffers
            key_buffer, visible_buffer = buffers
            if num_reqs > key_buffer.numel():
                raise ValueError(f"Sharded indexer request count {num_reqs} exceeds capacity {key_buffer.numel()}")
            actual_seq_lengths_key = key_buffer[:num_reqs]
            if self.use_native_dcp_qli:
                build_local_key_lengths(
                    seq_lens, actual_seq_lengths_key, group.world_size, group.rank_in_group, interleave
                )
                qli_dcp_args = dict(
                    global_seq_lens=seq_lens.to(torch.int32),
                    cp_world=group.world_size,
                    cp_rank=group.rank_in_group,
                    cp_interleave=interleave,
                )
            elif get_current_hardware_profile().supports(HardwareCapability.NATIVE_INT8_SPARSE):
                build_local_key_lengths(
                    seq_lens, actual_seq_lengths_key, group.world_size, group.rank_in_group, interleave
                )
            else:
                local_visible = visible_buffer[:num_input_tokens]
                build_local_lengths(
                    seq_lens,
                    cum_query_lens,
                    actual_seq_lengths_key,
                    local_visible,
                    group.world_size,
                    group.rank_in_group,
                    interleave,
                )
        if copy_rope:
            cos, sin = self._copy_rope_to_metadata_buffers(
                cos,
                sin,
                buffer_key,
            )

        num_decode_tokens = 0
        if is_pcp_sharded or self.use_dcp:
            # PCP's gather and native DCP's fused store need the decode boundary.
            _, _, num_decode_tokens, _ = split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.decode_threshold,
                treat_short_extends_as_decodes=(
                    self.use_dcp and is_pd_decode_recompute_scheduler_enabled(self.vllm_config)
                ),
            )

        metadata = AscendSFAIndexerMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            cum_query_lens=cum_query_lens,
            block_table=block_table,
            sin=sin,
            cos=cos,
            block_size=block_size,
            cache_scatter_indices=self._build_cache_scatter_indices(
                slot_mapping,
                num_input_tokens,
                common_attn_metadata.num_actual_tokens,
                num_decode_tokens,
                is_pcp_sharded,
                buffer_key,
            ),
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            num_decode_tokens=num_decode_tokens,
            uses_pcp_sharded_layout=is_pcp_sharded,
            local_visible_lengths=local_visible,
            qli_dcp_args=qli_dcp_args,
            query_start_loc=query_start_loc,
        )
        if self._device_metadata_enabled:
            self._defer_local_qli_metadata(metadata)
        return metadata
