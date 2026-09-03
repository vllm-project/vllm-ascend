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
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.distributed.utils import all_gather_async
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
from vllm_ascend.utils import enable_dsa_cp, vllm_version_is

if vllm_version_is("0.27.1"):
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
    cache view (block table, slot mapping), the rope tables, and the LI C8
    reshape-optim fields. The sequence lengths and rope tables reflect the
    unsharded batch; the parallel-layout sequence lengths the top-k kernel
    consumes are injected per forward by SFA (see
    ``actual_seq_lengths_query`` / ``actual_seq_lengths_key``).
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
    # Parallel-layout sequence lengths consumed by the top-k kernel, injected
    # by SFA per forward: base/PCP modes use the unsharded ``cum_query_lens``
    # / ``seq_lens`` equivalents, DSA-CP shards them per rank. Transported on
    # this metadata so the indexer forward interface stays layout-agnostic.
    actual_seq_lengths_query: torch.Tensor | None = None
    actual_seq_lengths_key: torch.Tensor | None = None
    # Decode-token count injected by SFA per forward; the PCP cache-write
    # gather splits the local prefill region on it (all-decode batches skip
    # the gather).
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
    LI C8 reshape-optim fields); SFA only injects the parallel-layout values
    (sequence lengths, decode count) onto that metadata per forward.

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
        cache_type: str = "",
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
        self.k_cache = getattr(vllm_indexer, "k_cache", None)
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
            AscendSFAIndexerBackend.q_hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu") / (
                128**0.5
            )
        if self.enable_sparse_li_c8 and AscendSFAIndexerBackend.k_hadamard is None:
            AscendSFAIndexerBackend.k_hadamard = torch.tensor(scipy.linalg.hadamard(128), dtype=torch.bfloat16, device="npu") / (
                128**0.5
            )

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
        return self.enable_sparse_li_c8 and get_ascend_config().c8_enable_reshape_optim

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
            gathered_tensors, slot_mapping = _gather_prefill_cache_inputs(tensors, slot_mapping, indexer_metadata.num_decode_tokens)
            k_li = gathered_tensors[0]
            assert slot_mapping.numel() == k_li.shape[0], (
                "PCP indexer cache write requires one slot per gathered token: "
                f"tokens={k_li.shape[0]}, slots={slot_mapping.numel()}."
            )
            if k_li_scale is not None:
                k_li_scale = gathered_tensors[1]
        elif self._dsa_cp_active:
            k_li, handle = all_gather_async(k_li, get_tp_group(), async_op=False)
            if handle is not None:
                handle.wait()
            if self.enable_sparse_li_c8:
                assert k_li_scale is not None
                k_li_scale, handle = all_gather_async(k_li_scale, get_tp_group(), async_op=False)
                if handle is not None:
                    handle.wait()
        return k_li, k_li_scale, slot_mapping

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_hidden_states: torch.Tensor,
        indexer_metadata: AscendSFAIndexerMetadata,
        compute_topk: bool = True,
    ) -> torch.Tensor | None:
        """Full indexer pipeline: k path -> cache write -> top-k selection.

        The k path output is persisted first because the selection kernel
        reads the freshly written cache. ``compute_topk=False`` (SFA layers
        sharing top-k indices) still runs the k path and the write so the
        cache stays up to date, and returns None.

        ``cos`` / ``sin`` come from SFA's metadata, not the indexer's own:
        DSA-CP shards them to the local token shard, matching the sharded
        inputs. ``indexer_metadata`` carries the indexer's own cache view
        plus the parallel-layout values injected by SFA."""
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
    so the slot mapping and block table mirror the ``*.attn`` layer's; the
    rope tables are rebuilt from the same positions via the shared helper.
    The slot mapping is emitted write-ready for the active parallel mode
    (full gather mapping under PCP). Variants with their own cache geometry
    override this construction to supply their layout's equivalents.
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
        self._pcp_active = vllm_config.parallel_config.prefill_context_parallel_size > 1

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendSFAIndexerMetadata:
        # common_prefix_len / fast_build are unused; kept for API compatibility.
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        if self._pcp_active:
            # PCP writes cover the gathered prefill region too, which
            # requires the full slot mapping.
            slot_mapping = common_attn_metadata.slot_mapping
        else:
            slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        block_size = self.kernel_block_size

        cos, sin = get_cos_and_sin_mla(input_positions, use_cache=True)

        if get_ascend_config().c8_enable_reshape_optim:
            torch.ops._C_ascend.store_kv_block_metadata(
                slot_mapping,
                common_attn_metadata.group_len,
                common_attn_metadata.group_key_idx,
                common_attn_metadata.group_key_cache_idx,
                block_size,
            )

        return AscendSFAIndexerMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            slot_mapping=slot_mapping,
            seq_lens=common_attn_metadata.seq_lens[:num_reqs],
            cum_query_lens=common_attn_metadata.query_start_loc[1 : num_reqs + 1],
            block_table=common_attn_metadata.block_table_tensor[:num_reqs],
            sin=sin[:num_input_tokens],
            cos=cos[:num_input_tokens],
            block_size=block_size,
            group_len=common_attn_metadata.group_len,
            group_key_idx=common_attn_metadata.group_key_idx,
            group_key_cache_idx=common_attn_metadata.group_key_cache_idx,
        )
