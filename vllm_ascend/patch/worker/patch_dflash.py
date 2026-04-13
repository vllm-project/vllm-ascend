from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
import torch_npu

logger = init_logger(__name__)

def ascend_precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None = None,
) -> None:
    """Precompute K/V for context states write them into each layer's KV cache.

    Input context states are projected to K/V, normed, and have RoPE applied.
    Since the context shape is different than the query shape, we can't rely on the
    regular forward pass to apply torch.compile and CUDA graphs to this section.
    As such, this function is optimized to minimize the number of torch ops present:
    we use fused vLLM kernels for RMSNorm and RoPE, fuse the GEMM into one
    large projection, and avoid cloning buffers (with .contiguous()) where possible.

    When context_slot_mapping is None (e.g. during dummy_run) only
    the computation runs, and no K/V is written to cache.
    """
    if not hasattr(self, "_num_attn_layers"):
        logger.warning_once(
            "DFlash buffer initialization was skipped. If dummy weights are not "
            "in use, this may indicate an error in weight loading."
        )
        self._build_fused_kv_buffers()

    num_ctx = context_states.shape[0]
    L = self._num_attn_layers
    kv = self._kv_size
    hd = self._head_dim
    nkv = self._num_kv_heads

    # --- Fused KV projection (one GEMM for all layers) ---
    normed_context_states, _ = torch_npu.npu_rms_norm(context_states, self._hidden_norm_weight, self._rms_norm_eps)

    all_kv_flat = F.linear(
        normed_context_states, self._fused_kv_weight, self._fused_kv_bias
    )
    # Single contiguous copy that separates K/V and transposes to
    # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
    # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
    all_kv = (
        all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
    )
    all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
    all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

    # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
    all_k_normed = torch.empty_like(all_k)
    for i in range(L):
        all_k_normed[i], _ = torch_npu.npu_rms_norm(all_k[i], self._k_norm_weights[i], self._rms_norm_eps)

    # --- Fused RoPE across all layers ---
    # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
    # In-place RoPE: pass K as the "query" arg with key=None.
    all_k_flat = all_k_normed.view(L * num_ctx, kv)
    positions_repeated = context_positions.repeat(L)
    cos_sin_cache = self._rope_cos_sin_cache
    if cos_sin_cache.dtype != all_k_flat.dtype:
        cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)

    all_k_flat, _ = RotaryEmbedding.forward_static(
        positions=positions_repeated,
        query=all_k_flat,
        key=None,
        head_size=self._rope_head_size,
        rotary_dim=self._rope_head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox_style=self._rope_is_neox,
    )

    if context_slot_mapping is None:
        return

    # --- Per-layer cache insert ---
    all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)

    for i in range(L):
        attn = self._attn_layers[i]
        kv_cache = attn.kv_cache
        attn.impl.do_kv_cache_update(
            attn,
            all_k_final[i],
            all_v[i],
            kv_cache,
            context_slot_mapping,
        )


DFlashQwen3Model.precompute_and_store_context_kv = ascend_precompute_and_store_context_kv