import torch
import torch.nn.functional as F
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)


def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | None = None,
) -> None:
    """CAPTURE_FUSION_V2: Optimized context KV precompute for DFlash.

    Optimizations vs original:
    1. Batch per-layer k_norm into stacked-weight RMSNorm (saves L-1 kernels)
    2. Avoid intermediate .contiguous() by keeping GEMM output in layer-major layout
    3. Remove wasteful .clone() for the RoPE dummy key: key=None makes the
       NPU RoPE path (forward_oot -> npu_mrope) use a 1-head throwaway buffer

    Correctness fixes vs original:
    4. k_norm.variance_epsilon (vllm RMSNorm has no .eps attribute)
    5. Rebind RoPE output from the return value: the Ascend RoPE path is
       out-of-place (mutates_args=[]), so discarding the return value
       leaves K un-rotated in the KV cache
    """
    if not hasattr(self, "_num_attn_layers"):
        self._build_fused_kv_buffers()

    num_ctx = context_states.shape[0]
    L = self._num_attn_layers
    kv = self._kv_size
    hd = self._head_dim
    nkv = self._num_kv_heads

    # --- Step 1: Fused KV projection (one GEMM for all layers) ---
    normed_context_states = self.hidden_norm(context_states)
    all_kv_flat = F.linear(normed_context_states, self._fused_kv_weight, self._fused_kv_bias)
    # all_kv_flat: [num_ctx, L * 2 * nkv * hd]
    # Reshape to [L, 2, num_ctx, nkv, hd] (layer-major) without .contiguous()
    # by using the fact that the fused weight is already layer-major ordered
    all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(1, 2, 0, 3, 4)
    # all_kv: [L, 2, num_ctx, nkv, hd] (view, not contiguous)
    all_k = all_kv[:, 0]  # [L, num_ctx, nkv, hd]
    all_v = all_kv[:, 1]  # [L, num_ctx, nkv, hd]

    # --- Step 2: Batched per-layer K RMSNorm ---
    # Stack per-layer k_norm weights into a single tensor for batched application
    if not hasattr(self, "_stacked_k_norm_weights"):
        self._stacked_k_norm_weights = torch.stack(
            [self.layers[i].self_attn.k_norm.weight.detach() for i in range(L)]
        )  # [L, hd]
        self._k_norm_eps = self.layers[0].self_attn.k_norm.variance_epsilon

    # all_k: [L, num_ctx, nkv, hd], weight: [L, hd] -> broadcast over nkv
    # RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    k_flat = all_k.reshape(L, num_ctx * nkv, hd).float()
    variance = k_flat.pow(2).mean(dim=-1, keepdim=True)
    k_normed = (k_flat * torch.rsqrt(variance + self._k_norm_eps)).to(all_k.dtype)
    # Apply per-layer weights: [L, 1, 1, hd] broadcast over [L, num_ctx, nkv, hd]
    k_norm_weight = self._stacked_k_norm_weights.unsqueeze(1).unsqueeze(1)
    all_k_normed = k_normed.reshape(L, num_ctx, nkv, hd) * k_norm_weight

    # --- Step 3: RoPE (K-only, no full-tensor clone) ---
    # key=None makes the NPU RoPE path (forward_oot -> npu_mrope) use a
    # 1-head throwaway key buffer instead of a clone of K.
    # NOTE: the Ascend RoPE path is out-of-place (registered with
    # mutates_args=[]), so the rotated K must be rebound from the return
    # value; discarding it would leave K un-rotated in the KV cache.
    all_k_flat = all_k_normed.reshape(L * num_ctx, kv).contiguous()
    positions_repeated = context_positions.repeat(L)
    all_k_flat, _ = self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, None)

    if context_slot_mapping is None:
        return

    # --- Step 4: Per-layer cache insert (keep per-layer, but K is now contiguous) ---
    all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
    all_v_contig = all_v.contiguous()
    per_layer = isinstance(context_slot_mapping, (list, tuple))
    for i in range(L):
        slot_mapping = context_slot_mapping[i] if per_layer else context_slot_mapping
        if slot_mapping is None:
            continue
        attn = self._attn_layers[i]
        kv_cache = attn.kv_cache
        attn.impl.do_kv_cache_update(
            attn,
            all_k_final[i],
            all_v_contig[i],
            kv_cache,
            slot_mapping,
        )


DFlashQwen3Model.precompute_and_store_context_kv = precompute_and_store_context_kv

_orig_read_mask_embedding = DFlashQwen3ForCausalLM._read_mask_embedding


def _patched_read_mask_embedding(self):
    try:
        return _orig_read_mask_embedding(self)
    except Exception:
        return None


DFlashQwen3ForCausalLM._read_mask_embedding = _patched_read_mask_embedding
