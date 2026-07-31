import torch
import torch.nn.functional as F
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)

from vllm_ascend.utils import is_310p


def apply_context_rope(*, layers, all_k_normed, context_positions, num_layers, num_ctx, kv_size):
    """Rotate the fused context K and return it as [num_layers * num_ctx, kv_size].

    The return value must be used on every device. On A2/A3 the rotary op reshapes
    with .contiguous() and mutates that storage, so the result aliases the input
    and taking it is a no-op; on 310P npu_apply_rotary_pos_emb allocates new
    tensors, so ignoring the result silently feeds unrotated K into the cache.

    310P additionally rotates one layer at a time. The fused [L * num_ctx] form
    cannot be used there because the drafting-time rotary path fills a module
    global cos/sin buffer sized max_num_batched_tokens, which L * num_ctx
    overflows for any realistic context with a multi-layer drafter.
    """
    if not is_310p():
        all_k_flat = all_k_normed.view(num_layers * num_ctx, kv_size)
        positions_repeated = context_positions.repeat(num_layers)
        rotated, _ = layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, all_k_flat.clone())
        return rotated

    per_layer = all_k_normed.view(num_layers, num_ctx, kv_size)
    for i in range(num_layers):
        layer_k = per_layer[i]
        # Each layer gets its own rotary module rather than layers[0]'s. They are
        # required to agree (_build_fused_kv_buffers asserts it), so this matches
        # the fused path while not depending on that assertion holding.
        rotated, _ = layers[i].self_attn.rotary_emb(context_positions, layer_k, layer_k.clone())
        per_layer[i] = rotated
    return all_k_normed.view(num_layers * num_ctx, kv_size)


def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | None = None,
) -> None:
    if not hasattr(self, "_num_attn_layers"):
        self._build_fused_kv_buffers()

    num_ctx = context_states.shape[0]
    L = self._num_attn_layers
    kv = self._kv_size
    hd = self._head_dim
    nkv = self._num_kv_heads

    # --- Fused KV projection (one GEMM for all layers) ---
    normed_context_states = self.hidden_norm(context_states)
    all_kv_flat = F.linear(normed_context_states, self._fused_kv_weight, self._fused_kv_bias)
    # Single contiguous copy that separates K/V and transposes to
    # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
    # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
    all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
    all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
    all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

    # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
    all_k_normed = torch.empty_like(all_k)
    for i in range(L):
        k_norm_layer = self.layers[i].self_attn.k_norm
        all_k_normed[i] = k_norm_layer(all_k[i])

    # --- Context RoPE (fused across layers off 310P, per layer on it) ---
    # K is passed in the "query" argument slot with a throwaway tensor as "key".
    all_k_flat = apply_context_rope(
        layers=self.layers,
        all_k_normed=all_k_normed,
        context_positions=context_positions,
        num_layers=L,
        num_ctx=num_ctx,
        kv_size=kv,
    )

    if context_slot_mapping is None:
        return

    # --- Per-layer cache insert ---
    all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
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
            all_v[i],
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
