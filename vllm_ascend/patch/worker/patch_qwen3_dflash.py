import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.rotary_embedding import AscendRotaryEmbedding

# Bound the extra live native norm outputs during small context updates.
_MAX_STACKED_CONTEXT_NORM_BYTES = 4 * 1024 * 1024

_orig_build_fused_kv_buffers = DFlashQwen3Model._build_fused_kv_buffers


def _build_fused_kv_buffers(self):
    _orig_build_fused_kv_buffers(self)
    self._use_q_only_context_rope = (
        HAS_TRITON
        and get_current_vllm_config().use_v2_model_runner
        and type(self.layers[0].self_attn.rotary_emb) is AscendRotaryEmbedding
    )


def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | None = None,
) -> None:
    if not hasattr(self, "_use_q_only_context_rope"):
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
    if self._use_q_only_context_rope and all_k.numel() * all_k.element_size() <= _MAX_STACKED_CONTEXT_NORM_BYTES:
        # Retain native norm arithmetic while replacing six small output copies
        # with one stack. Large prefills retain bounded temporary storage.
        all_k_normed = torch.stack([self.layers[i].self_attn.k_norm(all_k[i]) for i in range(L)])
    else:
        all_k_normed = torch.empty_like(all_k)
        for i in range(L):
            k_norm_layer = self.layers[i].self_attn.k_norm
            all_k_normed[i] = k_norm_layer(all_k[i])

    # --- Fused RoPE across all layers ---
    # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
    # In-place RoPE: pass K as the "query" arg with key=None.
    all_k_flat = all_k_normed.view(L * num_ctx, kv)
    positions_repeated = context_positions.repeat(L)
    if self._use_q_only_context_rope:
        # Context has only K. The cloned second RoPE input was discarded.
        all_k_flat, _ = self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, None)
    else:
        tmpv = all_k_flat.clone()
        self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)

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
DFlashQwen3Model._build_fused_kv_buffers = _build_fused_kv_buffers

_orig_read_mask_embedding = DFlashQwen3ForCausalLM._read_mask_embedding


def _patched_read_mask_embedding(self):
    try:
        return _orig_read_mask_embedding(self)
    except Exception:
        return None


DFlashQwen3ForCausalLM._read_mask_embedding = _patched_read_mask_embedding
