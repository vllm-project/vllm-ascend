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

    # --- Fused RoPE across all layers ---
    # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
    # In-place RoPE: pass K as the "query" arg with key=None.
    all_k_flat = all_k_normed.view(L * num_ctx, kv)
    positions_repeated = context_positions.repeat(L)
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


def _apply_dflash_structural_swa(self) -> None:
    """Enable per-layer structural SWA on the Ascend FIA impl for DFlash
    drafts whose config marks some layers as ``sliding_attention``.

    The DFlash draft builds every layer as a full-attention layer
    (``Attention.sliding_window`` is None -> ``FullAttentionSpec``, i.e. a
    single KV-cache group). Mixed sliding/full DFlash checkpoints such as
    ``Qwen3.6-35B-A3B-DFlash-SW`` (``layer_types = [sliding_attention x5,
    full_attention]``, ``sliding_window = 4096``) need the sliding layers to
    run causal-band sliding-window attention.

    Ascend V1 draft attention only plumbs a single KV-cache group, so we
    cannot split the layers into ``SlidingWindowSpec`` + ``FullAttentionSpec``
    groups (that is the V2-only path in upstream vLLM PR #47914). Instead we
    keep one ``FullAttentionSpec`` group and set ``sliding_window`` directly
    on each sliding layer's FIA impl. Ascend FIA then runs ``sparse_mode=4``
    (``pre_tokens=window``, ``next_tokens=0`` == causal band) for those
    layers -- exactly what upstream assigns them (SWA layers default causal
    in PR #47914 ``_resolve_layer_attention``). Full layers keep
    ``sliding_window=None`` -> ``sparse_mode=0`` (non-causal). Applying the
    causal-band mask over the fully-retained cache is numerically equivalent
    to the upstream windowed-cache path. See ``swa-dflash-design.md``.
    """
    cfg = getattr(self, "config", None)
    if cfg is None:
        return
    layer_types = getattr(cfg, "layer_types", None)
    sliding_window = getattr(cfg, "sliding_window", None)
    # Only mixed/structural SWA configs need this; an absent layer_types or
    # sliding_window (the "standard" all-full DFlash) is left untouched.
    if not layer_types or not sliding_window:
        return
    for i, layer in enumerate(self.layers):
        if i >= len(layer_types):
            break
        if layer_types[i] == "sliding_attention":
            # Attention.impl is created eagerly in Attention.__init__; setting
            # sliding_window here (model construction, before ACL-graph
            # capture) makes FIA bake sparse_mode=4 into the captured graph.
            layer.self_attn.attn.impl.sliding_window = sliding_window


_orig_dflash_model_init = DFlashQwen3Model.__init__


def _patched_dflash_model_init(self, *args, **kwargs) -> None:
    _orig_dflash_model_init(self, *args, **kwargs)
    _apply_dflash_structural_swa(self)


DFlashQwen3Model.__init__ = _patched_dflash_model_init

_orig_read_mask_embedding = DFlashQwen3ForCausalLM._read_mask_embedding


def _patched_read_mask_embedding(self):
    try:
        return _orig_read_mask_embedding(self)
    except Exception:
        return None


DFlashQwen3ForCausalLM._read_mask_embedding = _patched_read_mask_embedding
