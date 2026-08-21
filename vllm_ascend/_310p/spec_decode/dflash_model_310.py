#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""310P dflash/dspark model patches.

``patch_qwen3_dflash`` is not loaded on 310P (see ``patch/worker/__init__.py``),
so the generic ``precompute_and_store_context_kv`` never lands.  This module
provides a 310P-safe version that:

* Applies RoPE per layer over ``num_ctx`` positions (within the 310P cos/sin
  capacity buffer) with the drafting cos/sin refresh flag enabled, instead of
  the generic fused ``L * num_ctx`` RoPE call that overflows the buffer.
* Reuses ``do_kv_cache_update`` for the NZ 5D KV cache write (same as normal
  310P attention).
"""

import torch
import torch.nn.functional as F
from vllm.logger import logger
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM

from vllm_ascend._310p.ops.rotary_embedding import (
    AscendRotaryEmbedding310,
    set_full_decode_draft_rope_source_310,
)
from vllm_ascend.utils import vllm_version_is


def precompute_and_store_context_kv_310(
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
    all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
    all_k = all_kv[0]  # [L, num_ctx, nkv, hd]
    all_v = all_kv[1]  # [L, num_ctx, nkv, hd]
    context_probe = self.__dict__.get("_fdo_context_probe")
    if context_probe is not None:
        context_probe.capture_context_inputs(
            context_states=context_states,
            context_positions=context_positions,
            normed_context_states=normed_context_states,
            slot_mapping=context_slot_mapping,
        )

    # --- Per-layer RMSNorm K + RoPE (310P-safe) ---
    # Generic path fuses RoPE over [L * num_ctx, kv], which exceeds the 310P
    # cos/sin capacity buffer and never refreshes cos/sin without the drafting
    # flag.  Apply RoPE per layer over num_ctx positions instead.
    #
    # Two 310P-specific correctness requirements handled here:
    #   1. AscendRotaryEmbedding310._rope_forward_oot returns NEW tensors for
    #      the head_size==128 / rotary_dim==64 paths (npu_apply_rotary_pos_emb),
    #      so the RoPE'd K must be taken from the return value, not read back
    #      from the (unmodified) input buffer.
    #   2. The drafting cos/sin refresh flag may already be True (this runs
    #      inside AscendSpecDecodeBaseProposer310._run_merged_draft for the real
    #      draft flow) or False (profile/dummy_run). Save and restore the prior
    #      value so we never disable it for the enclosing draft-model forward.
    all_k_normed = torch.empty_like(all_k)
    prev_flag = AscendRotaryEmbedding310._is_drafting_update_enabled
    prev_rope_source = set_full_decode_draft_rope_source_310("context")
    AscendRotaryEmbedding310.set_rope_position_flag_310p(True)
    try:
        for i in range(L):
            k_norm_layer = self.layers[i].self_attn.k_norm
            k_normed = k_norm_layer(all_k[i]).reshape(num_ctx, kv)
            if context_probe is not None:
                context_probe.capture_context_k_norm(
                    layer_index=i,
                    k_norm_input=all_k[i],
                    k_norm_output=k_normed,
                )
            tmpv = k_normed.clone()
            k_roped, _ = self.layers[i].self_attn.rotary_emb(context_positions, k_normed, tmpv)
            all_k_normed[i] = k_roped.reshape(num_ctx, nkv, hd)
            if context_probe is not None:
                context_probe.capture_context_rope(
                    layer_index=i,
                    k_rope=k_roped,
                    value=all_v[i],
                )
    finally:
        AscendRotaryEmbedding310.set_rope_position_flag_310p(prev_flag)
        set_full_decode_draft_rope_source_310(prev_rope_source)

    if context_slot_mapping is None:
        return

    # --- Per-layer cache insert (NZ 5D via Ascend310PDeviceAdaptor) ---
    per_layer = isinstance(context_slot_mapping, (list, tuple))
    for i in range(L):
        slot_mapping = context_slot_mapping[i] if per_layer else context_slot_mapping
        if slot_mapping is None:
            continue
        attn = self._attn_layers[i]
        kv_cache = attn.kv_cache
        attn.impl.do_kv_cache_update(
            attn,
            all_k_normed[i],
            all_v[i],
            kv_cache,
            slot_mapping,
        )


def patch_dflash_read_mask_embedding_310() -> None:
    """Fallback when mask embedding weights are absent (same as generic patch)."""
    if vllm_version_is("0.24.0") or not hasattr(DFlashQwen3ForCausalLM, "_read_mask_embedding"):
        return

    _orig_read_mask_embedding = DFlashQwen3ForCausalLM._read_mask_embedding

    def _patched_read_mask_embedding(self):
        try:
            return _orig_read_mask_embedding(self)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "dflash _read_mask_embedding failed, falling back to None: %s: %s",
                type(exc).__name__,
                exc,
            )
            return None

    DFlashQwen3ForCausalLM._read_mask_embedding = _patched_read_mask_embedding  # type: ignore[method-assign]
