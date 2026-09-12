import os
import sys
import torch

print("[TURBOQUANT] patch_turboquant.py LOADED, env=", os.environ.get("VLLM_ASCEND_ENABLE_TURBOQUANT", "NOT SET"), flush=True)

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

_orig_reshape_and_cache = AscendAttentionBackendImpl.reshape_and_cache
_tq_cache: dict[int, object] = {}
_call_count = 0


def _get_tq(dim: int):
    if dim not in _tq_cache:
        from turboquant import TurboQuantMSE
        bits = int(os.environ.get("VLLM_ASCEND_TURBOQUANT_BITS", "4"))
        _tq_cache[dim] = TurboQuantMSE(dim=dim, bits=bits, device="cpu")
        print(f"[TURBOQUANT] created {bits}-bit quantizer for dim={dim}", flush=True)
    return _tq_cache[dim]


def _turboquant_reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
    global _call_count
    enabled = os.environ.get("VLLM_ASCEND_ENABLE_TURBOQUANT", "0") == "1"
    if _call_count == 0:
        print(f"[TURBOQUANT] reshape_and_cache CALLED, enabled={enabled}", flush=True)
    _call_count += 1

    if enabled and len(kv_cache) > 1:
        n = attn_metadata.num_actual_tokens
        orig_device = key.device
        orig_dtype = key.dtype
        head_dim = key.shape[-1]
        tq = _get_tq(head_dim)

        k_flat = key[:n].float().cpu().reshape(-1, head_dim)
        v_flat = value[:n].float().cpu().reshape(-1, head_dim)

        k_idx, k_norms = tq.quantize(k_flat)
        k_lossy = tq.dequantize(k_idx, k_norms).reshape(key[:n].shape)

        v_idx, v_norms = tq.quantize(v_flat)
        v_lossy = tq.dequantize(v_idx, v_norms).reshape(value[:n].shape)

        key = torch.cat([k_lossy.to(orig_dtype).to(orig_device), key[n:]], dim=0)
        value = torch.cat([v_lossy.to(orig_dtype).to(orig_device), value[n:]], dim=0)

    return _orig_reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output)


AscendAttentionBackendImpl.reshape_and_cache = _turboquant_reshape_and_cache
print("[TURBOQUANT] monkey-patch applied to AscendAttentionBackendImpl.reshape_and_cache", flush=True)
