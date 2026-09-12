# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Activate the TurboQuant KV cache on 310P (Tier 0).

Inert unless VLLM_ASCEND_TURBOQUANT=1.

    VLLM_ASCEND_TURBOQUANT=1 VLLM_ASCEND_TQ_BITS=3 vllm serve <model> ...

`platform.py` resolves the 310P attention backend by dotted path
(`..._310p.attention.attention_v1.AscendAttentionBackend310`), so rather than
registering a new backend -- which would also require teaching vLLM core a new
`kv_cache_dtype` string -- we patch that class's two extension points:
`get_kv_cache_shape` (packed shape) and `get_impl_cls` (quantizing impl).

Scope: PrefillNoCache + DecodeOnly. Chunked prefill and prefix caching read the
paged cache through `_npu_paged_attention_splitfuse`, which cannot parse packed
codes; launch with them disabled.
"""

import os

_ENABLED = os.environ.get("VLLM_ASCEND_TURBOQUANT", "0") == "1"
_BITS = int(os.environ.get("VLLM_ASCEND_TQ_BITS", "3"))
_K_BITS = int(os.environ.get("VLLM_ASCEND_TQ_K_BITS", str(_BITS)))
_V_BITS = int(os.environ.get("VLLM_ASCEND_TQ_V_BITS", str(_BITS)))
# Tier 2 = the custom AscendC kernels (dequant inside attention). Tier 0 keeps
# the gather+dense path. Opt-in because Tier 2 needs K and V at the same width.
_TIER2 = os.environ.get("VLLM_ASCEND_TQ_TIER2", "0") == "1"


def _log(msg):
    print(f"[TURBOQUANT] {msg}", flush=True)


def _report(shape, block_size, num_kv_heads, head_size, tag):
    """Log the ACTUAL allocated KV geometry so footprint can be compared without
    relying on vLLM's `GPU KV cache size`, which is derived from FullAttentionSpec
    (fp16 head_size) and therefore cannot see the packed layout."""
    elems = 1
    for d in shape[1:]:
        elems *= d
    kv_bytes = elems * 2 * 2                      # fp16 elements, x2 for K and V
    slots = shape[1] * block_size
    _log(f"{tag} shape={shape} block_size={block_size} kv_heads={num_kv_heads} head_dim={head_size}")
    _log(f"{tag} per-layer KV = {kv_bytes / 2**20:.1f} MiB over {shape[1]} blocks "
         f"({slots} slots) -> BYTES/TOKEN = {kv_bytes / slots:.1f}")


if not _ENABLED:
    # inert, but still report baseline geometry so the two runs are comparable
    from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackend310

    _base_shape_fn = AscendAttentionBackend310.get_kv_cache_shape

    def _spy(num_blocks, block_size, num_kv_heads, head_size, cache_type=""):
        s = _base_shape_fn(num_blocks, block_size, num_kv_heads, head_size, cache_type)
        _report(s, block_size, num_kv_heads, head_size, "BASELINE")
        return s

    AscendAttentionBackend310.get_kv_cache_shape = staticmethod(_spy)

def _patch_page_size_for_norms():
    """Budget the Tier-2 norm planes into page_size_bytes.

    vLLM sizes the cache as `num_blocks = kv_cache_tensor.size // page_size_bytes`,
    and the turboquant backend's get_kv_cache_shape describes only the PACKED
    payload. The norm planes are separate tensors the impl allocates, so without
    this they are invisible to the block accounting and the allocation is
    over-committed.

    `AttentionSpec.page_size_padded` is vLLM's own hook for exactly this: its
    page_size_bytes already grows for per-token-head scales, with the comment
    that they live in "separate tensors managed by the attention backend, but
    the memory is carved from the raw KV cache allocation so it must be budgeted
    here". That built-in formula assumes num_kv_heads*fp32, which UNDER-counts
    this layout 2x (16 lanes of fp16), so the mechanism is reused, not the
    constant.

    At d=256, NKV=4, block=128, b=4: payload 131072 B + norms 8192 B = +6.25%,
    making the honest saving vs fp16 3.77x rather than 4.00x.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec

    # imported here, not from the caller's scope: _enable() binds it locally
    from vllm_ascend.attention.turboquant_attn_310_tier2 import norm_bytes_per_block

    if getattr(AttentionSpec, "_tq_page_patched", False):
        return
    _orig = AttentionSpec.page_size_bytes.fget

    def _padded(self):
        base = _orig(self)
        if self.page_size_padded is not None:
            return base          # caller already pinned it; do not double-add
        return base + norm_bytes_per_block(self.block_size)

    AttentionSpec.page_size_bytes = property(_padded)
    AttentionSpec._tq_page_patched = True
    _log("page_size_bytes now budgets the norm planes "
         f"(+{norm_bytes_per_block(128)} B per 128-token block)")


if _ENABLED:
    from vllm_ascend._310p.attention.attention_v1 import AscendAttentionBackend310
    from vllm_ascend.attention.turboquant_attn_310 import (
        AscendTurboQuantAttentionBackend310,
        AscendTurboQuantAttentionBackendImpl310,
    )

    AscendTurboQuantAttentionBackend310.tq_k_bits = _K_BITS
    AscendTurboQuantAttentionBackend310.tq_v_bits = _V_BITS
    AscendTurboQuantAttentionBackendImpl310.tq_k_bits = _K_BITS
    AscendTurboQuantAttentionBackendImpl310.tq_v_bits = _V_BITS

    def _tq_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_type=""):
        shape = AscendTurboQuantAttentionBackend310.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_type
        )
        _report(shape, block_size, num_kv_heads, head_size, f"TURBOQUANT k{_K_BITS}v{_V_BITS}")
        return shape

    AscendAttentionBackend310.get_kv_cache_shape = staticmethod(_tq_kv_cache_shape)

    if _TIER2:
        from vllm_ascend.attention.turboquant_attn_310_tier2 import (
            AscendTurboQuantTier2AttentionBackendImpl310,
            norm_bytes_per_block,
        )

        if _K_BITS != _V_BITS:
            raise ValueError(
                f"VLLM_ASCEND_TQ_TIER2=1 needs matching widths; got k={_K_BITS} "
                f"v={_V_BITS}. The kernels take a single `bits` per op."
            )
        AscendAttentionBackend310.get_impl_cls = staticmethod(
            lambda: AscendTurboQuantTier2AttentionBackendImpl310
        )
        # OFF by default: padding page_size_bytes breaks
        #   assert kv_cache_tensor.size % kv_cache_spec.page_size_bytes == 0
        # in model_runner_310p, because vLLM sizes the KV tensor against the
        # UNPADDED page (32768 B here) and the padded one (36864 B) no longer
        # divides it. Consequence while off: the norm planes (+12.5% at
        # block_size=64) are outside vLLM's block accounting, so the allocation
        # is over-committed by that much -- compensate with a slightly lower
        # gpu_memory_utilization. Proper fix is to make the planner aware of the
        # padded page rather than to patch the property after the fact.
        if os.environ.get("VLLM_ASCEND_TQ_PAGE_PAD", "0") == "1":
            _patch_page_size_for_norms()
        _log(f"ENABLED TIER 2 (AscendC kernels) k={_K_BITS} v={_V_BITS} bits")
    else:
        AscendAttentionBackend310.get_impl_cls = staticmethod(
            lambda: AscendTurboQuantAttentionBackendImpl310
        )
        _log(f"ENABLED k={_K_BITS} v={_V_BITS} bits -- patched AscendAttentionBackend310")



# --- per-channel RMS diagnostic (VLLM_ASCEND_TQ_STATS=1) --------------------
# Must live in a patch module: importing vllm_ascend.attention directly from a
# standalone script circular-imports via vllm_ascend.device.device_op, because
# the platform plugin is only half-initialised at that point.
if os.environ.get("VLLM_ASCEND_TQ_STATS") == "1":
    import atexit
    import json

    import numpy as np
    import torch as _torch

    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl as _A

    _stats: dict = {}
    _orig_rac = _A.reshape_and_cache

    def _stats_rac(self, query, key, value, kv_cache, attn_metadata, output):
        n = getattr(attn_metadata, "num_actual_tokens", None)
        if n:
            for tag, t in (("K", key[:n]), ("V", value[:n])):
                x = t.detach().float().reshape(-1, t.shape[-1])
                s = _stats.setdefault((id(self), tag), [None, 0])
                sq = (x ** 2).sum(0).cpu().numpy()
                s[0] = sq if s[0] is None else s[0] + sq
                s[1] += x.shape[0]
        return _orig_rac(self, query, key, value, kv_cache, attn_metadata, output)

    _A.reshape_and_cache = _stats_rac

    @atexit.register
    def _dump():
        out = []
        for (lid, tag), (ss, cnt) in _stats.items():
            if ss is None or not cnt:
                continue
            rms = np.sqrt(ss / cnt)
            med = float(np.median(rms))
            if med < 1e-9:
                continue
            out.append({"tag": tag, "max_over_med": float(rms.max() / med),
                        "p99_over_med": float(np.percentile(rms, 99) / med),
                        "frac_gt_3x": float((rms > 3 * med).mean()),
                        "frac_gt_10x": float((rms > 10 * med).mean())})
        path = os.environ.get("VLLM_ASCEND_TQ_STATS_OUT", "/root/tq_ph0/chan_stats.json")
        try:
            with open(path, "w") as f:
                json.dump(out, f)
            _log(f"channel stats written: {path} ({len(out)} entries)")
        except Exception as e:  # noqa
            _log(f"stats dump failed: {e}")
