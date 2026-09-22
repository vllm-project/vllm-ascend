#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Honour the sliding window in the encoder-only (pooling) attention path.

Root cause
----------
``AscendAttentionBackendImpl._forward_encoder_attention`` calls
``torch_npu.npu_fusion_attention`` with the default ``sparse_mode=0`` and no
``atten_mask``. ``sparse_mode=0`` means "the mask (if any) is a full pairwise
mask", so passing no mask is equivalent to *unbounded* attention. Every
``sliding_attention`` layer of an encoder-only (``--runner pooling``) model
therefore degenerates into full attention.

The causal path (``forward_impl``) does not have this problem: it forwards
``self.sliding_window`` as ``pre_tokens`` with ``sparse_mode=4``. Only the
encoder-only branch drops the window.

Impact
------
Measured on a Laya decision encoder (28 layers, ``local_attention=128`` ->
``sliding_window=65``, 18 sliding layers) with a fp32 CPU reference of the same
checkpoint:

==================================  ==============
attention in the encoder             max |dlogit|
==================================  ==============
sliding window (reference)               0.1105
full attention (what vLLM did)          12.8125
==================================  ==============

So the missing window - and nothing else - is what made the ported model
disagree with the reference.

How
---
``npu_fusion_attention`` accepts an explicit ``[N, N]`` boolean ``atten_mask``
with ``sparse_mode=0``, where ``True`` blocks a (query, key) pair. In ``TND``
layout the mask is indexed by the *global* query index while the operator still
restricts every query to its own sequence, so a plain band
``abs(i - j) >= sliding_window`` implements exactly the same visibility as the
causal path (``sliding_window`` is inclusive-boundary, i.e. the window is
``2 * sliding_window - 1`` tokens wide).

Two further op constraints were verified on 910B4 / CANN 9.1.0:

* the mask must be no larger than the number of tokens passed as ``query``,
  hence the ``query/key/value`` are trimmed to ``actual_seq_qlen[-1]``
  (padding rows produce garbage anyway and are never pooled);
* ``actual_seq_qlen`` must stay a python ``list`` of cumulative lengths -
  trimming makes the trailing ``+ [0]`` of the original code unnecessary, and
  that trailing ``0`` would otherwise corrupt the mask indexing.

When the longest sequence fits inside a single window
(``max_seq_len <= sliding_window``, because the boundary is inclusive) the band
blocks nothing and the patch falls back to the original maskless call, costing
nothing extra. Laya requests are <= 512 tokens, so only long requests pay for
the mask.

``vllm_ascend.attention.attention_v1`` cannot be imported here: the platform
patch package is loaded from ``vllm_ascend/__init__.py``, before
``vllm_ascend.ops`` finishes importing, and reaching ``attention_v1`` at that
point raises a circular-import error through
``device_op`` <-> ``ops.fused_moe``. The class is therefore patched lazily from
``EncoderOnlyAttention.__init__``, which only runs once a model is built - long
after the import graph has settled.

Related PR (if no, explain why):
    Ascend-only: the bug lives in the vllm-ascend encoder-only attention
    branch, so the fix belongs here and cannot go to vLLM.

Future Plan:
    Drop this patch once ``_forward_encoder_attention`` forwards
    ``self.sliding_window`` (e.g. as ``sparse_mode=4`` + ``pre_tokens``) on its
    own, or once CANN exposes a windowed 2-D mask for ``TND`` at
    ``sparse_mode=0`` without the size restriction.
"""

import torch
import torch_npu
from vllm.model_executor.layers.attention import EncoderOnlyAttention

_ORIGINAL_ENCODER_ONLY_ATTENTION_INIT = EncoderOnlyAttention.__init__
_ORIGINAL_FORWARD_ENCODER_ATTENTION = None
_INSTALLED = False

# Band masks are keyed by (num_tokens, sliding_window). A serving batch only
# ever uses a handful of distinct token counts, so a small cache removes the
# per-layer rebuild while keeping the memory bounded.
_BAND_MASK_CACHE: dict[tuple[int, int], torch.Tensor] = {}
_BAND_MASK_CACHE_LIMIT = 32


def _get_band_mask(num_tokens: int, sliding_window: int, device: torch.device) -> torch.Tensor:
    """Boolean ``[num_tokens, num_tokens]`` mask, ``True`` where blocked.

    ``sliding_window`` is inclusive: position ``i`` attends to ``j`` when
    ``abs(i - j) <= sliding_window - 1``.
    """
    cache_key = (num_tokens, sliding_window)
    mask = _BAND_MASK_CACHE.get(cache_key)
    if mask is None or mask.device != device:
        index = torch.arange(num_tokens, device=device)
        mask = (index[:, None] - index[None, :]).abs() >= sliding_window
        if len(_BAND_MASK_CACHE) >= _BAND_MASK_CACHE_LIMIT:
            _BAND_MASK_CACHE.clear()
        _BAND_MASK_CACHE[cache_key] = mask
    return mask


def _forward_encoder_attention_with_sliding_window(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
) -> torch.Tensor:
    sliding_window = getattr(self, "sliding_window", None)
    if sliding_window is None:
        return _ORIGINAL_FORWARD_ENCODER_ATTENTION(self, query, key, value, attn_metadata, output)

    # Cumulative sequence lengths; the operator requires a plain python list.
    actual_seq_qlen = list(attn_metadata.actual_seq_lengths_q)
    num_tokens = actual_seq_qlen[-1]
    max_seq_len = max(
        actual_seq_qlen[i] - (actual_seq_qlen[i - 1] if i else 0) for i in range(len(actual_seq_qlen))
    )

    fia_kwargs = dict(
        query=query[:num_tokens],
        key=key[:num_tokens],
        value=value[:num_tokens],
        head_num=self.num_heads,
        input_layout="TND",
        scale=self.scale,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_qlen,
    )
    # A whole sequence fits in one window: nothing is masked, so keep the
    # original maskless fast path. The window boundary is inclusive, so every
    # pair of a sequence is visible as soon as ``max_seq_len <= sliding_window``.
    if max_seq_len > sliding_window:
        fia_kwargs["atten_mask"] = _get_band_mask(num_tokens, sliding_window, query.device)
        fia_kwargs["sparse_mode"] = 0

    attn_output = torch_npu.npu_fusion_attention(**fia_kwargs)[0]
    output[:num_tokens] = attn_output
    return output


def _install_sliding_window_patch() -> None:
    global _INSTALLED, _ORIGINAL_FORWARD_ENCODER_ATTENTION
    if _INSTALLED:
        return
    # Imported here, not at module import time: see the module docstring.
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

    _ORIGINAL_FORWARD_ENCODER_ATTENTION = AscendAttentionBackendImpl._forward_encoder_attention
    AscendAttentionBackendImpl._forward_encoder_attention = _forward_encoder_attention_with_sliding_window
    _INSTALLED = True


def _encoder_only_attention_init(self, *args, **kwargs) -> None:
    _ORIGINAL_ENCODER_ONLY_ATTENTION_INIT(self, *args, **kwargs)
    _install_sliding_window_patch()


EncoderOnlyAttention.__init__ = _encoder_only_attention_init
