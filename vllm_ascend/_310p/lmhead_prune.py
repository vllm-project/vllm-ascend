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
"""Vocabulary-pruned lm_head for 310P decode.

Decode on one 310P is weight-streaming-bound and the 248k-row lm_head is read
up to k+1 times per MTP step (~5.9 ms/read). Pruning the head to a keep-list
(calibration ids + specials + byte-fallback tokens + all-Cyrillic) cuts that
read cost ~3x. Rows, deq scales and quant bias are sliced from the SERVING
checkpoint's already-quantized lm_head, so kept logits are bit-identical to
the unpruned model - no additional quantization error. Pruned logits are
scattered into a full-vocab -inf tensor through a handful of contiguous
segment copies (NPU index_copy is pathologically slow), so token ids stay
canonical for the sampler, rejection sampler, detokenizer and embeddings.
Ids outside the keep-list can never be sampled; byte-fallback tokens keep
all text representable.

Enable with VLLM_LMHEAD_PRUNE_PACK=/path/pack.pt, a dict with
{"mode": "int8", "weight" [K,H] int8 ND, "deq_scale" [K] i64,
 "quant_bias" [K] i32, "keep_ids" [K] i64, "inv_map" [V] i64 (position in the
 pruned head, or K for pruned-out ids), "orig_vocab": int}.
"""

import os
import types

import torch
import torch.nn.functional as F

from vllm.logger import logger

from vllm_ascend.utils import maybe_trans_nz

_ENV = "VLLM_LMHEAD_PRUNE_PACK"


def _make_pruned_compute_logits(lm_head, inv_map: torch.Tensor, vocab_size: int):
    pad_cache: dict[int, torch.Tensor] = {}

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pruned = lm_head.quant_method.apply(lm_head, hidden_states)
        # Expand to the canonical vocab with a single last-dim gather; column
        # K of the cached pad buffer stays -inf for pruned-out ids. Kept to
        # two device ops - eager dispatch overhead at M=1 otherwise eats the
        # pruned matmul's savings.
        pruned2d = pruned.reshape(-1, pruned.shape[-1])
        m, k = pruned2d.shape
        pad = pad_cache.get(m)
        if pad is None:
            pad = torch.full((m, k + 1), torch.finfo(pruned.dtype).min, dtype=pruned.dtype, device=pruned.device)
            pad_cache[m] = pad
        pad[:, :k].copy_(pruned2d)
        full = pad.index_select(-1, inv_map)
        return full.reshape(*pruned.shape[:-1], vocab_size)

    return compute_logits


def maybe_prune_lm_head(*models: object) -> None:
    """Swap each model's compute_logits for the pruned projection.

    No-op unless VLLM_LMHEAD_PRUNE_PACK is set. Accepts any object exposing
    compute_logits + lm_head (target model, MTP drafter model). Models
    sharing one lm_head module are pruned once.
    """
    pack_path = os.environ.get(_ENV, "")
    if not pack_path:
        return
    pack = torch.load(pack_path, map_location="cpu")
    if pack.get("mode") != "int8":
        raise NotImplementedError(f"unsupported prune pack mode {pack.get('mode')!r}")
    inv_map = pack["inv_map"].npu()
    pruned_heads: set[int] = set()
    for model in models:
        if model is None or not hasattr(model, "compute_logits"):
            continue
        # lm_head may live on a nested language model (VL-style wrappers
        # delegate compute_logits); walk the usual containers to find it.
        owner = model
        for _ in range(4):
            if getattr(owner, "lm_head", None) is not None:
                break
            nxt = getattr(owner, "language_model", None) or getattr(owner, "model", None)
            if nxt is None or nxt is owner:
                break
            owner = nxt
        lm_head = getattr(owner, "lm_head", None)
        if lm_head is None:
            logger.warning("lm_head prune: no lm_head found on %s; skipped", type(model).__name__)
            continue
        proc = getattr(owner, "logits_processor", None)
        vocab_size = getattr(proc, "org_vocab_size", None) or pack["orig_vocab"]
        scale = getattr(proc, "scale", 1.0)
        if scale != 1.0 or getattr(proc, "soft_cap", None):
            raise NotImplementedError("lm_head pruning supports scale=1.0 / no soft cap only.")
        if id(lm_head) not in pruned_heads:
            # Mirror AscendW8A8Static process_weights_after_loading's weight
            # treatment on the sliced rows; activation-quant tensors
            # (aclnn_input_*) depend only on the hidden dim and stay valid.
            dev = lm_head.weight.data.device
            lm_head.weight.data = maybe_trans_nz(pack["weight"].to(dev)).transpose(0, 1)
            lm_head.deq_scale.data = pack["deq_scale"].to(dev)
            lm_head.quant_bias.data = pack["quant_bias"].to(dev)
            pruned_heads.add(id(lm_head))
        fn = _make_pruned_compute_logits(lm_head, inv_map[:vocab_size], vocab_size)
        model.compute_logits = types.MethodType(fn, model)
        logger.info(
            "lm_head pruned: %s vocab %d -> %d rows (embedding-gather scatter)",
            type(model).__name__,
            vocab_size,
            pack["weight"].shape[0],
        )
