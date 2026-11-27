#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Ascend Triton narrow collect for rare_chars (top1, chosen) logprobs.

Kept out of ``worker/v2/sample/logprob.py`` so the sampler logprob path stays
untouched by observability-only kernels.
"""

from __future__ import annotations

import torch
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON, tl, triton

_NARROW_COLLECT_UNAVAILABLE_LOGGED = False

# rare_chars narrow collect: top1 = -log S, chosen = (x_c - M) - log S.
_NARROW_COLLECT_BLOCK = 4096
_narrow_collect_workspace: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


@triton.jit
def _narrow_collect_partials(X, stride, PMax, PSum, N: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    offs = part * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row.to(tl.int64) * stride + offs, mask=offs < N, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    s = tl.sum(tl.exp(x - m), axis=0)
    s = tl.where(m == -float("inf"), 0.0, s)
    tl.store(PMax + row * PARTS + part, m)
    tl.store(PSum + row * PARTS + part, s)


@triton.jit
def _narrow_collect_finish(
    PMax, PSum, X, stride, Sampled, Out, N: tl.constexpr, PARTS: tl.constexpr, BLOCK_P: tl.constexpr
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_P)
    pm = tl.load(PMax + row * PARTS + offs, mask=offs < PARTS, other=-float("inf"))
    ps = tl.load(PSum + row * PARTS + offs, mask=offs < PARTS, other=0.0)
    m = tl.max(pm, axis=0)
    s = tl.sum(tl.where(pm == -float("inf"), 0.0, ps * tl.exp(pm - m)), axis=0)
    log_s = tl.log(s)
    tok = tl.load(Sampled + row).to(tl.int64)
    ok = (tok >= 0) & (tok < N)
    xc = tl.load(X + row.to(tl.int64) * stride + tok, mask=ok, other=float("nan")).to(tl.float32)
    bad = (m != m) | (m == -float("inf")) | (m == float("inf"))  # NaN or ±inf
    tl.store(Out + row * 2, tl.where(bad, float("nan"), -log_s))
    tl.store(Out + row * 2 + 1, tl.where(bad, float("nan"), (xc - m) - log_s))


def compute_top1_and_chosen_logprobs(
    logits: torch.Tensor,
    chosen_token_ids: torch.Tensor,
    *,
    packed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Return per-row (top1, chosen) logprobs; ``packed=True`` → ``[rows, 2]``."""
    global _NARROW_COLLECT_UNAVAILABLE_LOGGED
    if not HAS_TRITON:
        if not _NARROW_COLLECT_UNAVAILABLE_LOGGED:
            logger.warning("compute_top1_and_chosen_logprobs requires Triton (HAS_TRITON=False); fail closed")
            _NARROW_COLLECT_UNAVAILABLE_LOGGED = True
        raise RuntimeError("compute_top1_and_chosen_logprobs requires Triton")
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D [rows, vocab], got shape={tuple(logits.shape)}")
    batch_size, vocab_size = logits.shape
    ids = chosen_token_ids.reshape(-1).to(dtype=torch.int64, device=logits.device)
    if ids.numel() != batch_size:
        raise ValueError(f"chosen_token_ids length {ids.numel()} != logits rows {batch_size}")

    raw = logits if logits.stride(1) == 1 else logits.contiguous()
    if ids.stride(0) != 1:
        ids = ids.contiguous()

    parts = triton.cdiv(vocab_size, _NARROW_COLLECT_BLOCK)
    device_key = str(raw.device)
    ws = _narrow_collect_workspace.get(device_key)
    if ws is None or ws[0].shape[0] < batch_size or ws[0].shape[1] != parts:
        cap = triton.cdiv(batch_size, 64) * 64
        ws = _narrow_collect_workspace[device_key] = (
            torch.empty((cap, parts), dtype=torch.float32, device=raw.device),
            torch.empty((cap, parts), dtype=torch.float32, device=raw.device),
        )

    out = torch.empty((batch_size, 2), dtype=torch.float32, device=raw.device)
    block_p = max(16, triton.next_power_of_2(parts))
    _narrow_collect_partials[(batch_size, parts)](
        raw,
        raw.stride(0),
        ws[0],
        ws[1],
        vocab_size,
        parts,
        _NARROW_COLLECT_BLOCK,
        num_warps=4,
    )
    _narrow_collect_finish[(batch_size,)](
        ws[0],
        ws[1],
        raw,
        raw.stride(0),
        ids,
        out,
        vocab_size,
        parts,
        block_p,
        num_warps=1,
    )
    if packed:
        return out
    return out[:, 0], out[:, 1]
