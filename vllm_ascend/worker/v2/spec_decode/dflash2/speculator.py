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
# This file is a part of the vllm-ascend project.
#

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, spec_step_seed
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)


@triton.jit
def _selector_walk_kernel_ascend(
    scores_ptr,
    candidate_ptr,
    sample_pos_ptr,
    req_state_ptr,
    temperature_ptr,
    seeds_ptr,
    tokens_ptr,
    realized_scores_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """FP32 Ascend selector walk with upstream random-stream semantics.

    Reuse upstream's integer hash and stable Gumbel transform, which do not
    require libdevice log1p or Philox's unsupported int64 multiplies. Keep the
    explicit FP64 rejection since NPU sampling supports only FP32.
    """
    tl.static_assert(not USE_FP64, "fp64 gumbel is not supported on NPU")
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    req_state = tl.load(req_state_ptr + row * num_steps)
    valid = req_state >= 0
    temperature = tl.load(temperature_ptr + req_state, mask=valid, other=0.0)
    seed = tl.load(seeds_ptr + req_state, mask=valid, other=0)
    # Match upstream: a non-probabilistic draft is always greedy, regardless
    # of the request temperature
    effective_temp = temperature if SAMPLE_PROBABILISTIC else 0.0
    previous = 0
    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask & valid,
            other=float("-inf"),
        ).to(tl.float32)
        candidate_base = flat * top_k
        candidates = tl.load(
            candidate_ptr + candidate_base + offsets,
            mask=mask & valid,
            other=0,
        )

        if effective_temp == 0.0:
            best = tl.max(scores, axis=0)
            index = tl.min(tl.where(scores == best, offsets, BLOCK_K), axis=0)
        else:
            position = tl.load(sample_pos_ptr + flat, mask=valid, other=0) - 1
            _, index = gumbel_noised_argmax(
                scores,
                candidates,
                mask & valid,
                spec_step_seed(seed, step),
                position,
                effective_temp,
                IS_DRAFTING=True,
                USE_FP64=False,
            )

        tl.store(
            realized_scores_ptr + candidate_base + offsets,
            scores,
            mask=mask & valid,
        )
        token = tl.load(candidate_ptr + candidate_base + index, mask=valid, other=0)
        tl.store(tokens_ptr + flat, token, mask=valid)
        previous = index


class AscendDFlash2Speculator(DFlash2Speculator, AscendDFlashSpeculator):
    """DFlash2 speculative drafter on Ascend NPUs.

    Every Ascend-specific behavior is inherited from AscendDFlashSpeculator:
    the NPU attention-metadata building (build_draft_attn_metadatas), the
    aclgraph manager wiring (init_cudagraph_manager), KV-cache/slot bookkeeping
    (set_attn), and the propose wrapper. Placing DFlash2Speculator first in
    the MRO makes its ``__init__`` (selector buffers and the always-allocated
    full-vocab draft logits) and its ``_generate_draft`` override win: the
    candidate top-k, selector scoring, sequential path walk, and draft-logits
    scatter run instead of plain DFlash drafting -- including inside the
    aclgraph capture, which captures ``self._generate_draft``.
    """

    # _generate_draft -> get_top_k_tokens samples on the vocab-sharded head
    # outside sample_draft; reject lmhead TP at construction (as DSpark).
    _lmhead_tp_sample_draft_supported = False
