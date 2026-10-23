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

import torch
from vllm.forward_context import BatchDescriptor, create_forward_context, override_forward_context
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32
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
    """Ascend variant of upstream ``_selector_walk_kernel``.

    triton-ascend cannot lower ``tldevice.log1p`` (AST parse fails even for
    the greedy path), so the Gumbel noise uses the algebraically equivalent
    ``log(1 - u)`` transform. The signature mirrors upstream exactly so the
    inherited ``DFlash2Speculator._sample_path`` can call it unmodified:
    ``SAMPLE_PROBABILISTIC`` gates greedy vs. probabilistic, and ``USE_FP64``
    is rejected via ``tl.static_assert`` (NPU Triton has no fp64
    philox/rand path), matching the other vllm_ascend gumbel kernels.
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
            position = tl.load(sample_pos_ptr + flat) - 1
            # triton-ascend's philox requires int32 seed/offset operands
            # (int64 counters lower to 64-bit multiplies whose runtime
            # helper, __multi3, is unavailable); seeds, token ids and
            # positions all fit int32.
            gumbel_seed = tl.randint(seed.to(tl.int32), position.to(tl.int32))
            uniform = tl_rand32(gumbel_seed, candidates.to(tl.int32), includes_zero=False)
            noise = -tl.log(-tl.log(1.0 - uniform))
            sampled_scores = tl.where(mask, scores / effective_temp + noise, -float("inf"))
            best = tl.max(sampled_scores, axis=0)
            index = tl.min(tl.where(sampled_scores == best, offsets, BLOCK_K), axis=0)

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

    The common Ascend behavior is inherited from AscendDFlashSpeculator:
    the NPU attention-metadata building (build_draft_attn_metadatas), the
    aclgraph manager wiring (init_cudagraph_manager), KV-cache/slot bookkeeping
    (set_attn), and the propose wrapper. Placing DFlash2Speculator first in
    the MRO makes its ``__init__`` (selector buffers) and its
    ``_generate_draft`` override win: the
    candidate top-k, selector scoring, sequential path walk, and draft-logits
    scatter run instead of plain DFlash drafting -- including inside the
    aclgraph capture, which captures ``self._generate_draft``. Only selector
    scoring receives a local compilation context after the model forward exits.
    """

    # _generate_draft -> get_top_k_tokens samples on the vocab-sharded head
    # outside sample_draft; reject lmhead TP at construction (as DSpark).
    _lmhead_tp_sample_draft_supported = False

    def _score_candidates(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Upstream closes _run_model's context before scoring. The selector's
        # compiled graph wrapper still needs one, even though it has no
        # attention or distributed layers. Its dynamic axis counts requests,
        # not the flattened draft tokens used by the backbone.
        selector = self.model.model.candidate_selector
        forward_context = create_forward_context(
            None,
            selector.vllm_config,
            batch_descriptor=BatchDescriptor(num_tokens=candidate_ids.shape[0]),
        )
        # Keep the inner graph mode NONE: the outer DFlash ACL graph captures
        # the compiled selector. Creating the context directly also avoids
        # target-MoE DP synchronization or attention metadata in this local head.
        with override_forward_context(forward_context):
            return super()._score_candidates(candidate_ids, unary_logits, hidden_states)
