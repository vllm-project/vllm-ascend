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

from typing import Any

import torch
import torch.distributed as dist
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_pcp_group
from vllm.logger import logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.sample.gumbel import _DRAFT_NOISE_SALT, _log1p_neg_stable, tl_rand32
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

from vllm_ascend.ascend_config import validate_additional_config_bool
from vllm_ascend.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
from vllm_ascend.ops.vocab_parallel_embedding import VocabParallelMode
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.dflash2.pcp import PCPQueryBuffers


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

    The upstream stable ``log1p(-u)`` helper avoids libdevice log1p and
    preserves the positive Gumbel tail for small uniform draws.
    The signature mirrors upstream so the shared ``CandidateSampler`` can
    call it unmodified:
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
            # Verification uses a probability-ratio test, so draft and target
            # draws must use independent noise streams, as in Ascend's sampler.
            position = tl.load(sample_pos_ptr + flat) - 1 + _DRAFT_NOISE_SALT
            # triton-ascend's philox requires int32 seed/offset operands
            # (int64 counters lower to 64-bit multiplies whose runtime
            # helper, __multi3, is unavailable); seeds, token ids and
            # positions all fit int32.
            gumbel_seed = tl.randint(seed.to(tl.int32), position.to(tl.int32))
            uniform = tl_rand32(gumbel_seed, candidates.to(tl.int32), includes_zero=False)
            noise = -tl.log(-_log1p_neg_stable(uniform))
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
    aclgraph capture, which captures ``self._generate_draft``. Selector scoring
    inherits the upstream context-free call: any internal ACLGraphWrapper
    bypasses its own capture/replay, while device operations still participate
    in an active outer draft capture.
    """

    # _generate_draft -> get_top_k_tokens samples on the vocab-sharded head
    # outside sample_draft; reject lmhead TP at construction (as DSpark).
    _lmhead_tp_sample_draft_supported = False
    _pcp_sharding_enabled = False
    _pcp_real_proposal = False

    def __init__(self, vllm_config, device):
        # DFlash's constructor replaces PCP with 1 in its private config.
        # Retain target topology for eligibility and the actual PCP group.
        self._pcp_target_config = vllm_config
        self._pcp_sharding_requested = validate_additional_config_bool(
            (vllm_config.additional_config or {}).get("dflash_pcp_request_sharding", False),
            "additional_config.dflash_pcp_request_sharding",
        )
        self._pcp_sharding_enabled = False
        self._pcp_real_proposal = False
        super().__init__(vllm_config, device)

    def _pcp_sharding_ineligibility(self) -> str | None:
        config = self._pcp_target_config
        parallel = config.parallel_config
        if not config.use_v2_model_runner:
            return "requires Model Runner V2"
        if (
            parallel.tensor_parallel_size,
            parallel.data_parallel_size,
            parallel.decode_context_parallel_size,
            parallel.pipeline_parallel_size,
        ) != (1, 1, 1, 2):
            return "requires TP=1, DP=1, DCP=1 and PP=2"
        if parallel.prefill_context_parallel_size <= 1:
            return "requires PCP>1"
        if not config.model_config.enforce_eager or not self.speculative_config.enforce_eager:
            return "requires eager target and draft execution"
        if self.speculative_config.draft_sample_method != "greedy" or self.draft_logits is not None:
            return "requires greedy drafting without a draft logits cache"
        if getattr(self, "enable_adaptive_verification", False):
            return "adaptive verification is not supported"
        if getattr(self, "draft_watermarker", None) is not None:
            return "draft watermarking is not supported"
        if type(self.model) is not DFlash2Qwen3ForCausalLM:
            return "only the Ascend Qwen3 DFlash2 implementation has been audited"
        for name, layer in (("embedding", self.model.model.embed_tokens), ("LM head", self.model.lm_head)):
            if getattr(layer, "tp_size", None) != 1 or getattr(layer, "parallel_mode", None) not in (
                VocabParallelMode.STANDARD,
                VocabParallelMode.REPLICATED,
            ):
                return f"draft {name} must have full local weights without PCP/TP collectives"
        return None

    def load_model(self, target_model):
        super().load_model(target_model)
        self._pcp_sharding_enabled = False
        if not self._pcp_sharding_requested:
            return
        reason = self._pcp_sharding_ineligibility()
        reasons = [reason]
        group = None
        if self._pcp_target_config.parallel_config.prefill_context_parallel_size > 1:
            group = get_pcp_group()
            # Startup only: use the CPU group, never a device-to-host sync in
            # the inference hot path. A local weight-layout difference must
            # disable the optimization on every rank before any query runs.
            reasons = [None] * group.world_size
            dist.all_gather_object(reasons, reason, group=group.cpu_group)
        failures = sorted({reason for reason in reasons if reason is not None})
        if failures:
            logger.warning("DFlash2 PCP request sharding disabled: %s. Using replicated drafts.", "; ".join(failures))
            return
        assert group is not None
        self._pcp_sharding_enabled = True
        self._pcp_query_group = group
        logger.info(
            "DFlash2 PCP request sharding enabled across %d ranks; context KV remains replicated.", group.world_size
        )

    def set_attn(self, *args, **kwargs):
        super().set_attn(*args, **kwargs)
        if self._pcp_sharding_enabled:
            group = self._pcp_query_group
            self._pcp_query_buffers = PCPQueryBuffers(self, group.rank_in_group, group.world_size)

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync_state=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
        num_speculative_tokens=None,
    ):
        previous = self._pcp_real_proposal
        self._pcp_real_proposal = not dummy_run and not is_profile
        try:
            if self._pcp_sharding_enabled and self._pcp_real_proposal:
                if num_speculative_tokens not in (None, 0, self.num_speculative_steps):
                    raise ValueError("DFlash2 PCP request sharding requires a fixed speculative length")
                if input_batch.num_reqs == 0:
                    # Upstream max(seq_lens) is undefined for an empty batch.
                    return self.draft_tokens[:0]
            return super().propose(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                dp_sync_state,
                dummy_run,
                skip_attn_for_dummy_run,
                mm_inputs,
                is_profile=is_profile,
                num_speculative_tokens=num_speculative_tokens,
            )
        finally:
            self._pcp_real_proposal = previous

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        if not self._pcp_sharding_enabled or not self._pcp_real_proposal:
            return super()._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        if cudagraph_runtime_mode != CUDAGraphMode.NONE or num_tokens_across_dp is not None:
            raise RuntimeError("DFlash2 PCP request sharding received graph/DP execution after eager-only validation")
        buffers = self._pcp_query_buffers
        local_count = buffers.pack(self, num_reqs)
        if local_count:
            # The parent propose has already updated valid context KV for ALL
            # requests. Only the query pass uses this private subset.
            with buffers.activate(self):
                local_tokens = local_count * self.num_query_per_req
                bounds = self.input_batch.seq_lens_cpu_upper_bound[buffers.rank : num_reqs : buffers.world_size]
                local_metadata = self._build_uniform_attn_metadata(
                    num_reqs=local_count,
                    batch_desc=BatchExecutionDescriptor(
                        cg_mode=CUDAGraphMode.NONE,
                        num_tokens=local_tokens,
                        num_reqs=local_count,
                    ),
                    num_query_per_req=self.num_query_per_req,
                    seq_lens_cpu_upper_bound=bounds,
                    step=self.num_query_per_req,
                    causal=self._group_causal,
                )
                self._update_draft_attn_metadata(local_metadata, local_count)
                local_slots = build_slot_mappings_by_layer(
                    self.block_tables.slot_mappings[:, :local_tokens],
                    self.kv_cache_config,
                    self.attn_groups,
                )
                super()._generate_draft(
                    local_count, local_tokens, local_metadata, local_slots, None, CUDAGraphMode.NONE
                )
        buffers.gather(self._pcp_query_group, self.draft_tokens, num_reqs)
