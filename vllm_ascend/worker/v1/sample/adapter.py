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

import numpy as np
import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.ascend_config import SamplingConfig
from vllm_ascend.worker.v1.sample.context import V1MappingContext
from vllm_ascend.worker.v1.sample.logits_processor import LogitsProcessor

_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max
_UINT64_MODULUS = 2**64
_PLACEHOLDER_TOKEN_ID = -1


class V1SamplerAdapter:
    """Bridges v1 model runner's _sample() to the new sampling pipeline
    with Ascend-optimized kernels.

    Usage: Created in NPUModelRunner.__init__() when sampling_config enables
           the optimization. Called from NPUModelRunner._sample() instead of
           self.sampler().
    """

    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        logprobs_mode: str = "raw_logprobs",
        num_speculative_tokens: int = 1,
        sampling_config: SamplingConfig | None = None,
    ):
        self._max_num_reqs = max_num_reqs
        self._vocab_size = vocab_size
        self._device = device
        self._logprobs_mode = logprobs_mode
        self._num_speculative_tokens = num_speculative_tokens
        self._sampling_config = sampling_config or SamplingConfig()
        self._request_seeds: dict[str, int] = {}

        # Logits processing pipeline
        self._logits_processor = LogitsProcessor(self._sampling_config.logits_processing_mode)

    def __call__(
        self,
        logits: torch.Tensor,  # [num_logits, vocab_size]
        sampling_metadata: SamplingMetadata,
        num_reqs: int,
        positions: torch.Tensor,  # [num_logits]
        input_ids: torch.Tensor,  # [num_logits]
        req_indices: torch.Tensor,  # [num_logits]
        req_ids: tuple[str, ...] | None = None,
    ) -> SamplerOutput:
        """Main entry point. Replaces self.sampler(logits, sampling_metadata)."""

        # 1. Build mapping context from v1 data
        ctx = V1MappingContext.from_v1_logits(
            num_reqs=num_reqs,
            positions_at_logits=positions,
            input_ids_at_logits=input_ids,
            req_indices_at_logits=req_indices,
            device=self._device,
            req_ids=req_ids,
        )

        # 2. Logits processing (mode-controlled pipeline)
        processed_logits = self._logits_processor.apply(
            logits,
            sampling_metadata,
            ctx,
            self._num_speculative_tokens,
        )

        # 3. Sampling (Gumbel sampling with Ascend kernel)
        sampled = self._sample(processed_logits, sampling_metadata, ctx)

        # 4. Logprobs computation
        logprobs_tensors = self._compute_logprobs(
            logits, processed_logits, sampled, sampling_metadata, ctx
        )

        # 5. Convert to int32 to match the upstream SamplerOutput dtype.
        #    _bookkeeping_sync uses sampled_token_ids_pinned_cpu which is int32.
        sampled = sampled.to(torch.int32)
        sampled_token_ids = self._format_sampled_token_ids(sampled, ctx)

        # 6. Construct the engine-facing vllm.v1.outputs.SamplerOutput.
        return SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=logprobs_tensors,
        )

    def _sample(
        self,
        logits: torch.Tensor,  # [num_logits, vocab_size] - after processing
        sampling_metadata: SamplingMetadata,
        ctx: V1MappingContext,
    ) -> torch.Tensor:
        """Sample tokens using Gumbel sampling with Ascend kernel."""
        from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

        # Prepare temperature tensor on device
        temp = self._temperature_for_sampling(sampling_metadata, ctx)

        # Prepare seeds tensor
        seeds = self._compute_seeds(sampling_metadata, ctx)

        sampled = gumbel_sample(
            logits=logits,
            idx_mapping=ctx.expanded_idx_mapping,
            temperature=temp,
            seed=seeds,
            pos=ctx.pos,
            apply_temperature=False,  # temperature already applied in logits_processor
        )
        return sampled

    def _temperature_for_sampling(
        self,
        sampling_metadata: SamplingMetadata,
        ctx: V1MappingContext,
    ) -> torch.Tensor:
        temperature = sampling_metadata.temperature
        if temperature is None:
            return torch.zeros(ctx.num_reqs, dtype=torch.float32, device=self._device)
        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor(temperature, dtype=torch.float32, device=self._device)
        elif temperature.device != self._device:
            temperature = temperature.to(self._device)
        if temperature.ndim == 0:
            return temperature.expand(ctx.num_reqs).to(dtype=torch.float32)
        if int(temperature.shape[0]) < ctx.num_reqs:
            raise ValueError("temperature must have at least one entry per active request")
        return temperature[:ctx.num_reqs].to(dtype=torch.float32)

    def _format_sampled_token_ids(
        self,
        sampled: torch.Tensor,
        ctx: V1MappingContext,
    ) -> torch.Tensor:
        if ctx.is_identity_request_mapping:
            return sampled.view(-1, 1)

        counts = ctx.num_logits_per_req_np
        max_count = int(counts.max()) if counts.size else 0
        output = torch.full(
            (ctx.num_reqs, max_count),
            _PLACEHOLDER_TOKEN_ID,
            dtype=sampled.dtype,
            device=sampled.device,
        )
        if sampled.numel() == 0:
            return output
        output[ctx.expanded_idx_mapping.to(torch.long), ctx.expanded_local_pos.to(torch.long)] = sampled
        return output

    def _compute_seeds(
        self,
        sampling_metadata: SamplingMetadata,
        ctx: V1MappingContext,
    ) -> torch.Tensor:
        """Return per-request seeds for Gumbel sampling."""
        seeds = torch.empty(ctx.num_reqs, dtype=torch.int64, device=self._device)
        req_ids = ctx.req_ids
        if req_ids is None:
            req_ids = tuple(f"__slot_{idx}" for idx in range(ctx.num_reqs))

        active_req_ids = set(req_ids)
        for cached_req_id in tuple(self._request_seeds):
            if cached_req_id not in active_req_ids:
                del self._request_seeds[cached_req_id]

        generators = sampling_metadata.generators
        for req_idx, req_id in enumerate(req_ids):
            seed = self._request_seeds.get(req_id)
            if seed is None:
                generator = generators.get(req_idx) if generators else None
                seed = (
                    self._normalize_seed(generator.initial_seed())
                    if generator is not None
                    else self._new_random_seed()
                )
                self._request_seeds[req_id] = seed
            seeds[req_idx] = seed
        return seeds

    @staticmethod
    def _normalize_seed(seed: int) -> int:
        seed = int(seed)
        if seed > _NP_INT64_MAX:
            seed -= _UINT64_MODULUS
        return seed

    @staticmethod
    def _new_random_seed() -> int:
        return int(np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX, dtype=np.int64))

    def _compute_logprobs(
        self,
        raw_logits: torch.Tensor,
        processed_logits: torch.Tensor,
        sampled: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        ctx: V1MappingContext,
    ):
        """Compute top-k logprobs if requested."""
        from vllm.v1.outputs import LogprobsTensors

        max_num_logprobs = sampling_metadata.max_num_logprobs
        if max_num_logprobs is None:
            return None

        # Select logits source based on logprobs_mode
        if self._logprobs_mode in ("processed_logprobs", "processed_logits"):
            logits_for_logprobs = processed_logits
        else:  # "raw_logprobs"
            logits_for_logprobs = raw_logits

        if max_num_logprobs == -1:
            # Return the full unsorted and unranked logprobs.
            raw_logprobs = logits_for_logprobs.log_softmax(dim=-1, dtype=torch.float32)
            return LogprobsTensors(
                torch.empty(0, device=raw_logprobs.device, dtype=torch.int32),
                raw_logprobs,
                torch.empty(0, device=raw_logprobs.device, dtype=torch.int32),
                ctx.cu_num_logits_np.tolist() if ctx.expanded_logits else None,
            )

        if max_num_logprobs < 0:
            return None

        # Compute log softmax for the raw/processed logits
        logprobs = logits_for_logprobs.log_softmax(dim=-1, dtype=torch.float32)

        # Use int64 for indexing (required by gather), convert to int32 later
        sampled_int64 = sampled.to(torch.int64)
        sampled_token_ids = sampled_int64.unsqueeze(-1)

        # Get top-k logprobs. max_num_logprobs == 0 returns sampled-token-only
        # logprobs, matching upstream v1 semantics.
        topk_logprobs, topk_indices = torch.topk(logprobs, max_num_logprobs, dim=-1)
        sampled_logprobs = logprobs.gather(-1, sampled_token_ids)

        logprob_token_ids = torch.cat([sampled_token_ids, topk_indices], dim=1)
        logprob_values = torch.cat([sampled_logprobs, topk_logprobs], dim=1)

        # Compute token ranks
        sampled_logits = logits_for_logprobs.gather(-1, sampled_token_ids)
        token_ranks = (logits_for_logprobs >= sampled_logits).sum(dim=-1)

        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids.to(torch.int32),
            logprobs=logprob_values,
            selected_token_ranks=token_ranks,
            cu_num_generated_tokens=ctx.cu_num_logits_np.tolist() if ctx.expanded_logits else None,
        )
