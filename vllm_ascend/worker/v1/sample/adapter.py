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

from types import SimpleNamespace

import numpy as np
import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu.sample.output import SamplerOutput as GpuSamplerOutput
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler as GpuRejectionSampler

from vllm_ascend.ascend_config import SamplingConfig
from vllm_ascend.worker.v1.sample.context import V1MappingContext
from vllm_ascend.worker.v1.sample.logits_processor import LogitsProcessor

_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max
_UINT64_MODULUS = 2**64
_PLACEHOLDER_TOKEN_ID = -1


class GpuSamplerBridge:
    """Bridges the old v1 runner to the worker.gpu sampling interfaces.

    ``sample_from_v1`` is the old model-runner entry point. ``__call__`` and
    ``apply_sampling_params`` are the sampler facade expected by
    ``vllm.v1.worker.gpu.spec_decode.rejection_sampler.RejectionSampler``.
    """

    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        logprobs_mode: str = "raw_logprobs",
        num_speculative_tokens: int = 1,
        spec_config=None,
        sampling_config: SamplingConfig | None = None,
    ):
        self._max_num_reqs = max_num_reqs
        self._vocab_size = vocab_size
        self._device = device
        self._logprobs_mode = logprobs_mode
        self._num_speculative_tokens = num_speculative_tokens
        self._spec_config = spec_config
        self._sampling_config = sampling_config or SamplingConfig()
        self._request_seeds: dict[str, int] = {}
        self.logprobs_mode = logprobs_mode
        self.compute_nans = False
        self.sampling_states = None
        self._active_sampling_metadata: SamplingMetadata | None = None
        self._active_ctx: V1MappingContext | None = None
        self._active_processed_logits: torch.Tensor | None = None

        # Logits processing pipeline
        self._logits_processor = LogitsProcessor(self._sampling_config.logits_processing_mode)

    def sample_from_v1(
        self,
        logits: torch.Tensor,  # [num_logits, vocab_size]
        sampling_metadata: SamplingMetadata,
        num_reqs: int,
        positions: torch.Tensor,  # [num_logits]
        input_ids: torch.Tensor,  # [num_logits]
        req_indices: torch.Tensor,  # [num_logits]
        req_ids: tuple[str, ...] | None = None,
        spec_decode_metadata: SpecDecodeMetadata | None = None,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        """Model-runner entry point. Replaces self.sampler(logits, metadata)."""
        # 1. Build mapping context from v1 data
        ctx = V1MappingContext.from_v1_logits(
            num_reqs=num_reqs,
            positions_at_logits=positions,
            input_ids_at_logits=input_ids,
            req_indices_at_logits=req_indices,
            device=self._device,
            req_ids=req_ids,
        )

        if spec_decode_metadata is not None:
            return self._sample_spec_decode(
                logits,
                sampling_metadata,
                spec_decode_metadata,
                ctx,
                draft_logits,
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

    def _sample_spec_decode(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        ctx: V1MappingContext,
        draft_logits: torch.Tensor | None,
    ) -> SamplerOutput:
        if ctx.cu_num_logits_np is None:
            raise ValueError("speculative decoding requires grouped logits rows")
        expected_counts = np.asarray(
            [num_draft_tokens + 1 for num_draft_tokens in spec_decode_metadata.num_draft_tokens],
            dtype=np.int32,
        )
        if not np.array_equal(ctx.num_logits_per_req_np, expected_counts):
            raise ValueError("speculative logits-row mapping does not match spec decode metadata")
        spec_config = self._spec_config or SimpleNamespace(
            num_speculative_tokens=self._num_speculative_tokens,
            rejection_sample_method="strict",
            synthetic_acceptance_rates=None,
        )
        rejection_sample_method = getattr(spec_config, "rejection_sample_method", "strict")
        if rejection_sample_method == "probabilistic" and draft_logits is None:
            raise NotImplementedError(
                "probabilistic rejection sampling requires draft logits from the drafter."
            )

        self._activate_gpu_sampler(
            sampling_metadata,
            ctx,
            disable_gpu_logprobs=rejection_sample_method == "probabilistic",
        )
        try:
            rejection_sampler = GpuRejectionSampler(self, spec_config, self._device)
            input_batch = _GpuRejectionInputBatch.from_context(ctx)
            output = rejection_sampler(logits, input_batch, draft_logits)
            sampled_token_ids = self._mask_unsampled_tokens(
                output.sampled_token_ids,
                output.num_sampled,
            )
            logprobs_tensors = output.logprobs_tensors
            if rejection_sample_method == "probabilistic":
                if self._active_processed_logits is None:
                    raise RuntimeError("probabilistic rejection sampling did not process logits")
                sampled_for_logprobs = self._flatten_sampled_token_ids(
                    sampled_token_ids,
                    output.num_sampled,
                    ctx,
                )
                logprobs_tensors = self._compute_logprobs(
                    logits,
                    self._active_processed_logits,
                    sampled_for_logprobs,
                    sampling_metadata,
                    ctx,
                )
        finally:
            self._deactivate_gpu_sampler()
        return SamplerOutput(
            sampled_token_ids=sampled_token_ids.to(torch.int32),
            logprobs_tensors=logprobs_tensors,
        )

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: "_GpuRejectionInputBatch",
    ) -> GpuSamplerOutput:
        sampling_metadata, _ = self._active_gpu_sampler_state()
        ctx = self._ctx_from_input_batch(input_batch)
        processed_logits = self.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            input_batch.positions[input_batch.logits_indices],
            input_batch.input_ids[input_batch.logits_indices],
            input_batch.expanded_local_pos,
        )
        sampled = self._sample(processed_logits, sampling_metadata, ctx)
        logprobs_tensors = self._compute_logprobs(
            logits,
            processed_logits,
            sampled,
            sampling_metadata,
            ctx,
        )
        return GpuSamplerOutput(
            sampled_token_ids=sampled.view(-1, 1).to(torch.int32),
            logprobs_tensors=logprobs_tensors,
            num_nans=None,
            num_sampled=torch.ones(ctx.num_reqs, dtype=torch.int32, device=logits.device),
        )

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> torch.Tensor:
        sampling_metadata, active_ctx = self._active_gpu_sampler_state()
        del idx_mapping_np
        ctx = V1MappingContext.from_v1_logits(
            num_reqs=active_ctx.num_reqs,
            positions_at_logits=pos,
            input_ids_at_logits=input_ids,
            req_indices_at_logits=expanded_idx_mapping,
            device=logits.device,
            req_ids=active_ctx.req_ids,
            expanded_local_pos=expanded_local_pos,
            cu_num_logits_np=active_ctx.cu_num_logits_np,
        )
        processed_logits = self._logits_processor.apply(
            logits,
            sampling_metadata,
            ctx,
            self._num_speculative_tokens,
        )
        self._active_processed_logits = processed_logits
        return processed_logits

    def _ctx_from_input_batch(self, input_batch: "_GpuRejectionInputBatch") -> V1MappingContext:
        _, active_ctx = self._active_gpu_sampler_state()
        return V1MappingContext.from_v1_logits(
            num_reqs=active_ctx.num_reqs,
            positions_at_logits=input_batch.positions[input_batch.logits_indices],
            input_ids_at_logits=input_batch.input_ids[input_batch.logits_indices],
            req_indices_at_logits=input_batch.expanded_idx_mapping,
            device=input_batch.expanded_idx_mapping.device,
            req_ids=active_ctx.req_ids,
            expanded_local_pos=input_batch.expanded_local_pos,
            cu_num_logits_np=active_ctx.cu_num_logits_np,
        )

    def _activate_gpu_sampler(
        self,
        sampling_metadata: SamplingMetadata,
        ctx: V1MappingContext,
        disable_gpu_logprobs: bool = False,
    ) -> None:
        self._active_sampling_metadata = sampling_metadata
        self._active_ctx = ctx
        self.sampling_states = _GpuSamplingStatesBridge(
            self,
            sampling_metadata,
            ctx,
            disable_gpu_logprobs,
        )

    def _deactivate_gpu_sampler(self) -> None:
        self._active_sampling_metadata = None
        self._active_ctx = None
        self._active_processed_logits = None
        self.sampling_states = None

    def _active_gpu_sampler_state(self) -> tuple[SamplingMetadata, V1MappingContext]:
        if self._active_sampling_metadata is None or self._active_ctx is None:
            raise RuntimeError("GpuSamplerBridge is not active for a worker.gpu sampler call")
        return self._active_sampling_metadata, self._active_ctx

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

    @staticmethod
    def _mask_unsampled_tokens(
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
    ) -> torch.Tensor:
        steps = torch.arange(
            sampled_token_ids.shape[1],
            device=sampled_token_ids.device,
            dtype=torch.long,
        )
        valid_mask = steps.unsqueeze(0) < num_sampled.to(
            device=sampled_token_ids.device,
            dtype=torch.long,
        ).unsqueeze(1)
        return sampled_token_ids.masked_fill(~valid_mask, _PLACEHOLDER_TOKEN_ID)

    @staticmethod
    def _flatten_sampled_token_ids(
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        ctx: V1MappingContext,
    ) -> torch.Tensor:
        req_indices = ctx.expanded_idx_mapping.to(
            device=sampled_token_ids.device,
            dtype=torch.long,
        )
        local_pos = ctx.expanded_local_pos.to(
            device=sampled_token_ids.device,
            dtype=torch.long,
        )
        num_sampled_per_row = num_sampled.to(
            device=sampled_token_ids.device,
            dtype=torch.long,
        )[req_indices]
        valid_mask = local_pos < num_sampled_per_row
        sampled = sampled_token_ids[req_indices, local_pos]
        return torch.where(valid_mask, sampled, torch.zeros_like(sampled))

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
        """Compute logprobs/logits tensors if requested."""
        from vllm.v1.outputs import LogprobsTensors

        max_num_logprobs = sampling_metadata.max_num_logprobs
        logprob_token_ids = getattr(sampling_metadata, "logprob_token_ids", None)
        if max_num_logprobs is None and not logprob_token_ids:
            return None

        logprob_values_source = self._logprob_values_source(raw_logits, processed_logits)
        if logprob_token_ids:
            return self._gather_specific_token_logprobs(
                logprob_values_source,
                logprob_token_ids,
                sampled,
                ctx,
            )

        if max_num_logprobs == -1:
            return LogprobsTensors(
                torch.empty(0, device=logprob_values_source.device, dtype=torch.int32),
                logprob_values_source,
                torch.empty(0, device=logprob_values_source.device, dtype=torch.int32),
                ctx.cu_num_logits_np.tolist() if ctx.expanded_logits else None,
            )

        if max_num_logprobs < 0:
            return None

        # Use int64 for indexing (required by gather), convert to int32 later
        sampled_int64 = sampled.to(torch.int64)
        sampled_token_ids = sampled_int64.unsqueeze(-1)

        # Get top-k logprobs. max_num_logprobs == 0 returns sampled-token-only
        # logprobs, matching upstream v1 semantics.
        topk_logprobs, topk_indices = torch.topk(logprob_values_source, max_num_logprobs, dim=-1)
        sampled_logprobs = logprob_values_source.gather(-1, sampled_token_ids)

        logprob_token_ids = torch.cat([sampled_token_ids, topk_indices], dim=1)
        logprob_values = torch.cat([sampled_logprobs, topk_logprobs], dim=1)

        # Compute token ranks
        sampled_values = logprob_values_source.gather(-1, sampled_token_ids)
        token_ranks = (logprob_values_source >= sampled_values).sum(dim=-1)

        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids.to(torch.int32),
            logprobs=logprob_values,
            selected_token_ranks=token_ranks,
            cu_num_generated_tokens=ctx.cu_num_logits_np.tolist() if ctx.expanded_logits else None,
        )

    def _logprob_values_source(
        self,
        raw_logits: torch.Tensor,
        processed_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self._logprobs_mode == "raw_logits":
            return raw_logits.to(dtype=torch.float32)
        if self._logprobs_mode == "processed_logits":
            return processed_logits.to(dtype=torch.float32)
        if self._logprobs_mode == "processed_logprobs":
            return processed_logits.log_softmax(dim=-1, dtype=torch.float32)
        if self._logprobs_mode == "raw_logprobs":
            return raw_logits.log_softmax(dim=-1, dtype=torch.float32)
        raise NotImplementedError(f"Unsupported logprobs_mode: {self._logprobs_mode}")

    def _gather_specific_token_logprobs(
        self,
        logprob_values_source: torch.Tensor,
        logprob_token_ids: dict[int, list[int]],
        sampled: torch.Tensor,
        ctx: V1MappingContext,
    ):
        from vllm.v1.outputs import LogprobsTensors

        max_num_tokens = max((len(token_ids) for token_ids in logprob_token_ids.values()), default=0)
        width = max_num_tokens + 1
        token_ids = torch.zeros(
            (ctx.num_logits, width),
            dtype=torch.int64,
            device=logprob_values_source.device,
        )
        valid_mask = torch.zeros(
            (ctx.num_logits, width),
            dtype=torch.bool,
            device=logprob_values_source.device,
        )
        token_ids[:, 0] = sampled.to(device=logprob_values_source.device, dtype=torch.int64)
        valid_mask[:, 0] = True

        for row, req_idx in enumerate(ctx.idx_mapping_np.tolist()):
            req_token_ids = logprob_token_ids.get(req_idx)
            if not req_token_ids:
                continue
            count = len(req_token_ids)
            token_ids[row, 1 : count + 1] = torch.tensor(
                req_token_ids,
                dtype=torch.int64,
                device=logprob_values_source.device,
            )
            valid_mask[row, 1 : count + 1] = True

        values = logprob_values_source.gather(-1, token_ids)
        values = values.masked_fill(~valid_mask, float("-inf"))
        sampled_values = logprob_values_source.gather(-1, token_ids[:, :1])
        token_ranks = (logprob_values_source >= sampled_values).sum(dim=-1)
        return LogprobsTensors(
            logprob_token_ids=token_ids.to(torch.int32),
            logprobs=values,
            selected_token_ranks=token_ranks,
            cu_num_generated_tokens=ctx.cu_num_logits_np.tolist() if ctx.expanded_logits else None,
        )


class _GpuRejectionInputBatch(SimpleNamespace):
    @classmethod
    def from_context(cls, ctx: V1MappingContext) -> "_GpuRejectionInputBatch":
        if ctx.cu_num_logits_np is None:
            raise ValueError("cu_num_logits_np is required for speculative decoding")
        device = ctx.expanded_idx_mapping.device
        num_reqs = ctx.num_reqs
        idx_mapping = torch.arange(num_reqs, device=device, dtype=torch.int32)
        logits_indices = torch.arange(ctx.num_logits, device=device, dtype=torch.long)
        cu_num_logits = torch.from_numpy(ctx.cu_num_logits_np).to(device=device, dtype=torch.int32)
        return cls(
            num_reqs=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
            expanded_idx_mapping=ctx.expanded_idx_mapping,
            expanded_local_pos=ctx.expanded_local_pos,
            positions=ctx.pos,
            input_ids=ctx.input_ids,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=ctx.cu_num_logits_np,
            seq_lens=torch.zeros(num_reqs, device=device, dtype=torch.int32),
        )


class _GpuSamplingStatesBridge:
    def __init__(
        self,
        bridge: GpuSamplerBridge,
        sampling_metadata: SamplingMetadata,
        ctx: V1MappingContext,
        disable_gpu_logprobs: bool = False,
    ):
        self.temperature = SimpleNamespace(
            gpu=bridge._temperature_for_sampling(sampling_metadata, ctx)
        )
        self.seeds = SimpleNamespace(gpu=bridge._compute_seeds(sampling_metadata, ctx))
        self._max_num_logprobs = sampling_metadata.max_num_logprobs
        self._disable_gpu_logprobs = disable_gpu_logprobs

    def max_num_logprobs(self, idx_mapping_np: np.ndarray) -> int:
        del idx_mapping_np
        if self._disable_gpu_logprobs:
            return -1
        return -1 if self._max_num_logprobs is None else self._max_num_logprobs
