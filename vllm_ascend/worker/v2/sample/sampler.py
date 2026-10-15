# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import NamedTuple

import numpy as np
import torch
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_ascend.ops.triton.v2.sample.prepare import prepare_sampling_inputs
from vllm_ascend.sample.sampler import apply_top_k_top_p, random_sample
from vllm_ascend.worker.v2.sample.logprob import compute_topk_scores

_SAMPLING_EPS = 1e-5


class _SeededDecodeParams(NamedTuple):
    temperature: torch.Tensor
    top_k: torch.Tensor | None
    top_p: torch.Tensor | None
    generators: dict[int, torch.Generator]
    logprobs_dims: tuple[int, int] | None


class AscendSampler(Sampler):
    """Preserve MRV1's per-request RNG stream for explicitly seeded sampling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generators: dict[int, torch.Generator] = {}
        self.preempted_generators: dict[str, torch.Generator] = {}
        self._decode_params_key: tuple[int, ...] | None = None
        self._decode_params: _SeededDecodeParams | None = None
        num_reqs = len(self.needs_logits_processing)
        self._decode_num_sampled = torch.ones(num_reqs, dtype=torch.int32, device=self.req_states.device)
        self._decode_num_rejected = torch.zeros(num_reqs, dtype=torch.int32, device=self.req_states.device)
        self._decode_counts_num_reqs = -1
        self._decode_counts = None

    def add_request(self, req_idx: int, prompt_len: int, sampling_params: SamplingParams) -> None:
        super().add_request(req_idx, prompt_len, sampling_params)
        self._decode_params_key = None
        self._decode_params = None
        self.generators.pop(req_idx, None)
        if sampling_params.seed is not None:
            req_id = self.req_states.index_to_req_id[req_idx]
            generator = self.preempted_generators.pop(req_id, None)
            if generator is None:
                generator = torch.Generator(device=self.req_states.device)
                generator.manual_seed(sampling_params.seed)
            self.generators[req_idx] = generator

    def finish_requests(self, finished_req_ids: set[str], preempted_req_ids: set[str] | None) -> None:
        # MRV1 retains a request's generator while its KV cache is recomputed.
        # Store by request ID because MRV2 releases and reuses the request slot.
        if finished_req_ids or preempted_req_ids:
            self._decode_params_key = None
            self._decode_params = None
        for req_id in finished_req_ids.union(preempted_req_ids or ()):
            idx = self.req_states.req_id_to_index.get(req_id)
            generator = self.generators.pop(idx, None)
            if req_id in finished_req_ids:
                self.preempted_generators.pop(req_id, None)
            elif generator is not None:
                self.preempted_generators[req_id] = generator

    def __call__(self, logits, input_batch):
        if (
            not self.generators
            and not self.compute_nans
            and self.trace_replay_state is None
            and not self.return_sampling_mask
            and logits.shape[0] == input_batch.num_reqs
            and len(input_batch.idx_mapping_np) == input_batch.num_reqs
            and not input_batch.num_draft_tokens
            and not np.any(input_batch.idx_mapping_np < 0)
            and not np.any(self.needs_logits_processing[input_batch.idx_mapping_np])
            and not np.any(self.sampling_states.temperature.np[input_batch.idx_mapping_np])
            and self.get_logprobs_dims(input_batch.idx_mapping_np) is None
        ):
            return self._sample_plain_greedy(logits, input_batch)
        if not self.generators:
            return self._sample_with_preparation(logits, input_batch)
        discarded = (
            input_batch.num_computed_prefill_tokens_np + input_batch.num_scheduled_tokens < input_batch.prefill_len_np
        )
        offsets = {
            int(idx): self.generators[int(idx)].get_offset()
            for idx in input_batch.idx_mapping_np[discarded]
            if int(idx) in self.generators
        }
        output = self._sample_with_preparation(logits, input_batch)
        for idx, offset in offsets.items():
            self.generators[idx].set_offset(offset)
        return output

    def _sample_with_preparation(self, logits, input_batch):
        if (
            self.compute_nans
            or self.trace_replay_state is not None
            or self.return_sampling_mask
            or self.use_flashinfer
            or input_batch.num_draft_tokens
            or logits.shape[0] != input_batch.num_reqs
            or logits.shape[0] != len(input_batch.idx_mapping_np)
            or np.any(input_batch.idx_mapping_np < 0)
        ):
            return super().__call__(logits, input_batch)
        indices = input_batch.idx_mapping_np
        if not input_batch.has_prefill:
            params = self._get_seeded_decode_params(indices)
            if params is not None:
                return self._sample_seeded_decode(logits, input_batch, params)
        states = self.sampling_states
        top_k = states.top_k.gpu if np.any(states.top_k.np[indices] != states.vocab_size) else None
        top_p = states.top_p.gpu if np.any(states.top_p.np[indices] != 1.0) else None
        prepared = prepare_sampling_inputs(
            input_batch.positions,
            input_batch.input_ids,
            input_batch.logits_indices,
            input_batch.idx_mapping,
            input_batch.expanded_idx_mapping,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            self.req_states.prefill_len.gpu,
            top_k,
            top_p,
        )
        processed = self.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping,
            indices,
            prepared.positions,
            prepared.input_ids,
            input_batch.expanded_local_pos,
            skip_top_k_top_p=True,
        )
        sampled, processed = self._sample_random(
            processed,
            input_batch.expanded_idx_mapping,
            indices,
            prepared.positions,
            prepared.top_k,
            prepared.top_p,
            False,
        )
        logprobs_tensors = None
        dimensions = self.get_logprobs_dims(indices)
        if dimensions is not None:
            num_logprobs, max_token_ids = dimensions
            scores = processed if self.logprobs_mode in PROCESSED_LOGPROBS_MODES else logits
            logprobs_tensors = compute_topk_scores(
                scores,
                num_logprobs,
                sampled,
                logprob_token_ids_state=self.logprob_token_ids_state,
                expanded_idx_mapping=input_batch.expanded_idx_mapping,
                max_per_req_token_ids=max_token_ids,
                logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),
            )
        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=None,
            num_sampled=prepared.num_sampled,
            num_rejected=prepared.num_rejected,
        )

    def _get_seeded_decode_params(self, indices):
        # Parameters are constant for each request. Rebuild on batch reordering
        # or request lifecycle changes, including slot reuse after preemption.
        key = tuple(indices.tolist())
        if key == self._decode_params_key:
            return self._decode_params
        self._decode_params_key = key
        self._decode_params = None
        states = self.sampling_states
        temperature = states.temperature.np[indices]
        if (
            not np.all(states.seeds_set[indices])
            or np.any(temperature < _SAMPLING_EPS)
            or np.any(self.logit_bias_state.use_logit_bias[indices])
            or np.any(self.penalties_state.use_penalty[indices])
            or np.any(self.bad_words_state.num_bad_words.np[indices])
            or (self.thinking_budget_state.enabled and np.any(self.thinking_budget_state.use_thinking_budget[indices]))
            or np.any(states.min_p.np[indices])
        ):
            return None
        device = self.req_states.device
        top_k = states.top_k.np[indices]
        top_p = states.top_p.np[indices]
        # Stage immutable parameters in pinned memory to avoid blocking the
        # worker on queued model execution. The pinned allocator records the
        # copy stream, so invalidating this cache cannot reuse an in-flight
        # source buffer before its upload completes.
        self._decode_params = _SeededDecodeParams(
            async_tensor_h2d(temperature, device=device).unsqueeze(1),
            async_tensor_h2d(top_k, device=device) if np.any(top_k != states.vocab_size) else None,
            async_tensor_h2d(top_p, device=device) if np.any(top_p != 1.0) else None,
            {row: self.generators[idx] for row, idx in enumerate(key)},
            self.get_logprobs_dims(indices),
        )
        return self._decode_params

    def _sample_seeded_decode(self, logits, input_batch, params):
        # Match MRV1's FP32 division. The out-of-place result preserves raw
        # logits for logprobs and avoids a separate copy and temperature kernel.
        processed = logits.to(torch.float32) / params.temperature
        processed = apply_top_k_top_p(processed, params.top_k, params.top_p)
        sampled = random_sample(processed.softmax(dim=-1, dtype=torch.float32), params.generators)
        logprobs_tensors = None
        if params.logprobs_dims is not None:
            num_logprobs, max_token_ids = params.logprobs_dims
            scores = processed if self.logprobs_mode in PROCESSED_LOGPROBS_MODES else logits
            logprobs_tensors = compute_topk_scores(
                scores,
                num_logprobs,
                sampled,
                logprob_token_ids_state=self.logprob_token_ids_state,
                expanded_idx_mapping=input_batch.expanded_idx_mapping,
                max_per_req_token_ids=max_token_ids,
                logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),
            )
        num_sampled, num_rejected = self._get_decode_counts(input_batch.num_reqs)
        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def _get_decode_counts(self, num_reqs):
        # Count tensors are immutable. Retain their views until the batch size
        # changes; older steps can safely keep their existing views.
        if num_reqs != self._decode_counts_num_reqs:
            self._decode_counts = (
                self._decode_num_sampled[:num_reqs],
                self._decode_num_rejected[:num_reqs],
            )
            self._decode_counts_num_reqs = num_reqs
        return self._decode_counts

    def _sample_plain_greedy(self, logits, input_batch):
        # No logits processors consume positions or input IDs in this path.
        # Keep the upstream chunked-prefill output-count semantics.
        sampled = logits.argmax(dim=-1, keepdim=True)
        if not input_batch.has_prefill:
            # Ordinary decode produces one token and no rejected draft tokens.
            # Downstream consumers only read these shared, immutable counts.
            num_sampled, num_rejected = self._get_decode_counts(input_batch.num_reqs)
        else:
            num_sampled, num_rejected = get_num_sampled_and_rejected(
                input_batch.seq_lens.new_ones(input_batch.num_reqs),
                input_batch.seq_lens,
                input_batch.cu_num_logits,
                input_batch.idx_mapping,
                self.req_states.prefill_len.gpu,
            )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def _sample_random(self, processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_flashinfer):
        temperature = self.sampling_states.temperature.np[idx_mapping_np]
        if np.all(temperature < _SAMPLING_EPS):
            # Greedy sampling needs no filtering or random-number generation.
            return processed_logits.argmax(dim=-1), processed_logits
        seeded = self.sampling_states.seeds_set[idx_mapping_np]
        if not np.any(seeded) or len(idx_mapping_np) != processed_logits.shape[0]:
            return super()._sample_random(
                processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_flashinfer
            )
        greedy = processed_logits.argmax(dim=-1) if np.any(temperature < _SAMPLING_EPS) else None
        processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
        generators = {row: self.generators[int(idx)] for row, idx in enumerate(idx_mapping_np) if seeded[row]}
        sampled = random_sample(processed_logits.softmax(dim=-1, dtype=torch.float32), generators)
        if greedy is not None:
            greedy_mask = self.sampling_states.temperature.gpu[expanded_idx_mapping] < _SAMPLING_EPS
            sampled = torch.where(greedy_mask, greedy, sampled)
        return sampled, processed_logits
