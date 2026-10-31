# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""Triton-free MRV2 sampler for Ascend 310P.

Aligns temperature / top-k / top-p post-processing with MRV1's
``AscendSampler310`` (inverse-CDF random sampling) while keeping the MRV2
``sampling_states`` surface required by MTP draft ``propose()``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_ascend._310p.worker.v2.sample import (
    apply_min_p,
    apply_temperature,
    apply_top_k_top_p_npu,
    gather_logprobs,
)
from vllm_ascend._310p.worker.v2.sample.allowed_token_ids import (
    apply_allowed_token_ids,
    materialize_allowed_mask,
)
from vllm_ascend._310p.worker.v2.sample.bad_words import apply_bad_words
from vllm_ascend._310p.worker.v2.sample.logit_bias import apply_logit_bias
from vllm_ascend._310p.worker.v2.sample.penalties import (
    apply_penalties,
    build_penalty_token_tensors,
    use_penalty,
)
from vllm_ascend._310p.worker.v2.sample_utils import (
    _prepare_cpu_generators_310p,
    _random_sample_310p,
)

_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max
_SAMPLING_EPS = 1e-5


class Ascend310PSampler:
    """Triton-free sampler for 310P MRV2 with MRV1-aligned temperature sampling.

    Exposes ``sampling_states.temperature/seeds/top_k/top_p/min_p`` so MTP draft
    ``propose()`` (and ``_dummy_run``) can read device buffers without UVA
    SamplingStates.
    """

    # TODO: Refactor this sampler to register 310P implementations through
    # Triton Dispatcher after vLLM RFC #45133 lands.

    def __init__(
        self,
        max_num_reqs: int = 1,
        device: torch.device | str | None = None,
        vocab_size: int | None = None,
        req_states: Any | None = None,
    ) -> None:
        # Keep attribute for upstream post_update compatibility (no persistent bins).
        self.penalties_state = SimpleNamespace(output_bin_counts=None)
        self.req_states = req_states
        self.max_num_reqs = max_num_reqs
        self.vocab_size = int(vocab_size) if vocab_size is not None else 0
        if device is None:
            device = torch.device("cpu")
        elif not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device

        temperature_gpu = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
        seeds_gpu = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        top_k_gpu = torch.full(
            (max_num_reqs,),
            fill_value=max(self.vocab_size, 1),
            dtype=torch.int32,
            device=device,
        )
        top_p_gpu = torch.ones(max_num_reqs, dtype=torch.float32, device=device)
        min_p_gpu = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
        repetition_gpu = torch.ones(max_num_reqs, dtype=torch.float32, device=device)
        frequency_gpu = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
        presence_gpu = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
        self.temperature_cpu = torch.zeros(max_num_reqs, dtype=torch.float32, device="cpu")
        self.seeds_cpu = torch.zeros(max_num_reqs, dtype=torch.int64, device="cpu")
        self.top_k_cpu = torch.full(
            (max_num_reqs,),
            fill_value=max(self.vocab_size, 1),
            dtype=torch.int32,
            device="cpu",
        )
        self.top_p_cpu = torch.ones(max_num_reqs, dtype=torch.float32, device="cpu")
        self.min_p_cpu = torch.zeros(max_num_reqs, dtype=torch.float32, device="cpu")
        self.repetition_penalty_cpu = torch.ones(max_num_reqs, dtype=torch.float32, device="cpu")
        self.frequency_penalty_cpu = torch.zeros(max_num_reqs, dtype=torch.float32, device="cpu")
        self.presence_penalty_cpu = torch.zeros(max_num_reqs, dtype=torch.float32, device="cpu")
        self._sampling_state_dirty = False
        self.sampling_states = SimpleNamespace(
            temperature=SimpleNamespace(gpu=temperature_gpu),
            seeds=SimpleNamespace(gpu=seeds_gpu),
            top_k=SimpleNamespace(gpu=top_k_gpu),
            top_p=SimpleNamespace(gpu=top_p_gpu),
            min_p=SimpleNamespace(gpu=min_p_gpu),
            repetition_penalty=SimpleNamespace(gpu=repetition_gpu),
            frequency_penalty=SimpleNamespace(gpu=frequency_gpu),
            presence_penalty=SimpleNamespace(gpu=presence_gpu),
            vocab_size=self.vocab_size,
        )
        self._temperature_np = np.zeros(max_num_reqs, dtype=np.float32)
        self._top_k_np = np.full(max_num_reqs, fill_value=max(self.vocab_size, 1), dtype=np.int32)
        self._top_p_np = np.ones(max_num_reqs, dtype=np.float32)
        self._min_p_np = np.zeros(max_num_reqs, dtype=np.float32)
        self._repetition_penalty_np = np.ones(max_num_reqs, dtype=np.float32)
        self._frequency_penalty_np = np.zeros(max_num_reqs, dtype=np.float32)
        self._presence_penalty_np = np.zeros(max_num_reqs, dtype=np.float32)
        self._use_penalty = np.zeros(max_num_reqs, dtype=bool)
        self._use_allowed = np.zeros(max_num_reqs, dtype=bool)
        self._allowed_token_ids: dict[int, list[int]] = {}
        self._allowed_mask: torch.Tensor | None = None
        self._use_logit_bias = np.zeros(max_num_reqs, dtype=bool)
        self._logit_bias: dict[int, dict[int, float]] = {}
        self._use_bad_words = np.zeros(max_num_reqs, dtype=bool)
        self._bad_words_token_ids: dict[int, list[list[int]]] = {}
        self._num_logprobs = np.full(max_num_reqs, fill_value=-1, dtype=np.int32)
        self._seeds_set = np.zeros(max_num_reqs, dtype=bool)
        # Source generators keyed by req_idx (MRV1 CPU RNG cache semantics).
        self._source_generators: dict[int, torch.Generator] = {}

    def add_request(
        self,
        req_idx: int,
        sampling_params: SamplingParams,
    ) -> None:
        # vLLM #56497 dropped the prompt_len argument from
        # Sampler.add_request (and added custom logits processors).
        self._add_request_impl(req_idx, sampling_params)

    def _add_request_impl(
        self,
        req_idx: int,
        sampling_params: SamplingParams,
    ) -> None:
        unsupported = []
        bad_words = getattr(sampling_params, "bad_words", None) or []
        bad_words_token_ids = getattr(sampling_params, "bad_words_token_ids", None)
        if bad_words and not bad_words_token_ids:
            unsupported.append("bad_words (not tokenized)")
        if unsupported:
            raise NotImplementedError(
                f"Unsupported sampling parameters on model runner v2 for 310P: {', '.join(unsupported)}."
            )
        if not (0 <= req_idx < self.max_num_reqs):
            return

        temperature = float(sampling_params.temperature)
        top_p = float(sampling_params.top_p)
        top_k = int(sampling_params.top_k)
        if self.vocab_size <= 0:
            # Late-bind from logits on first sample if caller omitted vocab_size.
            vocab_size = max(top_k, 1)
        else:
            vocab_size = self.vocab_size
        if top_k <= 0 or top_k > vocab_size:
            top_k = vocab_size

        self._temperature_np[req_idx] = temperature
        self._top_p_np[req_idx] = top_p
        self._top_k_np[req_idx] = top_k
        self.temperature_cpu[req_idx] = temperature
        self.top_p_cpu[req_idx] = top_p
        self.top_k_cpu[req_idx] = top_k
        self._stage_postprocess_params(req_idx, sampling_params)

        allowed_token_ids = getattr(sampling_params, "allowed_token_ids", None)
        if allowed_token_ids:
            self._allowed_token_ids[req_idx] = list(allowed_token_ids)
            self._use_allowed[req_idx] = True
            if self.vocab_size > 0:
                self._materialize_allowed_mask(req_idx)
        else:
            self._allowed_token_ids.pop(req_idx, None)
            self._use_allowed[req_idx] = False
            if self._allowed_mask is not None:
                self._allowed_mask[req_idx].fill_(False)

        logit_bias = getattr(sampling_params, "logit_bias", None)
        if logit_bias:
            self._logit_bias[req_idx] = {int(k): float(v) for k, v in logit_bias.items()}
            self._use_logit_bias[req_idx] = True
        else:
            self._logit_bias.pop(req_idx, None)
            self._use_logit_bias[req_idx] = False

        if bad_words_token_ids:
            self._bad_words_token_ids[req_idx] = [list(seq) for seq in bad_words_token_ids]
            self._use_bad_words[req_idx] = True
        else:
            self._bad_words_token_ids.pop(req_idx, None)
            self._use_bad_words[req_idx] = False

        logprobs = sampling_params.logprobs
        self._num_logprobs[req_idx] = -1 if logprobs is None else int(logprobs)

        seed = getattr(sampling_params, "seed", None)
        self._seeds_set[req_idx] = seed is not None
        if seed is None:
            seed = int(np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX))
        self.seeds_cpu[req_idx] = int(seed)
        self._sampling_state_dirty = True

        if self._seeds_set[req_idx]:
            source = torch.Generator(device="cpu")
            source.manual_seed(int(seed))
            self._source_generators[req_idx] = source
        else:
            self._source_generators.pop(req_idx, None)

    def _stage_postprocess_params(self, req_idx: int, sampling_params: SamplingParams) -> None:
        """Keep CPU and NumPy post-processing state consistent on slot reuse."""
        for name in ("min_p", "repetition_penalty", "frequency_penalty", "presence_penalty"):
            value = float(getattr(sampling_params, name))
            getattr(self, f"_{name}_np")[req_idx] = value
            getattr(self, f"{name}_cpu")[req_idx] = value
        self._use_penalty[req_idx] = use_penalty(sampling_params)

    def _copy_postprocess_params_to_device(self) -> None:
        for name in ("min_p", "repetition_penalty", "frequency_penalty", "presence_penalty"):
            getattr(self.sampling_states, name).gpu.copy_(getattr(self, f"{name}_cpu"), non_blocking=True)

    def apply_staged_writes(self) -> None:
        if not self._sampling_state_dirty:
            return
        # MTP proposer is first device consumer. Bulk H2D copies replace
        # per-request scalar assignments.
        self.sampling_states.temperature.gpu.copy_(self.temperature_cpu, non_blocking=True)
        self.sampling_states.seeds.gpu.copy_(self.seeds_cpu, non_blocking=True)
        self.sampling_states.top_k.gpu.copy_(self.top_k_cpu, non_blocking=True)
        self.sampling_states.top_p.gpu.copy_(self.top_p_cpu, non_blocking=True)
        self._copy_postprocess_params_to_device()
        self._sampling_state_dirty = False

    def _maybe_bind_vocab_size(self, vocab_size: int) -> None:
        if self.vocab_size > 0 or vocab_size <= 0:
            return
        self.vocab_size = int(vocab_size)
        self.sampling_states.vocab_size = self.vocab_size
        # Replace unset top_k defaults (initialized to 1 when vocab was unknown).
        unset = self._top_k_np <= 1
        self._top_k_np[unset] = self.vocab_size
        self.top_k_cpu[unset] = self.vocab_size
        self.sampling_states.top_k.gpu[unset] = self.vocab_size
        for req_idx in list(self._allowed_token_ids):
            self._materialize_allowed_mask(req_idx)

    def _ensure_allowed_mask(self) -> torch.Tensor:
        if self._allowed_mask is None:
            if self.vocab_size <= 0:
                raise RuntimeError("allowed_token_ids requires vocab_size before apply.")
            self._allowed_mask = torch.zeros(
                self.max_num_reqs,
                self.vocab_size,
                dtype=torch.bool,
                device=self.device,
            )
        return self._allowed_mask

    def _materialize_allowed_mask(self, req_idx: int) -> None:
        allowed = self._allowed_token_ids.get(req_idx)
        if not allowed:
            return
        materialize_allowed_mask(self._ensure_allowed_mask(), req_idx, allowed, self.vocab_size)

    def _batch_needs_logits_processing(self, idx_mapping_np: np.ndarray) -> bool:
        temp = self._temperature_np[idx_mapping_np]
        return bool(
            np.any((temp != 0.0) & (temp != 1.0))
            or np.any(self._top_k_np[idx_mapping_np] != max(self.vocab_size, 1))
            or np.any(self._top_p_np[idx_mapping_np] != 1.0)
            or np.any(self._min_p_np[idx_mapping_np] != 0.0)
            or np.any(self._use_penalty[idx_mapping_np])
            or np.any(self._use_allowed[idx_mapping_np])
            or np.any(self._use_logit_bias[idx_mapping_np])
            or np.any(self._use_bad_words[idx_mapping_np])
        )

    def _any_random(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self._temperature_np[idx_mapping_np] >= _SAMPLING_EPS))

    def _batch_max_num_logprobs(self, idx_mapping_np: np.ndarray) -> int | None:
        vals = self._num_logprobs[idx_mapping_np]
        active = vals[vals >= 0]
        if active.size == 0:
            return None
        return int(active.max())

    def _apply_penalties(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor | None = None,
        expanded_local_pos: torch.Tensor | None = None,
    ) -> None:
        if not np.any(self._use_penalty[idx_mapping_np]):
            return
        if self.req_states is None:
            raise RuntimeError("310P MRV2 penalties require req_states bound on Ascend310PSampler.")
        expanded_np = expanded_idx_mapping.detach().to("cpu").numpy()
        prompt_t, output_t = build_penalty_token_tensors(
            req_states=self.req_states,
            use_penalty_mask=self._use_penalty,
            expanded_idx_mapping_np=expanded_np,
            vocab_size=self.vocab_size,
            device=self.device,
            input_ids=input_ids,
            expanded_local_pos=expanded_local_pos,
        )
        apply_penalties(
            logits,
            prompt_t,
            output_t,
            self.sampling_states.presence_penalty.gpu[expanded_idx_mapping],
            self.sampling_states.frequency_penalty.gpu[expanded_idx_mapping],
            self.sampling_states.repetition_penalty.gpu[expanded_idx_mapping],
        )

    def _apply_allowed_token_ids(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if not np.any(self._use_allowed[idx_mapping_np]):
            return
        for req_idx in np.unique(idx_mapping_np):
            req_idx = int(req_idx)
            if req_idx >= 0 and self._use_allowed[req_idx] and req_idx in self._allowed_token_ids:
                self._materialize_allowed_mask(req_idx)
        assert self._allowed_mask is not None
        apply_allowed_token_ids(logits, self._allowed_mask, expanded_idx_mapping)

    def _apply_logit_bias(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if not np.any(self._use_logit_bias[idx_mapping_np]):
            return
        apply_logit_bias(
            logits,
            expanded_idx_mapping,
            logit_bias=self._logit_bias,
            use_logit_bias=self._use_logit_bias,
        )

    def _apply_bad_words(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor | None = None,
        expanded_local_pos: torch.Tensor | None = None,
    ) -> None:
        if not np.any(self._use_bad_words[idx_mapping_np]):
            return
        if self.req_states is None:
            raise RuntimeError("310P MRV2 bad_words require req_states bound on Ascend310PSampler.")
        apply_bad_words(
            logits,
            req_states=self.req_states,
            bad_words_token_ids=self._bad_words_token_ids,
            use_bad_words=self._use_bad_words,
            expanded_idx_mapping=expanded_idx_mapping,
            input_ids=input_ids,
            expanded_local_pos=expanded_local_pos,
        )

    def _apply_postprocess(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor | None = None,
        expanded_local_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Allowed → logit_bias → bad_words → penalties → temp → min_p → top-k/p."""
        self._apply_allowed_token_ids(logits, expanded_idx_mapping, idx_mapping_np)
        self._apply_logit_bias(logits, expanded_idx_mapping, idx_mapping_np)
        self._apply_bad_words(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids=input_ids,
            expanded_local_pos=expanded_local_pos,
        )
        self._apply_penalties(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids=input_ids,
            expanded_local_pos=expanded_local_pos,
        )
        apply_temperature(logits, expanded_idx_mapping, self.sampling_states.temperature.gpu)
        apply_min_p(logits, expanded_idx_mapping, self.sampling_states.min_p.gpu)
        do_top_k = np.any(self._top_k_np[idx_mapping_np] != max(self.vocab_size, 1))
        do_top_p = np.any(self._top_p_np[idx_mapping_np] != 1.0)
        top_k = self.sampling_states.top_k.gpu[expanded_idx_mapping] if do_top_k else None
        top_p = self.sampling_states.top_p.gpu[expanded_idx_mapping] if do_top_p else None
        return apply_top_k_top_p_npu(logits, top_k, top_p)

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor | None = None,
        expanded_local_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """FP32 copy + post-process for MTP rejection verify.

        Mirrors upstream ``Sampler.apply_sampling_params`` so
        ``RejectionSampler310V2`` matches MRV1 tempered verify.
        """
        self._maybe_bind_vocab_size(logits.shape[-1])
        processed = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
        if not self._batch_needs_logits_processing(idx_mapping_np):
            return processed
        return self._apply_postprocess(
            processed,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids=input_ids,
            expanded_local_pos=expanded_local_pos,
        )

    def _build_row_generators(
        self,
        expanded_idx_mapping_np: np.ndarray,
    ) -> dict[int, torch.Generator]:
        """Map logit-row index → CPU generator for seeded random requests."""
        sources: dict[int, torch.Generator] = {}
        for row_idx, req_idx in enumerate(expanded_idx_mapping_np.tolist()):
            if req_idx < 0:
                continue
            if self._temperature_np[req_idx] < _SAMPLING_EPS:
                continue
            source = self._source_generators.get(int(req_idx))
            if source is not None:
                sources[row_idx] = source
        return _prepare_cpu_generators_310p(sources)

    def __call__(self, logits: torch.Tensor, input_batch) -> SamplerOutput:
        self._maybe_bind_vocab_size(logits.shape[-1])

        expanded_idx_mapping = input_batch.expanded_idx_mapping
        idx_mapping_np = input_batch.idx_mapping_np
        num_reqs = input_batch.num_reqs
        input_ids = getattr(input_batch, "input_ids", None)
        if input_ids is not None:
            input_ids = input_ids[input_batch.logits_indices]
        expanded_local_pos = getattr(input_batch, "expanded_local_pos", None)
        max_num_logprobs = self._batch_max_num_logprobs(idx_mapping_np)
        # Gather logprobs from raw logits before any post-process masking.
        raw_logprobs = None
        if max_num_logprobs is not None:
            raw_logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)

        # Fast path: all greedy and no logits processors → argmax.
        if not self._any_random(idx_mapping_np) and not self._batch_needs_logits_processing(idx_mapping_np):
            sampled = logits.argmax(dim=-1).to(torch.int32)
            num_sampled = input_batch.seq_lens.new_ones(num_reqs)
            logprobs_tensors = None
            if raw_logprobs is not None:
                logprobs_tensors = gather_logprobs(raw_logprobs, max_num_logprobs, sampled)
            return SamplerOutput(
                sampled_token_ids=sampled.view(-1, 1),
                logprobs_tensors=logprobs_tensors,
                num_nans=None,
                num_sampled=num_sampled,
                num_rejected=torch.zeros_like(num_sampled),
            )

        # Copy to FP32 before in-place post-process, matching upstream MRV2
        # Sampler.apply_sampling_params.
        processed = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
        if self._batch_needs_logits_processing(idx_mapping_np):
            processed = self._apply_postprocess(
                processed,
                expanded_idx_mapping,
                idx_mapping_np,
                input_ids=input_ids,
                expanded_local_pos=expanded_local_pos,
            )

        if not self._any_random(idx_mapping_np):
            sampled = processed.argmax(dim=-1).to(torch.int32)
        else:
            # Greedy rows keep argmax; random rows use MRV1 inverse-CDF sampling.
            greedy = processed.argmax(dim=-1)
            probs = processed.softmax(dim=-1, dtype=torch.float32)
            expanded_np = expanded_idx_mapping.detach().to("cpu").numpy()
            generators = self._build_row_generators(expanded_np)
            random_sampled = _random_sample_310p(probs, generators)
            is_random = self.sampling_states.temperature.gpu[expanded_idx_mapping] >= _SAMPLING_EPS
            sampled = torch.where(is_random, random_sampled, greedy).to(torch.int32)

        logprobs_tensors = None
        if raw_logprobs is not None:
            logprobs_tensors = gather_logprobs(raw_logprobs, max_num_logprobs, sampled)

        num_sampled = input_batch.seq_lens.new_ones(num_reqs)
        sampled_2d = sampled.view(-1, 1)
        return SamplerOutput(
            sampled_token_ids=sampled_2d,
            logprobs_tensors=logprobs_tensors,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=torch.zeros_like(num_sampled),
        )
