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

import numpy as np
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_ascend._310p.sample.sampler import (
    _prepare_cpu_generators_310p,
    _random_sample_310p,
)

_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max
_SAMPLING_EPS = 1e-5


def _apply_temperature_pytorch(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """In-place temperature scaling (Triton-free; skips temp in {0, 1})."""
    temps = temperature[expanded_idx_mapping].to(dtype=torch.float32)
    need_scale = (temps != 0.0) & (temps != 1.0)
    if not bool(need_scale.any().item()):
        return
    # Work in FP32 for numerical stability, write back to logits dtype.
    scaled = logits.to(torch.float32) / temps.unsqueeze(-1)
    logits.copy_(torch.where(need_scale.unsqueeze(-1), scaled.to(logits.dtype), logits))


class Ascend310PSampler:
    """Triton-free sampler for 310P MRV2 with MRV1-aligned temperature sampling.

    Exposes ``sampling_states.temperature/seeds/top_k/top_p`` so MTP draft
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
    ) -> None:
        self.penalties_state = SimpleNamespace(output_bin_counts=None)
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
        self.sampling_states = SimpleNamespace(
            temperature=SimpleNamespace(gpu=temperature_gpu),
            seeds=SimpleNamespace(gpu=seeds_gpu),
            top_k=SimpleNamespace(gpu=top_k_gpu),
            top_p=SimpleNamespace(gpu=top_p_gpu),
            vocab_size=self.vocab_size,
        )
        self._temperature_np = np.zeros(max_num_reqs, dtype=np.float32)
        self._top_k_np = np.full(max_num_reqs, fill_value=max(self.vocab_size, 1), dtype=np.int32)
        self._top_p_np = np.ones(max_num_reqs, dtype=np.float32)
        self._seeds_set = np.zeros(max_num_reqs, dtype=bool)
        # Source generators keyed by req_idx (MRV1 CPU RNG cache semantics).
        self._source_generators: dict[int, torch.Generator] = {}

    def add_request(
        self,
        req_idx: int,
        prompt_len: int,
        sampling_params: SamplingParams,
    ) -> None:
        del prompt_len
        unsupported = []
        if sampling_params.min_p != 0.0:
            unsupported.append("min_p")
        if sampling_params.repetition_penalty != 1.0:
            unsupported.append("repetition_penalty")
        if sampling_params.presence_penalty != 0.0 or sampling_params.frequency_penalty != 0.0:
            unsupported.append("presence/frequency penalty")
        if sampling_params.logprobs is not None or sampling_params.prompt_logprobs is not None:
            unsupported.append("logprobs")
        if (
            getattr(sampling_params, "bad_words", None)
            or getattr(sampling_params, "logit_bias", None)
            or getattr(sampling_params, "allowed_token_ids", None)
        ):
            unsupported.append("logits processors")
        if unsupported:
            # TODO: Support penalties / logprobs / min_p in a later 310P MRV2 iteration.
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
        self.sampling_states.temperature.gpu[req_idx] = temperature
        self.sampling_states.top_p.gpu[req_idx] = top_p
        self.sampling_states.top_k.gpu[req_idx] = top_k

        seed = getattr(sampling_params, "seed", None)
        self._seeds_set[req_idx] = seed is not None
        if seed is None:
            seed = int(np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX))
        self.sampling_states.seeds.gpu[req_idx] = int(seed)

        if self._seeds_set[req_idx]:
            source = torch.Generator(device="cpu")
            source.manual_seed(int(seed))
            self._source_generators[req_idx] = source
        else:
            self._source_generators.pop(req_idx, None)

    def apply_staged_writes(self) -> None:
        # Params are written directly to device buffers in ``add_request``.
        pass

    def _maybe_bind_vocab_size(self, vocab_size: int) -> None:
        if self.vocab_size > 0 or vocab_size <= 0:
            return
        self.vocab_size = int(vocab_size)
        self.sampling_states.vocab_size = self.vocab_size
        # Replace unset top_k defaults (initialized to 1 when vocab was unknown).
        unset = self._top_k_np <= 1
        self._top_k_np[unset] = self.vocab_size
        self.sampling_states.top_k.gpu[unset] = self.vocab_size

    def _batch_needs_logits_processing(self, idx_mapping_np: np.ndarray) -> bool:
        temp = self._temperature_np[idx_mapping_np]
        return bool(
            np.any((temp != 0.0) & (temp != 1.0))
            or np.any(self._top_k_np[idx_mapping_np] != max(self.vocab_size, 1))
            or np.any(self._top_p_np[idx_mapping_np] != 1.0)
        )

    def _any_random(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self._temperature_np[idx_mapping_np] >= _SAMPLING_EPS))

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor:
        """FP32 copy + temperature + top-k/top-p for MTP rejection verify.

        Mirrors upstream ``Sampler.apply_sampling_params`` (without penalties /
        logit bias) so ``RejectionSampler310V2`` matches MRV1 tempered verify.
        """
        self._maybe_bind_vocab_size(logits.shape[-1])
        processed = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
        if not self._batch_needs_logits_processing(idx_mapping_np):
            return processed
        _apply_temperature_pytorch(
            processed,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
        )
        do_top_k = np.any(self._top_k_np[idx_mapping_np] != max(self.vocab_size, 1))
        do_top_p = np.any(self._top_p_np[idx_mapping_np] != 1.0)
        top_k = self.sampling_states.top_k.gpu[expanded_idx_mapping] if do_top_k else None
        top_p = self.sampling_states.top_p.gpu[expanded_idx_mapping] if do_top_p else None
        return apply_top_k_top_p(processed, top_k, top_p)

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

        # Fast path: all greedy and no top-k/top-p → argmax (previous behavior).
        if not self._any_random(idx_mapping_np) and not self._batch_needs_logits_processing(idx_mapping_np):
            sampled = logits.argmax(dim=-1).to(torch.int32)
            num_sampled = input_batch.seq_lens.new_ones(num_reqs)
            return SamplerOutput(
                sampled_token_ids=sampled.view(-1, 1),
                logprobs_tensors=None,
                num_nans=None,
                num_sampled=num_sampled,
                num_rejected=torch.zeros_like(num_sampled),
            )

        # Copy to FP32 before in-place temperature / top-k / top-p, matching
        # upstream MRV2 Sampler.apply_sampling_params.
        processed = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        if self._batch_needs_logits_processing(idx_mapping_np):
            _apply_temperature_pytorch(
                processed,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
            )
            do_top_k = np.any(self._top_k_np[idx_mapping_np] != max(self.vocab_size, 1))
            do_top_p = np.any(self._top_p_np[idx_mapping_np] != 1.0)
            top_k = self.sampling_states.top_k.gpu[expanded_idx_mapping] if do_top_k else None
            top_p = self.sampling_states.top_p.gpu[expanded_idx_mapping] if do_top_p else None
            processed = apply_top_k_top_p(processed, top_k, top_p)

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

        num_sampled = input_batch.seq_lens.new_ones(num_reqs)
        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=torch.zeros_like(num_sampled),
        )
