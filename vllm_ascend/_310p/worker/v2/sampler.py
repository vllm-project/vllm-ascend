# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""Triton-free MRV2 sampler for Ascend 310P.

Aligns temperature / top-k / top-p / min_p / penalties / allowed_token_ids /
logit_bias / bad_words post-processing with MRV1's ``AscendSampler310``
(inverse-CDF random sampling) while keeping the MRV2 ``sampling_states``
surface required by MTP draft ``propose()``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from vllm.model_executor.layers.utils import apply_penalties as apply_penalties_vllm
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.bad_words import apply_bad_words
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


def _apply_min_p_pytorch(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    min_p: torch.Tensor,
) -> None:
    """In-place min_p filter matching MRV1 ``MinPLogitsProcessor`` semantics.

    Keeps tokens whose probability is at least ``min_p * max_prob``; equivalent
    to the upstream MRV2 log-space form ``logit >= max_logit + log(min_p)``.
    """
    min_p_rows = min_p[expanded_idx_mapping].to(dtype=torch.float32)
    need = min_p_rows != 0.0
    if not bool(need.any().item()):
        return
    # Probability-space form used by MRV1 builtin MinPLogitsProcessor.
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    max_probs = probs.amax(dim=-1, keepdim=True)
    thresholds = max_probs * min_p_rows.unsqueeze(-1)
    invalid = (probs < thresholds) & need.unsqueeze(-1)
    logits.masked_fill_(invalid, float("-inf"))


def _use_penalty_params(sampling_params: SamplingParams) -> bool:
    return bool(
        sampling_params.repetition_penalty != 1.0
        or sampling_params.presence_penalty != 0.0
        or sampling_params.frequency_penalty != 0.0
    )


class Ascend310PSampler:
    """Triton-free sampler for 310P MRV2 with MRV1-aligned post-processing.

    Exposes ``sampling_states.temperature/seeds/top_k/top_p/min_p`` so MTP draft
    ``propose()`` (and ``_dummy_run``) can read device buffers without UVA
    SamplingStates. Penalties rebuild prompt/output token tensors from
    ``req_states`` at apply time (MRV1 ``apply_penalties`` path).
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
        # True = disallowed (masked to -inf), matching MRV1 allowed_token_ids_mask.
        self._allowed_mask: torch.Tensor | None = None
        self._use_logit_bias = np.zeros(max_num_reqs, dtype=bool)
        self._logit_bias: dict[int, dict[int, float]] = {}
        self._use_bad_words = np.zeros(max_num_reqs, dtype=bool)
        self._bad_words_token_ids: dict[int, list[list[int]]] = {}
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
        if sampling_params.logprobs is not None or sampling_params.prompt_logprobs is not None:
            unsupported.append("logprobs")
        bad_words = getattr(sampling_params, "bad_words", None) or []
        bad_words_token_ids = getattr(sampling_params, "bad_words_token_ids", None)
        if bad_words and not bad_words_token_ids:
            unsupported.append("bad_words (not tokenized)")
        if unsupported:
            # TODO: Support logprobs in a later 310P MRV2 iteration.
            raise NotImplementedError(
                f"Unsupported sampling parameters on model runner v2 for 310P: {', '.join(unsupported)}."
            )
        if not (0 <= req_idx < self.max_num_reqs):
            return

        temperature = float(sampling_params.temperature)
        top_p = float(sampling_params.top_p)
        top_k = int(sampling_params.top_k)
        min_p = float(sampling_params.min_p)
        repetition_penalty = float(sampling_params.repetition_penalty)
        frequency_penalty = float(sampling_params.frequency_penalty)
        presence_penalty = float(sampling_params.presence_penalty)
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
        self._min_p_np[req_idx] = min_p
        self._repetition_penalty_np[req_idx] = repetition_penalty
        self._frequency_penalty_np[req_idx] = frequency_penalty
        self._presence_penalty_np[req_idx] = presence_penalty
        self._use_penalty[req_idx] = _use_penalty_params(sampling_params)
        self.temperature_cpu[req_idx] = temperature
        self.top_p_cpu[req_idx] = top_p
        self.top_k_cpu[req_idx] = top_k
        self.min_p_cpu[req_idx] = min_p
        self.repetition_penalty_cpu[req_idx] = repetition_penalty
        self.frequency_penalty_cpu[req_idx] = frequency_penalty
        self.presence_penalty_cpu[req_idx] = presence_penalty

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

    def apply_staged_writes(self) -> None:
        if not self._sampling_state_dirty:
            return
        # MTP proposer is first device consumer. Bulk H2D copies replace
        # per-request scalar assignments.
        self.sampling_states.temperature.gpu.copy_(self.temperature_cpu, non_blocking=True)
        self.sampling_states.seeds.gpu.copy_(self.seeds_cpu, non_blocking=True)
        self.sampling_states.top_k.gpu.copy_(self.top_k_cpu, non_blocking=True)
        self.sampling_states.top_p.gpu.copy_(self.top_p_cpu, non_blocking=True)
        self.sampling_states.min_p.gpu.copy_(self.min_p_cpu, non_blocking=True)
        self.sampling_states.repetition_penalty.gpu.copy_(self.repetition_penalty_cpu, non_blocking=True)
        self.sampling_states.frequency_penalty.gpu.copy_(self.frequency_penalty_cpu, non_blocking=True)
        self.sampling_states.presence_penalty.gpu.copy_(self.presence_penalty_cpu, non_blocking=True)
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
        mask = self._ensure_allowed_mask()
        mask[req_idx] = True
        ids = torch.tensor(allowed, dtype=torch.int64, device=self.device)
        ids = ids[(ids >= 0) & (ids < self.vocab_size)]
        if ids.numel():
            mask[req_idx, ids] = False

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

    def _build_penalty_token_tensors(
        self,
        expanded_idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor | None,
        expanded_local_pos: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build per-logit-row prompt/output token tensors (MRV1 pad=vocab_size)."""
        assert self.req_states is not None
        vocab_size = max(self.vocab_size, 1)
        num_rows = int(expanded_idx_mapping_np.shape[0])
        all_token_ids_cpu = self.req_states.all_token_ids.cpu
        prompt_lens = self.req_states.prompt_len.np
        total_lens = self.req_states.total_len.np

        prompt_lists: list[list[int]] = []
        output_lists: list[list[int]] = []
        max_prompt = 0
        max_output = 0
        local_pos_np = None if expanded_local_pos is None else expanded_local_pos.detach().to("cpu").numpy()
        input_ids_cpu = None if input_ids is None else input_ids.detach().to("cpu")

        for row_idx, req_idx in enumerate(expanded_idx_mapping_np.tolist()):
            req_idx = int(req_idx)
            if req_idx < 0 or not self._use_penalty[req_idx]:
                prompt_lists.append([])
                output_lists.append([])
                continue
            prompt_len = int(prompt_lens[req_idx])
            total_len = int(total_lens[req_idx])
            prompt = all_token_ids_cpu[req_idx, :prompt_len].tolist()
            output = all_token_ids_cpu[req_idx, prompt_len:total_len].tolist()
            # Match upstream MRV2 draft accumulation into output bin counts.
            if local_pos_np is not None and input_ids_cpu is not None:
                pos = int(local_pos_np[row_idx])
                if pos > 0:
                    start_idx = row_idx - pos
                    for prev_pos in range(pos):
                        output.append(int(input_ids_cpu[start_idx + prev_pos + 1].item()))
            prompt_lists.append(prompt)
            output_lists.append(output)
            max_prompt = max(max_prompt, len(prompt))
            max_output = max(max_output, len(output))

        max_prompt = max(max_prompt, 1)
        max_output = max(max_output, 1)
        prompt_t = torch.full((num_rows, max_prompt), fill_value=vocab_size, dtype=torch.int64, device=self.device)
        output_t = torch.full((num_rows, max_output), fill_value=vocab_size, dtype=torch.int64, device=self.device)
        for row_idx, (prompt, output) in enumerate(zip(prompt_lists, output_lists)):
            if prompt:
                prompt_t[row_idx, : len(prompt)] = torch.tensor(prompt, dtype=torch.int64)
            if output:
                output_t[row_idx, : len(output)] = torch.tensor(output, dtype=torch.int64)
        return prompt_t, output_t

    def _apply_penalties_pytorch(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor | None = None,
        expanded_local_pos: torch.Tensor | None = None,
    ) -> None:
        """In-place penalties via MRV1 ``apply_penalties`` (torch repetition path)."""
        if not np.any(self._use_penalty[idx_mapping_np]):
            return
        if self.req_states is None:
            raise RuntimeError("310P MRV2 penalties require req_states bound on Ascend310PSampler.")
        expanded_np = expanded_idx_mapping.detach().to("cpu").numpy()
        prompt_t, output_t = self._build_penalty_token_tensors(expanded_np, input_ids, expanded_local_pos)
        repetition = self.sampling_states.repetition_penalty.gpu[expanded_idx_mapping]
        frequency = self.sampling_states.frequency_penalty.gpu[expanded_idx_mapping]
        presence = self.sampling_states.presence_penalty.gpu[expanded_idx_mapping]
        apply_penalties_vllm(logits, prompt_t, output_t, presence, frequency, repetition)

    def _apply_allowed_token_ids(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        """Mask disallowed tokens to -inf (MRV1 allowed_token_ids_mask)."""
        if not np.any(self._use_allowed[idx_mapping_np]):
            return
        for req_idx in np.unique(idx_mapping_np):
            req_idx = int(req_idx)
            if req_idx >= 0 and self._use_allowed[req_idx] and req_idx in self._allowed_token_ids:
                self._materialize_allowed_mask(req_idx)
        assert self._allowed_mask is not None
        row_mask = self._allowed_mask[expanded_idx_mapping]
        logits.masked_fill_(row_mask, float("-inf"))

    def _apply_logit_bias(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        """Add per-token logit biases (MRV1 LogitBiasLogitsProcessor)."""
        if not np.any(self._use_logit_bias[idx_mapping_np]):
            return
        expanded_np = expanded_idx_mapping.detach().to("cpu").numpy()
        for row_idx, req_idx in enumerate(expanded_np.tolist()):
            req_idx = int(req_idx)
            if req_idx < 0 or not self._use_logit_bias[req_idx]:
                continue
            bias_map = self._logit_bias.get(req_idx)
            if not bias_map:
                continue
            for token_id, bias in bias_map.items():
                if 0 <= token_id < logits.shape[-1]:
                    logits[row_idx, token_id] += bias

    def _apply_bad_words_pytorch(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor | None = None,
        expanded_local_pos: torch.Tensor | None = None,
    ) -> None:
        """Ban completing bad-word sequences (MRV1 ``apply_bad_words``)."""
        if not np.any(self._use_bad_words[idx_mapping_np]):
            return
        if self.req_states is None:
            raise RuntimeError("310P MRV2 bad_words require req_states bound on Ascend310PSampler.")
        expanded_np = expanded_idx_mapping.detach().to("cpu").numpy()
        local_pos_np = None if expanded_local_pos is None else expanded_local_pos.detach().to("cpu").numpy()
        input_ids_cpu = None if input_ids is None else input_ids.detach().to("cpu")
        all_token_ids_cpu = self.req_states.all_token_ids.cpu
        total_lens = self.req_states.total_len.np

        bad_by_row: dict[int, list[list[int]]] = {}
        past_tokens: list[list[int]] = []
        for row_idx, req_idx in enumerate(expanded_np.tolist()):
            req_idx = int(req_idx)
            if req_idx < 0 or not self._use_bad_words[req_idx]:
                past_tokens.append([])
                continue
            total_len = int(total_lens[req_idx])
            past = all_token_ids_cpu[req_idx, :total_len].tolist()
            if local_pos_np is not None and input_ids_cpu is not None:
                pos = int(local_pos_np[row_idx])
                if pos > 0:
                    start_idx = row_idx - pos
                    for prev_pos in range(pos):
                        past.append(int(input_ids_cpu[start_idx + prev_pos + 1].item()))
            past_tokens.append(past)
            bad_by_row[row_idx] = self._bad_words_token_ids[req_idx]
        apply_bad_words(logits, bad_by_row, past_tokens)

    def _apply_postprocess(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor | None = None,
        expanded_local_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Allowed-ids → logit_bias → bad_words → penalties → temp → min_p → top-k/p."""
        self._apply_allowed_token_ids(logits, expanded_idx_mapping, idx_mapping_np)
        self._apply_logit_bias(logits, expanded_idx_mapping, idx_mapping_np)
        self._apply_bad_words_pytorch(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids=input_ids,
            expanded_local_pos=expanded_local_pos,
        )
        self._apply_penalties_pytorch(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids=input_ids,
            expanded_local_pos=expanded_local_pos,
        )
        _apply_temperature_pytorch(
            logits,
            expanded_idx_mapping,
            self.sampling_states.temperature.gpu,
        )
        _apply_min_p_pytorch(
            logits,
            expanded_idx_mapping,
            self.sampling_states.min_p.gpu,
        )
        do_top_k = np.any(self._top_k_np[idx_mapping_np] != max(self.vocab_size, 1))
        do_top_p = np.any(self._top_p_np[idx_mapping_np] != 1.0)
        top_k = self.sampling_states.top_k.gpu[expanded_idx_mapping] if do_top_k else None
        top_p = self.sampling_states.top_p.gpu[expanded_idx_mapping] if do_top_p else None
        return apply_top_k_top_p(logits, top_k, top_p)

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor | None = None,
        expanded_local_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """FP32 copy + penalties / temperature / min_p / top-k/top-p for verify."""
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
        expanded_local_pos = getattr(input_batch, "expanded_local_pos", None)

        # Fast path: all greedy and no logits post-process → argmax.
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

        # Copy to FP32 before in-place post-process.
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

        num_sampled = input_batch.seq_lens.new_ones(num_reqs)
        sampled_2d = sampled.view(-1, 1)
        return SamplerOutput(
            sampled_token_ids=sampled_2d,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=torch.zeros_like(num_sampled),
        )
