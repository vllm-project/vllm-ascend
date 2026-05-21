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

from collections.abc import Iterable

import numpy as np
import torch
from vllm.logger import logger
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu.sample.bad_words import apply_bad_words as _apply_bad_words_op
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature as _apply_temperature
from vllm.v1.worker.gpu.sample.min_p import apply_min_p as _apply_min_p
from vllm.v1.worker.gpu.sample.penalties import apply_penalties as _apply_penalties_op
from vllm.v1.worker.gpu.sample.states import apply_top_k_top_p as _apply_top_k_top_p

from vllm_ascend.worker.v1.sample.context import V1MappingContext


class LogitsProcessor:
    """Configurable logits processing pipeline for the GPU sampler bridge.

    The default pipeline follows the upstream sampler stage order while
    directly calling upstream sampling ops:
    1. Logit bias stage (allowed token IDs + non-argmax-invariant processors)
    2. Penalties (repetition, frequency, presence)
    3. Bad words masking
    4. Temperature scaling
    5. Argmax-invariant logits processors (min_p, etc.)
    6. Top-k / Top-p filtering

    Mode "default": uses upstream v1 sampling ops where the per-step metadata
                    can be mapped safely to logits rows.
    Mode "skip":    skips all processing, converts to float32, warns on
                    incompatible parameters.
    Mode "fused":   reserved for Phase 3 fused kernel implementation.
    """

    # Parameters that are incompatible with skip mode
    _SKIP_INCOMPATIBLE = {
        "penalties": (
            "repetition_penalty != 1.0",
            "frequency_penalty != 0.0",
            "presence_penalty != 0.0",
        ),
        "bad_words": ("bad_words is not empty",),
        "logit_bias": (
            "logit_bias is not empty",
            "allowed_token_ids is not empty",
        ),
        "non_argmax_invariant": (
            "non_argmax_invariant logits processors are active",
        ),
        "filtering": (
            "top_k != -1 (or vocab_size)",
            "top_p != 1.0",
            "min_p != 0.0",
        ),
    }

    def __init__(self, mode: str):
        self.mode: str = mode  # "default" | "skip" | "fused"
        self._skip_warnings_issued: set[str] = set()

    def apply(
        self,
        logits: torch.Tensor,  # [num_logits, vocab_size]
        sampling_metadata: SamplingMetadata,
        ctx: V1MappingContext,
        num_speculative_tokens: int = 1,
    ) -> torch.Tensor:
        """Apply logits processing. Returns processed logits."""
        if self.mode == "default":
            return self._apply_default(logits, sampling_metadata, ctx, num_speculative_tokens)
        elif self.mode == "skip":
            return self._apply_skip(logits, sampling_metadata, ctx)
        elif self.mode == "fused":
            raise NotImplementedError(
                "fused logits_processing_mode is not yet implemented. "
                "This will be available in Phase 3."
            )
        raise ValueError(f"Unknown logits_processing_mode: {self.mode}")

    # ------------------------------------------------------------------
    # Default mode: full pipeline with Ascend-optimized kernels
    # ------------------------------------------------------------------

    def _apply_default(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        ctx: V1MappingContext,
        num_speculative_tokens: int,
    ) -> torch.Tensor:
        """Full logits processing pipeline using upstream v1 sampling ops."""
        processed_logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
        self._apply_allowed_token_ids(processed_logits, sampling_metadata, ctx)
        processed_logits = self._apply_non_argmax_invariant(
            processed_logits,
            sampling_metadata,
            ctx,
        )
        self._apply_penalties(
            processed_logits,
            sampling_metadata,
            ctx,
            num_speculative_tokens,
        )
        self._apply_bad_words(processed_logits, sampling_metadata, ctx)
        self._apply_temperature(processed_logits, sampling_metadata, ctx)
        processed_logits = self._apply_argmax_invariant(
            processed_logits,
            sampling_metadata,
            ctx,
        )
        return self._apply_top_k_top_p(processed_logits, sampling_metadata, ctx)

    def _apply_allowed_token_ids(self, logits, sampling_metadata, ctx):
        """Apply allowed token IDs whitelist masking."""
        allowed_mask = sampling_metadata.allowed_token_ids_mask
        if allowed_mask is None:
            return
        logits.masked_fill_(self._expand_request_rows(allowed_mask, ctx, logits.device), float("-inf"))

    def _apply_bad_words(self, logits, sampling_metadata, ctx):
        """Apply bad words masking using upstream worker/gpu/sample kernel."""
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        if not bad_words_token_ids:
            return

        (
            flat_bad_word_token_ids,
            bad_word_offsets,
            num_bad_words,
            all_token_ids,
            prompt_len,
            total_len,
            max_num_bad_words,
        ) = self._build_bad_words_state_tensors(
            bad_words_token_ids,
            sampling_metadata.output_token_ids,
            ctx,
            logits,
        )
        if max_num_bad_words == 0:
            return

        _apply_bad_words_op(
            logits,
            ctx.expanded_idx_mapping,
            flat_bad_word_token_ids,
            bad_word_offsets,
            num_bad_words,
            all_token_ids,
            prompt_len,
            total_len,
            ctx.input_ids.to(device=logits.device, dtype=torch.int32),
            ctx.expanded_local_pos,
            max_num_bad_words,
        )

    def _apply_non_argmax_invariant(self, logits, sampling_metadata, ctx):
        """Apply logits processors that can impact greedy sampling.

        These include logit_bias, min_tokens, thinking_token_budget,
        and any custom logits processors.
        """
        return self._apply_logits_processors(
            logits,
            sampling_metadata.logitsprocs.non_argmax_invariant,
            ctx,
        )

    def _apply_penalties(self, logits, sampling_metadata, ctx, num_speculative_tokens):
        """Apply penalties using upstream worker/gpu/sample kernel."""
        if sampling_metadata.no_penalties:
            return
        prompt_bin_mask, output_bin_counts = self._build_penalty_state_tensors(
            sampling_metadata.prompt_token_ids,
            sampling_metadata.output_token_ids,
            ctx,
            logits,
        )
        _apply_penalties_op(
            logits,
            ctx.expanded_idx_mapping,
            ctx.input_ids.to(device=logits.device, dtype=torch.int32),
            ctx.expanded_local_pos,
            self._request_param_tensor(
                sampling_metadata.repetition_penalties,
                ctx,
                logits.device,
                dtype=torch.float32,
            ),
            self._request_param_tensor(
                sampling_metadata.frequency_penalties,
                ctx,
                logits.device,
                dtype=torch.float32,
            ),
            self._request_param_tensor(
                sampling_metadata.presence_penalties,
                ctx,
                logits.device,
                dtype=torch.float32,
            ),
            prompt_bin_mask,
            output_bin_counts,
            num_speculative_tokens,
        )

    def _apply_temperature(self, logits, sampling_metadata, ctx):
        """Apply temperature scaling using Ascend Triton kernel."""
        temp = sampling_metadata.temperature
        if temp is None:
            return
        temp = self._request_param_tensor(temp, ctx, logits.device, dtype=torch.float32)
        if temp.is_cpu:
            temp_cpu = temp.numpy()
        else:
            temp_cpu = temp.cpu().numpy()
        if np.all((temp_cpu == 0.0) | (temp_cpu == 1.0)):
            return

        _apply_temperature(logits, ctx.expanded_idx_mapping, temp)

    def _apply_argmax_invariant(self, logits, sampling_metadata, ctx):
        """Apply logits processors that do not affect greedy sampling.

        These include min_p and any custom argmax-invariant processors.
        Applied after temperature, same as the v1 Sampler.
        """
        return self._apply_logits_processors(
            logits,
            sampling_metadata.logitsprocs.argmax_invariant,
            ctx,
        )

    def _apply_top_k_top_p(self, logits, sampling_metadata, ctx):
        """Apply top-k and top-p filtering using upstream op."""
        k = self._expand_optional_request_rows(sampling_metadata.top_k, ctx, logits.device)
        p = self._expand_optional_request_rows(sampling_metadata.top_p, ctx, logits.device)
        return _apply_top_k_top_p(logits, k, p)

    def _apply_logits_processors(
        self,
        logits: torch.Tensor,
        processors: Iterable,
        ctx: V1MappingContext,
    ) -> torch.Tensor:
        for processor in processors:
            if ctx.is_identity_request_mapping:
                logits = processor.apply(logits)
            elif self._try_apply_min_p_processor(logits, processor, ctx):
                continue
            elif self._try_apply_logit_bias_processor(logits, processor, ctx):
                continue
            elif hasattr(processor, "apply_with_spec_decode"):
                logits = processor.apply_with_spec_decode(
                    logits,
                    ctx.num_logits_per_req_np.tolist(),
                )
            else:
                raise NotImplementedError(
                    f"{type(processor).__name__} does not support expanded logits "
                    "in GpuSamplerBridge. Add a mapping-aware implementation before "
                    "enabling this processor on the bridge path."
                )
        return logits

    def _try_apply_min_p_processor(
        self,
        logits: torch.Tensor,
        processor,
        ctx: V1MappingContext,
    ) -> bool:
        if not (hasattr(processor, "min_p_count") and hasattr(processor, "get_min_p_by_index")):
            return False
        if processor.min_p_count <= 0:
            return True
        min_p = torch.empty(ctx.num_reqs, dtype=torch.float32, device=logits.device)
        for req_idx in range(ctx.num_reqs):
            min_p[req_idx] = float(processor.get_min_p_by_index(req_idx))
        _apply_min_p(logits, ctx.expanded_idx_mapping, min_p)
        return True

    def _try_apply_logit_bias_processor(
        self,
        logits: torch.Tensor,
        processor,
        ctx: V1MappingContext,
    ) -> bool:
        biases = getattr(processor, "biases", None)
        if biases is None:
            return False
        if not biases:
            return True

        row_indices: list[int] = []
        token_ids: list[int] = []
        bias_values: list[float] = []
        for row, req_idx in enumerate(ctx.idx_mapping_np.tolist()):
            req_biases = biases.get(req_idx)
            if not req_biases:
                continue
            for token_id, bias in req_biases.items():
                row_indices.append(row)
                token_ids.append(int(token_id))
                bias_values.append(float(bias))
        if row_indices:
            rows = torch.tensor(row_indices, dtype=torch.long, device=logits.device)
            tokens = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
            values = torch.tensor(bias_values, dtype=torch.float32, device=logits.device)
            logits[rows, tokens] += values
        return True

    def _build_penalty_state_tensors(
        self,
        prompt_token_ids,
        output_token_ids: list[list[int]],
        ctx: V1MappingContext,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_token_ids is None:
            raise ValueError("prompt_token_ids is required when penalties are active")

        vocab_size = int(logits.shape[-1])
        prompt_bin_mask_np = np.zeros((ctx.num_reqs, (vocab_size + 31) // 32), dtype=np.int32)
        prompt_rows = self._request_rows(prompt_token_ids, ctx, logits.device).detach().cpu().tolist()
        for req_idx, tokens in enumerate(prompt_rows):
            for token_id in tokens:
                token_id = int(token_id)
                if 0 <= token_id < vocab_size:
                    prompt_bin_mask_np[req_idx, token_id // 32] |= 1 << (token_id % 32)

        output_bin_counts_np = np.zeros((ctx.num_reqs, vocab_size), dtype=np.int32)
        for req_idx, tokens in enumerate(output_token_ids[: ctx.num_reqs]):
            for token_id in tokens:
                token_id = int(token_id)
                if 0 <= token_id < vocab_size:
                    output_bin_counts_np[req_idx, token_id] += 1

        return (
            torch.from_numpy(prompt_bin_mask_np).to(device=logits.device),
            torch.from_numpy(output_bin_counts_np).to(device=logits.device),
        )

    def _build_bad_words_state_tensors(
        self,
        bad_words_token_ids: dict[int, list[list[int]]],
        output_token_ids: list[list[int]],
        ctx: V1MappingContext,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        active_bad_words = {
            req_idx: words
            for req_idx, words in bad_words_token_ids.items()
            if req_idx < ctx.num_reqs and words
        }
        max_num_bad_words = max((len(words) for words in active_bad_words.values()), default=0)
        max_total_bad_word_tokens = max(
            (sum(len(word) for word in words) for words in active_bad_words.values()),
            default=1,
        )
        max_total_bad_word_tokens = max(max_total_bad_word_tokens, 1)

        bad_word_token_ids_np = np.zeros(
            (ctx.num_reqs, max_total_bad_word_tokens),
            dtype=np.int32,
        )
        bad_word_offsets_np = np.zeros(
            (ctx.num_reqs, max_num_bad_words + 1),
            dtype=np.int32,
        )
        num_bad_words_np = np.zeros(ctx.num_reqs, dtype=np.int32)
        for req_idx, words in active_bad_words.items():
            offset = 0
            num_bad_words_np[req_idx] = len(words)
            for word_idx, word in enumerate(words):
                next_offset = offset + len(word)
                bad_word_token_ids_np[req_idx, offset:next_offset] = word
                offset = next_offset
                bad_word_offsets_np[req_idx, word_idx + 1] = offset

        output_lens = [len(tokens) for tokens in output_token_ids[: ctx.num_reqs]]
        max_output_len = max(output_lens, default=0)
        all_token_ids_np = np.zeros((ctx.num_reqs, max(max_output_len, 1)), dtype=np.int32)
        for req_idx, tokens in enumerate(output_token_ids[: ctx.num_reqs]):
            if tokens:
                all_token_ids_np[req_idx, : len(tokens)] = tokens

        device = logits.device
        return (
            torch.from_numpy(bad_word_token_ids_np).to(device=device),
            torch.from_numpy(bad_word_offsets_np).to(device=device),
            torch.from_numpy(num_bad_words_np).to(device=device),
            torch.from_numpy(all_token_ids_np).to(device=device),
            torch.zeros(ctx.num_reqs, dtype=torch.int32, device=device),
            torch.tensor(output_lens, dtype=torch.int32, device=device),
            max_num_bad_words,
        )

    def _request_rows(
        self,
        value,
        ctx: V1MappingContext,
        device: torch.device,
    ) -> torch.Tensor:
        if value is None:
            raise ValueError("request tensor is required for this sampling stage")
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=device)
        elif value.device != device:
            value = value.to(device)
        if int(value.shape[0]) < ctx.num_reqs:
            raise ValueError("request tensor must cover all active requests")
        return value[: ctx.num_reqs]

    def _request_param_tensor(
        self,
        value,
        ctx: V1MappingContext,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=device, dtype=dtype)
        elif value.device != device:
            value = value.to(device)
        if dtype is not None and value.dtype != dtype:
            value = value.to(dtype=dtype)
        if value.ndim == 0:
            return value.expand(ctx.num_reqs)
        if int(value.shape[0]) < ctx.num_reqs:
            raise ValueError("request parameter tensor must cover all active requests")
        return value[:ctx.num_reqs]

    def _expand_optional_request_rows(
        self,
        value,
        ctx: V1MappingContext,
        device: torch.device,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        return self._expand_request_rows(value, ctx, device)

    def _expand_request_rows(
        self,
        value,
        ctx: V1MappingContext,
        device: torch.device,
    ) -> torch.Tensor:
        if value is None:
            raise ValueError("request tensor is required for this sampling stage")
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=device)
        elif value.device != device:
            value = value.to(device)
        if value.ndim == 0:
            return value.expand(ctx.num_logits)
        if ctx.is_identity_request_mapping and int(value.shape[0]) == ctx.num_logits:
            return value
        if int(value.shape[0]) < ctx.num_reqs:
            raise ValueError("request tensor must cover all active requests")
        return value.index_select(0, ctx.expanded_idx_mapping.to(device=value.device, dtype=torch.long))

    def _expand_bad_words(
        self,
        bad_words_token_ids: dict[int, list[list[int]]],
        ctx: V1MappingContext,
    ) -> dict[int, list[list[int]]]:
        if ctx.is_identity_request_mapping:
            return bad_words_token_ids
        return {
            row: bad_words_token_ids[req_idx]
            for row, req_idx in enumerate(ctx.idx_mapping_np.tolist())
            if req_idx in bad_words_token_ids
        }

    def _expand_output_token_ids(
        self,
        output_token_ids: list[list[int]],
        ctx: V1MappingContext,
    ) -> list[list[int]]:
        if not output_token_ids or ctx.is_identity_request_mapping:
            return output_token_ids

        local_pos = ctx.expanded_local_pos.detach().cpu().numpy()
        input_ids = ctx.input_ids.detach().cpu().tolist()
        expanded: list[list[int]] = []
        for row, req_idx in enumerate(ctx.idx_mapping_np.tolist()):
            tokens = list(output_token_ids[req_idx])
            pos = int(local_pos[row])
            if pos > 0:
                previous_rows = [
                    prev_row
                    for prev_row in range(row)
                    if int(ctx.idx_mapping_np[prev_row]) == req_idx
                ]
                tokens.extend(input_ids[prev_row] for prev_row in previous_rows[-pos:])
            expanded.append(tokens)
        return expanded

    # ------------------------------------------------------------------
    # Skip mode: skip all processing
    # ------------------------------------------------------------------

    def _apply_skip(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        ctx: V1MappingContext,
    ) -> torch.Tensor:
        """Skip all logits processing. Warn if incompatible params are present."""
        self._check_skip_compatibility(sampling_metadata, ctx)
        if logits.dtype == torch.float32:
            return logits
        return logits.to(dtype=torch.float32)

    def _check_skip_compatibility(self, sampling_metadata, ctx):
        """Issue warnings for parameters that skip mode ignores."""
        # Check penalties
        if not sampling_metadata.no_penalties:
            self._warn_once("penalties", self._SKIP_INCOMPATIBLE["penalties"])

        # Check bad words
        bad_words = sampling_metadata.bad_words_token_ids
        if bad_words and any(bool(bw) for bw in bad_words.values()):
            self._warn_once("bad_words", self._SKIP_INCOMPATIBLE["bad_words"])

        # Check non-argmax-invariant processors (logit_bias, min_tokens)
        logitsprocs = sampling_metadata.logitsprocs
        if logitsprocs is not None:
            if list(logitsprocs.non_argmax_invariant):
                self._warn_once(
                    "non_argmax_invariant",
                    self._SKIP_INCOMPATIBLE["non_argmax_invariant"],
                )
            if list(logitsprocs.argmax_invariant):
                self._warn_once("filtering", self._SKIP_INCOMPATIBLE["filtering"])

        # Check allowed_token_ids_mask
        if sampling_metadata.allowed_token_ids_mask is not None:
            self._warn_once("logit_bias", self._SKIP_INCOMPATIBLE["logit_bias"])

        # Check filtering params (top-k, top-p, min-p are skipped in skip mode)
        if self._has_top_k_or_top_p(sampling_metadata, ctx):
            self._warn_once("filtering", self._SKIP_INCOMPATIBLE["filtering"])

        if logitsprocs is not None and "filtering" not in self._skip_warnings_issued:
            for proc in logitsprocs.argmax_invariant:
                if hasattr(proc, "min_p_count") and proc.min_p_count > 0:
                    self._warn_once("filtering", ("min_p != 0.0 - min_p filtering is skipped in skip mode",))
                    break

    def _has_top_k_or_top_p(self, sampling_metadata, ctx) -> bool:
        if sampling_metadata.top_k is not None:
            return True

        top_p = sampling_metadata.top_p
        if isinstance(top_p, torch.Tensor):
            top_p_rows = self._expand_request_rows(top_p, ctx, top_p.device)
            if bool((top_p_rows != 1.0).any()):
                return True
        elif top_p is not None:
            return bool(np.any(np.asarray(top_p) != 1.0))
        return False

    def _warn_once(self, category: str, details: tuple[str, ...]):
        if category not in self._skip_warnings_issued:
            self._skip_warnings_issued.add(category)
            logger.warning(
                "logits_processing_mode='skip' but active requests use %s. "
                "Output may differ from default mode. Incompatible: %s",
                category,
                ", ".join(details),
            )
