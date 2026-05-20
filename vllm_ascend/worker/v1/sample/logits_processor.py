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
from vllm.v1.worker.gpu.sample.sampler import Sampler as _RefactoredSampler

from vllm_ascend.worker.v1.sample.context import V1MappingContext


class _ApplySamplingParamsBridge:
    """Adapts old v1 SamplingMetadata to upstream apply_sampling_params."""

    def __init__(self, owner: "LogitsProcessor"):
        self._owner = owner
        self._sampling_metadata: SamplingMetadata | None = None
        self._ctx: V1MappingContext | None = None
        self.num_speculative_tokens = 1

        self.logit_bias_state = _MetadataLogitBiasState(self)
        self.penalties_state = _MetadataPenaltiesState(self)
        self.bad_words_state = _MetadataBadWordsState(self)
        self.sampling_states = _MetadataSamplingStates(self)

    def apply(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        ctx: V1MappingContext,
        num_speculative_tokens: int,
    ) -> torch.Tensor:
        self._sampling_metadata = sampling_metadata
        self._ctx = ctx
        self.num_speculative_tokens = num_speculative_tokens
        try:
            return _RefactoredSampler.apply_sampling_params(
                self,
                logits,
                ctx.expanded_idx_mapping,
                ctx.idx_mapping_np,
                ctx.pos,
                ctx.input_ids,
                ctx.expanded_local_pos,
            )
        finally:
            self._sampling_metadata = None
            self._ctx = None

    @property
    def owner(self) -> "LogitsProcessor":
        return self._owner

    def current(self) -> tuple[SamplingMetadata, V1MappingContext]:
        assert self._sampling_metadata is not None
        assert self._ctx is not None
        return self._sampling_metadata, self._ctx


class _MetadataLogitBiasState:

    def __init__(self, bridge: _ApplySamplingParamsBridge):
        self._bridge = bridge

    def apply_logit_bias(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
    ) -> None:
        sampling_metadata, ctx = self._bridge.current()
        owner = self._bridge.owner
        owner._apply_allowed_token_ids(logits, sampling_metadata, ctx)
        processed_logits = owner._apply_non_argmax_invariant(
            logits,
            sampling_metadata,
            ctx,
        )
        if processed_logits is not logits:
            logits.copy_(processed_logits)


class _MetadataPenaltiesState:

    def __init__(self, bridge: _ApplySamplingParamsBridge):
        self._bridge = bridge

    def apply_penalties(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        num_speculative_tokens: int,
    ) -> None:
        sampling_metadata, ctx = self._bridge.current()
        self._bridge.owner._apply_penalties(
            logits,
            sampling_metadata,
            ctx,
            num_speculative_tokens,
        )


class _MetadataBadWordsState:

    def __init__(self, bridge: _ApplySamplingParamsBridge):
        self._bridge = bridge

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        sampling_metadata, ctx = self._bridge.current()
        self._bridge.owner._apply_bad_words(logits, sampling_metadata, ctx)


class _MetadataSamplingStates:

    def __init__(self, bridge: _ApplySamplingParamsBridge):
        self._bridge = bridge

    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        sampling_metadata, ctx = self._bridge.current()
        self._bridge.owner._apply_temperature(logits, sampling_metadata, ctx)

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        sampling_metadata, ctx = self._bridge.current()
        processed_logits = self._bridge.owner._apply_argmax_invariant(
            logits,
            sampling_metadata,
            ctx,
        )
        if processed_logits is not logits:
            logits.copy_(processed_logits)

    def apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor:
        sampling_metadata, ctx = self._bridge.current()
        return self._bridge.owner._apply_top_k_top_p(logits, sampling_metadata, ctx)


class LogitsProcessor:
    """Configurable logits processing pipeline for the v1 sampler adapter.

    The default pipeline delegates stage orchestration to upstream
    ``Sampler.apply_sampling_params``:
    1. Logit bias stage (allowed token IDs + non-argmax-invariant processors)
    2. Penalties (repetition, frequency, presence)
    3. Bad words masking
    4. Temperature scaling
    5. Argmax-invariant logits processors (min_p, etc.)
    6. Top-k / Top-p filtering

    Mode "default": uses upstream sampler orchestration and Ascend kernels
                    where the v1 per-step metadata can be mapped safely to
                    logits rows.
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
        self._apply_sampling_params_bridge = _ApplySamplingParamsBridge(self)

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
        """Full logits processing pipeline using upstream orchestration."""
        return self._apply_sampling_params_bridge.apply(
            logits,
            sampling_metadata,
            ctx,
            num_speculative_tokens,
        )

    def _apply_allowed_token_ids(self, logits, sampling_metadata, ctx):
        """Apply allowed token IDs whitelist masking."""
        allowed_mask = sampling_metadata.allowed_token_ids_mask
        if allowed_mask is None:
            return
        logits.masked_fill_(self._expand_request_rows(allowed_mask, ctx, logits.device), float("-inf"))

    def _apply_bad_words(self, logits, sampling_metadata, ctx):
        """Apply bad words masking using upstream v1 implementation."""
        bad_words_token_ids = sampling_metadata.bad_words_token_ids
        if not bad_words_token_ids:
            return
        from vllm.v1.sample.ops.bad_words import apply_bad_words

        apply_bad_words(
            logits,
            self._expand_bad_words(bad_words_token_ids, ctx),
            self._expand_output_token_ids(sampling_metadata.output_token_ids, ctx),
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
        """Apply repetition/frequency/presence penalties using Ascend kernel."""
        if sampling_metadata.no_penalties:
            return
        from vllm_ascend.sample.penalties import apply_all_penalties

        apply_all_penalties(
            logits,
            self._expand_request_rows(sampling_metadata.prompt_token_ids, ctx, logits.device),
            self._expand_request_rows(sampling_metadata.presence_penalties, ctx, logits.device),
            self._expand_request_rows(sampling_metadata.frequency_penalties, ctx, logits.device),
            self._expand_request_rows(sampling_metadata.repetition_penalties, ctx, logits.device),
            self._expand_output_token_ids(sampling_metadata.output_token_ids, ctx),
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

        from vllm_ascend.worker.v2.sample.gumbel import apply_temperature

        apply_temperature(logits, ctx.expanded_idx_mapping, temp)

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
        """Apply top-k and top-p filtering using existing Ascend implementation."""
        from vllm_ascend.sample.sampler import apply_top_k_top_p

        k = self._expand_optional_request_rows(sampling_metadata.top_k, ctx, logits.device)
        p = self._expand_optional_request_rows(sampling_metadata.top_p, ctx, logits.device)
        return apply_top_k_top_p(logits, k, p)

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
                    "in V1SamplerAdapter. Add a mapping-aware implementation before "
                    "enabling this processor on the adapter path."
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
        from vllm_ascend.worker.v2.sample.min_p import apply_min_p

        apply_min_p(logits, ctx.expanded_idx_mapping, min_p)
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
        return torch.empty_like(logits, dtype=torch.float32).copy_(logits)

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
