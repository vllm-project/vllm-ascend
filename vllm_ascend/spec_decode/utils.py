# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import vllm.distributed.parallel_state as _ps  # type: ignore[import-not-found]
from vllm.config import CompilationMode
from vllm.forward_context import get_forward_context
from vllm_ascend.spec_decode.dynamic import (
    HardwareAwarePrefixPolicy,
    HardwareCostModel,
    SequentialTemperatureScaler,
)


logger = logging.getLogger(__name__)


def update_num_computed_tokens_for_batch_change(
    num_computed_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    prev_positions: torch.Tensor,
    valid_sampled_token_count: torch.Tensor,
    prev_num_draft_tokens: torch.Tensor,
    cpu_num_computed_tokens: torch.Tensor,
) -> None:
    """Correct num_computed_tokens for async spec decode drift.

    Requests that had drafts: corrected = prev_gpu + valid_count.
    New requests or non-draft (e.g. prefills): use CPU value directly.
    """
    # Clamp because prev_positions can be -1 for new requests
    gather_indices = prev_positions.clamp(min=0)

    valid_counts = valid_sampled_token_count[gather_indices]
    prev_computed = num_computed_tokens[gather_indices]
    prev_drafts = prev_num_draft_tokens[gather_indices]

    participating = (prev_positions >= 0) & (prev_drafts > 0)
    corrected = prev_computed + valid_counts.int()

    n = prev_positions.shape[0]
    num_computed_tokens[:n].copy_(torch.where(participating, corrected, cpu_num_computed_tokens))
    num_accepted_tokens.copy_(torch.where(participating, valid_counts, num_accepted_tokens))


def correct_optimistic_seq_lens_cpu(
    optimistic_seq_lens_cpu_np: np.ndarray,
    prev_positions_np: np.ndarray,
    prev_num_draft_tokens_np: np.ndarray,
    valid_sampled_token_count_np: np.ndarray,
    num_reqs: int,
) -> None:
    """Correct ``optimistic_seq_lens_cpu`` for async spec decode drift.

    The scheduler optimistically advances ``num_computed_tokens_cpu`` by the
    full number of tokens scheduled in the previous step (``prev_drafts + 1``
    per spec-decode request), assuming all drafts were accepted. The actual
    number of valid sampled tokens is ``valid_count = 1 + accepted_drafts``.
    The drift, equal to the number of rejected tokens, is therefore::

        rejected = prev_drafts + 1 - valid_count

    Subtracting this from the optimistic seq_lens recovers the true seq_lens
    that ``self.seq_lens`` (GPU) carries for participating requests, without
    touching the device. New requests (``prev_positions < 0``) and prefills
    (``prev_drafts == 0``) need no correction.

    Mirrors ``update_num_computed_tokens_for_batch_change`` on the CPU side.

    All arrays are sliced to ``num_reqs``; ``optimistic_seq_lens_cpu_np`` is
    modified in place.
    """
    prev_positions = prev_positions_np[:num_reqs]
    # Clamp negative entries (new requests) to 0; the participating mask zeroes
    # out their correction so the gathered values are don't-care.
    gather_indices = np.maximum(prev_positions, 0)
    prev_drafts = prev_num_draft_tokens_np[gather_indices]
    valid_counts = valid_sampled_token_count_np[gather_indices]

    participating = (prev_positions >= 0) & (prev_drafts > 0)
    # rejected_for_participating == correction; non-participating reqs end up
    # at zero via the mask multiply.
    correction = (prev_drafts + 1 - valid_counts) * participating
    optimistic_seq_lens_cpu_np[:num_reqs] -= correction.astype(optimistic_seq_lens_cpu_np.dtype, copy=False)


class SlidingWindowAdapter:
    """
    Sliding-window draft attention for the draft model (EAGLE3 / DFlash / DSpark).
    Caps the draft model's attention to the most recent ``window_size`` (W) tokens
    by (a) cropping its block table to the window's blocks and (b) keeping every
    KV-length tensor the FIA kernel can read (notably ``_seq_lens_cpu`` for EAGLE3,
    GPU ``seq_lens`` for DFlash/DSpark ``parallel_drafting``) capped at W.
    Slot-mapping is untouched and still addresses the full, absolute KV cache via
    :attr:`full_block_table`.

    ``future_offset`` is the number of tokens beyond ``seq_lens`` (at :meth:`apply`
    time) that the window end must cover:
      * EAGLE3 passes ``num_speculative_tokens`` — its ``seq_lens`` is context-only
        and the K draft positions lie beyond it, so ``final = seq_lens + K``.
      * DFlash / DSpark pass ``0`` — ``set_inputs_first_pass`` already bakes the
        query stretch (bonus + mask) into ``seq_lens``, so ``final = seq_lens``.
    """

    def __init__(
        self,
        window_size: int,
        block_size: int,
        max_num_reqs: int,
        future_offset: int,
        device: torch.device,
    ) -> None:
        self.window_size: int = window_size
        self.block_size: int = block_size
        self.window_blocks = (window_size + block_size - 1) // block_size
        self.max_window_blocks = self.window_blocks + 1
        self._future_offset: int = future_offset
        self._block_table_clone = torch.zeros(
            (max_num_reqs, self.max_window_blocks),
            dtype=torch.int32,
            device=device,
        )

    def compute_sliding_window_block_table(
        self,
        common_attn_metadata,
        out: torch.Tensor,
    ) -> None:
        k_future = self._future_offset
        w = self.window_size
        b = self.block_size
        num_reqs = common_attn_metadata.seq_lens.shape[0]
        full_cols = self.full_block_table.shape[1]

        # Window math on the (NPU) seq_lens. Pure arithmetic -> stays on NPU.
        self.start_tokens_in_window_rounding = ((common_attn_metadata.seq_lens + k_future - w).clamp(min=0) // b) * b
        self._windowed_seq_lens = common_attn_metadata.seq_lens - self.start_tokens_in_window_rounding
        start_block_indices = self.start_tokens_in_window_rounding // b

        # column offset grid [1, max_window_blocks]
        cols = torch.arange(self.max_window_blocks, device=self.full_block_table.device).unsqueeze(0)
        # source column per (row, col): start_block_indices[:, None] + cols
        src_cols = start_block_indices.unsqueeze(1) + cols
        # clamp to the valid full-block-table column range so gather never goes OOB
        src_cols_clamped = src_cols.clamp(max=full_cols - 1)

        gathered = torch.gather(self.full_block_table, 1, src_cols_clamped)

        needed = torch.clamp((self._windowed_seq_lens + b - 1) // b, max=self.max_window_blocks).unsqueeze(1)
        valid_mask = (cols < needed) & (src_cols < full_cols)
        out[:num_reqs].copy_(gathered * valid_mask.to(gathered.dtype))

    def apply(
        self,
        common_attn_metadata,
    ) -> None:
        self.full_block_table = common_attn_metadata.block_table_tensor
        num_reqs = common_attn_metadata.seq_lens.shape[0]
        k_future = self._future_offset
        w = self.window_size
        b = self.block_size

        self.compute_sliding_window_block_table(common_attn_metadata, self._block_table_clone)
        common_attn_metadata.block_table_tensor = self._block_table_clone[:num_reqs]

        # update NPU seq_lens: reuse the value computed in compute().
        common_attn_metadata.seq_lens = self._windowed_seq_lens

        for name in ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound"):
            src = getattr(common_attn_metadata, name, None)
            if src is not None:
                _windowed = src - ((src + k_future - w).clamp(min=0) // b) * b
                setattr(common_attn_metadata, name, _windowed)


@contextmanager
def patch_tensor_parallel_group(tp_group):
    """Temporarily swap the global TP group for draft-model spec decode.

    vllm-ascend local implementation for swapping the global TP group so the
    draft model can run with a TP degree that differs from the target model.
    """
    old_tp_group = _ps.get_tp_group()
    _ps._TP_STATE_PATCHED = True
    _ps._TP = tp_group
    try:
        yield
    finally:
        _ps._TP_STATE_PATCHED = False
        _ps._TP = old_tp_group


# TODO: Remove it when the bug of fx-graph is solved
# patch vllm_config to be in CompilationMode.NONE temporarily
@contextmanager
def _maybe_eager_context(vllm_config):
    target_compilation_config = vllm_config.compilation_config
    draft_compilation_config = replace(
        target_compilation_config,
        mode=CompilationMode.NONE,
    )
    # Model layers use these registries even when compilation is disabled.
    draft_compilation_config.static_forward_context = target_compilation_config.static_forward_context
    draft_compilation_config.static_all_moe_layers = target_compilation_config.static_all_moe_layers
    vllm_config.compilation_config = draft_compilation_config
    try:
        yield
    finally:
        vllm_config.compilation_config = target_compilation_config


class DynamicSpecScheduler:
    """Dynamic verification scheduler shared by DFlash and DSpark.

    Both DFlash and DSpark use the same scheduling algorithm:

       method-specific confidence
           -> token acceptance probabilities [B, D]
           -> cumulative survival probabilities [B, D]
           -> shared verify budget
           -> per-request verify lengths [B]

    The only method-specific part is how token acceptance probabilities are
    estimated:

    * DFlash:
       probability of the argmax draft token.

    * DSpark:
       sigmoid output of the confidence head.

    Everything after token-probability estimation is identical.
    """

    def __init__(
        self,
        *,
        method: str,
        policy: str = "confidence_budget",
        method_params: dict[str, Any],
        max_batch_size: int,
        num_speculative_tokens: int,
        device: torch.device,
    ) -> None:
        if method not in ("dflash", "dspark"):
            raise ValueError(f"Unsupported dynamic speculative method: {method}")

        self.method = method
        self.policy_name = policy or method_params.get("policy", "confidence_budget")
        if self.policy_name not in ("confidence_budget", "hardware_aware"):
            raise ValueError(f"Unsupported dynamic speculative policy: {self.policy_name}")

        self.max_batch_size = max_batch_size
        self.num_speculative_tokens = num_speculative_tokens
        self.device = device
        self._method_params = dict(method_params)

        # PR #47808 smooths request confidence with an EMA.  The alias keeps
        # the upstream terminology available while ``ema_alpha`` remains
        # convenient for Ascend deployments.  Legacy confidence-budget mode
        # stays unchanged unless the user explicitly enables EMA there.
        default_ema_alpha = 0.8 if self.policy_name == "hardware_aware" else 0.0
        self.ema_alpha = float(
            method_params.get(
                "adaptive_verification_ema_alpha",
                method_params.get("ema_alpha", default_ema_alpha),
            )
        )
        if not 0.0 <= self.ema_alpha <= 1.0:
            raise ValueError("adaptive_verification_ema_alpha must be in [0, 1]")
        self.auto_profile_enabled = bool(
            method_params.get(
                "auto_profile",
                method_params.get("startup_profile", False),
            )
        )

        # Shared configuration

        self.initial_verify_budget_per_req = int(
            method_params.get(
                "initial_verify_budget_per_req",
                5,
            )
        )

        self.budget_update_interval = int(
            method_params.get(
                "budget_update_interval",
                16,
            )
        )

        self.confidence_update_interval = int(
            method_params.get(
                "confidence_update_interval",
                1,
            )
        )
        if self.confidence_update_interval <= 0:
            raise ValueError("confidence_update_interval must be > 0")

        self.budget_threshold = float(
            method_params.get(
                "budget_threshold",
                0.3,
            )
        )

        configured_min_k = method_params.get("min_verify_tokens")
        self.min_k = int(
            configured_min_k
            if configured_min_k is not None
            else (0 if self.policy_name == "hardware_aware" else 1)
        )
        if self.min_k < 0 or self.min_k > self.num_speculative_tokens:
            raise ValueError(
                "min_verify_tokens must be between 0 and num_speculative_tokens"
            )

        # Request-to-prefix mapping is lowered to device operators on NPU.
        # Keep an explicit escape hatch for bring-up and A/B comparisons; the
        # default is enabled only for the hardware-aware policy.
        self.device_allocation_enabled = bool(
            method_params.get(
                "device_allocation_enabled",
                self.policy_name == "hardware_aware",
            )
        )

        # Confidence calibration is a no-op unless temperatures are supplied
        # explicitly or in the hardware profile.
        self.cost_model: HardwareCostModel | None = None
        self.hardware_policy: HardwareAwarePrefixPolicy | None = None
        profile_temperatures: tuple[float, ...] = ()
        self._configured_temperatures = method_params.get("confidence_temperatures")
        if self.policy_name == "hardware_aware":
            try:
                profile = method_params.get("profile")
                profile_path = method_params.get("profile_path")
                if profile is not None:
                    self.cost_model = HardwareCostModel.from_dict(
                        profile,
                        expected_fingerprint=method_params.get("profile_fingerprint"),
                        strict_fingerprint=bool(method_params.get("strict_profile_fingerprint", True)),
                    )
                elif profile_path:
                    self.cost_model = HardwareCostModel.from_json(
                        profile_path,
                        expected_fingerprint=method_params.get("profile_fingerprint"),
                        strict_fingerprint=bool(method_params.get("strict_profile_fingerprint", True)),
                    )
                elif not self.auto_profile_enabled:
                    raise ValueError("profile_path or inline profile is required")
                if self.cost_model is not None:
                    self._install_hardware_policy(self.cost_model)
                    profile_temperatures = self.cost_model.confidence_temperatures
            except (OSError, TypeError, ValueError) as exc:
                if self.auto_profile_enabled and not method_params.get("profile") and not method_params.get("profile_path"):
                    # The runner will collect a real profile after model/KV
                    # initialization. Keep the policy selected so the profile
                    # can be installed without reconstructing the proposer.
                    logger.info(
                        "Hardware-aware dynamic speculative scheduling is pending "
                        "startup profiling: %s",
                        exc,
                    )
                else:
                    self._fallback_to_confidence_budget(exc)

        configured_temperatures = self._configured_temperatures
        if configured_temperatures is None and profile_temperatures:
            configured_temperatures = profile_temperatures
        self.calibrator = SequentialTemperatureScaler.from_config(
            configured_temperatures,
            self.num_speculative_tokens,
        )

        self.budget_k = max(
            self.min_k,
            min(
                self.initial_verify_budget_per_req,
                self.num_speculative_tokens,
            ),
        )

        # A stale hardware profile must not reduce the proposal budget far
        # below the confidence-budget policy that is already known to work.
        # The floor is expressed as a ratio so it follows the confidence
        # scheduler when its budget is updated. Set to 0 to disable it after
        # a workload-specific profile has been validated.
        self.hardware_min_budget_ratio = float(
            method_params.get("hardware_min_budget_ratio", 0.8)
        )
        if not 0.0 <= self.hardware_min_budget_ratio <= 1.0:
            raise ValueError("hardware_min_budget_ratio must be in [0, 1]")

        # Hybrid hardware-aware scheduling keeps the cheap full-width path
        # for small/high-acceptance batches, and only pays for confidence plus
        # profile allocation when a larger or low-acceptance batch can benefit
        # from a shorter logical K. It is opt-in so existing deployments keep
        # the exact hardware-aware behavior unless explicitly enabled.
        self.hybrid_policy_enabled = bool(
            method_params.get("hybrid_policy_enabled", False)
        )
        self.hybrid_min_batch_size = int(
            method_params.get("hybrid_min_batch_size", 8)
        )
        if self.hybrid_min_batch_size <= 0:
            raise ValueError("hybrid_min_batch_size must be > 0")
        self.hybrid_acceptance_threshold = float(
            method_params.get("hybrid_acceptance_threshold", 0.6)
        )
        if not 0.0 <= self.hybrid_acceptance_threshold <= 1.0:
            raise ValueError("hybrid_acceptance_threshold must be in [0, 1]")
        self.hybrid_probe_interval = int(
            method_params.get("hybrid_probe_interval", 16)
        )
        if self.hybrid_probe_interval <= 0:
            raise ValueError("hybrid_probe_interval must be > 0")
        self.hybrid_full_width_goodput_margin = float(
            method_params.get("hybrid_full_width_goodput_margin", 0.0)
        )
        if self.hybrid_full_width_goodput_margin < 0.0:
            raise ValueError("hybrid_full_width_goodput_margin must be >= 0")

        # V2 Ascend FULL graphs have one physical query width for the whole
        # batch.  The hardware policy still computes per-request logical
        # prefixes, but a mixed result cannot be represented by the current
        # FIA metadata.  Collapse the result to one batch width at the output
        # boundary so the next scheduled batch can use a smaller captured
        # graph instead of silently falling back to max-K.  The percentile is
        # configurable because it trades off tail-request acceptance against
        # draft-model work.
        self.v2_uniform_batch_k = bool(
            method_params.get(
                "v2_uniform_batch_k",
                method_params.get("v2_varlen_physical_k", False),
            )
        )
        self.v2_batch_k_percentile = float(
            method_params.get("v2_batch_k_percentile", 50.0)
        )
        if not 0.0 <= self.v2_batch_k_percentile <= 100.0:
            raise ValueError("v2_batch_k_percentile must be in [0, 100]")
        self._last_v2_batch_physical_k: int | None = None
        self._last_v2_proposal_lengths: list[int] | None = None
        self._last_v2_request_ids: tuple[Any, ...] | None = None
        self._v2_result_generation = 0
        self._last_v2_published_generation = -1

        self._steps_since_budget_update = 0

        self._hybrid_last_acceptance: float | None = None
        self._hybrid_last_num_reqs: int | None = None
        self._hybrid_last_num_draft_tokens: int | None = None
        self._hybrid_steps_since_probe = 0
        self._hybrid_full_width_active = False
        self._hybrid_last_dynamic_goodput: float | None = None
        self._hybrid_last_full_width_goodput: float | None = None

        # Shared buffers

        # Conditional acceptance probability for every proposed token.
        # token_probs[b, i] ~= P(token_i accepted | prefix accepted)
        # Shape: [B, D]
        self._token_probs_buffer = torch.empty(
            (
                self.max_batch_size,
                self.num_speculative_tokens,
            ),
            dtype=torch.float32,
            device=device,
        )

        # The EMA state is kept separate from the current confidence buffer so
        # the current batch can be reordered by request id without mixing rows.
        self._ema_token_probs_buffer = torch.empty_like(self._token_probs_buffer)
        self._ema_request_to_row: dict[Any, int] = {}
        self._ema_previous_num_reqs: int | None = None
        self._ema_previous_num_draft_tokens: int | None = None

        # Cumulative survival probability.
        # survival[b, i] = prod(token_probs[b, :i + 1])
        # Shape: [B, D]
        self._survival_buffer = torch.empty(
            (
                self.max_batch_size,
                self.num_speculative_tokens,
            ),
            dtype=torch.float32,
            device=device,
        )

        # Final verification length selected for each request.
        # Shape: [B]
        self._num_verify_tokens_buffer = torch.empty(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
        )

        # Reused scatter_add source.
        self._scatter_ones_buffer = torch.ones(
            self.max_batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=device,
        )

        # Latest result consumed by the model runner.
        self.num_verify_tokens: torch.Tensor | None = None
        self.reused_last_result = False
        self._cached_num_verify_tokens: torch.Tensor | None = None
        self._confidence_steps = 0
        self._last_confidence_num_reqs: int | None = None
        self._last_confidence_num_draft_tokens: int | None = None
        self._last_confidence_request_ids: tuple[Any, ...] | None = None

    def update_from_token_probs(
        self,
        token_probs: torch.Tensor,
        *,
        request_ids: Sequence[Any] | None = None,
    ) -> torch.Tensor:
        """Update the policy from confidence probabilities already computed.

        V2 DSpark computes the confidence head inside the upstream speculator.
        Recomputing that head in :meth:`update` would duplicate the most
        expensive part of the decision path, so V2 publishes the existing
        probabilities through this small adapter.
        """

        num_reqs, num_draft_tokens = token_probs.shape
        current_request_ids = (
            tuple(request_ids) if request_ids is not None else None
        )

        # The V2 DSpark confidence head is part of the captured draft graph,
        # so it still runs every replay.  The expensive Python/device policy
        # work does not need to run at the same cadence: reuse the previous
        # prefix while the batch identity and physical width are unchanged.
        # A request reorder, shape change, or cadence expiry forces an
        # immediate refresh so cached lengths cannot be applied to a new row.
        if (
            self.hardware_policy is not None
            and self.confidence_update_interval > 1
            and self._cached_num_verify_tokens is not None
            and self._last_confidence_num_reqs == int(num_reqs)
            and self._last_confidence_num_draft_tokens == int(num_draft_tokens)
            and self._last_confidence_request_ids == current_request_ids
        ):
            self._confidence_steps += 1
            if self._confidence_steps < self.confidence_update_interval:
                self.num_verify_tokens = self._cached_num_verify_tokens[:num_reqs]
                self.reused_last_result = True
                return self.num_verify_tokens

        self._confidence_steps = 0
        self.reused_last_result = False
        result = self._finish_token_probs_update(
            token_probs,
            request_ids=request_ids,
        )
        self._v2_result_generation += 1
        if self.hardware_policy is not None and self.confidence_update_interval > 1:
            self._cached_num_verify_tokens = result
            self._last_confidence_num_reqs = int(num_reqs)
            self._last_confidence_num_draft_tokens = int(num_draft_tokens)
            self._last_confidence_request_ids = current_request_ids
        return result

    def proposal_lengths_for_v2(
        self,
        request_ids: Sequence[Any],
        *,
        max_k: int,
    ) -> list[int] | None:
        """Return the logical K that V2 should schedule on the next step.

        A V2 FULL Ascend graph currently has a single physical K.  When
        requested, use a robust batch percentile of the hardware policy's
        per-request result.  This preserves dynamic K while avoiding the
        mixed-batch fallback in ``physical_k_scope``.
        """

        if self.num_verify_tokens is None:
            return None
        current_request_ids = tuple(request_ids)
        if (
            self._last_v2_proposal_lengths is not None
            and self._last_v2_request_ids == current_request_ids
            and self._last_v2_published_generation == self._v2_result_generation
        ):
            return self._last_v2_proposal_lengths
        values = self.num_verify_tokens.detach().to("cpu").tolist()
        lengths = [
            max(0, min(int(values[idx]), max_k))
            for idx in range(min(len(request_ids), len(values)))
        ]
        if len(lengths) != len(request_ids):
            return None
        if not lengths:
            return []

        if self.v2_uniform_batch_k:
            ordered = sorted(lengths)
            rank = int(round((len(ordered) - 1) * self.v2_batch_k_percentile / 100.0))
            batch_k = max(1, min(max_k, ordered[rank]))
            lengths = [batch_k] * len(lengths)
            self._last_v2_batch_physical_k = batch_k
        else:
            self._last_v2_batch_physical_k = max(lengths, default=0)
        self._last_v2_proposal_lengths = lengths
        self._last_v2_request_ids = current_request_ids
        self._last_v2_published_generation = self._v2_result_generation
        return lengths

    def _install_hardware_policy(self, cost_model: HardwareCostModel) -> None:
        """Install or replace the profile-backed prefix policy."""

        self.cost_model = cost_model
        self.hardware_policy = HardwareAwarePrefixPolicy(
            cost_model=cost_model,
            min_k=self.min_k,
            max_batch_size=self.max_batch_size,
            max_draft_tokens=self.num_speculative_tokens,
            device=self.device,
            decision_interval=self._hardware_decision_interval(self._method_params),
            allocation_interval=self._hardware_allocation_interval(self._method_params),
            device_allocation_enabled=self.device_allocation_enabled,
        )
        if hasattr(self, "_cached_num_verify_tokens"):
            self._cached_num_verify_tokens = None
            self._last_confidence_num_reqs = None
            self._last_confidence_num_draft_tokens = None
            self._last_confidence_request_ids = None

    def set_hardware_profile(
        self,
        profile: Mapping[str, Any] | HardwareCostModel,
        *,
        source: str = "startup",
    ) -> None:
        """Install a freshly collected or externally supplied hardware profile."""

        if self.policy_name != "hardware_aware":
            raise RuntimeError("cannot install a hardware profile for confidence_budget policy")
        if isinstance(profile, HardwareCostModel):
            cost_model = profile
        else:
            cost_model = HardwareCostModel.from_dict(
                profile,
                expected_fingerprint=self._method_params.get("profile_fingerprint"),
                strict_fingerprint=bool(
                    self._method_params.get("strict_profile_fingerprint", True)
                ),
                source=source,
            )
        self._install_hardware_policy(cost_model)
        if self._configured_temperatures is None and cost_model.confidence_temperatures:
            self.calibrator = SequentialTemperatureScaler.from_config(
                cost_model.confidence_temperatures,
                self.num_speculative_tokens,
            )
        logger.info(
            "Installed hardware-aware dynamic speculative profile source=%s shapes=%d",
            cost_model.source,
            len(cost_model.latency_ms),
        )

    def fallback_to_confidence_budget(self, reason: Exception | str) -> None:
        """Disable hardware allocation while keeping dynamic confidence scheduling."""

        self._fallback_to_confidence_budget(reason)

    def _fallback_to_confidence_budget(self, reason: Exception | str) -> None:
        logger.warning(
            "Unable to enable hardware-aware dynamic speculative scheduling; "
            "falling back to confidence_budget: %s",
            reason,
        )
        self.policy_name = "confidence_budget"
        self.cost_model = None
        self.hardware_policy = None
        if self._method_params.get("min_verify_tokens") is None:
            self.min_k = 1
        self.budget_k = max(self.min_k, min(self.initial_verify_budget_per_req, self.num_speculative_tokens))

    @staticmethod
    def _hardware_decision_interval(method_params: dict[str, Any]) -> int:
        """Return a safe recomputation interval for hardware-aware policy.

        Recomputing the hardware allocation performs a device-side sort and
        transfers the winning candidate index back to Python.  Doing that on
        every decode step (``decision_interval=1``) makes the scheduler
        host-bound for small batches.  Keep the interval configurable, but
        protect the hot path with a conservative minimum.  Users that have a
        workload-specific calibration can explicitly lower
        ``min_decision_interval``.
        """
        configured = int(method_params.get("decision_interval", 16))
        minimum = int(method_params.get("min_decision_interval", 8))
        if configured <= 0 or minimum <= 0:
            raise ValueError(
                "decision_interval and min_decision_interval must be > 0"
            )
        interval = max(configured, minimum)
        if interval != configured:
            logger.info(
                "Clamping hardware-aware decision_interval from %d to %d "
                "to avoid per-step scheduler synchronization",
                configured,
                interval,
            )
        return interval

    @staticmethod
    def _hardware_allocation_interval(method_params: dict[str, Any]) -> int:
        """Return the cadence for remapping survival scores to request prefixes.

        The hardware optimum is intentionally searched at ``decision_interval``
        cadence, but the request-level top-k/scatter mapping can be held for a
        shorter, independently tuned interval.  Keep the default at one so
        existing deployments retain their exact per-step allocation semantics.
        """
        configured = int(method_params.get("allocation_interval", 1))
        if configured <= 0:
            raise ValueError("allocation_interval must be > 0")
        return configured

    def _hybrid_should_hold_full_width(
        self,
        *,
        num_reqs: int,
        physical_k: int,
    ) -> bool:
        """Decide whether to use the low-overhead full-width branch.

        Small batches do not amortize the confidence/policy host overhead.
        For larger batches, use the previous confidence estimate as an
        acceptance signal and periodically probe the dynamic path so the
        decision can recover when acceptance changes.
        """
        if not self.hybrid_policy_enabled or physical_k > self.budget_k:
            return False
        if num_reqs < self.hybrid_min_batch_size:
            return True
        if self._hybrid_last_acceptance is None:
            return False
        if (
            self._hybrid_last_num_reqs != num_reqs
            or self._hybrid_last_num_draft_tokens != physical_k
        ):
            return False
        if self._hybrid_steps_since_probe >= self.hybrid_probe_interval:
            return False
        if self._hybrid_last_acceptance < self.hybrid_acceptance_threshold:
            return False
        if (
            self._hybrid_last_dynamic_goodput is not None
            and self._hybrid_last_full_width_goodput is not None
            and self._hybrid_last_full_width_goodput
            * (1.0 + self.hybrid_full_width_goodput_margin)
            + 1e-6 * max(1.0, self._hybrid_last_dynamic_goodput)
            < self._hybrid_last_dynamic_goodput
        ):
            return False
        return True

    def _finish_token_probs_update(
        self,
        token_probs: torch.Tensor,
        *,
        request_ids: Sequence[Any] | None = None,
    ) -> torch.Tensor:
        """Apply the policy to probabilities produced by any DSpark path."""

        num_reqs, num_draft_tokens = token_probs.shape
        if num_reqs == 0 or num_draft_tokens == 0:
            self.num_verify_tokens = self._num_verify_tokens_buffer[:num_reqs]
            self.num_verify_tokens.zero_()
            return self.num_verify_tokens

        # Keep the same low-overhead hybrid/full-width semantics as the normal
        # update path.  V2 has already paid for the confidence head by the
        # time this method is called, but it can still skip sorting/allocation.
        if self.hardware_policy is not None:
            physical_k = min(num_draft_tokens, self.num_speculative_tokens)
            use_hybrid_fast_path = self._hybrid_should_hold_full_width(
                num_reqs=int(num_reqs),
                physical_k=physical_k,
            )
            use_legacy_saturated_fast_path = (
                not self.hybrid_policy_enabled
                and self.hardware_min_budget_ratio >= 1.0
                and physical_k <= self.budget_k
            )
            if use_hybrid_fast_path or use_legacy_saturated_fast_path:
                same_shape = (
                    self._hybrid_full_width_active
                    and self._hybrid_last_num_reqs == int(num_reqs)
                    and self._hybrid_last_num_draft_tokens == physical_k
                )
                self.num_verify_tokens = self._num_verify_tokens_buffer[:num_reqs]
                self.num_verify_tokens.fill_(physical_k)
                self.reused_last_result = same_shape
                self._hybrid_last_num_reqs = int(num_reqs)
                self._hybrid_last_num_draft_tokens = physical_k
                self._hybrid_full_width_active = True
                if int(num_reqs) >= self.hybrid_min_batch_size:
                    self._hybrid_steps_since_probe += 1
                return self.num_verify_tokens

        result = self._update_from_token_probs(
            token_probs,
            request_ids=request_ids,
        )
        if self.hybrid_policy_enabled and self.hardware_policy is not None:
            # The full-prefix survival is a cheap batch-level acceptance
            # signal used by the next cadence decision.
            self._hybrid_last_acceptance = float(
                self._survival_buffer[:num_reqs, num_draft_tokens - 1].mean().item()
            )
            self._hybrid_last_dynamic_goodput = self.hardware_policy.last_goodput
            self._hybrid_last_full_width_goodput = self.hardware_policy.full_width_goodput(
                self._survival_buffer[:num_reqs, :num_draft_tokens]
            )
            self._hybrid_last_num_reqs = int(num_reqs)
            self._hybrid_last_num_draft_tokens = int(num_draft_tokens)
            self._hybrid_steps_since_probe = 0
            self._hybrid_full_width_active = False
        if (
            self.hardware_policy is not None
            and self.confidence_update_interval > 1
        ):
            self._cached_num_verify_tokens = result
            self._last_confidence_num_reqs = int(num_reqs)
            self._last_confidence_num_draft_tokens = int(num_draft_tokens)
        return result

    def _apply_confidence_ema(
        self,
        token_probs: torch.Tensor,
        request_ids: Sequence[Any] | None = None,
    ) -> torch.Tensor:
        """Smooth conditional acceptance probabilities before cumprod.

        Request IDs are preferred because vLLM can reorder a running batch.
        When the caller does not provide IDs, positional smoothing is used as
        a compatibility fallback and is reset whenever the batch shape changes.
        The operation runs only when a fresh confidence result is computed;
        cached confidence steps do not pay this cost.
        """

        if self.ema_alpha <= 0.0 or token_probs.numel() == 0:
            return token_probs

        num_reqs, num_draft_tokens = token_probs.shape
        alpha = self.ema_alpha
        if request_ids is None:
            if self._ema_previous_num_draft_tokens == num_draft_tokens:
                previous = self._ema_token_probs_buffer[:num_reqs, :num_draft_tokens].clone()
                token_probs.mul_(1.0 - alpha).add_(previous, alpha=alpha)
            self._ema_token_probs_buffer[:num_reqs, :num_draft_tokens].copy_(token_probs)
            self._ema_previous_num_reqs = num_reqs
            self._ema_previous_num_draft_tokens = num_draft_tokens
            return token_probs

        ids = list(request_ids)
        if len(ids) != num_reqs:
            raise ValueError(
                "request_ids must have one entry per dynamic speculative request"
            )
        try:
            if len(set(ids)) != len(ids):
                raise ValueError("request_ids must be unique within a batch")
        except TypeError as exc:
            raise TypeError("request_ids must contain hashable values") from exc

        previous_state = self._ema_token_probs_buffer[
            : (self._ema_previous_num_reqs or 0), :num_draft_tokens
        ].clone()
        previous_width = self._ema_previous_num_draft_tokens
        previous_rows = self._ema_request_to_row
        for row, request_id in enumerate(ids):
            previous_row = previous_rows.get(request_id)
            if previous_width == num_draft_tokens and previous_row is not None:
                token_probs[row].mul_(1.0 - alpha).add_(
                    previous_state[previous_row],
                    alpha=alpha,
                )
            self._ema_token_probs_buffer[row, :num_draft_tokens].copy_(token_probs[row])
        self._ema_request_to_row = {request_id: row for row, request_id in enumerate(ids)}
        self._ema_previous_num_reqs = num_reqs
        self._ema_previous_num_draft_tokens = num_draft_tokens
        return token_probs

    def update(
        self,
        *,
        logits: torch.Tensor | None = None,
        model=None,
        last_hidden_states: torch.Tensor | None = None,
        draft_token_ids: torch.Tensor | None = None,
        num_reqs: int | None = None,
        request_ids: Sequence[Any] | None = None,
    ) -> torch.Tensor:
        self.reused_last_result = False

        # A full hardware budget does not need confidence estimation.  Keep
        # this check in the regular path for V1; V2 calls the same policy
        # through ``update_from_token_probs`` after its confidence head.
        if (
            self.hardware_policy is not None
            and num_reqs is not None
            and draft_token_ids is not None
        ):
            physical_k = max(int(draft_token_ids.shape[1]) - 1, 0)
            if self._hybrid_should_hold_full_width(
                num_reqs=int(num_reqs), physical_k=physical_k
            ) or (
                not self.hybrid_policy_enabled
                and self.hardware_min_budget_ratio >= 1.0
                and physical_k <= self.budget_k
            ):
                self.num_verify_tokens = self._num_verify_tokens_buffer[:num_reqs]
                self.num_verify_tokens.fill_(physical_k)
                self.reused_last_result = True
                self._hybrid_full_width_active = True
                return self.num_verify_tokens

        self._hybrid_full_width_active = False

        # The confidence head and the cumulative-prefix policy are device
        # work.  For a hardware-aware profile, holding the last safe prefix
        # for a short cadence avoids repeating that work when the physical
        # draft width and batch shape are unchanged.  A width/shape change
        # always refreshes immediately so cached lengths cannot exceed the
        # current draft tensor.
        num_draft_tokens = None
        if draft_token_ids is not None:
            num_draft_tokens = max(int(draft_token_ids.shape[1]) - 1, 0)
        elif logits is not None and num_reqs:
            num_draft_tokens = int(logits.shape[0]) // int(num_reqs)

        if (
            self.hardware_policy is not None
            and self.confidence_update_interval > 1
            and self._cached_num_verify_tokens is not None
            and num_reqs is not None
            and num_draft_tokens is not None
            and self._last_confidence_num_reqs == int(num_reqs)
            and self._last_confidence_num_draft_tokens == num_draft_tokens
        ):
            self._confidence_steps += 1
            if self._confidence_steps < self.confidence_update_interval:
                self.num_verify_tokens = self._cached_num_verify_tokens[:num_reqs]
                self.reused_last_result = True
                return self.num_verify_tokens

        if self.hardware_policy is not None and self.confidence_update_interval > 1:
            self._confidence_steps = 0

        if self.method == "dflash":
            if logits is None:
                raise ValueError("DFlash requires logits.")

            token_probs = self._compute_dflash_token_probs(
                logits,
                num_reqs=num_reqs,
            )
        elif self.method == "dspark":
            if num_reqs is None:
                raise ValueError("DSpark requires num_reqs.")

            token_probs = self._compute_dspark_token_probs(
                model,
                last_hidden_states,
                draft_token_ids,
                num_reqs,
            )
        else:
            raise RuntimeError(f"Unsupported dynamic speculative method: {self.method}")

        return self._finish_token_probs_update(token_probs, request_ids=request_ids)

    def _compute_dflash_token_probs(
        self,
        logits: torch.Tensor,
        num_reqs: int | None = None,
    ) -> torch.Tensor:
        """Estimate DFlash token acceptance probabilities.

        DFlash has no confidence head, so the softmax probability of the
        argmax draft token is used as the acceptance-confidence proxy.

        Input:
            logits: [B * D, V]

        Output:
            token_probs: [B, D]
        """
        num_rows = logits.shape[0]
        if num_reqs is None:
            num_draft_tokens = self.num_speculative_tokens
            num_reqs = num_rows // max(num_draft_tokens, 1)
        else:
            num_reqs = int(num_reqs)
            num_draft_tokens = num_rows // max(num_reqs, 1)
        if num_reqs <= 0 or num_draft_tokens <= 0:
            return self._token_probs_buffer[:0, :0]

        token_probs = self._token_probs_buffer[:num_reqs, :num_draft_tokens]
        # max(softmax(logits)) per row; PyTorch keeps this ACLGraph-safe.
        raw_probs = torch.softmax(logits.float(), dim=-1).max(dim=-1).values.view(
            num_reqs,
            num_draft_tokens,
        )
        token_probs.copy_(self.calibrator.calibrate_probabilities(raw_probs))
        token_probs.clamp_(
            min=1e-6,
            max=1.0,
        )

        return token_probs

    def _compute_dspark_token_probs(
        self,
        model,
        last_hidden_states: torch.Tensor,
        draft_token_ids: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        """Estimate DSpark token acceptance probabilities.

        ``compute_confidence`` already returns per-position acceptance
        probabilities (sigmoid of the confidence-head logits).

        Output:
            token_probs: [B, D]
        """
        num_draft_tokens = max(int(draft_token_ids.shape[1]) - 1, 0)
        num_tokens = num_reqs * num_draft_tokens
        if num_reqs <= 0 or num_draft_tokens <= 0:
            return self._token_probs_buffer[:0, :0]

        flat_hidden = last_hidden_states.reshape(
            num_tokens,
            last_hidden_states.shape[-1],
        )

        # draft_token_ids normally has shape [B, D + 1] for DSpark:
        # [seed, draft_1, ..., draft_D]
        # The confidence prediction for D positions uses the first D
        # Markov inputs.
        markov_embs = model.markov_embed(
            draft_token_ids[
                :num_reqs,
                :num_draft_tokens,
            ]
        )

        flat_markov = markov_embs.reshape(
            num_tokens,
            markov_embs.shape[-1],
        ).to(flat_hidden.dtype)

        confidence = model.compute_confidence(
            flat_hidden,
            flat_markov,
        )

        token_probs = self._token_probs_buffer[:num_reqs, :num_draft_tokens]

        token_probs.copy_(
            self.calibrator.calibrate_probabilities(
                confidence.reshape(
                    num_reqs,
                    num_draft_tokens,
                )
            )
        )
        token_probs.clamp_(
            min=1e-6,
            max=1.0,
        )

        return token_probs

    def _update_from_token_probs(
        self,
        token_probs: torch.Tensor,
        *,
        request_ids: Sequence[Any] | None = None,
    ) -> torch.Tensor:
        """Run the shared dynamic speculative scheduling pipeline."""
        num_reqs, num_draft_tokens = token_probs.shape

        token_probs = self._apply_confidence_ema(token_probs, request_ids)

        survival = self._survival_buffer[:num_reqs, : token_probs.shape[1]]

        # survival[b, i] estimates the probability that request b reaches
        # and accepts the draft prefix through position i.
        torch.cumprod(
            token_probs,
            dim=1,
            out=survival,
        )

        if self.hardware_policy is not None:
            # Keep the confidence budget alive as a cheap safety signal. The
            # update itself is amortized by budget_update_interval and avoids
            # the profile selecting a much smaller K solely because a sparse
            # latency table rounded a candidate to the wrong graph shape.
            self.compute_verify_budget(survival)
            min_total_tokens = math.ceil(
                num_reqs
                * self.budget_k
                * self.hardware_min_budget_ratio
            )
            if min_total_tokens >= num_reqs * num_draft_tokens:
                # The hardware floor already covers every physical draft
                # position. Reuse the scheduler buffer and bypass the
                # hardware policy entirely; its result is necessarily the
                # full width and no profile decision can improve it.
                self.num_verify_tokens = self._num_verify_tokens_buffer[:num_reqs]
                self.num_verify_tokens.fill_(num_draft_tokens)
            else:
                self.num_verify_tokens = self.hardware_policy.allocate(
                    survival,
                    min_total_tokens=min_total_tokens,
                )
        else:
            self.compute_verify_budget(survival)
            self.num_verify_tokens = self.allocate_verify_budget(survival)

        return self.num_verify_tokens

    def compute_verify_budget(
        self,
        survival: torch.Tensor,
    ) -> None:
        """Periodically recompute the shared per-request verify budget."""
        self._steps_since_budget_update += 1

        if self._steps_since_budget_update < self.budget_update_interval:
            return

        self._steps_since_budget_update = 0

        num_reqs = survival.shape[0]

        if num_reqs == 0:
            return

        # Count cumulative-prefix positions whose estimated probability of
        # being reached and accepted exceeds the configured threshold.
        # `.item()` introduces an NPU -> CPU synchronization, but only on
        # budget-update steps.
        mean_k = float((survival >= self.budget_threshold).sum().item()) / float(num_reqs)

        new_budget_k = math.ceil(mean_k)

        # Previously measured on Qwen3-8B on A3:
        # verification costs of adjacent budgets differ only slightly,
        # and the next odd speculative budget may be approximately equal
        # to or cheaper than the previous even one.
        # Example: batch=64 K=6 -> 52.9 K=7 -> 54.3
        # Verification also includes the bonus token, so an odd K gives an
        # even verification width. Current kernels can process these widths
        # more efficiently, potentially due to padding / next_power_of_2().
        if new_budget_k % 2 == 0 and new_budget_k < self.num_speculative_tokens:
            new_budget_k += 1

        self.budget_k = max(
            self.min_k,
            min(
                new_budget_k,
                self.num_speculative_tokens,
            ),
        )

    def allocate_verify_budget(
        self,
        survival: torch.Tensor,
    ) -> torch.Tensor:
        """Distribute the global verification budget across requests.

        Every request receives at least `min_k` tokens.

        The remaining global token budget is assigned to the largest
        cumulative survival probabilities across the whole batch.

        Because cumulative survival is monotonically non-increasing inside
        each request, selecting the globally highest positions naturally
        produces prefix lengths.
        """
        num_reqs, num_draft_tokens = survival.shape

        keep_lens = self._num_verify_tokens_buffer[:num_reqs]

        mandatory = min(self.min_k, num_draft_tokens)
        keep_lens.fill_(mandatory)

        extra_budget_per_req = max(
            self.budget_k - mandatory,
            0,
        )

        # Positions [0:mandatory] have already been guaranteed.
        candidate_window = survival[
            :,
            mandatory:,
        ]

        num_candidates = candidate_window.numel()

        num_budget_tokens = min(
            num_reqs * extra_budget_per_req,
            num_candidates,
        )

        if num_budget_tokens > 0:
            candidate_cols = num_draft_tokens - mandatory

            flat_survival = candidate_window.reshape(-1)

            _, top_indices = torch.topk(
                flat_survival,
                k=num_budget_tokens,
                largest=True,
                sorted=False,
            )

            chosen_requests = torch.div(
                top_indices,
                candidate_cols,
                rounding_mode="floor",
            )

            keep_lens.scatter_add_(
                0,
                chosen_requests,
                self._scatter_ones_buffer[:num_budget_tokens],
            )

        keep_lens.clamp_(
            min=mandatory,
            max=num_draft_tokens,
        )

        return keep_lens
