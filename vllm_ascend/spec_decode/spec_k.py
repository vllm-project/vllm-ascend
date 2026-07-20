# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from vllm_ascend.ascend_config import SpecKConfig


class SpecKPolicy:
    """Convert draft logits into per-token expert budgets."""

    def __init__(
        self,
        config: SpecKConfig,
        base_top_k: int,
        device: torch.device,
    ) -> None:
        if base_top_k <= 0:
            raise ValueError("Spec-K requires a MoE target with a positive top-k.")
        if len(config.ppl_thresholds) >= base_top_k:
            raise ValueError(
                "spec_k_config.ppl_thresholds must contain fewer values than "
                f"the model top-k ({base_top_k}) so at least one expert remains active."
            )
        if base_top_k > torch.iinfo(torch.uint8).max:
            raise ValueError("Spec-K supports model top-k values up to 255.")

        self.base_top_k = base_top_k
        self._apply_last_token = config.apply_last_token

        thresholds = config.ppl_thresholds + (0.0,) * (base_top_k - len(config.ppl_thresholds))
        self._log_perplexity_boundaries = torch.tensor(thresholds[::-1], dtype=torch.float32, device=device).log_()

    def top_ks_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3:
            raise ValueError(f"Spec-K logits must have shape [B, S, V], got {logits.shape}.")

        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        # xlogy defines 0 * log(0) as zero for reduced-vocabulary logits.
        entropy = -torch.xlogy(probabilities, probabilities).sum(dim=-1)
        token_top_ks = (entropy.unsqueeze(-1) >= self._log_perplexity_boundaries).sum(dim=-1)
        token_top_ks = token_top_ks.to(dtype=torch.int32)

        if self._apply_last_token:
            last_top_k = token_top_ks.to(torch.float32).mean(dim=1).to(torch.int32)
        else:
            last_top_k = torch.full(
                (logits.shape[0],),
                self.base_top_k,
                dtype=torch.int32,
                device=logits.device,
            )

        return torch.cat((token_top_ks, last_top_k[:, None]), dim=1).contiguous()


@dataclass(frozen=True, slots=True)
class SpecKHistoryUpdate:
    new_output_length: int
    accepted_top_ks: torch.Tensor


@dataclass(slots=True)
class SpecKRequestState:
    base_top_k: int
    output_top_ks: torch.Tensor = field(init=False)
    pending_draft_top_ks: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.output_top_ks = torch.empty(0, dtype=torch.uint8, device="cpu")

    def reconcile_output_length(self, output_length: int) -> None:
        current_length = self.output_top_ks.shape[0]
        if current_length > output_length:
            self.output_top_ks = self.output_top_ks[:output_length].clone()
        elif current_length < output_length:
            padding = torch.full(
                (output_length - current_length,),
                self.base_top_k,
                dtype=torch.uint8,
                device="cpu",
            )
            self.output_top_ks = torch.cat((self.output_top_ks, padding), dim=0)

    def finalize_step(
        self,
        update: SpecKHistoryUpdate,
        sampled_token_top_k: torch.Tensor,
    ) -> None:
        num_new_tokens = update.accepted_top_ks.shape[0] + 1
        previous_length = update.new_output_length - num_new_tokens
        if previous_length < 0:
            raise ValueError("Spec-K output history length became negative.")
        self.reconcile_output_length(previous_length)
        top_ks = torch.cat(
            (
                update.accepted_top_ks.to(dtype=torch.uint8, device="cpu"),
                sampled_token_top_k.to(dtype=torch.uint8, device="cpu").reshape(1),
            ),
            dim=0,
        )
        self.output_top_ks = torch.cat((self.output_top_ks, top_ks), dim=0)

    def set_pending_draft_top_ks(self, top_ks: torch.Tensor) -> None:
        self.pending_draft_top_ks = top_ks.to(dtype=torch.uint8, device="cpu").clone()

    def top_ks_for_positions(
        self,
        positions: np.ndarray,
        num_prompt_tokens: int,
        pending_draft_start_position: int,
        use_pending_draft: bool,
    ) -> torch.Tensor:
        result = torch.full(
            (len(positions),),
            self.base_top_k,
            dtype=torch.int32,
            device="cpu",
        )
        position_tensor = torch.from_numpy(positions.astype(np.int64, copy=False))

        output_indices = position_tensor - num_prompt_tokens
        history_mask = (output_indices >= 0) & (output_indices < self.output_top_ks.shape[0])
        if history_mask.any():
            result[history_mask] = self.output_top_ks[output_indices[history_mask]].to(torch.int32)

        pending = self.pending_draft_top_ks
        if use_pending_draft and pending is not None:
            pending_indices = position_tensor - pending_draft_start_position
            pending_mask = (pending_indices >= 0) & (pending_indices < pending.shape[0])
            if pending_mask.any():
                result[pending_mask] = pending[pending_indices[pending_mask]].to(torch.int32)
        return result
