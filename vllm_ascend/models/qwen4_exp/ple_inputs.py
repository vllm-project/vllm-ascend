# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp PLE inputs for the asynchronous v1 runner."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_ascend.models.qwen4_exp.ple_history import Qwen4ExpPLEHistory


class Qwen4ExpPLEInputs:
    """Adapt scheduler state and repaired inputs to a real PLE token history.

    This is separate from the synchronous CPU history path and from the
    captured model inputs. It only runs for PLE models using async scheduling.
    """

    def __init__(self, history: Qwen4ExpPLEHistory) -> None:
        self.history = history

    @classmethod
    def from_runner(cls, runner: Any) -> Qwen4ExpPLEInputs | None:
        config = runner.model_config.hf_text_config
        if not getattr(config, "ple_layer_ids", None) or not runner.use_async_scheduling:
            return None
        if runner.num_spec_tokens:
            raise NotImplementedError(
                "Qwen4Exp asynchronous PLE does not yet support speculative decoding; "
                "use --no-async-scheduling with speculative decoding"
            )
        parallel_config = runner.parallel_config
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("Qwen4Exp asynchronous PLE requires pipeline_parallel_size=1")
        if parallel_config.decode_context_parallel_size > 1 or parallel_config.prefill_context_parallel_size > 1:
            raise NotImplementedError("Qwen4Exp asynchronous PLE does not yet support context parallelism")
        return cls(
            Qwen4ExpPLEHistory(
                max_num_reqs=runner.max_num_reqs,
                max_model_len=runner.max_model_len,
                max_num_tokens=runner.max_num_tokens,
                context_len=int(config.ngram_size) - 1,
                eos_token_id=int(config.eos_token_id),
                device=runner.device,
                pin_memory=runner.pin_memory,
            )
        )

    def update_requests(self, input_batch: Any, scheduler_output: Any, requests: dict[str, Any]) -> None:
        """Follow the final batch after upstream removal, resume, and reorder.

        Only active batch requests occupy history slots. The scheduler supplies
        real token snapshots when an asynchronous request rejoins the batch.
        Never rebuild generated history from an unchecked CPU placeholder.
        """
        req_ids = input_batch.req_ids
        finished_req_ids = scheduler_output.finished_req_ids
        # A finished ID can be resubmitted in the same step. Release its old
        # identity even if it is present in the new batch.
        self.history.retain_requests([req_id for req_id in req_ids if req_id not in finished_req_ids])
        new_req_ids = {req.req_id for req in scheduler_output.scheduled_new_reqs}
        snapshots = scheduler_output.scheduled_cached_reqs.all_token_ids
        for req_idx, req_id in enumerate(req_ids):
            if req_id not in new_req_ids and self.history.has_request(req_id):
                # Keep the device history for a forced resume in the same
                # batch; an authoritative CPU snapshot may not be available.
                continue
            req_state = requests[req_id]
            if req_state.prompt_token_ids is None:
                raise ValueError("Qwen4Exp asynchronous PLE requires prompt token IDs")
            if req_id in new_req_ids:
                token_ids = req_state.prompt_token_ids
            else:
                token_ids = snapshots.get(req_id)
                if token_ids is None:
                    token_ids = req_state.prompt_token_ids + req_state.output_token_ids
            self.history.initialize_request(req_id, token_ids, int(input_batch.num_computed_tokens_cpu[req_idx]))

    def prepare_inputs(
        self,
        input_batch: Any,
        query_lens: np.ndarray,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> None:
        """Record corrected device inputs before the model consumes PLE context."""
        num_reqs = input_batch.num_reqs
        self.history.prepare_inputs(
            req_ids=input_batch.req_ids,
            context_ends=input_batch.num_computed_tokens_cpu[:num_reqs].tolist(),
            query_lens=query_lens.tolist(),
            input_ids=input_ids,
            positions=positions,
            query_start_loc=query_start_loc,
        )

    def ngram_context(self, num_reqs_padded: int) -> torch.Tensor:
        # The history's runtime buffer is distinct from the captured buffer,
        # whose padding routine fills it before copying the active rows.
        return self.history.context_buffer[:num_reqs_padded]
