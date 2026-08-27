# SPDX-License-Identifier: Apache-2.0
"""DFlash proposer cost profiling for D-Cut."""
from __future__ import annotations

import os

import torch
from vllm.config import CUDAGraphMode

from .globals import ENV_PROFILE_FORCE_EAGER, logger
from .utils import _npu_event


@torch.inference_mode()
def _adaptive_profile_draft_run(
    self,
    batch_size: int,
    context_tokens: int,
    n_warmup: int = 3,
    n_measure: int = 5,
):
    """Profile one complete DFlash proposer run for a D-Cut candidate.

    D-Cut truncates the verifier input after the proposer has produced the
    configured full K draft tokens. The next proposer invocation therefore
    still runs with ``B * (K + 1)`` query tokens. DFlash additionally prepares
    context K/V from this verifier step's Q target hidden states, so Q is passed
    separately as ``context_tokens`` instead of being conflated with the fixed
    draft query shape.

    The dummy buffers are overwritten by every real DFlash call. Profiling them
    after graph warmup and before serving requests therefore cannot leak draft
    context into a request.

    Returns ``(runtime_mode, avg_ms, padded_draft_query_tokens)``.
    """
    drafter = getattr(self, "drafter", None)
    if drafter is None:
        return "none", 0.0, 0
    if getattr(drafter, "method", None) != "dflash":
        if not getattr(self, "_dcut_draft_profile_fallback_logged", False):
            self._dcut_draft_profile_fallback_logged = True
            logger.warning(
                "D-Cut draft-cost profiling supports DFlash only; using "
                "target-only costs for method=%r.",
                getattr(drafter, "method", None),
            )
        return "unsupported", 0.0, 0

    draft_query_tokens = batch_size * (drafter.num_speculative_tokens + 1)
    has_lora = bool(self.input_batch.lora_id_to_lora_request)

    profile_force_eager = (
        os.environ.get(ENV_PROFILE_FORCE_EAGER, "1").lower()
        not in ("0", "false", "no")
    )
    if drafter.use_cuda_graph and not profile_force_eager:
        _, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens=draft_query_tokens,
            uniform_decode=True,
            has_lora=has_lora,
        )
        draft_input_tokens = batch_desc.num_tokens
        draft_input_tokens, _, _ = self._sync_metadata_across_dp(
            draft_input_tokens,
            is_draft_model=True,
        )
        runtime_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens=draft_input_tokens,
            uniform_decode=True,
            has_lora=has_lora,
        )
        draft_input_tokens = batch_desc.num_tokens
    else:
        runtime_mode = CUDAGraphMode.NONE
        batch_desc = None
        draft_input_tokens = draft_query_tokens

    def _draft_forward() -> None:
        drafter.dummy_run(
            num_tokens=draft_input_tokens,
            num_reqs=batch_size,
            aclgraph_runtime_mode=runtime_mode,
            batch_descriptor=batch_desc,
            is_profile=False,
            context_num_tokens=context_tokens,
        )

    for _ in range(max(n_warmup, 0)):
        _draft_forward()
    torch.npu.synchronize()

    avg_ms = 0.0
    if n_measure > 0:
        start_ev = _npu_event(enable_timing=True)
        end_ev = _npu_event(enable_timing=True)
        start_ev.record()
        for _ in range(n_measure):
            _draft_forward()
        end_ev.record()
        torch.npu.synchronize()
        avg_ms = start_ev.elapsed_time(end_ev) / n_measure

    mode_names = {
        CUDAGraphMode.FULL: "FCG",
        CUDAGraphMode.PIECEWISE: "PCG",
        CUDAGraphMode.NONE: "eager",
    }
    return (
        mode_names.get(runtime_mode, str(runtime_mode)),
        avg_ms,
        int(draft_input_tokens),
    )
