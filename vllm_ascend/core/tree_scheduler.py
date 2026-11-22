# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.spec_decode.metrics import SpecDecodingStats

from vllm_ascend.worker.v2.spec_decode.tree.shape import TreeShape


class TreeScheduler(Scheduler):
    """Reserve target tree rows while keeping the DFlash2 draft depth unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = TreeShape.from_config(self.vllm_config)
        assert shape is not None
        self.tree_budget = shape.draft_node_budget

        # DFlash2 drafts beyond the target query rows. The KV admission
        # estimate covers both widths, and Mamba accounts for its own states.
        step_tokens = shape.target_verification_width + self.num_lookahead_tokens
        self._spec_decode_step_blocks = sum(
            cdiv(step_tokens, manager.block_size)
            for manager in self.kv_cache_manager.coordinator.single_type_managers
            if not isinstance(manager.kv_cache_spec, MambaSpec)
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        # Padding, KV admission, and async placeholders use the target's
        # verifier width. Workers still generate at the draft depth.
        depth = self.num_spec_tokens
        self.num_spec_tokens = self.tree_budget
        try:
            output = super().schedule(throttle_prefills=throttle_prefills)
        finally:
            self.num_spec_tokens = depth
        output.num_spec_tokens_to_schedule = depth
        return output

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        num_draft_tokens -= (num_invalid_spec_tokens or {}).get(request_id, 0)
        # Log the whole tree in the scalar draft count. Per-position metric
        # arrays retain the draft depth expected by upstream logging.
        positions = min(num_draft_tokens, self.num_spec_tokens)
        spec_decoding_stats.observe_draft(positions, num_accepted_tokens)
        spec_decoding_stats.num_draft_tokens += num_draft_tokens - positions
        return spec_decoding_stats


class TreeAsyncScheduler(AsyncScheduler, TreeScheduler):
    """Combine native asynchronous request publication with Tree scheduling."""
