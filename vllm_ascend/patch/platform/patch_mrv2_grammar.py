# SPDX-License-Identifier: Apache-2.0
"""[Role] Backport vLLM #54442 scheduler metadata for MRv2.
[Invariants] Only the draft prefix inspected by the FSM may be accepted.
[Timeouts & Locks] No new waits or locks; retain the original bitmask producer.
[Side-Effect Policy] Patch the scheduler method; never rewrite upstream files.

Remove when the supported vLLM release includes #54442.
"""

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.structured_output.utils import strip_speculative_padding

from vllm_ascend.core.grammar_output import GrammarOutput

_ORIGINAL_GET_GRAMMAR_BITMASK = Scheduler.get_grammar_bitmask


def get_grammar_bitmask(self, scheduler_output):
    grammar = _ORIGINAL_GET_GRAMMAR_BITMASK(self, scheduler_output)
    if grammar is None:
        return None
    drafts = scheduler_output.scheduled_spec_decode_tokens
    limits = [
        len(strip_speculative_padding(drafts.get(req_id, []))) for req_id in grammar.structured_output_request_ids
    ]
    return GrammarOutput(grammar.structured_output_request_ids, grammar.grammar_bitmask, limits)


# Patch the base class before the balance scheduler and other subclasses load.
Scheduler.get_grammar_bitmask = get_grammar_bitmask
