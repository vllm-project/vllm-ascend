# SPDX-License-Identifier: Apache-2.0
"""[Role] Carry MRv2 grammar coverage on stock vLLM 0.30.0.
[Invariants] Limits follow structured_output_request_ids across worker transport.
[Timeouts & Locks] No waits, locks, or device operations.
[Side-Effect Policy] Defines an extended dataclass without modifying vLLM files.
"""

from dataclasses import dataclass

from vllm.v1.core.sched.output import GrammarOutput as VllmGrammarOutput


@dataclass
class GrammarOutput(VllmGrammarOutput):
    # None (warmup/legacy metadata) conservatively rejects structured drafts.
    num_acceptable_drafts: list[int] | None = None
