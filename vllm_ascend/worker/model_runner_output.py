# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from vllm.v1.outputs import ModelRunnerOutput


@dataclass
class AscendModelRunnerOutput(ModelRunnerOutput):
    """Ascend V1 output metadata needed by the PP + spec-decode protocol.

    The field stays on the Ascend-owned output type instead of being injected
    into vLLM's process-wide ``ModelRunnerOutput`` class.
    """

    draft_token_ids: list[list[int]] | None = None
