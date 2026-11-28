# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


class ProjectedDraftAuxStates(list[torch.Tensor]):
    """Explicit carrier for globally restored, already-combined DSpark states.

    A list keeps the runner's existing return/propose signatures intact.
    Consumers must retain this type; a singleton plain list is still raw aux.
    """

    def __init__(self, context_states: torch.Tensor) -> None:
        super().__init__([context_states])

    @property
    def context_states(self) -> torch.Tensor:
        if len(self) != 1:
            raise ValueError("Projected DSpark states must contain exactly one tensor.")
        return self[0]
