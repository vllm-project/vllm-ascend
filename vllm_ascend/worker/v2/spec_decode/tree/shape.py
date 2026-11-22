# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only tree dimensions shared by configuration, capture and publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class TreeShape:
    """Separate draft depth from the root-excluded tree budget.

    Node 0 is the accepted root; token/parent/depth storage contains nodes
    1..B. The DFlash2 draft runs one anchor query and one query per draft step.
    """

    draft_depth: int
    draft_node_budget: int

    def __post_init__(self) -> None:
        if self.draft_depth != 15 or not 1 <= self.draft_node_budget <= 128:
            raise ValueError("MegaTree requires draft depth 15 and a node budget from 1 through 128")

    @property
    def target_verification_width(self) -> int:
        """Maximum target rows per request, including its accepted root."""
        return 1 + self.draft_node_budget

    @classmethod
    def from_config(cls, vllm_config: VllmConfig) -> TreeShape | None:
        """Resolve early platform defaults without requiring Ascend's singleton."""
        from vllm_ascend.ascend_config import get_tree_spec_config

        tree_config = get_tree_spec_config(vllm_config)
        if tree_config is None:
            return None
        spec = vllm_config.speculative_config
        if spec is None or spec.num_speculative_tokens is None:
            raise ValueError("MegaTree requires speculative_config with num_speculative_tokens=15")
        return cls(draft_depth=spec.num_speculative_tokens, draft_node_budget=tree_config.budget)
