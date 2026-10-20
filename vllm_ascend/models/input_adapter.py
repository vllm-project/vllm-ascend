# SPDX-License-Identifier: Apache-2.0
"""Model-owned adapters for auxiliary Model Runner V1 inputs.

Keep model-specific buffer lifetimes and padding rules outside the generic
runner.  New model families can provide an adapter without adding another
architecture branch to ``model_runner_v1.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch


@dataclass(frozen=True)
class ModelInputContext:
    """Runtime state available to model-specific input adapters."""

    num_reqs: int
    num_reqs_padded: int
    is_dummy: bool
    query_start_loc: torch.Tensor
    num_computed_tokens: np.ndarray
    token_ids: np.ndarray


class ModelInputAdapter(Protocol):
    """Adds model-owned auxiliary tensors to a forward call."""

    def prepare(
        self,
        model_kwargs: dict[str, Any],
        context: ModelInputContext,
    ) -> None: ...


def build_model_input_adapter(
    model_config: Any,
    *,
    max_num_reqs: int,
    device: torch.device,
) -> ModelInputAdapter | None:
    """Build the adapter advertised by the model configuration, if any."""
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is None or not getattr(text_config, "ple_layer_ids", None):
        return None

    # Import lazily so unrelated models do not acquire Qwen runtime imports.
    from vllm_ascend.models.qwen4_exp.ple_inputs import (
        Qwen4ExpPLEInputAdapter,
    )

    return Qwen4ExpPLEInputAdapter(
        text_config,
        max_num_reqs=max_num_reqs,
        device=device,
    )


__all__ = [
    "ModelInputAdapter",
    "ModelInputContext",
    "build_model_input_adapter",
]
