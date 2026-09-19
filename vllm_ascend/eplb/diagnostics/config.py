# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from pydantic import Field

from vllm_ascend.config_utils import config


@config
class EplbDiagnosticsConfig:
    """Print per-rank, per-layer MoE workload with EP enabled and EPLB disabled."""

    mode: Literal["off", "observe"] = "off"
    window_size: int = Field(default=32, ge=1)
    warmup_steps: int = Field(default=32, ge=0)
    max_windows: int = Field(default=0, ge=0)  # Zero observes the entire run.
