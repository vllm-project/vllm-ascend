# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Reuse upstream suffix orchestration with Ascend kernel launch adapters."""

from __future__ import annotations

from vllm.v1.worker.gpu.spec_decode.suffix import speculator as suffix_speculator


class AscendSuffixSpeculator(suffix_speculator.SuffixSpeculator):
    """Inherit buffers, lifecycle and drafting flow; adapt device launches only."""
