# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft decode graph mode selection for the Ascend AR speculator."""

import pytest
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import (
    resolve_draft_decode_graph_mode,
)

ENV = "VLLM_ASCEND_SPEC_DECODE_DRAFT_GRAPH_MODE"


def test_default_keeps_draft_decode_eager(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert resolve_draft_decode_graph_mode(CUDAGraphMode.PIECEWISE, 3, False) == CUDAGraphMode.NONE


def test_piecewise_opt_in_captures_draft_decode(monkeypatch):
    monkeypatch.setenv(ENV, "piecewise")
    assert resolve_draft_decode_graph_mode(CUDAGraphMode.PIECEWISE, 3, False) == CUDAGraphMode.PIECEWISE


@pytest.mark.parametrize(
    ("num_speculative_steps", "enforce_eager"),
    [(1, False), (3, True)],
)
def test_piecewise_opt_in_ignored_without_a_multi_step_graph(monkeypatch, num_speculative_steps, enforce_eager):
    monkeypatch.setenv(ENV, "piecewise")
    assert (
        resolve_draft_decode_graph_mode(CUDAGraphMode.PIECEWISE, num_speculative_steps, enforce_eager)
        == CUDAGraphMode.NONE
    )


def test_full_target_decode_wins_over_the_opt_in(monkeypatch):
    monkeypatch.setenv(ENV, "none")
    assert resolve_draft_decode_graph_mode(CUDAGraphMode.FULL, 3, False) == CUDAGraphMode.FULL_DECODE_ONLY


def test_unknown_value_is_rejected(monkeypatch):
    monkeypatch.setenv(ENV, "full")
    with pytest.raises(ValueError, match="VLLM_ASCEND_SPEC_DECODE_DRAFT_GRAPH_MODE"):
        resolve_draft_decode_graph_mode(CUDAGraphMode.PIECEWISE, 3, False)
