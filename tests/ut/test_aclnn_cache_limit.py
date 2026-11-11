# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Unit tests for the ACLNN operator-descriptor cache advisory.

The advisory exists because CANN caches at most ``ACLNN_CACHE_LIMIT`` operator
descriptors per worker. Model Runner V2 with speculative decoding and ACL Graph
can exhaust that cache, which adds host-side tiling/workspace work; the variable
is exposed so that this can be diagnosed and, if it helps, tuned (see
https://github.com/vllm-project/vllm-ascend/issues/21741).
"""

from types import SimpleNamespace

import pytest
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.platform import (
    ACLNN_CACHE_LIMIT_RECOMMENDED,
    aclnn_cache_limit_advisory,
)


def _config(cudagraph_mode: CUDAGraphMode, speculative: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=object() if speculative else None,
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
    )


def test_advises_for_spec_decode_with_full_cudagraph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACLNN_CACHE_LIMIT", raising=False)
    message = aclnn_cache_limit_advisory(_config(CUDAGraphMode.FULL_AND_PIECEWISE))
    assert message is not None
    assert f"ACLNN_CACHE_LIMIT={ACLNN_CACHE_LIMIT_RECOMMENDED}" in message
    assert "diagnostic knob" in message
    assert "re-measure" in message.lower() or "keep the value only if it actually helps" in message


def test_advises_for_full_decode_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACLNN_CACHE_LIMIT", raising=False)
    assert aclnn_cache_limit_advisory(_config(CUDAGraphMode.FULL_DECODE_ONLY)) is not None


@pytest.mark.parametrize("value", ["100000", "20000"])
def test_silent_when_limit_already_set(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("ACLNN_CACHE_LIMIT", value)
    assert aclnn_cache_limit_advisory(_config(CUDAGraphMode.FULL_AND_PIECEWISE)) is None


def test_silent_without_spec_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACLNN_CACHE_LIMIT", raising=False)
    assert aclnn_cache_limit_advisory(_config(CUDAGraphMode.FULL_AND_PIECEWISE, speculative=False)) is None


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE])
def test_silent_without_full_cudagraphs(monkeypatch: pytest.MonkeyPatch, mode: CUDAGraphMode) -> None:
    monkeypatch.delenv("ACLNN_CACHE_LIMIT", raising=False)
    assert aclnn_cache_limit_advisory(_config(mode)) is None
