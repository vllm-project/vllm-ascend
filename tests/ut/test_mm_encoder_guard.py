# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""The multimodal encoder graph option must be rejected on Ascend, not hang.

``cudagraph_mm_encoder`` reaches ``GroupCoordinator.graph_capture()``, which asserts a
Cuda/XPU communicator. On Ascend the assertion fires inside the worker, the multiprocess
executor loses the message, and the client waits forever, so the platform validates the
option up front.
"""

from types import SimpleNamespace

import pytest

from vllm_ascend.platform import _reject_mm_encoder_cudagraph


def _config(enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mm_encoder=enabled))


def test_rejects_enabled_mm_encoder_graph() -> None:
    with pytest.raises(ValueError, match="cudagraph_mm_encoder=True is not supported on Ascend"):
        _reject_mm_encoder_cudagraph(_config(True))


def test_allows_disabled_mm_encoder_graph() -> None:
    _reject_mm_encoder_cudagraph(_config(False))


def test_tolerates_missing_compilation_config() -> None:
    _reject_mm_encoder_cudagraph(SimpleNamespace(compilation_config=None))
