# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.patch.worker import patch_model_runner


@pytest.mark.parametrize(
    "mode,enabled,skip_compiled,expected",
    [
        (CUDAGraphMode.NONE, True, False, True),
        (CUDAGraphMode.NONE, False, False, False),
        (CUDAGraphMode.NONE, None, False, False),
        (CUDAGraphMode.NONE, False, True, True),
        (CUDAGraphMode.PIECEWISE, True, False, False),
        (CUDAGraphMode.FULL, True, False, False),
    ],
)
def test_bypass_compiled_eager_steps(monkeypatch, mode, enabled, skip_compiled, expected):
    config = None
    if enabled is not None:
        compilation_config = SimpleNamespace(bypass_compiled_eager_steps=enabled)
        config = SimpleNamespace(ascend_compilation_config=compilation_config)
    lookup = Mock(return_value=config)
    forward_context = Mock()
    monkeypatch.setattr(patch_model_runner, "get_initialized_ascend_config", lookup)
    monkeypatch.setattr(patch_model_runner, "_set_forward_context", forward_context)

    attn_metadata, vllm_config = object(), object()
    result = patch_model_runner.set_forward_context(
        attn_metadata, vllm_config, cudagraph_runtime_mode=mode, skip_compiled=skip_compiled
    )

    assert result is forward_context.return_value
    forward_context.assert_called_once_with(
        attn_metadata, vllm_config, cudagraph_runtime_mode=mode, skip_compiled=expected
    )
