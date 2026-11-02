# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph as graph_module


@pytest.mark.parametrize(
    "capturing,mode,expected",
    [
        (True, CUDAGraphMode.FULL, True),
        (False, CUDAGraphMode.FULL, False),
    ],
)
def test_dflash_context_prefix_wrapper_sets_capture_context_and_forwards(monkeypatch, capturing, mode, expected):
    from vllm_ascend.worker.v2 import aclgraph_utils as wrapper_module

    original = SimpleNamespace(precompute_and_store_context_kv=MagicMock(return_value="stored"))
    wrapper = wrapper_module.ModelWithContext(original, is_draft_model=True)
    monkeypatch.setattr(wrapper_module, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=mode))
    config = object()
    context = MagicMock(return_value=nullcontext())
    monkeypatch.setattr(wrapper_module, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(wrapper_module, "set_forward_context", context)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(wrapper_module, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False, capturing=False))
    states, positions, slots = SimpleNamespace(shape=(16, 4096)), object(), [object(), object()]
    assert wrapper.precompute_and_store_context_kv(states, positions, slots) == "stored"
    original.precompute_and_store_context_kv.assert_called_once_with(states, positions, slots)
    context.assert_called_once_with(None, config, num_tokens=16, cudagraph_runtime_mode=CUDAGraphMode.FULL)
    assert wrapper_module._EXTRA_CTX.is_draft_model is True
    assert wrapper_module._EXTRA_CTX.capturing is expected


def test_native_capture_forwards_prefix_once_and_restores_model_on_error(monkeypatch):
    from vllm_ascend.worker.v2 import aclgraph_utils as wrapper_module

    original = SimpleNamespace(precompute_and_store_context_kv=MagicMock())
    spec = SimpleNamespace(model=original)
    manager = graph_module.DFlashAclGraphManager.__new__(graph_module.DFlashAclGraphManager)
    manager.speculator = spec
    manager.vllm_config = object()
    calls = []

    def capture(self, forward_fn, *args, precompute_context_kv=None, prepare_forward=None, **kwargs):
        assert spec.model.get_original_model() is original
        hook = precompute_context_kv if vllm_version_is("0.31.0") else prepare_forward
        hook(2)
        forward_fn(2)
        raise ValueError("capture failed")

    monkeypatch.setattr(graph_module.DFlashCudaGraphManager, "capture", capture)
    monkeypatch.setattr(graph_module, "communicator_switch", lambda: nullcontext())
    monkeypatch.setattr(graph_module, "set_current_vllm_config", lambda config: nullcontext())
    monkeypatch.setattr(graph_module, "model_capture_wrapper", wrapper_module.model_capture_wrapper)
    prefix_kwargs = (
        {"precompute_context_kv": lambda n: calls.append(("context", n))}
        if vllm_version_is("0.31.0")
        else {"prepare_forward": lambda n: calls.append(("context", n))}
    )
    with pytest.raises(ValueError, match="capture failed"):
        manager.capture(
            lambda n: calls.append(("query", n)),
            input_buffers=MagicMock(),
            block_tables=MagicMock(),
            attn_groups=[],
            kv_cache_config=MagicMock(),
            max_model_len=4096,
            **prefix_kwargs,
        )
    assert calls == [("context", 2), ("query", 2)]
    assert spec.model is original
