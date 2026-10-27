# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.dflash import speculator as module
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator


@pytest.mark.parametrize("flash", [False, True])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("dummy_run", [False, True])
@pytest.mark.parametrize("draft_cls", [module.AscendDFlashSpeculator, AscendDFlash2Speculator])
def test_context_kv_is_ready_before_query_and_deferred_only_for_flash_eager(
    monkeypatch, flash, eager, dummy_run, draft_cls
):
    calls = []
    cache = torch.zeros(1)
    draft = draft_cls.__new__(draft_cls)
    draft.speculative_config = SimpleNamespace(enforce_eager=eager)
    draft.flash_attn_layers = {}
    draft.flash_attn_metadata_executor = object() if flash else None
    draft._pending_context_kv = None

    @contextmanager
    def prepare(*_args):
        calls.append("metadata")
        yield
        calls.append("retire")

    def precompute(_self, start, end, dummy):
        assert (start, end, dummy) == (2, 8, dummy_run)
        calls.append("context")
        cache.fill_(7)

    def query(_self, *_args):
        assert cache.item() == 7
        calls.append("query")
        return cache.clone()

    monkeypatch.setattr(module, "prepare_flash_attn_metadata", prepare)
    monkeypatch.setattr(module.DFlashSpeculator, "_precompute_context_kv", precompute)
    monkeypatch.setattr(module.DFlashSpeculator, "_run_model", query)
    draft._precompute_context_kv(2, 8, dummy_run)
    if flash and eager:
        assert calls == []
    else:
        assert calls == ["context"]
    result = draft._run_model(8, {}, None, None)
    assert result.item() == 7
    expected = (
        ["metadata", "context", "query", "retire"] if flash and eager else ["context", "metadata", "query", "retire"]
    )
    assert calls == expected
    assert draft._pending_context_kv is None


def test_deferred_context_is_consumed_once_even_if_query_raises(monkeypatch):
    draft = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)
    draft.flash_attn_layers = {}
    draft.flash_attn_metadata_executor = None
    draft._pending_context_kv = (0, 8, False)
    calls = []
    monkeypatch.setattr(module.DFlashSpeculator, "_precompute_context_kv", lambda *_: calls.append("context"))

    def fail(*_args):
        raise RuntimeError("query failed")

    monkeypatch.setattr(module.DFlashSpeculator, "_run_model", fail)
    with pytest.raises(RuntimeError, match="query failed"):
        draft._run_model(8, None, None, None)
    assert calls == ["context"]
    assert draft._pending_context_kv is None
