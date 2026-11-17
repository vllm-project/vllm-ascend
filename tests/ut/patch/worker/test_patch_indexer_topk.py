# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Tests for the per_row wiring of vLLM's sparse indexer top-k on Ascend."""

import types

import pytest
import torch
from vllm import _custom_ops as vllm_custom_ops
from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk

from vllm_ascend.patch.worker import patch_indexer_topk

ARGS8 = (1, 2, 3, 4, 5, 6, 7, 8)


def _recording_op(name, calls):
    def op(*args):
        calls.append((name, args))

    return op


def _fake_ops(calls, prefix):
    return {
        patch_indexer_topk.PREFILL_ENTRY: _recording_op(f"{prefix}-prefill", calls),
        patch_indexer_topk.DECODE_ENTRY: _recording_op(f"{prefix}-decode", calls),
    }


def test_dispatch_prefers_the_ascend_operator(monkeypatch):
    calls = []
    monkeypatch.setattr(patch_indexer_topk, "_LOOKUP", _fake_ops(calls, "ascend").get)

    target = types.SimpleNamespace(**_fake_ops(calls, "vllm"))
    assert patch_indexer_topk.install(ops_module=target) is True

    target.top_k_per_row_prefill(*ARGS8)
    target.top_k_per_row_decode(*ARGS8)

    assert calls == [("ascend-prefill", ARGS8), ("ascend-decode", ARGS8)]


def test_dispatch_falls_back_while_the_operator_is_not_registered(monkeypatch):
    """Before ``vllm_ascend_C`` is imported the patch must not change behaviour."""
    calls = []
    monkeypatch.setattr(patch_indexer_topk, "_LOOKUP", lambda name: None)

    target = types.SimpleNamespace(**_fake_ops(calls, "vllm"))
    assert patch_indexer_topk.install(ops_module=target) is True

    target.top_k_per_row_prefill(*ARGS8)
    target.top_k_per_row_decode(*ARGS8)

    assert calls == [("vllm-prefill", ARGS8), ("vllm-decode", ARGS8)]


def test_dispatch_raises_without_any_implementation(monkeypatch):
    monkeypatch.setattr(patch_indexer_topk, "_LOOKUP", lambda name: None)

    target = types.SimpleNamespace()
    assert patch_indexer_topk.install(ops_module=target) is True

    with pytest.raises(RuntimeError, match="TopKPerRow"):
        target.top_k_per_row_prefill(*ARGS8)


def test_install_is_idempotent(monkeypatch):
    monkeypatch.setattr(patch_indexer_topk, "_LOOKUP", lambda name: None)

    target = types.SimpleNamespace(**_fake_ops([], "vllm"))
    assert patch_indexer_topk.install(ops_module=target) is True
    rebound = target.top_k_per_row_prefill
    assert patch_indexer_topk.install(ops_module=target) is False
    assert target.top_k_per_row_prefill is rebound


def test_auto_backend_resolves_to_per_row_without_cuda():
    """The chain the NPU takes: cooperative and persistent both need CUDA."""
    selector = SparseIndexerTopk(backend="auto")
    selector._is_cuda = False
    logits = torch.zeros(2, 8)
    assert selector.resolve_backend(logits, 4, 2) == "per_row"


def test_indexer_per_row_backend_reaches_the_ascend_operator(monkeypatch):
    """vLLM's indexer must end up calling the AscendC operator, unmodified."""
    calls = []
    monkeypatch.setattr(patch_indexer_topk, "_LOOKUP", _fake_ops(calls, "ascend").get)

    num_rows, width, topk_tokens, next_n = 2, 8, 3, 1
    logits = torch.zeros(num_rows, width)
    seq_lens = torch.tensor([5, 8], dtype=torch.int32)
    topk_indices = torch.full((num_rows, topk_tokens), -1, dtype=torch.int32)

    selector = SparseIndexerTopk(backend="per_row")
    selector.forward(logits, seq_lens, next_n, topk_indices, topk_tokens, width)

    assert len(calls) == 1
    name, args = calls[0]
    assert name == "ascend-decode"
    assert args[0] is logits
    assert args[1] == next_n
    assert args[2] is seq_lens
    assert args[3] is topk_indices
    assert args[4:] == (num_rows, logits.stride(0), logits.stride(1), topk_tokens)


def test_vllm_custom_ops_is_rebound_at_import():
    """Importing the patch must leave vLLM's entry points on the wrapper."""
    assert getattr(vllm_custom_ops.top_k_per_row_prefill, "_ascend_dispatch", False)
    assert getattr(vllm_custom_ops.top_k_per_row_decode, "_ascend_dispatch", False)
