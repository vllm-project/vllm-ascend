# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Tests for the per_row wiring of vLLM's sparse indexer top-k on Ascend."""

import types

import torch
from vllm import _custom_ops as vllm_custom_ops
from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk

from vllm_ascend.patch.worker import patch_indexer_topk


def _recording_op(name, calls):
    def op(*args):
        calls.append((name, args))

    return op


def test_install_is_a_noop_while_the_ascend_operator_is_missing():
    target = types.SimpleNamespace()
    assert patch_indexer_topk.install(ops_module=target, get_op=lambda name: None) is False
    assert not hasattr(target, patch_indexer_topk.PREFILL_ENTRY)
    assert not hasattr(target, patch_indexer_topk.DECODE_ENTRY)


def test_install_needs_both_entry_points():
    target = types.SimpleNamespace()
    only_decode = {patch_indexer_topk.DECODE_ENTRY: lambda *args: None}
    assert patch_indexer_topk.install(ops_module=target, get_op=only_decode.get) is False
    assert not hasattr(target, patch_indexer_topk.PREFILL_ENTRY)


def test_install_rebinds_with_the_vllm_signatures():
    target = types.SimpleNamespace()
    calls = []
    ops = {
        patch_indexer_topk.PREFILL_ENTRY: _recording_op("prefill", calls),
        patch_indexer_topk.DECODE_ENTRY: _recording_op("decode", calls),
    }
    assert patch_indexer_topk.install(ops_module=target, get_op=ops.get) is True

    target.top_k_per_row_prefill(1, 2, 3, 4, 5, 6, 7, 8)
    target.top_k_per_row_decode(9, 10, 11, 12, 13, 14, 15, 16)

    assert calls == [
        ("prefill", (1, 2, 3, 4, 5, 6, 7, 8)),
        ("decode", (9, 10, 11, 12, 13, 14, 15, 16)),
    ]


def test_auto_backend_resolves_to_per_row_without_cuda():
    """The chain the NPU takes: cooperative and persistent both need CUDA."""
    selector = SparseIndexerTopk(backend="auto")
    selector._is_cuda = False
    selector._cooperative_capable = False
    logits = torch.zeros(2, 8)
    assert selector.resolve_backend(logits, 4, 2) == "per_row"


def test_indexer_per_row_backend_reaches_the_ascend_operator(monkeypatch):
    """vLLM's indexer must end up calling the AscendC operator, unmodified."""
    calls = []
    # Register the restore before installing, so the rebind does not leak.
    monkeypatch.setattr(vllm_custom_ops, "top_k_per_row_decode", vllm_custom_ops.top_k_per_row_decode)
    monkeypatch.setattr(vllm_custom_ops, "top_k_per_row_prefill", vllm_custom_ops.top_k_per_row_prefill)
    ops = {
        patch_indexer_topk.PREFILL_ENTRY: _recording_op("prefill", calls),
        patch_indexer_topk.DECODE_ENTRY: _recording_op("decode", calls),
    }
    assert patch_indexer_topk.install(ops_module=vllm_custom_ops, get_op=ops.get) is True

    num_rows, width, topk_tokens, next_n = 2, 8, 3, 1
    logits = torch.zeros(num_rows, width)
    seq_lens = torch.tensor([5, 8], dtype=torch.int32)
    topk_indices = torch.full((num_rows, topk_tokens), -1, dtype=torch.int32)

    selector = SparseIndexerTopk(backend="per_row")
    selector.forward(logits, seq_lens, next_n, topk_indices, topk_tokens, width)

    assert len(calls) == 1
    name, args = calls[0]
    assert name == "decode"
    assert args[0] is logits
    assert args[1] == next_n
    assert args[2] is seq_lens
    assert args[3] is topk_indices
    assert args[4:] == (num_rows, logits.stride(0), logits.stride(1), topk_tokens)
