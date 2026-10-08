# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Huawei Technologies Co., Ltd.

from types import SimpleNamespace

import pytest
import torch


def test_ngram_propose_uses_committed_history_and_handles_padding(monkeypatch):
    pytest.importorskip("torch_npu")

    from vllm_ascend.worker.v2.spec_decode import ngram

    history = torch.full((3, 8), -1, dtype=torch.int32)
    history[1, :6] = torch.tensor([1, 2, 3, 4, 8, 9])
    total_len = torch.tensor([0, 6, 0], dtype=torch.int32)
    speculator = object.__new__(ngram.AscendNgramSpeculator)
    speculator.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=history),
        total_len=SimpleNamespace(gpu=total_len),
    )
    speculator.drafts = torch.zeros((2, 2), dtype=torch.int64)
    speculator.draft_offsets = torch.arange(2)
    speculator.num_speculative_steps = 2
    speculator.vocab_size = 100
    speculator.min_n = 1
    speculator.max_n = 2

    captured = {}

    def fake_kernel(token_ids, num_tokens, sampled, discard, *args, **kwargs):
        captured.update(
            history=token_ids,
            num_tokens=num_tokens,
            sampled=sampled,
            discard=discard.clone(),
            idx_mapping=kwargs["idx_mapping"].clone(),
            num_sampled=kwargs["num_sampled"].clone(),
        )
        return (
            torch.zeros(2, dtype=torch.int32),
            torch.tensor([[11, 12], [-1, -1]], dtype=torch.int32),
            torch.tensor([2, 0], dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
        )

    monkeypatch.setattr(ngram, "triton_ngram_spec_decode", fake_kernel)
    input_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=torch.tensor([1, -1], dtype=torch.int32),
    )
    output = speculator.propose(
        input_batch,
        None,
        None,
        torch.empty(0),
        None,
        torch.tensor([2, 0], dtype=torch.int32),
        torch.zeros(2, dtype=torch.int32),
        torch.tensor([[0], [9], [0]], dtype=torch.int64),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
    )

    assert captured["history"] is history
    assert captured["num_tokens"] is total_len
    assert captured["sampled"] is None
    assert captured["idx_mapping"].tolist() == [1, -1]
    assert captured["num_sampled"].tolist() == [2, 0]
    assert captured["discard"].tolist() == [False, True]
    assert history[1, 4].item() == 8
    assert output.tolist() == [[11, 12], [-1, -1]]
