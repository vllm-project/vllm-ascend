# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.model_executor.warmup import deepseek_v41_triton_warmup as warmup


@pytest.mark.parametrize(
    "topk,cores,max_tokens,expected",
    [
        (2048, 40, 4096, [1, 41]),
        (2048, 40, 40, [1]),
        (2048, 48, 4096, [1, 49]),
        (128, 40, 4096, [1, 41, 81, 161, 321, 641]),
        (128, 40, 128, [1, 41, 81]),
        (7, 40, 1, [1]),
    ],
)
def test_collect_indexer_warmup_token_counts(topk, cores, max_tokens, expected):
    assert warmup.collect_indexer_warmup_token_counts(topk, cores, max_tokens) == expected


@pytest.mark.parametrize("model_type", ["deepseek_v4.1", "deepseek_v41", "deepseek_v4.1_text", "deepseek_v41_text"])
def test_warmup_covers_tiles_and_active_compression_ratios(monkeypatch, model_type):
    config = SimpleNamespace(
        model_type=model_type,
        num_hidden_layers=4,
        compress_ratios=[0, 1, 2, 2, 8],
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
    )
    worker = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=config, dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        device=torch.device("cpu"),
    )
    quantize, prepare = Mock(), Mock()
    monkeypatch.setattr(warmup, "HAS_TRITON", True)
    monkeypatch.setattr(warmup, "get_vectorcore_num", lambda: 40)
    monkeypatch.setattr(warmup, "quantize_indexer_query", quantize)
    monkeypatch.setattr(warmup, "prepare_indexer_indices", prepare)
    warmup.deepseek_v41_triton_warmup(worker)
    query = quantize.call_args.args[0]
    assert query.shape == (1, 64, 128)
    assert query.dtype == torch.bfloat16
    assert [(call.args[0].shape[0], call.args[2]) for call in prepare.call_args_list] == [
        (1, 1),
        (1, 2),
        (41, 1),
        (41, 2),
    ]
    assert all(call.args[1].dtype == torch.int64 for call in prepare.call_args_list)


@pytest.mark.parametrize(
    "has_triton,model_type,ratios", [(False, "deepseek_v41", [1]), (True, "other", [1]), (True, "deepseek_v41", [0])]
)
def test_warmup_skips_unused_indexer(monkeypatch, has_triton, model_type, ratios):
    worker = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, num_hidden_layers=1, compress_ratios=ratios)
        )
    )
    quantize, prepare = Mock(), Mock()
    monkeypatch.setattr(warmup, "HAS_TRITON", has_triton)
    monkeypatch.setattr(warmup, "quantize_indexer_query", quantize)
    monkeypatch.setattr(warmup, "prepare_indexer_indices", prepare)
    warmup.deepseek_v41_triton_warmup(worker)
    quantize.assert_not_called()
    prepare.assert_not_called()
