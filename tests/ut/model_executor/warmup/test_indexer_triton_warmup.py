# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

warmup = importlib.import_module("vllm_ascend.model_executor.warmup.indexer_triton_warmup")


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


def _make_worker():
    config = SimpleNamespace(
        model_type="deepseek_v41",
        num_hidden_layers=4,
        compress_ratios=[0, 1, 2, 2, 8],
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
    )
    model_config = SimpleNamespace(hf_text_config=config, dtype=torch.bfloat16)
    vllm_config = SimpleNamespace(
        model_config=model_config,
        kernel_config=SimpleNamespace(enable_jit_warmup=True),
    )
    return SimpleNamespace(
        model_config=model_config,
        vllm_config=vllm_config,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        device=torch.device("cpu"),
    )


def test_indexer_warmup_uses_compile_only_wrappers():
    worker = _make_worker()

    with (
        patch.object(warmup, "HAS_TRITON", True),
        patch.object(warmup, "is_deepseek_v41", return_value=True),
        patch.object(warmup, "get_vectorcore_num", return_value=40),
        patch.object(warmup._QUANTIZE_INDEXER_QUERY_KERNEL, "compile") as mock_quantize,
        patch.object(warmup._PREPARE_INDEXER_INDICES_KERNEL, "compile") as mock_prepare,
    ):
        warmup.indexer_triton_warmup(worker)

    mock_quantize.assert_called_once()
    assert mock_prepare.call_count > 0
