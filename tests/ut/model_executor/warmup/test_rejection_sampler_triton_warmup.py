# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``rejection_sampler_triton_warmup``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.ut.model_executor.warmup.helpers import make_mock_worker
from vllm_ascend.model_executor.warmup import rejection_sampler_triton_warmup as rw


def test_collect_warmup_rejection_block_sizes():
    with patch("vllm_ascend.ops.triton.reject_sample.get_vectorcore_num", return_value=4):
        sizes = rw.collect_warmup_rejection_block_sizes(8)
        block_sizes = {rw.cal_grid_and_block_size(b)[1] for b in sizes}

    assert sizes[0] == 1
    assert sizes[-1] == 8
    assert len(block_sizes) == len(sizes)


@patch.object(rw, "_warm_rejection_random")
@patch.object(rw, "_warm_greedy")
@patch.object(rw, "_warm_expand")
@patch.object(rw, "_warm_sample_recovered")
@patch.object(rw, "_warm_prepare_inputs")
@patch.object(rw, "get_ascend_config")
@patch("vllm_ascend.ops.triton.reject_sample.get_vectorcore_num", return_value=4)
@patch.object(rw, "HAS_TRITON", True)
def test_rejection_sampler_triton_warmup(
    mock_vectorcore,
    mock_get_config,
    mock_prepare,
    mock_recovered,
    mock_expand,
    mock_greedy,
    mock_random,
):
    mock_get_config.return_value = MagicMock(
        enable_reduce_sample=False,
        rejection_sampler_config=MagicMock(
            enable_block_verify=False,
            enable_entropy_verify=False,
            posterior_threshold=0.95,
            posterior_alpha=0.4,
        ),
    )
    worker = make_mock_worker(
        max_num_seqs=8,
        pipeline_parallel_size=1,
        speculative_config=SimpleNamespace(num_speculative_tokens=4, method="eagle"),
    )
    req_batch_sizes = rw.collect_warmup_rejection_block_sizes(8)
    rw.rejection_sampler_triton_warmup(worker)

    mock_prepare.assert_called_once()
    assert mock_recovered.call_count == 1
    assert mock_expand.call_count == len(req_batch_sizes)
    assert mock_greedy.call_count == len(req_batch_sizes) * 2
    assert mock_random.call_count == len(req_batch_sizes)
