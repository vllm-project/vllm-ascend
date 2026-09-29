# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for speculative-decoding Triton warmup registration."""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

rw = importlib.import_module("vllm_ascend.model_executor.warmup.rejection_sampler_triton_warmup")
reject_sample = importlib.import_module("vllm_ascend.ops.triton.reject_sample")
spec_utils = importlib.import_module("vllm_ascend.ops.triton.spec_decode.utils")


def _make_worker(
    *,
    enable_jit_warmup=True,
    enable_block_verify=False,
    num_speculative_tokens=4,
):
    spec_config = SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        method="eagle",
        use_ngram_gpu=lambda: False,
    )
    vllm_config = SimpleNamespace(
        speculative_config=spec_config,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        kernel_config=SimpleNamespace(enable_jit_warmup=enable_jit_warmup),
    )
    ascend_config = SimpleNamespace(
        enable_reduce_sample=False,
        rejection_sampler_config=SimpleNamespace(
            enable_block_verify=enable_block_verify,
            enable_entropy_verify=False,
        ),
    )
    worker = SimpleNamespace(
        vllm_config=vllm_config,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        model_runner=SimpleNamespace(
            rejection_sampler=SimpleNamespace(synthetic_mode=False),
            jit_warmup_registry=JitWarmupRegistry(vllm_config),
        ),
    )
    return worker, ascend_config


def test_collect_warmup_rejection_block_sizes():
    with patch.object(reject_sample, "get_vectorcore_num", return_value=4):
        sizes = rw.collect_warmup_rejection_block_sizes(8)
    assert sizes == (1, 2)


def _make_context(*, max_spec_len: int, synthetic_mode: bool = False):
    return rw.RejectionWarmupContext(
        block_sizes=(1, 2),
        max_spec_len=max_spec_len,
        no_draft_probs_values=(True,),
        enable_reduce_sampling=False,
        entropy_verify=False,
        synthetic_mode=synthetic_mode,
        block_verify=False,
    )


def test_greedy_key_selection_follows_runtime_dispatch():
    context = _make_context(max_spec_len=1)

    spec_len_1_keys = reject_sample._REJECTION_GREEDY_SPEC_LEN_1_KERNEL.get_warmup_keys(context)
    greedy_keys = reject_sample._REJECTION_GREEDY_KERNEL.get_warmup_keys(context)

    assert {key.block_size for key in spec_len_1_keys} == {1, 2}
    assert {key.synthetic_mode for key in spec_len_1_keys} == {False}
    assert {key.block_size for key in greedy_keys} == {1, 2}
    assert {key.has_is_greedy for key in greedy_keys} == {True}
    assert {key.synthetic_mode for key in greedy_keys} == {False}


def test_rejection_sampler_triton_warmup_registers_selected_kernels():
    worker, ascend_config = _make_worker(num_speculative_tokens=1)

    with (
        patch.object(rw, "get_ascend_config", return_value=ascend_config),
        patch.object(rw, "collect_warmup_rejection_block_sizes", return_value=(1,)),
        patch.object(spec_utils._PREPARE_INPUTS_PADDED_KERNEL, "compile") as mock_prepare,
        patch.object(reject_sample._EXPAND_KERNEL, "compile") as mock_expand,
        patch.object(reject_sample._REJECTION_GREEDY_SPEC_LEN_1_KERNEL, "compile") as mock_greedy_1,
        patch.object(reject_sample._REJECTION_GREEDY_KERNEL, "compile") as mock_greedy,
        patch.object(reject_sample._SAMPLE_RECOVERED_TOKENS_KERNEL, "compile") as mock_recovered,
        patch.object(reject_sample._REJECTION_RANDOM_SAMPLE_KERNEL, "compile") as mock_random,
        patch.object(reject_sample._REJECTION_RANDOM_SAMPLE_BLOCK_VERIFY_KERNEL, "compile") as mock_block,
    ):
        rw.rejection_sampler_triton_warmup(worker)

    mock_prepare.assert_called()
    mock_expand.assert_called()
    mock_greedy_1.assert_called()
    mock_greedy.assert_called()
    mock_recovered.assert_called()
    mock_random.assert_called()
    mock_block.assert_not_called()


def test_rejection_sampler_triton_warmup_selects_block_verify_kernel():
    worker, ascend_config = _make_worker(enable_block_verify=True)

    with (
        patch.object(rw, "get_ascend_config", return_value=ascend_config),
        patch.object(rw, "collect_warmup_rejection_block_sizes", return_value=(1,)),
        patch.object(reject_sample._REJECTION_RANDOM_SAMPLE_KERNEL, "compile") as mock_random,
        patch.object(reject_sample._REJECTION_RANDOM_SAMPLE_BLOCK_VERIFY_KERNEL, "compile") as mock_block,
    ):
        rw.rejection_sampler_triton_warmup(worker)

    mock_random.assert_not_called()
    mock_block.assert_called()


def test_rejection_sampler_triton_warmup_respects_disable_flag():
    worker, ascend_config = _make_worker(enable_jit_warmup=False)

    with (
        patch.object(rw, "get_ascend_config", return_value=ascend_config),
        patch.object(reject_sample._EXPAND_KERNEL, "compile") as mock_expand,
    ):
        rw.rejection_sampler_triton_warmup(worker)

    mock_expand.assert_not_called()
