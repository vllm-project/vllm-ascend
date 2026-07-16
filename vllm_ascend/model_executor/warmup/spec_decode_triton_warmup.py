# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up spec-decode Triton kernels used during rejection sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON, triton
from vllm.v1.sample.rejection_sampler import MAX_SPEC_LEN

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.triton.reject_sample import (
    cal_grid_and_block_size,
    expand_triton,
    rejection_greedy_sample_with_triton,
    rejection_random_sample_block_verify_kernel,
    rejection_random_sample_kernel,
    sample_recovered_tokens_kernel,
)
from vllm_ascend.ops.triton.spec_decode.utils import prepare_inputs_padded_kernel
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
from vllm_ascend.spec_decode.llm_base_proposer import _PREPARE_INPUTS_BLOCK_SIZE

if TYPE_CHECKING:
    from vllm_ascend.worker.worker import NPUWorker

# Keep dummy tensors small; JIT keys depend on constexpr flags, not vocab size.
_WARMUP_VOCAB_SIZE = 1024
_WARMUP_SELECTED_VOCAB_SIZE = 256
_SUB_BLOCK = 512
_VOCAB_BLOCK_SIZE = 512


def _collect_boundary_sizes(max_value: int) -> set[int]:
    sizes = {1, max_value}
    try:
        vectorcore_num = get_vectorcore_num()
        sizes.add(vectorcore_num)
        if vectorcore_num > 1:
            sizes.add(vectorcore_num - 1)
        sizes.add(vectorcore_num + 1)
    except AssertionError:
        pass
    return sizes


def collect_warmup_req_batch_sizes(max_num_reqs: int) -> list[int]:
    """Request batch sizes that cover distinct rejection/expand BLOCK_SIZE keys."""
    sizes = _collect_boundary_sizes(max_num_reqs)
    return sorted(size for size in sizes if 0 < size <= max_num_reqs)


def collect_warmup_token_sizes(
    max_num_tokens: int,
    cudagraph_capture_sizes: list[int] | None,
) -> list[int]:
    """Token batch sizes for kernels keyed on total token count."""
    sizes = _collect_boundary_sizes(max_num_tokens)
    if cudagraph_capture_sizes:
        for size in cudagraph_capture_sizes:
            if isinstance(size, int) and size > 0:
                sizes.add(min(size, max_num_tokens))
    return sorted(size for size in sizes if 0 < size <= max_num_tokens)


def collect_warmup_batch_sizes(
    max_num_reqs: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> list[int]:
    """Backward-compatible alias for request batch size collection."""
    del cudagraph_capture_sizes
    return collect_warmup_req_batch_sizes(max_num_reqs)


def _prepare_inputs_grid(num_reqs: int) -> tuple[int]:
    num_blocks = triton.cdiv(num_reqs, _PREPARE_INPUTS_BLOCK_SIZE)
    grid_size = min(num_blocks, get_vectorcore_num())
    return (max(grid_size, 1),)


def _warm_prepare_inputs_padded_kernel(
    device: torch.device,
    num_reqs: int,
) -> None:
    draft_lens = torch.ones(num_reqs, dtype=torch.int32, device=device)
    cu_num_draft_tokens = torch.cumsum(draft_lens, dim=0)
    valid_sampled_tokens_count = torch.ones(num_reqs, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(
        num_reqs + 1,
        dtype=torch.int32,
        device=device,
    )
    token_indices_to_sample = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_rejected_tokens_gpu = torch.empty(num_reqs, dtype=torch.int32, device=device)

    prepare_inputs_padded_kernel[_prepare_inputs_grid(num_reqs)](
        cu_num_draft_tokens,
        valid_sampled_tokens_count,
        query_start_loc,
        token_indices_to_sample,
        num_rejected_tokens_gpu,
        num_reqs,
        BLOCK_SIZE=_PREPARE_INPUTS_BLOCK_SIZE,
    )


def _warm_expand_kernel(device: torch.device, batch_size: int) -> None:
    cu_num_tokens = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
    num_tokens = int(cu_num_tokens[-1].item())
    x = torch.zeros(batch_size, dtype=torch.int32, device=device)
    expanded_x = torch.empty(num_tokens, dtype=torch.int32, device=device)
    expand_triton(
        batch_size,
        expanded_x,
        x,
        cu_num_tokens,
        replace_from=-1,
        replace_to=0,
        max_num_tokens=MAX_SPEC_LEN,
    )


def _make_rejection_tensors(
    batch_size: int,
    max_spec_len: int,
    vocab_size: int,
    device: torch.device,
    *,
    with_draft_probs: bool,
    enable_reduce_sampling: bool,
) -> dict[str, torch.Tensor | None]:
    num_draft_per_req = max_spec_len
    num_tokens = batch_size * num_draft_per_req
    num_draft_tokens = torch.full(
        (batch_size,),
        num_draft_per_req,
        dtype=torch.int32,
        device=device,
    )
    cu_num_draft_tokens = torch.cumsum(num_draft_tokens, dim=0)

    draft_token_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    draft_probs = None
    global_vocab = vocab_size
    if with_draft_probs:
        global_vocab = max(vocab_size, _WARMUP_VOCAB_SIZE)
        draft_probs = torch.rand(
            num_tokens,
            global_vocab,
            dtype=torch.float32,
            device=device,
        )

    if enable_reduce_sampling:
        prob_vocab = _WARMUP_SELECTED_VOCAB_SIZE
        global_vocab_size = (
            global_vocab if with_draft_probs else _WARMUP_SELECTED_VOCAB_SIZE
        )
    else:
        prob_vocab = global_vocab if with_draft_probs else vocab_size
        global_vocab_size = prob_vocab
    target_probs = torch.rand(
        num_tokens,
        prob_vocab,
        dtype=torch.float32,
        device=device,
    )
    target_indices = None
    if enable_reduce_sampling:
        target_indices = torch.randint(
            0,
            vocab_size,
            (num_tokens, prob_vocab),
            dtype=torch.int64,
            device=device,
        )

    bonus_token_ids = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
    recovered_token_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    uniform_probs = torch.full(
        (num_tokens,),
        0.5,
        dtype=torch.float32,
        device=device,
    )
    is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=device)
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        -1,
        dtype=torch.int32,
        device=device,
    )
    q = torch.full(
        (batch_size, prob_vocab),
        1.0,
        dtype=torch.float32,
        device=device,
    )

    return {
        "cu_num_draft_tokens": cu_num_draft_tokens,
        "draft_token_ids": draft_token_ids,
        "draft_probs": draft_probs,
        "target_probs": target_probs,
        "target_indices": target_indices,
        "bonus_token_ids": bonus_token_ids,
        "recovered_token_ids": recovered_token_ids,
        "uniform_probs": uniform_probs,
        "is_greedy": is_greedy,
        "output_token_ids": output_token_ids,
        "q": q,
        "global_vocab_size": global_vocab_size,
        "prob_vocab_size": prob_vocab,
    }


def _warm_sample_recovered_tokens_kernel(
    batch_size: int,
    max_spec_len: int,
    tensors: dict[str, torch.Tensor | None],
    *,
    with_draft_probs: bool,
    enable_reduce_sampling: bool,
    block_verify: bool,
) -> None:
    global_vocab_size = tensors["global_vocab_size"]
    assert isinstance(global_vocab_size, int)
    prob_vocab_size = tensors["prob_vocab_size"]
    assert isinstance(prob_vocab_size, int)

    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        tensors["recovered_token_ids"],
        tensors["cu_num_draft_tokens"],
        tensors["draft_token_ids"],
        tensors["draft_probs"],
        tensors["target_probs"],
        tensors["target_indices"],
        tensors["q"],
        prob_vocab_size,
        global_vocab_size,
        NO_DRAFT_PROBS=not with_draft_probs,
        BLOCK_VERIFY=block_verify,
        ENABLE_REDUCE_SAMPLING=enable_reduce_sampling,
        SUB_BLOCK=_SUB_BLOCK,
    )


def _warm_rejection_random_sample_kernel(
    batch_size: int,
    max_spec_len: int,
    block_size: int,
    grid: int,
    tensors: dict[str, torch.Tensor | None],
    *,
    with_draft_probs: bool,
    enable_reduce_sampling: bool,
    block_verify: bool,
) -> None:
    global_vocab_size = tensors["global_vocab_size"]
    assert isinstance(global_vocab_size, int)
    prob_vocab_size = tensors["prob_vocab_size"]
    assert isinstance(prob_vocab_size, int)
    uniform_probs = tensors["uniform_probs"]
    assert isinstance(uniform_probs, torch.Tensor)

    kernel_args = (
        tensors["output_token_ids"],
        tensors["cu_num_draft_tokens"],
        tensors["draft_token_ids"],
        tensors["draft_probs"],
        tensors["target_probs"],
        tensors["target_indices"],
        tensors["bonus_token_ids"],
        tensors["recovered_token_ids"],
        uniform_probs,
        tensors["is_greedy"],
        max_spec_len,
        prob_vocab_size,
        global_vocab_size,
        batch_size,
    )
    constexpr_kwargs = dict(
        NO_DRAFT_PROBS=not with_draft_probs,
        ENABLE_REDUCE_SAMPLING=enable_reduce_sampling,
        BLOCK_SIZE=block_size,
    )

    if block_verify:
        rejection_random_sample_block_verify_kernel[(grid,)](
            *kernel_args,
            SUB_BLOCK=_SUB_BLOCK,
            **constexpr_kwargs,
        )
    else:
        rejection_random_sample_kernel[(grid,)](
            *kernel_args,
            VOCAB_BLOCK_SIZE=_VOCAB_BLOCK_SIZE,
            **constexpr_kwargs,
        )


def _warm_greedy_rejection_kernels(
    batch_size: int,
    max_spec_len: int,
    block_size: int,
    grid: int,
    device: torch.device,
) -> None:
    num_draft_per_req = max_spec_len
    num_tokens = batch_size * num_draft_per_req
    num_draft_tokens_list = [num_draft_per_req] * batch_size
    cu_num_draft_tokens = torch.cumsum(
        torch.full((batch_size,), num_draft_per_req, dtype=torch.int32, device=device),
        dim=0,
    )
    draft_token_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    target_argmax = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    bonus_token_ids = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)

    output_all_greedy = torch.full(
        (batch_size, max_spec_len + 1),
        -1,
        dtype=torch.int32,
        device=device,
    )
    rejection_greedy_sample_with_triton(
        output_all_greedy,
        num_draft_tokens_list,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy=None,
        max_spec_len=max_spec_len,
        grid=grid,
        block_size=block_size,
    )

    output_mixed = torch.full(
        (batch_size, max_spec_len + 1),
        -1,
        dtype=torch.int32,
        device=device,
    )
    is_greedy = torch.zeros(batch_size, dtype=torch.bool, device=device)
    rejection_greedy_sample_with_triton(
        output_mixed,
        num_draft_tokens_list,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy=is_greedy,
        max_spec_len=max_spec_len,
        grid=grid,
        block_size=block_size,
    )


def _warm_rejection_random_path(
    device: torch.device,
    batch_size: int,
    max_spec_len: int,
    vocab_size: int,
    *,
    with_draft_probs: bool,
    enable_reduce_sampling: bool,
    block_verify: bool,
) -> None:
    grid, block_size = cal_grid_and_block_size(batch_size)
    tensors = _make_rejection_tensors(
        batch_size,
        max_spec_len,
        vocab_size,
        device,
        with_draft_probs=with_draft_probs,
        enable_reduce_sampling=enable_reduce_sampling,
    )
    _warm_sample_recovered_tokens_kernel(
        batch_size,
        max_spec_len,
        tensors,
        with_draft_probs=with_draft_probs,
        enable_reduce_sampling=enable_reduce_sampling,
        block_verify=block_verify,
    )
    _warm_rejection_random_sample_kernel(
        batch_size,
        max_spec_len,
        block_size,
        grid,
        tensors,
        with_draft_probs=with_draft_probs,
        enable_reduce_sampling=enable_reduce_sampling,
        block_verify=block_verify,
    )


@torch.inference_mode()
def spec_decode_triton_warmup(worker: NPUWorker) -> None:
    """JIT spec-decode Triton kernels before the first real request."""
    if not HAS_TRITON:
        return

    spec_config = worker.vllm_config.speculative_config
    if spec_config is None:
        return

    max_spec_len = spec_config.num_speculative_tokens
    if max_spec_len <= 0:
        return

    device = worker.device
    max_num_reqs = worker.scheduler_config.max_num_seqs
    vocab_size = min(worker.vllm_config.model_config.get_vocab_size(), _WARMUP_VOCAB_SIZE)

    enable_reduce_sampling = get_ascend_config().enable_reduce_sample

    req_batch_sizes = collect_warmup_req_batch_sizes(max_num_reqs)

    logger.info(
        "Warming up spec-decode Triton kernels: max_spec_len=%d, "
        "req_batch_sizes=%s, reduce_sample=%s",
        max_spec_len,
        req_batch_sizes,
        enable_reduce_sampling,
    )

    draft_prob_variants = (False, True)
    reduce_variants = (False, enable_reduce_sampling)

    for num_reqs in req_batch_sizes:
        _warm_prepare_inputs_padded_kernel(device, num_reqs)

    for batch_size in req_batch_sizes:
        _warm_expand_kernel(device, batch_size)
        grid, block_size = cal_grid_and_block_size(batch_size)
        _warm_greedy_rejection_kernels(
            batch_size,
            max_spec_len,
            block_size,
            grid,
            device,
        )
        for with_draft_probs in draft_prob_variants:
            for reduce_sampling in reduce_variants:
                if reduce_sampling and not enable_reduce_sampling:
                    continue
                # Match rejection_sampler: block verify needs draft_probs and
                # num_speculative_tokens >= 3.
                block_verify = max_spec_len >= 3 and with_draft_probs
                _warm_rejection_random_path(
                    device,
                    batch_size,
                    max_spec_len,
                    vocab_size,
                    with_draft_probs=with_draft_probs,
                    enable_reduce_sampling=reduce_sampling,
                    block_verify=block_verify,
                )

    if device.type == "npu":
        torch.npu.synchronize()
