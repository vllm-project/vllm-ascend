# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up rejection sampler Triton kernels used during speculative decoding."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
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

# Keep dummy tensors small; JIT keys are driven by constexpr flags.
_WARMUP_VOCAB_SIZE = 1024
_WARMUP_SELECTED_VOCAB_SIZE = 256
_SUB_BLOCK = 4096
_VOCAB_BLOCK_SIZE = 512
_EPSILON = 1e-10


def collect_warmup_rejection_block_sizes(max_num_reqs: int) -> list[int]:
    """Return one representative batch_size per distinct ``BLOCK_SIZE``.

    Rejection / expand / greedy kernels specialize on ``BLOCK_SIZE`` from
    ``cal_grid_and_block_size``, not on the raw batch size.
    """
    if max_num_reqs <= 0:
        return []

    block_size_to_batch: dict[int, int] = {}
    for batch_size in range(1, max_num_reqs + 1):
        _, block_size = cal_grid_and_block_size(batch_size)
        if block_size not in block_size_to_batch:
            block_size_to_batch[block_size] = batch_size

    # Ensure the largest batch is included for the top BLOCK_SIZE bucket.
    _, max_block_size = cal_grid_and_block_size(max_num_reqs)
    block_size_to_batch[max_block_size] = max_num_reqs
    return sorted(block_size_to_batch.values())


# Backward-compatible aliases.
collect_warmup_req_batch_sizes = collect_warmup_rejection_block_sizes


def collect_warmup_batch_sizes(
    max_num_reqs: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> list[int]:
    del cudagraph_capture_sizes
    return collect_warmup_rejection_block_sizes(max_num_reqs)


def _is_ngram_spec_method(spec_config) -> bool:
    method = getattr(spec_config, "method", None)
    if method in ("ngram", "ngram_gpu"):
        return True
    use_ngram_gpu = getattr(spec_config, "use_ngram_gpu", None)
    return bool(callable(use_ngram_gpu) and use_ngram_gpu())


def _collect_no_draft_probs_values(spec_config, pipeline_parallel_size: int) -> list[bool]:
    """Return ``NO_DRAFT_PROBS`` values that match the live rejection_sample path.

    ``model_runner_v1._sample`` passes ``draft_probs=None`` when PP==1.
    ngram never provides draft probs. Model-based drafts may supply probs only
    when PP>1.
    """
    if _is_ngram_spec_method(spec_config) or pipeline_parallel_size <= 1:
        return [True]
    return [False, True]


def _make_rejection_tensors(
    batch_size: int,
    max_spec_len: int,
    vocab_size: int,
    device: torch.device,
    *,
    with_draft_probs: bool,
    enable_reduce_sampling: bool,
) -> dict[str, Any]:
    """Build dummy tensors for recovered / random / greedy warmup launches."""
    num_tokens = batch_size * max_spec_len
    cu_num_draft_tokens = torch.cumsum(
        torch.full((batch_size,), max_spec_len, dtype=torch.int32, device=device),
        dim=0,
        dtype=torch.int32,
    )

    if with_draft_probs:
        global_vocab = max(vocab_size, _WARMUP_VOCAB_SIZE)
        draft_probs = torch.rand(
            num_tokens,
            global_vocab,
            dtype=torch.float32,
            device=device,
        )
    else:
        global_vocab = vocab_size
        draft_probs = None

    if enable_reduce_sampling:
        prob_vocab = _WARMUP_SELECTED_VOCAB_SIZE
        global_vocab_size = global_vocab if with_draft_probs else _WARMUP_SELECTED_VOCAB_SIZE
        target_indices = torch.randint(
            0,
            vocab_size,
            (num_tokens, prob_vocab),
            dtype=torch.int32,
            device=device,
        )
    else:
        prob_vocab = global_vocab if with_draft_probs else vocab_size
        global_vocab_size = prob_vocab
        target_indices = None

    return {
        "cu_num_draft_tokens": cu_num_draft_tokens,
        "draft_token_ids": torch.zeros(num_tokens, dtype=torch.int32, device=device),
        "draft_probs": draft_probs,
        "target_probs": torch.rand(num_tokens, prob_vocab, dtype=torch.float32, device=device),
        "target_indices": target_indices,
        "bonus_token_ids": torch.zeros(batch_size, 1, dtype=torch.int32, device=device),
        "recovered_token_ids": torch.zeros(num_tokens, dtype=torch.int32, device=device),
        "uniform_probs": torch.full((num_tokens,), 0.5, dtype=torch.float32, device=device),
        "is_greedy": torch.zeros(batch_size, dtype=torch.bool, device=device),
        "output_token_ids": torch.full(
            (batch_size, max_spec_len + 1),
            -1,
            dtype=torch.int32,
            device=device,
        ),
        "q": torch.full((batch_size, prob_vocab), 1.0, dtype=torch.float32, device=device),
        "ori_target_probs": torch.rand(
            num_tokens,
            prob_vocab,
            dtype=torch.float32,
            device=device,
        ),
        "target_argmax": torch.zeros(num_tokens, dtype=torch.int64, device=device),
        "global_vocab_size": global_vocab_size,
        "prob_vocab_size": prob_vocab,
    }


def _warm_prepare_inputs(device: torch.device, num_reqs: int) -> None:
    """Warm ``prepare_inputs_padded_kernel`` once.

    ``BLOCK_SIZE`` is fixed at ``_PREPARE_INPUTS_BLOCK_SIZE`` and ``num_reqs`` is
    ``do_not_specialize``, so a single launch covers all request counts.
    """
    num_blocks = triton.cdiv(num_reqs, _PREPARE_INPUTS_BLOCK_SIZE)
    grid_size = min(num_blocks, get_vectorcore_num())
    grid = (max(grid_size, 1),)

    cu_num_draft_tokens = torch.cumsum(
        torch.ones(num_reqs, dtype=torch.int32, device=device),
        dim=0,
        dtype=torch.int32,
    )
    valid_sampled_tokens_count = torch.ones(num_reqs, dtype=torch.int64, device=device)
    query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32, device=device)
    token_indices_to_sample = torch.empty(num_reqs, dtype=torch.int32, device=device)
    num_rejected_tokens_gpu = torch.empty(num_reqs, dtype=torch.int32, device=device)

    prepare_inputs_padded_kernel[grid](
        cu_num_draft_tokens,
        valid_sampled_tokens_count,
        query_start_loc,
        token_indices_to_sample,
        num_rejected_tokens_gpu,
        num_reqs,
        BLOCK_SIZE=_PREPARE_INPUTS_BLOCK_SIZE,
    )


def _warm_expand(device: torch.device, batch_size: int) -> None:
    """Warm expand kernel for one ``BLOCK_SIZE``, covering int32 and float32 dtypes."""
    cu_num_tokens = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
    # temperature/top_p use float32; top_k uses int32.
    for value_dtype in (torch.int32, torch.float32):
        x = torch.zeros(batch_size, dtype=value_dtype, device=device)
        expanded_x = torch.empty(batch_size, dtype=value_dtype, device=device)
        expand_triton(
            batch_size,
            expanded_x,
            x,
            cu_num_tokens,
            replace_from=-1,
            replace_to=0,
            max_num_tokens=MAX_SPEC_LEN,
        )


def _warm_greedy(
    batch_size: int,
    max_spec_len: int,
    block_size: int,
    grid: int,
    device: torch.device,
    is_greedy: torch.Tensor | None,
) -> None:
    """Warm greedy rejection for ``is_greedy=None`` and non-None paths."""
    tensors = _make_rejection_tensors(
        batch_size,
        max_spec_len,
        _WARMUP_VOCAB_SIZE,
        device,
        with_draft_probs=False,
        enable_reduce_sampling=False,
    )
    rejection_greedy_sample_with_triton(
        tensors["output_token_ids"],
        [max_spec_len] * batch_size,
        tensors["cu_num_draft_tokens"],
        tensors["draft_token_ids"],
        tensors["target_argmax"],
        tensors["bonus_token_ids"],
        is_greedy,
        max_spec_len,
        grid,
        block_size,
    )


def _warm_sample_recovered(
    batch_size: int,
    max_spec_len: int,
    tensors: dict[str, Any],
    *,
    no_draft_probs: bool,
    enable_reduce_sampling: bool,
) -> None:
    """Warm ``sample_recovered_tokens_kernel`` for one constexpr combination."""
    sample_recovered_tokens_kernel[(batch_size, max_spec_len)](
        tensors["recovered_token_ids"],
        tensors["cu_num_draft_tokens"],
        tensors["draft_token_ids"],
        tensors["draft_probs"],
        tensors["target_probs"],
        tensors["target_indices"],
        tensors["q"],
        tensors["prob_vocab_size"],
        tensors["global_vocab_size"],
        NO_DRAFT_PROBS=no_draft_probs,
        ENABLE_REDUCE_SAMPLING=enable_reduce_sampling,
        VOCAB_BLOCK_SIZE=_VOCAB_BLOCK_SIZE,
        SUB_BLOCK=_SUB_BLOCK,
        multibuffer=False,
    )


def _warm_rejection_random(
    batch_size: int,
    max_spec_len: int,
    block_size: int,
    grid: int,
    tensors: dict[str, Any],
    *,
    no_draft_probs: bool,
    enable_reduce_sampling: bool,
    block_verify: bool,
) -> None:
    """Warm random or block-verify rejection kernel for one constexpr combination."""
    rejection_config = get_ascend_config().rejection_sampler_config
    using_entropy_verify = bool(rejection_config.enable_entropy_verify)
    # Match rejection_sample: ori_target_probs only when entropy verify is enabled.
    ori_target_probs = tensors["ori_target_probs"] if using_entropy_verify else None
    draft_probs = None if no_draft_probs else tensors["draft_probs"]

    kernel_args = (
        tensors["output_token_ids"],
        tensors["cu_num_draft_tokens"],
        tensors["draft_token_ids"],
        draft_probs,
        tensors["target_probs"],
        tensors["target_indices"],
        tensors["bonus_token_ids"],
        tensors["recovered_token_ids"],
        tensors["uniform_probs"],
        tensors["is_greedy"],
        max_spec_len,
        tensors["prob_vocab_size"],
        tensors["global_vocab_size"],
        batch_size,
        ori_target_probs,
    )
    constexpr_kwargs = dict(
        NO_ORI_TARGET_PROBS=ori_target_probs is None,
        NO_DRAFT_PROBS=no_draft_probs,
        ENABLE_REDUCE_SAMPLING=enable_reduce_sampling,
        ENTROPY_VERIFY=using_entropy_verify,
        BLOCK_SIZE=block_size,
        POSTERIOR_THRESHOLD=float(rejection_config.posterior_threshold),
        POSTERIOR_ALPHA=float(rejection_config.posterior_alpha),
        SUB_BLOCK=_SUB_BLOCK,
        EPSILON=_EPSILON,
    )

    if block_verify:
        rejection_random_sample_block_verify_kernel[(grid,)](
            *kernel_args,
            **constexpr_kwargs,
        )
    else:
        rejection_random_sample_kernel[(grid,)](
            *kernel_args,
            None,  # synthetic_conditional_rates_ptr (unused unless SYNTHETIC_MODE)
            VOCAB_BLOCK_SIZE=_VOCAB_BLOCK_SIZE,
            SYNTHETIC_MODE=False,
            **constexpr_kwargs,
        )


@torch.inference_mode()
def rejection_sampler_triton_warmup(worker: NPUWorker) -> None:
    """JIT rejection sampler Triton kernels before the first spec-decode request."""
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

    ascend_config = get_ascend_config()
    enable_reduce_sampling = bool(ascend_config.enable_reduce_sample)
    # Match rejection_sample: block verify needs config and max_spec_len >= 3.
    block_verify = max_spec_len >= 3 and bool(ascend_config.rejection_sampler_config.enable_block_verify)
    no_draft_probs_values = _collect_no_draft_probs_values(
        spec_config,
        worker.vllm_config.parallel_config.pipeline_parallel_size,
    )
    req_batch_sizes = collect_warmup_rejection_block_sizes(max_num_reqs)

    # Fixed BLOCK_SIZE: warm once.
    _warm_prepare_inputs(device, max(max_num_reqs, 1))

    # No cal_grid_and_block_size BLOCK_SIZE: warm once per constexpr combo.
    for no_draft_probs in no_draft_probs_values:
        tensors = _make_rejection_tensors(
            1,
            max_spec_len,
            vocab_size,
            device,
            with_draft_probs=not no_draft_probs,
            enable_reduce_sampling=enable_reduce_sampling,
        )
        _warm_sample_recovered(
            1,
            max_spec_len,
            tensors,
            no_draft_probs=no_draft_probs,
            enable_reduce_sampling=enable_reduce_sampling,
        )

    # expand / greedy / random specialize on BLOCK_SIZE from batch_size.
    for batch_size in req_batch_sizes:
        _warm_expand(device, batch_size)

        grid, block_size = cal_grid_and_block_size(batch_size)
        for is_greedy in (
            None,
            torch.zeros(batch_size, dtype=torch.bool, device=device),
        ):
            _warm_greedy(
                batch_size,
                max_spec_len,
                block_size,
                grid,
                device,
                is_greedy,
            )

        for no_draft_probs in no_draft_probs_values:
            tensors = _make_rejection_tensors(
                batch_size,
                max_spec_len,
                vocab_size,
                device,
                with_draft_probs=not no_draft_probs,
                enable_reduce_sampling=enable_reduce_sampling,
            )
            _warm_rejection_random(
                batch_size,
                max_spec_len,
                block_size,
                grid,
                tensors,
                no_draft_probs=no_draft_probs,
                enable_reduce_sampling=enable_reduce_sampling,
                block_verify=block_verify,
            )
