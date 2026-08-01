# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Warm up rejection sampler Triton kernels used during speculative decoding."""

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

_REJECTION_TENSOR_DTYPES: dict[str, torch.dtype] = {
    "num_draft_tokens": torch.int32,
    "cu_num_draft_tokens": torch.int32,
    "draft_token_ids": torch.int32,
    "target_argmax": torch.int64,
    "draft_probs": torch.float32,
    "target_probs": torch.float32,
    "target_indices": torch.int32,
    "bonus_token_ids": torch.int32,
    "recovered_token_ids": torch.int32,
    "uniform_probs": torch.float32,
    "is_greedy": torch.bool,
    "output_token_ids": torch.int32,
    "q": torch.float32,
    "ori_target_probs": torch.float32,
}

_EPSILON = 1e-10


def collect_warmup_rejection_block_sizes(max_num_reqs: int) -> list[int]:
    """Batch sizes that cover every distinct ``BLOCK_SIZE`` from ``cal_grid_and_block_size``.

    Rejection/expand/greedy kernels key on ``BLOCK_SIZE`` (constexpr), not raw
    batch_size. Many batch sizes share the same ``BLOCK_SIZE``; we only need one
    representative batch per distinct value up to ``max_num_reqs``.
    """
    if max_num_reqs <= 0:
        return []

    block_size_to_batch: dict[int, int] = {}
    for batch_size in range(1, max_num_reqs + 1):
        _, block_size = cal_grid_and_block_size(batch_size)
        if block_size not in block_size_to_batch:
            block_size_to_batch[block_size] = batch_size

    # Ensure the largest batch is included (may be the only hit for top bucket).
    _, max_block_size = cal_grid_and_block_size(max_num_reqs)
    block_size_to_batch[max_block_size] = max_num_reqs
    return sorted(block_size_to_batch.values())


def collect_warmup_req_batch_sizes(max_num_reqs: int) -> list[int]:
    """Request batch sizes that cover distinct rejection/expand BLOCK_SIZE keys."""
    return collect_warmup_rejection_block_sizes(max_num_reqs)


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


# Warm up once per distinct BLOCK_SIZE; vllm_ascend currently uses int32
# for ``query_start_loc`` only.
def _warm_prepare_inputs_padded_kernel(
    device: torch.device,
    num_reqs: int,
) -> None:
    draft_lens = torch.ones(num_reqs, dtype=torch.int32, device=device)
    cu_num_draft_tokens = torch.cumsum(draft_lens, dim=0, dtype=torch.int32)
    valid_sampled_tokens_count = torch.ones(num_reqs, dtype=torch.int64, device=device)
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


# Warm up once per distinct BLOCK_SIZE. ``value_dtypes`` defaults mirror
# ``expand_batch_to_tokens``: float32 for temperature/top_p, int32 for top_k.
def _warm_expand_kernel(
    device: torch.device,
    batch_size: int,
    value_dtypes: tuple[torch.dtype, ...] = (torch.int32, torch.float32),
) -> None:
    cu_num_tokens = torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
    num_tokens = batch_size
    # expand_batch_to_tokens uses x.new_empty; temperature/top_p are float, top_k int.
    for value_dtype in value_dtypes:
        x = torch.zeros(batch_size, dtype=value_dtype, device=device)
        expanded_x = torch.empty(num_tokens, dtype=value_dtype, device=device)
        expand_triton(
            batch_size,
            expanded_x,
            x,
            cu_num_tokens,
            replace_from=-1,
            replace_to=0,
            max_num_tokens=MAX_SPEC_LEN,
        )

# recovered_token_ids: int32
# cu_num_draft_tokens: int32
# draft_token_ids: int32
# target_probs: float32
# target_indices: int32
# q: float32
def _make_rejection_tensors(
    batch_size: int,
    max_spec_len: int,
    vocab_size: int,
    device: torch.device,
    *,
    with_draft_probs: bool,
    enable_reduce_sampling: bool,
) -> dict[str, torch.Tensor | None]:
    dtypes = _REJECTION_TENSOR_DTYPES
    num_draft_per_req = max_spec_len
    num_tokens = batch_size * num_draft_per_req
    num_draft_tokens = torch.full(
        (batch_size,),
        num_draft_per_req,
        dtype=dtypes["num_draft_tokens"],
        device=device,
    )
    cu_num_draft_tokens = torch.cumsum(
        num_draft_tokens,
        dim=0,
        dtype=dtypes["cu_num_draft_tokens"],
    )

    draft_token_ids = torch.zeros(
        num_tokens,
        dtype=dtypes["draft_token_ids"],
        device=device,
    )
    draft_probs = None
    global_vocab = vocab_size
    if with_draft_probs:
        global_vocab = max(vocab_size, _WARMUP_VOCAB_SIZE)
        draft_probs = torch.rand(
            num_tokens,
            global_vocab,
            dtype=dtypes["draft_probs"],
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
        dtype=dtypes["target_probs"],
        device=device,
    )
    target_indices = None
    if enable_reduce_sampling:
        target_indices = torch.randint(
            0,
            vocab_size,
            (num_tokens, prob_vocab),
            dtype=dtypes["target_indices"],
            device=device,
        )

    bonus_token_ids = torch.zeros(
        batch_size,
        1,
        dtype=dtypes["bonus_token_ids"],
        device=device,
    )
    recovered_token_ids = torch.zeros(
        num_tokens,
        dtype=dtypes["recovered_token_ids"],
        device=device,
    )
    uniform_probs = torch.full(
        (num_tokens,),
        0.5,
        dtype=dtypes["uniform_probs"],
        device=device,
    )
    is_greedy = torch.zeros(
        batch_size,
        dtype=dtypes["is_greedy"],
        device=device,
    )
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        -1,
        dtype=dtypes["output_token_ids"],
        device=device,
    )
    q = torch.full(
        (batch_size, prob_vocab),
        1.0,
        dtype=dtypes["q"],
        device=device,
    )
    ori_target_probs = torch.rand(
        num_tokens,
        prob_vocab,
        dtype=dtypes["ori_target_probs"],
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
        "ori_target_probs": ori_target_probs,
        "global_vocab_size": global_vocab_size,
        "prob_vocab_size": prob_vocab,
    }


# Warm up twice for the two distinct ``global_vocab_size`` values.
# constexpr:
# NO_DRAFT_PROBS: both True and False
# ENABLE_REDUCE_SAMPLING: both True and False
# constexpr SUB_BLOCK=4 * 1024
def _warm_sample_recovered_tokens_kernel(
    batch_size: int,
    max_spec_len: int,
    tensors: dict[str, torch.Tensor | None],
    *,
    no_draft_probs: bool,
    enable_reduce_sampling: bool,
    block_verify: bool,
) -> None:
    logger.info(
        "Warming up sample recovered Triton kernels: no_draft_probs=%s, "
        "enable_reduce_sampling=%s, block_verify=%s",
        no_draft_probs,
        enable_reduce_sampling,
        block_verify,
    )

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
        NO_DRAFT_PROBS=no_draft_probs,
        BLOCK_VERIFY=block_verify,
        ENABLE_REDUCE_SAMPLING=enable_reduce_sampling,
        SUB_BLOCK=_SUB_BLOCK,
        multibuffer=False,
    )

# Warmup covers two ``global_vocab_size`` values and ``block_verify`` on/off.
# ``ENTROPY_VERIFY`` / ``NO_ORI_TARGET_PROBS`` follow ``rejection_sampler``:
# ori_target_probs is set only when ``enable_entropy_verify`` is True.
# other constexpr:
# ENABLE_REDUCE_SAMPLING=True
# SUB_BLOCK=4 * 1024
# EPSILON=1e-10
# POSTERIOR_THRESHOLD=rejection_sampler_config.posterior_threshold
# POSTERIOR_ALPHA=rejection_sampler_config.posterior_alpha
def _warm_rejection_random_sample_kernel(
    batch_size: int,
    max_spec_len: int,
    block_size: int,
    grid: int,
    tensors: dict[str, torch.Tensor | None],
    *,
    no_draft_probs: bool,
    enable_reduce_sampling: bool,
    block_verify: bool,
) -> None:
    global_vocab_size = tensors["global_vocab_size"]
    assert isinstance(global_vocab_size, int)
    prob_vocab_size = tensors["prob_vocab_size"]
    assert isinstance(prob_vocab_size, int)
    uniform_probs = tensors["uniform_probs"]
    assert isinstance(uniform_probs, torch.Tensor)
    ori_target_probs_tensor = tensors["ori_target_probs"]
    assert isinstance(ori_target_probs_tensor, torch.Tensor)
    draft_probs = None if no_draft_probs else tensors["draft_probs"]

    rejection_config = get_ascend_config().rejection_sampler_config
    using_entropy_verify = bool(rejection_config.enable_entropy_verify)
    posterior_threshold = float(rejection_config.posterior_threshold)
    posterior_alpha = float(rejection_config.posterior_alpha)

    # Match rejection_sample: ori_target_probs exists only when entropy verify
    # is enabled and original target logits are available.
    ori_target_probs = ori_target_probs_tensor if using_entropy_verify else None

    kernel_args = (
        tensors["output_token_ids"],
        tensors["cu_num_draft_tokens"],
        tensors["draft_token_ids"],
        draft_probs,
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
        ori_target_probs,
    )
    constexpr_kwargs = dict(
        NO_ORI_TARGET_PROBS=ori_target_probs is None,
        NO_DRAFT_PROBS=no_draft_probs,
        ENABLE_REDUCE_SAMPLING=enable_reduce_sampling,
        ENTROPY_VERIFY=using_entropy_verify,
        BLOCK_SIZE=block_size,
        POSTERIOR_THRESHOLD=posterior_threshold,
        POSTERIOR_ALPHA=posterior_alpha,
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
            VOCAB_BLOCK_SIZE=_VOCAB_BLOCK_SIZE,
            **constexpr_kwargs,
        )

# Warm up greedy rejection kernels for ``is_greedy=None`` and non-None paths.
# When ``num_draft_tokens`` are all 1 and ``is_greedy`` is None, a specialized
# kernel is used; otherwise ``rejection_greedy_sample_triton`` is launched.
def _warm_greedy_rejection_kernels(
    batch_size: int,
    max_spec_len: int,
    block_size: int,
    grid: int,
    device: torch.device,
    is_greedy: torch.Tensor | None,
) -> None:
    dtypes = _REJECTION_TENSOR_DTYPES
    num_draft_per_req = max_spec_len
    num_tokens = batch_size * num_draft_per_req
    num_draft_tokens_list = [num_draft_per_req] * batch_size
    cu_num_draft_tokens = torch.cumsum(
        torch.full(
            (batch_size,),
            num_draft_per_req,
            dtype=dtypes["num_draft_tokens"],
            device=device,
        ),
        dim=0,
        dtype=dtypes["cu_num_draft_tokens"],
    )
    draft_token_ids = torch.zeros(
        num_tokens,
        dtype=dtypes["draft_token_ids"],
        device=device,
    )
    target_argmax = torch.zeros(
        num_tokens,
        dtype=dtypes["target_argmax"],
        device=device,
    )
    bonus_token_ids = torch.zeros(
        batch_size,
        1,
        dtype=dtypes["bonus_token_ids"],
        device=device,
    )
    output_token_ids = torch.full(
        (batch_size, max_spec_len + 1),
        -1,
        dtype=dtypes["output_token_ids"],
        device=device,
    )
    rejection_greedy_sample_with_triton(
        output_token_ids,
        num_draft_tokens_list,
        cu_num_draft_tokens,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        is_greedy,
        max_spec_len,
        grid,
        block_size,
    )

# Warmup requirements:
# 1. Warm up twice for the two distinct ``global_vocab_size`` values.
# 2. Cover constexpr keys: global_vocab_size, NO_DRAFT_PROBS,
#    ENABLE_REDUCE_SAMPLING, BLOCK_SIZE, ENTROPY_VERIFY, and block_verify.
def _warm_rejection_random_path(
    device: torch.device,
    batch_size: int,
    max_spec_len: int,
    vocab_size: int,
) -> None:
    grid, block_size = cal_grid_and_block_size(batch_size)
    for enable_reduce_sampling in (False, True):
        for no_draft_probs in (False, True):
            with_draft_probs = not no_draft_probs
            # Match rejection_sampler: block verify needs draft_probs and spec_len >= 3.
            block_verify = max_spec_len >= 3 and with_draft_probs
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
                no_draft_probs=no_draft_probs,
                enable_reduce_sampling=enable_reduce_sampling,
                block_verify=block_verify,
            )
            _warm_rejection_random_sample_kernel(
                batch_size,
                max_spec_len,
                block_size,
                grid,
                tensors,
                no_draft_probs=no_draft_probs,
                enable_reduce_sampling=enable_reduce_sampling,
                block_verify=block_verify,
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

    enable_reduce_sampling = get_ascend_config().enable_reduce_sample

    req_batch_sizes = collect_warmup_req_batch_sizes(max_num_reqs)

    logger.info(
        "Warming up rejection sampler Triton kernels: max_spec_len=%d, "
        "req_batch_sizes=%s, reduce_sample=%s",
        max_spec_len,
        req_batch_sizes,
        enable_reduce_sampling,
    )

    for num_reqs in req_batch_sizes:
        _warm_prepare_inputs_padded_kernel(device, num_reqs)

    for batch_size in req_batch_sizes:
        _warm_expand_kernel(device, batch_size)
        grid, block_size = cal_grid_and_block_size(batch_size)
        for is_greedy in (
            None,
            torch.zeros(
                batch_size,
                dtype=_REJECTION_TENSOR_DTYPES["is_greedy"],
                device=device,
            ),
        ):
            _warm_greedy_rejection_kernels(
                batch_size,
                max_spec_len,
                block_size,
                grid,
                device,
                is_greedy,
            )
        _warm_rejection_random_path(
            device,
            batch_size,
            max_spec_len,
            vocab_size,
        )

    if device.type == "npu":
        torch.npu.synchronize()
