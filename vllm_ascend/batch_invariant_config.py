# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

SCHEDULER = "vllm_ascend.core.batch_invariant_scheduler.BatchInvariantScheduler"
ASYNC_SCHEDULER = "vllm_ascend.core.batch_invariant_scheduler.BatchInvariantAsyncScheduler"
BATCH_INVARIANT_BLOCK_SIZE = 128


def configure_batch_invariant(vllm_config) -> None:
    """Apply before compilation sizes and worker buffers are derived."""
    if not envs.VLLM_BATCH_INVARIANT:
        return

    if vllm_config.kv_transfer_config is not None:
        raise ValueError("Batch invariance does not support KV transfer with non-preemptive scheduling.")

    scheduler = vllm_config.scheduler_config
    scheduler.enable_chunked_prefill = False
    scheduler.long_prefill_token_threshold = 0
    vllm_config.cache_config.enable_prefix_caching = False
    # refresh_block_size no longer promotes the default after both prefix
    # caching and chunked prefill have been disabled. Standard Ascend attention
    # backends require 128-token blocks, including when a user supplied a size.
    vllm_config.cache_config.block_size = BATCH_INVARIANT_BLOCK_SIZE

    # A complete prompt must fit into one iteration. Update derived encoder
    # budgets too, since SchedulerConfig.__post_init__ has already run.
    scheduler.max_num_batched_tokens = max(scheduler.max_num_batched_tokens, vllm_config.model_config.max_model_len)
    if scheduler.max_num_scheduled_tokens is not None:
        scheduler.max_num_scheduled_tokens = max(
            scheduler.max_num_scheduled_tokens, vllm_config.model_config.max_model_len
        )
    scheduler.max_num_encoder_input_tokens = scheduler.max_num_batched_tokens
    scheduler.encoder_cache_size = scheduler.max_num_batched_tokens


def select_batch_invariant_scheduler(vllm_config) -> None:
    """Run after Ascend's specialized scheduler selection."""
    if not envs.VLLM_BATCH_INVARIANT:
        return
    scheduler = vllm_config.scheduler_config
    if scheduler.scheduler_cls not in (None, SCHEDULER, ASYNC_SCHEDULER):
        raise ValueError("Batch invariance does not support a custom or specialized scheduler.")
    scheduler.scheduler_cls = ASYNC_SCHEDULER if scheduler.async_scheduling else SCHEDULER
    logger.warning(
        "VLLM_BATCH_INVARIANT=1: Batch invariance disables chunked prefill, prefix caching and request preemption. "
        "block_size=%d, max_num_batched_tokens=%d; reserving generation KV capacity may reduce concurrency.",
        vllm_config.cache_config.block_size,
        scheduler.max_num_batched_tokens,
    )
