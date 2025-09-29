# SPDX-License-Identifier: Apache-2.0
# This code is from: https://github.com/vllm-project/vllm/tests/v1/kv_connector/unit/utils.py
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from typing import Any, Optional

import torch
from vllm import SamplingParams
from vllm.config import (CacheConfig, DeviceConfig, KVTransferConfig,
                         ModelConfig, SchedulerConfig, VllmConfig)
from vllm.utils import sha256
from vllm.v1.core.kv_cache_utils import (get_request_block_hasher,
                                         init_none_hash)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

EOS_TOKEN_ID = 50256
os.environ["VLLM_USE_V1"] = "1"


def assert_scheduler_empty(scheduler: Scheduler):
    """Confirm the scheduler is "empty" - i.e. no leaks."""
    # Scheduler Metadata.
    assert len(scheduler.requests) == 0
    assert len(scheduler.waiting) == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.finished_req_ids) == 0
    assert len(scheduler.finished_recving_kv_req_ids) == 0

    # EncoderCacheManager.
    assert len(scheduler.encoder_cache_manager.freed) == 0
    assert len(scheduler.encoder_cache_manager.cached) == 0

    # KVCache Manager.
    assert len(scheduler.kv_cache_manager.coordinator.single_type_managers[0].
               req_to_blocks) == 0
    assert len(scheduler.kv_cache_manager.coordinator.single_type_managers[0].
               num_cached_block) == 0
    num_free_blocks = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks)
    assert num_free_blocks == (
        scheduler.kv_cache_manager.block_pool.num_gpu_blocks - 1)

    # NOTE(rob): just the ref count on blocks will be 0. The hash
    # value, etc will remain since we lazily evict for prefix cache.
    for block in scheduler.kv_cache_manager.block_pool.blocks:
        assert block.ref_cnt == 0


def create_vllm_config(
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 1024,
    block_size: int = 128,
) -> VllmConfig:
    """Initialize VllmConfig For Testing."""
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_num_batched_tokens,
    )
    fake_weight_path = os.path.join(os.path.dirname(__file__), "..",
                                    "fake_weight")
    model_config = ModelConfig(
        model=fake_weight_path,
        skip_tokenizer_init=True,
    )
    # Cache config, optionally force APC
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector="LLMDataDistCMgrConnector",
        kv_role="kv_both",
        kv_connector_module_path=
        "vllm_ascend.distributed.llmdatadist_c_mgr_connector")
    return VllmConfig(scheduler_config=scheduler_config,
                      model_config=model_config,
                      cache_config=cache_config,
                      kv_transfer_config=kv_transfer_config,
                      device_config=DeviceConfig("cpu"))


def create_scheduler(
    vllm_config: VllmConfig,
    num_blocks: int = 10000,
) -> Scheduler:
    """Initialize Scheduler For Testing."""
    block_size = vllm_config.cache_config.block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,  # A large number of blocks to hold all requests
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(['layer'],
                             FullAttentionSpec(block_size, 1, 1, torch.float16,
                                               False))
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


_none_hash_initialized = False


def create_request(
    request_id: int,
    num_tokens: int = 10,
    max_tokens: int = 128,
    do_remote_decode: bool = False,
    do_remote_prefill: bool = False,
    use_all_1s_for_prompt_tokens: bool = False,
    num_remote_blocks: int = 3,
    block_size: int = 16,
) -> Request:
    """Make dummy request for testing."""
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    block_hasher = get_request_block_hasher(block_size, sha256)

    kv_transfer_params: Optional[dict[str, Any]] = None

    if do_remote_decode:
        assert not do_remote_prefill
        kv_transfer_params = dict(do_remote_prefill=False,
                                  do_remote_decode=True)
    elif do_remote_prefill:
        kv_transfer_params = dict(do_remote_prefill=True,
                                  do_remote_decode=False,
                                  remote_engine_id="my-engine-id",
                                  remote_block_ids=list(
                                      range(num_remote_blocks)),
                                  remote_host="my-host",
                                  remote_port=1234,
                                  remote_tp_size=1)

    max_tokens = 1 if do_remote_decode else max_tokens
    sampling_params = SamplingParams(max_tokens=max_tokens)

    if use_all_1s_for_prompt_tokens:
        prompt_token_ids = [1] * num_tokens
    else:
        prompt_token_ids = [i * request_id for i in range(num_tokens)]

    req = Request(
        request_id=f"id-{request_id}",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=[],
        eos_token_id=EOS_TOKEN_ID,
        block_hasher=block_hasher,
    )
    req.kv_transfer_params = kv_transfer_params
    return req


def create_model_runner_output(
    reqs: list[Request],
    finished_sending: Optional[list[str]] = None,
    finished_recving: Optional[list[str]] = None,
    use_eos: bool = False,
) -> ModelRunnerOutput:
    """Make dummy model runner output for testing."""

    # Make request data.
    req_ids = [req.request_id for req in reqs]
    req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

    # Make sampled tokens.
    sampled_token = EOS_TOKEN_ID if use_eos else 0
    sampled_token_ids = [[sampled_token] for _ in req_ids]

    # Make output data structure.
    extra_args = {}
    from vllm.v1.worker.kv_connector_model_runner_mixin import \
        KVConnectorOutput  # type: ignore  # noqa
    kv_connector_output = KVConnectorOutput(finished_sending=finished_sending,
                                            finished_recving=finished_recving)
    extra_args = {"kv_connector_output": kv_connector_output}

    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
        **extra_args,
    )

    return model_runner_output
