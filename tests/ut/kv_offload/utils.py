# SPDX-License-Identifier: Apache-2.0
# This code is from: https://github.com/vllm-project/vllm/tests/v1/kv_connector/unit/utils.py
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from typing import Any

import torch
from vllm import SamplingParams
from vllm.config import CacheConfig, DeviceConfig, KVTransferConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

EOS_TOKEN_ID = 50256


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
    assert len(scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks) == 0
    assert len(scheduler.kv_cache_manager.coordinator.single_type_managers[0].num_cached_block) == 0
    num_free_blocks = scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks
    assert num_free_blocks == (scheduler.kv_cache_manager.block_pool.num_gpu_blocks - 1)

    for block in scheduler.kv_cache_manager.block_pool.blocks:
        assert block.ref_cnt == 0


def create_vllm_config(
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 1024,
    block_size: int = 128,
) -> VllmConfig:
    """Initialize VllmConfig For Testing."""
    fake_weight_path = os.path.join(os.path.dirname(__file__), "..", "_fake_weight")
    model_config = ModelConfig(
        model=fake_weight_path,
        skip_tokenizer_init=True,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_num_batched_tokens,
        enable_chunked_prefill=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    kv_transfer_config = KVTransferConfig(kv_connector="MooncakeConnector", kv_role="kv_both")
    return VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig("cpu"),
    )


def create_scheduler(
    vllm_config: VllmConfig,
    num_blocks: int = 10000,
) -> Scheduler:
    """Initialize Scheduler For Testing."""
    block_size = vllm_config.cache_config.block_size
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float16)
            )
        ],
    )
    vllm_config.cache_config.num_gpu_blocks = num_blocks

    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        log_stats=True,
        block_size=block_size,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


_none_hash_initialized = False


def create_request(
    request_id: int,
    num_tokens: int = 10,
    max_tokens: int = 128,
    do_remote_decode: bool = False,
    do_remote_prefill: bool = False,
    num_remote_blocks: int = 3,
    block_size: int = 16,
) -> Request:
    """Make dummy request for testing."""
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    kv_transfer_params: dict[str, Any] | None = None

    if do_remote_decode:
        assert not do_remote_prefill
        kv_transfer_params = dict(
            do_remote_prefill=False,
            do_remote_decode=True,
            transfer_id=f"transfer-{request_id}",
        )
    elif do_remote_prefill:
        kv_transfer_params = dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_engine_id="my-engine-id",
            remote_block_ids=list(range(num_remote_blocks)),
            remote_host="my-host",
            remote_port=1234,
            remote_bootstrap_addr="my-bootstrap",
            transfer_id=f"transfer-{request_id}",
            remote_tp_size=1,
            remote_pcp_size=1,
            remote_dcp_size=1,
        )

    max_tokens = 1 if do_remote_decode else max_tokens
    sampling_params = SamplingParams(max_tokens=max_tokens)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)

    prompt_token_ids = [i * request_id for i in range(num_tokens)]

    block_hasher = get_request_block_hasher(block_size, sha256)

    req = Request(
        request_id=f"id-{request_id}",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=block_hasher,
    )
    req.kv_transfer_params = kv_transfer_params
    return req


def create_model_runner_output(
    reqs: list[Request],
    finished_sending: set[str] | None = None,
    finished_recving: set[str] | None = None,
    use_eos: bool = False,
) -> ModelRunnerOutput:
    """Make dummy model runner output for testing."""

    req_ids = [req.request_id for req in reqs]
    req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

    sampled_token = EOS_TOKEN_ID if use_eos else 0
    sampled_token_ids = [[sampled_token] for _ in req_ids]

    kv_connector_output = (
        None
        if (finished_sending is None and finished_recving is None)
        else KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
        )
    )

    model_runner_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=kv_connector_output,
    )

    return model_runner_output
