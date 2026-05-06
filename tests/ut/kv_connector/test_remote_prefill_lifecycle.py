#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/blob/main/tests/conftest.py
#
import copy

import vllm_ascend.envs as ascend_envs
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
from vllm.v1.request import RequestStatus

from vllm_ascend.core.laps_scheduler import LAPSRequestQueue
from vllm_ascend.core.recompute_scheduler import RecomputeScheduler
from tests.ut.kv_connector.utils import (assert_scheduler_empty,
                                         create_model_runner_output,
                                         create_request, create_scheduler,
                                         create_vllm_config)


def test_basic_lifecycle():
    """Test lifecycle of a remote prefill."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))
    START_FREE_BLOCK_QUEUE_SIZE = (
        scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks)

    request = create_request(request_id=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True,
                             block_size=BLOCK_SIZE)

    scheduler.add_request(request)
    request_id = request.request_id

    # STEP (1):
    # (1a): schedule()
    scheduler_output = scheduler.schedule()

    # Nothing running and empty scheduler output.
    assert len(scheduler.running) == 0
    assert len(scheduler_output.scheduled_new_reqs) == 0
    assert scheduler_output.scheduled_cached_reqs.num_reqs == 0
    assert len(scheduler_output.num_scheduled_tokens) == 0
    assert scheduler_output.total_num_scheduled_tokens == 0

    # Req waiting for KVs with no computed/scheduled toks ...
    assert len(scheduler.waiting) == 1
    assert request in scheduler.waiting
    assert (request.status == RequestStatus.WAITING_FOR_REMOTE_KVS)
    assert (request.num_computed_tokens == 0)

    # ... but should have (uncached) blocks allocated to it.
    block_pool = scheduler.kv_cache_manager.block_pool
    assert (block_pool.free_block_queue.num_free_blocks
            < START_FREE_BLOCK_QUEUE_SIZE)
    assert len(block_pool.cached_block_hash_to_block) == 0
    blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0].req_to_blocks[request_id]
    for block in blocks:
        assert block._block_hash is None

    # (1b): forward()
    model_runner_output = EMPTY_MODEL_RUNNER_OUTPUT

    # (1c): update_from_output()
    engine_core_outputs = scheduler.update_from_output(scheduler_output,
                                                       model_runner_output)
    assert not engine_core_outputs or not engine_core_outputs[0].outputs

    # STEP (2):
    # (2a): schedule(): nothing happens!
    scheduler_output = scheduler.schedule()
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 0

    # (2b): forward(): request finishes recv.
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    from vllm.v1.worker.kv_connector_model_runner_mixin import \
        KVConnectorOutput  # type: ignore  # noqa
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_recving=[request_id])

    # (2c): update_from_output():
    engine_core_outputs = scheduler.update_from_output(scheduler_output,
                                                       model_runner_output)
    assert len(scheduler.waiting) == 1
    assert (request_id in scheduler.finished_recving_kv_req_ids)

    # STEP (3):
    # (3a): schedule(): this should actually schedule.
    scheduler_output = scheduler.schedule()
    assert len(scheduler.running) == 1

    # Confirm the block are actually allocated.
    num_hashed_blocks = 0
    blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0].req_to_blocks[request_id]
    for block in blocks:
        assert block.ref_cnt == 1
        num_hashed_blocks += (1 if block._block_hash is not None else 0)
    assert num_hashed_blocks == NUM_EXTERNAL_FULL_BLOCKS

    # Confirm the rest of the prompt is scheduled in this step.
    scheduled_req = scheduler_output.scheduled_new_reqs[0]
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens[request_id]
    num_computed_tokens = scheduled_req.num_computed_tokens
    total_prompt_tokens = len(scheduled_req.prompt_token_ids)
    assert (num_scheduled_tokens == total_prompt_tokens - num_computed_tokens)

    # (3b): execute_model()
    model_runner_output = create_model_runner_output([request])
    # (3c): update_from_output()
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # Step (4): Hit EOS.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output([request], use_eos=True)
    engine_core_outputs = scheduler.update_from_output(scheduler_output,
                                                       model_runner_output)
    scheduler.schedule()

    assert_scheduler_empty(scheduler)


def test_no_spurious_prefix_caching():
    """
    With P/D, blocks can be allocated but uncomputed for
    multiple engine steps. This test confirms that we do
    not accidentally have cache hits against uncomputed
    blocks.
    """

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 and a half full external blocks.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    # Both of these requests have prompts like [1,1,1,1,1, ...]
    request_remote = create_request(
        request_id=1,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
        use_all_1s_for_prompt_tokens=True,
    )

    # Schedule the remote prefill request. This should not
    # cause any blocks to be cached.
    scheduler.add_request(request_remote)
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)
    assert len(scheduler.waiting) == 1

    remote_blocks = scheduler.kv_cache_manager.coordinator.single_type_managers[
        0].req_to_blocks[request_remote.request_id]

    # Remote blocks should not be cached.
    for block in remote_blocks:
        assert block.ref_cnt == 1
        assert block._block_hash is None


def test_full_block_prompt():
    """Test that we handle a prompt that is the full block size."""

    vllm_config = create_vllm_config()
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * NUM_EXTERNAL_FULL_BLOCKS)

    request = create_request(request_id=1,
                             num_tokens=NUM_TOKENS,
                             do_remote_prefill=True)

    scheduler.add_request(request)
    request_id = request.request_id

    # STEP (1): Initialize a recv.
    scheduler_output = scheduler.schedule()
    # All blocks should be allocated.
    num_blocks = len(scheduler.kv_cache_manager.coordinator.
                     single_type_managers[0].req_to_blocks[request_id])
    assert num_blocks == NUM_EXTERNAL_FULL_BLOCKS
    model_runner_output = EMPTY_MODEL_RUNNER_OUTPUT
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # # STEP (2): Recv.
    scheduler_output = scheduler.schedule()
    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    from vllm.v1.worker.kv_connector_model_runner_mixin import \
        KVConnectorOutput  # type: ignore  # noqa
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_recving=[request_id])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    assert len(scheduler.waiting) == 1
    assert (request_id in scheduler.finished_recving_kv_req_ids)

    # # STEP (3): Run as usual.
    scheduler_output = scheduler.schedule()

    # We need to recompute the final token of the prompt to generate
    # the first new token, so we should not have a new block.
    num_blocks = len(scheduler.kv_cache_manager.coordinator.
                     single_type_managers[0].req_to_blocks[request_id])
    assert num_blocks == NUM_EXTERNAL_FULL_BLOCKS
    assert (scheduler_output.scheduled_new_reqs[0].num_computed_tokens ==
            NUM_TOKENS - 1)
    assert (scheduler_output.num_scheduled_tokens[request_id] == 1)

    model_runner_output = create_model_runner_output([request])
    scheduler.update_from_output(scheduler_output, model_runner_output)

    # # Step (4): Hit EOS.
    scheduler_output = scheduler.schedule()
    model_runner_output = create_model_runner_output([request], use_eos=True)
    scheduler.schedule()

    assert_scheduler_empty(scheduler)


def test_resumed_waiting_request_with_attached_remote_blocks():
    """A resumed PD request must not rediscover attached remote KV blocks."""

    vllm_config = create_vllm_config()
    base_scheduler = create_scheduler(vllm_config)
    scheduler = RecomputeScheduler(
        vllm_config=vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        log_stats=True,
        block_size=vllm_config.cache_config.block_size,
        structured_output_manager=base_scheduler.structured_output_manager,
    )

    block_size = vllm_config.cache_config.block_size
    num_external_full_blocks = 2
    num_tokens = int(block_size * (num_external_full_blocks + 0.5))

    request = create_request(request_id=7,
                             num_tokens=num_tokens,
                             do_remote_prefill=True,
                             block_size=block_size)

    scheduler.add_request(request)
    request_id = request.request_id

    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)

    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    from vllm.v1.worker.kv_connector_model_runner_mixin import \
        KVConnectorOutput  # type: ignore  # noqa
    model_runner_output.kv_connector_output = KVConnectorOutput(
        finished_recving=[request_id])
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, model_runner_output)

    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output,
                                 create_model_runner_output([request]))

    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    attached_block_ids = tuple(
        block.block_id for block in manager.req_to_blocks[request_id]
    )
    cached_blocks = manager.num_cached_block[request_id]

    scheduler.running.remove(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
    request.num_external_computed_tokens = 0
    scheduler.waiting.prepend_request(request)

    scheduler_output = scheduler.schedule()

    assert request.status == RequestStatus.RUNNING
    assert request.request_id in scheduler_output.num_scheduled_tokens
    assert tuple(
        block.block_id for block in manager.req_to_blocks[request_id]
    ) == attached_block_ids
    assert manager.num_cached_block[request_id] == cached_blocks


def test_recompute_scheduler_uses_pd_aware_laps_queue(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SCHEDULING", "1")

    vllm_config = create_vllm_config()
    base_scheduler = create_scheduler(vllm_config)
    scheduler = RecomputeScheduler(
        vllm_config=vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        log_stats=True,
        block_size=vllm_config.cache_config.block_size,
        structured_output_manager=base_scheduler.structured_output_manager,
    )

    assert ascend_envs.VLLM_ASCEND_LAPS_SCHEDULING
    assert isinstance(scheduler.waiting, LAPSRequestQueue)

    request = create_request(request_id=9, num_tokens=32, do_remote_prefill=True)
    request.status = RequestStatus.PREEMPTED

    assert scheduler._should_bypass_laps_wait_window(request)


def test_recompute_scheduler_selects_schedulable_laps_subqueue(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SCHEDULING", "1")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "16")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_WINDOW_MS", "10")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_MAX_BATCH", "4")

    vllm_config = create_vllm_config()
    base_scheduler = create_scheduler(vllm_config)
    scheduler = RecomputeScheduler(
        vllm_config=vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        log_stats=True,
        block_size=vllm_config.cache_config.block_size,
        structured_output_manager=base_scheduler.structured_output_manager,
    )

    short_request = create_request(request_id=10, num_tokens=8)
    long_request = create_request(request_id=11, num_tokens=64)

    scheduler.waiting.add_request(short_request)
    scheduler.waiting.add_request(long_request)

    selected_queue = scheduler._select_waiting_queue_for_scheduling()
    assert selected_queue is not None
    assert selected_queue.peek_request() is long_request


def test_recompute_scheduler_falls_back_to_skipped_waiting_when_laps_waits(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SCHEDULING", "1")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "16")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_WINDOW_MS", "10")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_MAX_BATCH", "4")

    vllm_config = create_vllm_config()
    base_scheduler = create_scheduler(vllm_config)
    scheduler = RecomputeScheduler(
        vllm_config=vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        log_stats=True,
        block_size=vllm_config.cache_config.block_size,
        structured_output_manager=base_scheduler.structured_output_manager,
    )

    short_request = create_request(request_id=13, num_tokens=8)
    skipped_request = create_request(request_id=14, num_tokens=64, do_remote_prefill=True)
    skipped_request.status = RequestStatus.WAITING_FOR_REMOTE_KVS

    scheduler.waiting.add_request(short_request)
    scheduler.skipped_waiting.add_request(skipped_request)

    selected_queue = scheduler._select_waiting_queue_for_scheduling()
    assert selected_queue is scheduler.skipped_waiting
    assert selected_queue.peek_request() is skipped_request


def test_recompute_scheduler_preempted_request_is_forced_immediate(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SCHEDULING", "1")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "16")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_WINDOW_MS", "10")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_WAIT_MAX_BATCH", "4")

    vllm_config = create_vllm_config()
    base_scheduler = create_scheduler(vllm_config)
    scheduler = RecomputeScheduler(
        vllm_config=vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        log_stats=True,
        block_size=vllm_config.cache_config.block_size,
        structured_output_manager=base_scheduler.structured_output_manager,
    )

    request = create_request(request_id=12, num_tokens=8)
    scheduler.waiting.prepend_request(request, force_immediate=True)

    selected_queue = scheduler._select_waiting_queue_for_scheduling()
    assert selected_queue is not None
    assert selected_queue.peek_request() is request


def test_recompute_scheduler_applies_laps_budgeting(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SCHEDULING", "1")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "0")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0.25")

    vllm_config = create_vllm_config(max_num_batched_tokens=1024)
    base_scheduler = create_scheduler(vllm_config)
    scheduler = RecomputeScheduler(
        vllm_config=vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        log_stats=True,
        block_size=vllm_config.cache_config.block_size,
        structured_output_manager=base_scheduler.structured_output_manager,
    )

    short_request = create_request(request_id=15, num_tokens=64)
    long_request = create_request(request_id=16, num_tokens=800)

    scheduler.add_request(short_request)
    scheduler.add_request(long_request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[short_request.request_id] == 64
    assert output.num_scheduled_tokens[long_request.request_id] == 768
    waiting = scheduler.waiting
    assert isinstance(waiting, LAPSRequestQueue)
    assert waiting._last_short_reserved_tokens == 256
    assert waiting._last_short_actual_used_tokens == 64
    assert waiting._last_long_actual_used_tokens == 768


def test_recompute_scheduler_disables_laps_budgeting_under_priority_policy(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SCHEDULING", "1")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_THRESHOLD", "128")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_LONG_PREFILL_CAP", "256")
    monkeypatch.setenv("VLLM_ASCEND_LAPS_SHORT_RESERVED_RATIO", "0.25")

    vllm_config = create_vllm_config(max_num_batched_tokens=1024)
    vllm_config.scheduler_config.policy = "priority"
    base_scheduler = create_scheduler(vllm_config)
    base_scheduler.scheduler_config.policy = base_scheduler.vllm_config.scheduler_config.policy
    scheduler = RecomputeScheduler(
        vllm_config=vllm_config,
        kv_cache_config=base_scheduler.kv_cache_config,
        log_stats=True,
        block_size=vllm_config.cache_config.block_size,
        structured_output_manager=base_scheduler.structured_output_manager,
    )

    long_request = create_request(request_id=17, num_tokens=800)
    scheduler.add_request(long_request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens[long_request.request_id] == 800
    assert scheduler._laps_waiting_queue() is None or not scheduler._laps_long_budgeting_enabled()
