#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

"""Multi-rank sleep/wake coverage for the lightweight HCCL lifecycle.

The test drives a real TP=2 engine through active -> suspend -> resume:
the lightweight policy keeps the HCCL communicators alive across sleep,
while the destroy policy exercises the destroy/recreate fallback.  In both
cases the generated tokens after wake must match the cold-start output.

A request issued while the engine is asleep must be queued (not silently
dropped) and produce the same tokens once the engine wakes up.
"""

import asyncio
import inspect
import os

import pytest
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

# Small text checkpoint available on the two-card CI runners; overridable for
# local runs, matching tests/e2e/pull_request/one_card/rlhf/conftest.py.
# collective_rpc probes ship functions to the workers, like the DP pause test.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

MODEL = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TP_SIZE = 2
PROMPT = "The capital of France is"
MAX_TOKENS = 16


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _hccl_stats(worker) -> dict:
    return worker.sleep_wakeup_manager.hccl.suspend_stats.as_dict()


def _hccl_state(worker) -> str:
    return worker.sleep_wakeup_manager.hccl.state.value


def _engine_args(policy: str) -> AsyncEngineArgs:
    return AsyncEngineArgs(
        model=MODEL,
        tensor_parallel_size=TP_SIZE,
        enable_sleep_mode=True,
        enforce_eager=True,
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        additional_config={
            "weight_nz_mode": 0,
            "rl_config": {
                "enabled": True,
                "sleep_mode_extra_cleanup": True,
                "sleep_hccl_policy": policy,
            },
        },
    )


async def _generate(engine: AsyncLLM, request_id: str) -> list[int]:
    token_ids: list[int] = []
    async for output in engine.generate(
        request_id=request_id,
        prompt=PROMPT,
        sampling_params=SamplingParams(temperature=0, max_tokens=MAX_TOKENS),
    ):
        if output.outputs:
            token_ids = list(output.outputs[0].token_ids)
    return token_ids


async def _sleep_wake_cycles(policy: str, cycles: int) -> None:
    engine = AsyncLLM.from_engine_args(_engine_args(policy))
    suspended_handle_sets: list[tuple] = []
    try:
        baseline = await _generate(engine, f"{policy}-baseline")
        assert baseline, f"{policy}: baseline generation returned no tokens"
        for cycle in range(cycles):
            await _maybe_await(engine.sleep(level=1))
            assert await _maybe_await(engine.is_sleeping())
            if policy == "lightweight":
                stats = await engine.engine_core.collective_rpc_async(_hccl_stats)
                assert all(item["suspend_failed"] == 0 for item in stats), stats
                assert all(item["suspend_ok"] > 0 for item in stats), stats
                handles = tuple(tuple(item["suspended_handles"]) for item in stats)
                assert all(handles), "expected at least one suspended HCCL communicator per rank"
                suspended_handle_sets.append(handles)
            await _maybe_await(engine.wake_up())
            assert not await _maybe_await(engine.is_sleeping())
            if policy == "lightweight":
                stats = await engine.engine_core.collective_rpc_async(_hccl_stats)
                assert all(not item["suspended_handles"] for item in stats), stats
                states = await engine.engine_core.collective_rpc_async(_hccl_state)
                assert all(state == "idle" for state in states), states
            after = await _generate(engine, f"{policy}-cycle-{cycle}")
            assert after == baseline, f"{policy}: tokens diverged after wake cycle {cycle}: {after} != {baseline}"
        if suspended_handle_sets:
            assert len(set(suspended_handle_sets)) == 1, "communicator handles changed across sleep cycles"
    finally:
        engine.shutdown()


async def _request_while_sleeping() -> None:
    engine = AsyncLLM.from_engine_args(_engine_args("lightweight"))
    try:
        baseline = await _generate(engine, "queued-baseline")
        assert baseline
        await _maybe_await(engine.sleep(level=1))
        queued = asyncio.create_task(_generate(engine, "queued-during-sleep"))
        await asyncio.sleep(1.0)
        assert not queued.done(), "a request issued during sleep must be queued, not executed"
        await _maybe_await(engine.wake_up())
        after = await asyncio.wait_for(queued, timeout=180)
        assert after == baseline, f"queued request diverged after wake: {after} != {baseline}"
    finally:
        engine.shutdown()


def _disable_suspend_on_first_rank(worker) -> bool:
    rank = getattr(worker, "rank", None)
    if rank is None:
        rank = getattr(worker, "local_rank", 0)
    if rank == 0:
        worker.sleep_wakeup_manager.hccl._suspend_backend.available = lambda: False
    return True


async def _partial_rank_suspend_falls_back() -> None:
    engine = AsyncLLM.from_engine_args(_engine_args("lightweight"))
    try:
        baseline = await _generate(engine, "partial-baseline")
        assert baseline
        assert all(await engine.engine_core.collective_rpc_async(_disable_suspend_on_first_rank))
        await _maybe_await(engine.sleep(level=1))
        stats = await engine.engine_core.collective_rpc_async(_hccl_stats)
        assert all(item["fallbacks"] >= 1 for item in stats), stats
        await _maybe_await(engine.wake_up())
        after = await _generate(engine, "partial-after")
        assert after == baseline, f"partial-rank fallback diverged: {after} != {baseline}"
    finally:
        engine.shutdown()


async def _shutdown_while_asleep() -> None:
    engine = AsyncLLM.from_engine_args(_engine_args("lightweight"))
    baseline = await _generate(engine, "shutdown-baseline")
    assert baseline
    await _maybe_await(engine.sleep(level=1))
    engine.shutdown()
    replacement = AsyncLLM.from_engine_args(_engine_args("lightweight"))
    try:
        after = await _generate(replacement, "shutdown-replacement")
        assert after == baseline, f"engine restarted after sleeping shutdown diverged: {after} != {baseline}"
    finally:
        replacement.shutdown()


def test_lightweight_sleep_wake_preserves_tokens():
    asyncio.run(_sleep_wake_cycles("lightweight", cycles=2))


def test_destroy_sleep_wake_preserves_tokens():
    asyncio.run(_sleep_wake_cycles("destroy", cycles=1))


def test_request_during_sleep_is_queued_until_wake():
    asyncio.run(_request_while_sleeping())


def test_shutdown_while_asleep_releases_resources():
    asyncio.run(_shutdown_while_asleep())


def test_partial_rank_capability_failure_falls_back_for_all_ranks():
    asyncio.run(_partial_rank_suspend_falls_back())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
