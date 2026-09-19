#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm PR #52914
#   (tests/v1/distributed/test_async_llm_dp.py::test_dp_pause_completion_implies_device_idle)
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
#

"""NPU guard for "a resolved pause promise means an idle device" (vLLM #52914).

In data parallel the pause path itself launches work: ``_pause_complete`` forces
the every-N-steps dummy-batch burst so every rank reaches the pause consensus,
and the last dummy batch is launched inside the consensus iteration, where
``has_work()`` cannot see it. Upstream therefore puts a device barrier in
``EngineCoreProc._finish_pause`` before the caches are reset and the pause future
resolves. Ascend only gets that barrier if ``NPUWorker.synchronize_device()``
really waits, which is what this test checks end to end.

The upstream test inflates the dummy batches with ``torch.cuda._sleep``; that
call does not exist on NPU, so ``npu_stream_busy`` queues real matmuls instead.
"""

import asyncio
import os
from contextlib import ExitStack

import pytest
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from tests.e2e.pull_request.one_card.rlhf.state_transitions.npu_stream_busy import (
    enqueue_npu_busy_work,
    npu_stream_idle,
)

DP_SIZE = 2
# The tiny non-MoE model upstream uses for the same DP pause test: this test
# exercises the pause/DP machinery, not model quality, so it needs no real
# weights. VLLM_TEST_MODEL overrides it, as in
# tests/e2e/pull_request/one_card/rlhf/conftest.py.
DP_PAUSE_MODEL = os.environ.get("VLLM_TEST_MODEL", "hmellor/tiny-random-LlamaForCausalLM")
DP_PAUSE_PROMPT = "This is a test of data parallel pause"


def _has_pause_device_barrier() -> bool:
    """Whether the installed vLLM carries the #52914 pause device barrier."""
    try:
        from vllm.v1.engine.core import EngineCoreProc
    except ImportError:  # pragma: no cover - vLLM is a hard dependency of the plugin
        return False
    return hasattr(EngineCoreProc, "_finish_pause")


def _get_dp_pause_engine_args() -> AsyncEngineArgs:
    return AsyncEngineArgs(
        model=DP_PAUSE_MODEL,
        enforce_eager=True,
        data_parallel_size=DP_SIZE,
        data_parallel_backend="mp",
        max_model_len=512,
        gpu_memory_utilization=0.5,
    )


async def _pause_an_inflated_dp_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    with ExitStack() as after:
        # The probes below ship functions through collective_rpc.
        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        engine = AsyncLLM.from_engine_args(_get_dp_pause_engine_args())
        after.callback(engine.shutdown)

        async for _ in engine.generate(
            request_id="warmup",
            prompt=DP_PAUSE_PROMPT,
            sampling_params=SamplingParams(max_tokens=5),
        ):
            pass

        def inflate_dummy_batches(worker) -> bool:
            inner = worker.execute_dummy_batch

            def slow_dummy_batch() -> None:
                inner()
                # Keep the stream busy after launch, like a large-MoE forward.
                # NPU stand-in for the upstream torch.cuda._sleep(200_000_000):
                # torch.npu has no _sleep and torch.cuda is unusable on NPU.
                enqueue_npu_busy_work()

            worker.execute_dummy_batch = slow_dummy_batch
            return True

        assert all(await engine.engine_core.collective_rpc_async(inflate_dummy_batches))

        await asyncio.wait_for(engine.pause_generation(mode="abort"), timeout=60)

        def device_idle(worker) -> bool:
            return npu_stream_idle()

        idle = await engine.engine_core.collective_rpc_async(device_idle)
        assert all(idle), "pause_generation() returned while NPU work was still in flight"


@pytest.mark.skipif(
    not _has_pause_device_barrier(),
    reason="requires a vLLM with the #52914 pause device barrier (EngineCoreProc._finish_pause)",
)
def test_dp_pause_completion_implies_npu_idle(monkeypatch: pytest.MonkeyPatch):
    """A resolved pause future promises an idle device. Inflate the dummy
    batches the pause consensus manufactures so any work the pause fails to wait
    on stays visible, then require a quiet NPU stream the moment
    ``pause_generation()`` returns.

    Driven through ``asyncio.run`` on purpose: the repository does not depend on
    pytest-asyncio, and a bare async test would silently skip.
    """
    asyncio.run(_pause_an_inflated_dp_engine(monkeypatch))
