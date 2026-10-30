#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm PR #45586
#   (tests/entrypoints/serve/dev/rlhf/test_sleep_wake.py)
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

"""
End-to-end tests for the vLLM RL sleep/wake lifecycle on Ascend NPU.

Endpoint surface under test
---------------------------
sleep/api_router  : POST /sleep  POST /wake_up  GET /is_sleeping

All tests require:
  --enable-sleep-mode   KV cache allocated via CuMemAllocator; without this
                        flag sleep/wake are no-ops and the bug cannot trigger.
  VLLM_SERVER_DEV_MODE=1
  --additional-config '{"weight_nz_mode": 0}'
"""

from tests.e2e.pull_request.one_card.rlhf.conftest import (
    gen,
    health,
    npu_free_bytes,
    server,
    sleep,
    sleep_metrics,
    wake,
)

# ---------------------------------------------------------------------------
# TestPhysicalMemory
# ---------------------------------------------------------------------------


class TestPhysicalMemory:
    """Assert NPU free bytes change, not just flags.

    Guards against regressions where CuMemAllocator.sleep() silently no-ops
    (e.g. missing stream-sync, wrong tag registration) while returning 200.
    Each stage is cross-validated against the Prometheus sleep-state metrics.
    """

    def test_sleep_level1_frees_npu_memory(self):
        with server() as url:
            gen(url)  # warm up — allocate KV blocks
            free_awake = npu_free_bytes()

            assert sleep(url, level=1) == 200
            free_sleeping = npu_free_bytes()
            freed_gib = (free_sleeping - free_awake) / 2**30

            # 0.5 GiB threshold: sleep(1) offloads weights only
            # (~1.2 GiB for 0.6B bf16)
            assert freed_gib > 0.5, f"sleep(1) freed only {freed_gib:.2f} GiB — CuMemAllocator unmap may be a no-op"
            awake, wo, _ = sleep_metrics(url)
            assert awake == 0 and wo == 1, f"Prometheus sleep metrics inconsistent: awake={awake} wo={wo}"

            assert wake(url) == 200
            free_awake2 = npu_free_bytes()
            re_allocated_gib = (free_sleeping - free_awake2) / 2**30
            assert re_allocated_gib > 0.4, (
                f"wake_up re-allocated only {re_allocated_gib:.2f} GiB — remap may be incomplete"
            )

    def test_sleep_level2_frees_all_discards_all(self):
        with server() as url:
            gen(url)
            free_awake = npu_free_bytes()

            assert sleep(url, level=2) == 200
            freed_gib = (npu_free_bytes() - free_awake) / 2**30
            assert freed_gib > 1.5

            _, _, da = sleep_metrics(url)
            assert da == 1

            assert wake(url) == 200
            assert health(url) == 200


# ---------------------------------------------------------------------------
# TestOutputCorrectness
# ---------------------------------------------------------------------------


class TestOutputCorrectness:
    """Output must be deterministic and self-consistent across the lifecycle."""

    def test_staged_wake_restores_output(self):
        """sleep → wake(weights) → wake(kv_cache) — output matches golden."""
        with server() as url:
            golden_text = gen(url)["choices"][0]["text"]

            assert sleep(url, level=1) == 200
            assert wake(url, tags=["weights"]) == 200
            assert wake(url, tags=["kv_cache"]) == 200

            resp = gen(url)
            assert resp and resp["choices"][0]["text"] == golden_text
