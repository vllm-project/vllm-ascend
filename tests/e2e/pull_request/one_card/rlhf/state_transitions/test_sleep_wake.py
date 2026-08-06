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
  VLLM_ASCEND_ENABLE_NZ=0
"""

import requests

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

    def test_staged_release_each_step_changes_memory(self):
        """Each tag releases a distinct chunk of NPU memory."""
        with server() as url:
            gen(url)
            assert sleep(url, level=1) == 200

            assert wake(url, tags=["weights"]) == 200
            free_after_weights = npu_free_bytes()

            assert wake(url, tags=["kv_cache"]) == 200
            free_after_kv = npu_free_bytes()

            # waking kv_cache consumes more NPU memory than weights-only wake
            assert free_after_kv < free_after_weights, (
                "waking kv_cache should use more NPU memory than weights-only wake"
            )
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

    def test_multiple_cycles_stable(self):
        """3× sleep/wake cycles — output and engine stay stable.

        Guards against cumem bookkeeping corruption across repeated
        release+remap of the same physical pages.
        """
        with server() as url:
            golden_text = gen(url)["choices"][0]["text"]

            for i in range(3):
                assert sleep(url, level=1) == 200
                assert wake(url) == 200
                assert health(url) == 200

                resp = gen(url)
                assert resp and resp["choices"][0]["text"] == golden_text, (
                    f"output drifted on cycle {i} — cumem bookkeeping corrupted"
                )


# ---------------------------------------------------------------------------
# TestMemoryLeakCycle  (new — Tier 1A supplement)
# ---------------------------------------------------------------------------


class TestMemoryLeakCycle:
    """sleep/wake cycles must not accumulate NPU memory leaks.

    Reference: ROLL tests/third_party/vllm/test_vllm_mem_oom.py
               generate_memory() — 20 iterations tracking memory growth.

    We measure NPU free bytes when the engine is awake (same lifecycle stage
    on every cycle) rather than host RSS, because:
      (a) the vLLM server is a subprocess so its RSS is not directly readable
          from the test process, and
      (b) NPU memory is what cumem manages — leaks manifest there first.
    """

    def test_no_npu_memory_growth_over_5_cycles(self):
        """5 sleep/wake cycles: NPU free bytes (when awake) must be stable.

        After each wake, the engine should have remapped the same NPU pages.
        A growing delta (less free memory each cycle) indicates a leak in
        cumem bookkeeping or handle tracking.
        """
        with server() as url:
            free_samples = []

            for i in range(5):
                gen(url)
                assert sleep(url, level=1) == 200
                assert wake(url) == 200
                assert health(url) == 200

                if i >= 2:  # skip warm-up cycles
                    free_samples.append(npu_free_bytes())

            baseline = free_samples[0]
            # Allow 50 MiB tolerance for KV block allocation jitter
            min_free = min(free_samples)
            leak_gib = (baseline - min_free) / 2**30

            assert leak_gib < 0.05, (  # 50 MiB tolerance
                f"NPU free memory shrank by {leak_gib:.3f} GiB over 8 post-warmup "
                f"sleep/wake cycles (baseline={baseline / 2**30:.2f} GiB, "
                f"min={min_free / 2**30:.2f} GiB) — "
                "possible cumem handle leak or unmapped page accumulation"
            )


# ---------------------------------------------------------------------------
# TestLogprobsPrecision  (new — Tier 1C supplement)
# ---------------------------------------------------------------------------


class TestLogprobsPrecision:
    """logprobs values must be consistent before and after a sleep/wake cycle.

    Reference: ROLL tests/distributed/strategy/log_probs/ (9 files)
               test_fsdp_log_probs_full, test_fsdp_log_probs_cp_rmpad, etc.

    After sleep(level=1)/wake, weights are remapped from CPU backup.
    Calibrated scales (FP8-KV, etc.) must be restored; logprobs must match
    the pre-sleep values within a tight tolerance.
    """

    def test_logprobs_stable_after_sleepwake(self):
        """logprobs before and after sleep/wake must match within 1e-2.

        Reference: ROLL test_fsdp_log_probs_full — compares log_probs values
        across different parallelism configurations to within tight tolerance.
        """
        with server() as url:
            prompt = "The capital of France is Paris and the capital of Germany is"

            def _get_logprobs():
                r = requests.post(
                    f"{url}/v1/completions",
                    json={
                        "model": "m",
                        "prompt": prompt,
                        "max_tokens": 4,
                        "temperature": 0,
                        "logprobs": 5,
                    },
                    timeout=30,
                )
                resp = r.json()
                if "choices" not in resp or not resp["choices"]:
                    return None
                choice = resp["choices"][0]
                lp = choice.get("logprobs", {})
                return lp.get("token_logprobs", [])

            before = _get_logprobs()
            assert before is not None, "failed to get logprobs before sleep"
            assert len(before) > 0

            assert sleep(url, level=1) == 200
            assert wake(url) == 200
            assert health(url) == 200

            after = _get_logprobs()
            assert after is not None, "failed to get logprobs after sleep/wake"
            assert len(after) == len(before), "logprobs length changed after sleep/wake"

            compared = 0
            for i, (b, a) in enumerate(zip(before, after)):
                if b is None or a is None:
                    continue
                compared += 1
                diff = abs(b - a)
                # BF16 has ~3 significant decimal digits; 1e-2 is achievable
                # for identical greedy decodes across a sleep/wake cycle.
                assert diff < 1e-2, (
                    f"logprob[{i}] drifted after sleep/wake: "
                    f"before={b:.6f} after={a:.6f} diff={diff:.2e} — "
                    "weight restore or KV-scale recalibration may be incorrect"
                )
            assert compared > 0, "no non-None logprob pairs were compared — logprobs response may be empty or malformed"
