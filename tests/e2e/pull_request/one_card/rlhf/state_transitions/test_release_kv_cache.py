#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
End-to-end tests for releasing only the KV cache on Ascend NPU.

Endpoint surface under test
---------------------------
sleep/api_router  : POST /sleep  POST /release_kv_cache_memory  POST /wake_up
                    GET /is_sleeping

The documented sequence is sleep(level=0, mode="keep") ->
release_kv_cache_memory() -> wake_up(["kv_cache"]). The engine rejects a
release unless the pause has completed and all executor memory is resident.
"""

from tests.e2e.common.rlhf.common import (
    gen,
    is_sleeping,
    npu_free_bytes,
    release_kv_cache,
    server,
    sleep,
    wake,
)

# The KV cache takes most of the 0.75 memory budget, while the Qwen3-0.6B bf16
# weights are about 1.2 GiB, so only the KV cache can account for this much.
KV_CACHE_MIN_GIB = 2.0


def text(url) -> str:
    resp = gen(url)
    assert resp and resp.get("choices"), f"completion failed: {resp}"
    return resp["choices"][0]["text"]


class TestReleaseKvCacheMemory:
    def test_release_frees_kv_cache_and_keeps_weights(self):
        """The KV cache is freed and restored under default weight_nz_mode=1.

        Weights stay resident across a KV-only release/wake, so FRACTAL_NZ
        weights remain valid and produce the same output. A second release while
        the KV cache is released is refused and changes nothing.
        """
        with server(weight_nz_mode=None) as url:
            golden_text = text(url)
            free_awake = npu_free_bytes()

            assert sleep(url, level=0, mode="keep") == 200
            assert release_kv_cache(url) == 200
            free_released = npu_free_bytes()
            freed_gib = (free_released - free_awake) / 2**30
            assert freed_gib > KV_CACHE_MIN_GIB, f"release freed only {freed_gib:.2f} GiB"
            # Scheduling stays paused until the KV cache wakes.
            assert is_sleeping(url)

            assert release_kv_cache(url) != 200
            assert is_sleeping(url)

            assert wake(url, tags=["kv_cache"]) == 200
            re_allocated_gib = (free_released - npu_free_bytes()) / 2**30
            assert re_allocated_gib > KV_CACHE_MIN_GIB, f"wake_up re-allocated only {re_allocated_gib:.2f} GiB"
            assert not is_sleeping(url)
            assert text(url) == golden_text

    def test_release_during_sleep_is_rejected(self):
        """A release after a level-1 sleep is refused; a full wake-up still restores serving."""
        with server() as url:
            golden_text = text(url)

            assert sleep(url, level=1) == 200
            assert release_kv_cache(url) != 200
            assert is_sleeping(url)

            assert wake(url) == 200
            assert text(url) == golden_text

    def test_sleep_after_release_then_staged_wake(self):
        """A level-1 sleep after a release offloads the weights; a staged wake restores both."""
        with server() as url:
            golden_text = text(url)

            assert sleep(url, level=0, mode="keep") == 200
            assert release_kv_cache(url) == 200
            assert sleep(url, level=1) == 200

            assert wake(url, tags=["weights"]) == 200
            # The KV cache is still released, so scheduling stays paused.
            assert is_sleeping(url)
            assert wake(url, tags=["kv_cache"]) == 200
            assert not is_sleeping(url)
            assert text(url) == golden_text
