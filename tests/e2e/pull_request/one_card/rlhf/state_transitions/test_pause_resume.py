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
End-to-end tests for the vLLM RL /pause /resume /is_paused lifecycle on Ascend NPU.

Endpoint surface under test
---------------------------
rlhf/api_router   : POST /pause  POST /resume   GET /is_paused
"""

import threading
import time

from tests.e2e.pull_request.one_card.rlhf.conftest import (
    gen,
    ok,
    pause,
    resume,
    server,
)

# ---------------------------------------------------------------------------
# TestPauseResume
# ---------------------------------------------------------------------------


class TestPauseResume:
    """POST /pause  POST /resume  GET /is_paused are independent of sleep.

    /pause blocks scheduling without releasing NPU memory (level=0 equivalent
    from the NPU side, but a distinct code path and distinct state flag).
    """

    def test_pause_mode_wait_drains_inflight_request(self):
        """mode='wait' lets an in-flight request complete, then blocks new ones."""
        with server() as url:
            result: dict = {}

            def _bg():
                result["r"] = gen(url, max_tokens=32, timeout=60)

            t = threading.Thread(target=_bg)
            t.start()
            time.sleep(0.5)

            assert pause(url, mode="wait") == 200
            t.join(timeout=30)
            assert result.get("r") is not None, "in-flight request not completed after pause(mode=wait)"

            resp = gen(url, timeout=5)
            assert not ok(resp)
            assert resume(url) == 200

    def test_pause_mode_keep_resumes_frozen_request(self):
        """mode='keep' freezes the request; it must complete after /resume."""
        with server() as url:
            result: dict = {}

            def _bg():
                result["r"] = gen(url, max_tokens=128, timeout=60)

            t = threading.Thread(target=_bg)
            t.start()
            time.sleep(0.1)

            assert pause(url, mode="keep") == 200
            time.sleep(1)

            # request must NOT have completed yet
            assert not ok(result.get("r")), "request completed before resume in mode=keep"

            assert resume(url) == 200
            t.join(timeout=30)
            assert ok(result.get("r")), "request not completed after resume in mode=keep"
