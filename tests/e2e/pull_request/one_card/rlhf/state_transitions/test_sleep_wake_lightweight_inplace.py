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

"""NPU gate for Sleep Mode lightweight HCCL + inplace ACL Graph (TP=1).

Teacher review on PR #20481 asks whether these hold at the same time:
  P1 suspend HCCL (not destroy)
  P2 memory-release benefit
  P3 no ACL Graph recapture after wake

This case runs multi-round sleep/wake and the first generate after wake on
one card with ``sleep_mode_extra_cleanup`` enabled. Worker logs emit a
structured ``Sleep cycle report: {...}`` dict with
``hccl_action_actual`` / ``capture_model_called`` / ``hbm_freed_bytes`` for
evidence collection. TP=2 Suspend vs ``fallback_destroy`` conclusions are
documented in ``docs/community_task/design_sleep_mode_lifecycle.md``.
"""

from __future__ import annotations

from tests.e2e.common.rlhf.common import gen, npu_free_bytes, server, sleep, wake

_RL_LIGHTWEIGHT_INPLACE = {
    "weight_nz_mode": 0,
    "rl_config": {
        "enabled": True,
        "sleep_mode_extra_cleanup": True,
        "sleep_hccl_policy": "lightweight",
        "sleep_acl_graph_policy": "inplace",
        "sleep_allow_fallback": True,
    },
}


class TestLightweightInplaceSleepWake:
    def test_multi_round_sleep_wake_generate_and_memory(self):
        with server(additional_config=_RL_LIGHTWEIGHT_INPLACE, port=8771) as url:
            first = gen(url)
            assert first and first["choices"][0]["text"]

            for _ in range(2):
                free_before = npu_free_bytes()
                assert sleep(url, level=1) == 200
                free_sleeping = npu_free_bytes()
                freed_gib = (free_sleeping - free_before) / 2**30
                assert freed_gib > 0.3, f"sleep freed only {freed_gib:.3f} GiB"

                assert wake(url) == 200
                after = gen(url)
                assert after and after["choices"][0]["text"]
