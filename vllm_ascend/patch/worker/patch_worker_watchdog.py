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
import vllm.envs as envs
from vllm.v1.executor.multiproc_executor import WorkerProc

from vllm_ascend.common.utils.watch_dog import get_watch_dog

_watchdog = get_watch_dog()

_original_worker_busy_loop = WorkerProc.worker_busy_loop


def _patched_worker_busy_loop(*args, **kwargs):
    _watchdog.setup("worker", timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
    _watchdog.start()
    return _original_worker_busy_loop(*args, **kwargs)


WorkerProc.worker_busy_loop = _patched_worker_busy_loop
