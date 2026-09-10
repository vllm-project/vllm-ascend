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
import queue
from logging import DEBUG

import vllm.envs as envs
from vllm.v1.engine.core import EngineCoreProc, logger

from vllm_ascend.common.utils.watch_dog import get_watch_dog

_WAIT_INPUT_TIMEOUT = 5

_watchdog = get_watch_dog()

_original_run_engine_core = EngineCoreProc.run_engine_core
_original_process_engine_step = EngineCoreProc._process_engine_step


def _patched_run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
    _watchdog.setup("engine", timeout=envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS)
    _watchdog.start()
    return _original_run_engine_core(*args, dp_rank=dp_rank, local_dp_rank=local_dp_rank, **kwargs)


def _patched_process_engine_step(self) -> bool:
    _watchdog.feed()
    return _original_process_engine_step(self)


def _patched_process_input_queue(self):
    waited = False
    while not self.has_work() and self.is_running():
        # Notify callbacks waiting for engine to become idle.
        self._notify_idle_state_callbacks()
        if self.input_queue.empty():
            # Drain aborts queue; all aborts are also processed via input_queue.
            with self.aborts_queue.mutex:
                self.aborts_queue.queue.clear()
            if logger.isEnabledFor(DEBUG):
                logger.debug("EngineCore waiting for work.")
                waited = True
        block = self.process_input_queue_block
        try:
            req = self.input_queue.get(block=block, timeout=_WAIT_INPUT_TIMEOUT)
            self._handle_client_request(*req)
        except queue.Empty:
            _watchdog.feed()
            if block:
                continue
            else:
                break
        if not block:
            break

    if waited:
        logger.debug("EngineCore loop active.")

    # Handle any more client requests.
    while not self.input_queue.empty():
        req = self.input_queue.get_nowait()
        self._handle_client_request(*req)


EngineCoreProc.run_engine_core = staticmethod(_patched_run_engine_core)
EngineCoreProc._process_engine_step = _patched_process_engine_step
EngineCoreProc._process_input_queue = _patched_process_input_queue
