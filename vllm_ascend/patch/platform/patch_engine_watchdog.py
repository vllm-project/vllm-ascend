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
import logging
import queue

from vllm.v1.engine.core import EngineCoreProc, logger

from vllm_ascend.watch_dog import get_watch_dog, start_watch_dog

_original_process_engine_step = EngineCoreProc._process_engine_step


def _patched_process_engine_step(self) -> bool:
    get_watch_dog().feed()
    return _original_process_engine_step(self)


def _patched_process_input_queue(self):
    """Exits when an engine step needs to be performed."""
    waited = False
    while not self.has_work() and self.is_running():
        # Notify callbacks waiting for engine to become idle.
        self._notify_idle_state_callbacks()
        if self.input_queue.empty():
            # Drain aborts queue; all aborts are also processed via input_queue.
            with self.aborts_queue.mutex:
                self.aborts_queue.queue.clear()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("EngineCore waiting for work.")
                waited = True
        block = self.process_input_queue_block
        try:
            with get_watch_dog().disable():
                req = self.input_queue.get(block=block)
            self._handle_client_request(*req)
        except queue.Empty:
            break
        if not block:
            break

    if waited:
        logger.debug("EngineCore loop active.")

    # Handle any more client requests.
    while not self.input_queue.empty():
        req = self.input_queue.get_nowait()
        self._handle_client_request(*req)


def patch_watchdog_for_engine_core(dp_rank):
    EngineCoreProc._process_engine_step = _patched_process_engine_step
    EngineCoreProc._process_input_queue = _patched_process_input_queue

    start_watch_dog(f"engine_{dp_rank}", logger)
