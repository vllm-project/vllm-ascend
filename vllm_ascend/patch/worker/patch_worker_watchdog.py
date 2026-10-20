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
from threading import Thread

from vllm.v1.executor.multiproc_executor import WorkerProc, logger

from vllm_ascend.watch_dog import get_watch_dog, start_watch_dog


def _patched_worker_busy_loop(self, *args, **kwargs):
    rank = kwargs.get("rank", 0)
    start_watch_dog(f"worker_{rank}", logger)
    """Main busy loop for Multiprocessing Workers."""
    assert self.rpc_broadcast_mq is not None
    while True:
        with get_watch_dog().disable():
            rpc_request = self.rpc_broadcast_mq.dequeue(indefinite=True)
        get_watch_dog().feed()
        self._execute_worker_rpc(rpc_request)
        del rpc_request


def _patched_monitor_death_pipe(self, death_pipe, shutdown_requested):
    if death_pipe is None:
        return

    def death_pipe_monitor(queues_to_shutdown):
        try:
            # This will block until parent process exits (pipe closes)
            death_pipe.recv()
        except EOFError:
            logger.info_once("Parent process exited, terminating worker queues")
            get_watch_dog().handle_failure("shutdown")
            shutdown_requested.set()
            for mq in queues_to_shutdown:
                if mq is not None:
                    mq.shutdown()
        except Exception as e:
            logger.warning("Death monitoring error: %s", e)

    # Pass queue references directly to avoid gc issues if passing self
    Thread(
        target=death_pipe_monitor,
        args=([self.rpc_broadcast_mq, self.worker_response_mq],),
        daemon=True,
        name="DeathPipeMonitor",
    ).start()


WorkerProc.worker_busy_loop = _patched_worker_busy_loop
WorkerProc.monitor_death_pipe = _patched_monitor_death_pipe
