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
"""Tolerate late/duplicate KV-transfer receive terminals on the pinned vLLM core.

A PD-disaggregated decode engine can receive a finished_recving terminal for
a request that already left WAITING_FOR_REMOTE_KVS (a duplicate transfer
round, or a failure terminal racing a request that already started decoding).
The pinned vLLM core treats this as a state-machine violation and asserts,
which kills the whole engine group. This patch fails only the affected
request in a controlled way and emits a client-facing error output for it.
"""

from __future__ import annotations

from vllm.logger import logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.request import Request, RequestStatus


def _update_from_kv_xfer_finished(self, kv_connector_output):
    """Scheduler._update_from_kv_xfer_finished with controlled failure.

    Differences from the pinned core:
    * a terminal for an unknown request is logged and skipped instead of
      indexing ``self.requests`` directly;
    * a terminal for a request that already left WAITING_FOR_REMOTE_KVS is
      no longer asserted away: the request is failed via ``finish_requests``
      (controlled FINISHED_ERROR, blocks released through the normal free
      path) and stashed so the ``update_from_output`` wrapper below can
      emit a client-facing error output for it.
    """
    if self.connector is not None:
        self.connector.update_connector_output(kv_connector_output)

    terminal_errors: list[Request] = []
    for req_id in kv_connector_output.finished_recving or ():
        logger.debug("Finished recving KV transfer for request %s", req_id)
        req = self.requests.get(req_id)
        if req is None:
            # Very late completion after the request was cleaned up.
            logger.warning(
                "finished_recving for unknown request %s; ignoring.", req_id
            )
            continue
        if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            self.finished_recving_kv_req_ids.add(req_id)
        elif RequestStatus.is_finished(req.status):
            self._free_blocks(req)
        else:
            # A terminal for a request that already left the waiting state
            # (duplicate transfer round / failure racing an in-flight
            # decode) is a state-machine violation. The KV of the first
            # round can no longer be trusted; fail the request in a
            # controlled way instead of killing the engine.
            logger.error(
                "Duplicate or out-of-order finished_recving for request %s "
                "in state %s (computed=%s); failing the request.",
                req_id,
                req.status.name,
                req.num_computed_tokens,
            )
            terminal_errors.extend(
                self.finish_requests(req_id, RequestStatus.FINISHED_ERROR)
            )
    for req_id in kv_connector_output.finished_sending or ():
        logger.debug("Finished sending KV transfer for request %s", req_id)
        req = self.requests.get(req_id)
        if req is None:
            logger.warning(
                "finished_sending for unknown request %s; ignoring.", req_id
            )
            continue
        self._free_blocks(req)

    # Consumed by the update_from_output wrapper below.
    self._pd_terminal_errors = terminal_errors
    return terminal_errors


_orig_update_from_output = Scheduler.update_from_output


def _update_from_output(self, scheduler_output, model_runner_output):
    engine_core_outputs = _orig_update_from_output(
        self, scheduler_output, model_runner_output
    )
    terminal_errors = getattr(self, "_pd_terminal_errors", None)
    if terminal_errors:
        self._pd_terminal_errors = []
        for request in terminal_errors:
            # Emit exactly one terminal frame: drop any nonterminal frame
            # already emitted for this request so the stream ends with the
            # error instead of a token followed by a dangling connection.
            error_frame = EngineCoreOutput(
                request_id=request.request_id,
                new_token_ids=[],
                finish_reason=request.get_finished_reason(),
                events=request.take_events(),
                trace_headers=request.trace_headers,
            )
            eco = engine_core_outputs.get(request.client_index)
            if eco is None:
                engine_core_outputs[request.client_index] = EngineCoreOutputs(
                    outputs=[error_frame]
                )
                continue
            eco.outputs[:] = [
                out
                for out in eco.outputs
                if out.request_id != request.request_id
            ]
            eco.outputs.append(error_frame)
    return engine_core_outputs


Scheduler._update_from_kv_xfer_finished = _update_from_kv_xfer_finished
Scheduler.update_from_output = _update_from_output
