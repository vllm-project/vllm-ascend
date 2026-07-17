#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import enum
import sys

import vllm.v1.request as request_module


def patch_request_status_for_nonbsp() -> None:
    old_status = request_module.RequestStatus
    if hasattr(old_status, "LB_PAUSED"):
        return

    class RequestStatus(enum.IntEnum):
        WAITING = enum.auto()
        WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
        WAITING_FOR_REMOTE_KVS = enum.auto()
        WAITING_FOR_STREAMING_REQ = enum.auto()
        RUNNING = enum.auto()
        LB_PAUSED = enum.auto()
        PREEMPTED = enum.auto()
        FINISHED_STOPPED = enum.auto()
        FINISHED_LENGTH_CAPPED = enum.auto()
        FINISHED_ABORTED = enum.auto()
        FINISHED_IGNORED = enum.auto()
        FINISHED_ERROR = enum.auto()
        FINISHED_REPETITION = enum.auto()

        def __str__(self):
            return self.name

        @staticmethod
        def is_finished(status: "RequestStatus") -> bool:
            return status > RequestStatus.PREEMPTED

        @staticmethod
        def get_finished_reason(status: "RequestStatus"):
            return request_module._FINISHED_REASON_MAP.get(status)

    request_module.RequestStatus = RequestStatus
    for module_name in (
        "vllm.v1.engine.core",
        "vllm.v1.core.sched.scheduler",
        "vllm.v1.core.sched.utils",
        "vllm.v1.core.sched.async_scheduler",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "RequestStatus"):
            module.RequestStatus = RequestStatus
    request_module._FINISHED_REASON_MAP = {
        RequestStatus.FINISHED_STOPPED: request_module.FinishReason.STOP,
        RequestStatus.FINISHED_LENGTH_CAPPED: request_module.FinishReason.LENGTH,
        RequestStatus.FINISHED_ABORTED: request_module.FinishReason.ABORT,
        RequestStatus.FINISHED_IGNORED: request_module.FinishReason.LENGTH,
        RequestStatus.FINISHED_ERROR: request_module.FinishReason.ERROR,
        RequestStatus.WAITING_FOR_STREAMING_REQ: request_module.FinishReason.STOP,
        RequestStatus.FINISHED_REPETITION: request_module.FinishReason.REPETITION,
    }


patch_request_status_for_nonbsp()
