# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.v1.worker.gpu.kv_connector import KVConnector

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput


class MambaCopyKVConnector(KVConnector):
    """Keep Mamba bulk copies between synchronous KV loads and forward.

    Wrap only this runner's connector. In particular, do not mutate the
    upstream class or the shared no-op connector used during profiling.
    """

    def __init__(self, connector: KVConnector, copy_mamba_state: Callable[[], None]) -> None:
        self._connector = connector
        self._copy_mamba_state = copy_mamba_state

    def pre_forward(self, scheduler_output: "SchedulerOutput", *args: Any, **kwargs: Any) -> None:
        self._connector.pre_forward(scheduler_output, *args, **kwargs)
        self._copy_mamba_state()

    def finish_forward(self) -> None:
        self._connector.finish_forward()

    def post_forward(self, finished_req_ids: set[str], *args: Any, **kwargs: Any) -> "KVConnectorOutput | None":
        return self._connector.post_forward(finished_req_ids, *args, **kwargs)

    def no_forward(self, scheduler_output: "SchedulerOutput") -> "ModelRunnerOutput":
        # No state preprocessing or model execution on this path.
        return self._connector.no_forward(scheduler_output)

    def set_disabled(self, disabled: bool) -> None:
        self._connector.set_disabled(disabled)

    def reset_capture_state(self) -> None:
        self._connector.reset_capture_state()
