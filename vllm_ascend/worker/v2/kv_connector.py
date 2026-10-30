# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.v1.worker.gpu.kv_connector import KVConnector

from vllm_ascend.worker.v2.pd_decode_queue import PDDecodeQueueState

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

    def finalize_step(self) -> None:
        self._connector.finalize_step()

    def post_forward(self, finished_req_ids: set[str], *args: Any, **kwargs: Any) -> "KVConnectorOutput | None":
        return self._connector.post_forward(finished_req_ids, *args, **kwargs)

    def no_forward(self, scheduler_output: "SchedulerOutput") -> "ModelRunnerOutput":
        # No state preprocessing or model execution on this path.
        return self._connector.no_forward(scheduler_output)

    def set_disabled(self, disabled: bool) -> None:
        self._connector.set_disabled(disabled)

    def reset_capture_state(self) -> None:
        self._connector.reset_capture_state()


class PDDecodeQueueKVConnector(KVConnector):
    """Observe the existing Mooncake callbacks without consuming notifications.

    The wrapped connector retains all transfer, ACK, finalization and error
    behavior. Only its already-returned completion output feeds the host proof.
    """

    def __init__(self, connector: KVConnector, transfer_group: Any, metadata_type: type) -> None:
        self._connector = connector
        self.transfer_group = transfer_group
        self._metadata_type = metadata_type
        self._disabled = False
        self.decode_queue_state = PDDecodeQueueState()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connector, name)

    def observe_scheduler_output(self, scheduler_output: "SchedulerOutput") -> None:
        if self._disabled:
            return
        metadata = scheduler_output.kv_connector_metadata
        known = (
            type(metadata) is self._metadata_type
            and type(getattr(metadata, "requests", None)) is dict
            and type(getattr(metadata, "requests_to_send", None)) is dict
            and type(getattr(metadata, "reqs_in_batch", None)) is set
            and set(vars(metadata)) == {"requests", "requests_to_send", "reqs_in_batch"}
        )
        self.decode_queue_state.observe_step(scheduler_output, metadata_known=known)

    def pre_forward(self, scheduler_output: "SchedulerOutput", *args: Any, **kwargs: Any) -> None:
        self.observe_scheduler_output(scheduler_output)
        try:
            self._connector.pre_forward(scheduler_output, *args, **kwargs)
        except Exception:
            self.decode_queue_state.invalidate()
            raise

    def finish_forward(self) -> None:
        try:
            self._connector.finish_forward()
        except Exception:
            self.decode_queue_state.invalidate()
            raise

    def post_forward(self, finished_req_ids: set[str], *args: Any, **kwargs: Any) -> "KVConnectorOutput | None":
        try:
            output = self._connector.post_forward(finished_req_ids, *args, **kwargs)
        except Exception:
            self.decode_queue_state.invalidate()
            raise
        if not self._disabled:
            self.decode_queue_state.observe_completion(output)
        return output

    def finalize_step(self) -> None:
        try:
            self._connector.finalize_step()
        except Exception:
            self.decode_queue_state.invalidate()
            raise

    def no_forward(self, scheduler_output: "SchedulerOutput") -> "ModelRunnerOutput":
        self.observe_scheduler_output(scheduler_output)
        try:
            output = self._connector.no_forward(scheduler_output)
        except Exception:
            self.decode_queue_state.invalidate()
            raise
        if not self._disabled:
            self.decode_queue_state.observe_completion(output.kv_connector_output)
        return output

    def set_disabled(self, disabled: bool) -> None:
        self._connector.set_disabled(disabled)
        if disabled != self._disabled:
            self.decode_queue_state.invalidate()
        self._disabled = disabled

    def reset_capture_state(self) -> None:
        self._connector.reset_capture_state()
        self.decode_queue_state.invalidate()


def wrap_pd_decode_queue_connector(connector: KVConnector, transfer_group: Any) -> KVConnector:
    # Lazy imports keep Mooncake's engine dependencies out of ordinary workers.
    # MooncakeConnector is an alias of this registered pull implementation.
    if type(transfer_group).__module__ != "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector":
        return connector

    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakePullConnector
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeConnectorMetadata

    inner = connector._connector if type(connector) is MambaCopyKVConnector else connector
    if (
        type(inner) is ActiveKVConnector
        and inner.kv_connector is transfer_group
        and type(transfer_group) is MooncakePullConnector
        and transfer_group.kv_transfer_config.is_kv_consumer
    ):
        return PDDecodeQueueKVConnector(connector, transfer_group, MooncakeConnectorMetadata)
    return connector
