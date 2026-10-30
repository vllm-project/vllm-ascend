# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from typing import TYPE_CHECKING, Any

from vllm.v1.worker.gpu.kv_connector import KVConnector

from vllm_ascend.models.qwen4_exp.pd_decode_queue import PDDecodeQueueState
from vllm_ascend.worker.v2.kv_connector import MambaCopyKVConnector

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput


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
        known = False
        try:
            metadata = scheduler_output.kv_connector_metadata
            fields = getattr(metadata, "__dict__", None)
            # Unknown fields may add transfer actions: retain the baseline path
            # until their semantics are supported, including slot-only metadata.
            known = (
                type(metadata) is self._metadata_type
                and type(fields) is dict
                and set(fields) == {"requests", "requests_to_send", "reqs_in_batch"}
                and type(getattr(metadata, "requests", None)) is dict
                and type(getattr(metadata, "requests_to_send", None)) is dict
                and type(getattr(metadata, "reqs_in_batch", None)) is set
            )
        except Exception:
            # Optional queue admission must not interrupt the connector itself.
            pass
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
            # The baseline Mamba wrapper only handles the pre-forward copy.
            # Forward finalization to its actual connector in this adapter.
            connector = self._connector._connector if type(self._connector) is MambaCopyKVConnector else self._connector
            connector.finalize_step()
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


def unwrap_pd_decode_queue_connector(connector: KVConnector) -> KVConnector:
    """Retire owned observers while preserving known Mamba copy callbacks."""
    if type(connector) is PDDecodeQueueKVConnector:
        connector.decode_queue_state.invalidate()
        return unwrap_pd_decode_queue_connector(connector._connector)
    if type(connector) is MambaCopyKVConnector:
        inner = unwrap_pd_decode_queue_connector(connector._connector)
        if inner is not connector._connector:
            return MambaCopyKVConnector(inner, connector._copy_mamba_state)
    return connector


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
