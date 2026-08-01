# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU IPC-based weight transfer engine using Ascend IPC for communication."""

import pickle
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import pybase64 as base64
import torch
from vllm import envs
from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
)
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerSendWeightsArgs,
    IPCWeightTransferUpdateInfo,
)

from vllm_ascend.distributed.weight_transfer.device_mapping import get_npu_ipc_uuid
from vllm_ascend.distributed.weight_transfer.lifecycle import WeightUpdateLifecyclePolicy, get_weight_update_lifecycle_policy
from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    packed_npu_ipc_consumer,
    packed_npu_ipc_producer,
)
from vllm_ascend.distributed.weight_transfer.trainer_send import (
    TrainerProcessCoordinator,
    collect_parameter_metadata,
    default_parameter_tensor,
    dispatch_update_info,
)


@dataclass
class NPUIPCTrainerSendWeightsArgs(IPCTrainerSendWeightsArgs):
    """NPU IPC variant — inherits all fields and validation from the CUDA IPC
    base class.  Only the ``send_mode`` callable type is widened to accept the
    NPU update-info type."""

    send_mode: str | Callable[["NPUIPCWeightTransferUpdateInfo"], None]


@dataclass
class NPUIPCWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for NPU IPC weight transfer backend.

    No initialization needed for NPU IPC.
    """

    pass


@dataclass
class NPUIPCWeightTransferUpdateInfo(IPCWeightTransferUpdateInfo):
    """NPU IPC variant — inherits all fields and validation from the CUDA IPC
    base class.  No overrides needed; the field types and ``__post_init__`` are
    identical."""


def npu_generate_uuid() -> str:
    """Generate a unique identifier for the current process's physical NPU chip."""
    return get_npu_ipc_uuid()


class NPUIPCWeightTransferEngine(WeightTransferEngine[NPUIPCWeightTransferInitInfo, NPUIPCWeightTransferUpdateInfo]):
    """
    Weight transfer engine using NPU IPC for communication between
    trainer and workers.

    This implementation uses Ascend NPU IPC to transfer weights from the
    trainer (rank 0) to all inference workers. IPC handles are used to
    share memory between processes on the same node.

    Requires ``torch_npu`` to be imported (which patches
    ``torch.multiprocessing.reductions.reduce_tensor`` to support
    NPU tensors via ``_share_npu_()`` / ``rebuild_npu_tensor``).
    """

    init_info_cls = NPUIPCWeightTransferInitInfo
    update_info_cls = NPUIPCWeightTransferUpdateInfo
    supports_draft_weight_update = False

    def __init__(  # type: ignore[misc]
        self,
        config: WeightTransferConfig,
        vllm_config: VllmConfig,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self._weight_update_lifecycle_policy: WeightUpdateLifecyclePolicy = get_weight_update_lifecycle_policy(False)

    def set_weight_update_lifecycle_policy(self, policy: WeightUpdateLifecyclePolicy) -> None:
        self._weight_update_lifecycle_policy = policy

    def parse_update_info(self, update_dict: dict[str, Any]) -> NPUIPCWeightTransferUpdateInfo:
        """Parse update dict, deserializing pickled IPC handles if present.

        HTTP transport sends IPC handles as a base64-encoded pickle under the
        key ``ipc_handles_pickled``. This method deserializes them back into
        ``ipc_handles`` before constructing the typed dataclass, keeping
        serialization concerns out of the dataclass itself.

        Requires ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` because the
        payload is deserialized via ``pickle.loads``.
        """
        if "ipc_handles_pickled" in update_dict:
            if "ipc_handles" in update_dict:
                raise ValueError("Cannot specify both `ipc_handles` and `ipc_handles_pickled`")

            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise ValueError(
                    "Refusing to deserialize `ipc_handles_pickled` without VLLM_ALLOW_INSECURE_SERIALIZATION=1"
                )

            pickled = update_dict.pop("ipc_handles_pickled")
            update_dict["ipc_handles"] = pickle.loads(base64.b64decode(pickled))

        return super().parse_update_info(update_dict)

    def init_transfer_engine(self, init_info: NPUIPCWeightTransferInitInfo) -> None:
        """No initialization needed for NPU IPC backend."""
        pass

    def start_weight_update(self) -> None:
        self._weight_update_lifecycle_policy.start(self.model)

    def finish_weight_update(self) -> None:
        self._weight_update_lifecycle_policy.finish(self.model, self.model_config)

    def receive_weights(self, update_info: NPUIPCWeightTransferUpdateInfo) -> None:
        """Receive weights from the trainer via NPU IPC handles.

        Args:
            update_info: NPU IPC update info containing parameter names,
                dtypes, shapes, and IPC handles.
        """
        load_weights = self._weight_update_lifecycle_policy.make_load_weights(self.model)
        device_index = torch.accelerator.current_device_index()
        physical_npu_id = npu_generate_uuid()

        if update_info.packed:
            assert update_info.tensor_sizes is not None
            assert isinstance(update_info.ipc_handles, dict)
            weights = packed_npu_ipc_consumer(
                ipc_handle=update_info.ipc_handles,
                physical_npu_id=physical_npu_id,
                names=update_info.names,
                shapes=update_info.shapes,
                dtype_names=update_info.dtype_names,
                tensor_sizes=update_info.tensor_sizes,
                device_index=device_index,
            )
            load_weights(weights)
        else:
            # Lazy import: ``rebuild_npu_tensor`` lives in ``torch_npu`` and
            # must not be imported at module load time on non-NPU hosts.
            from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

            assert isinstance(update_info.ipc_handles, list)
            weights = []
            for name, ipc_handle in zip(
                update_info.names,
                update_info.ipc_handles,
            ):
                if physical_npu_id not in ipc_handle:
                    raise ValueError(
                        f"IPC handle not found for NPU UUID {physical_npu_id}. "
                        f"Available UUIDs: {list(ipc_handle.keys())}. "
                        f"This may indicate that the trainer and worker are "
                        f"not co-located on the same physical NPU (node)."
                    )

                args = ipc_handle[physical_npu_id]
                list_args = list(args)
                # Index 6 is the device_index parameter in torch's
                # IPC handle tuple (rebuild_npu_tensor). Update it
                # to the current device since the logical index can
                # differ between sender and receiver.
                list_args[6] = device_index
                weight = rebuild_npu_tensor(*list_args)
                weights.append((name, weight))

            load_weights(weights)

    def shutdown(self) -> None:
        pass

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | NPUIPCTrainerSendWeightsArgs,
    ) -> None:
        """Send weights from trainer to inference workers via NPU IPC.

        Supports two transport modes ('ray' and 'http') and two transfer
        strategies:
        - Non-packed (default): all weights in a single API call.
        - Packed (packed=True): chunked transfer with bounded NPU memory.

        For multi-NPU training, all ranks must call this method in
        parallel. IPC handles are all-gathered across ranks and merged
        so that each vLLM worker can find its own NPU UUID. Only rank 0
        sends the payload to vLLM.

        .. note::
            This method calls ``update_weights`` internally. The caller must
            handle ``pause`` / ``start_weight_update`` / ``finish_weight_update``
            / ``resume`` before and after this method.

        Args:
            iterator: Iterator of (name, tensor) pairs.
            trainer_args: NPUIPCTrainerSendWeightsArgs or equivalent dict.
        """
        args = NPUIPCTrainerSendWeightsArgs(**trainer_args) if isinstance(trainer_args, dict) else trainer_args
        npu_uuid = npu_generate_uuid()
        if args.packed:
            NPUIPCWeightTransferEngine._send_packed(iterator, args, npu_uuid)
        else:
            NPUIPCWeightTransferEngine._send_unpacked(iterator, args, npu_uuid)

    @staticmethod
    def _send_unpacked(
        iterator: Iterator[tuple[str, torch.Tensor]],
        args: NPUIPCTrainerSendWeightsArgs,
        npu_uuid: str,
    ) -> None:
        """Send all weights in a single API call (non-packed mode)."""
        names, dtype_names, shapes, ipc_handles, weight_refs = collect_parameter_metadata(iterator, npu_uuid)
        ipc_handles = TrainerProcessCoordinator.all_gather_and_merge_handles(ipc_handles)

        if TrainerProcessCoordinator.is_rank_zero():
            NPUIPCWeightTransferEngine._do_send(
                args=args,
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
                ipc_handles=ipc_handles,
            )

        TrainerProcessCoordinator.post_send_sync()
        _ = weight_refs

    @staticmethod
    def _send_packed(
        iterator: Iterator[tuple[str, torch.Tensor]],
        args: NPUIPCTrainerSendWeightsArgs,
        npu_uuid: str,
    ) -> None:
        """Send weights in bounded-memory chunks (packed mode)."""
        post_iter_func: Callable = default_parameter_tensor

        for chunk in packed_npu_ipc_producer(
            iterator=iterator,
            npu_uuid=npu_uuid,
            post_iter_func=post_iter_func,
            buffer_size_bytes=args.packed_buffer_size_bytes,
        ):
            ipc_handle = TrainerProcessCoordinator.all_gather_and_merge_handles([chunk["ipc_handle"]])[0]

            if TrainerProcessCoordinator.is_rank_zero():
                NPUIPCWeightTransferEngine._do_send(
                    args=args,
                    names=chunk["names"],
                    dtype_names=chunk["dtype_names"],
                    shapes=chunk["shapes"],
                    ipc_handles=ipc_handle,
                    tensor_sizes=chunk["tensor_sizes"],
                    packed=True,
                )

            TrainerProcessCoordinator.post_send_sync()

    @staticmethod
    def _do_send(
        args: NPUIPCTrainerSendWeightsArgs,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        ipc_handles: list[dict[str, tuple]] | dict[str, tuple],
        tensor_sizes: list[int] | None = None,
        packed: bool = False,
    ) -> None:
        """Send a single update payload via the configured transport."""
        update_fields: dict[str, Any] = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "packed": packed,
        }
        if tensor_sizes is not None:
            update_fields["tensor_sizes"] = tensor_sizes

        update_fields["ipc_handles"] = ipc_handles
        update_info = NPUIPCWeightTransferUpdateInfo(**update_fields)

        dispatch_update_info(
            args=args,
            update_info=update_info,
            update_fields=update_fields,
            ipc_handles=ipc_handles,
        )
