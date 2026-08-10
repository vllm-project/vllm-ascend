# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse runtime weight transfer through Ascend NPU IPC handles."""

import pickle
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

import pybase64 as base64
import requests
import torch
from torch.multiprocessing.reductions import reduce_tensor
from vllm import envs
from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferUpdateInfo,
)

from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCTrainerSendWeightsArgs,
    NPUIPCWeightTransferInitInfo,
)
from vllm_ascend.distributed.weight_transfer.npu_ipc_common import (
    all_gather_and_merge_handles,
    is_rank_zero,
    npu_generate_uuid,
    post_send_sync,
)
from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import (
    SparseWeightPatch,
    apply_sparse_patch,
)

__all__ = [
    "SparseNPUIPCTrainerSendWeightsArgs",
    "SparseNPUIPCWeightTransferInitInfo",
    "SparseNPUIPCWeightTransferUpdateInfo",
    "SparseNPUIPCWeightTransferEngine",
]


@dataclass
class SparseNPUIPCTrainerSendWeightsArgs(NPUIPCTrainerSendWeightsArgs):
    """Trainer arguments for sparse NPU IPC updates."""

    send_mode: str | Callable[["SparseNPUIPCWeightTransferUpdateInfo"], None]
    parameter_shapes: dict[str, list[int]] | None = None


@dataclass
class SparseNPUIPCWeightTransferInitInfo(NPUIPCWeightTransferInitInfo):
    """Sparse NPU IPC needs no data-plane rendezvous."""


@dataclass
class SparseNPUIPCWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Metadata and paired IPC handles for sparse runtime patches."""

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    num_updates_list: list[int]
    indices_ipc_handles: list[dict[str, tuple]] | None = None
    values_ipc_handles: list[dict[str, tuple]] | None = None
    indices_ipc_handles_pickled: str | None = None
    values_ipc_handles_pickled: str | None = None
    packed: bool = False

    def __post_init__(self) -> None:
        self.indices_ipc_handles = self._deserialize_handles(
            "indices", self.indices_ipc_handles, self.indices_ipc_handles_pickled
        )
        self.values_ipc_handles = self._deserialize_handles(
            "values", self.values_ipc_handles, self.values_ipc_handles_pickled
        )
        if self.packed:
            raise ValueError("Sparse NPU IPC updates do not support `packed=True`")

        num_params = len(self.names)
        fields = {
            "dtype_names": self.dtype_names,
            "shapes": self.shapes,
            "num_updates_list": self.num_updates_list,
            "indices_ipc_handles": self.indices_ipc_handles,
            "values_ipc_handles": self.values_ipc_handles,
        }
        if not self.num_updates_list:
            raise ValueError("`num_updates_list` cannot be empty for sparse updates")
        for field_name, values in fields.items():
            if values is None or len(values) != num_params:
                actual = 0 if values is None else len(values)
                raise ValueError(
                    f"`{field_name}` should be of the same size as `names`: "
                    f"got {actual} and {num_params}"
                )
        if any(count < 0 for count in self.num_updates_list):
            raise ValueError("Sparse `num_updates_list` entries must be non-negative")

    @staticmethod
    def _deserialize_handles(
        label: str,
        handles: list[dict[str, tuple]] | None,
        encoded: str | None,
    ) -> list[dict[str, tuple]]:
        if handles is not None and encoded is not None:
            raise ValueError(
                f"Cannot specify both `{label}_ipc_handles` and "
                f"`{label}_ipc_handles_pickled`"
            )
        if encoded is not None:
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise ValueError(
                    "Refusing to deserialize pickled IPC handles without "
                    "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
                )
            handles = pickle.loads(base64.b64decode(encoded))
        if handles is None:
            raise ValueError(f"Sparse NPU IPC requires `{label}_ipc_handles`")
        return handles


def _rebuild_sparse_ipc_tensor(
    handle: dict[str, tuple],
    physical_npu_id: str,
    device_index: int,
) -> torch.Tensor:
    if physical_npu_id not in handle:
        raise ValueError(
            f"IPC handle not found for NPU UUID {physical_npu_id}. "
            f"Available UUIDs: {list(handle.keys())}. Trainer and worker must "
            "share the same physical NPU."
        )
    from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

    rebuild_args = list(handle[physical_npu_id])
    rebuild_args[6] = device_index
    return rebuild_npu_tensor(*rebuild_args)


class SparseNPUIPCWeightTransferEngine(
    WeightTransferEngine[
        SparseNPUIPCWeightTransferInitInfo,
        SparseNPUIPCWeightTransferUpdateInfo,
    ]
):
    """Apply sparse flat-index patches shared through NPU IPC."""

    init_info_cls = SparseNPUIPCWeightTransferInitInfo
    update_info_cls = SparseNPUIPCWeightTransferUpdateInfo
    supports_draft_weight_update = False

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: VllmConfig,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)

    def init_transfer_engine(
        self, init_info: SparseNPUIPCWeightTransferInitInfo
    ) -> None:
        pass

    def start_weight_update(self) -> None:
        if self.parallel_config.world_size != 1:
            raise NotImplementedError(
                "Sparse weight updates currently require TP=1 and PP=1"
            )

    def finish_weight_update(self) -> None:
        pass

    def receive_weights(
        self, update_info: SparseNPUIPCWeightTransferUpdateInfo
    ) -> None:
        device_index = self.device.index
        if device_index is None:
            raise ValueError("Sparse NPU IPC requires an indexed NPU device")
        physical_npu_id = npu_generate_uuid(device_index)
        assert update_info.indices_ipc_handles is not None
        assert update_info.values_ipc_handles is not None

        for name, dtype_name, shape, count, index_handle, value_handle in zip(
            update_info.names,
            update_info.dtype_names,
            update_info.shapes,
            update_info.num_updates_list,
            update_info.indices_ipc_handles,
            update_info.values_ipc_handles,
        ):
            indices = _rebuild_sparse_ipc_tensor(
                index_handle, physical_npu_id, device_index
            )
            values = _rebuild_sparse_ipc_tensor(
                value_handle, physical_npu_id, device_index
            )
            if indices.numel() != count or values.numel() != count:
                raise ValueError(
                    f"Rebuilt sparse patch length does not match declared count "
                    f"{count} for {name}"
                )
            actual_dtype_name = str(values.dtype).split(".")[-1]
            if actual_dtype_name != dtype_name:
                raise ValueError(
                    f"Rebuilt sparse values dtype {actual_dtype_name} does not "
                    f"match declared dtype {dtype_name} for {name}"
                )
            apply_sparse_patch(
                self.model,
                SparseWeightPatch(name, indices, values),
                expected_shape=shape,
            )

    def shutdown(self) -> None:
        pass

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[SparseWeightPatch],
        trainer_args: dict[str, Any] | SparseNPUIPCTrainerSendWeightsArgs,
    ) -> None:
        args = (
            SparseNPUIPCTrainerSendWeightsArgs(**trainer_args)
            if isinstance(trainer_args, dict)
            else trainer_args
        )
        if args.packed:
            raise ValueError("Sparse NPU IPC updates do not support `packed=True`")
        if args.parameter_shapes is None:
            raise ValueError("Sparse NPU IPC requires `parameter_shapes`")

        npu_uuid = npu_generate_uuid()
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        counts: list[int] = []
        index_handles: list[dict[str, tuple]] = []
        value_handles: list[dict[str, tuple]] = []
        tensor_refs: list[torch.Tensor] = []

        for patch in iterator:
            if patch.name not in args.parameter_shapes:
                raise ValueError(f"Missing parameter shape for {patch.name}")
            indices = patch.indices.detach().contiguous()
            values = patch.values.detach().contiguous()
            if indices.dtype != torch.int32 or indices.ndim != 1:
                raise ValueError("Sparse NPU IPC requires 1D int32 indices")
            if values.ndim != 1 or values.numel() != indices.numel():
                raise ValueError("Sparse indices and values must be matching 1D tensors")
            tensor_refs.extend((indices, values))
            _, index_args = reduce_tensor(indices)
            _, value_args = reduce_tensor(values)
            names.append(patch.name)
            dtype_names.append(str(values.dtype).split(".")[-1])
            shapes.append(args.parameter_shapes[patch.name])
            counts.append(indices.numel())
            index_handles.append({npu_uuid: index_args})
            value_handles.append({npu_uuid: value_args})

        index_handles = all_gather_and_merge_handles(index_handles)
        value_handles = all_gather_and_merge_handles(value_handles)
        if is_rank_zero():
            SparseNPUIPCWeightTransferEngine._do_send(
                args,
                names,
                dtype_names,
                shapes,
                counts,
                index_handles,
                value_handles,
            )
        post_send_sync()
        del tensor_refs

    @staticmethod
    def _do_send(
        args: SparseNPUIPCTrainerSendWeightsArgs,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        counts: list[int],
        index_handles: list[dict[str, tuple]],
        value_handles: list[dict[str, tuple]],
    ) -> None:
        fields: dict[str, Any] = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "num_updates_list": counts,
            "indices_ipc_handles": index_handles,
            "values_ipc_handles": value_handles,
        }
        update_info = SparseNPUIPCWeightTransferUpdateInfo(**fields)
        if callable(args.send_mode):
            args.send_mode(update_info)
        elif args.send_mode == "ray":
            import ray

            handles = (
                args.llm_handle
                if isinstance(args.llm_handle, list)
                else [args.llm_handle]
            )
            ray.get(
                [
                    handle.update_weights.remote(
                        dict(update_info=asdict(update_info))
                    )
                    for handle in handles
                ]
            )
        elif args.send_mode == "http":
            http_fields = {
                key: value
                for key, value in fields.items()
                if key not in {"indices_ipc_handles", "values_ipc_handles"}
            }
            http_fields["indices_ipc_handles_pickled"] = base64.b64encode(
                pickle.dumps(index_handles)
            ).decode("utf-8")
            http_fields["values_ipc_handles_pickled"] = base64.b64encode(
                pickle.dumps(value_handles)
            ).decode("utf-8")
            response = requests.post(
                f"{args.url}/update_weights",
                json={"update_info": http_fields},
                timeout=300,
            )
            response.raise_for_status()
        else:
            raise ValueError(f"Unsupported send_mode: {args.send_mode}")
