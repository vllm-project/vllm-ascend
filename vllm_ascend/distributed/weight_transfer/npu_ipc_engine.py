# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU IPC-based weight transfer engine using Ascend IPC for communication."""

import os
import pickle
import socket
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import pybase64 as base64
import requests
import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm import envs
from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)


@dataclass
class NPUIPCTrainerSendWeightsArgs:
    """Arguments for NPU IPC trainer_send_weights method."""

    mode: str
    """Transport mode: 'http' or 'ray'."""
    llm_handle: Any = None
    """Ray ObjectRef to LLM handle (required for 'ray' mode)."""
    url: str | None = None
    """Base URL for HTTP endpoint (required for 'http' mode)."""

    def __post_init__(self):
        """Validate that required arguments are provided for the selected mode."""
        if self.mode == "ray" and self.llm_handle is None:
            raise ValueError("llm_handle is required for 'ray' mode")
        if self.mode == "http" and self.url is None:
            raise ValueError("url is required for 'http' mode")
        if self.mode not in ("ray", "http"):
            raise ValueError(f"mode must be 'ray' or 'http', got {self.mode}")


@dataclass
class NPUIPCWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for NPU IPC weight transfer backend.

    No initialization needed for NPU IPC."""

    pass


@dataclass
class NPUIPCWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for NPU IPC weight transfer backend.

    Accepts IPC handles either directly via ``ipc_handles`` (Ray transport)
    or as a base64-encoded pickle via ``ipc_handles_pickled`` (HTTP transport).
    Exactly one of the two must be provided; if ``ipc_handles_pickled`` is set
    it is unpickled into ``ipc_handles`` during ``__post_init__``.
    """

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    ipc_handles: list[dict[str, tuple[Callable, tuple]]] | None = None
    """IPC handles mapping physical NPU UUID to (func, args) tuple.
    Each handle is a dictionary mapping NPU UUID strings to IPC handle tuples."""
    ipc_handles_pickled: str | None = None
    """Base64-encoded pickled IPC handles, used for HTTP transport."""

    def __post_init__(self):
        if self.ipc_handles_pickled is not None:
            if self.ipc_handles is not None:
                raise ValueError(
                    "Cannot specify both `ipc_handles` and `ipc_handles_pickled`"
                )

            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise ValueError(
                    "Refusing to deserialize `ipc_handles_pickled` without "
                    "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
                )

            self.ipc_handles = pickle.loads(
                base64.b64decode(self.ipc_handles_pickled)
            )
            self.ipc_handles_pickled = None

        if self.ipc_handles is None:
            raise ValueError(
                "Either `ipc_handles` or `ipc_handles_pickled` must be provided"
            )

        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {len(self.names)}"
            )
        if len(self.ipc_handles) != num_params:
            raise ValueError(
                f"`ipc_handles` should be of the same size as `names`: "
                f"got {len(self.ipc_handles)} and {len(self.names)}"
            )


@lru_cache(maxsize=1)
def get_ip() -> str:
    try:
        # try to get ip from network interface
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:  # noqa: BLE001
        # fallback to get ip from hostname
        return socket.gethostbyname(socket.gethostname())


@lru_cache(maxsize=1)
def npu_generate_uuid() -> str:
    """Generate a unique identifier for the current process's physical NPU chip.

    Returns ``{host_ip}-{physical_chip_id}`` where ``host_ip`` is the local
    machine's IP address and ``physical_chip_id`` is derived from the current
    logical device index mapped through ``ASCEND_RT_VISIBLE_DEVICES``.

    On Ascend NPU, ``torch.accelerator.current_device_index()`` returns the
    *logical* device index. When ``ASCEND_RT_VISIBLE_DEVICES`` is set, it
    maps logical indices to physical chip IDs (e.g., ``ASCEND_RT_VISIBLE_DEVICES=2,3``
    means logical device 0 → physical chip 2, logical device 1 → physical chip 3).
    If the env var is not set, the logical index is used directly as the
    physical chip ID (identity mapping).

    The result is cached because it is constant for the lifetime of the
    process. Both the trainer and inference worker processes co-located
    on the same physical NPU chip will produce the same UUID, which is
    required for NPU IPC handle matching.
    """
    logical_device = torch.accelerator.current_device_index()
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", None)
    if visible_devices:
        physical_device = int(visible_devices.split(",")[logical_device].strip())
    else:
        physical_device = logical_device
    return f"{get_ip()}-{physical_device}"


class NPUIPCWeightTransferEngine(
    WeightTransferEngine[
        NPUIPCWeightTransferInitInfo, NPUIPCWeightTransferUpdateInfo
    ]
):
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

    def __init__(
        self, config: WeightTransferConfig, parallel_config: ParallelConfig
    ) -> None:
        super().__init__(config, parallel_config)

    def init_transfer_engine(
        self, init_info: NPUIPCWeightTransferInitInfo
    ) -> None:
        """No initialization needed for NPU IPC backend."""
        pass

    def receive_weights(
        self,
        update_info: NPUIPCWeightTransferUpdateInfo,
        load_weights: Callable[[list[tuple[str, torch.Tensor]]], None],
    ) -> None:
        """Receive weights from the trainer via NPU IPC handles.

        Args:
            update_info: NPU IPC update info containing parameter names,
                dtypes, shapes, and IPC handles. Each IPC handle is a
                mapping between physical NPU UUID and the IPC handle tuple
                (func, args).
            load_weights: Callable that loads weights into the model.
        """
        assert update_info.ipc_handles is not None
        device_index = torch.accelerator.current_device_index()
        physical_npu_id = npu_generate_uuid()

        weights = []
        for name, _dtype_name, _shape, ipc_handle in zip(
            update_info.names,
            update_info.dtype_names,
            update_info.shapes,
            update_info.ipc_handles,
        ):
            if physical_npu_id not in ipc_handle:
                raise ValueError(
                    f"IPC handle not found for NPU UUID {physical_npu_id}. "
                    f"Available UUIDs: {list(ipc_handle.keys())}. "
                    f"This may indicate that the trainer and worker are "
                    f"not co-located on the same physical NPU (node)."
                )

            handle = ipc_handle[physical_npu_id]

            func, args = handle
            list_args = list(args)  # type: ignore
            # Index 6 is the device_index parameter in torch's
            # IPC handle tuple (rebuild_npu_tensor). Update it
            # to the current device since the logical index can
            # differ between sender and receiver.
            list_args[6] = device_index
            weight = func(*list_args)  # type: ignore
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

        Supports two modes:
        - 'ray': Sends weights via Ray RPC to a Ray-based LLM handle
        - 'http': Sends weights via HTTP POST to a vLLM HTTP server

        Args:
            iterator: Iterator of model parameters. Returns (name, tensor)
                tuples. Tensors should be on the same NPU as the inference
                workers.
            trainer_args: Dictionary or NPUIPCTrainerSendWeightsArgs instance.

        Example (Ray mode):
            >>> from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
            ...     NPUIPCWeightTransferEngine,
            ...     NPUIPCTrainerSendWeightsArgs,
            ... )
            >>> param_iter = ((n, p) for n, p in model.named_parameters())
            >>> args = NPUIPCTrainerSendWeightsArgs(
            ...     mode="ray", llm_handle=llm_handle
            ... )
            >>> NPUIPCWeightTransferEngine.trainer_send_weights(
            ...     param_iter, asdict(args)
            ... )

        Example (HTTP mode):
            >>> args = NPUIPCTrainerSendWeightsArgs(
            ...     mode="http", url="http://localhost:8000"
            ... )
            >>> NPUIPCWeightTransferEngine.trainer_send_weights(
            ...     param_iter, asdict(args)
            ... )
        """
        if isinstance(trainer_args, dict):
            args = NPUIPCTrainerSendWeightsArgs(**trainer_args)
        else:
            args = trainer_args

        # Get physical NPU UUID
        npu_uuid = npu_generate_uuid()

        names = []
        dtype_names = []
        shapes = []
        ipc_handles = []

        for name, tensor in iterator:
            names.append(name)
            dtype_names.append(str(tensor.dtype).split(".")[-1])
            shapes.append(list(tensor.shape))

            # Create IPC handle for this weight tensor
            # The tensor must remain in memory for IPC to work
            weight = tensor.detach().contiguous()
            ipc_handle = reduce_tensor(weight)
            ipc_handles.append({npu_uuid: ipc_handle})

        if args.mode == "ray":
            import ray

            update_info = asdict(
                NPUIPCWeightTransferUpdateInfo(
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    ipc_handles=ipc_handles,
                )
            )
            ray.get(
                args.llm_handle.update_weights.remote(
                    dict(update_info=update_info)
                )
            )
        elif args.mode == "http":
            pickled_handles = base64.b64encode(
                pickle.dumps(ipc_handles)
            ).decode("utf-8")

            url = f"{args.url}/update_weights"
            payload = {
                "update_info": {
                    "names": names,
                    "dtype_names": dtype_names,
                    "shapes": shapes,
                    "ipc_handles_pickled": pickled_handles,
                }
            }
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
