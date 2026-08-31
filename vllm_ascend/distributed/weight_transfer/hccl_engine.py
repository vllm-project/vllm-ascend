# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HCCL-based weight transfer engine."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferUpdateInfo,
)

from vllm_ascend.distributed.weight_transfer.hccl_common import (
    HCCLWeightTransferInitInfo,
    trainer_init,
    worker_init_process_group,
)
from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    packed_broadcast_consumer,
)

# Re-exported for backward compatibility; canonical home is hccl_common.
__all__ = [
    "HCCLWeightTransferInitInfo",
    "HCCLTrainerSendWeightsArgs",
    "HCCLWeightTransferUpdateInfo",
    "HCCLWeightTransferEngine",
]


@dataclass
class HCCLTrainerSendWeightsArgs:
    """Arguments for HCCL trainer_send_weights method."""

    group: Any
    """Process group (PyHcclCommunicator) for HCCL communication."""
    src: int = 0
    """Source rank (default 0, trainer is typically rank 0)."""
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor] | None = None
    """Optional function to apply to each (name, tensor) pair before broadcasting.
    If None, extracts just the tensor."""
    packed: bool = False
    """Whether to use packed tensor broadcasting for efficiency.
    When True, multiple tensors are batched together before broadcasting
    to reduce HCCL communication overhead."""
    stream: torch.npu.Stream | None = None
    """ACL stream to use for broadcasting if packed is False.
    If packed is True, new streams will be created for each buffer."""
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    """Size in bytes for each packed tensor buffer.
    Must match the value used in HCCLWeightTransferUpdateInfo."""
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS
    """Number of buffers for double/triple buffering during packed transfer.
    Must match the value used in HCCLWeightTransferUpdateInfo."""


@dataclass
class HCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for HCCL weight transfer backend."""

    names: list[str]
    """Names of the parameters to transfer (e.g. ``model.layers.0.weight``)."""
    dtype_names: list[str]
    """Torch dtype names (e.g. ``bfloat16``, ``float32``) for each parameter."""
    shapes: list[list[int]]
    """Shapes of each parameter as integer lists."""
    packed: bool = False
    """Whether to use packed tensor broadcasting for efficiency.
    When True, multiple tensors are batched together before broadcasting
    to reduce HCCL communication overhead."""
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    """Size in bytes for each packed tensor buffer.
    Both producer and consumer must use the same value."""
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS
    """Number of buffers for double/triple buffering during packed transfer.
    Both producer and consumer must use the same value."""

    def __post_init__(self):
        """Validate that all lists have the same length."""
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: got {len(self.shapes)} and {len(self.names)}"
            )


class HCCLWeightTransferEngine(WeightTransferEngine[HCCLWeightTransferInitInfo, HCCLWeightTransferUpdateInfo]):
    """
    Weight transfer engine using HCCL for communication between trainer and workers.

    This implementation uses HCCL broadcast operations to transfer weights from
    the trainer (rank 0) to all inference workers in a process group.
    """

    # Define backend-specific dataclass types
    init_info_cls = HCCLWeightTransferInitInfo
    update_info_cls = HCCLWeightTransferUpdateInfo

    def __init__(  # type: ignore[misc]
        self,
        config: WeightTransferConfig,
        vllm_config: VllmConfig,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self.model_update_group: PyHcclCommunicator | None = None  # type: ignore[no-redef]

    def start_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import (
            initialize_layerwise_reload,
        )

        initialize_layerwise_reload(self.model)

    def finish_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
        )

        finalize_layerwise_reload(self.model, self.model_config)

    def init_transfer_engine(self, init_info: HCCLWeightTransferInitInfo) -> None:
        """
        Initialize HCCL process group with the trainer.

        Args:
            init_info: HCCL initialization info containing master address, port,
                      rank offset, and world size
        """

        self.model_update_group = worker_init_process_group(
            init_info,
            self.parallel_config,
        )

    def receive_weights(
        self,
        update_info: HCCLWeightTransferUpdateInfo,
    ) -> None:
        """
        Receive weights from trainer via HCCL broadcast and load them incrementally.

        If update_info.packed is True, uses packed tensor broadcasting for
        efficient transfer of multiple weights in batches. Otherwise, uses simple
        one-by-one broadcasting.

        Args:
            update_info: HCCL update info containing parameter names, dtypes, shapes,
                        and packed flag
        """
        if self.model_update_group is None:
            raise RuntimeError("HCCL weight transfer not initialized. Call init_transfer_engine() first.")

        if update_info.packed:
            # Build iterator of (name, (shape, dtype)) from update_info
            def state_dict_info_iterator():
                for name, dtype_name, shape in zip(update_info.names, update_info.dtype_names, update_info.shapes):
                    dtype = getattr(torch, dtype_name)
                    yield (name, (shape, dtype))

            packed_broadcast_consumer(
                iterator=state_dict_info_iterator(),
                group=self.model_update_group,
                src=0,
                post_unpack_func=self.model.load_weights,
                buffer_size_bytes=update_info.packed_buffer_size_bytes,
                num_buffers=update_info.packed_num_buffers,
            )
        else:
            # Use simple one-by-one broadcasting
            for name, dtype_name, shape in zip(update_info.names, update_info.dtype_names, update_info.shapes):
                dtype = getattr(torch, dtype_name)
                weight = torch.empty(shape, dtype=dtype, device="npu")
                self.model_update_group.broadcast(weight, src=0, stream=torch.npu.current_stream())
                self.model.load_weights([(name, weight)])
                del weight

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            # Clean up the communicator by removing the reference
            self.model_update_group = None

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | HCCLTrainerSendWeightsArgs,
    ) -> None:
        """Broadcast weights from trainer to vLLM workers.

        Args:
            iterator: Iterator of model parameters. Returns (name, tensor) tuples
            trainer_args: Dictionary or HCCLTrainerSendWeightsArgs instance containing
                         HCCL-specific arguments. If a dict, should contain keys from
                         HCCLTrainerSendWeightsArgs.

        Example:
            >>> from vllm.distributed.weight_transfer.hccl_engine import (
            ...     HCCLWeightTransferEngine,
            ...     HCCLTrainerSendWeightsArgs,
            ... )
            >>> param_iter = ((n, p) for n, p in model.named_parameters())
            >>> args = HCCLTrainerSendWeightsArgs(group=group, packed=True)
            >>> HCCLWeightTransferEngine.trainer_send_weights(param_iter, args)
        """
        # Parse trainer args - accept either dict or dataclass instance
        if isinstance(trainer_args, dict):
            args = HCCLTrainerSendWeightsArgs(**trainer_args)
        else:
            args = trainer_args

        if args.post_iter_func is None:
            # Default: extract just the tensor from (name, tensor) tuple
            post_iter_func = lambda x: x[1]
        else:
            post_iter_func = args.post_iter_func

        if args.packed:
            # Use packed tensor broadcasting for efficiency
            from vllm_ascend.distributed.weight_transfer.packed_tensor import (
                packed_broadcast_producer,
            )

            packed_broadcast_producer(
                iterator=iterator,
                group=args.group,
                src=args.src,
                post_iter_func=post_iter_func,
                buffer_size_bytes=args.packed_buffer_size_bytes,
                num_buffers=args.packed_num_buffers,
            )
        else:
            # Use simple one-by-one broadcasting
            for item in iterator:
                tensor = post_iter_func(item)
                args.group.broadcast(
                    tensor,
                    src=args.src,
                    stream=args.stream or torch.npu.current_stream(),
                )

    # Trainer-side process-group setup. Delegates to the shared helper so the
    # sparse engine can reuse the exact same rendezvous without subclassing.
    trainer_init = staticmethod(trainer_init)
