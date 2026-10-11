# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse HCCL weight transfer engine.

Sparse patches use checkpoint names, shapes, and flat indices. The model's native
weight loader maps them to rank-local runtime parameters, including TP shards and
packed parameters.
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngine,
    WeightTransferUpdateInfo,
)
from vllm.model_executor.model_loader.checkpoint_weight_patch import (
    CheckpointWeightPatch,
    load_checkpoint_weight_patches,
)

from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLWeightTransferEngine,
    HCCLWeightTransferInitInfo,
)
from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import (
    SPARSE_HCCL_VALUE_DTYPES,
    SparseWeightPatch,
    validate_sparse_patch,
)

__all__ = [
    "SparseWeightPatch",
    "SparseHCCLTrainerInitInfo",
    "SparseHCCLWeightTransferUpdateInfo",
    "SparseHCCLWeightTransferEngine",
    "SparseHCCLTrainerWeightTransferEngine",
]


def checkpoint_loader_views(model: torch.nn.Module) -> ExitStack:
    """Expose loader layouts while preserving the inference tensor storage."""
    stack = ExitStack()
    try:
        for layer in model.modules():
            method = getattr(layer, "quant_method", None)
            view = getattr(method, "checkpoint_weight_loader_view", None)
            if view is not None:
                stack.enter_context(view(layer))
    except BaseException:
        stack.close()
        raise
    return stack


@dataclass
class SparseHCCLTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for the sparse HCCL weight transfer backend.

    Same rendezvous shape as the dense HCCL backend (the sender opens its
    endpoint as HCCL rank 0), but with no packed wire params: sparse transfers
    are never packed. `backend` is the factory dispatch key."""

    backend: ClassVar[str] = "sparse_hccl"

    master_address: str
    master_port: int
    world_size: int


@dataclass
class SparseHCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for the sparse HCCL weight transfer backend."""

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    num_updates_list: list[int]
    """Number of sparse entries to receive for each parameter in ``names``."""

    def __post_init__(self) -> None:
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
        if len(self.num_updates_list) == 0:
            raise ValueError("`num_updates_list` cannot be empty for sparse updates")
        if len(self.num_updates_list) != num_params:
            raise ValueError(
                f"`num_updates_list` should be of the same size as `names`: "
                f"got {len(self.num_updates_list)} and {len(self.names)}"
            )
        if any(num_updates < 0 for num_updates in self.num_updates_list):
            raise ValueError("Sparse `num_updates_list` entries must be non-negative")
        for name, dtype_name, shape, count in zip(
            self.names, self.dtype_names, self.shapes, self.num_updates_list, strict=True
        ):
            dtype = getattr(torch, dtype_name, None)
            if not isinstance(name, str) or not name:
                raise ValueError("Sparse checkpoint name must be non-empty")
            if dtype not in SPARSE_HCCL_VALUE_DTYPES:
                raise ValueError(f"Sparse checkpoint dtype must be floating: {name}")
            if any(type(dim) is not int or dim < 0 for dim in shape):
                raise ValueError(f"Invalid checkpoint shape: {name}")
            if type(count) is not int:
                raise ValueError(f"Sparse update count must be an integer: {name}")


class SparseHCCLWeightTransferEngine(
    WeightTransferEngine[HCCLWeightTransferInitInfo, SparseHCCLWeightTransferUpdateInfo]
):
    """
    Sparse weight transfer engine using HCCL.

    Receives checkpoint-coordinate patches broadcast from the trainer and applies
    them through the model's native weight loader. Sparse updates modify initialized
    model tensors in place, so the layerwise reload lifecycle is not used.
    """

    init_info_cls = HCCLWeightTransferInitInfo
    update_info_cls = SparseHCCLWeightTransferUpdateInfo
    supports_draft_weight_update = False

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self.model_update_group: PyHcclCommunicator | None = None

    def init_transfer_engine(self, init_info: HCCLWeightTransferInitInfo) -> None:
        """Initialize the HCCL process group with the trainer."""
        self.start_weight_update()
        workers = self.parallel_config.world_size * self.parallel_config.data_parallel_size
        if init_info.rank_offset != 1 or init_info.world_size != workers + 1:
            raise ValueError(
                "Sparse HCCL requires one trainer and all TP/DP workers: world_size=1+TP*DP, rank_offset=1"
            )
        HCCLWeightTransferEngine.init_transfer_engine(self, init_info)

    def start_weight_update(self) -> None:
        """Validate support without reinitializing the existing model weights."""
        if self.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("Sparse HCCL requires PP=1")
        if self.parallel_config.enable_eplb:
            raise NotImplementedError("Sparse HCCL does not support EPLB")
        if self.model_config.quantization is not None:
            raise NotImplementedError("Sparse HCCL requires unquantized floating-point weights")
        for layer in self.model.modules():
            method = getattr(layer, "quant_method", None)
            validate = getattr(method, "validate_checkpoint_weight_loader_view", None)
            if validate is not None:
                validate(layer)
        for parameter in self.model.parameters():
            if parameter.device.type == "npu":
                import torch_npu

                if torch_npu.get_npu_format(parameter) not in (0, 2):
                    raise NotImplementedError("Sparse HCCL requires ND model weights (weight_nz_mode=0)")

    def finish_weight_update(self) -> None:
        """No-op: sparse patches are applied in place, no layerwise reload."""
        pass

    def receive_weights(self, update_info: SparseHCCLWeightTransferUpdateInfo) -> None:
        """Receive sparse flat-index patches from the trainer and apply them."""
        if self.model_update_group is None:
            raise RuntimeError("HCCL weight transfer not initialized. Call init_transfer_engine() first.")

        # Receive on the communicator's device rather than relying on the
        # ambient current device in the RPC thread.
        device = self.model_update_group.device
        with torch.npu.device(device):
            stream = torch.npu.current_stream(device=device)
            patches = []
            for name, dtype_name, shape, num_updates in zip(
                update_info.names,
                update_info.dtype_names,
                update_info.shapes,
                update_info.num_updates_list,
                strict=True,
            ):
                dtype = getattr(torch, dtype_name)
                indices = torch.empty(num_updates, dtype=torch.int32, device=device)
                values = torch.empty(num_updates, dtype=dtype, device=device)
                if num_updates:
                    self.model_update_group.broadcast(indices, src=0, stream=stream)
                    self.model_update_group.broadcast(values, src=0, stream=stream)
                patches.append(
                    CheckpointWeightPatch(
                        name=name,
                        shape=tuple(shape),
                        dtype=dtype,
                        values=values,
                        indices=indices,
                    )
                )
            with checkpoint_loader_views(self.model):
                load_checkpoint_weight_patches(self.model, patches)

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group.close()
            self.model_update_group = None


class SparseHCCLTrainerWeightTransferEngine(TrainerWeightTransferEngine[SparseHCCLTrainerInitInfo]):
    """Trainer-side sparse HCCL weight transfer engine.

    Broadcasts flat-index (indices, values) patches from HCCL rank 0 while the
    inference-side `update_weights` runs concurrently on a side thread (the
    worker's recvs rendezvous inside the same HCCL broadcasts). `send_weights`
    owns a complete one-shot lifecycle. RL infrastructure that owns the generic
    client lifecycle can call `send_weight_chunk` between one start and finish.

    Sparse patches differ every round, so they are not a stable `WeightSource`:
    the engine takes no `source`, and patches are passed directly to the send
    methods. An empty patch list is a no-op.

    Only the designated trainer sender joins the transfer group; other trainer
    ranks skip sparse sends.
    """

    init_info_cls = SparseHCCLTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
        is_sender: bool = True,
    ) -> None:
        # Sparse is a delta backend: it takes per-round patches via
        # send_weights, so a `source` would silently never be sent. The
        # parameter exists only to match the base/factory signature.
        if source is not None:
            raise ValueError(
                "Sparse HCCL weight transfer takes no WeightSource; pass each "
                "round's patches to send_weights(patches) instead."
            )
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.model_update_group: PyHcclCommunicator | None = None

    @classmethod
    def trainer_init(
        cls,
        init_info: SparseHCCLTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> "SparseHCCLTrainerWeightTransferEngine":
        engine = cls(client=client, source=source, is_sender=init_info.is_sender)
        if not engine.is_sender:
            return engine
        if init_info.world_size < 2:
            raise ValueError("Sparse HCCL needs a trainer and at least one worker")

        # Workers sit at rank_offset 1, after the single trainer sender rank 0.
        # Sparse transfers are never packed, so the worker keeps the unpacked
        # defaults on its init info.
        worker_init_info = HCCLWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=1,
            world_size=init_info.world_size,
        )

        # The inference workers block inside init_weight_transfer_engine waiting
        # for the HCCL rendezvous, so we kick that off on a side thread while we
        # open the trainer endpoint (rank 0); both sides must rendezvous together.
        with ThreadPoolExecutor(max_workers=1) as exe:
            future = exe.submit(
                engine.client.init_weight_transfer_engine,
                asdict(worker_init_info),
            )
            # Open the trainer endpoint as HCCL rank 0 on the current device
            # (the init info satisfies the helper's rendezvous protocol).
            engine.model_update_group = HCCLWeightTransferEngine.trainer_init(asdict(init_info))
            future.result()  # surface any inference-side init error

        return engine

    def send_weights(self, patches: Iterable[SparseWeightPatch] | None = None) -> None:
        """Broadcast one sparse update through a one-shot lifecycle."""
        patches = self._prepare_patches(patches)
        if not patches:
            return

        self.client.start_weight_update()
        self._broadcast_chunk(patches)
        self.client.finish_weight_update()

    def send_weight_chunk(self, patches: Iterable[SparseWeightPatch] | None = None) -> None:
        """Broadcast one chunk inside a caller-owned weight update lifecycle."""
        patches = self._prepare_patches(patches)
        if not patches:
            return
        self._broadcast_chunk(patches)

    def _prepare_patches(self, patches: Iterable[SparseWeightPatch] | None) -> list[SparseWeightPatch]:
        if not self.is_sender:
            return []

        prepared = list(patches) if patches is not None else []
        if not prepared:
            return []
        if self.model_update_group is None:
            raise RuntimeError("trainer_init() must be called before sending weights.")
        for patch in prepared:
            self._validate_patch(patch)
        device = self.model_update_group.device
        # Normalize views and CPU payloads before workers enter collectives.
        with torch.npu.device(device):
            return [
                SparseWeightPatch(
                    name=patch.name,
                    full_shape=tuple(patch.full_shape),
                    indices=patch.indices.to(device=device).contiguous(),
                    values=patch.values.to(device=device).contiguous(),
                )
                for patch in prepared
            ]

    def _broadcast_chunk(self, patches: list[SparseWeightPatch]) -> None:
        assert self.model_update_group is not None
        device = self.model_update_group.device
        with torch.npu.device(device):
            update_info = SparseHCCLWeightTransferUpdateInfo(
                names=[patch.name for patch in patches],
                dtype_names=[str(patch.values.dtype).split(".")[-1] for patch in patches],
                shapes=[list(patch.full_shape) for patch in patches],
                num_updates_list=[patch.indices.numel() for patch in patches],
            )

            # update_weights (workers receive) must run concurrently with the
            # trainer-side broadcasts — both rendezvous inside the same HCCL calls.
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(self.client.update_weights, asdict(update_info))
                # Surface an RPC that failed before the worker entered HCCL rather
                # than broadcasting to a peer that will never receive.
                if future.done():
                    future.result()
                stream = torch.npu.current_stream()
                for patch in patches:
                    if not patch.indices.numel():
                        continue
                    self.model_update_group.broadcast(patch.indices, src=0, stream=stream)
                    self.model_update_group.broadcast(patch.values, src=0, stream=stream)
                future.result()  # surface inference-side errors
            finally:
                # A failed broadcast can leave the RPC blocked in the matching
                # receive. Waiting for that thread would hide the original error.
                executor.shutdown(wait=False, cancel_futures=True)
            self._post_send_sync()

    @staticmethod
    def _validate_patch(patch: SparseWeightPatch) -> None:
        """Reject a malformed patch before starting the HCCL transfer."""
        validate_sparse_patch(patch)

    def _post_send_sync(self) -> None:
        """Wait for the broadcasts to land before returning, so a caller may
        rebuild or free the patch tensors as soon as a send method returns rather
        than relying on same-stream ordering."""
        if torch.npu.is_available():
            torch.npu.current_stream().synchronize()

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group.close()
            self.model_update_group = None
