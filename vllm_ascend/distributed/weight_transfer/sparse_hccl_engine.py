# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HCCL transport for upstream sparse weight transfer engines."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

if TYPE_CHECKING:
    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

from vllm.distributed.weight_transfer.base import (
    VLLMWeightSyncClient,
    WeightSource,
)
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
    SparseNCCLTrainerInitInfo,
    SparseNCCLTrainerWeightTransferEngine,
    SparseNCCLWeightTransferEngine,
)
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
    SparseNCCLWeightTransferUpdateInfo as SparseHCCLWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
    SparseWeightPatch as SparseWeightPatch,
)

from vllm_ascend.distributed.weight_transfer import hccl_common
from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLWeightTransferEngine,
    HCCLWeightTransferInitInfo,
)

__all__ = [
    "SparseWeightPatch",
    "SparseHCCLTrainerInitInfo",
    "SparseHCCLWeightTransferUpdateInfo",
    "SparseHCCLWeightTransferEngine",
    "SparseHCCLTrainerWeightTransferEngine",
]


def _checkpoint_loader_layers(model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

    for layer in model.modules():
        if isinstance(getattr(layer, "quant_method", None), AscendUnquantizedFusedMoEMethod):
            yield layer


def _validate_checkpoint_weight_loader_view(layer: torch.nn.Module) -> None:
    import torch_npu

    parameters = [getattr(layer, name, None) for name in ("w13_weight", "w2_weight")]
    if any(parameter is None for parameter in parameters):
        raise NotImplementedError("Checkpoint sparse updates require unsplit ND expert weights")
    if any(torch_npu.get_npu_format(parameter) not in (0, 2) for parameter in parameters):
        raise NotImplementedError("Checkpoint sparse updates require ND expert weights")


@contextmanager
def _checkpoint_weight_loader_view(layer: torch.nn.Module) -> Iterator[None]:
    """Expose [expert, output, input] views over Ascend's transposed storage.

    Restore inference views even if loading fails; written values are not rolled back.
    """
    _validate_checkpoint_weight_loader_view(layer)
    parameters = [layer.w13_weight, layer.w2_weight]
    originals = [parameter.data for parameter in parameters]
    try:
        for parameter, original in zip(parameters, originals):
            parameter.data = original.transpose(1, 2)
        yield
    finally:
        for parameter, original in zip(parameters, originals):
            parameter.data = original


@contextmanager
def checkpoint_loader_views(model: torch.nn.Module) -> Iterator[None]:
    """Expose loader layouts while preserving the inference tensor storage."""
    with ExitStack() as stack:
        for layer in _checkpoint_loader_layers(model):
            stack.enter_context(_checkpoint_weight_loader_view(layer))
        yield


@dataclass
class SparseHCCLTrainerInitInfo(SparseNCCLTrainerInitInfo):
    backend: ClassVar[str] = "sparse_hccl"


class SparseHCCLWeightTransferEngine(SparseNCCLWeightTransferEngine):
    _stateless_init_process_group = staticmethod(hccl_common.stateless_init_process_group)
    init_info_cls = HCCLWeightTransferInitInfo
    update_info_cls = SparseHCCLWeightTransferUpdateInfo
    model_update_group: "PyHcclCommunicator | None"

    def init_transfer_engine(self, init_info: HCCLWeightTransferInitInfo) -> None:
        self._validate_model_support()
        HCCLWeightTransferEngine.init_transfer_engine(self, init_info)

    def _validate_model_support(self) -> None:
        if self.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("Sparse HCCL does not support pipeline parallelism")
        if self.parallel_config.enable_eplb:
            raise NotImplementedError("Sparse HCCL does not support EPLB")
        if self.model_config.quantization is not None:
            raise NotImplementedError("Sparse HCCL requires unquantized floating-point weights")
        for layer in _checkpoint_loader_layers(self.model):
            _validate_checkpoint_weight_loader_view(layer)
        for parameter in self.model.parameters():
            if parameter.device.type == "npu":
                import torch_npu

                if torch_npu.get_npu_format(parameter) not in (0, 2):
                    raise NotImplementedError("Sparse HCCL requires ND model weights (weight_nz_mode=0)")

    def update_weights(self, update_info: dict[str, Any]) -> None:
        # NOTE: Ascend transposes MoE weights for inference; temporarily expose the checkpoint loader layout
        # so upstream loading updates the existing storage correctly, then restore the inference views.
        with checkpoint_loader_views(self.model):
            # WARNING: The upstream receiver passes int32 indices to the checkpoint loader, whose bounds check
            # can reject valid indices for tensors with 2**31 elements. Pending an upstream bounds-check fix.
            super().update_weights(update_info)

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            # NOTE: Upstream sparse engines only clear the communicator reference; explicitly destroy
            # the HCCL communicator here to release native resources before dropping the reference.
            self.model_update_group.close()
            self.model_update_group = None


class SparseHCCLTrainerWeightTransferEngine(SparseNCCLTrainerWeightTransferEngine):
    init_info_cls = SparseHCCLTrainerInitInfo
    model_update_group: "PyHcclCommunicator | None"

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
        worker_init_info = HCCLWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=1,
            world_size=init_info.world_size,
        )

        with ExitStack() as cleanup:
            cleanup.callback(engine.shutdown)
            # Trainer and workers must enter HCCL initialization concurrently.
            with ThreadPoolExecutor(max_workers=1) as exe:
                future = exe.submit(
                    engine.client.init_weight_transfer_engine,
                    asdict(worker_init_info),
                )
                engine.model_update_group = hccl_common.trainer_init(init_info, rank=init_info.rank)
                future.result()
            cleanup.pop_all()

        return engine

    def _broadcast_chunk(self, patches: list[SparseWeightPatch]) -> None:
        from vllm_ascend.worker.v2.utils import torch_cuda_wrapper

        # The HCCL receiver skips zero-length collectives.
        patches = [patch for patch in patches if patch.indices.numel()]
        if not patches:
            return
        # NOTE: Map the upstream torch.cuda.current_stream() call to NPU explicitly,
        # so standalone trainers do not depend on worker initialization.
        with torch_cuda_wrapper():
            super()._broadcast_chunk(patches)

    def _post_send_sync(self) -> None:
        if torch.npu.is_available():
            torch.npu.current_stream().synchronize()

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            # NOTE: Upstream sparse engines only clear the communicator reference; explicitly destroy
            # the HCCL communicator here to release native resources before dropping the reference.
            self.model_update_group.close()
            self.model_update_group = None
