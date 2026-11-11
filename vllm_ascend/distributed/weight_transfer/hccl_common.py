# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TCP-rendezvous helpers for the HCCL weight-transfer backend.

Upstream's `nccl_common` builds a `PyNcclCommunicator` inside every helper, so
the HCCL variants cannot delegate to it. Everything else is reused from
upstream: the worker init payload serializer and the rank arithmetic (kept
inline in the engine, mirroring `nccl_common.worker_init_process_group`).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.weight_transfer.nccl_common import NCCLWeightTransferInitInfo

if TYPE_CHECKING:
    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator


def _validate_packed_params(info: Any) -> None:
    """Validate optional packed settings without conflating None and False."""
    if info.packed is not None and not isinstance(info.packed, bool):
        raise ValueError(f"`packed` must be a boolean or None, got {info.packed!r}")
    for name in ("packed_buffer_size_bytes", "packed_num_buffers"):
        value = getattr(info, name)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError(f"`{name}` must be a positive integer or None, got {value!r}")


@dataclass(kw_only=True)
class HCCLWeightTransferInitInfo(NCCLWeightTransferInitInfo):
    """Upstream worker init contract with TCP rendezvous and optional packed settings.

    HCCL requires the TCP endpoint; NCCL unique-ID rendezvous is unsupported.
    ``None`` packed settings remain distinguishable from explicit values so
    legacy trainers can still supply them per update.
    """

    master_address: str = field()
    master_port: int = field()
    packed: bool | None = None  # type: ignore[assignment]
    packed_buffer_size_bytes: int | None = None  # type: ignore[assignment]
    packed_num_buffers: int | None = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.nccl_unique_id_b64 is not None:
            raise ValueError("HCCL weight transfer requires TCP rendezvous; `nccl_unique_id_b64` is not supported")
        super().__post_init__()
        _validate_packed_params(self)


def stateless_init_process_group(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: int,
) -> "PyHcclCommunicator":
    """Create an HCCL communicator without touching torch's default group."""
    from vllm.distributed.utils import StatelessProcessGroup

    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=rank,
        world_size=world_size,
    )
    communicator = PyHcclCommunicator(pg, device=device)
    if not communicator.available or communicator.disabled:
        raise RuntimeError(
            "HCCL weight-transfer communicator is unavailable or disabled; "
            "refusing to continue as if the broadcast succeeded"
        )
    return communicator


def trainer_init(init_info: Any, rank: int = 0) -> "PyHcclCommunicator":
    """Open the trainer-side endpoint as ``rank``, defaulting to the sender."""
    fields = ("master_address", "master_port", "world_size")
    address, port, world_size = (
        init_info[name] if isinstance(init_info, dict) else getattr(init_info, name) for name in fields
    )
    device = torch.accelerator.current_device_index()
    if device is None:
        raise ValueError("HCCL trainer requires an explicit current NPU device")
    return stateless_init_process_group(address, port, rank, world_size, device)
