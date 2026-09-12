"""Mooncake backend state supplied by the model worker to the transfer process."""

import torch

from ..backend.mooncake_backend import MooncakeBackend


class MPMooncakeBackend(MooncakeBackend):
    """Use worker-provided identity without joining its distributed groups."""

    def __init__(self, device_index: int, global_rank: int, lazy_init: bool = False):
        self._device_index = device_index
        super().__init__(None, lazy_init=lazy_init, process_global_rank=global_rank)  # type: ignore[arg-type]

    def set_device(self) -> None:
        torch.npu.set_device(self._device_index)


__all__ = ["MPMooncakeBackend"]
