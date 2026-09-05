"""YuanRong backend owned by one multiprocess Worker service."""

from typing import Any

import torch

from .....backend.yuanrong_backend import YuanrongBackend


class MPYuanrongBackend(YuanrongBackend):
    """Use YuanRong modes that do not leave process-owned registrations behind."""

    def __init__(self, parallel_config: Any, device_index: int, lazy_init: bool = False):
        del lazy_init
        self.device_index = device_index
        super().__init__(parallel_config)

    def set_device(self) -> None:
        torch.npu.set_device(self.device_index)

    def register_buffer(self, ptrs: list[int], lengths: list[int]) -> None:
        if self._needs_dev_mem_pregister:
            raise NotImplementedError(
                "YuanRong device-memory pre-registration cannot be safely released by an MP Worker service"
            )
        super().register_buffer(ptrs, lengths)

    def unregister_buffer(self) -> None:
        self._registered_buffers = None
        self._buffers_registered = False

    def close(self) -> None:
        self.unregister_buffer()
        close = getattr(self.store, "close", None)
        if callable(close):
            close()
