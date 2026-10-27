from __future__ import annotations

import os
from collections.abc import Sequence

import torch
from vllm.logger import logger
from vllm.v1.worker.gpu import buffer_utils


def is_uva_available() -> bool:
    """Whether the platform can expose a real NPU view of pinned CPU memory."""
    return "pinned_mem_register:True" in os.environ.get("PYTORCH_NPU_ALLOC_CONF", {})


class UvaBufferWrapper:
    """Explicit-copy replacement for A's UvaBuffer on Ascend.

    ``cpu`` is the single source of truth and ``np`` is a plain numpy view of
    it.  A's UvaBufferPool performs ordinary CPU writes into those attributes
    and then calls ``uva(n)``.  Keep that contract simple: copy the requested
    prefix to the NPU synchronously in ``uva()``.  No per-row Python hooks and
    no hidden dirty-row bookkeeping are involved.
    """

    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self.np = self.cpu.numpy()
        self._use_real_uva = False
        view = None
        if is_uva_available():
            try:
                import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401

                view = torch.ops._C_ascend.get_npu_view_from_cpu_tensor(self.cpu)
            except Exception as exc:  # noqa: BLE001
                logger.warning_once("NPU UVA view is unavailable: %s", exc)
        if view is not None:
            self._uva = view
            self._use_real_uva = True
        else:
            self._uva = torch.zeros_like(self.cpu, device="npu")

    @property
    def uva(self):
        return self._uva

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            n = self.cpu.shape[0]
        if not self._use_real_uva:
            self._uva[:n].copy_(self.cpu[:n], non_blocking=False)
        return self._uva[:n]

    def uva_tensor(self, n: int | None = None) -> torch.Tensor:
        return self.copy_to_uva(n)


# A's UvaBufferPool calls ``buf.uva(n)`` as a method.  Keep the class-shaped
# contract while routing all writes through the explicit copy above.
UvaBufferWrapper.uva = UvaBufferWrapper.copy_to_uva  # type: ignore[method-assign]
buffer_utils.UvaBuffer = UvaBufferWrapper
