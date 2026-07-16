"""Patch CpuGpuBuffer with double-buffer rotation for async H2D safety.

Under async scheduling, the next iteration's CPU writes to ``buffer.cpu``
can race with the previous iteration's in-flight ``non_blocking=True``
H2D copy, corrupting the data seen by GPU consumers (e.g. query_start_loc).

This patch keeps ``self.cpu`` / ``self.gpu`` / ``self.np`` addresses fixed
(important for long-term references and graph capture/replay), but routes
``copy_to_gpu`` through two pre-allocated pinned CPU shadow buffers that
rotate each call. With ``max_concurrent_batches=1`` the double-buffer is
naturally safe: by the time buffer[0] is reused (two iterations later),
its previous H2D has certainly completed.
"""

import torch
from vllm.v1.utils import CpuGpuBuffer

_orig_init = CpuGpuBuffer.__init__
_orig_copy_to_gpu = CpuGpuBuffer.copy_to_gpu


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    pin_memory = self.cpu.is_pinned()
    self._shadow_buffers = [
        torch.empty_like(self.cpu, pin_memory=pin_memory) for _ in range(2)
    ]
    self._shadow_idx = 0


def _patched_copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
    shadow = self._shadow_buffers[self._shadow_idx]
    if n is None:
        shadow.copy_(self.cpu)
        result = self.gpu.copy_(shadow, non_blocking=True)
    else:
        shadow[:n].copy_(self.cpu[:n])
        result = self.gpu[:n].copy_(shadow[:n], non_blocking=True)
    self._shadow_idx = (self._shadow_idx + 1) % 2
    return result


CpuGpuBuffer.__init__ = _patched_init
CpuGpuBuffer.copy_to_gpu = _patched_copy_to_gpu
