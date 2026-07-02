# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HybridTransferTracker: PD separation control layer.
Decouples data payload from state readiness signaling.
Like a gate controlling when KV and Mamba channels are both ready.

Architecture:
  P node sends: KV blocks (data channel) + Mamba state (data channel)
  D node receives: TransferTracker (control channel) independently tracks
  each channel's readiness. Only when BOTH are ready does decode begin.

This prevents the common mistake of embedding control signals in payload
dicts, which creates race conditions under async transfer.
"""

import threading
from collections.abc import Callable


class HybridTransferTracker:
    """原子就绪屏障: KV 和 Mamba 独立通道都到达后才通知调度器。

    与数据通道完全解耦 — 不修改 payload 结构。
    每个请求创建一个实例, 线程安全。
    """

    def __init__(self, request_id: str, on_ready: Callable | None = None):
        self.request_id = request_id
        self._kv_ready = threading.Event()
        self._mamba_ready = threading.Event()
        self._on_ready = on_ready

    @property
    def is_ready(self) -> bool:
        return self._kv_ready.is_set() and self._mamba_ready.is_set()

    def mark_kv_received(self) -> None:
        """KV blocks 通道就绪"""
        self._kv_ready.set()
        self._notify_if_ready()

    def mark_mamba_received(self) -> None:
        """Mamba state 通道就绪"""
        self._mamba_ready.set()
        self._notify_if_ready()

    def _notify_if_ready(self) -> None:
        if self.is_ready and self._on_ready:
            self._on_ready(self.request_id)

    def reset(self) -> None:
        """重置状态 (用于请求复用)"""
        self._kv_ready.clear()
        self._mamba_ready.clear()
