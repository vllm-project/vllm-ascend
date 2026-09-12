# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import GroupCoordinator


class BroadcastKVPPTransport:
    """Broadcast allocated cache bytes using the existing KVPP device group."""

    def __init__(
        self,
        kvpp_group: GroupCoordinator,
        layer_owner_ranks: dict[str, int],
        layer_buffers: dict[str, torch.Tensor],
    ) -> None:
        self._device_group = kvpp_group.device_group
        self._owner_global_ranks = {name: kvpp_group.ranks[owner] for name, owner in layer_owner_ranks.items()}
        self._layer_buffers = layer_buffers

    def prefetch(self, layer_name: str, cache_ready: Any, transfer_stream: Any) -> None:
        with torch.npu.stream(transfer_stream):
            transfer_stream.wait_event(cache_ready)
            work = dist.broadcast(
                self._layer_buffers[layer_name],
                src=self._owner_global_ranks[layer_name],
                group=self._device_group,
                async_op=True,
            )
            work.wait()
            done = torch.npu.Event()
            done.record(transfer_stream)
        # A Future must cover device completion, including the owner's source
        # reads, before attention can overwrite the persistent cache.
        done.synchronize()
