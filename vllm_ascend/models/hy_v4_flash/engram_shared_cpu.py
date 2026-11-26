# SPDX-License-Identifier: Apache-2.0
"""Shared CPU storage for HY4 CPU index_select; no device UVA registration."""

import math
from multiprocessing import shared_memory
from unittest.mock import patch

import torch
import torch.distributed as dist


class PageableEngramBuffer:
    def __init__(self, shape, dtype, group):
        self.shm = None
        self.tensor = None
        self.group = group
        count = math.prod(shape)
        size = count * torch.empty((), dtype=dtype).element_size()
        leader = dist.get_global_rank(group.cpu_group, 0)
        payload = [None]
        if group.rank_in_group == 0:
            try:
                self.shm = shared_memory.SharedMemory(create=True, size=size)
                payload[0] = self.shm.name
            except Exception as exc:
                payload[0] = "ERROR: " + repr(exc)
        dist.broadcast_object_list(payload, src=leader, group=group.cpu_group)
        if payload[0] is None or payload[0].startswith("ERROR:"):
            raise RuntimeError(f"CPU shared Engram creation failed: {payload[0]}")
        error = None
        try:
            if self.shm is None:
                with patch("multiprocessing.resource_tracker.register", lambda *a, **k: None):
                    self.shm = shared_memory.SharedMemory(name=payload[0])
            self.tensor = torch.frombuffer(self.shm.buf, dtype=dtype, count=count).reshape(shape)
        except Exception as exc:
            error = repr(exc)
        errors = [None] * group.world_size
        dist.all_gather_object(errors, error, group=group.cpu_group)
        dist.barrier(group=group.cpu_group)
        if group.rank_in_group == 0:
            self.shm.unlink()
        if any(errors):
            raise RuntimeError(f"CPU shared Engram mapping failed: {errors}")

    def close(self):
        self.tensor = None
        if self.shm is not None:
            try:
                self.shm.close()
            except BufferError:
                return
            self.shm = None
