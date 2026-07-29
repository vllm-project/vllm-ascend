# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side helpers for Ascend weight transfer."""

import pickle
from collections.abc import Callable, Iterator
from dataclasses import asdict
from typing import Any

import pybase64 as base64
import requests
import torch
from torch.multiprocessing.reductions import reduce_tensor


class TrainerProcessCoordinator:
    """Small wrapper around torch distributed synchronization used by trainers."""

    @staticmethod
    def is_rank_zero() -> bool:
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    @staticmethod
    def all_gather_and_merge_handles(handles: list[dict[str, tuple]]) -> list[dict[str, tuple]]:
        """All-gather and merge IPC handle dicts across ranks.

        Each rank contributes a list of ``{npu_uuid: ipc_args}`` dicts. Rank 0
        collects and merges per-index; other ranks receive a list of empty dicts.
        No-op when no distributed group exists.
        """
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            return handles

        world_size = torch.distributed.get_world_size()
        gathered: list[list[dict[str, tuple]] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, handles)
        torch.distributed.barrier()
        torch.npu.synchronize()

        if torch.distributed.get_rank() == 0:
            merged: list[dict[str, tuple]] = []
            for param_idx in range(len(handles)):
                m: dict[str, tuple] = {}
                for rank_handles in gathered:
                    if rank_handles is not None:
                        m.update(rank_handles[param_idx])
                merged.append(m)
            return merged
        return [{} for _ in handles]

    @staticmethod
    def post_send_sync() -> None:
        """Barrier + synchronize after a send; no-op if single-NPU."""
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()
        torch.npu.synchronize()


def default_parameter_tensor(item: tuple[str, torch.Tensor]) -> torch.Tensor:
    return item[1]


def collect_parameter_metadata(
    iterator: Iterator[tuple[str, torch.Tensor]],
    npu_uuid: str,
) -> tuple[list[str], list[str], list[list[int]], list[dict[str, tuple]], list[torch.Tensor]]:
    """Collect metadata and IPC handles for non-packed NPU IPC transfer."""
    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    ipc_handles: list[dict[str, tuple]] = []
    # Hold strong refs to every contiguous copy until the send + post-send sync
    # completes. ``reduce_tensor``'s returned args do not keep storage alive.
    weight_refs: list[torch.Tensor] = []

    for name, tensor in iterator:
        names.append(name)
        dtype_names.append(str(tensor.dtype).split(".")[-1])
        shapes.append(list(tensor.shape))

        weight = tensor.detach().contiguous()
        weight_refs.append(weight)
        # Store only the rebuild args (drop the func); the consumer rebuilds with
        # the well-known rebuild function, mirroring upstream's CUDA IPC engine.
        _, ipc_args = reduce_tensor(weight)
        ipc_handles.append({npu_uuid: ipc_args})

    return names, dtype_names, shapes, ipc_handles, weight_refs


def dispatch_update_info(
    *,
    args: Any,
    update_info: Any,
    update_fields: dict[str, Any],
    ipc_handles: list[dict[str, tuple]] | dict[str, tuple],
) -> None:
    """Send a weight update payload through callable, Ray, or HTTP transport."""
    if callable(args.send_mode):
        args.send_mode(update_info)
    elif args.send_mode == "ray":
        import ray

        handles = args.llm_handle if isinstance(args.llm_handle, list) else [args.llm_handle]
        ray.get([h.update_weights.remote(dict(update_info=asdict(update_info))) for h in handles])
    elif args.send_mode == "http":
        pickled_handles = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")
        http_fields = {k: v for k, v in update_fields.items() if k != "ipc_handles"}
        http_fields["ipc_handles_pickled"] = pickled_handles

        url = f"{args.url}/update_weights"
        payload = {"update_info": http_fields}
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
