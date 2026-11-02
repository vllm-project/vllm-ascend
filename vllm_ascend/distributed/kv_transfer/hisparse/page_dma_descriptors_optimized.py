# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Candidate: batch page-DMA descriptor arithmetic, preserving submission order.

This detached module does not install itself. Cache payloads, page ownership,
transfer IDs, DMA submission and all completion events retain the original path.
"""

import hashlib
import inspect
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import numpy as np

_COMMON_WORKER_SHA256 = "1953b3e582502d16df38bba4a5966e0b98492feb67b65cdac7f2b0397ca4b229"


@dataclass(frozen=True)
class _PageGeometry:
    signature: tuple
    owners: tuple
    source_indices: np.ndarray
    source_addresses: np.ndarray
    destination_addresses: np.ndarray
    source_block_bytes: np.ndarray
    row_bytes: np.ndarray
    source_block_counts: np.ndarray
    destination_row_counts: np.ndarray


def _geometry(worker):
    """Recheck current storage each call; read each metadata property once."""
    handles = worker.cache_handles
    sources = worker.resident_caches
    destinations = worker.host_caches
    if not handles or len(handles) != len(sources) or len(handles) != len(destinations):
        return None
    signature = []
    facts = []
    for cache, source, destination in zip(handles, sources, destinations, strict=True):
        source_shape = tuple(source.shape)
        destination_shape = tuple(destination.shape)
        if len(source_shape) != 3 or len(destination_shape) != 2:
            return None
        source_stride = tuple(source.stride())
        destination_stride = tuple(destination.stride())
        source_size = source.element_size()
        destination_size = destination.element_size()
        if (
            source_shape[-1] != destination_shape[-1]
            or source_size != destination_size
            or source_stride[2] != 1
            or source_stride[1] != source_shape[-1]
            or destination_stride[1] != 1
            or destination_stride[0] != destination_shape[-1]
            or source_shape[1] != worker.kernel_block_size
        ):
            return None
        source_index = cache.runtime.resident_source_index
        if source_index < 0:
            return None
        source_address = source.data_ptr()
        destination_address = destination.data_ptr()
        signature.append(
            (
                id(source),
                source_address,
                source_shape,
                source_stride,
                source.dtype,
                source.device,
                source_size,
                id(destination),
                destination_address,
                destination_shape,
                destination_stride,
                destination.dtype,
                destination.device,
                destination_size,
                source_index,
            )
        )
        facts.append(
            (
                source_index,
                source_address,
                destination_address,
                source_stride[0] * source_size,
                source_shape[-1] * source_size,
                source_shape[0],
                destination_shape[0],
            )
        )
    signature = (worker.kernel_block_size, tuple(signature))
    previous = getattr(worker, "_ascend_page_descriptor_geometry", None)
    if previous is not None:
        if previous.signature != signature:
            raise RuntimeError("HiSparse page DMA storage changed; drain and recreate the worker")
        return previous

    def vector(column):
        array = np.asarray([row[column] for row in facts], dtype=np.int64)
        array.setflags(write=False)
        return array

    result = _PageGeometry(
        signature=signature,
        owners=tuple(sources) + tuple(destinations),
        source_indices=vector(0),
        source_addresses=vector(1),
        destination_addresses=vector(2),
        source_block_bytes=vector(3),
        row_bytes=vector(4),
        source_block_counts=vector(5),
        destination_row_counts=vector(6),
    )
    worker._ascend_page_descriptor_geometry = result
    return result


def enqueue_transfers(worker, transfers) -> bool:
    """Return False for unsupported layouts before any descriptor submission."""
    if not transfers or not worker.is_host_writer:
        return True
    geometry = _geometry(worker)
    if geometry is None:
        return False
    source_blocks_by_transfer = np.asarray([transfer.resident_block_ids for transfer in transfers], dtype=np.int64)
    if source_blocks_by_transfer.ndim != 2:
        raise RuntimeError("HiSparse spill DMA source mappings must be rectangular.")
    if np.any(geometry.source_indices >= source_blocks_by_transfer.shape[1]):
        raise RuntimeError("HiSparse spill DMA source index is out of range.")
    source_blocks = source_blocks_by_transfer[:, geometry.source_indices]
    if np.any(source_blocks < 0) or np.any(source_blocks >= geometry.source_block_counts[None, :]):
        raise RuntimeError("HiSparse spill DMA source is out of range.")
    destination_rows = np.fromiter(
        (transfer.host_block_id * worker.kernel_block_size for transfer in transfers),
        dtype=np.int64,
        count=len(transfers),
    )
    if np.any(destination_rows < 0) or np.any(
        destination_rows[:, None] + worker.kernel_block_size > geometry.destination_row_counts[None, :]
    ):
        raise RuntimeError("HiSparse spill DMA destination is out of range.")

    num_layers = len(worker.cache_handles)
    descriptor_count = len(transfers) * num_layers
    descriptors = worker._acquire_dma_descriptors(descriptor_count)
    # Original order is transfer-major, then layer; keep every byte address,
    # extent, duplicate transfer, transfer ID and downstream lifetime unchanged.
    src = descriptors.src_np[:descriptor_count].reshape(len(transfers), num_layers)
    dst = descriptors.dst_np[:descriptor_count].reshape(len(transfers), num_layers)
    sizes = descriptors.sizes_np[:descriptor_count].reshape(len(transfers), num_layers)
    np.multiply(source_blocks, geometry.source_block_bytes[None, :], out=src)
    np.add(src, geometry.source_addresses[None, :], out=src)
    np.multiply(destination_rows[:, None], geometry.row_bytes[None, :], out=dst)
    np.add(dst, geometry.destination_addresses[None, :], out=dst)
    sizes[:] = worker.kernel_block_size * geometry.row_bytes[None, :]
    worker._submit_dma_descriptors(
        descriptors, descriptor_count, transfer_ids=tuple(transfer.transfer_id for transfer in transfers)
    )
    return True


def install() -> None:
    """Explicit Ascend-only activation; other backends retain the original."""
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker

    original = HiSparseConnectorWorker._enqueue_transfers
    if getattr(original, "_ascend_vector_page_descriptors", False):
        return
    source = Path(inspect.getsourcefile(original))
    if hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != _COMMON_WORKER_SHA256:
        raise RuntimeError("Unreviewed HiSparse worker page-transfer call site")

    @wraps(original)
    def wrapped(worker, transfers):
        handles = worker.cache_handles
        if handles and handles[0].runtime.device.type == "npu" and enqueue_transfers(worker, transfers):
            return
        return original(worker, transfers)

    wrapped._ascend_vector_page_descriptors = True
    wrapped._ascend_original = original
    HiSparseConnectorWorker._enqueue_transfers = wrapped
