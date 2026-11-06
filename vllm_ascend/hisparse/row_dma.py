# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Private experiment for the frozen Core HiSparse row-mirror worker."""

from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
from pathlib import Path

import numpy as np

_WORKER_SHA256 = "864e2c7ce4e2cff320b4f68f8dccc9691adb829f0e50fdf6a27dc9c7885040e5"
# Large prefill batches keep the scalar path to bound temporary memory.
_MAX_DESCRIPTORS = 32768


@dataclass(frozen=True)
class _Geometry:
    sources: tuple
    destinations: tuple
    block_size: int
    source_columns: np.ndarray
    source_addresses: np.ndarray
    source_page_bytes: np.ndarray
    source_pages: np.ndarray
    destination_addresses: np.ndarray
    destination_rows: np.ndarray
    row_bytes: np.ndarray

    @classmethod
    def bind(cls, worker):
        sources, destinations = worker.resident_caches, worker.host_caches
        block_size = worker.kernel_block_size
        facts = []
        for handle, source, destination in zip(worker.cache_handles, sources, destinations, strict=True):
            column = handle.runtime.resident_source_index
            if column < 0:
                return None
            item_size = source.element_size()
            facts.append(
                (
                    column,
                    source.data_ptr(),
                    source.stride(0) * item_size,
                    source.shape[0],
                    destination.data_ptr(),
                    destination.shape[0],
                    source.shape[-1] * item_size,
                )
            )
        if not facts:
            return None
        columns = np.ascontiguousarray(np.asarray(facts, dtype=np.int64).T)
        columns.flags.writeable = False
        # Core _init_dma already validated the rows and bound the cache owners.
        return cls(sources, destinations, block_size, *columns)


def _try_enqueue(worker, geometry, layer_indices, ready_event) -> bool:
    if not layer_indices or not worker._row_mirror_num_rows or not worker.is_host_writer:
        return True
    num_layers = len(layer_indices)
    num_mirrors = len(worker._row_mirrors)
    count = num_mirrors * num_layers
    if (
        geometry is None
        or num_layers < 2
        or count > _MAX_DESCRIPTORS
        or geometry.sources is not worker.resident_caches
        or geometry.destinations is not worker.host_caches
        or geometry.block_size != worker.kernel_block_size
    ):
        return False

    layers = np.asarray(layer_indices, dtype=np.intp)
    starts = worker._row_mirror_source_starts
    columns = geometry.source_columns[layers]
    if np.any(columns >= starts.shape[1]):
        return False
    pages, offsets = np.divmod(starts[:, columns], geometry.block_size)
    row_counts = worker._row_mirror_counts[:, None]
    destinations = worker._row_mirror_destination_starts[:, None]
    # Delegate invalid inputs to the original path for its checks and errors.
    if (
        np.any(pages < 0)
        or np.any(pages >= geometry.source_pages[layers])
        or np.any(offsets + row_counts > geometry.block_size)
        or np.any(destinations < 0)
        or np.any(destinations + row_counts > geometry.destination_rows[layers])
    ):
        return False

    descriptors = worker._acquire_dma_descriptors(count)
    shape = (num_mirrors, num_layers)
    src = descriptors.src_np[:count].reshape(shape)
    dst = descriptors.dst_np[:count].reshape(shape)
    row_bytes = geometry.row_bytes[layers]
    # The native submission consumes mirror-major descriptors in this order.
    np.multiply(pages, geometry.source_page_bytes[layers], out=src)
    src += geometry.source_addresses[layers]
    src += offsets * row_bytes
    np.multiply(destinations, row_bytes, out=dst)
    dst += geometry.destination_addresses[layers]
    descriptors.sizes_np[:count].reshape(shape)[:] = row_counts * row_bytes
    snapshot = getattr(worker, "_ascend_target_snapshot", None)
    if snapshot is None or not snapshot.try_submit(worker, descriptors, count, layer_indices, ready_event):
        worker._submit_dma_descriptors(descriptors, count, ready_event=ready_event)
    return True


def _install_worker(worker_type, backend_type) -> None:
    original_init = worker_type._init_dma
    original_enqueue = worker_type._enqueue_row_dma
    if getattr(original_enqueue, "_ascend_row_dma", False):
        return

    @wraps(original_init)
    def init(worker):
        original_init(worker)
        worker._ascend_row_dma_geometry = None
        if isinstance(worker.backend, backend_type) and worker.is_host_writer and worker.shared_host_region is None:
            worker._ascend_row_dma_geometry = _Geometry.bind(worker)

    @wraps(original_enqueue)
    def enqueue(worker, layer_indices, ready_event=None):
        geometry = getattr(worker, "_ascend_row_dma_geometry", None)
        if geometry is None or not _try_enqueue(worker, geometry, layer_indices, ready_event):
            return original_enqueue(worker, layer_indices, ready_event=ready_event)

    enqueue._ascend_row_dma = True
    worker_type._init_dma = init
    worker_type._enqueue_row_dma = enqueue


def install(backend_type) -> None:
    """Only adapt the exact Core implementation reviewed for this experiment."""
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import worker

    if sha256(Path(worker.__file__).read_bytes()).hexdigest() == _WORKER_SHA256:
        _install_worker(worker.HiSparseConnectorWorker, backend_type)
