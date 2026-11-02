# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU descriptor contracts; no DMA, graph or model qualification is implied."""

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# These modules only inspect CPU descriptors and tensor geometry. File loading
# keeps this test portable without initializing the Ascend platform plugin.
BUILDER = load_file(
    "hisparse_test_page_builder", ROOT / "vllm_ascend/distributed/kv_transfer/hisparse/page_dma_descriptors.py"
)


@dataclass
class Geometry:
    shape: tuple
    strides: tuple
    address: int
    dtype: str = "uint8"
    device: str = "npu:0"
    itemsize: int = 1

    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def data_ptr(self):
        return self.address

    def element_size(self):
        return self.itemsize


def worker(layers, groups, width=656, itemsize=1):
    page_stride = 419840 if itemsize == 1 else 589824
    return SimpleNamespace(
        kernel_block_size=128,
        is_host_writer=True,
        cache_handles=[
            SimpleNamespace(
                runtime=SimpleNamespace(resident_source_index=i % groups, device=SimpleNamespace(type="npu"))
            )
            for i in range(layers)
        ],
        resident_caches=[
            Geometry(
                (80 + i, 128, width),
                (page_stride // itemsize, width, 1),
                0x100000000 + i * 0x10000000,
                itemsize=itemsize,
            )
            for i in range(layers)
        ],
        host_caches=[
            Geometry((12800, width), (width, 1), 0x8000000000 + i * 0x10000000, device="cpu", itemsize=itemsize)
            for i in range(layers)
        ],
        _acquire_dma_descriptors=lambda count: SimpleNamespace(
            **{name: np.full(count + 7, -777, dtype=np.int64) for name in ("src_np", "dst_np", "sizes_np")}
        ),
        _submit_dma_descriptors=MagicMock(),
    )


def transfers(count, groups, shift=0):
    values = [
        SimpleNamespace(
            resident_block_ids=[1 + (i + g + shift) % 64 for g in range(groups)],
            host_block_id=1 + (i + shift) % 60,
            transfer_id=1000 + i,
        )
        for i in range(count)
    ]
    if count > 1:
        values[-1] = values[0]  # Duplicates and their original transfer order must survive.
    return values


@pytest.mark.parametrize("layers,groups", [(1, 1), (7, 3), (79, 21)])
@pytest.mark.parametrize("count", [0, 1, 8, 32])
@pytest.mark.parametrize("width,itemsize", [(656, 1), (576, 2)])
def test_current_descriptor_bytes_match_independent_scalar_reference(layers, groups, count, width, itemsize):
    owner = worker(layers, groups, width, itemsize)
    for iteration, shift in enumerate((0, 3, 7), start=1):
        current = transfers(count, groups, shift)
        assert BUILDER.enqueue_transfers(owner, current)
        if not count:
            owner._submit_dma_descriptors.assert_not_called()
            continue
        args, kwargs = owner._submit_dma_descriptors.call_args
        descriptors, actual_count = args
        assert actual_count == count * layers
        assert kwargs == {"transfer_ids": tuple(t.transfer_id for t in current)}
        expected = []
        for transfer in current:
            for handle, source, destination in zip(
                owner.cache_handles, owner.resident_caches, owner.host_caches, strict=True
            ):
                block = transfer.resident_block_ids[handle.runtime.resident_source_index]
                expected.append(
                    (
                        source.data_ptr() + block * source.stride(0) * itemsize,
                        destination.data_ptr() + transfer.host_block_id * 128 * width * itemsize,
                        128 * width * itemsize,
                    )
                )
        vectors = (descriptors.src_np, descriptors.dst_np, descriptors.sizes_np)
        np.testing.assert_array_equal(np.stack([v[:actual_count] for v in vectors], axis=1), expected)
        assert all(np.all(v[actual_count:] == -777) for v in vectors)
        assert owner._submit_dma_descriptors.call_count == iteration


def test_real_tensor_offsets_strides_and_changed_storage():
    owner = worker(1, 1, width=7)
    backing = torch.zeros(400, dtype=torch.uint8)
    source = backing.as_strided((9, 4, 7), (40, 7, 1), storage_offset=11)
    host = torch.zeros((40, 7), dtype=torch.uint8)
    owner.kernel_block_size = 4
    owner.resident_caches, owner.host_caches = [source], [host]
    current = [SimpleNamespace(resident_block_ids=[3], host_block_id=6, transfer_id=42)]
    assert BUILDER.enqueue_transfers(owner, current)
    descriptor = owner._submit_dma_descriptors.call_args.args[0]
    assert descriptor.src_np[0] == source.data_ptr() + 3 * 40
    assert descriptor.dst_np[0] == host.data_ptr() + 6 * 4 * 7
    assert descriptor.sizes_np[0] == 28
    owner.resident_caches = [torch.zeros_like(source)]
    with pytest.raises(RuntimeError, match="storage changed"):
        BUILDER.enqueue_transfers(owner, current)
    assert owner._submit_dma_descriptors.call_count == 1


@pytest.mark.parametrize("bad", ["negative_source", "source_bound", "negative_host", "host_bound", "source_index"])
def test_bad_addresses_never_submit(bad):
    owner = worker(1, 1)
    current = transfers(1, 1)
    if bad == "negative_source":
        current[0].resident_block_ids[0] = -1
    elif bad == "source_bound":
        current[0].resident_block_ids[0] = 80
    elif bad == "negative_host":
        current[0].host_block_id = -1
    elif bad == "host_bound":
        current[0].host_block_id = 100
    else:
        owner.cache_handles[0].runtime.resident_source_index = 1
    with pytest.raises(RuntimeError, match="out of range"):
        BUILDER.enqueue_transfers(owner, current)
    owner._submit_dma_descriptors.assert_not_called()


def test_unsupported_layout_fallback_and_nonwriter_noop():
    owner = worker(1, 1)
    owner.resident_caches[0].strides = (419840, 657, 1)
    assert not BUILDER.enqueue_transfers(owner, transfers(1, 1))
    owner.is_host_writer = False
    assert BUILDER.enqueue_transfers(owner, transfers(1, 1))
    owner._submit_dma_descriptors.assert_not_called()


class OriginalWorker:
    def _enqueue_transfers(self, values):
        self.original_calls.append(values)
        return "original"


def test_cold_install_scopes_npu_falls_back_and_never_overwrites_foreign_owner(monkeypatch):
    target = "vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker"
    builder_name = "vllm_ascend.distributed.kv_transfer.hisparse.page_dma_descriptors"
    original = OriginalWorker._enqueue_transfers
    cls = type("Worker", (), {"_enqueue_transfers": original})
    monkeypatch.setitem(sys.modules, target, SimpleNamespace(HiSparseConnectorWorker=cls))
    monkeypatch.setitem(sys.modules, builder_name, BUILDER)
    patch = load_file("hisparse_test_page_patch", ROOT / "vllm_ascend/patch/worker/patch_hisparse_page_descriptors.py")
    assert not patch.install()
    assert cls._enqueue_transfers is original
    monkeypatch.setattr(patch, "_WORKER_SOURCE_SHA256", {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    assert patch.install()
    installed = cls._enqueue_transfers
    assert installed._ascend_original is original
    assert patch.install() and cls._enqueue_transfers is installed
    npu = worker(2, 1)
    npu.original_calls = []
    installed(npu, transfers(2, 1))
    assert npu._submit_dma_descriptors.call_count == 1 and not npu.original_calls
    npu.resident_caches[0].strides = (419840, 657, 1)
    assert installed(npu, transfers(1, 1)) == "original"
    cpu = worker(1, 1)
    cpu.original_calls = []
    cpu.cache_handles[0].runtime.device.type = "cpu"
    assert installed(cpu, transfers(1, 1)) == "original"
    cpu._submit_dma_descriptors.assert_not_called()

    def foreign(self, values):
        return "foreign"

    foreign.__wrapped__ = original
    cls._enqueue_transfers = foreign
    assert not patch.install() and cls._enqueue_transfers is foreign
    cls._enqueue_transfers = len
    assert not patch.install() and cls._enqueue_transfers is len
