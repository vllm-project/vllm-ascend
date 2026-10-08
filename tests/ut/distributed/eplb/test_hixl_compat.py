# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import ctypes
import struct
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tests.ut.distributed.eplb.test_hixl_communicator import _Engine, _Tensor
from vllm_ascend.distributed.eplb import eplb_communicator, hixl_compat


def test_compat_layout_matches_probed_cann_abi():
    assert ctypes.sizeof(hixl_compat.MemDesc) == 144
    assert ctypes.sizeof(hixl_compat.TransferOpDesc) == 24
    assert ctypes.sizeof(hixl_compat.TransferArgs) == 128

    assert hixl_compat.SUCCESS == 0
    assert hixl_compat.MemType.MEM_DEVICE == 0
    assert hixl_compat.TransferOp.READ == 0
    assert [status.name for status in hixl_compat.TransferStatus] == [
        "WAITING",
        "COMPLETED",
        "TIMEOUT",
        "FAILED",
    ]

    desc = hixl_compat.TransferOpDesc(local_addr=1, remote_addr=2, len=3)
    assert (desc.local_addr, desc.remote_addr, desc.len) == (1, 2, 3)
    mem_desc = hixl_compat.MemDesc(4096, 8192)
    assert (mem_desc.addr, mem_desc.len, mem_desc.remote_accessible) == (4096, 8192, True)


def test_compat_empty_options_map_self_points_at_header():
    options_map = hixl_compat.Hixl._empty_options_map()
    header = ctypes.addressof(options_map) + hixl_compat._EMPTY_MAP_HEADER_OFFSET
    left, right = struct.unpack_from("=QQ", options_map, hixl_compat._EMPTY_MAP_SELF_POINTER_OFFSET)
    assert (left, right) == (header, header)
    node_count_offset = hixl_compat._EMPTY_MAP_SELF_POINTER_OFFSET + 16
    assert struct.unpack_from("=Q", options_map, node_count_offset)[0] == 0


def test_resolve_prefers_the_official_hixl_package(monkeypatch):
    official = SimpleNamespace(SUCCESS=0)
    monkeypatch.setitem(sys.modules, "hixl", official)

    assert eplb_communicator._resolve_hixl_module() is official


def test_resolve_falls_back_to_hixl_compat(monkeypatch):
    monkeypatch.setitem(sys.modules, "hixl", None)
    monkeypatch.setattr(hixl_compat, "ensure_available", MagicMock())

    assert eplb_communicator._resolve_hixl_module() is hixl_compat


def test_resolve_raises_when_both_hixl_paths_are_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "hixl", None)

    def _unavailable():
        raise RuntimeError("No CANN HIXL libraries found")

    monkeypatch.setattr(hixl_compat, "ensure_available", _unavailable)

    with pytest.raises(RuntimeError, match="HIXL EPLB requires"):
        eplb_communicator._resolve_hixl_module()


def test_hixl_communicator_reads_through_hixl_compat(monkeypatch):
    engine = _Engine()
    monkeypatch.setitem(sys.modules, "hixl", None)
    monkeypatch.setattr(hixl_compat, "ensure_available", MagicMock())
    monkeypatch.setattr(hixl_compat, "Hixl", lambda: engine)
    monkeypatch.setattr(eplb_communicator, "is_weak_contiguous", lambda _tensor: True)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            set_device=MagicMock(),
            memory_snapshot=MagicMock(return_value=[{"device": 0, "address": 0, "total_size": 8_388_608}]),
        ),
        raising=False,
    )
    monkeypatch.setattr(eplb_communicator, "get_ip", lambda: "192.0.2.1")
    monkeypatch.setattr(eplb_communicator, "get_open_port", lambda: 12345)

    group = MagicMock()
    group.rank.return_value = 0
    group.size.return_value = 2

    def all_gather(gathered, local_state, *, group):
        gathered[0] = local_state
        gathered[1] = (
            "192.0.2.2:12346",
            {
                key: (tuple(address + 10_000 for address in addresses), stride)
                for key, (addresses, stride) in local_state[1].items()
            },
        )

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather)

    def all_reduce(completed, *, group, async_op):
        completed.fill_(2)
        return MagicMock()

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    device = SimpleNamespace(type="npu", index=0)
    # Each layer holds one stacked view and one per-expert tensor list; the
    # heterogeneous shape is intentional, so keep the annotation loose.
    weights: list[list[Any]] = [
        [_Tensor(1000, device), [_Tensor(2100, device, (4,), 16), _Tensor(2200, device, (4,), 16)]],
        [_Tensor(3000, device), [_Tensor(4100, device, (4,), 16), _Tensor(4200, device, (4,), 16)]],
    ]
    buffers = [
        _Tensor(5000, device),
        [_Tensor(6100, device, (4,), 16), _Tensor(6200, device, (4,), 16)],
    ]
    communicator = eplb_communicator.AscendHixlEplbCommunicator(group, weights, buffers)

    assert communicator._hixl is hixl_compat
    assert engine.options == {}
    assert len(engine.registered) == 1
    assert isinstance(engine.registered[0], hixl_compat.MemDesc)

    communicator.set_transfer_context(np.array([0, 1, 2, 3]), layer_idx=1)
    communicator.add_recv(
        [_Tensor(5000, device, shape=(4,), nbytes=16), weights[0][1][0]],
        src_rank=1,
        expert_id=3,
    )
    communicator.execute()

    remote_engine, operation, descriptors = engine.transfers[0]
    assert remote_engine == "192.0.2.2:12346"
    assert operation == hixl_compat.TransferOp.READ
    assert all(isinstance(desc, hixl_compat.TransferOpDesc) for desc in descriptors)

    communicator._close()
    assert engine.finalized
