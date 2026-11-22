# SPDX-License-Identifier: Apache-2.0
"""Unit tests for owner-sharded KV transport staging."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.utils import owner_transfer


def _stager(capacity: int):
    buffer = torch.zeros(capacity, dtype=torch.int8)
    return owner_transfer.OwnerTransferStager(torch.device("cpu"), buffer=buffer)


def test_put_stages_complete_keys_and_splits_to_capacity(monkeypatch):
    copies = []

    def copy(src, dst, sizes):
        copies.append((list(src), list(dst), list(sizes)))

    monkeypatch.setattr(owner_transfer, "_device_copy", copy)
    stager = _stager(16)
    backend = SimpleNamespace(puts=[])

    def put(keys, addrs, sizes):
        backend.puts.append((list(keys), [list(a) for a in addrs], [list(s) for s in sizes]))
        return [True] * len(keys)

    backend.put_fn = put

    keys = ["k0", "k1", "k2"]
    addrs = [[100, 200], [300], [400]]
    sizes = [[4, 4], [8], [8]]  # key0=8, key1=8, key2=8

    class Backend:
        def put(self, keys, addrs, sizes):
            return put(keys, addrs, sizes)

    result = stager.put(Backend(), keys, addrs, sizes)

    assert result == [True, True, True]
    # capacity 16 fits two 8-byte keys per batch.
    assert [put_keys for put_keys, _, _ in backend.puts] == [["k0", "k1"], ["k2"]]
    # The staged addresses are inside the staging buffer and preserve entry order.
    assert backend.puts[0][1] == [
        [stager.ptr, stager.ptr + 4],
        [stager.ptr + 8],
    ]
    assert backend.puts[1][1] == [[stager.ptr]]
    assert [sizes for _, _, sizes in copies] == [[4, 4, 8], [8]]
    # Every source range is copied once.
    assert copies[0][0] == [100, 200, 300]
    assert copies[1][0] == [400]


def test_put_rejects_a_single_key_larger_than_staging():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(owner_transfer, "_device_copy", lambda *args: None)
    stager = _stager(8)
    backend = SimpleNamespace(put=lambda *args: [True])
    with pytest.raises(ValueError, match="staging buffer"):
        stager.put(backend, ["too-big"], [[0]], [[9]])
    monkeypatch.undo()


def test_get_stages_into_owner_pages(monkeypatch):
    copies = []

    def copy(src, dst, sizes):
        copies.append((list(src), list(dst), list(sizes)))

    monkeypatch.setattr(owner_transfer, "_device_copy", copy)
    stager = _stager(32)

    class Backend:
        def get(self, keys, addrs, sizes):
            return [0] * len(keys)

    result = stager.get(Backend(), ["k0"], [[500, 600]], [[4, 4]])
    assert result == [0]
    assert copies[0][1] == [500, 600]
    assert copies[0][2] == [4, 4]
    assert copies[0][0] == [stager.ptr, stager.ptr + 4]


def test_owner_backed_tensor_detection_is_lazy(monkeypatch):
    # The helper only imports the VMM module when it is called.
    from vllm_ascend.attention.context_parallel import pcp_owner_vmm

    owned = torch.zeros(4, dtype=torch.int8)
    monkeypatch.setattr(pcp_owner_vmm, "owner_slab_for_tensor", lambda tensor: object() if tensor is owned else None)
    assert owner_transfer.owner_backed_tensor({"a": torch.zeros(4, dtype=torch.int8)}) is None
    assert owner_transfer.owner_backed_tensor({"a": (owned,)}) is owned
