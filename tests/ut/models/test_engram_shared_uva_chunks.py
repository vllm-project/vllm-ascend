# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared CPU backing and UVA registration ownership, without NPU imports.

Small real shared-memory buffers exercise the staged allocator. CANN device
addresses and collectives are mocked; these tests do not establish hardware UVA
support or prove that chunking resolves driver allocation failures.
"""

import ast
import ctypes
import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/npu.py"
CHUNK = 1024
WIDTH = 32
ROWS = CHUNK * 2 + 3


def _allocator(*, members=1, limit=CHUNK * WIDTH * 2, register_failure=None, pointer_failure=None):
    local = threading.local()
    local.rank = 0
    fence = threading.Barrier(members)
    slots = {}
    errors_by_rank = [None] * members
    state = SimpleNamespace(
        registrations=[],
        pointer_queries=[],
        unregistrations=[],
        created=[],
        mappings={},
        attempts={},
        unregister_failures=set(),
        local=local,
    )

    def broadcast(payload, src, group):
        if local.rank == 0:
            slots["name"] = payload[0]
            state.created.append(payload[0])
        fence.wait(timeout=20)
        payload[0] = slots["name"]
        fence.wait(timeout=20)

    def gather(errors, error, group):
        errors_by_rank[local.rank] = error
        fence.wait(timeout=20)
        errors[:] = errors_by_rank
        fence.wait(timeout=20)

    class Library:
        def __init__(self):
            self.rank = local.rank

        def aclrtHostRegisterV2(self, pointer, size, flags):
            index = state.attempts.get(self.rank, 0)
            state.attempts[self.rank] = index + 1
            failure = (self.rank, index) == register_failure
            state.registrations.append((self.rank, index, pointer.value, size, flags, failure))
            if failure:
                return 207001
            # Deliberately noncontiguous: deriving later chunk addresses from
            # the first registration must produce incorrect lookup results.
            device = (1 << 40) + self.rank * (1 << 36) + index * (1 << 30)
            state.mappings[self.rank, pointer.value] = (device, size, index)
            return 0

        def aclrtHostGetDevicePointer(self, pointer, out, flags):
            device, size, index = state.mappings[self.rank, pointer.value]
            state.pointer_queries.append((self.rank, index, pointer.value))
            if (self.rank, index) == pointer_failure:
                return 207001
            out._obj.value = device
            return 0

        def aclrtHostUnregister(self, pointer):
            key = self.rank, pointer.value
            state.unregistrations.append(key)
            if key in state.unregister_failures:
                state.unregister_failures.remove(key)
                return 207001
            assert key in state.mappings, "double unregister or unowned range"
            del state.mappings[key]
            return 0

    namespace = {
        "torch": torch,
        "ctypes": ctypes,
        "mmap": mmap,
        "os": os,
        "shared_memory": shared_memory,
        "patch": patch,
        "_host_library": Library,
        "HostUvaBuffer": object,
        "CHUNK_ROWS": CHUNK,
        "MAX_SINGLE_REGISTRATION_BYTES": limit,
        "ACL_HOST_REG_MAPPED": 2,
        "ACL_HOST_REG_PINNED": 0x10000000,
        "logger": Mock(),
        "dist": SimpleNamespace(
            get_global_rank=lambda group, rank: rank,
            broadcast_object_list=broadcast,
            all_gather_object=gather,
            barrier=lambda group: fence.wait(timeout=20),
        ),
    }
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name
        in {"EngramBufferInitializationError", "_MemfdSharedMemory", "SharedUvaBuffer", "gather_dequantize_host_uva"}
    ]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    state.cls = namespace["SharedUvaBuffer"]
    state.error_cls = namespace["EngramBufferInitializationError"]
    state.lookup = namespace["gather_dequantize_host_uva"]
    state.namespace = namespace
    state.group = lambda rank=0: SimpleNamespace(world_size=members, rank_in_group=rank, cpu_group="storage")
    return state


def test_production_registration_limit_is_32_gib():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MAX_SINGLE_REGISTRATION_BYTES" for target in node.targets
        )
    )
    assert eval(compile(ast.Expression(node.value), str(SOURCE), "eval")) == 32 * 1024**3


@pytest.mark.parametrize("rows,registration_count", [(CHUNK * 2 - 1, 1), (CHUNK * 2, 1), (CHUNK * 2 + 1, 3)])
def test_threshold_boundary_preserves_small_tp_single_registration(rows, registration_count):
    state = _allocator()
    buffer = state.cls((rows, WIDTH), torch.int8, "cpu", state.group())
    try:
        assert len(state.registrations) == registration_count
        assert len(buffer.ptrs) == (rows + CHUNK - 1) // CHUNK
        if registration_count == 1:
            base = int(buffer.ptrs[0])
            assert buffer.ptrs.tolist() == [base + start * WIDTH for start in range(0, rows, CHUNK)]
    finally:
        buffer.close()


def test_noncontiguous_device_addresses_tail_and_page_aligned_host_boundaries():
    state = _allocator()
    buffer = state.cls((ROWS, WIDTH), torch.int8, "cpu", state.group())
    try:
        registrations = state.registrations
        assert [row[3] for row in registrations] == [CHUNK * WIDTH, CHUNK * WIDTH, 3 * WIDTH]
        base = registrations[0][2]
        assert [row[2] - base for row in registrations] == [0, CHUNK * WIDTH, CHUNK * WIDTH * 2]
        assert all(row[2] % 4096 == 0 for row in registrations)
        assert all(row[4] == 2 | 0x10000000 for row in registrations)
        expected = [state.mappings[0, row[2]][0] for row in registrations]
        assert buffer.ptrs.tolist() == expected
        assert expected[1] != expected[0] + CHUNK * WIDTH
        assert buffer.tensor.shape == (ROWS, WIDTH)
    finally:
        buffer.close()
    assert [pointer for rank, pointer in state.unregistrations] == [row[2] for row in registrations][::-1]
    assert state.mappings == {}
    buffer.close()
    assert len(state.unregistrations) == 3


@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_midregistration_failure_unwinds_only_successful_ranges(failure_index):
    state = _allocator(register_failure=(0, failure_index))
    with pytest.raises(RuntimeError, match=rf"aclrtHostRegisterV2 failed: rc=207001 .*chunk={failure_index}"):
        state.cls((ROWS, WIDTH), torch.int8, "cpu", state.group())
    assert len(state.registrations) == failure_index + 1
    successful = [row[2] for row in state.registrations if not row[-1]]
    assert [pointer for rank, pointer in state.unregistrations] == successful[::-1]
    assert len(state.pointer_queries) == failure_index
    assert state.mappings == {}
    name = state.created[0]
    with pytest.raises(FileNotFoundError):
        if isinstance(name, tuple):
            os.open(f"/proc/{name[1]}/fd/{name[2]}", os.O_RDWR)
        else:
            shared_memory.SharedMemory(name=name)


def test_get_pointer_failure_unwinds_that_successfully_registered_chunk():
    state = _allocator(pointer_failure=(0, 1))
    with pytest.raises(RuntimeError, match="aclrtHostGetDevicePointer failed: rc=207001 chunk=1"):
        state.cls((ROWS, WIDTH), torch.int8, "cpu", state.group())
    assert len(state.registrations) == len(state.pointer_queries) == 2
    assert [pointer for rank, pointer in state.unregistrations] == [row[2] for row in state.registrations][::-1]
    assert state.mappings == {}


@pytest.mark.parametrize("failure", ["register", "pointer"])
def test_chunk_failure_on_one_peer_reaches_and_cleans_every_storage_member(failure):
    controls = {"register_failure" if failure == "register" else "pointer_failure": (1, 1)}
    state = _allocator(members=2, **controls)

    def attach(rank):
        state.local.rank = rank
        try:
            return state.cls((ROWS, WIDTH), torch.int8, "cpu", state.group(rank))
        except RuntimeError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attach, range(2)))
    assert all(isinstance(value, RuntimeError) for value in results)
    assert all("rank 1: RuntimeError:" in str(value) and "chunk=1" in str(value) for value in results)
    assert len(state.created) == 1
    assert state.mappings == {}
    expected = 4 if failure == "register" else 5
    assert len(state.unregistrations) == expected


def test_unregister_failure_keeps_backing_and_retries_without_double_unregister():
    state = _allocator()
    buffer = state.cls((ROWS, WIDTH), torch.int8, "cpu", state.group())
    addresses = [row[2] for row in state.registrations]
    state.unregister_failures.add((0, addresses[1]))
    tensor, backing = buffer.tensor, buffer.shm
    real_close = backing.close
    backing.close = Mock(wraps=real_close)
    try:
        with pytest.raises(RuntimeError, match="aclrtHostUnregister failed for shared Engram"):
            buffer.close()
        backing.close.assert_not_called()
        assert buffer.shm is backing and buffer.tensor is tensor
        assert [pointer.value for pointer in buffer._registered_host_ptrs] == addresses[:2]
        assert buffer.pointer.value == addresses[0]
        assert state.unregistrations == [(0, addresses[2]), (0, addresses[1])]
        buffer.close()
        assert state.unregistrations == [(0, addresses[2]), (0, addresses[1]), (0, addresses[1]), (0, addresses[0])]
        assert buffer._registered_host_ptrs == []
        assert buffer.shm is buffer.tensor is buffer.ptrs is None
        assert not buffer.pointer.value
        backing.close.assert_called_once()
        buffer.close()
        backing.close.assert_called_once()
    finally:
        state.unregister_failures.clear()
        buffer.close()


def test_constructor_cleanup_failure_retains_each_outstanding_registration_for_retry():
    state = _allocator(pointer_failure=(0, 1))
    register = state.cls.__init__.__globals__["_host_library"]

    def library():
        value = register()
        original = value.aclrtHostGetDevicePointer

        def get_pointer(pointer, out, flags):
            result = original(pointer, out, flags)
            if result:
                state.unregister_failures.add((0, pointer.value))
            return result

        value.aclrtHostGetDevicePointer = get_pointer
        return value

    state.cls.__init__.__globals__["_host_library"] = library
    with pytest.raises(state.error_cls, match="release failed") as info:
        state.cls((ROWS, WIDTH), torch.int8, "cpu", state.group())
    buffer = info.value.buffer
    assert buffer.shm is not None
    assert len(buffer._registered_host_ptrs) == 2
    assert len(state.unregistrations) == 1
    buffer.close()
    assert buffer.shm is None and buffer._registered_host_ptrs == []
    assert len(state.unregistrations) == 3
    assert state.mappings == {}


@pytest.mark.parametrize("closing_member", ["codes", "scales"])
def test_actual_uva_lookup_rejects_partial_close_before_native_imports(closing_member):
    state = _allocator()
    buffer = state.cls((ROWS, WIDTH), torch.int8, "cpu", state.group())
    addresses = [row[2] for row in state.registrations]
    state.unregister_failures.add((0, addresses[1]))
    try:
        with pytest.raises(RuntimeError, match="aclrtHostUnregister failed"):
            buffer.close()
        assert buffer._closing
        assert (0, addresses[2]) not in state.mappings  # Tail chunk is already unmapped.
        assert buffer.shm is not None and buffer.tensor is not None
        other = SimpleNamespace()
        codes, scales = (buffer, other) if closing_member == "codes" else (other, buffer)
        ids = torch.zeros((1, 1), dtype=torch.int64)
        with (
            patch("builtins.__import__", side_effect=AssertionError("closed lookup reached native imports")),
            pytest.raises(RuntimeError, match="closing or closed shared buffer"),
        ):
            state.lookup(codes, scales, ids)
        buffer.close()
        assert buffer.shm is None
        with pytest.raises(RuntimeError, match="closing or closed shared buffer"):
            state.lookup(codes, scales, ids)
    finally:
        state.unregister_failures.clear()
        buffer.close()


def _read_device_row(state, rank, buffer, row, dtype):
    chunk, local = divmod(row, CHUNK)
    address = int(buffer.ptrs[chunk]) + local * buffer.row_bytes
    for (owner, host), (device, size, index) in state.mappings.items():
        if owner == rank and device <= address < device + size:
            raw = (ctypes.c_ubyte * buffer.row_bytes).from_address(host + address - device)
            return torch.frombuffer(raw, dtype=dtype).clone()
    raise AssertionError(f"lookup selected an unregistered device address: {address:#x}")


def test_two_members_lookup_same_real_cpu_backing_across_noncontiguous_chunks():
    state = _allocator(members=2)

    def attach(rank):
        state.local.rank = rank
        codes = state.cls((ROWS, WIDTH), torch.int8, "cpu", state.group(rank))
        scales = state.cls((ROWS, 1), torch.float32, "cpu", state.group(rank))
        return codes, scales

    buffers = []
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            buffers = list(pool.map(attach, range(2)))
        assert len(state.created) == 2  # One codes segment, one scales segment.
        codes, scales = buffers[0]
        codes.tensor.copy_((torch.arange(ROWS) % 127).to(torch.int8)[:, None].expand(ROWS, WIDTH))
        scales.tensor.copy_((1 + torch.arange(ROWS) % 3).float()[:, None])
        torch.testing.assert_close(buffers[1][0].tensor, codes.tensor)
        torch.testing.assert_close(buffers[1][1].tensor, scales.tensor)
        for rank, (peer_codes, peer_scales) in enumerate(buffers):
            # Scales remain one small registration; codes use three real
            # independently translated ranges. Query rows straddle every edge.
            assert len(peer_codes._registered_host_ptrs) == 3
            assert len(peer_scales._registered_host_ptrs) == 1
            for row in [0, CHUNK - 1, CHUNK, CHUNK * 2 - 1, CHUNK * 2, ROWS - 1]:
                actual = _read_device_row(state, rank, peer_codes, row, torch.int8).float()
                actual *= _read_device_row(state, rank, peer_scales, row, torch.float32)[0]
                expected = codes.tensor[row].float() * scales.tensor[row, 0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        for pair in buffers:
            for buffer in pair:
                buffer.close()
    assert state.mappings == {}
