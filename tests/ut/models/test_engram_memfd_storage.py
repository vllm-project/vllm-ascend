# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real memfd CPU backing and collective ownership; CANN alone is mocked."""

import ast
import ctypes
import mmap
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from test_engram_shared_uva_chunks import ROWS, WIDTH, _allocator

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/npu.py"


@pytest.fixture
def backing_class():
    namespace = {"mmap": mmap, "os": os, "ctypes": ctypes, "logger": Mock()}
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"EngramBufferInitializationError", "_MemfdSharedMemory"}
    ]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_MemfdSharedMemory"]


@pytest.mark.parametrize("size", [4096, 4099, 2 * 1024**2 + 37])
def test_real_memfd_tail_sharing_parameter_alias_and_creator_close(backing_class, size):
    creator = peer = None
    try:
        creator = backing_class(size)
        peer = backing_class(size, creator.name)
        creator_info, peer_info = os.fstat(creator._fd), os.fstat(peer._fd)
        assert (creator_info.st_dev, creator_info.st_ino, creator_info.st_size) == (
            peer_info.st_dev,
            peer_info.st_ino,
            size,
        )
        assert bytes(creator.buf) == bytes(size)
        tensor = torch.frombuffer(creator.buf, dtype=torch.int8)
        parameter = torch.nn.Parameter(tensor, requires_grad=False)
        assert parameter.data_ptr() == tensor.data_ptr()
        parameter[0] = 43
        parameter[-1] = 79
        assert (peer.buf[0], peer.buf[-1]) == (43, 79)
        peer.buf[size // 2] = 61
        assert int(parameter[size // 2]) == 61
        creator.close()
        # The embedding cleanup drops Parameter aliases immediately after the
        # buffer closes. No access to that closed creator mapping is permitted.
        del parameter, tensor
        assert creator._fd is creator._mapping is creator.buf is None
        assert (peer.buf[0], peer.buf[-1], peer.buf[size // 2]) == (43, 79, 61)
        peer.buf[-1] = 101
        assert peer.buf[-1] == 101
        creator.close()
    finally:
        if peer is not None:
            peer.close()
        if creator is not None:
            creator.close()


@pytest.mark.parametrize("size", [0, -1, True, 4096.0, "4096"])
def test_invalid_memfd_size_is_rejected_before_creation(backing_class, size):
    with (
        patch.object(os, "memfd_create", side_effect=AssertionError("invalid size allocated backing")),
        pytest.raises(ValueError, match="positive integer"),
    ):
        backing_class(size)


@pytest.mark.parametrize("invalid", ["list", "tag", "short", "bool_pid", "negative_fd", "wrong_size"])
def test_memfd_descriptor_type_and_size_validation_precedes_open(backing_class, invalid):
    creator = backing_class(4099)
    try:
        descriptor = creator.name
        values = {
            "list": list(descriptor),
            "tag": ("other", *descriptor[1:]),
            "short": descriptor[:-1],
            "bool_pid": (descriptor[0], True, *descriptor[2:]),
            "negative_fd": (*descriptor[:2], -1, *descriptor[3:]),
            "wrong_size": (*descriptor[:5], descriptor[5] + 1),
        }
        with (
            patch.object(os, "open", side_effect=AssertionError("invalid descriptor opened a process FD")),
            pytest.raises(ValueError, match="descriptor or size"),
        ):
            backing_class(4099, values[invalid])
    finally:
        creator.close()


def test_closed_creator_fd_cannot_be_attached(backing_class):
    creator = backing_class(4099)
    descriptor = creator.name
    creator.close()
    with pytest.raises(FileNotFoundError):
        backing_class(4099, descriptor)


def test_reused_fd_identity_is_rejected_and_opened_fd_is_closed(backing_class):
    creator = backing_class(4099)
    replacement = backing_class(4099)
    opened = []

    def reused_descriptor(path, flags):
        descriptor = os.dup(replacement._fd)
        opened.append(descriptor)
        return descriptor

    try:
        with (
            patch.object(os, "open", side_effect=reused_descriptor),
            pytest.raises(ValueError, match="identity changed"),
        ):
            backing_class(4099, creator.name)
        with pytest.raises(OSError):
            os.fstat(opened[0])
        assert creator.buf[0] == replacement.buf[0] == 0
    finally:
        replacement.close()
        creator.close()


def test_changed_backing_size_is_rejected_before_peer_mapping(backing_class):
    creator = backing_class(8192)
    try:
        os.ftruncate(creator._fd, 4096)
        with pytest.raises(ValueError, match="identity changed"):
            backing_class(8192, creator.name)
    finally:
        creator.close()


def test_exported_memoryview_close_failure_keeps_fd_and_mapping_for_retry(backing_class):
    buffer = backing_class(4099)
    alias = buffer.buf[:]
    descriptor = buffer._fd
    try:
        with pytest.raises(BufferError):
            buffer.close()
        assert buffer._fd == descriptor and buffer._mapping is not None and buffer.buf is not None
        assert alias[0] == buffer.buf[0] == 0
        alias.release()
        buffer.close()
        assert buffer.buf is buffer._fd is buffer._mapping is None
        buffer.close()
    finally:
        alias.release()
        buffer.close()


@pytest.mark.parametrize("failure", ["create", "pretouch", "peer_advice", "peer_identity"])
def test_memfd_failure_is_collective_and_never_falls_back_to_posix(failure):
    state = _allocator(members=2)
    original = state.namespace["_MemfdSharedMemory"]
    original_broadcast = state.namespace["dist"].broadcast_object_list

    def broadcast(payload, src, group):
        original_broadcast(payload, src, group)
        if failure == "peer_identity" and state.local.rank == 1:
            descriptor = payload[0]
            payload[0] = (*descriptor[:4], descriptor[4] + 1, descriptor[5])

    def allocate(size, name=None):
        if failure == "create" and name is None:
            raise AttributeError("memfd_create unavailable")
        if failure == "pretouch" and name is None:
            original.__init__.__globals__["bytes"] = Mock(side_effect=MemoryError("creator page pre-touch failed"))
        if failure == "peer_advice" and name is not None:
            with patch.object(mmap, "MADV_HUGEPAGE", 2**31 - 1):
                return original(size, name)
        return original(size, name)

    state.namespace["_MemfdSharedMemory"] = allocate
    state.namespace["dist"].broadcast_object_list = broadcast

    def attach(rank):
        state.local.rank = rank
        try:
            return state.cls((ROWS, WIDTH), torch.int8, "cpu", state.group(rank))
        except RuntimeError as exc:
            return exc

    with (
        patch(
            "multiprocessing.shared_memory.SharedMemory",
            side_effect=AssertionError("large memfd fell back to POSIX"),
        ),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        results = list(pool.map(attach, range(2)))
    assert all(isinstance(value, RuntimeError) for value in results)
    assert state.mappings == {}
    if failure in {"create", "pretouch"}:
        assert state.registrations == []
        assert all("creation failed" in str(value) for value in results)
    else:
        assert all("rank 1:" in str(value) for value in results)
        assert len(state.unregistrations) == 3


def test_memfd_pretouch_completes_before_broadcast_and_registration():
    state = _allocator(members=2)
    real_backing = state.namespace["_MemfdSharedMemory"]
    real_broadcast = state.namespace["dist"].broadcast_object_list
    captured = []

    def allocate(size, name=None):
        value = real_backing(size, name)
        if name is None:
            assert bytes(value.buf) == bytes(size)
            value.buf[-1] = 79
            captured.append(value)
        return value

    def broadcast(payload, src, group):
        if state.local.rank == 0:
            assert captured[0].buf[-1] == 79
            assert state.registrations == []
        real_broadcast(payload, src, group)

    state.namespace["_MemfdSharedMemory"] = allocate
    state.namespace["dist"].broadcast_object_list = broadcast

    def attach(rank):
        state.local.rank = rank
        return state.cls((ROWS, WIDTH), torch.int8, "cpu", state.group(rank))

    buffers = []
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            buffers = list(pool.map(attach, range(2)))
        assert int(buffers[0].tensor[-1, -1]) == int(buffers[1].tensor[-1, -1]) == 79
        assert len(state.created) == 1
        assert captured[0]._fd is not None  # Leader FD survives the all-peer fence.
        assert all(len(buffer._registered_host_ptrs) == 3 for buffer in buffers)
    finally:
        for buffer in buffers:
            buffer.close()
    assert state.mappings == {}


@pytest.mark.parametrize("collective", ["broadcast_object_list", "all_gather_object", "barrier"])
def test_collective_api_failure_releases_this_ranks_backing_and_registered_ranges(collective):
    state = _allocator()
    getattr(state.namespace["dist"], collective)
    setattr(state.namespace["dist"], collective, Mock(side_effect=RuntimeError("CPU collective timed out")))
    with pytest.raises(RuntimeError, match="failed.*CPU collective timed out"):
        state.cls((ROWS, WIDTH), torch.int8, "cpu", state.group())
    assert state.mappings == {}
    assert len(state.unregistrations) == (0 if collective == "broadcast_object_list" else 3)


@pytest.mark.parametrize("collective", ["all_gather_object", "barrier"])
def test_collective_failure_and_unregister_failure_preserve_owner_for_retry(collective):
    state = _allocator()
    original = state.cls.__init__.__globals__["_host_library"]

    def library():
        result = original()
        register = result.aclrtHostRegisterV2

        def register_and_fail_cleanup(pointer, size, flags):
            rc = register(pointer, size, flags)
            state.unregister_failures.add((0, pointer.value))
            return rc

        result.aclrtHostRegisterV2 = register_and_fail_cleanup
        return result

    state.cls.__init__.__globals__["_host_library"] = library
    setattr(state.namespace["dist"], collective, Mock(side_effect=RuntimeError("CPU collective timed out")))
    with pytest.raises(state.error_cls, match="release failed") as info:
        state.cls((ROWS, WIDTH), torch.int8, "cpu", state.group())
    buffer = info.value.buffer
    assert len(buffer._registered_host_ptrs) == 3
    assert buffer.shm.buf is not None and buffer.shm._fd is not None
    assert buffer._closing
    state.unregister_failures.clear()
    buffer.close()
    assert buffer.shm is None and state.mappings == {}


def test_creator_pretouch_and_backing_close_failure_retain_outer_owner_for_retry():
    state = _allocator()
    aliases = []

    def fail_pretouch(length):
        backing = sys._getframe(1).f_locals["self"]
        aliases.append(backing.buf[:])
        raise MemoryError("creator pre-touch failed with an exported view")

    state.namespace["bytes"] = fail_pretouch
    try:
        with pytest.raises(state.error_cls, match="creation failed.*release failed") as info:
            state.cls((ROWS, WIDTH), torch.int8, "cpu", state.group())
        buffer = info.value.buffer
        assert buffer.shm.buf is not None and buffer.shm._fd is not None
        assert state.registrations == [] and state.mappings == {}
        assert len(aliases) == 1 and aliases[0][0] == 0
        aliases.pop().release()
        buffer.close()
        assert buffer.shm is None
    finally:
        for alias in aliases:
            alias.release()
