# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for the 3FS USRBIO (RDMA) KV tier backend.

The native ``hf3fs`` binding is not available in CI, so these tests verify two
things that do not require it:

* :func:`probe` degrades gracefully to ``None`` (buffered fallback) when the
  binding is missing, guaranteeing USRBIO can never break startup.
* :func:`store_block` / :func:`load_block` preserve the buffered tier's file
  semantics (atomic temp+replace, skip-existing, short-read removal, bit-exact
  round trip) when driven by a stand-in backend. The only USRBIO-specific part
  -- the registered-buffer DMA -- is isolated behind ``backend.write`` /
  ``backend.read``; everything around it is exercised here.
"""

import os

import pytest

from vllm_ascend.kv_offload import usrbio


class _FakeBackend:
    """Backend stand-in that performs the transfer with ordinary syscalls."""

    def write(self, fd, view_slice, length):
        return os.write(fd, view_slice[:length])

    def read(self, fd, out_slice, length):
        data = os.read(fd, length)
        out_slice[: len(data)] = data
        return len(data)

    def close(self):
        pass


def test_probe_without_binding_returns_none():
    # The hf3fs binding is absent in CI; probe must fall back, not raise.
    assert usrbio.probe("/nonexistent/3fs/path", 128) is None


def test_store_then_load_round_trip(tmp_path):
    backend = _FakeBackend()
    block_size = 4099  # not a multiple of any alignment
    payload = bytes((i % 251) for i in range(block_size))
    src = bytearray(block_size * 3)
    offset = block_size
    src[offset : offset + block_size] = payload

    dest = str(tmp_path / "sub" / "block.bin")
    usrbio.store_block(backend, dest, memoryview(src), offset, block_size)

    assert os.path.exists(dest)
    assert not any(p.name.endswith(".tmp") for p in (tmp_path / "sub").iterdir())

    dst = bytearray(block_size * 2)
    load_offset = block_size
    usrbio.load_block(
        backend, dest, memoryview(dst), load_offset, block_size
    )
    assert bytes(dst[load_offset : load_offset + block_size]) == payload


def test_store_skips_existing_block(tmp_path):
    backend = _FakeBackend()
    block_size = 64
    dest = str(tmp_path / "block.bin")
    with open(dest, "wb") as f:
        f.write(b"\x01" * block_size)

    buf = memoryview(bytearray(b"\x02" * block_size))
    usrbio.store_block(backend, dest, buf, 0, block_size)
    with open(dest, "rb") as f:
        assert f.read() == b"\x01" * block_size


def test_load_short_file_removed(tmp_path):
    backend = _FakeBackend()
    block_size = 128
    dest = str(tmp_path / "short.bin")
    with open(dest, "wb") as f:
        f.write(b"\x00" * (block_size // 2))

    dst = memoryview(bytearray(block_size))
    with pytest.raises(OSError):
        usrbio.load_block(backend, dest, dst, 0, block_size)
    assert not os.path.exists(dest)


def test_store_short_write_cleans_temp(tmp_path):
    class _ShortWriteBackend(_FakeBackend):
        def write(self, fd, view_slice, length):
            return os.write(fd, view_slice[: length // 2])  # short

    block_size = 128
    dest = str(tmp_path / "sub" / "block.bin")
    buf = memoryview(bytearray(b"\x05" * block_size))

    with pytest.raises(OSError):
        usrbio.store_block(_ShortWriteBackend(), dest, buf, 0, block_size)

    assert not os.path.exists(dest)
    # No temp file must be left behind.
    leftovers = list((tmp_path / "sub").iterdir())
    assert leftovers == []
