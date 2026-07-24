# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Regression tests for the Ascend filesystem secondary tier.

The upstream ``fs_python`` tier opens block files with ``O_DIRECT``, which
fails with ``EINVAL`` on filesystems / buffers that do not meet its alignment
requirements (e.g. 3FS via FUSE). The Ascend tier performs the same per-block
I/O with buffered flags by default; these tests exercise that buffered path.
"""

import os

import pytest

from vllm_ascend.kv_offload import fs_tier

# Buffered flags (no O_DIRECT), matching AscendFileSystemTierManager defaults.
STORE_FLAGS = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC
LOAD_FLAGS = os.O_RDONLY


def test_store_then_load_round_trip(tmp_path):
    block_size = 4099  # deliberately not a multiple of 512/4096
    payload = bytes((i % 251) for i in range(block_size))
    # Embed the payload at a non-zero offset to exercise offset slicing.
    src = bytearray(block_size * 3)
    offset = block_size  # block index 1
    src[offset : offset + block_size] = payload
    src_view = memoryview(src)

    dest = str(tmp_path / "sub" / "block.bin")
    fs_tier.store_block(dest, src_view, offset, block_size, STORE_FLAGS)

    assert os.path.exists(dest)
    # Temp file must have been renamed away.
    assert not any(p.name.endswith(".tmp") for p in (tmp_path / "sub").iterdir())

    dst = bytearray(block_size * 2)
    dst_view = memoryview(dst)
    load_offset = block_size  # block index 1
    fs_tier.load_block(dest, dst_view, load_offset, block_size, LOAD_FLAGS)
    assert bytes(dst[load_offset : load_offset + block_size]) == payload


def test_store_skips_existing_block(tmp_path):
    block_size = 64
    dest = str(tmp_path / "block.bin")
    with open(dest, "wb") as f:
        f.write(b"\x01" * block_size)

    buf = memoryview(bytearray(b"\x02" * block_size))
    # Should be a no-op because the destination already exists.
    fs_tier.store_block(dest, buf, 0, block_size, STORE_FLAGS)
    with open(dest, "rb") as f:
        assert f.read() == b"\x01" * block_size


def test_load_missing_block_raises(tmp_path):
    dst = memoryview(bytearray(64))
    with pytest.raises(OSError):
        fs_tier.load_block(str(tmp_path / "missing.bin"), dst, 0, 64, LOAD_FLAGS)


def test_load_short_file_removed(tmp_path):
    block_size = 128
    dest = str(tmp_path / "short.bin")
    with open(dest, "wb") as f:
        f.write(b"\x00" * (block_size // 2))  # too short

    dst = memoryview(bytearray(block_size))
    with pytest.raises(OSError):
        fs_tier.load_block(dest, dst, 0, block_size, LOAD_FLAGS)
    # A corrupt/short entry must be removed so it is not served again.
    assert not os.path.exists(dest)


def test_ensure_dir_caches_makedirs(tmp_path, monkeypatch):
    """The directory creation is cached so makedirs is not re-issued."""
    target = str(tmp_path / "a" / "b")
    fs_tier._created_dirs.discard(target)

    calls = []
    real_makedirs = os.makedirs

    def counting_makedirs(path, *args, **kwargs):
        calls.append(path)
        return real_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(fs_tier.os, "makedirs", counting_makedirs)

    fs_tier._ensure_dir(target)
    fs_tier._ensure_dir(target)
    fs_tier._ensure_dir(target)

    assert os.path.isdir(target)
    # makedirs runs only on the first (uncached) call.
    assert calls == [target]


def test_store_recreates_vanished_dir(tmp_path):
    """store_block recovers if a cached directory was removed underneath it."""
    block_size = 64
    sub = tmp_path / "vanish"
    dest = str(sub / "block.bin")

    payload = bytes((i % 251) for i in range(block_size))
    buf = memoryview(bytearray(payload))

    # Prime the cache, then delete the directory to simulate eviction.
    fs_tier._ensure_dir(str(sub))
    if sub.exists():
        for child in sub.iterdir():
            child.unlink()
        sub.rmdir()

    fs_tier.store_block(dest, buf, 0, block_size, STORE_FLAGS)
    assert os.path.exists(dest)
    with open(dest, "rb") as f:
        assert f.read() == payload
