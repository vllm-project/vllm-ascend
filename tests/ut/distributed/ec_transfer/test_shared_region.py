# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0

import mmap
import os
import tempfile
import uuid

import vllm_ascend.distributed.ec_transfer.ec_connector.cpu.ec_shared_region as region_mod
from vllm_ascend.distributed.ec_transfer.ec_connector.cpu.ec_shared_region import (
    AscendECSharedRegion,
)


def _temporary_mmap(size: int):
    # Ownership is transferred to the caller, which closes both resources.
    backing_file = tempfile.TemporaryFile()  # noqa: SIM115
    backing_file.truncate(size)
    mmap_obj = mmap.mmap(
        backing_file.fileno(),
        size,
        flags=mmap.MAP_SHARED,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    return backing_file, mmap_obj


def test_fallback_populate_write_preserves_bytes():
    size = 3 * mmap.PAGESIZE
    backing_file, mmap_obj = _temporary_mmap(size)

    try:
        expected = bytes(index % 251 for index in range(size))
        mmap_obj[:] = expected
        region_mod._fallback_populate_write(mmap_obj, 0, size)
        assert mmap_obj[:] == expected
    finally:
        mmap_obj.close()
        backing_file.close()


def test_region_initializes_when_madvise_is_unsupported(monkeypatch):
    fallback_calls = []
    real_fallback = region_mod._fallback_populate_write

    def spy_fallback(mmap_obj, offset, length):
        fallback_calls.append((offset, length))
        real_fallback(mmap_obj, offset, length)

    monkeypatch.setattr(
        region_mod.mmap,
        "MADV_POPULATE_WRITE",
        0x7FFFFFFF,
        raising=False,
    )
    monkeypatch.setattr(region_mod, "_fallback_populate_write", spy_fallback)
    region = AscendECSharedRegion(
        engine_id=f"ut_{uuid.uuid4().hex}",
        num_blocks=4,
        block_size_bytes=64,
    )
    path = region._mmap_path

    try:
        assert region.blocks.shape == (4, 64)
        assert region._is_creator
        assert fallback_calls == [(0, 256)]
    finally:
        region.cleanup()

    assert not os.path.exists(path)


def test_regions_share_upstream_layout_and_cleanup():
    engine_id = f"ut_{uuid.uuid4().hex}"
    creator = AscendECSharedRegion(
        engine_id=engine_id,
        num_blocks=4,
        block_size_bytes=64,
    )
    joiner = AscendECSharedRegion(
        engine_id=engine_id,
        num_blocks=4,
        block_size_bytes=64,
    )
    path = creator._mmap_path

    try:
        creator.blocks[2, 7] = 42
        assert joiner.blocks[2, 7].item() == 42

        joiner.cleanup()
        assert os.path.exists(path)
    finally:
        joiner.cleanup()
        creator.cleanup()

    assert not os.path.exists(path)
