# SPDX-License-Identifier: Apache-2.0
import json
import multiprocessing
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from vllm_ascend.model_loader.engram_ssd_artifact import prepare_artifact


def checkpoint(root):
    root.mkdir()
    mapping = {}
    for layer, rows in [(1, 3), (14, 5)]:
        weight = (
            ((np.arange(rows * 256).reshape(rows, 256) + np.arange(rows)[:, None] * 7) % 256).astype(np.uint8).tobytes()
        )
        scales = bytes([0, 1, 126, 127, 128, 129, 254, 255]) * rows
        name = f"layer-{layer}.safetensors"
        keys = [f"layers.{layer}.engram.embed.{part}" for part in ["weight", "scale"]]
        header = {
            keys[0]: dict(dtype="I8", shape=[rows, 256], data_offsets=[0, len(weight)]),
            keys[1]: dict(dtype="F8_E8M0", shape=[rows, 8], data_offsets=[len(weight), len(weight) + len(scales)]),
        }
        raw = json.dumps(header).encode()
        (root / name).write_bytes(struct.pack("<Q", len(raw)) + raw + weight + scales)
        mapping.update(dict.fromkeys(keys, name))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    return root


def prepare(model, cache):
    return prepare_artifact(model, cache, {1: 3, 14: 5})


def test_pack_and_reuse_without_rewriting(tmp_path):
    model = checkpoint(tmp_path / "model")
    result = prepare(model, tmp_path / "cache")
    table = Path(result["table_paths"][0])
    meta = json.loads(Path(result["manifest"]).read_text())
    assert meta["complete"] is True
    assert meta["layer_row_bases"] == [0, 4096]
    data = table.read_bytes()
    expected_bits = np.array(
        [0x00400000, 1 << 23, 126 << 23, 127 << 23, 128 << 23, 129 << 23, 254 << 23, 0x7FC00000], dtype="<u4"
    )
    for base, rows in [(0, 3), (4096, 5)]:
        for row in range(rows):
            start = (base + row) * 288
            assert data[start : start + 256] == bytes((value + row * 7) % 256 for value in range(256))
            assert data[start + 256 : start + 288] == expected_bits.tobytes()
    before = (table.stat().st_ino, table.stat().st_mtime_ns, Path(result["manifest"]).stat().st_mtime_ns)
    assert prepare(model, tmp_path / "cache") == result
    assert before == (table.stat().st_ino, table.stat().st_mtime_ns, Path(result["manifest"]).stat().st_mtime_ns)


def test_changed_table_with_restored_mtime_is_not_reused(tmp_path):
    model = checkpoint(tmp_path / "model")
    result = prepare(model, tmp_path / "cache")
    table = Path(result["table_paths"][0])
    stat = table.stat()
    with table.open("r+b") as stream:
        stream.write(b"broken!")
    os.utime(table, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    prepare(model, tmp_path / "cache")
    assert table.read_bytes()[:7] == bytes(range(7))


def test_unindexed_checkpoint_ignores_shared_safetensors_metadata(tmp_path):
    model = checkpoint(tmp_path / "model")
    (model / "model.safetensors.index.json").unlink()
    for source in model.glob("*.safetensors"):
        data = source.read_bytes()
        size = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + size])
        header["__metadata__"] = {"format": "pt"}
        raw = json.dumps(header).encode()
        source.write_bytes(struct.pack("<Q", len(raw)) + raw + data[8 + size :])
    result = prepare(model, tmp_path / "cache")
    assert Path(result["table_paths"][0]).exists()


def test_source_change_uses_new_version(tmp_path):
    model = checkpoint(tmp_path / "model")
    first = prepare(model, tmp_path / "cache")
    source = model / "layer-1.safetensors"
    stat = source.stat()
    with source.open("r+b") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        stream.seek(8 + header_size)
        stream.write(b"\x63")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = prepare(model, tmp_path / "cache")
    assert first["artifact_version"] != second["artifact_version"]
    assert Path(first["table_paths"][0]).exists()
    assert Path(second["table_paths"][0]).read_bytes()[0] == 99


def test_truncated_artifact_is_rebuilt(tmp_path):
    model = checkpoint(tmp_path / "model")
    result = prepare(model, tmp_path / "cache")
    table = Path(result["table_paths"][0])
    expected = table.read_bytes()
    table.write_bytes(b"partial")
    assert prepare(model, tmp_path / "cache") == result
    assert table.read_bytes() == expected


def test_failed_write_never_publishes_complete_manifest(tmp_path, monkeypatch):
    model = checkpoint(tmp_path / "model")
    from vllm_ascend.model_loader import engram_ssd_artifact as module

    real = module._write_table

    def interrupted(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(module, "_write_table", interrupted)
    with pytest.raises(OSError, match="write failure"):
        prepare(model, tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*/manifest.json"))
    monkeypatch.setattr(module, "_write_table", real)
    result = prepare(model, tmp_path / "cache")
    assert Path(result["table_paths"][0]).stat().st_size == 8192 * 288


def test_invalid_shape_rejected_before_output(tmp_path):
    model = checkpoint(tmp_path / "model")
    with pytest.raises(ValueError, match="shape"):
        prepare_artifact(model, tmp_path / "cache", {1: 4, 14: 5})
    assert not list(tmp_path.glob("cache/**/*.bin"))


def _concurrent_prepare(model, cache, queue):
    try:
        result = prepare(model, cache)
        table = Path(result["table_paths"][0])
        queue.put((table.stat().st_ino, table.stat().st_mtime_ns))
    except Exception as error:
        queue.put(str(error))


def test_concurrent_initialization_publishes_once(tmp_path):
    model = checkpoint(tmp_path / "model")
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    processes = [ctx.Process(target=_concurrent_prepare, args=(model, tmp_path / "cache", queue)) for _ in range(3)]
    for proc in processes:
        proc.start()
    values = [queue.get(timeout=60) for _ in processes]
    for proc in processes:
        proc.join(60)
        assert proc.exitcode == 0
    assert all(isinstance(v, tuple) for v in values), values
    assert len(set(values)) == 1


def test_insufficient_space_fails_without_publishing(tmp_path, monkeypatch):
    model = checkpoint(tmp_path / "model")
    from vllm_ascend.model_loader import engram_ssd_artifact as module

    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: type("Usage", (), {"free": 0})())
    with pytest.raises(OSError, match="space"):
        prepare(model, tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*/manifest.json"))
