# SPDX-License-Identifier: Apache-2.0
"""CPU-only, persistent preparation of local Engram SSD tables.

Checkpoint tensors are streamed directly into the runtime's INT8[256] +
FP32[8] row layout. Completed files are immutable from the serving path.
"""

import fcntl
import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path

import numpy as np
from vllm.logger import logger

ROW_BYTES = 288
TILE_ROWS = 4096
CHUNK_ROWS = 65536
MAX_HEADER_BYTES = 64 * 1024 * 1024
FORMAT_VERSION = "engram-int8-fp32-aos288-v1"
SPACE_MARGIN_BYTES = 64 * 1024 * 1024
SAMPLE_BYTES = 64 * 1024


def _sample_hash(path):
    """Bounded integrity check; intentionally not a full-file checksum."""
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for offset in sorted({0, max(0, (size - SAMPLE_BYTES) // 2), max(0, size - SAMPLE_BYTES)}):
            stream.seek(offset)
            digest.update(struct.pack("<Q", offset))
            digest.update(stream.read(SAMPLE_BYTES))
    return digest.hexdigest()


def _file_identity(path):
    st = path.stat()
    return dict(
        path=str(path),
        size=st.st_size,
        inode=st.st_ino,
        mtime_ns=st.st_mtime_ns,
        ctime_ns=st.st_ctime_ns,
        sample_hash=_sample_hash(path),
    )


def _header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size <= MAX_HEADER_BYTES:
            raise ValueError(f"Invalid safetensors header size: {path}")
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError(f"Truncated safetensors header: {path}")
    return json.loads(raw), size + 8, hashlib.sha256(raw).hexdigest()


def _sources(model, layers):
    model = Path(model).resolve(strict=True)
    if not model.is_dir() or not layers:
        raise ValueError("Engram SSD preparation requires a local checkpoint and nonempty layers")
    indexes = sorted(model.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError("Ambiguous safetensors indexes in checkpoint")
    headers = {}
    if indexes:
        mapping = json.loads(indexes[0].read_text())["weight_map"]
    else:
        mapping = {}
        for path in sorted(model.glob("*.safetensors")):
            headers[path] = _header(path)
            for key in headers[path][0]:
                if key == "__metadata__":
                    continue
                if key in mapping:
                    raise ValueError(f"Duplicate safetensors key: {key}")
                mapping[key] = path.name
    result = []
    for layer, rows in sorted(layers.items()):
        entry = dict(layer=layer, rows=rows)
        for part, width, dtypes in [("weight", 256, ("I8",)), ("scale", 8, ("F8_E8M0", "F32"))]:
            suffix = f"layers.{layer}.engram.embed.{part}"
            keys = [key for key in mapping if key == suffix or key.endswith("." + suffix)]
            if len(keys) != 1:
                raise ValueError(f"Expected exactly one checkpoint tensor {suffix}")
            path = (model / mapping[keys[0]]).resolve(strict=True)
            if path not in headers:
                headers[path] = _header(path)
            head, offset, digest = headers[path]
            tensor = head[keys[0]]
            if rows <= 0 or tensor.get("shape") != [rows, width] or tensor.get("dtype") not in dtypes:
                raise ValueError(f"Unsupported Engram tensor dtype/shape: {keys[0]} {tensor}")
            begin, end = tensor["data_offsets"]
            itemsize = 4 if tensor["dtype"] == "F32" else 1
            if begin < 0 or end - begin != rows * width * itemsize or offset + end > path.stat().st_size:
                raise ValueError(f"Invalid tensor offsets: {keys[0]}")
            entry[part] = tensor
            entry[part + "_file"] = _file_identity(path)
            entry[part + "_offset"] = offset + begin
            entry[part + "_header_hash"] = digest
        result.append(entry)
    return result


def _read_rows(stream, offset, start, count, width):
    stream.seek(offset + start * width)
    raw = stream.read(count * width)
    if len(raw) != count * width:
        raise OSError("Checkpoint was truncated during Engram preparation")
    return np.frombuffer(raw, dtype=np.uint8).reshape(count, width)


def _write_table(path, sources, bases, file_bytes):
    digest = hashlib.sha256()
    started = last_log = time.monotonic()
    with path.open("wb") as out:
        os.posix_fallocate(out.fileno(), 0, file_bytes)
        for src, base in zip(sources, bases):
            with (
                Path(src["weight_file"]["path"]).open("rb") as weight,
                Path(src["scale_file"]["path"]).open("rb") as scale,
            ):
                padded = (src["rows"] + TILE_ROWS - 1) // TILE_ROWS * TILE_ROWS
                for start in range(0, padded, CHUNK_ROWS):
                    count = min(CHUNK_ROWS, padded - start)
                    valid = max(0, min(count, src["rows"] - start))
                    packed = np.zeros((count, ROW_BYTES), dtype=np.uint8)
                    packed[:valid, :256] = _read_rows(weight, src["weight_offset"], start, valid, 256)
                    width = 32 if src["scale"]["dtype"] == "F32" else 8
                    raw = _read_rows(scale, src["scale_offset"], start, valid, width)
                    if width == 8:
                        bits = raw.astype("<u4") << 23
                        bits[raw == 0] = 0x00400000
                        bits[raw == 255] = 0x7FC00000
                        raw = bits.view(np.uint8).reshape(valid, 32)
                    packed[:valid, 256:] = raw
                    data = packed.tobytes()
                    out.write(data)
                    digest.update(data)
                    if time.monotonic() - last_log >= 10:
                        logger.info(
                            "Preparing Engram SSD table: %.1f%% (%.1f seconds)",
                            100 * out.tell() / file_bytes,
                            time.monotonic() - started,
                        )
                        last_log = time.monotonic()
                assert out.tell() == (base + padded) * ROW_BYTES
        out.flush()
        os.fsync(out.fileno())
    return digest.hexdigest()


def _reusable(manifest, table, version, sources, bases, file_bytes):
    try:
        meta = json.loads(manifest.read_text())
        st = table.stat()
        return (
            meta["complete"] is True
            and meta["version"] == version
            and meta["sources"] == sources
            and meta["layer_row_bases"] == bases
            and meta["format"] == FORMAT_VERSION
            and meta["file_bytes"] == st.st_size == file_bytes
            and meta["table_mtime_ns"] == st.st_mtime_ns
            and meta["table_ctime_ns"] == st.st_ctime_ns
            and meta["table_inode"] == st.st_ino
            and meta["table_sample_hash"] == _sample_hash(table)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def prepare_artifact(model: str | Path, directory: str | Path, layers: dict[int, int]) -> dict:
    """Return resolved legacy fields, generating only absent/invalid managed data.

    Fast reuse checks source headers, filesystem identity and bounded samples,
    not all checkpoint bytes. Treat checkpoint files as immutable while serving.
    Sampling also catches changes in sampled regions on coarse-timestamp filesystems.
    """
    sources = _sources(model, layers)
    identity = dict(format=FORMAT_VERSION, sources=sources)
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    bases = []
    total_rows = 0
    for src in sources:
        bases.append(total_rows)
        total_rows += (src["rows"] + TILE_ROWS - 1) // TILE_ROWS * TILE_ROWS
    file_bytes = total_rows * ROW_BYTES
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / version
    manifest, table = target / "manifest.json", target / "table.bin"
    result = dict(manifest=str(manifest), artifact_version=version, table_paths=[str(table)])
    with (root / ".prepare.lock").open("a+b") as lock:
        logger.info("Checking Engram SSD artifacts in %s (waiting for preparation lock if needed)", root)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        # The checkpoint might have changed while another launcher held the lock.
        if _sources(model, layers) != sources:
            raise ValueError("Checkpoint changed while waiting to prepare Engram SSD; restart initialization")
        if _reusable(manifest, table, version, sources, bases, file_bytes):
            logger.info("Reusing complete Engram SSD table: %s; no conversion required", table)
            return result
        target.mkdir(exist_ok=True)
        temporary = target / "table.bin.incomplete"
        meta_tmp = target / "manifest.json.incomplete"
        # Only these two private incomplete files are reclaimed under our lock.
        temporary.unlink(missing_ok=True)
        meta_tmp.unlink(missing_ok=True)
        if shutil.disk_usage(root).free < file_bytes + SPACE_MARGIN_BYTES:
            raise OSError(f"Insufficient SSD space in {root}: need {file_bytes + SPACE_MARGIN_BYTES} free bytes")
        logger.info("Preparing Engram SSD table from %s: %d bytes -> %s", model, file_bytes, table)
        try:
            checksum = _write_table(temporary, sources, bases, file_bytes)
            if _sources(model, layers) != sources:
                raise ValueError("Checkpoint changed during Engram SSD preparation; result discarded")
            os.replace(temporary, table)
            table_stat = table.stat()
            meta = dict(
                complete=True,
                version=version,
                format=FORMAT_VERSION,
                sources=sources,
                layer_row_bases=bases,
                file_bytes=file_bytes,
                sha256=checksum,
                table_mtime_ns=table_stat.st_mtime_ns,
                table_ctime_ns=table_stat.st_ctime_ns,
                table_inode=table_stat.st_ino,
                table_sample_hash=_sample_hash(table),
            )
            with meta_tmp.open("w") as stream:
                json.dump(meta, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(meta_tmp, manifest)
            _sync_directory(target)
            _sync_directory(root)
        finally:
            temporary.unlink(missing_ok=True)
            meta_tmp.unlink(missing_ok=True)
        logger.info("Engram SSD preparation complete: %s", table)
    return result
