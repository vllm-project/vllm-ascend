# SPDX-License-Identifier: Apache-2.0
"""Minimal streaming safetensors reader/writer (stdlib only).

The reducer copies raw tensor bytes without interpreting values, so it never
needs torch/numpy and is dtype-agnostic (BF16, FP8, packed int4 all pass
through unchanged). Reads are bounded: header parse plus chunked byte copies.

Parsing is strict by design: malformed headers (duplicate keys, negative or
fractional offsets, boolean dimensions, overlapping ranges, truncated data
sections) are rejected before any byte is copied, so a corrupt source cannot
silently yield a plausible-looking output.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass, field

from .errors import UnsupportedFormatError

_HEADER_LEN = struct.Struct("<Q")
_COPY_CHUNK = 8 * 1024 * 1024

# Dtype name -> item size in bytes. F4/F6 packed dtypes are intentionally not
# listed; a checkpoint using them fails explicitly in ``_validate`` rather than
# being silently mishandled.
_DTYPE_SIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

MAX_HEADER_BYTES = 256 * 1024 * 1024


@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int  # byte offset relative to the start of the data region
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


@dataclass
class SafetensorsFile:
    path: str
    tensors: dict[str, TensorInfo] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    data_offset: int = 0  # absolute file offset of the data region


def _is_int(value) -> bool:
    # bool is a subclass of int; reject it explicitly.
    return isinstance(value, int) and not isinstance(value, bool)


def _validate(name: str, entry: dict, begin: int, end: int) -> TensorInfo:
    dtype = entry.get("dtype")
    shape = entry.get("shape")
    if dtype not in _DTYPE_SIZE:
        raise UnsupportedFormatError(
            f"tensor {name!r} uses unsupported dtype {dtype!r}; "
            f"supported dtypes: {sorted(_DTYPE_SIZE)}. Packed dtypes (e.g. F4) are not byte-copyable "
            "by this tool yet."
        )
    if not isinstance(shape, list) or any(not _is_int(d) or d < 0 for d in shape):
        raise UnsupportedFormatError(f"tensor {name!r} has malformed shape {shape!r}")
    expected = _DTYPE_SIZE[dtype]
    for dim in shape:
        expected *= dim
    nbytes = end - begin
    if nbytes != expected:
        raise UnsupportedFormatError(
            f"tensor {name!r} byte range ({nbytes}) does not match dtype/shape product ({expected}); "
            "the checkpoint is corrupt or uses a packing this tool does not understand"
        )
    return TensorInfo(name=name, dtype=dtype, shape=tuple(shape), begin=begin, end=end)


def _no_duplicate_keys(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise UnsupportedFormatError(f"safetensors header has duplicate key {key!r}")
        result[key] = value
    return result


def read_safetensors(path: str) -> SafetensorsFile:
    """Parse and fully validate a safetensors header without reading tensor data."""
    file_size = os.path.getsize(path)
    with open(path, "rb") as handle:
        raw = handle.read(_HEADER_LEN.size)
        if len(raw) != _HEADER_LEN.size:
            raise UnsupportedFormatError(f"{path}: file too small to be safetensors")
        (header_len,) = _HEADER_LEN.unpack(raw)
        if header_len <= 0 or header_len > MAX_HEADER_BYTES:
            raise UnsupportedFormatError(f"{path}: implausible safetensors header length {header_len}")
        header_bytes = handle.read(header_len)
        if len(header_bytes) != header_len:
            raise UnsupportedFormatError(f"{path}: truncated safetensors header")
    try:
        header = json.loads(header_bytes.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise UnsupportedFormatError(f"{path}: invalid safetensors header encoding: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise UnsupportedFormatError(f"{path}: invalid safetensors header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise UnsupportedFormatError(f"{path}: safetensors header must be a JSON object")
    result = SafetensorsFile(path=path, data_offset=_HEADER_LEN.size + header_len)
    ranges: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict) or any(not isinstance(v, str) for v in (entry or {}).values()):
                raise UnsupportedFormatError(f"{path}: __metadata__ must be a string->string object")
            result.metadata = dict(entry or {})
            continue
        if not isinstance(entry, dict):
            raise UnsupportedFormatError(f"{path}: tensor {name!r} entry must be a JSON object")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(_is_int(o) for o in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise UnsupportedFormatError(
                f"{path}: tensor {name!r} has invalid data_offsets {offsets!r} "
                "(must be two non-negative ordered integers)"
            )
        result.tensors[name] = _validate(name, entry, offsets[0], offsets[1])
        ranges.append((offsets[0], offsets[1], name))
    # Reject overlapping tensor ranges and ranges past the end of the file.
    data_size = file_size - result.data_offset
    if data_size < 0:
        raise UnsupportedFormatError(f"{path}: header extends past end of file")
    ranges.sort()
    prev_end = 0
    for begin, end, name in ranges:
        if begin < prev_end:
            raise UnsupportedFormatError(f"{path}: tensor {name!r} byte range overlaps another tensor")
        prev_end = max(prev_end, end)
    if ranges and ranges[-1][1] > data_size:
        raise UnsupportedFormatError(
            f"{path}: tensor data extends past the end of the file "
            f"(needs {ranges[-1][1]} bytes, data section has {data_size}); the shard is truncated"
        )
    return result


def copy_tensor_data(src: SafetensorsFile, info: TensorInfo, dst, digest: hashlib._Hash) -> None:
    """Stream one tensor's raw bytes from ``src`` into open file ``dst``."""
    with open(src.path, "rb") as handle:
        handle.seek(src.data_offset + info.begin)
        remaining = info.nbytes
        while remaining > 0:
            chunk = handle.read(min(_COPY_CHUNK, remaining))
            if not chunk:
                raise UnsupportedFormatError(f"{src.path}: truncated data while reading {info.name!r}")
            dst.write(chunk)
            digest.update(chunk)
            remaining -= len(chunk)


def sha256_of_tensor(src: SafetensorsFile, info: TensorInfo) -> str:
    digest = hashlib.sha256()
    with open(src.path, "rb") as handle:
        handle.seek(src.data_offset + info.begin)
        remaining = info.nbytes
        while remaining > 0:
            chunk = handle.read(min(_COPY_CHUNK, remaining))
            if not chunk:
                raise UnsupportedFormatError(f"{src.path}: truncated data while reading {info.name!r}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


@dataclass
class OutTensor:
    """A tensor to write: raw bytes are pulled lazily from a source shard."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    src: SafetensorsFile
    src_info: TensorInfo


def plan_shards(tensors: list[OutTensor], max_shard_bytes: int) -> list[list[OutTensor]]:
    """Greedy in-order packing of tensors into output shards."""
    if max_shard_bytes <= 0:
        raise UnsupportedFormatError(f"max_shard_bytes must be positive, got {max_shard_bytes}")
    shards: list[list[OutTensor]] = []
    current: list[OutTensor] = []
    current_bytes = 0
    for tensor in tensors:
        if current and current_bytes + tensor.nbytes > max_shard_bytes:
            shards.append(current)
            current, current_bytes = [], 0
        current.append(tensor)
        current_bytes += tensor.nbytes
    if current:
        shards.append(current)
    return shards


def shard_filename(index: int, count: int) -> str:
    if count == 1:
        return "model.safetensors"
    return f"model-{index + 1:05d}-of-{count:05d}.safetensors"


def write_shard(path: str, tensors: list[OutTensor]) -> tuple[int, dict[str, str]]:
    """Write one shard; returns (total data bytes, per-tensor sha256)."""
    header: dict[str, dict] = {}
    offset = 0
    for tensor in tensors:
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes
    header_bytes = json.dumps(header).encode("utf-8")
    digests: dict[str, str] = {}
    tmp_path = path + ".partial"
    with open(tmp_path, "wb") as handle:
        handle.write(_HEADER_LEN.pack(len(header_bytes)))
        handle.write(header_bytes)
        for tensor in tensors:
            digest = hashlib.sha256()
            copy_tensor_data(tensor.src, tensor.src_info, handle, digest)
            digests[tensor.name] = digest.hexdigest()
    os.replace(tmp_path, path)
    return offset, digests


def write_safetensors_from_bytes(path: str, tensors: list[tuple[str, str, tuple[int, ...], bytes]]) -> None:
    """Write a safetensors file from in-memory bytes (fixtures/tests/small files)."""
    header: dict[str, dict] = {}
    offset = 0
    for name, dtype, shape, data in tensors:
        info = _validate(name, {"dtype": dtype, "shape": list(shape)}, 0, len(data))
        header[name] = {
            "dtype": info.dtype,
            "shape": list(info.shape),
            "data_offsets": [offset, offset + info.nbytes],
        }
        offset += info.nbytes
    header_bytes = json.dumps(header).encode("utf-8")
    tmp_path = path + ".partial"
    with open(tmp_path, "wb") as handle:
        handle.write(_HEADER_LEN.pack(len(header_bytes)))
        handle.write(header_bytes)
        for _, _, _, data in tensors:
            handle.write(data)
    os.replace(tmp_path, path)


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_COPY_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
