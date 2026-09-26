# SPDX-License-Identifier: Apache-2.0
"""Tests for the streaming safetensors reader/writer (real byte round-trips).

Includes the strict-parsing regressions from the independent review: invalid
offsets, overlapping ranges, truncated payloads, boolean shape dimensions and
duplicate header keys must all be rejected before any byte is copied.
"""

import json
import struct

import pytest

from tools.glm_reduced.errors import ReductionError, UnsupportedFormatError
from tools.glm_reduced.safetensors_io import (
    OutTensor,
    plan_shards,
    read_safetensors,
    sha256_of_tensor,
    shard_filename,
    write_safetensors_from_bytes,
    write_shard,
)

from .conftest import f16_tensor


def _write_raw(path, entries: dict, data: bytes = b"abcdefgh"):
    header = json.dumps(entries).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def test_roundtrip_bytes_and_header(tmp_path):
    tensors = [
        ("a.weight", *f16_tensor((4, 4), seed=1)),
        ("b.weight", "F8_E4M3", (128, 128), bytes(range(256)) * 64),
    ]
    path = tmp_path / "model.safetensors"
    write_safetensors_from_bytes(str(path), tensors)

    parsed = read_safetensors(str(path))
    assert set(parsed.tensors) == {"a.weight", "b.weight"}
    assert parsed.tensors["a.weight"].shape == (4, 4)
    assert parsed.tensors["b.weight"].nbytes == 128 * 128
    assert sha256_of_tensor(parsed, parsed.tensors["a.weight"]) == sha256_of_tensor(parsed, parsed.tensors["a.weight"])


def test_official_reader_preserves_real_tensor_values(tmp_path):
    st_numpy = pytest.importorskip("safetensors.numpy")
    np = pytest.importorskip("numpy")
    arrays = {
        "x": np.arange(12, dtype=np.float16).reshape(3, 4),
        "scale": np.array([0.125], dtype=np.float32),
    }
    src_path = tmp_path / "input.safetensors"
    st_numpy.save_file(arrays, str(src_path))
    src = read_safetensors(str(src_path))
    tensors = [OutTensor(k, v.dtype, v.shape, v.nbytes, src, v) for k, v in src.tensors.items()]
    dst = tmp_path / "output.safetensors"
    write_shard(str(dst), tensors)
    result = st_numpy.load_file(str(dst))
    assert result.keys() == arrays.keys()
    for name in arrays:
        np.testing.assert_array_equal(result[name], arrays[name])


@pytest.mark.parametrize("offsets", [[-8, 0], [0.0, 8.0], [True, 9], [8, 0], [0]])
def test_reject_invalid_offsets(tmp_path, offsets):
    path = tmp_path / "bad.safetensors"
    _write_raw(path, {"x": {"dtype": "I64", "shape": [1], "data_offsets": offsets}})
    with pytest.raises(ReductionError):
        read_safetensors(str(path))


def test_reject_overlapping_ranges(tmp_path):
    path = tmp_path / "bad.safetensors"
    entry = {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}
    _write_raw(path, {"x": entry, "y": entry})
    with pytest.raises(ReductionError, match="overlap"):
        read_safetensors(str(path))


def test_reject_truncated_payload_at_parse(tmp_path):
    path = tmp_path / "bad.safetensors"
    _write_raw(path, {"x": {"dtype": "I64", "shape": [2], "data_offsets": [0, 16]}})
    with pytest.raises(ReductionError, match="truncated"):
        read_safetensors(str(path))


def test_reject_boolean_shape(tmp_path):
    path = tmp_path / "bad.safetensors"
    _write_raw(path, {"x": {"dtype": "I64", "shape": [True], "data_offsets": [0, 8]}})
    with pytest.raises(ReductionError, match="shape"):
        read_safetensors(str(path))


def test_reject_duplicate_header_keys(tmp_path):
    path = tmp_path / "bad.safetensors"
    entry = json.dumps({"dtype": "I64", "shape": [1], "data_offsets": [0, 8]})
    header = b'{"x": %s, "x": %s}' % (entry.encode(), entry.encode())
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"abcdefgh")
    with pytest.raises(ReductionError, match="duplicate"):
        read_safetensors(str(path))


def test_reject_bad_metadata_type(tmp_path):
    path = tmp_path / "bad.safetensors"
    _write_raw(
        path,
        {"__metadata__": {"k": 3}, "x": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}},
    )
    with pytest.raises(ReductionError, match="__metadata__"):
        read_safetensors(str(path))


def test_rejects_bad_shape_dtype_combo(tmp_path):
    path = tmp_path / "bad.safetensors"
    with pytest.raises(UnsupportedFormatError, match="byte range"):
        write_safetensors_from_bytes(str(path), [("x", "F16", (3, 3), b"\x00" * 10)])


def test_rejects_unknown_dtype(tmp_path):
    path = tmp_path / "bad.safetensors"
    with pytest.raises(UnsupportedFormatError, match="unsupported dtype"):
        write_safetensors_from_bytes(str(path), [("x", "F4", (8,), b"\x00" * 4)])


def test_plan_shards_respects_budget():
    def fake(nbytes):
        return OutTensor("t", "F16", (nbytes // 2,), nbytes, None, None)

    shards = plan_shards([fake(10), fake(10), fake(7), fake(10)], max_shard_bytes=20)
    assert [[t.nbytes for t in shard] for shard in shards] == [[10, 10], [7, 10]]
    with pytest.raises(UnsupportedFormatError):
        plan_shards([fake(10)], max_shard_bytes=0)


def test_shard_filename():
    assert shard_filename(0, 1) == "model.safetensors"
    assert shard_filename(0, 3) == "model-00001-of-00003.safetensors"
    assert shard_filename(2, 3) == "model-00003-of-00003.safetensors"
