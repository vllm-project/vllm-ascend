# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for packed NOVA artifacts and allocator aliases."""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.quantization.methods.kv_cache.nova_kv.codec import (
    FORMAT_VERSION,
    NovaCodec,
    cache_views,
    pack_int2,
    packet_bytes,
    unpack_int2,
)


def make_codec(heads=2, dim=8):
    matrix = torch.eye(dim).repeat(heads, 1, 1)
    books = torch.ones(heads, dim // 4, 256, 4)
    return NovaCodec(torch.zeros(heads, dim), matrix, matrix, matrix, books)


def test_every_int2_byte_roundtrips():
    payload = torch.arange(256).to(torch.uint8).reshape(16, 16)
    assert torch.equal(pack_int2(unpack_int2(payload)), payload)


@pytest.mark.parametrize("dim", [0, -4, 5])
def test_invalid_packet_dimension(dim):
    with pytest.raises(ValueError):
        packet_bytes(dim)


@pytest.mark.parametrize("count", [0, 1, 17])
def test_exact_centroid_and_constant_values(count):
    codec = make_codec()
    key = torch.ones(count, 2, 8)
    value = torch.full_like(key, 7)
    encoded = codec.encode(key, value, chunk_size=3)
    assert encoded[0].shape == encoded[1].shape == (count, 2, 2)
    assert encoded[2].shape == (count, 2, 3)
    reconstructed_key, reconstructed_value = codec.decode(*encoded, output_dtype=torch.bfloat16)
    torch.testing.assert_close(reconstructed_key.float(), key)
    torch.testing.assert_close(reconstructed_value.float(), value)


def test_nonorthogonal_key_transform():
    codec = make_codec(1)
    codec.key_transform *= 2
    # make_codec shares the identity tensor; assign independent inverse/value matrices.
    codec.key_inverse = torch.eye(8)[None] / 2
    codec.value_transform = torch.eye(8)[None]
    key = torch.ones(3, 1, 8)
    reconstructed_key, _ = codec.decode(*codec.encode(key, torch.zeros_like(key)))
    torch.testing.assert_close(reconstructed_key, key)


def test_chunk_size_and_input_geometry():
    codec = make_codec()
    with pytest.raises(ValueError, match="chunk_size"):
        codec.encode(torch.ones(1, 2, 8), torch.ones(1, 2, 8), chunk_size=0)
    with pytest.raises(ValueError, match="geometry"):
        codec.encode(torch.ones(1, 1, 8), torch.ones(1, 1, 8))


@pytest.mark.parametrize("heads,tp_size,rank,local_heads,start", [(4, 2, 1, 2, 2), (2, 4, 3, 1, 1)])
def test_artifact_tensor_parallel_sharding(tmp_path, heads, tp_size, rank, local_heads, start):
    codec = make_codec(heads)
    codec.key_mean[:, 0] = torch.arange(heads)
    artifact = {field.name: getattr(codec, field.name) for field in fields(codec)}
    artifact["format_version"] = FORMAT_VERSION
    path = tmp_path / "layer.pt"
    torch.save(artifact, path)
    loaded = NovaCodec.load(str(path), torch.device("cpu"), rank, tp_size, local_heads)
    torch.testing.assert_close(loaded.key_mean, codec.key_mean[start : start + local_heads])


@pytest.mark.parametrize("rank,tp_size,heads", [(2, 2, 1), (0, 3, 1), (0, 1, 1), (0, 0, 1)])
def test_artifact_rejects_invalid_parallel_geometry(tmp_path, rank, tp_size, heads):
    codec = make_codec(2)
    artifact = {field.name: getattr(codec, field.name) for field in fields(codec)}
    artifact["format_version"] = FORMAT_VERSION
    path = tmp_path / "layer.pt"
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="geometry"):
        NovaCodec.load(str(path), torch.device("cpu"), rank, tp_size, heads)


def test_artifact_rejects_nonfinite_transform(tmp_path):
    codec = make_codec()
    artifact = {field.name: getattr(codec, field.name) for field in fields(codec)}
    artifact["format_version"] = FORMAT_VERSION
    artifact["key_transform"][0, 0, 0] = float("nan")
    path = tmp_path / "layer.pt"
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="artifact tensor"):
        NovaCodec.load(str(path), torch.device("cpu"), 0, 1, 2)


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
def test_page_padding_and_nonzero_storage_offset(dtype):
    spec = SimpleNamespace(block_size=128, num_kv_heads=2, head_size=256, page_size_bytes=65536)
    backing = torch.zeros(3 * spec.page_size_bytes + 64, dtype=dtype)
    cache = cache_views(backing[64:], spec, 128)
    cache[1, 0, 0, 0] = 17
    assert backing[64 + spec.page_size_bytes] == 17
    assert cache.shape == (3, 128, 2, 134)
    assert not backing[:64].any()
    assert not backing[64 + 34304 : 64 + spec.page_size_bytes].any()


def test_cache_rejects_partial_or_split_pages():
    spec = SimpleNamespace(block_size=128, num_kv_heads=2, head_size=256, page_size_bytes=65536)
    with pytest.raises(ValueError, match="complete"):
        cache_views(torch.zeros(65535, dtype=torch.int8), spec, 128)
    with pytest.raises(ValueError, match="split"):
        cache_views(torch.zeros(65536, dtype=torch.int8), spec, 64)
