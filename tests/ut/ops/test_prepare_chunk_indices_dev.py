# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity checks for the device-side varlen chunk-index builder.

``prepare_chunk_indices`` builds the ``[total_chunks, 2]`` ``(seq_id, chunk_in_seq)`` tensor
with ``.tolist()`` in the middle, i.e. a device->host synchronisation per kernel launch. The
device-side variant is used on the prefill hot path instead, so it has to produce exactly the
same rows - only the row count may differ, because it has to be shape-derived.
"""

import pytest
import torch

from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices, prepare_chunk_indices_dev
from vllm_ascend.ops.triton.kda.utils import prepare_chunk_indices as kda_prepare_chunk_indices
from vllm_ascend.ops.triton.kda.utils import prepare_chunk_indices_dev as kda_prepare_chunk_indices_dev

SEQ_LEN_SETS = (
    (1024,),
    (1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024),
    tuple([1024] * 32),
    (777, 1024, 130),
    (1, 64, 65, 63),
    (37, 96, 192),
    (1216, 1217, 608),
    (8192, 8192, 8192, 8192, 8192, 8192, 8192, 8192),
)
CHUNK_SIZES = (16, 64, 1216)
DTYPES = (torch.int32, torch.int64)
BUILDERS = (
    ("fla", prepare_chunk_indices, prepare_chunk_indices_dev),
    ("kda", kda_prepare_chunk_indices, kda_prepare_chunk_indices_dev),
)


def _cu_seqlens(seq_lens, dtype):
    ends = torch.cumsum(torch.tensor(seq_lens, dtype=torch.int64), 0).tolist()
    return torch.tensor([0, *ends], dtype=dtype)


@pytest.mark.parametrize("seq_lens", SEQ_LEN_SETS)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name,host_builder,device_builder", BUILDERS)
def test_device_builder_matches_host_builder(seq_lens, chunk_size, dtype, name, host_builder, device_builder):
    cu_seqlens = _cu_seqlens(seq_lens, dtype)
    expected = host_builder(cu_seqlens, chunk_size)
    actual = device_builder(cu_seqlens, sum(seq_lens), chunk_size)

    assert actual.dtype == cu_seqlens.dtype
    assert actual.shape[1] == 2
    assert actual.shape[0] >= expected.shape[0]
    # Only the trailing phantom rows may differ, and the upper bound adds at most one row per
    # sequence on top of the exact count.
    assert actual.shape[0] - expected.shape[0] <= len(seq_lens)
    assert torch.equal(actual[: expected.shape[0]], expected)


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("name,host_builder,device_builder", BUILDERS)
def test_phantom_rows_address_a_chunk_past_their_sequence(chunk_size, name, host_builder, device_builder):
    """Any row past the exact count must fall outside its sequence, where kernels mask it."""
    seq_lens = (777, 1024, 130)
    cu_seqlens = _cu_seqlens(seq_lens, torch.int64)
    expected = host_builder(cu_seqlens, chunk_size)
    actual = device_builder(cu_seqlens, sum(seq_lens), chunk_size)
    phantom = actual[expected.shape[0] :]

    assert phantom.shape[0] > 0
    chunks_per_seq = (torch.tensor(seq_lens) + chunk_size - 1) // chunk_size
    assert bool((phantom[:, 1] >= chunks_per_seq[phantom[:, 0]]).all())


@pytest.mark.parametrize("name,host_builder,device_builder", BUILDERS)
def test_builder_row_count_is_the_documented_upper_bound(name, host_builder, device_builder):
    """Aligned sequence lengths give ``total_tokens // chunk_size + num_seqs`` rows."""
    seq_lens = [1024] * 32
    chunk_size = 64
    cu_seqlens = _cu_seqlens(seq_lens, torch.int64)

    actual = device_builder(cu_seqlens, sum(seq_lens), chunk_size)

    assert actual.shape[0] == sum(seq_lens) // chunk_size + len(seq_lens) == 544


@pytest.mark.parametrize("name,host_builder,device_builder", BUILDERS)
def test_builder_returns_an_empty_tensor_without_sequences(name, host_builder, device_builder):
    cu_seqlens = torch.tensor([0], dtype=torch.int64)

    actual = device_builder(cu_seqlens, 0, 64)

    assert actual.shape == (0, 2)
    assert actual.dtype == cu_seqlens.dtype


def test_device_builder_does_not_read_back_to_the_host():
    """The point of the helper: no ``.tolist()``/``.item()`` on the device->host path."""
    import inspect

    source = inspect.getsource(prepare_chunk_indices_dev)

    assert ".tolist()" not in source
    assert ".item()" not in source
    assert ".cpu()" not in source
