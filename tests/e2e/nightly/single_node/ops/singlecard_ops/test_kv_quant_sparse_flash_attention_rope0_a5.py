# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 INT8/FP8 NoPE512: compact input, accuracy, and RoPE64 regression.

The A5 kernel currently writes attention only. These tests retain that contract
and do not claim support for the A2/A3 softmax-max/sum output feature.
"""

import importlib
import math
from dataclasses import dataclass
from itertools import accumulate

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

from .test_kv_quant_sparse_flash_attention import (
    BF16_ATOL,
    BF16_RTOL,
    _make_inputs,
    _reference_attention,
    _run_custom_op,
)

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Requires the A5 KvQuantSparseFlashAttention kernel.",
)


@pytest.fixture(scope="module", autouse=True)
def _load_a5_operator_extension():
    # A5 disables generic runtime custom ops; this is an explicit operator test.
    torch.npu.set_device("npu:0")
    importlib.import_module("vllm_ascend.vllm_ascend_C")


NOPE_DIM = 512
ROPE_DIM = 64
QUANT_TILE_SIZE = 128
PAGE_SIZE = 128
SCALE_GROUPS = NOPE_DIM // QUANT_TILE_SIZE
FP16_ATOL = 2.5e-5
FP16_RTOL = 5e-3
SCALE_VALUE = (NOPE_DIM + ROPE_DIM) ** -0.5


@dataclass(frozen=True)
class RopeCase:
    name: str
    query_layout: str
    kv_layout: str
    heads: int
    sparse_block_size: int
    max_tokens: int
    query_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]


def _stored_query(tensor, case):
    if case.query_layout == "TND":
        return torch.cat([tensor[b, :length] for b, length in enumerate(case.query_lengths)])
    return tensor


def _make_cpu_case(case, dtype, rope_dim, *, kv_dtype=torch.int8, uniform=False, value_sign=1):
    generator = torch.Generator().manual_seed(20260915)
    batch = len(case.query_lengths)
    query_seq = max(case.query_lengths)
    kv_seq = max(case.kv_lengths)
    query_shape = (batch, query_seq, case.heads, NOPE_DIM)
    key_shape = (batch, kv_seq, 1, NOPE_DIM)
    if uniform:
        query = torch.zeros(query_shape, dtype=dtype)
        query[..., 0] = (torch.arange(math.prod(query_shape[:-1])).reshape(query_shape[:-1]) % 7 + 1).to(dtype)
        key_values = (torch.arange(math.prod(key_shape)).reshape(key_shape) % 31 + 8) * value_sign
        key_values[..., 0] = 0
        key = key_values.to(kv_dtype)
        exponent = (torch.arange(batch * kv_seq)[:, None] + torch.arange(SCALE_GROUPS)[None, :]) % 4 - 3
        scales = torch.pow(2.0, exponent).reshape(batch, kv_seq, 1, SCALE_GROUPS)
    else:
        query = (torch.rand(query_shape, generator=generator) * 2 - 1).to(dtype)
        key = torch.randint(-64, 64, key_shape, generator=generator, dtype=torch.int8).to(kv_dtype)
        scales = torch.rand((batch, kv_seq, 1, SCALE_GROUPS), generator=generator) * 0.095 + 0.005

    query_rope = torch.zeros((*query_shape[:-1], rope_dim), dtype=dtype)
    key_rope = torch.zeros((*key_shape[:-1], rope_dim), dtype=dtype)
    packed_bytes = torch.cat(
        (
            key.contiguous().view(torch.uint8),
            key_rope.contiguous().view(torch.uint8),
            scales.contiguous().view(torch.uint8),
        ),
        dim=-1,
    )
    packed = packed_bytes.view(kv_dtype)
    index_width = case.max_tokens // case.sparse_block_size
    sparse = torch.full((batch, query_seq, 1, index_width), -1, dtype=torch.int32)
    for batch_index, query_length in enumerate(case.query_lengths):
        for query_index in range(query_length):
            threshold = case.kv_lengths[batch_index] - query_length + query_index + 1
            available = math.ceil(threshold / case.sparse_block_size)
            count = min(index_width, available)
            # Always include the causal boundary block, including partially valid blocks.
            sparse[batch_index, query_index, 0, : count - 1] = torch.randperm(
                available - 1, generator=generator, dtype=torch.int32
            )[: count - 1]
            sparse[batch_index, query_index, 0, count - 1] = available - 1

    block_table = None
    if case.kv_layout == "PA_BSND":
        pages_per_sequence = tuple(math.ceil(length / PAGE_SIZE) for length in case.kv_lengths)
        page_count = sum(pages_per_sequence)
        physical_pages = torch.randperm(page_count, generator=generator, dtype=torch.int32)
        block_table = torch.full((batch, math.ceil(kv_seq / PAGE_SIZE)), -1, dtype=torch.int32)
        packed_storage = torch.zeros((page_count, PAGE_SIZE, 1, packed.shape[-1]), dtype=kv_dtype)
        value_storage = torch.zeros((page_count, PAGE_SIZE, 1, NOPE_DIM), dtype=kv_dtype)
        page_index = 0
        for batch_index, pages in enumerate(pages_per_sequence):
            for logical_page in range(pages):
                physical_page = int(physical_pages[page_index])
                block_table[batch_index, logical_page] = physical_page
                begin = logical_page * PAGE_SIZE
                end = min(begin + PAGE_SIZE, case.kv_lengths[batch_index])
                packed_storage[physical_page, : end - begin] = packed[batch_index, begin:end]
                value_storage[physical_page, : end - begin] = key[batch_index, begin:end]
                page_index += 1
    else:
        packed_storage = torch.cat([packed[b, :length] for b, length in enumerate(case.kv_lengths)])
        value_storage = torch.cat([key[b, :length] for b, length in enumerate(case.kv_lengths)])

    query_ends = tuple(accumulate(case.query_lengths)) if case.query_layout == "TND" else case.query_lengths
    kv_ends = tuple(accumulate(case.kv_lengths)) if case.kv_layout == "TND" else case.kv_lengths
    return {
        "query": _stored_query(torch.cat((query, query_rope), dim=-1), case),
        "key": packed_storage,
        "value": value_storage,
        "sparse_indices": _stored_query(sparse, case),
        "block_table": block_table,
        "actual_seq_lengths_query": torch.tensor(query_ends, dtype=torch.int32),
        "actual_seq_lengths_kv": torch.tensor(kv_ends, dtype=torch.int32),
        "scale_value": SCALE_VALUE,
        "sparse_block_size": case.sparse_block_size,
        "layout_query": case.query_layout,
        "layout_kv": case.kv_layout,
        "sparse_mode": 3,
        "attention_mode": 2,
        "quant_scale_repo_mode": 1,
        "tile_size": QUANT_TILE_SIZE,
        "rope_head_dim": rope_dim,
        "key_quant_mode": 2,
        "value_quant_mode": 2,
        "cpu": {"query": query, "key": key, "scales": scales, "sparse_indices": sparse},
    }


def _to_npu(inputs):
    return {name: value.npu() if isinstance(value, torch.Tensor) else value for name, value in inputs.items()}


def _pad_zero_rope(inputs):
    padded = dict(inputs)
    query = inputs["query"]
    key = inputs["key"]
    padded["query"] = torch.cat((query, query.new_zeros((*query.shape[:-1], ROPE_DIM))), dim=-1)
    padded["key"] = torch.cat(
        (key[..., :NOPE_DIM], key.new_zeros((*key.shape[:-1], ROPE_DIM * 2)), key[..., NOPE_DIM:]), dim=-1
    )
    padded["rope_head_dim"] = ROPE_DIM
    assert torch.equal(
        padded["key"][..., NOPE_DIM + ROPE_DIM * 2 :].view(torch.uint8),
        key[..., NOPE_DIM:].view(torch.uint8),
    )
    return padded


def _uniform_reference(inputs, case):
    """Uniform attention is the selected V mean; LSE is log(valid token count)."""
    cpu = inputs["cpu"]
    query = cpu["query"]
    assert torch.all(query[..., 0] != 0)
    assert torch.count_nonzero(query[..., 1:]) == 0
    assert torch.count_nonzero(cpu["key"][..., 0].float()) == 0
    values = cpu["key"].double() * cpu["scales"].double().repeat_interleave(QUANT_TILE_SIZE, dim=-1)
    assert torch.equal(values, values.to(query.dtype).double())
    expected = torch.zeros_like(query, dtype=torch.float64)
    counts = torch.zeros(query.shape[:-1], dtype=torch.float32)
    for batch_index, query_length in enumerate(case.query_lengths):
        for query_index in range(query_length):
            threshold = case.kv_lengths[batch_index] - query_length + query_index + 1
            token_ids: list[int] = []
            for block_id in cpu["sparse_indices"][batch_index, query_index, 0].tolist():
                if block_id < 0:
                    break
                begin = block_id * case.sparse_block_size
                token_ids.extend(range(begin, min(begin + case.sparse_block_size, threshold)))
            expected[batch_index, query_index] = values[batch_index, token_ids, 0].sum(dim=0) / len(token_ids)
            counts[batch_index, query_index] = len(token_ids)
    expected = _stored_query(expected.to(query.dtype), case)
    counts = _stored_query(counts, case)
    # N2=1: BSND LSE is [B,1,S,N], TND LSE is [1,T,N].
    counts = counts.unsqueeze(0) if case.query_layout == "TND" else counts.unsqueeze(1)
    return expected, counts


def _check_outputs(outputs, inputs, case, return_lse):
    attention, maximum, denominator = outputs
    output_shape = (*inputs["query"].shape[:-1], NOPE_DIM)
    assert attention.shape == output_shape
    assert attention.dtype == inputs["query"].dtype
    assert torch.isfinite(attention).all()
    assert maximum.dtype == denominator.dtype == torch.float32
    if not return_lse:
        assert maximum.numel() == denominator.numel() == 0
        return
    lse_shape: tuple[int, ...]
    if case.query_layout == "TND":
        lse_shape = (1, sum(case.query_lengths), case.heads)
    else:
        lse_shape = (len(case.query_lengths), 1, max(case.query_lengths), case.heads)
    assert maximum.shape == denominator.shape == lse_shape
    valid = _valid_lse_rows(case).to(maximum.device)
    assert torch.isfinite(maximum[valid]).all()
    assert torch.isfinite(denominator[valid]).all()


def _valid_lse_rows(case):
    if case.query_layout == "TND":
        return torch.ones((1, sum(case.query_lengths), case.heads), dtype=torch.bool)
    valid = torch.zeros((len(case.query_lengths), 1, max(case.query_lengths), case.heads), dtype=torch.bool)
    for batch_index, length in enumerate(case.query_lengths):
        valid[batch_index, 0, :length] = True
    return valid


def _assert_outputs_equal(actual, expected, case, return_lse):
    assert torch.equal(actual[0], expected[0])
    if return_lse:
        valid = _valid_lse_rows(case)
        for observed, control in zip(actual[1:], expected[1:]):
            # Padding query rows do not have a specified LSE value.
            assert torch.equal(observed.cpu()[valid], control.cpu()[valid])
    else:
        assert actual[1].numel() == actual[2].numel() == expected[1].numel() == expected[2].numel() == 0


def _check_uniform(outputs, expected, counts, return_lse):
    attention, maximum, denominator = (tensor.cpu() for tensor in outputs)
    atol, rtol = (BF16_ATOL, BF16_RTOL) if attention.dtype == torch.bfloat16 else (FP16_ATOL, FP16_RTOL)
    torch.testing.assert_close(attention.float(), expected.float(), atol=atol, rtol=rtol)
    if return_lse:
        valid = counts > 0
        # Only valid rows have a specified LSE. The maximum is zero and exp-sum is an integer.
        torch.testing.assert_close(maximum[valid], torch.zeros_like(maximum[valid]), atol=0, rtol=0)
        torch.testing.assert_close(denominator[valid], counts[valid], atol=0, rtol=0)
        lse = maximum[valid] + denominator[valid].log()
        torch.testing.assert_close(lse, counts[valid].log(), atol=0, rtol=0)


CASES = (
    RopeCase("bsnd_page_tail", "BSND", "PA_BSND", 4, 1, 640, (1, 3), (249, 761)),
    RopeCase("tnd_page_tail", "TND", "PA_BSND", 48, 1, 640, (2, 3), (505, 761)),
    RopeCase("tnd_packed_tail", "TND", "TND", 2, 1, 640, (2, 3), (249, 505)),
    RopeCase("split_heads", "BSND", "PA_BSND", 128, 1, 640, (1, 2), (249, 761)),
)
KV_DTYPES = (torch.int8, torch.float8_e4m3fn)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("kv_dtype", KV_DTYPES, ids=("int8", "float8_e4m3fn"))
@torch.inference_mode()
def test_a5_compact_matches_zero_rope(case, dtype, kv_dtype):
    cpu = _make_cpu_case(case, dtype, 0, kv_dtype=kv_dtype)
    compact = _to_npu(cpu)
    padded = _to_npu(_pad_zero_rope(cpu))
    actual = _run_custom_op(compact)
    expected = _run_custom_op(padded)
    repeated = _run_custom_op(compact)
    for result in (actual, expected, repeated):
        _check_outputs(result, compact, case, False)
    _assert_outputs_equal(actual, expected, case, False)
    _assert_outputs_equal(actual, repeated, case, False)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("kv_dtype", KV_DTYPES, ids=("int8", "float8_e4m3fn"))
@pytest.mark.parametrize("rope_dim", (0, 64))
@torch.inference_mode()
def test_a5_independent_selected_value_mean(case, dtype, kv_dtype, rope_dim):
    cpu = _make_cpu_case(case, dtype, rope_dim, kv_dtype=kv_dtype, uniform=True)
    expected, counts = _uniform_reference(cpu, case)
    inputs = _to_npu(cpu)
    outputs = _run_custom_op(inputs)
    _check_outputs(outputs, inputs, case, False)
    _check_uniform(outputs, expected, counts, False)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("kv_dtype", KV_DTYPES, ids=("int8", "float8_e4m3fn"))
@pytest.mark.parametrize("rope_dim", (0, 64))
@torch.inference_mode()
def test_a5_graph_replay_updates_kv(dtype, kv_dtype, rope_dim):
    case = CASES[1]
    inputs = _to_npu(_make_cpu_case(case, dtype, rope_dim, kv_dtype=kv_dtype, uniform=True))
    _run_custom_op(inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = _run_custom_op(inputs)
    pointers = tuple(tensor.data_ptr() for tensor in actual)
    for sign in (1, -1):
        cpu = _make_cpu_case(case, dtype, rope_dim, kv_dtype=kv_dtype, uniform=True, value_sign=sign)
        expected, counts = _uniform_reference(cpu, case)
        replacement = _to_npu(cpu)
        inputs["key"].copy_(replacement["key"])
        inputs["value"].copy_(replacement["value"])
        graph.replay()
        torch.npu.synchronize()
        assert tuple(tensor.data_ptr() for tensor in actual) == pointers
        _check_uniform(actual, expected, counts, False)
        _assert_outputs_equal(actual, _run_custom_op(inputs), case, False)


@pytest.mark.parametrize(
    "rope_dim,query_dim,key_dim",
    ((-1, 512, 528), (32, 544, 592), (128, 640, 784), (0, 576, 528), (0, 512, 656), (64, 512, 656), (64, 576, 528)),
)
@torch.inference_mode()
def test_a5_invalid_shape(rope_dim, query_dim, key_dim):
    inputs = _to_npu(_make_cpu_case(CASES[0], torch.float16, 0))
    _run_custom_op(inputs)
    inputs["query"] = inputs["query"].new_zeros((*inputs["query"].shape[:-1], query_dim))
    inputs["key"] = inputs["key"].new_zeros((*inputs["key"].shape[:-1], key_dim))
    inputs["rope_head_dim"] = rope_dim
    with pytest.raises(RuntimeError):
        _run_custom_op(inputs)


@pytest.mark.parametrize("field,value", (("tile_size", 64), ("key_quant_mode", 1), ("quant_scale_repo_mode", 0)))
@torch.inference_mode()
def test_a5_quantization_contract_unchanged(field, value):
    inputs = _to_npu(_make_cpu_case(CASES[0], torch.float16, 0))
    _run_custom_op(inputs)
    inputs[field] = value
    with pytest.raises(RuntimeError):
        _run_custom_op(inputs)


@torch.inference_mode()
def test_a5_original_nonzero_rope_accuracy():
    inputs = _make_inputs()
    reference = _reference_attention(inputs)
    output, maximum, denominator = _run_custom_op(inputs)
    assert maximum.numel() == denominator.numel() == 0
    torch.testing.assert_close(output.cpu().float(), reference, atol=BF16_ATOL, rtol=BF16_RTOL)
