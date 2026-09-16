# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 INT8 NoPE512: compact input, independent accuracy, and RoPE64 regression.

The A5 kernel currently writes attention only. These tests retain that contract
and do not claim support for the A2/A3 softmax-max/sum output feature.
"""

import importlib

import pytest
import torch

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

from .kv_quant_sparse_flash_attention_test_utils import (
    RopeCase,
    _assert_outputs_equal,
    _check_outputs,
    _check_uniform,
    _make_cpu_case,
    _pad_zero_rope,
    _to_npu,
    _uniform_reference,
)
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


CASES = (
    RopeCase("bsnd_page_tail", "BSND", "PA_BSND", 4, 1, 640, (1, 3), (249, 761)),
    RopeCase("tnd_page_tail", "TND", "PA_BSND", 48, 1, 640, (2, 3), (505, 761)),
    RopeCase("tnd_packed_tail", "TND", "TND", 2, 1, 640, (2, 3), (249, 505)),
    RopeCase("split_heads", "BSND", "PA_BSND", 128, 1, 640, (1, 2), (249, 761)),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@torch.inference_mode()
def test_a5_compact_matches_zero_rope(case, dtype):
    cpu = _make_cpu_case(case, dtype, 0)
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
@pytest.mark.parametrize("rope_dim", (0, 64))
@torch.inference_mode()
def test_a5_independent_selected_value_mean(case, dtype, rope_dim):
    cpu = _make_cpu_case(case, dtype, rope_dim, uniform=True)
    expected, counts = _uniform_reference(cpu, case)
    inputs = _to_npu(cpu)
    outputs = _run_custom_op(inputs)
    _check_outputs(outputs, inputs, case, False)
    _check_uniform(outputs, expected, counts, False)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("rope_dim", (0, 64))
@torch.inference_mode()
def test_a5_graph_replay_updates_kv(dtype, rope_dim):
    case = CASES[1]
    inputs = _to_npu(_make_cpu_case(case, dtype, rope_dim, uniform=True))
    _run_custom_op(inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        actual = _run_custom_op(inputs)
    pointers = tuple(tensor.data_ptr() for tensor in actual)
    for sign in (1, -1):
        cpu = _make_cpu_case(case, dtype, rope_dim, uniform=True, value_sign=sign)
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
