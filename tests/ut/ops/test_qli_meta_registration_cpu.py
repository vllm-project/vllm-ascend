# SPDX-License-Identifier: Apache-2.0
"""Compile the actual QLI Meta functions without CANN or NPU execution.

Only aclrtGetSocName is stubbed. The schema and complete Meta function bodies
are extracted from production sources, so missing arguments fail registration.
Requires a C++ compiler, ninja and PyTorch CPU extension headers.
"""

import ctypes
import json
import re
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[3]


def _function(source, name):
    start = source.index("std::tuple<at::Tensor, at::Tensor> " + name + "(")
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def meta_extension(tmp_path_factory):
    binding = (ROOT / "csrc/torch_binding.cpp").read_text(encoding="utf-8")
    source = (ROOT / "csrc/torch_binding_meta.cpp").read_text(encoding="utf-8")
    schema = None
    for match in re.finditer(r'ops\.def\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)\)', binding):
        value = "".join(json.loads(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"', match[1]))
        if value.startswith("npu_quant_lightning_indexer_v2("):
            schema = value
            break
    assert schema is not None
    cpp = """
#include <ATen/ATen.h>
#include <torch/library.h>
#include <string>
static int test_soc = 0;
extern "C" void set_test_soc(int value) { test_soc = value; }
const char* aclrtGetSocName() {
    if (test_soc == 0) return "Ascend910_9382";
    if (test_soc == 1) return "Ascend950PR_9599";
    return nullptr;
}
"""
    cpp += _function(source, "construct_quant_lightning_indexer_v2_output_tensor")
    cpp += _function(source, "npu_quant_lightning_indexer_v2_meta")
    cpp += "\nTORCH_LIBRARY(qli_meta_contract_test, m) {\n"
    cpp += f'm.def(R"SCHEMA({schema})SCHEMA");\n'
    cpp += 'm.impl("npu_quant_lightning_indexer_v2", c10::kMeta, TORCH_FN(npu_quant_lightning_indexer_v2_meta));\n}\n'
    build = tmp_path_factory.mktemp("qli_meta_cpp")
    path = build / "qli_meta.cpp"
    path.write_text(cpp, encoding="utf-8")
    library = load(
        name="qli_meta_contract_test",
        sources=[str(path)],
        build_directory=str(build),
        extra_cflags=["-O0"],
        is_python_module=False,
        verbose=False,
    )
    handle = ctypes.CDLL(str(library))
    handle.set_test_soc.argtypes = [ctypes.c_int]
    return handle


@pytest.mark.parametrize("soc", [0, 1, 2], ids=["A3", "A5", "unknown_matches_runtime"])
@pytest.mark.parametrize("return_value", [0, 1])
@pytest.mark.parametrize("layout", ["TND", "BSND"])
def test_qli_meta_registration_shape_and_dtype(meta_extension, soc, return_value, layout):
    meta_extension.set_test_soc(soc)
    shape = (3, 64, 128) if layout == "TND" else (2, 3, 64, 128)
    # Use INT8 for both devices: device family, not Q dtype, selects score dtype.
    q = torch.empty(shape, dtype=torch.int8, device="meta")
    k = torch.empty((4, 768, 1, 128), dtype=torch.int8, device="meta")
    scale = torch.empty((3, 64), dtype=torch.float16, device="meta")
    indices, scores = torch.ops.qli_meta_contract_test.npu_quant_lightning_indexer_v2(
        q,
        k,
        scale,
        scale,
        scale,
        2048,
        2,
        layout_q=layout,
        layout_k="PA_BBND",
        return_value=return_value,
        enable_dcp=True,
        cp_world=16,
        cp_rank=7,
        cp_kv_cache_interleave_size=768,
        causal_seqlens_kv_global=torch.empty(2, dtype=torch.int32, device="meta"),
    )
    expected_shape = (3, 1, 2048) if layout == "TND" else (2, 3, 1, 2048)
    assert tuple(indices.shape) == expected_shape
    assert indices.dtype == torch.int32
    assert tuple(scores.shape) == (expected_shape if return_value else (0,))
    assert scores.dtype == (torch.float32 if return_value and soc != 1 else torch.bfloat16)
    assert scores.device.type == "meta"
