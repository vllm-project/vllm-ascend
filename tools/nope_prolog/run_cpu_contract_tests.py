#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run source-isolated contract tests without installing vLLM or torch_npu.

This executes selected, unchanged production function bodies from the supplied
checkout. Module imports, NPU operators, and infrastructure are stubbed. It is
NOT a replacement for repository UTs, device execution, or graph capture.
Requires torch and pytest. Usage: python run_cpu_contract_tests.py /path/to/repo
"""

from __future__ import annotations

import ast
import enum
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn


def make_module(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    if "." in name:
        parent, child = name.rsplit(".", 1)
        if parent not in sys.modules:
            make_module(parent)
        setattr(sys.modules[parent], child, module)
    return module


def load_selected(repo, name, relative_path, functions=(), classes=None, constants=(), namespace=None):
    """Extract definitions, preserving their executable bodies exactly."""
    path = repo / relative_path
    source = ast.parse(path.read_text())
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in (classes or {}):
            methods = classes[node.name]
            if methods is not None:
                node.body = [
                    member
                    for member in node.body
                    if (isinstance(member, ast.FunctionDef) and member.name in methods)
                    or (isinstance(member, ast.AnnAssign) and getattr(member.target, "id", None) == "act_quant_type")
                ]
                node.decorator_list = []
                node.bases = [
                    ast.Attribute(
                        value=ast.Name(id="nn", ctx=ast.Load()),
                        attr="Module",
                        ctx=ast.Load(),
                    )
                ]
            body.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(target, "id", None) in constants for target in node.targets):
            body.append(node)
    module = make_module(name)
    module.__file__ = str(path)
    module.__dict__.update(torch=torch, nn=nn, dataclass=dataclass, enum=enum, **(namespace or {}))
    exec(  # noqa: S102 - Execute the explicitly supplied checkout's selected function bodies.
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(path),
            "exec",
        ),
        module.__dict__,
    )
    return module


def unavailable(*args, **kwargs):
    raise AssertionError("An infrastructure or NPU operation was not explicitly mocked by the test")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python run_cpu_contract_tests.py /path/to/vllm-ascend")
    repo = Path(sys.argv[1]).resolve()
    # Load the actual pure-Python hardware matrix; only device discovery is stubbed.
    for module_name, relative in [
        ("vllm_ascend.device.hardware", "vllm_ascend/device/hardware.py"),
        (
            "vllm_ascend.device.hardware_profile",
            "vllm_ascend/device/hardware_profile.py",
        ),
    ]:
        if module_name.endswith("hardware_profile"):
            device_config = make_module("vllm_ascend.device.device_config")
            family = sys.modules["vllm_ascend.device.hardware"].AscendDeviceType.A2
            device_config.get_device_config = lambda family=family: (types.SimpleNamespace(_device_type=family))
        module = make_module(module_name)
        module.__file__ = str(repo / relative)
        exec(  # noqa: S102 - Load the supplied checkout's pure-Python hardware definitions.
            compile((repo / relative).read_text(), module.__file__, "exec"),
            module.__dict__,
        )
    hardware_profile = sys.modules["vllm_ascend.device.hardware_profile"]
    # Stub only dependencies absent from this CPU-only environment.
    compilation = make_module("vllm.config.compilation")
    compilation.CUDAGraphMode = enum.Enum("CUDAGraphMode", ["NONE", "FULL", "PIECEWISE"])
    linear = make_module("vllm.model_executor.layers.linear")
    for name in ("ReplicatedLinear", "ColumnParallelLinear", "RowParallelLinear"):
        setattr(linear, name, type(name, (nn.Module,), {}))
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
    prepared = load_selected(
        repo,
        "vllm_ascend.quantization.prepared_linear_input",
        "vllm_ascend/quantization/prepared_linear_input.py",
        classes={"PreparedLinearInput": None},
    )
    npu = make_module("torch_npu")
    npu.float8_e8m0fnu = torch.float8_e8m0fnu
    npu.npu_quant_matmul = npu.npu_dynamic_quant = npu.npu_dynamic_mx_quant = unavailable
    quant_namespace = {
        "PreparedLinearInput": prepared.PreparedLinearInput,
        "torch_npu": npu,
    }
    int8 = load_selected(
        repo,
        "vllm_ascend.quantization.methods.w8a8.w8a8_dynamic",
        "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py",
        classes={"AscendW8A8DynamicLinearMethod": {"apply", "__init__"}},
        namespace=quant_namespace,
    )
    mx = load_selected(
        repo,
        "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8",
        "vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py",
        classes={"AscendW8A8MXFP8DynamicLinearMethod": {"apply"}},
        namespace=quant_namespace,
    )
    query_contract = load_selected(
        repo,
        "vllm_ascend.attention.prolog_query",
        "vllm_ascend/attention/prolog_query.py",
        functions={
            "mxfp8_query_scale_width",
            "_mxfp8_query_padding",
            "is_unquantized_query_projection",
            "prolog_query_unsupported_reasons",
            "normalize_prolog_query",
        },
        constants={"MXFP8_QUERY_GROUP_SIZE"},
    )
    utils = load_selected(
        repo,
        "vllm_ascend.attention.utils",
        "vllm_ascend/attention/utils.py",
        functions={"mark_fused_preprocess_weights", "get_sfa_qsfa_packed_head_dim"},
        classes={"PreprocessType": None},
        constants={"SFA_QSFA_TILE_SIZE"},
        namespace={"get_dtype_size": lambda dtype: torch.empty(0, dtype=dtype).element_size()},
    )
    load_selected(
        repo,
        "vllm_ascend.attention.sfa_v1",
        "vllm_ascend/attention/sfa_v1.py",
        functions={"_int64_kv_slots", "_prolog_kv_slots"},
        constants={
            "NOPE_PROLOG_QK_HEAD_DIM",
            "NOPE_PROLOG_KV_RANK",
            "NOPE_PROLOG_Q_RANKS",
        },
        classes={
            "AscendSFAImpl": {
                "forward",
                "_get_layer_quant_method",
                "_resolve_preprocess_type",
                "_fused_preprocess_type",
                "_get_fused_type_unsupported_reasons",
                "_try_enable_type",
                "_sfa_preprocess_prolog_v3",
            }
        },
        namespace={
            "CUDAGraphMode": compilation.CUDAGraphMode,
            "PreprocessType": utils.PreprocessType,
            "AscendW8A8DynamicLinearMethod": int8.AscendW8A8DynamicLinearMethod,
            "AscendW8A8MXFP8DynamicLinearMethod": mx.AscendW8A8MXFP8DynamicLinearMethod,
            "AscendW8A8LinearMethod": type("UnsupportedStaticScheme", (), {}),
            "prolog_query_unsupported_reasons": query_contract.prolog_query_unsupported_reasons,
            "normalize_prolog_query": query_contract.normalize_prolog_query,
            "is_unquantized_query_projection": query_contract.is_unquantized_query_projection,
            "HardwareCapability": hardware_profile.HardwareCapability,
            "get_current_hardware_profile": hardware_profile.get_current_hardware_profile,
            "SFA_QSFA_TILE_SIZE": 128,
            "get_sfa_qsfa_packed_head_dim": utils.get_sfa_qsfa_packed_head_dim,
            "torch_npu": npu,
            "logger": MagicMock(),
            **{
                name: unavailable
                for name in [
                    "get_forward_context",
                    "wait_for_kv_layer_from_connector",
                    "notify_kv_cache_written",
                    "maybe_save_kv_layer_to_connector",
                    "attention_transfer_window",
                ]
            },
        },
    )
    load_selected(
        repo,
        "vllm_ascend.attention.indexer_kpool",
        "vllm_ascend/attention/indexer_kpool.py",
        classes={
            "AscendIndexerKPoolMetadata": None,
            "AscendIndexerKPoolQueryMetadata": None,
            "AscendIndexerKPoolTailMetadata": None,
            "Glm5NextKPoolIndexerBackend": {
                "_project_query",
                "_bound_cache",
                "forward",
            },
        },
        namespace={
            "CUDAGraphMode": compilation.CUDAGraphMode,
            "PreparedLinearInput": prepared.PreparedLinearInput,
            "F": torch.nn.functional,
            "get_forward_context": unavailable,
            "prolog_query_unsupported_reasons": query_contract.prolog_query_unsupported_reasons,
            "mxfp8_query_scale_width": query_contract.mxfp8_query_scale_width,
        },
    )
    print(
        f"Source-isolated CPU contracts; torch={torch.__version__}; repo={repo}",
        flush=True,
    )
    print(
        "No full module imports, NPU kernels, or graph capture are validated.",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="nope-contracts-") as directory:
        test = Path(directory) / "test_nope_prolog_preparation.py"
        test.write_bytes((repo / "tests/ut/attention/test_nope_prolog_preparation.py").read_bytes())
        return pytest.main([str(test), "-q", "--noconftest", "-p", "no:cacheprovider"])


if __name__ == "__main__":
    raise SystemExit(main())
