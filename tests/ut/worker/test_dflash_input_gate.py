# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Check lazy DFlash adapters against the production kernel's signature."""

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import create_autospec

import pytest
import torch


@pytest.mark.parametrize("release", [False, True])
@pytest.mark.parametrize("block_stride", [384, 1536])
@pytest.mark.parametrize("num_reqs", [None, 5])
def test_dflash_adapter_defers_import_and_preserves_manager_slots(release, block_stride, num_reqs):
    root = Path(__file__).resolve().parents[3]
    kernel_source = root / "vllm_ascend/ops/triton/spec_decode/prepare_dflash_inputs.py"
    kernel_def = next(
        node
        for node in ast.parse(kernel_source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_dflash_inputs_triton"
    )
    # Use the real launcher signature without importing Triton or initializing NPU.
    kernel_def.body = [ast.Pass()]
    ast.fix_missing_locations(kernel_def)
    namespace = {"torch": torch, "InputBuffers": object, "InputBatch": object}
    exec(compile(ast.Module(body=[kernel_def], type_ignores=[]), str(kernel_source), "exec"), namespace)
    kernel = create_autospec(namespace["prepare_dflash_inputs_triton"])
    imports = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("vllm_ascend.ops.triton"):
            imports.append(name)
            return SimpleNamespace(prepare_dflash_inputs_triton=kernel)
        return original_import(name, *args, **kwargs)

    source = root / "vllm_ascend/worker/v2/spec_decode/dflash/speculator.py"
    names = {
        "install_ascend_dflash_input_preparation",
        "_prepare_dflash_inputs_with_block_stride",
        "prepare_dflash_inputs",
    }
    helpers: list[ast.stmt] = [
        node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    upstream = SimpleNamespace()
    namespace.update(
        Any=Any,
        vllm_version_is=lambda version: release and version == "0.34.0",
        dflash_speculator=upstream,
        __builtins__={**vars(builtins), "__import__": guarded_import},
    )
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(source), "exec"), namespace)
    namespace["install_ascend_dflash_input_preparation"]()
    assert imports == []
    inputs = [object() for _ in range(28)]
    inputs[17], inputs[-1] = 384, True
    native = inputs if release else [*inputs[:18], block_stride, *inputs[18:]]
    upstream.prepare_dflash_inputs(*native, num_reqs_override=num_reqs)
    kernel.assert_called_once_with(*inputs, num_reqs_override=num_reqs, block_stride=None if release else block_stride)
    assert imports == ["vllm_ascend.ops.triton.spec_decode.prepare_dflash_inputs"]
