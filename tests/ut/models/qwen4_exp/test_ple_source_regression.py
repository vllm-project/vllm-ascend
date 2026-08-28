# SPDX-License-Identifier: Apache-2.0
"""Source and CPU regressions for the Qwen4Exp PLE graph-safe path."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
NVIDIA_PLE = (
    ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "nvidia" / "ple_layer.py"
)
NVIDIA_MODEL = (
    ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "nvidia" / "model.py"
)
AMD_PLE = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "amd" / "ple_layer.py"
AMD_MODEL = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "amd" / "model.py"


def _method_node(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"method {class_name}.{method_name} not found in {path}")


def _standalone_method(method_name: str) -> Callable:
    method = copy.deepcopy(
        _method_node(NVIDIA_PLE, "Qwen4ExpPLELayer", method_name)
    )
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, NVIDIA_PLE, "exec"), namespace)
    return namespace[method_name]


def test_short_conv_cross_correlation_matches_conv1d() -> None:
    implementation = _standalone_method("_short_conv_cross_correlation")
    torch.manual_seed(20260827)
    channels = 8
    kernel_size = 4
    dilation = 3
    effective_kernel_size = dilation * (kernel_size - 1) + 1
    weights = torch.randn(channels, kernel_size)

    for history_len in (effective_kernel_size, effective_kernel_size + 4):
        history = torch.randn(2, channels, history_len)
        expected = F.conv1d(
            history,
            weights.unsqueeze(1),
            groups=channels,
            dilation=dilation,
        )
        actual = implementation(history, weights, dilation)
        torch.testing.assert_close(actual, expected)


def test_conv_state_rows_use_contiguous_storage_layout() -> None:
    gather = _standalone_method("_gather_conv_state_rows")
    scatter = _standalone_method("_scatter_conv_state_rows")
    storage = torch.arange(4 * 7 * 5, dtype=torch.float32).reshape(4, 7, 5)
    conv_state = storage.transpose(-1, -2)
    indices = torch.tensor([3, 1])
    assert conv_state.stride() == (35, 1, 5)
    assert conv_state.transpose(-1, -2).is_contiguous()

    selected = gather(conv_state, indices)
    torch.testing.assert_close(selected, conv_state.index_select(0, indices))

    updated = selected + 1000
    scatter(conv_state, indices, updated)
    torch.testing.assert_close(conv_state.index_select(0, indices), updated)


def test_ple_state_shape_reserves_speculative_history() -> None:
    for path in (NVIDIA_MODEL, AMD_MODEL):
        source = ast.unparse(
            _method_node(
                path,
                "Qwen4ExpForCausalLM",
                "get_ple_mamba_state_shape_from_config",
            )
        )
        assert "vllm_config.num_speculative_tokens" in source

    for path in (NVIDIA_PLE, AMD_PLE):
        source = ast.unparse(
            _method_node(path, "Qwen4ExpPLELayer", "get_state_shape")
        )
        assert "self.num_spec_tokens" in source


def test_nvidia_model_does_not_compile_query_start_loc_as_dynamic() -> None:
    tree = ast.parse(NVIDIA_MODEL.read_text())
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen4ExpModel"
    )
    assignment = next(
        node
        for node in model.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "dynamic_arg_dims"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Dict)
    keys = {
        key.value
        for key in assignment.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert "query_start_loc" not in keys
