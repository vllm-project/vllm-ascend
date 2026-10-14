# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression for the measured nine-argument Ascend launcher contract.

The binary launcher is emulated; these checks do not establish Triton/NPU ABI
compatibility. Run with --noconftest without importing the device plugins.
"""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


class RuntimeOnlyLauncher:
    """Represent a compiled launcher whose constexpr values are already bound."""

    def __init__(self, to_cache):
        self.to_cache = to_cache
        self.calls = []

    def __getitem__(self, grid):
        def launch(state, packed, indices, flags, rows, page_stride, payload, packed_stride, use_flags):
            self.calls.append(grid)
            assert rows == state.shape[0]
            assert page_stride == state.stride(0)
            assert payload == state[0].numel()
            assert packed_stride == 1
            assert use_flags == (0 if self.to_cache else 1)
            for row, index in enumerate(indices.tolist()):
                valid = 0 <= index < rows
                if self.to_cache:
                    assert flags is indices
                    if valid:
                        state[index].copy_(packed[row])
                elif valid and bool(flags[row]):
                    packed[row].copy_(state[index])
                else:
                    packed[row].zero_()

        return launch


@pytest.fixture
def launch_method():
    path = ROOT / "vllm_ascend/ops/kda_state_copy_plan.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "KDAStateCopyPlan")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_launch")
    scope = {"nullcontext": nullcontext, "torch": SimpleNamespace(npu=SimpleNamespace(current_device=lambda: None))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope["_launch"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("to_cache", [False, True])
def test_direct_launch_consumes_only_runtime_arguments(launch_method, dtype, index_dtype, to_cache):
    backing = torch.full((4 * 20,), -7, dtype=dtype)
    state = backing.as_strided((4, 2, 2, 4), (20, 8, 4, 1))
    state.copy_(torch.arange(state.numel(), dtype=dtype).reshape(state.shape))
    before = backing.clone()
    indices = torch.tensor([2, -1, 0, 9], dtype=index_dtype)
    flags = torch.tensor([True, True, False, True])
    packed = torch.full((4, *state.shape[1:]), 31, dtype=dtype)
    launcher = RuntimeOnlyLauncher(to_cache)
    plan = SimpleNamespace(_compiled={(index_dtype, to_cache): launcher}, _tiles=1, _scalars=(4, 20, 16, 1))
    MethodType(launch_method, plan)(state, packed, indices, flags, to_cache=to_cache)
    assert launcher.calls == [(4, 1, 1)]
    if to_cache:
        torch.testing.assert_close(state[2], packed[0])
        torch.testing.assert_close(state[0], packed[2])
        torch.testing.assert_close(state[1], before.as_strided(state.shape, state.stride())[1])
        torch.testing.assert_close(state[3], before.as_strided(state.shape, state.stride())[3])
        torch.testing.assert_close(backing.reshape(4, 20)[:, 16:], before.reshape(4, 20)[:, 16:])
    else:
        torch.testing.assert_close(packed[0], state[2])
        assert not torch.count_nonzero(packed[1:])
        torch.testing.assert_close(backing, before)


def test_empty_direct_launch_does_not_access_compiled_kernel(launch_method):
    plan = SimpleNamespace()
    MethodType(launch_method, plan)(None, None, torch.empty(0, dtype=torch.int64), None, to_cache=False)


def test_direct_launch_rejects_unaligned_metadata(launch_method):
    state = torch.empty(1, 2, 2, 4)
    packed = torch.empty_like(state)
    indices = torch.zeros(2, dtype=torch.int64)[1:]
    flags = torch.ones(1, dtype=torch.bool)
    plan = SimpleNamespace()
    with pytest.raises(RuntimeError, match="unaligned normalized KDA buffer"):
        MethodType(launch_method, plan)(state, packed, indices, flags, to_cache=False)
