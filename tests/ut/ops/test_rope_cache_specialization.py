# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(params=["qsa_rmsnorm_mrope", "split_qkv_rmsnorm_mrope"])
def wrapper(request):
    # Execute the real host wrapper with a recorded launch, without Triton/NPU.
    name = request.param
    path = Path(__file__).resolve().parents[3] / f"vllm_ascend/ops/triton/linearnorm/{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_validate_rope_positions", f"triton_{name}")
    ]
    launches = []

    class RecordedKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(kwargs)

            return launch

    namespace = {
        "torch": torch,
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        "get_vectorcore_num": lambda: 48,
        "_qsa_rmsnorm_mrope_kernel": RecordedKernel(),
        "split_qkv_rmsnorm_mrope_kernel": RecordedKernel(),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)

    def invoke(rows, cache, positions=None):
        if name == "qsa_rmsnorm_mrope":
            namespace[f"triton_{name}"](
                tensor=torch.empty(rows, 4, 128, dtype=torch.bfloat16),
                weight=torch.empty(128, dtype=torch.bfloat16),
                cos_sin=cache,
                eps=1e-6,
                mrope_section=[11, 11, 10],
                rope_dim=64,
                positions=positions,
            )
        else:
            namespace[f"triton_{name}"](
                qkv=torch.empty(rows, 1536, dtype=torch.bfloat16),
                q_weight=torch.empty(256, dtype=torch.bfloat16),
                k_weight=torch.empty(256, dtype=torch.bfloat16),
                cos_sin=cache,
                num_q_heads=2,
                num_kv_heads=1,
                head_size=256,
                eps=1e-6,
                mrope_section=[11, 11, 10],
                is_interleaved=True,
                rope_dim=64,
                has_gate=True,
                positions=positions,
            )
        return launches[-1]

    return name, invoke


def test_legacy_direct_cache_constants_do_not_follow_tokens(wrapper):
    _, invoke = wrapper
    constants = []
    for rows in (1, 32, 1, 129, 32):
        cache = torch.empty(3, rows, 64, dtype=torch.bfloat16)
        launch = invoke(rows, cache)
        assert launch["positions_ptr"] is None
        assert launch["GATHER_CACHE"] is False
        assert launch["POSITION_STRIDE_AXIS"] == launch["POSITION_STRIDE_TOKEN"] == 0
        constants.append((launch["CACHE_ROWS"], launch.get("CACHE_STRIDE", 0)))
    assert constants == [(0, 0)] * 5


def test_legacy_qsa_rope_cache_rows_do_not_follow_tokens(wrapper):
    name, invoke = wrapper
    if name != "qsa_rmsnorm_mrope":
        pytest.skip("The shared QKV operator's legacy cache has three axes")
    for rows in (1, 32, 129):
        launch = invoke(rows, torch.empty(rows, 64, dtype=torch.bfloat16))
        assert launch["CACHE_ROWS"] == 0
        assert launch["GATHER_CACHE"] is False


@pytest.mark.parametrize("rows", [1, 32, 129])
@pytest.mark.parametrize("axes", [1, 3])
def test_direct_cache_keeps_actual_table_and_position_strides(wrapper, rows, axes):
    _, invoke = wrapper
    cache = torch.empty(257, 80, dtype=torch.bfloat16)[:, :64]
    positions = torch.empty(3, rows * 2, dtype=torch.int64)[:, ::2]
    if axes == 1:
        positions = positions[0]
    launch = invoke(rows, cache, positions)
    assert launch["positions_ptr"] is positions
    assert launch["GATHER_CACHE"] is True
    assert launch["CACHE_ROWS"] == 257
    if "CACHE_STRIDE" in launch:
        assert launch["CACHE_STRIDE"] == 80
    assert launch["POSITION_STRIDE_TOKEN"] == 2
    assert launch["POSITION_STRIDE_AXIS"] == (rows * 2 if axes == 3 else 0)
