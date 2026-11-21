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
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append({"kernel": self.name, **kwargs})

            return launch

    namespace = {
        "torch": torch,
        "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        "get_vectorcore_num": lambda: 48,
        "_qsa_rmsnorm_mrope_kernel": RecordedKernel("legacy"),
        "_qsa_rmsnorm_mrope_with_positions_kernel": RecordedKernel("positions"),
        "split_qkv_rmsnorm_mrope_kernel": RecordedKernel("legacy"),
        "_split_qkv_rmsnorm_mrope_with_positions_kernel": RecordedKernel("positions"),
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


def test_legacy_launch_has_no_direct_cache_metadata(wrapper):
    name, invoke = wrapper
    cache_axes = (1, 3) if name == "qsa_rmsnorm_mrope" else (3,)
    for axes in cache_axes:
        for rows in (1, 32, 1, 129, 32):
            shape = (rows, 64) if axes == 1 else (3, rows, 64)
            launch = invoke(rows, torch.empty(shape, dtype=torch.bfloat16))
            assert launch["kernel"] == "legacy"
            assert (
                not {
                    "positions_ptr",
                    "position_stride_axis",
                    "position_stride_token",
                    "cache_rows",
                    "cache_stride",
                    "GATHER_CACHE",
                }
                & launch.keys()
            )


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
    assert launch["kernel"] == "positions"
    assert launch["cache_rows"] == 257
    if "cache_stride" in launch:
        assert launch["cache_stride"] == 80
    assert launch["position_stride_token"] == 2
    assert launch["position_stride_axis"] == (rows * 2 if axes == 3 else 0)


@pytest.mark.parametrize("name", ["qsa_rmsnorm_mrope", "split_qkv_rmsnorm_mrope"])
def test_dynamic_cache_metadata_uses_runtime_parameters_without_specialization(name):
    path = Path(__file__).resolve().parents[3] / f"vllm_ascend/ops/triton/linearnorm/{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kernel_name = f"_{name}_with_positions_kernel"
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == kernel_name)
    arguments = {arg.arg: arg for arg in kernel.args.args}
    metadata = {"position_stride_axis", "position_stride_token", "cache_rows"}
    if name == "split_qkv_rmsnorm_mrope":
        metadata.add("cache_stride")
    decorator = kernel.decorator_list[0]
    assert isinstance(decorator, ast.Call)
    controls = next(keyword.value for keyword in decorator.keywords if keyword.arg == "do_not_specialize")
    assert metadata <= set(ast.literal_eval(controls))
    for parameter in metadata:
        assert arguments[parameter].annotation is None
    # Each entry fixes the mode; vector widths remain compile-time parameters.
    assert "GATHER_CACHE" not in arguments
    assert ast.unparse(arguments["rope_dim"].annotation) == "tl.constexpr"


@pytest.mark.parametrize("layout", ["contiguous", "fixed_capacity"])
def test_direct_position_axis_stride_tracks_layout_at_runtime(wrapper, layout):
    _, invoke = wrapper
    for rows in (1, 16, 32, 129, 1):
        capacity = rows if layout == "contiguous" else 2049
        positions = torch.empty(3, capacity, dtype=torch.int64)[:, :rows]
        cache = torch.empty(257, 64, dtype=torch.bfloat16)
        launch = invoke(rows, cache, positions)
        assert launch["position_stride_axis"] == capacity
        assert launch["position_stride_token"] == 1


@pytest.mark.parametrize("name", ["qsa_rmsnorm_mrope", "split_qkv_rmsnorm_mrope"])
def test_legacy_entry_keeps_original_signature_and_shared_body(name):
    path = Path(__file__).resolve().parents[3] / f"vllm_ascend/ops/triton/linearnorm/{name}.py"
    functions = {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }
    legacy_name = "_qsa_rmsnorm_mrope_kernel" if name == "qsa_rmsnorm_mrope" else "split_qkv_rmsnorm_mrope_kernel"
    legacy = functions[legacy_name]
    direct = functions[f"_{name}_with_positions_kernel"]
    extra = ["positions_ptr", "position_stride_axis", "position_stride_token"]
    if name == "split_qkv_rmsnorm_mrope":
        extra.append("cache_stride")
    extra.append("cache_rows")
    legacy_args = [arg.arg for arg in legacy.args.args]
    assert not set(extra) & set(legacy_args)
    assert [arg.arg for arg in direct.args.args] == legacy_args + extra
    # Both launch entries delegate to one body. Legacy metadata stays literal,
    # so it cannot become unused runtime arguments in the device entry point.
    for entry, gather in ((legacy, False), (direct, True)):
        assert len(entry.body) == 1
        call = entry.body[0].value
        assert call.func.id == f"_{name}"
        assert ast.literal_eval(call.args[-1]) is gather
    legacy_call = legacy.body[0].value
    assert [ast.literal_eval(arg) for arg in legacy_call.args[len(legacy_args) : -1]] == [None] + [0] * (len(extra) - 1)
