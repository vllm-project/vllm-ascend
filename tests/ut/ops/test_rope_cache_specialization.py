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
    functions: list[ast.stmt] = [
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
                launches.append({"kernel": self.name, "args": args, **kwargs})

            return launch

    namespace = {
        "torch": torch,
        "triton": SimpleNamespace(
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(), cdiv=lambda n, d: (n + d - 1) // d
        ),
        "get_vectorcore_num": lambda: 48,
        "_qsa_rmsnorm_mrope_kernel": RecordedKernel("qsa"),
        "split_qkv_rmsnorm_mrope_kernel": RecordedKernel("split"),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)

    def invoke(rows, cache, positions=None, weight=None):
        if name == "qsa_rmsnorm_mrope":
            namespace[f"triton_{name}"](
                tensor=torch.empty(rows, 4, 128, dtype=torch.bfloat16),
                weight=torch.empty(128, dtype=torch.bfloat16) if weight is None else weight,
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
        return launches[-1] if launches else None

    return name, invoke, launches


def test_legacy_launch_isolates_unused_direct_cache_metadata(wrapper):
    name, invoke, _ = wrapper
    cache_axes = (1, 3) if name == "qsa_rmsnorm_mrope" else (3,)
    for axes in cache_axes:
        for rows in (1, 32, 1, 129, 32):
            shape = (rows, 64) if axes == 1 else (3, rows, 64)
            launch = invoke(rows, torch.empty(shape, dtype=torch.bfloat16))
            if name == "qsa_rmsnorm_mrope":
                assert launch["kernel"] == "qsa"
                assert launch["GATHER_CACHE"] is False
                assert launch["positions_ptr"] is None
                for parameter in ("position_stride_axis", "position_stride_token"):
                    assert launch[parameter] == 0
                assert launch["cache_rows"] == shape[0]
                continue
            assert launch["kernel"] == "split"
            assert launch["GATHER_CACHE"] is False
            for parameter in (
                "positions_ptr",
                "position_stride_axis",
                "position_stride_token",
                "cache_rows",
                "cache_stride",
            ):
                assert launch[parameter] is None


@pytest.mark.parametrize("rows", [1, 32, 129])
@pytest.mark.parametrize("axes", [1, 3])
def test_direct_cache_keeps_actual_table_and_position_strides(wrapper, rows, axes):
    name, invoke, _ = wrapper
    cache = torch.empty(257, 80, dtype=torch.bfloat16)[:, :64]
    positions = torch.empty(3, rows * 2, dtype=torch.int64)[:, ::2]
    if axes == 1:
        positions = positions[0]
    launch = invoke(rows, cache, positions)
    assert launch["positions_ptr"] is positions
    assert launch["kernel"] == ("qsa" if name == "qsa_rmsnorm_mrope" else "split")
    assert launch["GATHER_CACHE"] is True
    assert launch["cache_rows"] == 257
    if "cache_stride" in launch:
        assert launch["cache_stride"] == 80
    assert launch["position_stride_token"] == 2
    assert launch["position_stride_axis"] == (rows * 2 if axes == 3 else 0)


@pytest.mark.parametrize("name", ["qsa_rmsnorm_mrope", "split_qkv_rmsnorm_mrope"])
def test_dynamic_cache_metadata_uses_runtime_parameters_without_specialization(name):
    path = Path(__file__).resolve().parents[3] / f"vllm_ascend/ops/triton/linearnorm/{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kernel_name = "_qsa_rmsnorm_mrope_kernel" if name == "qsa_rmsnorm_mrope" else "split_qkv_rmsnorm_mrope_kernel"
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
    for parameter in ("GATHER_CACHE", "ROPE_DIM"):
        annotation = arguments[parameter].annotation
        assert annotation is not None
        assert ast.unparse(annotation) == "tl.constexpr"
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == f"_{name}_with_positions_kernel" for node in tree.body
    )


@pytest.mark.parametrize("layout", ["contiguous", "fixed_capacity"])
def test_direct_position_axis_stride_tracks_layout_at_runtime(wrapper, layout):
    _, invoke, _ = wrapper
    for rows in (1, 16, 32, 129, 1):
        capacity = rows if layout == "contiguous" else 2049
        positions = torch.empty(3, capacity, dtype=torch.int64)[:, :rows]
        cache = torch.empty(257, 64, dtype=torch.bfloat16)
        launch = invoke(rows, cache, positions)
        assert launch["position_stride_axis"] == capacity
        assert launch["position_stride_token"] == 1


@pytest.mark.parametrize("wrapper", ["qsa_rmsnorm_mrope"], indirect=True)
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("rows", [0, 1])
def test_qsa_passes_noncontiguous_weight_stride(wrapper, direct, rows):
    _, invoke, launches = wrapper
    weight = torch.empty(256, dtype=torch.bfloat16)[::2]
    cache = torch.empty(16 if direct else rows, 64, dtype=torch.bfloat16)
    positions = torch.arange(rows) if direct else None
    launch = invoke(rows, cache, positions, weight=weight)
    if rows:
        assert launch["args"][1] is weight
        assert launch["args"][6] == 2
    else:
        assert not launches


@pytest.mark.parametrize("wrapper", ["qsa_rmsnorm_mrope"], indirect=True)
@pytest.mark.parametrize("direct", [False, True])
def test_qsa_accepts_contiguous_weight_with_storage_offset(wrapper, direct):
    _, invoke, launches = wrapper
    weight = torch.empty(129, dtype=torch.bfloat16)[1:]
    assert weight.is_contiguous() and weight.storage_offset() == 1
    cache = torch.empty(16 if direct else 1, 64, dtype=torch.bfloat16)
    positions = torch.zeros(1, dtype=torch.int64) if direct else None
    invoke(1, cache, positions, weight=weight)
    assert len(launches) == 1
