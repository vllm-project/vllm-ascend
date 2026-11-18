# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A2/A3 融合 GMM1+SiTU+量化入口的 CPU 路由测试。

用 AST 把 ``AscendW4A8DynamicFusedMoEMethod._fused_gmm1_situ_quant`` 抠出来，放进只带
``torch.ops`` 和 NPU 存储格式查询桩的 scope 里执行，断言 aclnn 入口和输入形态。
不校验设备侧数学，也不需要 NPU / torch_npu：

    pytest tests/ut/quantization/methods/w4a8/test_w4a8_situ_route.py \
        --confcutdir=tests/ut/quantization/methods/w4a8
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[5]
W4A8 = "vllm_ascend/quantization/methods/w4a8/w4a8.py"
CLASS = "AscendW4A8DynamicFusedMoEMethod"
METHOD = "_fused_gmm1_situ_quant"

ND_ENTRY = "grouped_matmul_situ_quant"
NZ_ENTRY = "grouped_matmul_situ_quant_weight_nz"
ND_FORMAT = 2
NZ_FORMAT = 29
NZ_C0_16_FORMAT = 50


def _load_method(fake_torch, name=METHOD, **scope_vars):
    tree = ast.parse((ROOT / W4A8).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.decorator_list = []
    scope = {
        "torch": fake_torch,
        "torch_npu": SimpleNamespace(get_npu_format=lambda weight: getattr(weight, "npu_format", ND_FORMAT)),
        "ACL_FORMAT_FRACTAL_NZ": NZ_FORMAT,
        "ACL_FORMAT_FRACTAL_NZ_C0_16": NZ_C0_16_FORMAT,
        **scope_vars,
    }
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), W4A8, "exec"), scope)
    return scope[name]


@pytest.fixture
def entry():
    def make(name):
        plain = Mock(name=name)
        plain.list = Mock(name=f"{name}.list")
        return plain

    nd_op, nz = make(ND_ENTRY), make(NZ_ENTRY)
    fake_torch = SimpleNamespace(ops=SimpleNamespace(_C_ascend=SimpleNamespace(**{ND_ENTRY: nd_op, NZ_ENTRY: nz})))
    method = _load_method(fake_torch)
    return SimpleNamespace(call=method, nd_op=nd_op, nz=nz)


def _invoke(method, weight, weight_scale, weight_assist_matrix):
    return method(
        None,
        x=torch.zeros(4, 32, dtype=torch.int8),
        weight=weight,
        weight_scale=weight_scale,
        weight_assist_matrix=weight_assist_matrix,
        x_scale=torch.ones(4),
        group_list=torch.tensor([4], dtype=torch.int64),
        group_list_type=0,
        beta=2.0,
        linear_beta=0.5,
    )


def test_stacked_nd_uses_plain_entry(entry):
    weight = torch.zeros(2, 32, 8, dtype=torch.int32)  # [E, K, N/8]
    scale = torch.zeros(2, 64)
    assist = torch.zeros(2, 64)

    _invoke(entry.call, weight, scale, assist)

    entry.nd_op.assert_called_once()
    entry.nd_op.list.assert_not_called()
    entry.nz.assert_not_called()
    kwargs = entry.nd_op.call_args.kwargs
    assert kwargs["weight"] is weight
    assert kwargs["weight_scale"] is scale
    assert kwargs["weight_assist_matrix"] is assist
    assert kwargs["bias"] is None
    assert kwargs["smooth_scale"] is None
    assert kwargs["dequant_mode"] == 0
    assert kwargs["dequant_dtype"] == 0
    assert kwargs["quant_mode"] == 0
    assert kwargs["beta"] == 2.0
    assert kwargs["linear_beta"] == 0.5


@pytest.mark.parametrize("npu_format", [NZ_FORMAT, NZ_C0_16_FORMAT])
def test_stacked_nz_uses_plain_nz_entry(entry, npu_format):
    weight = torch.zeros(2, 32, 8, dtype=torch.int32)  # NZ retains the logical [E, K, N/8] shape.
    weight.npu_format = npu_format
    scale = torch.zeros(2, 64)
    assist = torch.zeros(2, 64)

    _invoke(entry.call, weight, scale, assist)

    entry.nz.assert_called_once()
    entry.nz.list.assert_not_called()
    entry.nd_op.assert_not_called()
    assert entry.nz.call_args.kwargs["weight"] is weight


def test_per_expert_nd_keeps_lists(entry):
    weight = [torch.zeros(32, 8, dtype=torch.int32), torch.zeros(32, 8, dtype=torch.int32)]
    scale = [torch.zeros(64), torch.zeros(64)]
    assist = [torch.zeros(64), torch.zeros(64)]

    _invoke(entry.call, weight, scale, assist)

    entry.nd_op.list.assert_called_once()
    entry.nd_op.assert_not_called()
    kwargs = entry.nd_op.list.call_args.kwargs
    assert kwargs["weight"] == weight
    assert kwargs["weight"][0] is weight[0]
    assert kwargs["weight"][0].dim() == 2  # 多专家形态原样透传，不补 E 维
    assert kwargs["weight_scale"] == scale
    assert kwargs["weight_assist_matrix"] == assist


def test_single_expert_nd_gets_expert_dim(entry):
    # len(list) == 1 时 aclnn 按单张量形态处理，三份输入都必须带 E 维。
    weight = [torch.zeros(32, 8, dtype=torch.int32)]
    scale = [torch.zeros(64)]
    assist = [torch.zeros(64)]

    _invoke(entry.call, weight, scale, assist)

    entry.nd_op.list.assert_called_once()
    kwargs = entry.nd_op.list.call_args.kwargs
    assert [w.dim() for w in kwargs["weight"]] == [3]
    assert [s.dim() for s in kwargs["weight_scale"]] == [2]
    assert [a.dim() for a in kwargs["weight_assist_matrix"]] == [2]
    assert kwargs["weight"][0].shape == (1, 32, 8)
    assert kwargs["weight"][0] is not weight[0]


def test_single_expert_nz_gets_expert_dim(entry):
    weight = [torch.zeros(32, 8, dtype=torch.int32)]
    weight[0].npu_format = NZ_FORMAT
    scale = [torch.zeros(64)]
    assist = [torch.zeros(64)]

    _invoke(entry.call, weight, scale, assist)

    entry.nz.list.assert_called_once()
    entry.nd_op.list.assert_not_called()
    kwargs = entry.nz.list.call_args.kwargs
    assert kwargs["weight"][0].shape == (1, 32, 8)


@pytest.mark.parametrize("shape", [(1, 4, 16, 8), (2, 1, 4, 16, 8)])
def test_physical_nz_like_nd_shape_is_rejected(entry, shape):
    weight = torch.zeros(shape, dtype=torch.int32)
    scale = torch.zeros(64)
    assist = torch.zeros(64)

    with pytest.raises(ValueError, match="2-D per-expert or 3-D stacked weights"):
        _invoke(entry.call, weight, scale, assist)

    assert all(not op.called for op in (entry.nd_op, entry.nd_op.list, entry.nz, entry.nz.list))


@pytest.mark.parametrize(
    "shape,scale_shape,expected",
    [
        ((4, 64), (2, 128), True),
        ((0, 64), (2, 128), True),
        ((4, 48), (2, 128), False),
        ((4, 64), (2, 64), False),
        ((4, 0), (2, 128), False),
        ((4, 64), (), False),
    ],
)
def test_fused_situ_shape_constraints(shape, scale_shape, expected):
    method = _load_method(
        torch,
        "_supports_fused_situ_shape",
        _SITU_GMM_INPUT_ALIGNMENT=64,
        _SITU_GMM_OUTPUT_ALIGNMENT=64,
    )
    assert method(torch.empty(shape), [torch.empty(scale_shape)]) is expected


@pytest.mark.parametrize(
    "k,n,has_compensation,expected_fused",
    [(64, 128, True, True), (48, 128, True, False), (64, 64, True, False), (64, 128, False, False)],
)
def test_situ_gmm1_falls_back_when_fusion_is_unsupported(k, n, has_compensation, expected_fused):
    quantized = torch.empty(4, k, dtype=torch.int8)
    token_scale = torch.ones(4)
    scales = [torch.ones(2, n, dtype=torch.int64)]
    bias = [torch.ones(2, n)] if has_compensation else None
    fallback_gmm = Mock(return_value=["bf16"])
    fallback_situ = Mock(return_value=("separate", "separate_scale"))
    fake_torch = SimpleNamespace(
        int64=torch.int64,
        bfloat16=torch.bfloat16,
        ops=SimpleNamespace(_C_ascend=SimpleNamespace(dequant_situ_quant=fallback_situ)),
    )
    supports_shape = _load_method(
        fake_torch,
        "_supports_fused_situ_shape",
        _SITU_GMM_INPUT_ALIGNMENT=64,
        _SITU_GMM_OUTPUT_ALIGNMENT=64,
    )
    call = _load_method(
        fake_torch,
        "apply_gmm1_act_quant",
        torch_npu=SimpleNamespace(npu_grouped_matmul=fallback_gmm),
        MoEActivation=SimpleNamespace(SITU="situ"),
        dispose_tensor=Mock(),
    )
    method = SimpleNamespace(
        _quant_hidden_states=Mock(return_value=(quantized, token_scale)),
        _get_mlp_weights=Mock(return_value=([object()], scales, [object()], [torch.ones(1, dtype=torch.int64)])),
        _get_mlp_scale_bias=Mock(return_value=(bias, None)),
        _maybe_convert_group_list=Mock(return_value=(torch.tensor([2, 2]), 1)),
        _fuse_gmm_situ_quant=True,
        _supports_fused_situ_shape=supports_shape,
        _fused_gmm1_situ_quant=Mock(return_value=("fused", "fused_scale")),
    )
    inputs = SimpleNamespace(
        hidden_states=object(),
        dynamic_scale=None,
        layer=object(),
        activation="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    result = call(method, inputs)

    if expected_fused:
        assert result == ("fused", "fused_scale")
        method._fused_gmm1_situ_quant.assert_called_once()
        fallback_gmm.assert_not_called()
    else:
        assert result == ("separate", "separate_scale")
        method._fused_gmm1_situ_quant.assert_not_called()
        fallback_gmm.assert_called_once()
        fallback_situ.assert_called_once()


@pytest.mark.parametrize("overlap", [False, True, None], ids=["disabled", "enabled", "uninitialized"])
def test_fused_mlp_preserves_shared_expert_event_contract(overlap):
    path = ROOT / "vllm_ascend/ops/fused_moe/moe_mlp.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_moe_mlp")
    config = Mock()
    if overlap is None:
        config.side_effect = RuntimeError("uninitialized config")
    else:
        config.return_value = SimpleNamespace(multistream_overlap_shared_expert=overlap)
    event = Mock(return_value="event")
    scope = {"get_ascend_config": config, "maybe_record_event": event}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    inputs = SimpleNamespace(quant=SimpleNamespace(is_quant=False), lora_context=None, activation="situ")
    method = Mock()
    method.try_apply_fused_mlp.return_value = "fully_fused"
    method.supports_fused_activation.return_value = True
    method.apply_gmm1_act_quant.return_value = ("intermediate", "scale")
    method.apply_gmm2.return_value = "separate"

    result = scope["apply_moe_mlp"](inputs, method)

    if overlap is False:
        assert result == ("fully_fused", None)
        event.assert_not_called()
        method.apply_gmm2.assert_not_called()
    else:
        assert result == ("separate", "event")
        method.try_apply_fused_mlp.assert_not_called()
        event.assert_called_once()


@pytest.mark.parametrize(
    "expert_list,lora,weight_format,experts,width,up_width,capacity",
    [
        (False, None, 29, 7, 3584, 6144, 2048),
        (False, None, 29, 14, 7168, 7168, 128),
        (True, None, 29, 7, 3584, 6144, 16),
        (False, object(), 29, 7, 3584, 6144, 16),
        (False, None, 2, 7, 3584, 6144, 16),
        (False, None, 29, 7, 3584, 6144, 2049),
        (False, None, 29, 14, 7168, 7168, 129),
    ],
)
def test_fused_mlp_uses_stacked_nz_weights(expert_list, lora, weight_format, experts, width, up_width, capacity):
    op = Mock()
    dispose = Mock()
    fake_torch = SimpleNamespace(
        int32=torch.int32,
        int64=torch.int64,
        ops=SimpleNamespace(_C_ascend=SimpleNamespace(grouped_matmul_situ_quant_mlp=op)),
    )
    method = _load_method(
        fake_torch,
        "try_apply_fused_mlp",
        MoEActivation=SimpleNamespace(SITU="situ"),
        SITU_MLP_SHAPES={(7, 3584): 6144, (14, 7168): 7168},
        SITU_MLP_CAPACITY_LIMITS={7: 2048, 14: 128},
        dispose_tensor=dispose,
    )
    x = torch.empty((capacity, width), dtype=torch.int8, device="meta")
    x_scale = torch.ones((capacity, 1))
    up = torch.empty((experts, width, up_width // 8), dtype=torch.int32, device="meta")
    down = torch.empty((experts, up_width // 2, width // 8), dtype=torch.int32, device="meta")
    up.npu_format = weight_format
    down.npu_format = weight_format
    layer = SimpleNamespace(
        w13_weight=up,
        w2_weight=down,
        w13_scale_bias=torch.ones((experts, up_width)),
        w2_scale_bias=torch.ones((experts, width)),
        w13_weight_scale=torch.ones((experts, up_width), dtype=torch.int64),
        w2_weight_scale=torch.ones((experts, 1, width), dtype=torch.int64),
    )
    inputs = SimpleNamespace(
        dynamic_eplb=False,
        lora_context=lora,
        activation="situ",
        group_list_type=0,
        group_list=torch.full((experts,), 16, dtype=torch.int32),
        layer=layer,
        hidden_states=x,
        dynamic_scale=x_scale,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    scheme = SimpleNamespace(
        _fuse_gmm_situ_quant=True,
        use_expert_weight_list=expert_list,
        _quant_hidden_states=Mock(return_value=(x, x_scale)),
    )
    result = method(scheme, inputs)
    capacity_limit = 2048 if experts == 7 else 128
    if expert_list or lora is not None or weight_format != NZ_FORMAT or capacity > capacity_limit:
        assert result is None
        op.assert_not_called()
        dispose.assert_not_called()
    else:
        assert result is op.return_value
        args = op.call_args.args
        assert args[0] is x and args[3] is up and args[6] is down
        assert args[1].shape == (capacity,) and args[2].dtype == torch.int64
        assert args[9:] == (0, 4.0, 25.0)
        dispose.assert_called_once_with(x)
