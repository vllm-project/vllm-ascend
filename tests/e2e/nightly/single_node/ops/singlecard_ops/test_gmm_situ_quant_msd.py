# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 A8W4 MSD GroupedMatmulSituQuant correctness, native NZ, and graph regressions."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend.vllm_ascend_C")

BETA = 4.0
LINEAR_BETA = 25.0


@pytest.fixture(autouse=True)
def require_a3():
    if not torch.npu.is_available():
        pytest.skip("GroupedMatmulSituQuant A8W4 MSD requires an NPU")
    soc_version = torch_npu.npu.get_soc_version()
    if not 250 <= soc_version <= 256:
        pytest.skip("GroupedMatmulSituQuant A8W4 MSD regression requires Ascend A3")
    assert hasattr(torch.ops._C_ascend, "grouped_matmul_situ_quant"), "A3 extension must register the fused op"
    assert hasattr(torch.ops._C_ascend, "grouped_matmul_situ_quant_weight_nz"), "NZ entry must be registered"
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") != b"disable"
    torch.npu.config.allow_internal_format = True
    try:
        yield
    finally:
        torch.npu.synchronize()
        torch.npu.config.allow_internal_format = previous


def _pack_int4(weight):
    lanes = weight.reshape(weight.shape[0], -1, 8).to(torch.int64)
    packed = torch.zeros(lanes.shape[:-1], dtype=torch.int64)
    for lane in range(8):
        packed |= (lanes[..., lane] & 15) << (4 * lane)
    return packed.to(torch.int32)


def _layer(experts, k, n, nz, seed):
    generator = torch.Generator().manual_seed(seed)
    logical = [torch.randint(-8, 8, (k, n), generator=generator, dtype=torch.int8) for _ in range(experts)]
    scales = [torch.rand(n, generator=generator) * 0.001 + 0.0001 for _ in range(experts)]
    packed = torch.stack([_pack_int4(weight) for weight in logical]).npu()
    if nz:
        # The loader owns INT8 NZ storage and exposes an INT32 carrier view.
        # Casting INT32 storage to NZ creates an incompatible descriptor.
        native = torch_npu.npu_format_cast(packed.view(torch.int8).clone(), 29)
        packed = native.view(torch.int32)
        assert native.data_ptr() == packed.data_ptr()
        # The loader clones INT8 NZ storage before creating INT32 carriers.
        weights = [w.clone().view(torch.int32) for w in native.unbind()]
    else:
        weights = [w.clone() for w in packed.unbind()]
    assert all(torch_npu.get_npu_format(weight) == (29 if nz else 2) for weight in weights)
    # Match the model's INT64 carrier of packed UINT64 scale bits.
    encoded = torch.stack([scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF for scale in scales]).npu()
    assist = torch.stack([8 * w.float().sum(0) * s for w, s in zip(logical, scales)]).npu()
    return weights, list(encoded.unbind()), logical, scales, packed, encoded, assist


def _reference(x, x_scale, layer, counts, external_assist=False, beta=BETA, linear_beta=LINEAR_BETA):
    outputs, scales, activations = [], [], []
    offset = 0
    for weight, weight_scale, rows in zip(layer[2], layer[3], counts):
        if rows:
            # Independently reproduce MSD's two INT4 products and FP16 cube
            # rounding, followed by FP32 compensation and token scaling.
            tokens = x[offset : offset + rows].to(torch.int32)
            high, low = tokens // 16, (tokens & 15) - 8
            high_mm = ((high.float() @ weight.float()) * weight_scale).half().float()
            low_mm = ((low.float() @ weight.float()) * weight_scale).half().float()
            compensation = 8 * weight.float().sum(0) * weight_scale
            if not external_assist:
                compensation = compensation.half().float()
            hidden = (16 * high_mm + low_mm + compensation) * x_scale[offset : offset + rows, None]
            gate, up = hidden.chunk(2, dim=-1)
            gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
            if linear_beta > 0:
                up = linear_beta * torch.tanh(up / linear_beta)
            act = gate * up
            row_max = act.abs().amax(dim=-1)
            inverse = torch.where(row_max > 0, 127 / row_max, torch.zeros_like(row_max))
            outputs.append((act * inverse[:, None]).round().clamp(-128, 127).to(torch.int8))
            scales.append(row_max / 127)
            activations.append(act)
        offset += rows
    if not outputs:
        n = layer[2][0].shape[1]
        return torch.empty((0, n // 2), dtype=torch.int8), torch.empty(0), torch.empty((0, n // 2))
    return torch.cat(outputs), torch.cat(scales), torch.cat(activations)


def _check(output, expected, capacity):
    y, scale = (tensor.cpu() for tensor in output)
    ref_y, ref_scale, act = expected
    assert y.dtype == torch.int8 and scale.dtype == torch.float32
    assert y.shape == (capacity, ref_y.shape[1]) and scale.shape == (capacity,)
    rows = ref_y.shape[0]
    if not rows:
        return  # Padding contents are unspecified by the operator contract.
    y, scale = y[:rows], scale[:rows]
    torch.testing.assert_close(scale, ref_scale, rtol=1e-3, atol=1e-5)
    # External FP32 compensation cancels FP16 partial products on a zero
    # input row. SiTU can retain ~1e-11 residuals, for which normalizing by
    # a ~1e-13 scale makes raw codes unstable. Check those reconstructed
    # values with a strict absolute bound instead of comparing codes.
    meaningful = act.abs().amax(-1) > 1e-8
    difference = (y[meaningful].to(torch.int16) - ref_y[meaningful].to(torch.int16)).abs()
    if difference.numel():
        assert difference.max().item() <= 1
        assert (difference == 1).float().mean().item() < 0.02
    torch.testing.assert_close(y[~meaningful].float() * scale[~meaningful, None], act[~meaningful], rtol=0, atol=1e-8)
    # Quantization alone can differ from the unquantized activation by half
    # a row's step (up to 0.394 for saturated beta=4, linear_beta=25).
    # Include that rounding error and the one-code allowance above; retain
    # the separately checked 0.1% scale error in reconstructed values.
    error_bound = 1.5 * ref_scale[:, None] + 1e-3 * act.abs() + 1e-5
    assert torch.all((y.float() * scale[:, None] - act).abs() <= error_bound)


def _call(
    x,
    x_scale,
    layer,
    group_list,
    group_list_type,
    nz=False,
    stacked=False,
    external_assist=False,
    beta=BETA,
    linear_beta=LINEAR_BETA,
):
    op = (
        torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz if nz else torch.ops._C_ascend.grouped_matmul_situ_quant
    )
    if stacked:
        weight, weight_scale = layer[4:6]
        assist = layer[6] if external_assist else None
    else:
        op = op.list
        weight, weight_scale = layer[:2]
        assist = list(layer[6].unbind()) if external_assist else None
    return op(
        x,
        weight,
        weight_scale,
        assist,
        None,  # bias is unsupported by contract
        x_scale,
        None,  # smooth_scale is unsupported by contract
        group_list.to(torch.int64),
        dequant_mode=0,
        dequant_dtype=0,
        quant_mode=0,
        group_list_type=group_list_type,
        tuning_config=None,
        beta=beta,
        linear_beta=linear_beta,
    )


def _groups(counts, dtype, group_list_type):
    groups = torch.tensor(counts, dtype=dtype)
    return groups.cumsum(0) if group_list_type == 0 else groups


@pytest.mark.parametrize("capacity", [0, 8])
@pytest.mark.parametrize("fake", [False, True], ids=["meta", "fake_npu"])
def test_gmm_situ_quant_msd_dispatcher_shape_inference(capacity, fake):
    op_name = "_C_ascend::grouped_matmul_situ_quant"
    assert torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "PrivateUse1")
    assert torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "Meta")
    device = "npu" if fake else "meta"
    with FakeTensorMode() if fake else torch.device("meta"):
        x = torch.empty((capacity, 320), dtype=torch.int8, device=device)
        weight = torch.empty((320, 96), dtype=torch.int32, device=device)
        weight_scale = torch.empty(768, dtype=torch.int64, device=device)
        x_scale = torch.empty(capacity, dtype=torch.float32, device=device)
        group_list = torch.empty(1, dtype=torch.int64, device=device)
        # Exercise the dispatcher schema's default scalar arguments as well.
        y, scale = torch.ops._C_ascend.grouped_matmul_situ_quant(
            x=x,
            weight=[weight],
            weight_scale=[weight_scale],
            weight_assist_matrix=None,
            bias=None,
            x_scale=x_scale,
            smooth_scale=None,
            group_list=group_list,
            dequant_mode=0,
            dequant_dtype=0,
            quant_mode=0,
            group_list_type=1,
            tuning_config=None,
            beta=BETA,
            linear_beta=LINEAR_BETA,
        )
        assert y.shape == (capacity, 384) and y.dtype == torch.int8
        assert scale.shape == (capacity,) and scale.dtype == torch.float32
        assert y.device == x.device and scale.device == x.device


@pytest.mark.parametrize("nz", [False, True], ids=["nd_weights", "native_nz"])
@pytest.mark.parametrize("experts,k,n", [(1, 320, 768), (2, 320, 768), (3, 576, 1280)])
@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize("external_assist", [False, True])
def test_gmm_situ_quant_msd_eager(nz, experts, k, n, group_list_type, stacked, external_assist):
    capacity = 8
    generator = torch.Generator().manual_seed(71)
    x = torch.randint(-32, 32, (capacity, k), generator=generator, dtype=torch.int8)
    x[0].zero_()  # Exercise zero activation and zero quantization scale.
    x_scale = torch.rand(capacity, generator=generator) * 0.01 + 0.001
    counts = [3, 0, 2][:experts]
    layer = _layer(experts, k, n, nz, 101)
    groups = _groups(counts, torch.int64, group_list_type).npu()
    output = _call(x.npu(), x_scale.npu(), layer, groups, group_list_type, nz, stacked, external_assist)
    _check(output, _reference(x, x_scale, layer, counts, external_assist), capacity)


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("external_assist", [False, True])
def test_gmm_situ_quant_msd_graph_dynamic_groups(dtype, group_list_type, external_assist):
    # Distinct shapes, expert counts and layouts share one capture. Each call
    # must retain its own scratch storage and device-side routing metadata.
    cases = []
    for index, (experts, k, n, nz) in enumerate([(2, 320, 768, False), (3, 576, 1280, True)]):
        generator = torch.Generator().manual_seed(81 + index)
        x = torch.randint(-32, 32, (8, k), generator=generator, dtype=torch.int8)
        x_scale = torch.rand(8, generator=generator) * 0.01 + 0.001
        layer = _layer(experts, k, n, nz, 201 + index)
        groups = _groups([1] * experts, dtype, group_list_type).npu()
        cases.append((x, x_scale, layer, x.npu(), x_scale.npu(), groups, nz))
    for _, _, layer, x_device, scale_device, groups, nz in cases:
        _call(x_device, scale_device, layer, groups, group_list_type, nz, external_assist=external_assist)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="global"):
        outputs = [
            _call(xd, sd, layer, groups, group_list_type, nz, external_assist=external_assist)
            for _, _, layer, xd, sd, groups, nz in cases
        ]
    for routes in [([3, 0], [0, 2, 3]), ([0, 4], [2, 0, 1]), ([0, 0], [0, 0, 0]), ([1, 2], [1, 2, 1])]:
        for case, counts in zip(cases, routes):
            case[-2].copy_(_groups(counts, dtype, group_list_type))
        graph.replay()
        torch.npu.synchronize()
        for output, (x, x_scale, layer, *_), counts in zip(outputs, cases, routes):
            _check(output, _reference(x, x_scale, layer, counts, external_assist), x.shape[0])


@pytest.mark.parametrize("nz", [False, True])
@pytest.mark.parametrize("beta,linear_beta", [(1.0, 0.0), (2.0, 3.0), (0.5, -1.0)])
def test_gmm_situ_quant_msd_activation_parameters(nz, beta, linear_beta):
    generator = torch.Generator().manual_seed(911)
    x = torch.randint(-128, 128, (4, 320), generator=generator, dtype=torch.int8)
    x_scale = torch.tensor([0.0, 0.01, 0.1, 1.0])
    layer = _layer(1, 320, 768, nz, 913)
    groups = torch.tensor([4], dtype=torch.int64, device="npu")
    output = _call(
        x.npu(),
        x_scale.npu(),
        layer,
        groups,
        1,
        nz=nz,
        external_assist=True,
        beta=beta,
        linear_beta=linear_beta,
    )
    expected = _reference(x, x_scale, layer, [4], external_assist=True, beta=beta, linear_beta=linear_beta)
    _check(output, expected, 4)


@pytest.mark.parametrize("nz", [False, True])
def test_gmm_situ_quant_msd_large_token_scale(nz):
    generator = torch.Generator().manual_seed(901)
    x = torch.randint(-128, 128, (4, 320), generator=generator, dtype=torch.int8)
    x_scale = torch.tensor([0.0, 0.01, 10.0, 100.0])
    layer = _layer(1, 320, 768, nz, 903)
    groups = torch.tensor([4], dtype=torch.int64, device="npu")
    output = _call(x.npu(), x_scale.npu(), layer, groups, 1, nz=nz, external_assist=True)
    _check(output, _reference(x, x_scale, layer, [4], external_assist=True), 4)


@pytest.mark.parametrize("nz", [False, True])
def test_gmm_situ_quant_msd_zero_capacity(nz):
    layer = _layer(2, 320, 768, nz, 301)
    x = torch.empty((0, 320), dtype=torch.int8)
    scale = torch.empty(0)
    counts = [0, 0]
    output = _call(x.npu(), scale.npu(), layer, torch.tensor(counts, device="npu"), 1, nz)
    _check(output, _reference(x, scale, layer, counts), 0)


@pytest.mark.parametrize(
    "experts,k,n,rows,capacity",
    [(1, 320, 768, 5, 8), (3, 576, 1280, 33, 64), (14, 3584, 6144, 128, 7168), (14, 7168, 7168, 128, 7168)],
)
@pytest.mark.parametrize("stacked", [False, True])
def test_gmm_situ_quant_msd_model_split_reference(experts, k, n, rows, capacity, stacked):
    """Compare against the production BF16 GMM + dequant_situ_quant chain.

    The fused kernel retains FP32 after merging MSD's FP16 partial products;
    the split chain rounds that merge to BF16. Compare dequantized outputs
    within the resulting BF16/INT8 rounding budget, not byte equality.
    """
    generator = torch.Generator().manual_seed(431)
    x = torch.randint(-128, 128, (capacity, k), dtype=torch.int8, generator=generator).npu()
    x_scale = (torch.rand(capacity, generator=generator) * 0.01 + 0.001).npu()
    layer = _layer(experts, k, n, True, 433)
    counts = [rows // experts + int(i < rows % experts) for i in range(experts)]
    groups = torch.tensor(counts, dtype=torch.int64, device="npu")
    # The split CANN GMM treats a list of length one as a stacked tensor.
    # Normalize only the reference call; the fused entry also supports [K,N/8].
    reference_stacked = stacked or experts == 1
    weights = [layer[4]] if reference_stacked else layer[0]
    scales = [layer[5]] if reference_stacked else layer[1]
    assist = [layer[6]] if reference_stacked else list(layer[6].unbind())
    intermediate = torch_npu.npu_grouped_matmul(
        # The CANN A8W4 split kernel rewrites x while preparing MSD nibbles.
        # Give the reference its own input so both paths see identical data.
        x=[x.clone()],
        weight=weights,
        scale=[s.unsqueeze(-2) for s in scales],
        bias=assist,
        per_token_scale=[x_scale],
        split_item=2,
        group_type=0,
        group_list=groups,
        group_list_type=1,
        output_dtype=torch.bfloat16,
    )[0]
    reference = torch.ops._C_ascend.dequant_situ_quant(
        x=intermediate,
        weight_scale=None,
        activation_scale=None,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=groups,
        beta=BETA,
        linear_beta=LINEAR_BETA,
        activate_left=True,
        quant_mode="dynamic",
    )
    actual = _call(x, x_scale, layer, groups, 1, nz=True, stacked=stacked, external_assist=True)
    actual_value = actual[0][:rows].float() * actual[1][:rows, None]
    reference_value = reference[0][:rows].float() * reference[1][:rows, None]
    # BF16 merge rounding and two independent INT8 quantizations contribute
    # bounded absolute error proportional to each row's quantization step.
    error = (actual_value - reference_value).abs().cpu()
    budget = (3 * reference[1][:rows, None] + 0.02 * reference_value.abs()).cpu()
    assert torch.isfinite(actual_value).all()
    assert torch.all(error <= budget + 1e-5), (error - budget).max().item()


@pytest.mark.parametrize(
    "invalid",
    [
        "group_dtype",
        "weight_dtype",
        "weight_offset",
        "scale_dtype",
        "assist_size",
        "bias",
        "smooth_scale",
        "mode",
        "beta",
    ],
)
def test_gmm_situ_quant_msd_rejects_invalid_metadata(invalid):
    layer = _layer(1, 320, 768, True, 501)
    kwargs = dict(
        x=torch.ones(4, 320, dtype=torch.int8, device="npu"),
        weight=layer[4],
        weight_scale=layer[5],
        weight_assist_matrix=layer[6],
        bias=None,
        x_scale=torch.ones(4, device="npu"),
        smooth_scale=None,
        group_list=torch.tensor([4], dtype=torch.int64, device="npu"),
        dequant_mode=0,
        dequant_dtype=0,
        quant_mode=0,
        group_list_type=1,
        tuning_config=None,
        beta=BETA,
        linear_beta=LINEAR_BETA,
    )
    if invalid == "group_dtype":
        kwargs["group_list"] = kwargs["group_list"].to(torch.int32)
    elif invalid == "weight_dtype":
        kwargs["weight"] = kwargs["weight"].view(torch.int8)
    elif invalid == "weight_offset":
        kwargs["weight"] = _layer(2, 320, 768, True, 503)[4][1:]
        assert kwargs["weight"].storage_offset() > 0
    elif invalid == "scale_dtype":
        kwargs["weight_scale"] = kwargs["weight_scale"].float()
    elif invalid == "assist_size":
        kwargs["weight_assist_matrix"] = torch.zeros(1, device="npu")
    elif invalid in ("bias", "smooth_scale"):
        kwargs[invalid] = torch.zeros(768, device="npu")
    elif invalid == "mode":
        kwargs["quant_mode"] = 1
    else:
        kwargs["beta"] = 0.0
    with pytest.raises(RuntimeError, match="SiTU MSD"):
        torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz(**kwargs)
