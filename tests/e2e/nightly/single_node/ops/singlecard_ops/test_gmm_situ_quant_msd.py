# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A2/A3 A8W4 MSD GroupedMatmulSituQuant correctness, native NZ, and graph regressions."""

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend.vllm_ascend_C")

BETA = 4.0
LINEAR_BETA = 25.0


@pytest.fixture(autouse=True)
def require_a2a3():
    if not torch.npu.is_available():
        pytest.skip("GroupedMatmulSituQuant A8W4 MSD requires an NPU")
    soc_version = torch_npu.npu.get_soc_version()
    if not (220 <= soc_version <= 225 or 250 <= soc_version <= 256):
        pytest.skip("GroupedMatmulSituQuant A8W4 MSD regression requires Ascend A2/A3")
    assert hasattr(torch.ops._C_ascend, "grouped_matmul_situ_quant"), "A2/A3 extension must register the fused op"
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


def _layer(experts, k, n, nz, seed, *, reference_k=None):
    generator = torch.Generator().manual_seed(seed)
    logical_k = k if reference_k is None else reference_k
    logical = [torch.randint(-8, 8, (logical_k, n), generator=generator, dtype=torch.int8) for _ in range(experts)]
    scales = [torch.rand(n, generator=generator) * 0.001 + 0.0001 for _ in range(experts)]
    packed = torch.stack([_pack_int4(weight) for weight in logical]).npu()
    if logical_k != k:
        # Zero-pad the physical weights and keep the CPU reference small.
        padded = packed.new_zeros((experts, k, n // 8))
        padded[:, :logical_k].copy_(packed)
        packed = padded
    if nz:
        # The loader owns INT8 NZ storage and exposes an INT32 carrier view.
        # Casting INT32 storage to NZ creates an incompatible descriptor.
        native = torch_npu.npu_format_cast(packed.view(torch.int8).clone(), 29)
        packed = native.view(torch.int32)
        assert native.data_ptr() == packed.data_ptr()
    weights = list(packed.unbind())
    if experts > 1:
        assert weights[1].storage_offset() > 0
    assert all(torch_npu.get_npu_format(weight) == (29 if nz else 2) for weight in weights)
    # MSD reads the per-channel scale as packed UINT64 bits; torch has no uint64,
    # so the flattened float32 bits are carried in int64 unchanged.
    encoded = torch.stack([scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF for scale in scales]).npu()
    return weights, list(encoded.unbind()), logical, scales, (packed, encoded)


def _reference(x, x_scale, layer, counts, provided_assist=False, beta=BETA, linear_beta=LINEAR_BETA):
    outputs, scales, activations = [], [], []
    offset = 0
    for weight, weight_scale, rows in zip(layer[2], layer[3], counts):
        if rows:
            # Independent CPU MSD arithmetic: each signed INT4 product is
            # scaled and rounded to FP16 before the two partials are merged.
            values = x[offset : offset + rows].to(torch.int32)
            high = torch.div(values, 16, rounding_mode="floor").float()
            low = (values.remainder(16) - 8).float()
            high_product = ((high @ weight.float()) * weight_scale).half().float()
            low_product = ((low @ weight.float()) * weight_scale).half().float()
            compensation = 8 * weight.float().sum(0) * weight_scale
            if not provided_assist:
                compensation = compensation.half().float()
            hidden = (16 * high_product + low_product + compensation) * x_scale[offset : offset + rows, None]
            gate, up = hidden.chunk(2, dim=-1)
            gate = ((2 * torch.sigmoid(gate * (2 / beta)) - 1) * torch.sigmoid(gate)) * beta
            if linear_beta > 0:
                up = (2 * torch.sigmoid(up * (2 / linear_beta)) - 1) * linear_beta
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
    difference = (y.to(torch.int16) - ref_y.to(torch.int16)).abs()
    assert difference.max().item() <= 1
    assert (difference == 1).float().mean().item() < 0.02
    torch.testing.assert_close(y.float() * scale[:, None], act, rtol=2e-2, atol=2e-2)


def _call(
    x,
    x_scale,
    layer,
    group_list,
    group_list_type,
    nz=False,
    assist=None,
    stacked=False,
    beta=BETA,
    linear_beta=LINEAR_BETA,
):
    op = (
        torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz if nz else torch.ops._C_ascend.grouped_matmul_situ_quant
    )
    if not stacked:
        op = op.list
    weight, weight_scale = layer[4] if stacked else layer[:2]
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
def test_gmm_situ_quant_msd_eager(nz, experts, k, n, group_list_type):
    capacity = 8
    generator = torch.Generator().manual_seed(71)
    x = torch.randint(-32, 32, (capacity, k), generator=generator, dtype=torch.int8)
    x[0].zero_()  # Exercise zero activation and zero quantization scale.
    x_scale = torch.rand(capacity, generator=generator) * 0.01 + 0.001
    counts = [3, 0, 2][:experts]
    layer = _layer(experts, k, n, nz, 101)
    groups = _groups(counts, torch.int64, group_list_type).npu()
    output = _call(x.npu(), x_scale.npu(), layer, groups, group_list_type, nz)
    _check(output, _reference(x, x_scale, layer, counts), capacity)


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("group_list_type", [0, 1])
def test_gmm_situ_quant_msd_graph_dynamic_groups(dtype, group_list_type):
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
        _call(x_device, scale_device, layer, groups, group_list_type, nz)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="global"):
        outputs = [_call(xd, sd, layer, groups, group_list_type, nz) for _, _, layer, xd, sd, groups, nz in cases]
    for routes in [([3, 0], [0, 2, 3]), ([0, 4], [2, 0, 1]), ([0, 0], [0, 0, 0]), ([1, 2], [1, 2, 1])]:
        for case, counts in zip(cases, routes):
            case[-2].copy_(_groups(counts, dtype, group_list_type))
        graph.replay()
        torch.npu.synchronize()
        for output, (x, x_scale, layer, *_), counts in zip(outputs, cases, routes):
            _check(output, _reference(x, x_scale, layer, counts), x.shape[0])


@pytest.mark.parametrize("nz", [False, True])
def test_gmm_situ_quant_msd_zero_capacity(nz):
    layer = _layer(2, 320, 768, nz, 301)
    x = torch.empty((0, 320), dtype=torch.int8)
    scale = torch.empty(0)
    counts = [0, 0]
    output = _call(x.npu(), scale.npu(), layer, torch.tensor(counts, device="npu"), 1, nz)
    _check(output, _reference(x, scale, layer, counts), 0)


@pytest.mark.parametrize("provided_assist", [False, True])
@pytest.mark.parametrize("group_list_type", [0, 1])
def test_gmm_situ_quant_msd_graph_workspace_boundaries(provided_assist, group_list_type):
    # This shape crosses the 64 MiB double-buffer workspace limit at about
    # 1424 rows. Replay one captured graph across zero, one and two chunks;
    # sample both sides of the chunk boundary against independent CPU MSD.
    k, n, capacity = 7168, 4096, 1536
    layer = _layer(3, k, n, True, 827)
    generator = torch.Generator().manual_seed(828)
    x = torch.randint(-128, 128, (capacity, k), generator=generator, dtype=torch.int8)
    x_scale = torch.full((capacity,), 0.01)
    xd, sd = x.npu(), x_scale.npu()
    assist = None
    if provided_assist:
        assist = torch.stack([8 * w.float().sum(0) * s for w, s in zip(layer[2], layer[3])]).npu()
    groups = _groups([3, 0, 2], torch.int64, group_list_type).npu()
    _call(xd, sd, layer, groups, group_list_type, True, assist, True)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [_call(xd, sd, layer, groups, group_list_type, True, assist, True) for _ in range(2)]
    for counts in ([3, 0, 2], [768, 0, 768], [0, 0, 0], [0, 1536, 0], [2, 1, 1]):
        groups.copy_(_groups(counts, torch.int64, group_list_type))
        graph.replay()
        torch.npu.synchronize()
        # Keep CPU reference work bounded while exercising the full NPU shape.
        indices, sampled_counts, start = [], [], 0
        for count in counts:
            selected = sorted({start, start + count - 1, *range(max(start, 1418), min(start + count, 1428))})
            selected = [i for i in selected if start <= i < start + count]
            indices.extend(selected)
            sampled_counts.append(len(selected))
            start += count
        if not indices:
            continue
        expected = _reference(x[indices], x_scale[indices], layer, sampled_counts, provided_assist)
        for output in outputs:
            _check(tuple(t.cpu()[indices] for t in output), expected, len(indices))


@pytest.mark.parametrize("nz", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize("experts", [1, 3])
def test_gmm_situ_quant_msd_weight_compensation(nz, stacked, experts):
    torch.manual_seed(411)
    layer = _layer(experts, 320, 768, nz, 411)
    assist = torch.stack([8 * w.float().sum(0) * s for w, s in zip(layer[2], layer[3])]).npu()
    if not stacked:
        assist = list(assist.unbind())
    x = torch.randint(-128, 128, (8, 320), dtype=torch.int8)
    x_scale = torch.full((8,), 0.01)
    counts = [2] * experts
    output = _call(x.npu(), x_scale.npu(), layer, torch.tensor(counts, device="npu"), 1, nz, assist, stacked)
    _check(output, _reference(x, x_scale, layer, counts, provided_assist=True), 8)


@pytest.mark.parametrize("beta,linear_beta", [(1.0, 0.0), (2.0, -1.0)])
def test_gmm_situ_quant_msd_without_linear_transform(beta, linear_beta):
    torch.manual_seed(418)
    data = _layer(2, 320, 768, True, 418)
    x = torch.randint(-128, 128, (8, 320), dtype=torch.int8)
    x_scale = torch.full((8,), 0.01)
    counts = [2, 3]
    output = _call(
        x.npu(), x_scale.npu(), data, torch.tensor(counts, device="npu"), 1, True, beta=beta, linear_beta=linear_beta
    )
    _check(output, _reference(x, x_scale, data, counts, beta=beta, linear_beta=linear_beta), 8)


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("k,n", [(320, 768), (7168, 4096)])
def test_w4a8_method_native_situ(as_list, k, n):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod

    torch.manual_seed(512)
    data = _layer(3, k, n, True, 512)
    assist = torch.stack([8 * w.float().sum(0) * s for w, s in zip(data[2], data[3])]).npu()
    weights, scales = data[4]
    layer = SimpleNamespace()
    for prefix in ("w13", "w2"):
        for suffix, value in (("weight", weights), ("weight_scale", scales), ("scale_bias", assist)):
            if as_list:
                # Match the loader's per-expert INT8 NZ storage.
                items = list(value.unbind())
                if suffix == "weight":
                    items = [item.view(torch.int8) for item in items]
                setattr(layer, f"{prefix}_{suffix}_list", items)
            else:
                setattr(layer, f"{prefix}_{suffix}", value)
    method = AscendW4A8DynamicFusedMoEMethod.__new__(AscendW4A8DynamicFusedMoEMethod)
    method.use_expert_weight_list = as_list
    method.is_per_channel_weight = True
    x = torch.randint(-128, 128, (8, k), dtype=torch.int8)
    x_scale = torch.full((8,), 0.01)
    inputs = SimpleNamespace(
        hidden_states=x.npu(),
        dynamic_scale=x_scale.npu(),
        layer=layer,
        group_list=torch.tensor([3, 3, 5], dtype=torch.int32, device="npu"),
        group_list_type=0,
        activation=MoEActivation.SITU,
        activation_situ_beta=BETA,
        activation_situ_linear_beta=LINEAR_BETA,
    )
    output = method.apply_gmm1_act_quant(inputs)
    _check(output, _reference(x, x_scale, data, [3, 0, 2], provided_assist=True), 8)


@pytest.mark.parametrize(
    "invalid,match",
    [("weight", "INT32"), ("weight_scale", "INT64"), ("group_list", "int64"), ("beta", "finite beta")],
)
def test_gmm_situ_quant_msd_rejects_invalid_metadata(invalid, match):
    data = _layer(1, 320, 768, False, 613)
    kwargs = dict(
        x=torch.zeros((1, 320), dtype=torch.int8, device="npu"),
        weight=data[0],
        weight_scale=data[1],
        weight_assist_matrix=None,
        bias=None,
        x_scale=torch.ones(1, device="npu"),
        smooth_scale=None,
        group_list=torch.ones(1, dtype=torch.int64, device="npu"),
        dequant_mode=0,
        dequant_dtype=0,
        quant_mode=0,
        group_list_type=1,
        tuning_config=None,
        beta=BETA,
        linear_beta=LINEAR_BETA,
    )
    if invalid == "beta":
        kwargs[invalid] = 0.0
    elif invalid == "group_list":
        kwargs[invalid] = kwargs[invalid].int()
    else:
        kwargs[invalid] = [kwargs[invalid][0].float()]
    with pytest.raises(RuntimeError, match=match):
        torch.ops._C_ascend.grouped_matmul_situ_quant.list(**kwargs)


@pytest.mark.parametrize("group_list_type", [0, 1], ids=["prefix", "counts"])
@pytest.mark.parametrize(
    "k,n,capacity,reference_k,routes,external_assist",
    [
        (
            7168,
            7168,
            1024,
            256,
            [
                [67] * 11 + [66] * 3,  # 935 rows and an eligible expert boundary.
                [70] * 12 + [0, 97],  # 937 rows and a zero-sized expert.
                [935] + [0] * 13,  # No eligible expert boundary.
                [16] + [0] * 13,
            ],
            False,
        ),
        (
            3584,
            6144,
            1280,
            64,
            [
                [0, 43, 113, 0, 149, 160, 175],  # Exactly fills one 640-row slot.
                [641] + [0] * 6,  # One expert spans both slots.
                [180] * 6 + [200],  # Replay the full captured capacity.
                [16] + [0] * 6,
                [0] * 7,
            ],
            False,
        ),
        (
            3584,
            6144,
            7168,
            64,
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0] * 7,
                [0] * 6 + [16],
                [2, 2, 2, 2, 2, 3, 3],
                [32] + [0] * 6,
                [0, 33] + [0] * 5,
                [171, 171, 170] + [0] * 4,
                [91] * 6 + [94],
                [0] * 6 + [641],
                [0] * 7,
                [1] + [0] * 6,
            ],
            True,
        ),
        (
            3584,
            6144,
            16,
            64,
            [
                [0] * 6 + [1],
                [0] * 7,
                [16] + [0] * 6,
                [0, 3, 0, 8, 0, 5, 0],
                [0] * 6 + [16],
            ],
            True,
        ),
        (
            3584,
            6144,
            2048,
            64,
            [
                [0, 3, 0, 8, 0, 5, 0],  # Sparse decode at the 16-row boundary.
                [0, 0, 0, 0, 0, 0, 17],
                [0] * 7,
                [0, 0, 0, 1, 0, 0, 0],
                [2, 2, 2, 2, 2, 3, 3],
            ],
            True,
        ),
        (
            7168,
            7168,
            16,
            64,
            [
                [0] * 13 + [1],
                [0] * 14,
                [0, 4, 0, 12] + [0] * 10,
                [0] * 13 + [16],
                [8] + [0] * 13,
                [0] * 5 + [9] + [0] * 8,
                [1] + [0] * 13,
            ],
            True,
        ),
        (
            7168,
            7168,
            2048,
            64,
            [
                [0, 0, 0, 1] + [0] * 10,
                [16] + [0] * 13,
                [0] * 13 + [15],
                [0] * 13 + [17],
                [0] * 13 + [32],
                [0, 33] + [0] * 12,
                [9] * 13 + [11],
                [0] * 14,
                [0] * 13 + [1],
            ],
            True,
        ),
    ],
    ids=[
        "e14-boundary",
        "e7-compact-workspace",
        "e7-wide-decode-external-assist",
        "e7-tiny-decode-external-assist",
        "e7-decode-external-assist",
        "e14-single-token-shared-pack",
        "e14-single-token-local-pack",
    ],
)
def test_gmm_situ_quant_msd_graph_expert_boundaries(
    k, n, capacity, reference_k, routes, external_assist, group_list_type
):
    experts = len(routes[0])
    generator = torch.Generator().manual_seed(91)
    x_reference = torch.randint(-32, 32, (capacity, reference_k), generator=generator, dtype=torch.int8)
    if not external_assist:
        x_reference[0].zero_()
    x = torch.zeros((capacity, k), dtype=torch.int8)
    x[:, :reference_k] = x_reference
    x_scale = torch.rand(capacity, generator=generator) * 0.01 + 0.001
    layer = _layer(experts, k, n, True, 401, reference_k=reference_k)
    assist = None
    if external_assist:
        assist = torch.stack([8 * w.float().sum(dim=0) * scale for w, scale in zip(layer[2], layer[3])]).npu()
    x_device, scale_device = x.npu(), x_scale.npu()
    groups = _groups(routes[0], torch.int64, group_list_type).npu()
    _call(x_device, scale_device, layer, groups, group_list_type, nz=True, assist=assist, stacked=True)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="global"):
        output = _call(x_device, scale_device, layer, groups, group_list_type, nz=True, assist=assist, stacked=True)
    for counts in routes:
        groups.copy_(_groups(counts, torch.int64, group_list_type))
        graph.replay()
        torch.npu.synchronize()
        _check(output, _reference(x_reference, x_scale, layer, counts, external_assist), capacity)


@pytest.mark.parametrize("group_list_type", [0, 1], ids=["prefix", "counts"])
@pytest.mark.parametrize("capacity", [16, 640, 641, 2048, 7168])
def test_gmm_situ_mlp_graph_matches_existing_fusion(group_list_type, capacity):
    experts, width, up_width = 7, 3584, 6144
    up = _layer(experts, width, up_width, True, 604, reference_k=64)
    down = _layer(experts, up_width // 2, width, True, 605, reference_k=64)

    def compensation(weights):
        return torch.stack([8 * weight.float().sum(0) * scale for weight, scale in zip(weights[2], weights[3])]).npu()

    up_compensation, down_compensation = compensation(up), compensation(down)
    rng = torch.Generator().manual_seed(606)
    x = torch.randint(-127, 128, (capacity, width), dtype=torch.int8, generator=rng).npu()
    x_scale = (torch.rand(capacity, generator=rng) * 0.02 + 0.001).npu()
    routes = [[16] + [0] * 6, [0] * 7, [0] * 6 + [1], [0, 3, 0, 8, 0, 5, 0]]
    routes.extend([[0] * 6 + [2], [0, 0, 8, 0, 0, 0, 0], [0] * 6 + [16]])
    if capacity >= 17:
        routes.append([0] * 6 + [17])
    if capacity >= 128:
        routes.extend([[0, 8, 0, 9, 0, 0, 0], [18] * 6 + [20], [0] * 6 + [128]])
    if capacity >= 129:
        routes.extend([[18] * 6 + [21], [93] * 6 + [83]])
    if 640 <= capacity <= 641:
        routes.extend([[0] * 6 + [rows] for rows in range(639, capacity + 1)])
        routes.append([0] * 6 + [8])
    if capacity >= 7168:
        # Cross the reused down-projection workspace slot and prefill-tile boundaries.
        routes.extend([[0] * 6 + [rows] for rows in (1256, 1257, 1258, 2049)])
        routes.extend([[293] * 6 + [290], [585] * 6 + [586], [1024] * 7, [0] * 7, [1] + [0] * 6])
    groups = _groups(routes[0], torch.int64, group_list_type).npu()
    op = torch.ops._C_ascend.grouped_matmul_situ_quant_mlp

    def fused():
        return op(
            x,
            x_scale,
            groups,
            up[4][0],
            up[4][1],
            up_compensation,
            down[4][0],
            down[4][1].unsqueeze(1),
            down_compensation,
            group_list_type,
            BETA,
            LINEAR_BETA,
        )

    fused()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fused()
    for counts in routes:
        groups.copy_(_groups(counts, torch.int64, group_list_type))
        graph.replay()
        rows = sum(counts)
        if rows:
            q, scale = _call(x, x_scale, up, groups, group_list_type, True, up_compensation, stacked=True)
            reference = torch_npu.npu_grouped_matmul(
                x=[q],
                weight=[down[4][0]],
                scale=[down[4][1].unsqueeze(1)],
                bias=[down_compensation],
                per_token_scale=[scale],
                split_item=2,
                group_type=0,
                group_list=_groups(counts, torch.int64, 1).npu(),
                group_list_type=1,
                output_dtype=torch.bfloat16,
            )[0]
            torch.testing.assert_close(output[:rows].cpu(), reference[:rows].cpu(), rtol=0, atol=0)
