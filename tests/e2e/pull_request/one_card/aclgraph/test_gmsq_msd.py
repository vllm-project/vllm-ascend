# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded A3 MSD GMSQ correctness, native NZ, and graph regressions."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend.vllm_ascendC")

BETA = 4.0
LINEAR_BETA = 25.0


@pytest.fixture(autouse=True)
def require_a3():
    if not torch.npu.is_available():
        pytest.skip("GMSQ MSD requires an NPU")
    if not 250 <= torch_npu.npu.get_soc_version() <= 256:
        pytest.skip("GMSQ MSD regression requires Ascend A3")
    if not hasattr(torch.vllm_ascendC, "grouped_matmul_situ_quant"):
        pytest.skip("GMSQ custom operator is not built")
    previous = torch.npu.config.allow_internal_format
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
    weights = list(packed.unbind())
    if experts > 1:
        assert weights[1].storage_offset() > 0
    assert all(torch_npu.get_npu_format(weight) == (29 if nz else 2) for weight in weights)
    encoded = torch.stack([scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF for scale in scales]).npu()
    return weights, list(encoded.unbind()), logical, scales


def _reference(x, x_scale, layer, counts):
    outputs, scales, activations = [], [], []
    offset = 0
    for weight, weight_scale, rows in zip(layer[2], layer[3], counts):
        if rows:
            # Bounded integers keep CPU FP32 accumulation exact. The A3 MSD
            # path rounds the weight-scaled accumulator to FP16 before SiTU.
            acc = x[offset : offset + rows].float() @ weight.float()
            hidden = (acc * weight_scale).half().float() * x_scale[offset : offset + rows, None]
            gate, up = hidden.chunk(2, dim=-1)
            gate = ((2 * torch.sigmoid(gate * (2 / BETA)) - 1) * torch.sigmoid(gate)) * BETA
            up = (2 * torch.sigmoid(up * (2 / LINEAR_BETA)) - 1) * LINEAR_BETA
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


def _call(x, x_scale, layer, group_list, group_list_type):
    return torch.vllm_ascendC.grouped_matmul_situ_quant(
        x, layer[0], layer[1], x_scale, group_list, [], BETA, LINEAR_BETA, group_list_type
    )


def _groups(counts, dtype, group_list_type):
    groups = torch.tensor(counts, dtype=dtype)
    return groups.cumsum(0) if group_list_type == 0 else groups


@pytest.mark.parametrize("nz", [False, True], ids=["nd", "native_nz"])
@pytest.mark.parametrize("experts,k,n", [(2, 320, 768), (3, 576, 1280)])
@pytest.mark.parametrize("group_list_type", [0, 1])
def test_gmsq_msd_eager(nz, experts, k, n, group_list_type):
    capacity = 8
    generator = torch.Generator().manual_seed(71)
    x = torch.randint(-32, 32, (capacity, k), generator=generator, dtype=torch.int8)
    x[0].zero_()  # Exercise zero activation and zero quantization scale.
    x_scale = torch.rand(capacity, generator=generator) * 0.01 + 0.001
    counts = [3, 0] + ([2] if experts == 3 else [])
    layer = _layer(experts, k, n, nz, 101)
    groups = _groups(counts, torch.int64, group_list_type).npu()
    output = _call(x.npu(), x_scale.npu(), layer, groups, group_list_type)
    _check(output, _reference(x, x_scale, layer, counts), capacity)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
@pytest.mark.parametrize("group_list_type", [0, 1])
def test_gmsq_msd_graph_dynamic_groups(dtype, group_list_type):
    # Distinct shapes, expert counts and layouts share one capture. Each call
    # must retain its own scratch storage and device-side routing metadata.
    cases = []
    for index, (experts, k, n, nz) in enumerate([(2, 320, 768, False), (3, 576, 1280, True)]):
        generator = torch.Generator().manual_seed(81 + index)
        x = torch.randint(-32, 32, (8, k), generator=generator, dtype=torch.int8)
        x_scale = torch.rand(8, generator=generator) * 0.01 + 0.001
        layer = _layer(experts, k, n, nz, 201 + index)
        groups = _groups([1] * experts, dtype, group_list_type).npu()
        cases.append((x, x_scale, layer, x.npu(), x_scale.npu(), groups))
    for _, _, layer, x_device, scale_device, groups in cases:
        _call(x_device, scale_device, layer, groups, group_list_type)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="global"):
        outputs = [_call(xd, sd, layer, groups, group_list_type) for _, _, layer, xd, sd, groups in cases]
    for routes in [([3, 0], [0, 2, 3]), ([0, 4], [2, 0, 1]), ([0, 0], [0, 0, 0]), ([1, 2], [1, 2, 1])]:
        for case, counts in zip(cases, routes):
            case[-1].copy_(_groups(counts, dtype, group_list_type))
        graph.replay()
        torch.npu.synchronize()
        for output, (x, x_scale, layer, *_), counts in zip(outputs, cases, routes):
            _check(output, _reference(x, x_scale, layer, counts), x.shape[0])


@pytest.mark.parametrize("nz", [False, True])
def test_gmsq_msd_zero_capacity(nz):
    layer = _layer(2, 320, 768, nz, 301)
    x = torch.empty((0, 320), dtype=torch.int8)
    scale = torch.empty(0)
    counts = [0, 0]
    output = _call(x.npu(), scale.npu(), layer, torch.tensor(counts, device="npu"), 1)
    _check(output, _reference(x, scale, layer, counts), 0)


@pytest.mark.parametrize("invalid", ["mixed", "descriptor", "noncontiguous", "expert_limit"])
def test_gmsq_msd_rejects_invalid_weights(invalid):
    k, n = 320, 768
    nd = torch.zeros((k, n // 8), dtype=torch.int32, device="npu")
    native = torch_npu.npu_format_cast(torch.zeros((k, n // 2), dtype=torch.int8, device="npu"), 29)
    nz = native.view(torch.int32)
    if invalid == "mixed":
        weights, message = [nd, nz], "mixed"
    elif invalid == "descriptor":
        wrong = torch_npu.npu_format_cast(torch.zeros((k, n // 4), dtype=torch.float16, device="npu"), 29)
        weights, message = [wrong.view(torch.int32)], "native"
    elif invalid == "noncontiguous":
        # A square carrier preserves shape under transpose, so this reaches
        # the stride guard rather than failing the shape check first.
        k, n = 64, 512
        square = torch_npu.npu_format_cast(torch.zeros((k, n // 2), dtype=torch.int8, device="npu"), 29)
        weights, message = [square.view(torch.int32).transpose(0, 1)], "contiguous"
    else:
        weights, message = [nd] * 129, "at most 128"
    scale = torch.ones(n).view(torch.int32).to(torch.int64).npu()
    layer = weights, [scale] * len(weights)
    counts = torch.tensor([1] + [0] * (len(weights) - 1), dtype=torch.int64, device="npu")
    with pytest.raises(RuntimeError, match=message):
        _call(torch.ones((1, k), dtype=torch.int8, device="npu"), torch.ones(1, device="npu"), layer, counts, 1)


def test_gmsq_msd_rechecks_nd_strides_after_warmup():
    layer = _layer(1, 64, 512, False, 401)
    x = torch.ones((1, 64), dtype=torch.int8, device="npu")
    scale = torch.ones(1, device="npu")
    counts = torch.ones(1, dtype=torch.int64, device="npu")
    _call(x, scale, layer, counts, 1)
    torch.npu.synchronize()
    original = layer[0][0]
    layer[0][0] = original.transpose(0, 1)
    assert layer[0][0].shape == original.shape
    assert layer[0][0].data_ptr() == original.data_ptr()
    with pytest.raises(RuntimeError, match="contiguous"):
        _call(x, scale, layer, counts, 1)
