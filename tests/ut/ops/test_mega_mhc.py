# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import math
import weakref
from dataclasses import replace

import pytest
import torch

from vllm_ascend.ops.mhc import (
    A2OverlapMhcBackend,
    A3DecoupledMhcBackend,
    A3MegaMhcBackend,
    A5NativeMhcBackend,
    MhcHardwareCapability,
    MhcSeamInputs,
    MhcSeamOutputs,
    MhcSeamSpec,
    MhcWeightLifecycleManager,
    NativeFallbackMhcBackend,
    WeightFormat,
    build_c4_index_map,
    c4_weight_nbytes,
    pack_ozaki_weights,
    select_backend,
    should_prefer_side_stream,
    unpack_ozaki_weights,
)
from vllm_ascend.ops.mhc.reference import stable_sigmoid

DTYPES = frozenset((torch.bfloat16, torch.float16))
A3 = MhcHardwareCapability("Ascend910_9382", "arch22", 24, DTYPES, has_a3_fused_op=True)
A2 = MhcHardwareCapability("Ascend910B2", "arch20", 24, DTYPES, has_a2_overlap_op=True, graph_mode="full_decode_only")
A5 = MhcHardwareCapability("Ascend950", "arch35", 32, DTYPES, has_a5_native_op=True)
BF16_SEAM = MhcSeamSpec(C=4, D=5120, dtype=torch.bfloat16, T=7)
WEIGHT_BYTES = 1_966_080


def _seam_case(T: int = 3, C: int = 4, D: int = 64, dtype: torch.dtype = torch.bfloat16, device: str = "cpu"):
    generator = torch.Generator(device="cpu").manual_seed(11)

    def sample(shape: tuple[int, ...], out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu").to(out_dtype).to(device)

    k = C * D
    mix = C * (C + 2)
    inputs = MhcSeamInputs(
        y=sample((T, D), dtype),
        residual=sample((T, C, D), dtype),
        post=sample((T, C)),
        comb=sample((T, C, C)),
        pre_mix=sample((T, C)),
        gamma=sample((D,), dtype),
        hc_scale=sample((3,)),
        hc_base=sample((mix,)),
    )
    hc_fn = torch.randn((mix, k), generator=generator, device="cpu", dtype=torch.float32) / math.sqrt(k)
    return inputs, hc_fn


@pytest.mark.parametrize("fmt", [WeightFormat.OZAKI_BF16_ND_V1, WeightFormat.OZAKI_BF16_NZ_V1])
def test_ozaki_layout_roundtrip_and_memory_parity(fmt):
    k = 20_480
    gen = torch.Generator().manual_seed(17)
    source = torch.randn((k, 24), generator=gen, dtype=torch.float32).T
    hi = source.bfloat16()
    lo = (source - hi.float()).bfloat16()
    packed = pack_ozaki_weights(source, fmt)
    assert packed.is_contiguous()
    if fmt == WeightFormat.OZAKI_BF16_NZ_V1:
        assert packed.shape == (k // 16, 48, 16)
        torch.testing.assert_close(packed[27, 33, 9], lo[9, 27 * 16 + 9], rtol=0, atol=0)
        logical_nd = packed.transpose(0, 1).reshape(48, k)
    else:
        assert packed.shape == (48, k)
        logical_nd = packed
    assert torch.equal(logical_nd[:24], hi)
    assert torch.equal(logical_nd[24:], lo)
    reconstructed = unpack_ozaki_weights(packed, fmt)
    assert torch.equal(reconstructed, hi.float() + lo.float())
    torch.testing.assert_close(reconstructed, source, rtol=2**-16, atol=0)
    assert packed.nbytes == source.nbytes == WEIGHT_BYTES
    assert c4_weight_nbytes(k, fmt) == c4_weight_nbytes(k, WeightFormat.FP32_CANONICAL) == WEIGHT_BYTES


def test_ozaki_rejects_invalid_shapes_and_nz_alignment():
    source = torch.zeros(24, 17)
    with pytest.raises(ValueError, match="NZ packing"):
        pack_ozaki_weights(source)
    with pytest.raises(ValueError, match="float32"):
        pack_ozaki_weights(source.bfloat16())
    assert pack_ozaki_weights(source, WeightFormat.OZAKI_BF16_ND_V1).shape == (48, 17)


def test_c4_index_map_gathers_coefficient_layout():
    offsets = build_c4_index_map()
    assert (offsets.dtype, offsets.shape, offsets.nbytes) == (torch.int32, (408,), 1632)
    assert torch.all((offsets >= 0) & (offsets % 4 == 0) & (offsets < 192 * 1024))
    reduction, sig, comb, scale1, scale2, post = (offsets // 4).split((256, 32, 64, 8, 16, 32))
    matrix = torch.arange(64).reshape(4, 4, 4)
    expected = (
        matrix[:, :, [1, 0, 3, 2]],
        matrix[:, :, [2, 3, 0, 1]],
        matrix[:, [1, 0, 3, 2], :],
        matrix[:, [2, 3, 0, 1], :],
    )
    for indices, permuted in zip(reduction.split(64), expected, strict=True):
        assert torch.equal(matrix.flatten()[indices], permuted.flatten())
    logits = torch.arange(256).reshape(4, 64)
    assert torch.equal(logits.flatten()[sig], logits[:, :8].flatten())
    assert torch.equal(logits.flatten()[comb], logits[:, 8:24].flatten())
    assert scale1.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert scale2.tolist() == [2] * 16
    sigmoid = torch.arange(32).reshape(4, 8)
    assert torch.equal(sigmoid.flatten()[post].reshape(4, 8), sigmoid[:, 4:].repeat(1, 2))


def test_stable_sigmoid_boundary():
    x = torch.tensor([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=torch.float32)
    value = stable_sigmoid(x)
    assert torch.isfinite(value).all()
    torch.testing.assert_close(value, torch.sigmoid(x.double()).float(), rtol=0, atol=2**-149)


@pytest.mark.parametrize(
    ("spec", "cap", "expected"),
    [
        pytest.param(BF16_SEAM, A3, A3MegaMhcBackend, id="a3-qualified"),
        pytest.param(replace(BF16_SEAM, D=16_320), A3, A3MegaMhcBackend, id="a3-max-width"),
        pytest.param(replace(BF16_SEAM, D=16_384), A3, NativeFallbackMhcBackend, id="a3-width-limit"),
        pytest.param(replace(BF16_SEAM, D=5121), A3, NativeFallbackMhcBackend, id="a3-misalignment"),
        pytest.param(replace(BF16_SEAM, C=3), A3, NativeFallbackMhcBackend, id="a3-c3"),
        pytest.param(replace(BF16_SEAM, dtype=torch.float16), A3, NativeFallbackMhcBackend, id="a3-fp16"),
        pytest.param(replace(BF16_SEAM, is_layer0_entry=True), A3, NativeFallbackMhcBackend, id="layer0"),
        pytest.param(replace(BF16_SEAM, is_write_kv_path=True), A3, NativeFallbackMhcBackend, id="write-kv"),
        pytest.param(
            replace(BF16_SEAM, is_write_kv_path=True, write_kv_migrated=True),
            A3,
            A3MegaMhcBackend,
            id="migrated-write-kv",
        ),
        pytest.param(BF16_SEAM, replace(A3, aic_count=4), NativeFallbackMhcBackend, id="a3-cores-too-few"),
        pytest.param(
            BF16_SEAM,
            replace(A3, soc_name="Ascend910_9372", aic_count=20),
            A3MegaMhcBackend,
            id="a3-20aic-9372-qualified",
        ),
        pytest.param(
            BF16_SEAM,
            replace(A3, soc_name="Ascend910B2"),
            NativeFallbackMhcBackend,
            id="a3-rejects-a2-24aic-soc",
        ),
        pytest.param(replace(BF16_SEAM, T=2**33), A3, NativeFallbackMhcBackend, id="a3-t-uint32-limit"),
        pytest.param(
            BF16_SEAM,
            replace(A3, graph_mode="FULL_DECODE_ONLY"),
            A3MegaMhcBackend,
            id="a3-uppercase-graph-mode",
        ),
        pytest.param(
            BF16_SEAM,
            replace(A2, graph_mode="FULL_DECODE_ONLY"),
            A2OverlapMhcBackend,
            id="a2-uppercase-graph-mode",
        ),
        pytest.param(BF16_SEAM, replace(A3, has_a3_fused_op=False), NativeFallbackMhcBackend, id="a3-no-op"),
        pytest.param(BF16_SEAM, replace(A3, superkernel_enabled=True), NativeFallbackMhcBackend, id="a3-superkernel"),
        pytest.param(
            replace(BF16_SEAM, T=6, prefer_side_stream=True),
            replace(A3, has_a3_decoupled_op=True, superkernel_enabled=False),
            A3DecoupledMhcBackend,
            id="a3-decoupled-qualified",
        ),
        pytest.param(
            replace(BF16_SEAM, T=6, prefer_side_stream=True),
            replace(A3, has_a3_decoupled_op=False),
            A3MegaMhcBackend,
            id="a3-decoupled-falls-back-to-fused-when-split-op-missing",
        ),
        pytest.param(
            replace(BF16_SEAM, T=6, prefer_side_stream=True),
            replace(A3, has_a3_decoupled_op=True, superkernel_enabled=True),
            NativeFallbackMhcBackend,
            id="a3-decoupled-rejected-under-superkernel",
        ),
        pytest.param(
            replace(BF16_SEAM, sp_enabled=True),
            replace(A3, graph_mode="aclgraph"),
            A3MegaMhcBackend,
            id="a3-local-sp-seam",
        ),
        pytest.param(replace(BF16_SEAM, T=72), A2, A2OverlapMhcBackend, id="a2-qualified"),
        pytest.param(replace(BF16_SEAM, T=73), A2, NativeFallbackMhcBackend, id="a2-row-limit"),
        pytest.param(replace(BF16_SEAM, sp_enabled=True), A2, NativeFallbackMhcBackend, id="a2-sp"),
        pytest.param(replace(BF16_SEAM, post_mult=1.5), A2, NativeFallbackMhcBackend, id="a2-post-mult"),
        pytest.param(BF16_SEAM, replace(A2, graph_mode="eager"), NativeFallbackMhcBackend, id="a2-runner-mode"),
        pytest.param(BF16_SEAM, replace(A2, soc_name="Ascend310P"), NativeFallbackMhcBackend, id="a2-wrong-soc"),
        pytest.param(BF16_SEAM, A5, A5NativeMhcBackend, id="a5-qualified"),
        pytest.param(BF16_SEAM, replace(A5, has_a5_native_op=False), NativeFallbackMhcBackend, id="a5-gated"),
        pytest.param(BF16_SEAM, replace(A3, has_a5_native_op=True), A3MegaMhcBackend, id="a5-wrong-arch"),
    ],
)
def test_backend_dispatch(spec, cap, expected):
    backend = select_backend(spec, cap)
    assert isinstance(backend, expected)
    supported, reason = backend.supports(spec, cap)
    assert supported and reason


@pytest.mark.parametrize(
    ("name", "bad_tensor", "message"),
    [
        ("comb", torch.empty(3, 4, 3), "comb must have shape"),
        ("gamma", torch.empty(64), "gamma must have dtype"),
        ("pre_mix", torch.empty(3, 4, device="meta"), "pre_mix must be on"),
    ],
)
def test_invalid_input_is_an_error_before_launch(name, bad_tensor, message):
    inputs, hc_fn = _seam_case()
    backend = select_backend(replace(BF16_SEAM, D=64), A3)
    prepared = backend.prepare({"hc_fn": hc_fn, "version_tag": "checkpoint"}, "cpu", A3)
    with pytest.raises(ValueError, match=message):
        backend.run(replace(inputs, **{name: bad_tensor}), prepared, {})


def test_native_fallback_runs_unaligned_fp16_seam():
    inputs, hc_fn = _seam_case(C=3, D=63, dtype=torch.float16)
    backend = select_backend(replace(BF16_SEAM, C=3, D=63, dtype=torch.float16), A3)
    prepared = backend.prepare({"hc_fn": hc_fn, "version_tag": "checkpoint"}, "cpu", A3)
    out = backend.run(inputs, prepared, {"return_v_fp32": True})
    assert out.v.dtype == torch.float16 and out.v.shape == inputs.y.shape
    assert torch.equal(out.v_fp32, out.v.bfloat16().float())
    assert torch.all(out.v_fp32.view(torch.int32) & 0xFFFF == 0)
    assert out.comb.shape == (3, 3, 3) and torch.isfinite(out.comb).all()


def test_a2_forwards_to_the_existing_overlap_graph(monkeypatch):
    inputs, hc_fn = _seam_case(T=7, D=5120)
    backend = select_backend(BF16_SEAM, A2)
    prepared = backend.prepare({"hc_fn": hc_fn, "version_tag": "checkpoint"}, "cpu", A2)

    def moe(*args):
        assert args[6] is hc_fn and args[-1] is True
        return inputs.residual, inputs.y, inputs.y.float(), inputs.post, inputs.comb, inputs.pre_mix

    monkeypatch.setattr(torch.ops.vllm, "dsv41_mhc_moe_input", moe, raising=False)
    result = backend.run(inputs, prepared, {"return_v_fp32": True, "wait_for_attention": True})
    assert result.v is inputs.y and result.pre is inputs.pre_mix


def test_a5_uses_runner_supplied_semantic_adapter():
    inputs, hc_fn = _seam_case()
    backend = select_backend(replace(BF16_SEAM, D=64), A5)
    expected = MhcSeamOutputs(inputs.residual, inputs.y, None, inputs.post, inputs.comb, inputs.pre_mix)

    def run_native(seam, weight, attrs):
        assert seam is inputs and weight is hc_fn and attrs == {}
        return expected

    prepared = backend.prepare({"hc_fn": hc_fn, "version_tag": "v1", "native_adapter": run_native}, "cpu", A5)
    assert backend.run(inputs, prepared, {}) is expected


@pytest.mark.parametrize(
    ("coefficient_time", "extra", "contention", "delta_c", "expected"),
    [
        (13, 3, 2, 0, True),
        (19, 3, 2, 0, True),
        (19, 4, 2, 0, False),
        (19, 5, 2, 0, False),
        (12, 2, 11, 0, False),
        (18, 2, 1, 3, False),
        (16, 2, 1, 2, True),
    ],
)
def test_side_stream_cost_model_boundary(coefficient_time, extra, contention, delta_c, expected):
    assert should_prefer_side_stream(21, 10, 12, coefficient_time, extra, contention, delta_c) is expected


def test_decoupled_backend_forwards_to_split_ops(monkeypatch):
    inputs, hc_fn = _seam_case(T=6, D=64)
    cap = replace(A3, has_a3_decoupled_op=True)
    spec = replace(BF16_SEAM, D=64, T=6, prefer_side_stream=True)
    backend = select_backend(spec, cap)
    assert isinstance(backend, A3DecoupledMhcBackend)
    prepared = replace(backend.prepare({"hc_fn": hc_fn, "version_tag": "v1"}, "cpu", cap), index_map_ready=True)
    r_sq = torch.ones(6, dtype=torch.float32)
    calls = []

    def fake_main(y, res, po, co, pr, g, *, norm_eps, return_v_fp32, return_r_sq=False):
        calls.append(("main", return_v_fp32, return_r_sq))
        return res, y, y.float() if return_v_fp32 else None, (r_sq if return_r_sq else None)

    def fake_coeff(res, pw, hs, hb, *, r_sq, weight_format, numerics_profile, norm_eps, hc_eps, iterations, post_mult):
        calls.append(("coeff", r_sq is not None, weight_format))
        return inputs.post, inputs.comb, inputs.pre_mix

    monkeypatch.setattr(torch.ops._C_ascend, "npu_mega_mhc_main", fake_main, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_mhc_coefficients_ozaki", fake_coeff, raising=False)
    out = backend.run(inputs, prepared, {"return_v_fp32": True})
    assert calls == [("main", True, False), ("coeff", False, int(WeightFormat.OZAKI_BF16_NZ_V1))]
    assert out.v_fp32 is not None and out.residual is inputs.residual

    calls.clear()
    r_o, v_o, vf_o, rsq_o = backend.run_main(inputs, prepared, {"return_v_fp32": False}, return_r_sq=False)
    assert vf_o is None and rsq_o is None
    po, co, pr = backend.run_coefficients(r_o, prepared, inputs.hc_scale, inputs.hc_base, {}, r_sq=None)
    assert calls == [("main", False, False), ("coeff", False, int(WeightFormat.OZAKI_BF16_NZ_V1))]
    assert po is inputs.post and co is inputs.comb and pr is inputs.pre_mix


def test_staging_bounds_and_preserves_layer0_and_write_kv():
    manager = MhcWeightLifecycleManager("cpu", A3)

    def load():
        return torch.zeros(24, 20_480)

    keys = ("l0", "l1", "l2", "l3")
    owners = (("layer0_entry",), ("write_kv_l1",), ("write_kv_l2",), ())
    refs = []
    for i, (key, extra_callers) in enumerate(zip(keys, owners, strict=True)):
        source = load()
        refs.append(weakref.ref(source))
        seam = f"seam_l{i}"
        manager.register_layer(key, (seam, *extra_callers), load_canonical=load, version_tag="v1", canonical=source)
        manager.mark_consumer_migrated(key, seam)
    del source

    for i, key in enumerate(keys[:3]):
        backend, prepared = manager.prepare_layer(key, BF16_SEAM)
        caller = owners[i][0]
        assert isinstance(backend, NativeFallbackMhcBackend)
        assert manager.fp32_consumers(key) == {caller}
        assert prepared.tensor is manager.canonical_for(key, caller) is refs[i]()
        assert not prepared.source_released
    del prepared
    assert manager.peak_staged_weight_bytes == 0

    _, prepared = manager.prepare_layer("l3", BF16_SEAM)
    assert prepared.source_released and refs[3]() is None
    assert prepared.tensor.nbytes == WEIGHT_BYTES
    manager.mark_consumer_migrated("l1", "write_kv_l1")
    _, prepared = manager.prepare_layer("l1", BF16_SEAM)
    assert prepared.source_released and refs[1]() is None
    assert refs[0]() is not None and refs[2]() is not None
    assert manager.peak_staged_weight_bytes == WEIGHT_BYTES


def test_lifecycle_refreshes_weights_and_constants_after_graph_retirement(monkeypatch):
    events = []
    prepare = A3MegaMhcBackend.prepare

    def with_constant(self, weights, device, cap):
        events.append("prepare constant")
        return replace(prepare(self, weights, device, cap), index_map_ready=True)

    monkeypatch.setattr(A3MegaMhcBackend, "prepare", with_constant)
    manager = MhcWeightLifecycleManager(
        "cpu",
        A3,
        retire_graphs=lambda: events.append("retire"),
        invalidate_device_constants=lambda device: events.append(f"invalidate {device}"),
    )
    spec = replace(BF16_SEAM, D=64)
    manager.register_layer("l0", ("seam_l0",), version_tag="v1", load_canonical=lambda: torch.zeros(24, 256))
    manager.mark_consumer_migrated("l0", "seam_l0")
    _, first = manager.prepare_layer("l0", spec)
    assert manager.prepare_layer("l0", spec)[1] is first
    old_graph = manager.generation
    manager.validate_graph(old_graph)

    manager.on_sleep()
    assert events == ["prepare constant", "retire", "invalidate cpu"]
    with pytest.raises(RuntimeError, match="asleep"):
        manager.prepare_layer("l0", spec)
    manager.on_wake()
    with pytest.raises(RuntimeError, match="recaptured"):
        manager.validate_graph(old_graph)
    _, awake = manager.prepare_layer("l0", spec)
    assert awake is not first and awake.index_map_ready
    assert torch.equal(awake.tensor, first.tensor)

    manager.on_weight_update(
        "l0", canonical=torch.ones(24, 256), version_tag="v2", load_canonical=lambda: torch.ones(24, 256)
    )
    _, updated = manager.prepare_layer("l0", spec)
    assert updated.version_tag == "v2"
    assert torch.equal(unpack_ozaki_weights(updated.tensor), torch.ones(24, 256))

    manager.on_device_move("meta", replace(A3, has_a3_fused_op=False))
    assert events[-2:] == ["retire", "invalidate cpu"]
    _, moved = manager.prepare_layer("l0", spec)
    assert moved.device == torch.device("meta") and moved.version_tag == "v2"
    assert moved.weight_format == WeightFormat.FP32_CANONICAL and not moved.index_map_ready


def test_canonical_layer_releases_storage_on_sleep_and_device_move():
    reloads = {"l0": 0, "l1": 0}

    def make_loader(key: str):
        def _load() -> torch.Tensor:
            reloads[key] += 1
            return torch.full((24, 256), float(reloads[key]), dtype=torch.float32)

        return _load

    manager = MhcWeightLifecycleManager("cpu", A3)
    spec = replace(BF16_SEAM, D=64)
    l0_src = torch.zeros(24, 256, dtype=torch.float32)
    l1_src = torch.zeros(24, 256, dtype=torch.float32)
    l0_ref = weakref.ref(l0_src)
    l1_ref = weakref.ref(l1_src)

    manager.register_layer(
        "l0", ("seam_l0", "layer0_entry"), load_canonical=make_loader("l0"), version_tag="v1", canonical=l0_src
    )
    manager.register_layer(
        "l1", ("seam_l1", "write_kv_l1"), load_canonical=make_loader("l1"), version_tag="v1", canonical=l1_src
    )
    manager.mark_consumer_migrated("l0", "seam_l0")
    manager.mark_consumer_migrated("l1", "seam_l1")
    del l0_src, l1_src

    _, p0 = manager.prepare_layer("l0", spec)
    _, p1 = manager.prepare_layer("l1", spec)
    assert p0.tensor is l0_ref() and p1.tensor is l1_ref()
    del p0, p1

    manager.on_sleep()
    assert l0_ref() is None and l1_ref() is None

    manager.on_wake()
    reloaded = manager.canonical_for("l0", "layer0_entry")
    reloaded_ref = weakref.ref(reloaded)
    assert reloads["l0"] == 1 and torch.equal(reloaded, torch.ones(24, 256))
    del reloaded

    manager.on_device_move("meta", replace(A3, has_a3_fused_op=False))
    assert reloaded_ref() is None
    _, moved = manager.prepare_layer("l0", spec)
    assert moved.device == torch.device("meta") and reloads["l0"] == 2


def test_is_mega_mhc_supported_tracks_custom_op_registration(monkeypatch):
    from vllm_ascend.ascend_config import is_mega_mhc_supported

    class _OpsWithout:
        pass

    class _OpsWith:
        npu_mega_mhc_seam = object()

    monkeypatch.setattr(torch.ops, "_C_ascend", _OpsWithout(), raising=False)
    assert is_mega_mhc_supported() is False
    monkeypatch.setattr(torch.ops, "_C_ascend", _OpsWith(), raising=False)
    assert is_mega_mhc_supported() is True
