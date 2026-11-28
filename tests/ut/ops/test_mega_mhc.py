# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from __future__ import annotations

import math
import weakref
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

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
    assert (offsets.dtype, offsets.shape, offsets.nbytes) == (torch.int32, (304,), 1216)
    assert torch.all((offsets >= 0) & (offsets % 4 == 0) & (offsets < 192 * 1024))
    sig, comb, base_comb, scale_sig, scale_comb, post, unpack_comb = (offsets // 4).split((32, 128, 32, 8, 8, 32, 64))
    logits = torch.arange(256).reshape(4, 64)
    assert torch.equal(logits.flatten()[sig], logits[:, :8].flatten())
    gathered_comb = logits.flatten()[comb].reshape(4, 4, 8)
    assert torch.equal(gathered_comb[:, :, :4], logits[:, 8:24].reshape(4, 4, 4))
    assert torch.all(gathered_comb[:, :, 4:] == logits[0, 0])
    params = torch.arange(40)
    expanded_base = params[base_comb].reshape(4, 8)
    expected_base = params[24:40].reshape(4, 4)
    assert torch.equal(expanded_base[:, :4], expected_base)
    assert torch.equal(expanded_base[:, 4:], expected_base[:, 3:4].expand(4, 4))
    assert scale_sig.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert scale_comb.tolist() == [2] * 8
    sigmoid = torch.arange(32).reshape(4, 8)
    assert torch.equal(sigmoid.flatten()[post].reshape(4, 8), sigmoid[:, 4:].repeat(1, 2))
    padded_comb = torch.arange(128).reshape(4, 4, 8)
    assert torch.equal(padded_comb.flatten()[unpack_comb].reshape(4, 4, 4), padded_comb[:, :, :4])


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
            replace(A3, soc_name="ascend910_9382"),
            A3MegaMhcBackend,
            id="a3-lowercase-9382-soc",
        ),
        pytest.param(
            BF16_SEAM,
            replace(A3, soc_name="ascend910_9372", aic_count=20),
            A3MegaMhcBackend,
            id="a3-lowercase-9372-soc",
        ),
        pytest.param(
            BF16_SEAM,
            replace(A3, soc_name="ascend910c"),
            A3MegaMhcBackend,
            id="a3-lowercase-910c-soc",
        ),
        pytest.param(
            BF16_SEAM,
            replace(A3, graph_mode="full"),
            A3MegaMhcBackend,
            id="a3-fused-full-graph-mode",
        ),
        pytest.param(
            BF16_SEAM,
            replace(A3, graph_mode="piecewise"),
            A3MegaMhcBackend,
            id="a3-fused-piecewise-graph-mode",
        ),
        pytest.param(
            BF16_SEAM,
            replace(A3, graph_mode="full_and_piecewise"),
            A3MegaMhcBackend,
            id="a3-fused-full-and-piecewise-graph-mode",
        ),
        pytest.param(
            replace(BF16_SEAM, T=6, prefer_side_stream=True),
            replace(A3, has_a3_decoupled_op=True, graph_mode="full"),
            A3DecoupledMhcBackend,
            id="a3-decoupled-full-graph-mode",
        ),
        pytest.param(
            replace(BF16_SEAM, T=6, prefer_side_stream=True),
            replace(A3, has_a3_decoupled_op=True, graph_mode="piecewise"),
            A3MegaMhcBackend,
            id="a3-decoupled-rejects-piecewise-falls-back-to-fused",
        ),
        pytest.param(
            replace(BF16_SEAM, T=6, prefer_side_stream=True),
            replace(A3, has_a3_decoupled_op=True, graph_mode="full_and_piecewise"),
            A3MegaMhcBackend,
            id="a3-decoupled-rejects-full-and-piecewise-falls-back-to-fused",
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
    owners: tuple[tuple[str, ...], ...] = (("layer0_entry",), ("write_kv_l1",), ("write_kv_l2",), ())
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


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"residual": torch.empty(6, 4, 64, dtype=torch.float32)}, "residual must be bfloat16 or float16"),
        ({"residual": torch.empty(6, 3, 64, dtype=torch.bfloat16)}, "C=4"),
        (
            {"residual": torch.empty(6, 64, 4, dtype=torch.bfloat16).transpose(1, 2)},
            "residual must be contiguous",
        ),
        ({"hc_scale": torch.empty(4, dtype=torch.float32)}, "hc_scale must have shape"),
        ({"hc_base": torch.empty(24, dtype=torch.bfloat16)}, "hc_base must have dtype"),
        ({"r_sq": torch.empty(5, dtype=torch.float32)}, "r_sq must have shape"),
        ({"r_sq": torch.empty(6, dtype=torch.bfloat16)}, "r_sq must have dtype"),
        ({"r_sq": torch.empty(6, dtype=torch.float32, device="meta")}, "r_sq must be on"),
        ({"prepared_tensor": torch.empty(1, 48, 16, dtype=torch.bfloat16)}, "prepared hc_fn must have shape"),
    ],
)
def test_decoupled_run_coefficients_rejects_invalid_inputs(override, message):
    inputs, hc_fn = _seam_case(T=6, D=64)
    cap = replace(A3, has_a3_decoupled_op=True)
    backend = A3DecoupledMhcBackend()
    prepared = replace(backend.prepare({"hc_fn": hc_fn, "version_tag": "v1"}, "cpu", cap), index_map_ready=True)
    if "prepared_tensor" in override:
        prepared = replace(prepared, tensor=override["prepared_tensor"])
    kwargs = {
        "residual": inputs.residual,
        "hc_scale": inputs.hc_scale,
        "hc_base": inputs.hc_base,
        "r_sq": None,
    }
    kwargs.update({k: v for k, v in override.items() if k != "prepared_tensor"})
    with pytest.raises(ValueError, match=message):
        backend.run_coefficients(
            kwargs["residual"],
            prepared,
            kwargs["hc_scale"],
            kwargs["hc_base"],
            {},
            r_sq=kwargs["r_sq"],
        )


def test_decoupled_launch_coefficients_async_records_streams_and_handles_capture_and_errors(monkeypatch):
    inputs, hc_fn = _seam_case(T=6, D=64)
    cap = replace(A3, has_a3_decoupled_op=True)
    backend = A3DecoupledMhcBackend()
    prepared = replace(backend.prepare({"hc_fn": hc_fn, "version_tag": "v1"}, "cpu", cap), index_map_ready=True)
    r_sq = torch.ones(6, dtype=torch.float32)

    stream_events: list[tuple[str, object]] = []
    recorded_streams: list[tuple[int, object]] = []
    capturing_state = {"value": False}

    class _FakeStream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_event(self, event) -> None:
            stream_events.append(("wait_event", (self.name, event.name)))

    class _FakeEvent:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self, stream=None) -> None:
            stream_events.append(("record", (self.name, None if stream is None else stream.name)))

    main_stream = _FakeStream("main")
    side_stream = _FakeStream("side")
    ready_event = _FakeEvent("ready")
    done_event = _FakeEvent("done")

    @contextmanager
    def fake_stream_ctx(stream):
        stream_events.append(("enter_stream", stream.name))
        try:
            yield
        finally:
            stream_events.append(("exit_stream", stream.name))

    fake_npu = SimpleNamespace(
        stream=fake_stream_ctx,
        current_stream=lambda: main_stream,
        is_current_stream_capturing=lambda: capturing_state["value"],
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda self, stream: recorded_streams.append((id(self), stream.name)),
    )

    out_post = torch.ones(6, 4, dtype=torch.float32)
    out_comb = torch.ones(6, 4, 4, dtype=torch.float32)
    out_pre = torch.ones(6, 4, dtype=torch.float32)

    def fake_coeff(res, pw, hs, hb, *, r_sq, weight_format, numerics_profile, norm_eps, hc_eps, iterations, post_mult):
        return out_post, out_comb, out_pre

    monkeypatch.setattr(torch.ops._C_ascend, "npu_mhc_coefficients_ozaki", fake_coeff, raising=False)

    outputs = backend.launch_coefficients_async(
        inputs.residual,
        prepared,
        inputs.hc_scale,
        inputs.hc_base,
        {},
        side_stream=side_stream,
        ready_event=ready_event,
        done_event=done_event,
        r_sq=r_sq,
    )
    assert outputs == (out_post, out_comb, out_pre)
    assert recorded_streams == [
        (id(inputs.residual), "side"),
        (id(prepared.tensor), "side"),
        (id(inputs.hc_scale), "side"),
        (id(inputs.hc_base), "side"),
        (id(r_sq), "side"),
        (id(out_post), "main"),
        (id(out_comb), "main"),
        (id(out_pre), "main"),
    ]
    assert stream_events == [
        ("record", ("ready", None)),
        ("enter_stream", "side"),
        ("wait_event", ("side", "ready")),
        ("record", ("done", "side")),
        ("exit_stream", "side"),
    ]

    # When graph capture is active on an explicit different main_stream, capture is
    # queried on that stream and record_stream is skipped.
    alt_main_stream = _FakeStream("alt_main")
    recorded_streams.clear()
    stream_events.clear()
    capturing_state["value"] = True
    backend.launch_coefficients_async(
        inputs.residual,
        prepared,
        inputs.hc_scale,
        inputs.hc_base,
        {},
        side_stream=side_stream,
        ready_event=ready_event,
        done_event=done_event,
        main_stream=alt_main_stream,
        r_sq=r_sq,
    )
    assert recorded_streams == []
    assert stream_events == [
        ("enter_stream", "alt_main"),
        ("exit_stream", "alt_main"),
        ("record", ("ready", "alt_main")),
        ("enter_stream", "side"),
        ("wait_event", ("side", "ready")),
        ("record", ("done", "side")),
        ("exit_stream", "side"),
    ]

    # Invalid inputs raise ValueError BEFORE touching ready_event, side_stream, or done_event.
    recorded_streams.clear()
    stream_events.clear()
    capturing_state["value"] = False
    with pytest.raises(ValueError, match="hc_scale must have shape"):
        backend.launch_coefficients_async(
            inputs.residual,
            prepared,
            torch.empty(2, dtype=torch.float32),
            inputs.hc_base,
            {},
            side_stream=side_stream,
            ready_event=ready_event,
            done_event=done_event,
            main_stream=main_stream,
        )
    assert stream_events == []
    assert recorded_streams == []

    with pytest.raises(ValueError, match="device constants must be prepared"):
        backend.launch_coefficients_async(
            inputs.residual,
            replace(prepared, index_map_ready=False),
            inputs.hc_scale,
            inputs.hc_base,
            {},
            side_stream=side_stream,
            ready_event=ready_event,
            done_event=done_event,
            main_stream=main_stream,
        )
    assert stream_events == []

    with pytest.raises(ValueError, match="Invalid mHC attributes"):
        backend.launch_coefficients_async(
            inputs.residual,
            prepared,
            inputs.hc_scale,
            inputs.hc_base,
            {"unknown_attr": 1},
            side_stream=side_stream,
            ready_event=ready_event,
            done_event=done_event,
            main_stream=main_stream,
        )
    assert stream_events == []

    # On runtime failure inside run_coefficients, done_event is still recorded in finally:.
    stream_events.clear()
    recorded_streams.clear()

    def failing_coeff(*args, **kwargs):
        raise RuntimeError("kernel launch failed")

    monkeypatch.setattr(torch.ops._C_ascend, "npu_mhc_coefficients_ozaki", failing_coeff, raising=False)
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        backend.launch_coefficients_async(
            inputs.residual,
            prepared,
            inputs.hc_scale,
            inputs.hc_base,
            {},
            side_stream=side_stream,
            ready_event=ready_event,
            done_event=done_event,
            main_stream=main_stream,
        )
    assert stream_events == [
        ("record", ("ready", "main")),
        ("enter_stream", "side"),
        ("wait_event", ("side", "ready")),
        ("record", ("done", "side")),
        ("exit_stream", "side"),
    ]

    # On runtime failure inside record_stream (after wait_event), done_event is STILL recorded.
    stream_events.clear()

    def failing_record_stream(self, stream):
        raise RuntimeError("record_stream failed")

    monkeypatch.setattr(torch.Tensor, "record_stream", failing_record_stream)
    with pytest.raises(RuntimeError, match="record_stream failed"):
        backend.launch_coefficients_async(
            inputs.residual,
            prepared,
            inputs.hc_scale,
            inputs.hc_base,
            {},
            side_stream=side_stream,
            ready_event=ready_event,
            done_event=done_event,
            main_stream=main_stream,
        )
    assert stream_events == [
        ("record", ("ready", "main")),
        ("enter_stream", "side"),
        ("wait_event", ("side", "ready")),
        ("record", ("done", "side")),
        ("exit_stream", "side"),
    ]


def test_native_fallback_npu_without_rms_norm_cast_falls_back_to_npu_rms_norm(monkeypatch):
    inputs, hc_fn = _seam_case(T=4, C=4, D=64, dtype=torch.bfloat16)
    backend = NativeFallbackMhcBackend()
    prepared = backend.prepare({"hc_fn": hc_fn, "version_tag": "v1"}, "cpu", A5)
    calls: list[str] = []

    class _AscendOpsNoCast:
        @staticmethod
        def npu_hc_post(y, residual, post, comb):
            calls.append("npu_hc_post")
            return residual

        @staticmethod
        def npu_hc_pre_v3(updated, weight, hc_scale, hc_base, pre_mix, **kwargs):
            calls.append("npu_hc_pre_v3")
            return updated[:, 0, :], inputs.post, inputs.comb, inputs.pre_mix

    class _NpuOps:
        @staticmethod
        def npu_rms_norm(collapsed, gamma, norm_eps):
            calls.append("npu_rms_norm")
            return collapsed.to(torch.bfloat16), None

    monkeypatch.setattr(MhcSeamInputs, "device", property(lambda self: SimpleNamespace(type="npu")))
    monkeypatch.setattr(torch.ops, "_C_ascend", _AscendOpsNoCast(), raising=False)
    monkeypatch.setattr(torch.ops, "npu", _NpuOps(), raising=False)

    out = backend.run(inputs, prepared, {"return_v_fp32": True})
    assert calls == ["npu_hc_post", "npu_hc_pre_v3", "npu_rms_norm"]
    assert out.v_fp32 is not None
    assert torch.equal(out.v_fp32, out.v.bfloat16().float())


def test_lifecycle_backend_switch_between_a3_fused_and_decoupled_reuses_weights_without_retiring():
    events: list[str] = []
    loads = {"count": 0}

    def load():
        loads["count"] += 1
        return torch.ones(24, 256, dtype=torch.float32)

    cap = replace(A3, has_a3_decoupled_op=True)
    manager = MhcWeightLifecycleManager("cpu", cap, retire_graphs=lambda: events.append("retire"))
    manager.register_layer("l0", ("seam_l0",), version_tag="v1", load_canonical=load)
    manager.mark_consumer_migrated("l0", "seam_l0")

    fused_spec = replace(BF16_SEAM, D=64, prefer_side_stream=False)
    decoupled_spec = replace(BF16_SEAM, D=64, prefer_side_stream=True)

    b1, p1 = manager.prepare_layer("l0", fused_spec)
    assert isinstance(b1, A3MegaMhcBackend) and not isinstance(b1, A3DecoupledMhcBackend)
    gen0 = manager.generation
    assert loads["count"] == 1 and events == []

    # Switching from fused to decoupled reuses packed Ozaki weights without retiring graphs or repacking.
    b2, p2 = manager.prepare_layer("l0", decoupled_spec)
    assert isinstance(b2, A3DecoupledMhcBackend)
    assert p2 is p1
    assert manager.generation == gen0
    assert loads["count"] == 1 and events == []

    # Switching back from decoupled to fused also reuses packed weights without retiring graphs.
    b3, p3 = manager.prepare_layer("l0", fused_spec)
    assert isinstance(b3, A3MegaMhcBackend) and not isinstance(b3, A3DecoupledMhcBackend)
    assert p3 is p1
    assert manager.generation == gen0
    assert loads["count"] == 1 and events == []

    # Changing weight_format (NZ -> ND) must still retire graphs and repack.
    nd_spec = replace(fused_spec, weight_format=WeightFormat.OZAKI_BF16_ND_V1)
    _, p4 = manager.prepare_layer("l0", nd_spec)
    assert p4 is not p1
    assert p4.weight_format == WeightFormat.OZAKI_BF16_ND_V1
    assert manager.generation == gen0 + 1
    assert events == ["retire"] and loads["count"] == 2


def test_resolve_mhc_device_handles_unspecified_npu_index(monkeypatch):
    from vllm_ascend.ops.mhc import registry as mhc_registry

    real_torch_device = torch.device

    class _FakeNpuDevice:
        def __init__(self, dev_type: str, index: int | None = None) -> None:
            self.type = dev_type
            self.index = index

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _FakeNpuDevice) and (self.type, self.index) == (other.type, other.index)

    def fake_device(dev, index=None):
        if isinstance(dev, _FakeNpuDevice):
            return dev
        if isinstance(dev, str) and dev.startswith("npu"):
            if index is not None:
                return _FakeNpuDevice("npu", int(index))
            parts = dev.split(":", 1)
            return _FakeNpuDevice("npu", int(parts[1]) if len(parts) == 2 else None)
        return real_torch_device(dev) if index is None else real_torch_device(dev, index)

    monkeypatch.setattr(torch, "device", fake_device)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: 3), raising=False)
    assert mhc_registry.resolve_mhc_device("npu") == _FakeNpuDevice("npu", 3)
    assert mhc_registry.resolve_mhc_device("npu:1") == _FakeNpuDevice("npu", 1)

    manager = MhcWeightLifecycleManager("npu", A3)
    assert manager._device == _FakeNpuDevice("npu", 3)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_device=lambda: 5), raising=False)
    manager.on_device_move("npu", A3)
    assert manager._device == _FakeNpuDevice("npu", 5)

    monkeypatch.delattr(torch, "npu", raising=False)
    with pytest.raises(ValueError, match="explicit NPU device index"):
        mhc_registry.resolve_mhc_device("npu")


def test_torch_split_seam_vectorized_matches_loop_reference():
    inputs, hc_fn = _seam_case(T=5, C=4, D=64, dtype=torch.bfloat16)
    backend = NativeFallbackMhcBackend()
    prepared = backend.prepare({"hc_fn": hc_fn, "version_tag": "v1"}, "cpu", A3)
    out = backend.run(inputs, prepared, {"return_v_fp32": True})

    # Reference explicit loop calculation for HcPost and collapse with default norm_eps=1e-20.
    ref_updated = inputs.y.float().unsqueeze(1) * inputs.post.unsqueeze(-1)
    for lane, comb_col in zip(inputs.residual.float().unbind(1), inputs.comb.unbind(1)):
        ref_updated = ref_updated + lane.unsqueeze(1) * comb_col.unsqueeze(-1)
    ref_updated = ref_updated.to(inputs.y.dtype)

    ref_pieces = (ref_updated.float() * inputs.pre_mix.unsqueeze(-1)).unbind(1)
    ref_collapsed = ref_pieces[0]
    for piece in ref_pieces[1:]:
        ref_collapsed = ref_collapsed + piece
    ref_collapsed = ref_collapsed.to(inputs.y.dtype).float()
    ref_denom = (ref_collapsed.square().mean(-1, keepdim=True) + 1e-20).sqrt()
    ref_v = ((ref_collapsed * ref_denom.reciprocal()) * inputs.gamma.float()).to(inputs.y.dtype)

    # BF16 outputs from identical FP32 math may differ by at most 1 BF16 ULP from summation order.
    assert torch.max((out.residual.view(torch.int16) - ref_updated.view(torch.int16)).abs()).item() <= 1
    assert torch.max((out.v.view(torch.int16) - ref_v.view(torch.int16)).abs()).item() <= 1
    assert out.v_fp32 is not None and out.v_fp32.dtype == torch.float32
    torch.testing.assert_close(out.v_fp32, out.v.bfloat16().float(), atol=0, rtol=0)

    # Small-norm case where norm_eps=1e-20 vs custom norm_eps=1e-4 produces a sharp, verifiable gap.
    tiny_inputs = replace(
        inputs,
        y=torch.full_like(inputs.y, 1e-4),
        residual=torch.zeros_like(inputs.residual),
        post=torch.ones_like(inputs.post),
        comb=torch.zeros_like(inputs.comb),
        pre_mix=torch.full_like(inputs.pre_mix, 0.25),
        gamma=torch.ones_like(inputs.gamma),
    )
    out_default_eps = backend.run(tiny_inputs, prepared, {"return_v_fp32": True})
    out_large_eps = backend.run(tiny_inputs, prepared, {"return_v_fp32": True, "norm_eps": 1e-4})
    torch.testing.assert_close(
        out_default_eps.v.float(), torch.ones_like(tiny_inputs.y, dtype=torch.float32), atol=1e-5, rtol=1e-5
    )
    tiny_val = float(torch.tensor(1e-4, dtype=torch.bfloat16).float())
    expected_large = torch.full_like(
        tiny_inputs.y,
        tiny_val / math.sqrt(tiny_val**2 + 1e-4),
    )
    torch.testing.assert_close(out_large_eps.v.float(), expected_large.float(), atol=1e-5, rtol=1e-5)


def test_torch_split_seam_avoids_bmm_when_npu_hf32_enabled(monkeypatch):
    from vllm_ascend.ops.mhc import registry as mhc_registry

    inputs, hc_fn = _seam_case(T=4, C=3, D=32, dtype=torch.bfloat16)
    attrs = mhc_registry._parse_attrs({})
    ref = mhc_registry._torch_split_seam(inputs, hc_fn, attrs)

    bmm_calls = {"count": 0}
    real_bmm = torch.bmm

    def spy_bmm(a, b):
        bmm_calls["count"] += 1
        return real_bmm(a, b)

    monkeypatch.setattr(torch, "bmm", spy_bmm)

    # When NPU HF32 is enabled, _torch_split_seam uses the exact FP32 vector path without calling bmm.
    monkeypatch.setattr(mhc_registry, "_npu_hf32_enabled", lambda dev: True)
    out_vec = mhc_registry._torch_split_seam(inputs, hc_fn, attrs)
    assert bmm_calls["count"] == 0
    assert torch.max((out_vec.residual.view(torch.int16) - ref.residual.view(torch.int16)).abs()).item() <= 1
    assert torch.max((out_vec.v.view(torch.int16) - ref.v.view(torch.int16)).abs()).item() <= 1

    # When NPU HF32 is disabled, _torch_split_seam uses the fast bmm path.
    monkeypatch.setattr(mhc_registry, "_npu_hf32_enabled", lambda dev: False)
    out_bmm = mhc_registry._torch_split_seam(inputs, hc_fn, attrs)
    assert bmm_calls["count"] == 1
    torch.testing.assert_close(out_bmm.residual, ref.residual, atol=0, rtol=0)
    torch.testing.assert_close(out_bmm.v, ref.v, atol=0, rtol=0)

    # Verify _npu_hf32_enabled inspects torch.npu.matmul.allow_hf32 without mutating it.
    monkeypatch.undo()
    fake_dev = SimpleNamespace(type="npu")
    monkeypatch.setattr(torch, "npu", SimpleNamespace(matmul=SimpleNamespace(allow_hf32=True)), raising=False)
    assert mhc_registry._npu_hf32_enabled(fake_dev) is True
    assert torch.npu.matmul.allow_hf32 is True
    monkeypatch.setattr(torch, "npu", SimpleNamespace(matmul=SimpleNamespace(allow_hf32=False)), raising=False)
    assert mhc_registry._npu_hf32_enabled(fake_dev) is False
    assert mhc_registry._npu_hf32_enabled(torch.device("cpu")) is False


def test_lifecycle_rejects_non_contiguous_canonical_weights():
    manager = MhcWeightLifecycleManager("cpu", A3)
    non_contig = torch.randn(256, 24, dtype=torch.float32).T
    assert not non_contig.is_contiguous()

    manager.register_layer(
        "l0",
        ("seam_l0", "layer0_entry"),
        load_canonical=lambda: non_contig,
        version_tag="v1",
        canonical=non_contig,
    )
    with pytest.raises(ValueError, match="contiguous 2D float32"):
        manager.canonical_for("l0", "layer0_entry")

    with pytest.raises(ValueError, match="contiguous 2D float32"):
        manager.on_weight_update(
            "l0", canonical=non_contig, version_tag="v2", load_canonical=lambda: torch.ones(24, 256)
        )

    with pytest.raises(ValueError, match="contiguous 2D float32"):
        manager.prepare_layer("l0", replace(BF16_SEAM, D=64))
