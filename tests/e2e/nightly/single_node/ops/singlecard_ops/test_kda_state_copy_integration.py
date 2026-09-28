# SPDX-License-Identifier: Apache-2.0
"""NPU regressions for startup plans and dynamic selected counts.

These tests exercise real device copies and deny JIT entry after sealing. They
are operator/integration tests, not model-serving or performance benchmarks.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.kda_state_copy import KDAStateCopyPlan, initialize_kda_state_copy
from vllm_ascend.ops.triton import kda_state_copy as lowlevel


def _forbid_compilation(monkeypatch):
    """Fail loudly on both kernel JIT and compiler API entry during serving."""

    def forbidden(*args, **kwargs):
        raise AssertionError("KDA steady-state compilation is forbidden")

    monkeypatch.setattr(lowlevel._kda_state_copy_kernel, "run", forbidden)
    for name in ("triton", "triton.compiler", "triton.compiler.compiler", "triton.runtime.jit"):
        module = importlib.import_module(name)
        if callable(getattr(module, "compile", None)):
            monkeypatch.setattr(module, "compile", forbidden)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 1, 8193)])
@torch.inference_mode()
def test_dynamic_grid_and_metadata_use_only_four_compiled_variants(dtype, shape, monkeypatch):
    """Cover 65+ counts without a per-selected cache; preserve gaps and offsets."""
    payload = shape[0] * shape[1] * shape[2]
    stride = payload * 2 + 16
    backing = torch.full((80 * stride + 1,), -23, dtype=dtype, device="npu")
    state = backing.as_strided((80, *shape), (stride, shape[1] * shape[2], shape[2], 1), 1)
    state.copy_(torch.arange(80, device="npu", dtype=torch.float32).to(dtype)[:, None, None, None])
    original = backing.cpu()
    plan = KDAStateCopyPlan.prepare(state, 129)
    torch.testing.assert_close(backing.cpu(), original, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="sealed"):
        plan.gather(state, torch.zeros(1, dtype=torch.int64, device="npu"), None)
    plan.seal()
    assert len(plan._compiled) == 4
    _forbid_compilation(monkeypatch)
    # Every count 0..65 is covered, plus the scheduler boundary. INT32/INT64
    # inputs alternate and intentionally start at a misaligned, strided offset.
    for selected in (*range(66), 80, 129):
        backing.copy_(original)
        index_dtype = torch.int32 if selected % 2 else torch.int64
        base = torch.zeros(selected * 2 + 1, dtype=index_dtype, device="npu")
        indices = base[1::2]
        indices.copy_(torch.arange(selected, dtype=index_dtype, device="npu"))
        flag_base = torch.zeros(selected * 2 + 1, dtype=torch.int32, device="npu")
        flags = flag_base[1::2]
        flags.copy_(torch.arange(selected, dtype=torch.int32, device="npu") % 2)
        if selected:
            indices[-1] = -1 if selected % 2 else 80
        packed = plan.gather(state, indices, flags)
        host_indices = indices.cpu().long()
        valid = (host_indices >= 0) & (host_indices < 80)
        expected = torch.zeros((selected, *shape), dtype=dtype)
        active = valid & flags.cpu().bool()
        expected[active] = host_indices[active].to(dtype)[:, None, None, None]
        torch.testing.assert_close(packed.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(backing.cpu(), original, rtol=0, atol=0)
        updates = torch.full((selected, *shape), 11, dtype=torch.float32, device="npu")
        plan.scatter(state, updates, indices)
        expected_backing = original.clone()
        expected_state = expected_backing.as_strided(state.shape, state.stride(), 1)
        expected_state[host_indices[valid]] = 11
        torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)
    assert len(plan._compiled) == 4


@torch.inference_mode()
def test_worker_plan_deduplication_and_failed_reinitialization(monkeypatch):
    """Startup shares compiled kernels and publishes nothing after failure."""
    state = torch.full((8, 1, 2, 3), 7, dtype=torch.float32, device="npu")
    layers = [SimpleNamespace(_requires_kda_state_copy=True, kv_cache=(None, state.clone())) for _ in range(3)]
    initialize_kda_state_copy(dict(enumerate(layers)), 8)
    plans = [layer._ascend_kda_state_copy for layer in layers]
    assert len({id(plan) for plan in plans}) == len(layers)
    assert all(plan._compiled is plans[0]._compiled for plan in plans)
    for layer in layers:
        torch.testing.assert_close(layer.kv_cache[1], state, rtol=0, atol=0)
    layers[-1].kv_cache = ()
    with pytest.raises(RuntimeError, match="bound"):
        initialize_kda_state_copy(dict(enumerate(layers)), 8)
    assert all(layer._ascend_kda_state_copy is None for layer in layers)


@torch.inference_mode()
def test_worker_plan_graph_replay_and_rejections(monkeypatch):
    """Captured gather uses changed device indices/flags without compiler entry."""
    state = torch.arange(8 * 24, dtype=torch.float32, device="npu").view(8, 2, 3, 4)
    plan = KDAStateCopyPlan.prepare(state, 8)
    plan.seal()
    _forbid_compilation(monkeypatch)
    indices = torch.tensor([0, 3], dtype=torch.int32, device="npu")
    flags = torch.tensor([True, False], device="npu")
    plan.gather(state, indices, flags)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        packed = plan.gather(state, indices, flags)
    indices.copy_(torch.tensor([7, -1], dtype=torch.int32, device="npu"))
    flags.fill_(True)
    graph.replay()
    torch.testing.assert_close(packed[0].cpu(), state[7].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(packed[1].cpu(), torch.zeros((2, 3, 4)), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="scheduler"):
        plan.gather(state, torch.zeros(9, dtype=torch.int64, device="npu"), None)
    with pytest.raises(RuntimeError, match="layout"):
        plan.gather(state.transpose(2, 3), indices, flags)
    with pytest.raises(RuntimeError, match="flags"):
        plan.gather(state, indices, flags[:1])


@torch.inference_mode()
def test_real_prefill_with_native_chunk_matches_existing_path(monkeypatch):
    """Run the actual prefill method and installed native chunk (no model load).

    The existing native binary is exercised, not rebuilt or attested here.
    This is a focused real-math prefill integration, not a full model server.
    """
    # Normal engine startup registers this extension; standalone tests must
    # explicitly load it. A missing native build is a test failure, not a skip.
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention

    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    torch.nn.Module.__init__(attention)
    attention.gate_lower_bound = -4.0
    attention.A_log = torch.nn.Parameter(torch.zeros(12, dtype=torch.float32, device="npu"))
    attention.dt_bias = torch.nn.Parameter(torch.zeros(12 * 128, dtype=torch.float32, device="npu"))
    state = torch.randn((8, 12, 128, 128), dtype=torch.float32, device="npu")
    attention.kv_cache = (None, state)
    initialize_kda_state_copy({"layer": attention}, 8)
    q = torch.randn((1, 16, 12, 128), dtype=torch.bfloat16, device="npu") * 0.01
    k, v = torch.randn_like(q) * 0.01, torch.randn_like(q) * 0.01
    gate = torch.zeros_like(q)
    beta = torch.full((1, 16, 12), 0.5, dtype=torch.float32, device="npu")
    indices = torch.tensor([1, 3], dtype=torch.int32, device="npu")
    flags = torch.tensor([True, False], device="npu")
    metadata = SimpleNamespace(
        cu_seqlens_host=(0, 8, 16), cu_seqlens_kern=None, keep_meta=None, chunk_indices_chunk64_host=(0, 0, 1, 0)
    )
    reference = state.clone()
    from vllm_ascend.ops.triton.fla.utils import clear_ssm_states

    class ReferencePlan:
        """Exercise the previous indexing/clearing path on valid indices."""

        def gather(self, cache, selected, initial_flags):
            packed = cache[selected].contiguous()
            clear_ssm_states(packed, initial_flags)
            return packed

        def scatter(self, cache, packed, selected):
            cache[selected] = packed.to(cache.dtype)

    prepared_plan = attention._ascend_kda_state_copy
    attention._ascend_kda_state_copy = ReferencePlan()
    expected = attention._run_prefill(q, k, v, gate, beta, reference, indices, flags, metadata)
    attention._ascend_kda_state_copy = prepared_plan
    _forbid_compilation(monkeypatch)
    actual = attention._run_prefill(q, k, v, gate, beta, state, indices, flags, metadata)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, reference, rtol=0, atol=0)


@torch.inference_mode()
def test_int32_fast_path_does_not_convert_or_scan_environment(monkeypatch):
    """Serving uses the caller's aligned INT32 pointer and no config scan/JIT."""
    from vllm_ascend.ops import kda_state_copy as production

    state = torch.randn((8, 2, 3, 4), device="npu")
    plan = KDAStateCopyPlan.prepare(state, 16)
    plan.seal()
    indices = torch.tensor([0, 7], dtype=torch.int32, device="npu")
    assert plan._indices(state, indices) is indices
    _forbid_compilation(monkeypatch)

    def forbidden():
        raise AssertionError("serving scanned compiler environment")

    monkeypatch.setattr(production, "_configuration", forbidden)
    for count in (1, 2, 8, 16):
        ids = torch.arange(count, dtype=torch.int32, device="npu") % 8
        result = plan.gather(state, ids, None)
        torch.testing.assert_close(result, state[ids], rtol=0, atol=0)
    plan.scatter(state, state[indices].clone(), indices)
    assert len(plan._compiled) == 4


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_plan_matches_pr17301_native_operator(monkeypatch, dtype, index_dtype):
    """Optional differential test against the separately built PR17301 op.

    A clean main/Triton build intentionally does not include that unmerged
    native operator. Primary Torch-reference tests above remain mandatory.
    """
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    if not hasattr(torch.ops._C_ascend, "kda_state_copy"):
        pytest.skip("optional native reference requires a separate PR17301 build")
    shape, rows, stride, offset = (12, 128, 128), 8, 5308416, 393216
    payload = 12 * 128 * 128
    backing = torch.full(((rows - 1) * stride + offset + payload + 32,), -23, dtype=dtype, device="npu")
    state = backing.as_strided((rows, *shape), (stride, 16384, 128, 1), offset)
    state.copy_(torch.randn_like(state))
    reference_backing = backing.clone()
    reference = reference_backing.as_strided(state.shape, state.stride(), offset)
    plan = KDAStateCopyPlan.prepare(state, 16)
    plan.seal()
    _forbid_compilation(monkeypatch)
    for count in (0, 1, 4, 8):
        indices = torch.arange(count, dtype=index_dtype, device="npu")
        if count > 1:
            indices[-1] = rows
            indices[0] = -1
        for flag in (None, True, False):
            flags = None if flag is None else torch.full((count,), flag, device="npu", dtype=torch.bool)
            expected = torch.empty((count, *shape), dtype=dtype, device="npu")
            torch.ops._C_ascend.kda_state_copy(reference, expected, indices, flags, False)
            actual = plan.gather(state, indices, flags)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        final = torch.randn((count, *shape), dtype=dtype, device="npu")
        torch.ops._C_ascend.kda_state_copy(reference, final, indices, None, True)
        plan.scatter(state, final, indices)
        torch.testing.assert_close(backing, reference_backing, rtol=0, atol=0)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("page_stride", [24, 64])
@torch.inference_mode()
def test_auto_fp16_fallback_is_precompiled(monkeypatch, index_dtype, page_stride):
    """Unsupported dtype uses byte-copy fallback; dynamic requests never JIT."""
    from vllm_ascend.ops import kda_state_copy as production

    backing = torch.full((8 * page_stride + 1,), -23, device="npu", dtype=torch.float16)
    state = backing.as_strided((8, 2, 3, 4), (page_stride, 12, 4, 1), 1)
    state.copy_(torch.arange(8, device="npu")[:, None, None, None])
    initial = backing.cpu()
    layer = SimpleNamespace(_requires_kda_state_copy=True, kv_cache=(None, state))
    initialize_kda_state_copy({"layer": layer}, 16)
    plan = layer._ascend_kda_state_copy
    assert isinstance(plan, production.StridedKDAFallbackPlan)
    torch.testing.assert_close(backing.cpu(), initial, rtol=0, atol=0)
    _forbid_compilation(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("fallback serving entered JIT")

    monkeypatch.setattr(production.batch_memcpy_kernel, "run", forbidden)
    for count in (0, 1, 4, 8, 16):
        backing.copy_(initial)
        ids = torch.arange(count, device="npu", dtype=index_dtype) - 1
        if count >= 4:
            ids[-2] = torch.iinfo(index_dtype).min
            ids[-1] = torch.iinfo(index_dtype).max
        flags = (ids % 2) == 0
        actual = plan.gather(state, ids, flags)
        # Build the oracle on CPU: NPU advanced indexing of an offset FP16
        # contiguous cache can itself hit an alignment error on this runtime.
        # The tested path must support that cache without invoking indexing.
        ids_cpu, flags_cpu = ids.cpu().long(), flags.cpu()
        valid = (ids_cpu >= 0) & (ids_cpu < 8)
        expected = torch.zeros(actual.shape, dtype=state.dtype)
        reference = initial.as_strided(state.shape, state.stride(), 1)
        expected[valid & flags_cpu] = reference[ids_cpu[valid & flags_cpu]]
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        final = torch.full_like(actual, 17)
        plan.scatter(state, final, ids)
        expected_backing = initial.clone()
        view = expected_backing.as_strided(state.shape, state.stride(), 1)
        view[ids_cpu[valid]] = 17
        torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)


@torch.inference_mode()
def test_scatter_graph_replay_int64_extremes(monkeypatch):
    """Graph replay consumes dynamic INT64 values, including invalid extremes."""
    state = torch.zeros((8, 2, 3, 4), device="npu")
    plan = KDAStateCopyPlan.prepare(state, 8)
    plan.seal()
    _forbid_compilation(monkeypatch)
    ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="npu")
    final = torch.ones((4, 2, 3, 4), device="npu")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        plan.scatter(state, final, ids)
    state.zero_()
    ids.copy_(torch.tensor([7, 5, -(2**63), 2**63 - 1], dtype=torch.int64, device="npu"))
    final.fill_(11)
    graph.replay()
    expected = torch.zeros_like(state)
    expected[5] = 11
    expected[7] = 11
    torch.testing.assert_close(state, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_production_plan_offsets_beyond_four_gib(monkeypatch):
    """The optimized INT32 pointer route still widens cache offsets to INT64."""
    stride, offset = 2**30 + 32, 16
    backing = torch.empty(stride + offset + 48, dtype=torch.float32, device="npu")
    backing[:64].fill_(-23)
    backing[-64:].fill_(-23)
    state = backing.as_strided((2, 1, 2, 8), (stride, 16, 8, 1), offset)
    state[0].fill_(3)
    state[1].fill_(7)
    plan = KDAStateCopyPlan.prepare(state, 8)
    plan.seal()
    _forbid_compilation(monkeypatch)
    ids = torch.tensor([1, 0, -1, 2], dtype=torch.int32, device="npu")
    packed = plan.gather(state, ids, None)
    expected = torch.zeros((4, 1, 2, 8))
    expected[0].fill_(7)
    expected[1].fill_(3)
    torch.testing.assert_close(packed.cpu(), expected, rtol=0, atol=0)
    packed.add_(10)
    plan.scatter(state, packed, ids)
    torch.testing.assert_close(state[0].cpu(), torch.full((1, 2, 8), 13.0), rtol=0, atol=0)
    torch.testing.assert_close(state[1].cpu(), torch.full((1, 2, 8), 17.0), rtol=0, atol=0)
    assert torch.all(backing[:offset] == -23).item()
    assert torch.all(backing[-32:] == -23).item()


@torch.inference_mode()
def test_automatic_startup_uses_fused_plan(monkeypatch):
    """Exercise automatic capability selection, not just explicit triton mode."""
    from vllm_ascend.ops import kda_state_copy as production

    state = torch.zeros((4, 2, 3, 4), device="npu")
    layer = SimpleNamespace(_requires_kda_state_copy=True, kv_cache=(None, state))
    assert production.supports_kda_state_copy(state)
    initialize_kda_state_copy({"layer": layer}, 8)
    assert isinstance(layer._ascend_kda_state_copy, KDAStateCopyPlan)
    assert layer._kda_state_copy_ready
    _forbid_compilation(monkeypatch)
    ids = torch.tensor([0, 1], dtype=torch.int32, device="npu")
    actual = layer._ascend_kda_state_copy.gather(state, ids, None)
    torch.testing.assert_close(actual, state[ids], rtol=0, atol=0)


@torch.inference_mode()
def test_prepared_plan_nan_inf_signed_zero_bits(monkeypatch):
    """Copying preserves payload bits, including NaN/Inf/-0, without arithmetic."""
    state = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0], device="npu")
    state = state.reshape(1, 1, 1, 4).repeat(2, 1, 1, 1)
    plan = KDAStateCopyPlan.prepare(state, 8)
    plan.seal()
    _forbid_compilation(monkeypatch)
    ids = torch.tensor([1, 0], dtype=torch.int32, device="npu")
    flags = torch.tensor([True, False], device="npu")
    actual = plan.gather(state, ids, flags)
    torch.testing.assert_close(actual[0].view(torch.int32), state[1].view(torch.int32), rtol=0, atol=0)
    assert torch.all(actual[1].view(torch.int32) == 0).item()
    plan.scatter(state, actual, ids)
    torch.testing.assert_close(state[1].view(torch.int32), actual[0].view(torch.int32), rtol=0, atol=0)


@torch.inference_mode()
def test_plan_current_stream_and_device_guard(monkeypatch):
    """A sealed plan resolves each current stream and restores another device."""
    device = torch.npu.current_device()
    state = torch.zeros((8, 2, 3, 4), device=f"npu:{device}")
    ids = torch.tensor([1, 3], dtype=torch.int32, device=state.device)
    flags = torch.ones(2, dtype=torch.bool, device=state.device)
    plan = KDAStateCopyPlan.prepare(state, 8)
    plan.seal()
    _forbid_compilation(monkeypatch)
    torch.npu.synchronize()
    for value in (7, 13):
        stream = torch.npu.Stream(device=device)
        with torch.npu.stream(stream):
            state.fill_(value)
            actual = plan.gather(state, ids, flags)
            plan.scatter(state, actual + 1, ids)
        stream.synchronize()
        torch.testing.assert_close(actual.cpu(), torch.full((2, 2, 3, 4), float(value)), rtol=0, atol=0)
        torch.testing.assert_close(state[ids].cpu(), actual.cpu() + 1, rtol=0, atol=0)
    if torch.npu.device_count() > 1:
        other = (device + 1) % torch.npu.device_count()
        with torch.npu.device(other):
            actual = plan.gather(state, ids, flags)
            assert torch.npu.current_device() == other
        torch.npu.synchronize(device)
        torch.testing.assert_close(actual, state[ids], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("page_stride", [24, 64])
@torch.inference_mode()
def test_fullgraph_state_copy_uses_sealed_context_plan(dtype, page_stride, monkeypatch):
    """Dynamo/FakeTensor tracing preserves gather allocation and scatter writes.

    The eager backend tests full-graph tracing, not Inductor performance or a
    full model engine. The context contains a real initialized layer/plan;
    only the outer engine's forward-context installation is substituted.
    """
    from vllm_ascend.ops import kda_state_copy as production

    backing = torch.full((8, page_stride), -23, dtype=dtype, device="npu")
    state = backing.as_strided((8, 2, 3, 4), (page_stride, 12, 4, 1))
    state.copy_(torch.randn_like(state))
    layer = SimpleNamespace(_requires_kda_state_copy=True, kv_cache=(None, state))
    initialize_kda_state_copy({"layer": layer}, 8)
    plan = layer._ascend_kda_state_copy
    monkeypatch.setattr(production, "get_forward_context", lambda: SimpleNamespace(no_compile_layers={"layer": layer}))
    _forbid_compilation(monkeypatch)
    if dtype == torch.float16:

        def forbidden(*args, **kwargs):
            raise AssertionError("fallback entered JIT after seal")

        monkeypatch.setattr(production.batch_memcpy_kernel, "run", forbidden)

    def copy_roundtrip(cache, indices, flags):
        packed = plan.gather(cache, indices, flags)
        plan.scatter(cache, packed + 1, indices)
        return packed

    # Parametrization reuses one Python code object for independent simulated
    # workers. Reset Dynamo between workers instead of spending its per-frame
    # cache limit on three dtypes times the special zero/one/dynamic sizes.
    # This does not reset or relax the sealed Triton plan/compiler guards.
    torch._dynamo.reset()
    compiled = torch.compile(copy_roundtrip, backend="eager", fullgraph=True, dynamic=True)
    for count in (0, 1, 2, 5, 3):
        indices = torch.arange(count, dtype=torch.int32, device="npu")
        flags = (indices % 2) == 0
        expected = state[indices].clone()
        expected.masked_fill_(~flags[:, None, None, None], 0)
        result = compiled(state, indices, flags)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(state[indices], expected + 1, rtol=0, atol=0)
        assert torch.all(backing[:, 24:] == -23).item()
    # Compose the traced callable with a real NPU graph, then change metadata
    # in-place. Invalid rows clear on gather and must not overwrite the cache.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = compiled(state, indices, flags)
    state.fill_(3)
    indices.copy_(torch.tensor([7, 6, -1], dtype=torch.int32, device="npu"))
    flags.copy_(torch.tensor([True, False, True], device="npu"))
    graph.replay()
    expected = torch.zeros((3, 2, 3, 4), dtype=dtype, device="npu")
    expected[0].fill_(3)
    torch.testing.assert_close(graph_output, expected, rtol=0, atol=0)
    torch.testing.assert_close(state[7], torch.full_like(state[7], 4), rtol=0, atol=0)
    torch.testing.assert_close(state[6], torch.ones_like(state[6]), rtol=0, atol=0)
    torch.testing.assert_close(state[:6], torch.full_like(state[:6], 3), rtol=0, atol=0)
    assert torch.all(backing[:, 24:] == -23).item()
    # Resolve the active worker context at execution, not a captured old plan.
    layer._kda_state_copy_ready = False
    with pytest.raises(RuntimeError, match="prepared worker layer"):
        compiled(state, indices, flags)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_fullgraph_same_layout_layers_keep_distinct_bindings(dtype, monkeypatch):
    """Trace each layer's name while sharing sealed kernels, including fallback."""
    from vllm_ascend.ops import kda_state_copy as production

    states = [torch.empty_strided((8, 2, 3, 4), (64, 12, 4, 1), dtype=dtype, device="npu") for _ in range(2)]
    layers = {
        name: SimpleNamespace(_requires_kda_state_copy=True, kv_cache=(None, state))
        for name, state in zip(("layer_0", "layer_31"), states)
    }
    initialize_kda_state_copy(layers, 8)
    plans = [layer._ascend_kda_state_copy for layer in layers.values()]
    assert plans[0] is not plans[1]
    assert plans[0]._compiled is plans[1]._compiled
    _forbid_compilation(monkeypatch)
    monkeypatch.setattr(production.batch_memcpy_kernel, "run", Mock(side_effect=AssertionError("serving JIT")))
    indices = torch.tensor([0, 2], dtype=torch.int32, device="npu")
    flags = torch.tensor([True, False], device="npu")
    for value, (name, layer) in enumerate(layers.items(), 3):
        # A layer-local context makes accidentally tracing a sibling fail closed.
        context = SimpleNamespace(no_compile_layers={name: layer})
        monkeypatch.setattr(production, "get_forward_context", lambda context=context: context)
        plan = layer._ascend_kda_state_copy
        state = layer.kv_cache[1]
        state.fill_(value)
        traced_names: list[str] = []

        def capture(graph, example_inputs, traced_names=traced_names):
            traced_names.extend(
                node.args[-1]
                for node in graph.graph.nodes
                if node.op == "call_function"
                and node.target in (torch.ops.vllm.kda_state_gather, torch.ops.vllm.kda_state_scatter)
            )
            return graph.forward

        def roundtrip(cache, ids, keep, plan=plan):
            packed = plan.gather(cache, ids, keep)
            plan.scatter(cache, packed + 1, ids)
            return packed

        torch._dynamo.reset()
        compiled = torch.compile(roundtrip, backend=capture, fullgraph=True)
        result = compiled(state, indices, flags)
        assert traced_names == [name, name]
        expected = torch.zeros_like(result)
        expected[0].fill_(value)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(state[indices], expected + 1, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("state_offset", [0, 1])
@torch.inference_mode()
def test_prepare_matches_alignment_of_unaligned_scratch_backing(dtype, state_offset, monkeypatch):
    """Scratch alignment follows both actual pointers, not an allocator assumption."""
    payload = 6
    source = torch.full((8 * payload + state_offset,), 7, dtype=dtype, device="npu")
    state = source.narrow(0, state_offset, 8 * payload).view(8, 1, 2, 3)
    scratch_size = payload + 16 // state.element_size()
    zeros = torch.zeros
    allocations = []

    def unaligned_zeros(size, *args, **kwargs):
        """Offset only the scratch allocation; index metadata stays aligned."""
        if size == scratch_size and kwargs.get("dtype") == dtype:
            offset = 8 // state.element_size()
            backing = zeros(size + offset, *args, **kwargs).narrow(0, offset, size)
            assert backing.data_ptr() % 16 == 8
            allocations.append(backing)
            return backing
        return zeros(size, *args, **kwargs)

    monkeypatch.setattr(torch, "zeros", unaligned_zeros)
    plan = KDAStateCopyPlan.prepare(state, 8)
    plan.seal()
    assert len(allocations) == 1
    assert len(plan._compiled) == 4
    # Startup must only mutate its disposable storage, never the bound cache.
    torch.testing.assert_close(state, torch.full_like(state, 7), rtol=0, atol=0)
    _forbid_compilation(monkeypatch)
    indices = torch.tensor([0, 7], dtype=torch.int32, device="npu")
    flags = torch.tensor([True, False], device="npu")
    packed = plan.gather(state, indices, flags)
    expected = torch.zeros_like(packed)
    expected[0].fill_(7)
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    plan.scatter(state, packed + 1, indices)
    torch.testing.assert_close(state[indices], expected + 1, rtol=0, atol=0)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_auto_contiguous_prefill_masks_invalid_rows(monkeypatch, index_dtype):
    """Exercise real prefill dispatch and copies; only chunk math is substituted."""
    from vllm_ascend.ops import kda_state_copy as production
    from vllm_ascend.ops import kimi_kda as kimi

    attention = kimi.AscendKimiK3DeltaAttention.__new__(kimi.AscendKimiK3DeltaAttention)
    torch.nn.Module.__init__(attention)
    attention.gate_lower_bound = None
    attention.A_log, attention.dt_bias = torch.zeros(1), torch.zeros(1)
    state = torch.arange(8 * 24, dtype=torch.float16, device="npu").reshape(8, 2, 3, 4)
    attention.kv_cache = (None, state)
    initialize_kda_state_copy({"layer": attention}, 8)
    limits = torch.iinfo(index_dtype)
    ids_cpu = torch.tensor([0, -1, 8, limits.min, limits.max, 3], dtype=index_dtype)
    flags_cpu = torch.tensor([True, True, True, True, True, False])
    indices, flags = ids_cpu.to("npu"), flags_cpu.to("npu")
    expected_cache = state.cpu()
    expected_gather = torch.zeros((6, 2, 3, 4), dtype=state.dtype)
    expected_gather[0] = expected_cache[0]
    final = torch.full((6, 2, 3, 4), 17, dtype=torch.float32, device="npu")
    chunk = Mock(return_value=("output", final))
    monkeypatch.setattr(kimi, "run_chunk_kda", chunk)
    _forbid_compilation(monkeypatch)
    monkeypatch.setattr(production.batch_memcpy_kernel, "run", Mock(side_effect=AssertionError("serving JIT")))
    metadata = SimpleNamespace(
        cu_seqlens_host=tuple(range(7)), cu_seqlens_kern=None, keep_meta=None, chunk_indices_chunk64_host=()
    )
    output = attention._run_prefill(None, None, None, None, None, state, indices, flags, metadata)
    assert output == "output"
    torch.testing.assert_close(chunk.call_args.args[5].cpu(), expected_gather, rtol=0, atol=0)
    expected_cache[[0, 3]] = 17
    torch.testing.assert_close(state.cpu(), expected_cache, rtol=0, atol=0)
