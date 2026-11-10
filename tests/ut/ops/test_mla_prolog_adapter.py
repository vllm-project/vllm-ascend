# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests; no vLLM, torch_npu or vendor installation is needed.

Run with ``pytest --confcutdir=tests/ut/ops tests/ut/ops/test_mla_prolog_adapter.py``
to bypass the repository-wide NPU fixtures. These tests validate adaptation,
not the numerical accuracy or device execution of the vendor kernel.
"""

import builtins
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

HIDDEN_SIZE = 1024
Q_LORA_RANK = 640
KV_LORA_RANK = 512
NOPE_DIM = 256
NUM_HEADS = 2
BLOCK_SIZE = 16
MX_GROUP_SIZE = 32
SCALE_PAIR_SIZE = 2
EPSILON = 1e-6
MODULE_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/mla_prolog.py"


def _load_adapter(monkeypatch):
    # A synthetic name deliberately avoids executing vllm_ascend/__init__.py.
    name = "_cpu_mla_prolog_adapter_under_test"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def packaged(monkeypatch):
    name = "vllm_ascend.ops.packaged_mla_prolog"
    if name not in sys.modules:
        path = MODULE_PATH.with_name("packaged_mla_prolog.py")
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module = sys.modules[name]
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.packaged_mla_prolog", module)
    module._get_dsl_op.cache_clear()
    # Only tests register CPU execution; production still dispatches PrivateUse1.
    library = torch.library.Library("vllm_ascend", "IMPL", "CPU")
    try:
        library.impl("quant_mla_prolog", module._quant_mla_prolog_impl)
        yield module
    finally:
        library._destroy()
        module._get_dsl_op.cache_clear()


@pytest.fixture
def vendor(monkeypatch, packaged):
    kernel = PublicKernel()
    public = ModuleType("ops.quant_mla_prolog")
    monkeypatch.setattr(public, "quant_mla_prolog", kernel, raising=False)
    package = ModuleType("ops")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "ops", package)
    monkeypatch.setitem(sys.modules, "ops.quant_mla_prolog", public)
    return kernel


@pytest.fixture
def npu(monkeypatch):
    module = ModuleType("torch_npu")
    monkeypatch.setattr(module, "npu_format_cast", Mock(side_effect=lambda tensor, fmt: tensor.clone()), raising=False)
    monkeypatch.setattr(
        module, "npu_dynamic_mx_quant", Mock(side_effect=AssertionError("already quantized input")), raising=False
    )
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    return module


@pytest.fixture
def adapter(monkeypatch, npu, packaged):
    return _load_adapter(monkeypatch)


def _bytes(shape, offset=0):
    # Nonuniform bytes catch transposes and numeric casts, including sign bits.
    return ((torch.arange(math.prod(shape)) + offset) % 256).to(torch.uint8).reshape(shape)


def _storage(tensor):
    return tensor.untyped_storage().data_ptr()


def _assert_bytes_equal(actual, expected):
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.fixture
def weights():
    return dict(
        weight_dq=_bytes((HIDDEN_SIZE, Q_LORA_RANK)).view(torch.float8_e4m3fn),
        weight_uq_qr=_bytes((Q_LORA_RANK, NUM_HEADS * NOPE_DIM), 17).view(torch.float8_e4m3fn),
        weight_dkv_kr=_bytes((HIDDEN_SIZE, KV_LORA_RANK), 29).view(torch.float8_e4m3fn),
        weight_uk=torch.arange(NUM_HEADS * NOPE_DIM * KV_LORA_RANK, dtype=torch.float32)
        .reshape(NUM_HEADS, NOPE_DIM, KV_LORA_RANK)
        .to(torch.bfloat16),
        dequant_scale_w_dq=_bytes((HIDDEN_SIZE // 64, Q_LORA_RANK, SCALE_PAIR_SIZE), 31),
        dequant_scale_w_uq_qr=_bytes((Q_LORA_RANK // 64, NUM_HEADS * NOPE_DIM, SCALE_PAIR_SIZE), 47),
        dequant_scale_w_dkv_kr=_bytes((HIDDEN_SIZE // 64, KV_LORA_RANK, SCALE_PAIR_SIZE), 59),
        rmsnorm_gamma_cq=torch.linspace(0.5, 1.5, Q_LORA_RANK, dtype=torch.bfloat16),
        rmsnorm_gamma_ckv=torch.linspace(1.5, 2.5, KV_LORA_RANK, dtype=torch.bfloat16),
        rmsnorm_epsilon_cq=EPSILON,
        rmsnorm_epsilon_ckv=EPSILON,
    )


class PublicKernel:
    """Strict public wheel signature: private/native entrypoints are absent."""

    def __init__(self):
        self.calls = []
        self.outputs = []

    def __call__(
        self,
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        quant_scale_kv=None,
        rope_sin=None,
        rope_cos=None,
        *,
        norm_eps=1e-5,
        w_qr_quant_mode,
        qkv_quant_mode,
        cache_mode,
        qr_scale=1.0,
        kvr_scale=1.0,
        rope_mode=1,
        return_qr=False,
    ):
        call = locals().copy()
        call.pop("self")
        self.calls.append(call)
        tokens = x.shape[0]
        heads = wkb.shape[0]
        rank = norm_weight_qa.numel()
        if tokens == 0:
            output = tuple(
                torch.empty(0, dtype=dtype)
                for dtype in (torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.float8_e8m0fnu)
            )
        else:
            output = (
                torch.full((tokens, heads, KV_LORA_RANK), 3.0, dtype=torch.bfloat16),
                torch.empty(0, dtype=torch.float32),
                _bytes((tokens, rank), 73).view(torch.float8_e4m3fn)
                if return_qr
                else torch.empty(0, dtype=torch.float8_e4m3fn),
                _bytes((tokens, rank // MX_GROUP_SIZE), 101).view(torch.float8_e8m0fnu)
                if return_qr
                else torch.empty(0, dtype=torch.float8_e8m0fnu),
            )
        self.outputs.append(output)
        return output


@pytest.fixture
def prepared(adapter, weights, vendor):
    return adapter.prepare_mla_prolog(**weights)


def _inputs(weights, tokens=3, *, cache_quant=False, query_norm=True):
    width = KV_LORA_RANK + (16 if cache_quant else 0)
    return dict(
        **weights,
        token_x=_bytes((tokens, HIDDEN_SIZE), 83).view(torch.float8_e4m3fn),
        dequant_scale_x=_bytes((tokens, HIDDEN_SIZE // MX_GROUP_SIZE), 127),
        kv_cache=torch.zeros((2, BLOCK_SIZE, width), dtype=torch.uint8 if cache_quant else torch.bfloat16),
        kr_cache=torch.empty((2, BLOCK_SIZE, 0), dtype=torch.bfloat16),
        cache_index=torch.arange(tokens, dtype=torch.int64),
        rope_sin=torch.empty((tokens, 0), dtype=torch.bfloat16),
        rope_cos=torch.empty((tokens, 0), dtype=torch.bfloat16),
        query_norm_flag=query_norm,
        weight_quant_mode=3,
        query_quant_mode=0,
        kv_cache_quant_mode=3 if cache_quant else 0,
        cache_mode="PA_BSND",
        qc_qr_scale=1.0,
        kc_scale=1.0,
        **(
            dict(ckvkr_repo_mode=1, quant_scale_repo_mode=1, tile_size=128, k_nope_clip_alpha=None)
            if cache_quant
            else {}
        ),
    )


def _forward(adapter, prepared, inputs):
    return adapter.mla_prolog(backend="cannbot", prepared=prepared, **inputs)


def test_default_import_isolated_and_five_outputs_pass_through(monkeypatch):
    original_import = builtins.__import__
    forbidden = ("torch_npu", "cannbotdsl", "cannbotDSL", "ops", "vllm_ascend")

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in forbidden, f"unexpected optional import: {name}"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = _load_adapter(monkeypatch)
    outputs = tuple(object() for _ in range(5))
    kernel = Mock(return_value=outputs)
    inputs = {"token_x": object(), "unknown_native_kwarg": object()}
    result = module.mla_prolog(torch_npu_prolog=kernel, **inputs)
    assert result is outputs
    kernel.assert_called_once_with(**inputs)


def test_default_resolves_only_mocked_torch_npu(adapter, npu, monkeypatch):
    outputs = tuple(object() for _ in range(5))
    npu.npu_mla_prolog_v3 = Mock(return_value=outputs)
    monkeypatch.setitem(sys.modules, "ops", None)
    assert adapter.mla_prolog(token_x=None) is outputs
    npu.npu_mla_prolog_v3.assert_called_once_with(token_x=None)


def test_missing_dependency_is_lazy_and_actionable(adapter, weights, monkeypatch):
    monkeypatch.setitem(sys.modules, "ops", None)
    monkeypatch.setitem(sys.modules, "ops.quant_mla_prolog", None)
    state = adapter.prepare_mla_prolog(**weights)
    with pytest.raises((ImportError, RuntimeError), match="(?i)cannbot|quant_mla_prolog"):
        _forward(adapter, state, _inputs(weights))


def test_resolves_public_vendor_entrypoint(adapter, weights, monkeypatch):
    public = ModuleType("ops.quant_mla_prolog")
    kernel = PublicKernel()
    monkeypatch.setattr(public, "quant_mla_prolog", kernel, raising=False)
    package = ModuleType("ops")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "ops", package)
    monkeypatch.setitem(sys.modules, "ops.quant_mla_prolog", public)
    state = adapter.prepare_mla_prolog(**weights)
    _forward(adapter, state, _inputs(weights))
    assert len(kernel.calls) == 1


@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e8m0fnu])
@pytest.mark.parametrize("strided", [False, True])
def test_prepare_layout_bitview_and_original_parameters_unchanged(adapter, weights, npu, scale_dtype, strided):
    # Loaded split weights/scales can have strides; prepacking must own copies.
    for name, value in list(weights.items()):
        if not isinstance(value, torch.Tensor):
            continue
        if name.startswith("dequant_scale"):
            value = value.view(scale_dtype)
        if strided and value.ndim >= 2:
            value = value.transpose(0, 1).contiguous().transpose(0, 1)
        weights[name] = value
    snapshots = {
        name: (value.clone(), value.stride(), _storage(value))
        for name, value in weights.items()
        if isinstance(value, torch.Tensor)
    }
    state = adapter.prepare_mla_prolog(**weights)
    for source, destination in (("weight_dq", "wqa"), ("weight_uq_qr", "wqb"), ("weight_dkv_kr", "wkva")):
        packed = getattr(state, destination)
        assert packed.dtype == torch.float8_e4m3fn
        assert packed.is_contiguous()
        assert _storage(packed) != _storage(weights[source])
        _assert_bytes_equal(packed, weights[source].t().contiguous())
    for source, destination in (
        ("dequant_scale_w_dq", "descale_wqa"),
        ("dequant_scale_w_uq_qr", "descale_wqb"),
        ("dequant_scale_w_dkv_kr", "descale_wkva"),
    ):
        packed = getattr(state, destination)
        assert packed.dtype == torch.float8_e8m0fnu
        assert packed.is_contiguous()
        assert _storage(packed) != _storage(weights[source])
        _assert_bytes_equal(packed, weights[source].transpose(0, 1).contiguous())
    assert state.wkb.is_contiguous()
    assert _storage(state.wkb) != _storage(weights["weight_uk"])
    torch.testing.assert_close(state.wkb, weights["weight_uk"])
    for source, destination in (("rmsnorm_gamma_cq", "norm_weight_qa"), ("rmsnorm_gamma_ckv", "norm_weight_kva")):
        packed = getattr(state, destination)
        assert packed.dtype == torch.float32
        torch.testing.assert_close(packed, weights[source].float())
    for name, (snapshot, stride, storage) in snapshots.items():
        _assert_bytes_equal(weights[name], snapshot)
        assert weights[name].stride() == stride
        assert _storage(weights[name]) == storage
    # Explicit empty+copy_ allocation is also a valid ND conversion; if the
    # adapter uses format_cast, it must request ND rather than device NZ.
    assert all(call.args[1] == 2 for call in npu.npu_format_cast.call_args_list)


@pytest.mark.parametrize("query_norm", [False, True])
@pytest.mark.parametrize("cache_quant", [False, True])
def test_five_outputs_keep_quantized_latent_separate_from_rope(
    adapter, prepared, weights, npu, query_norm, cache_quant, vendor
):
    inputs = _inputs(weights, cache_quant=cache_quant, query_norm=query_norm)
    output = _forward(adapter, prepared, inputs)
    assert len(output) == 5
    q, rope, q_scale, q_norm, q_norm_scale = output
    vendor_output = vendor.outputs[-1]
    assert q.shape == (3, NUM_HEADS, KV_LORA_RANK)
    assert q.dtype == torch.bfloat16
    torch.testing.assert_close(q, vendor_output[0])
    assert rope.shape == (3, NUM_HEADS, 0)
    assert rope.dtype == torch.bfloat16
    assert q_scale.shape == (0,) and q_scale.dtype == torch.float32
    assert q_norm.shape == ((3, Q_LORA_RANK) if query_norm else (0,))
    assert q_norm.dtype == torch.float8_e4m3fn
    assert q_norm_scale.shape == ((3, Q_LORA_RANK // MX_GROUP_SIZE) if query_norm else (0,))
    assert q_norm_scale.dtype == torch.float8_e8m0fnu
    _assert_bytes_equal(q_norm, vendor_output[2])
    _assert_bytes_equal(q_norm_scale, vendor_output[3])
    call = vendor.calls[-1]
    assert call["x"] is inputs["token_x"]
    assert call["descale_x"].dtype == torch.float8_e8m0fnu
    assert _storage(call["descale_x"]) == _storage(inputs["dequant_scale_x"])
    _assert_bytes_equal(call["descale_x"], inputs["dequant_scale_x"])
    assert call["w_qr_quant_mode"] == 1
    assert call["qkv_quant_mode"] == int(cache_quant)
    assert call["cache_mode"] == 0 and call["rope_mode"] == 0
    assert call["rope_sin"] is None and call["rope_cos"] is None
    assert call["norm_eps"] == EPSILON
    assert call["return_qr"] is query_norm
    npu.npu_dynamic_mx_quant.assert_not_called()


@pytest.mark.parametrize("query_norm", [False, True])
@pytest.mark.parametrize("cache_quant", [False, True])
def test_zero_tokens_preserve_five_output_shapes_without_launch(
    adapter, prepared, weights, query_norm, cache_quant, vendor
):
    output = _forward(adapter, prepared, _inputs(weights, 0, cache_quant=cache_quant, query_norm=query_norm))
    assert len(output) == 5
    assert [tuple(t.shape) for t in output] == [
        (0, NUM_HEADS, KV_LORA_RANK),
        (0, NUM_HEADS, 0),
        (0,),
        (0, Q_LORA_RANK) if query_norm else (0,),
        (0, Q_LORA_RANK // MX_GROUP_SIZE) if query_norm else (0,),
    ]
    assert [t.dtype for t in output] == [
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float8_e4m3fn,
        torch.float8_e8m0fnu,
    ]
    assert not vendor.calls


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.uint8, torch.int8, torch.float8_e4m3fn])
@pytest.mark.parametrize("rank", [3, 4])
def test_cache_views_alias_original_storage(adapter, prepared, weights, cache_dtype, rank, vendor):
    inputs = _inputs(weights, cache_quant=cache_dtype != torch.bfloat16)
    # An offset, contiguous slice must preserve its storage offset as well.
    original = inputs["kv_cache"].view(cache_dtype)
    backing = torch.zeros((3, *original.shape[1:]), dtype=original.dtype)
    cache = backing[1:]
    assert cache.is_contiguous() and cache.storage_offset() > 0
    if rank == 4:
        cache = cache.unsqueeze(2)
    inputs["kv_cache"] = cache
    _forward(adapter, prepared, inputs)
    passed = vendor.calls[-1]["kv_cache"]
    assert passed.shape == (2, BLOCK_SIZE, 1, cache.shape[-1])
    assert passed.dtype == (torch.bfloat16 if cache_dtype == torch.bfloat16 else torch.float8_e4m3fn)
    assert _storage(passed) == _storage(cache)
    assert passed.storage_offset() == cache.storage_offset()
    passed.view(torch.uint8).reshape(-1)[0] = 37
    assert cache.view(torch.uint8).reshape(-1)[0].item() == 37


@pytest.mark.parametrize("cache_quant", [False, True])
def test_noncontiguous_cache_rejected_without_copy_or_launch(adapter, prepared, weights, cache_quant, vendor):
    inputs = _inputs(weights, cache_quant=cache_quant)
    cache = inputs["kv_cache"]
    inputs["kv_cache"] = torch.zeros((*cache.shape[:-1], cache.shape[-1] * 2), dtype=cache.dtype)[..., ::2]
    assert not inputs["kv_cache"].is_contiguous()
    with pytest.raises((ValueError, TypeError), match="(?i)contiguous"):
        _forward(adapter, prepared, inputs)
    assert not vendor.calls


def test_unequal_norm_epsilons_rejected_at_prepare(adapter, weights, vendor):
    weights["rmsnorm_epsilon_ckv"] = EPSILON * 2
    with pytest.raises(ValueError, match="(?i)eps"):
        adapter.prepare_mla_prolog(**weights)
    assert not vendor.calls


@pytest.mark.parametrize(
    "change",
    [
        {"weight_quant_mode": 0},
        {"weight_quant_mode": 2},
        {"query_quant_mode": 1},
        {"kv_cache_quant_mode": 1},
        {"cache_mode": "PA_NZ"},
        {"cache_mode": "PA"},
        {"rmsnorm_epsilon_ckv": EPSILON * 2},
        {"rope_sin": torch.ones(3, 64, dtype=torch.bfloat16)},
        {"unknown_option": True},
    ],
)
def test_unsupported_combinations_do_not_launch(adapter, prepared, weights, change, vendor):
    inputs = _inputs(weights)
    inputs.update(change)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        _forward(adapter, prepared, inputs)
    assert not vendor.calls


@pytest.mark.parametrize(
    "change",
    [
        {"tile_size": 32},
        {"ckvkr_repo_mode": 0},
        {"quant_scale_repo_mode": 0},
        {"k_nope_clip_alpha": 0.9},
    ],
)
def test_unsupported_fp8_cache_modes_do_not_launch(adapter, prepared, weights, change, vendor):
    inputs = _inputs(weights, cache_quant=True)
    inputs.update(change)
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        _forward(adapter, prepared, inputs)
    assert not vendor.calls


@pytest.mark.parametrize(
    "name",
    [
        "weight_dq",
        "weight_uq_qr",
        "weight_dkv_kr",
        "dequant_scale_w_dq",
        "dequant_scale_w_uq_qr",
        "dequant_scale_w_dkv_kr",
        "weight_uk",
    ],
)
def test_invalid_loaded_weight_shape_rejected(adapter, weights, name, vendor):
    weights[name] = weights[name][:-1]
    with pytest.raises((ValueError, TypeError)):
        adapter.prepare_mla_prolog(**weights)
    assert not vendor.calls


@pytest.mark.parametrize("name", ["token_x", "dequant_scale_x", "cache_index", "kv_cache"])
def test_invalid_forward_shape_does_not_launch(adapter, prepared, weights, name, vendor):
    inputs = _inputs(weights)
    if name == "cache_index":
        inputs[name] = inputs[name][:-1]
    else:
        inputs[name] = inputs[name][..., :-1].contiguous()
    with pytest.raises((ValueError, TypeError)):
        _forward(adapter, prepared, inputs)
    assert not vendor.calls


@pytest.mark.parametrize("invalid_prepared", [None, object(), SimpleNamespace(wqa=object())])
def test_cannbot_requires_prepared_state_before_dispatch(adapter, monkeypatch, invalid_prepared):
    packaged_forward = Mock()
    native_forward = Mock()
    monkeypatch.setattr(adapter, "_cannbot_forward", packaged_forward)
    with pytest.raises(ValueError, match="^Cannbot requires prepare_mla_prolog first$"):
        adapter.mla_prolog(backend="cannbot", prepared=invalid_prepared, torch_npu_prolog=native_forward)
    packaged_forward.assert_not_called()
    native_forward.assert_not_called()


def test_unknown_backend_is_not_silently_fallback(adapter):
    kernel = Mock()
    with pytest.raises(ValueError, match="(?i)backend"):
        adapter.mla_prolog(backend="misspelled_backend", torch_npu_prolog=kernel)
    kernel.assert_not_called()


def test_unit_clip_scalar_is_supported(adapter, prepared, weights, vendor):
    inputs = _inputs(weights, cache_quant=True)
    inputs["k_nope_clip_alpha"] = 1.0
    assert len(_forward(adapter, prepared, inputs)) == 5
    assert len(vendor.calls) == 1


def test_tensor_clip_rejected_without_host_scalar_read(adapter, prepared, weights, monkeypatch, vendor):
    inputs = _inputs(weights, cache_quant=True)
    inputs["k_nope_clip_alpha"] = torch.ones(1)
    forbidden = Mock(side_effect=AssertionError("must not synchronize tensor to host"))
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    with pytest.raises(ValueError, match="(?i)clip|tensor"):
        _forward(adapter, prepared, inputs)
    forbidden.assert_not_called()
    assert not vendor.calls


def test_repeated_forward_never_repacks(adapter, prepared, weights, npu, monkeypatch, vendor):
    inputs = _inputs(weights)
    before = npu.npu_format_cast.call_count
    forbidden = Mock(side_effect=AssertionError("forward must not prepare weights"))
    monkeypatch.setattr(adapter, "prepare_mla_prolog", forbidden)
    _forward(adapter, prepared, inputs)
    _forward(adapter, prepared, inputs)
    assert npu.npu_format_cast.call_count == before
    forbidden.assert_not_called()
    first, second = vendor.calls
    for name in (
        "wqa",
        "wqb",
        "wkva",
        "wkb",
        "descale_wqa",
        "descale_wqb",
        "descale_wkva",
        "norm_weight_qa",
        "norm_weight_kva",
    ):
        # Dispatcher scales are fresh byte/E8M0 views, never repacked buffers.
        assert first[name].untyped_storage().data_ptr() == second[name].untyped_storage().data_ptr()
        assert first[name].storage_offset() == second[name].storage_offset()
        assert first[name].shape == second[name].shape
        assert first[name].dtype == second[name].dtype
