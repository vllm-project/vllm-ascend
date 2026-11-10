# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU dispatcher contracts, not NPU numerical or ACLGraph validation.

Run with --confcutdir=tests/ut/ops to avoid unrelated NPU fixtures.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

MODULE_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/packaged_mla_prolog.py"


def _load_packaged():
    # Reuse one registration across this file and the adapter contract tests.
    name = "vllm_ascend.ops.packaged_mla_prolog"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture
def packaged():
    return _load_packaged()


def _inputs(tokens=3, mode=0, device="cpu"):
    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device=device)

    return dict(
        x=empty((tokens, 1024), torch.float8_e4m3fn),
        wqa=empty((640, 1024), torch.float8_e4m3fn),
        wqb=empty((512, 640), torch.float8_e4m3fn),
        wkva=empty((512, 1024), torch.float8_e4m3fn),
        wkb=empty((2, 256, 512), torch.bfloat16),
        descale_x=empty((tokens, 32), torch.uint8),
        descale_wqa=empty((640, 16, 2), torch.uint8),
        descale_wqb=empty((512, 10, 2), torch.uint8),
        descale_wkva=empty((512, 16, 2), torch.uint8),
        norm_weight_qa=empty((640,), torch.float32),
        norm_weight_kva=empty((512,), torch.float32),
        kv_cache=empty((2, 16, 512 if mode == 0 else 528), torch.bfloat16 if mode == 0 else torch.uint8),
        cache_index=torch.arange(tokens, device=device, dtype=torch.int64),
        norm_eps=1e-6,
        kv_cache_quant_mode=mode,
        return_qr=True,
    )


def test_schema_declares_only_original_cache_mutation(packaged):
    schema = packaged.quant_mla_prolog._schema
    writes = [arg.name for arg in schema.arguments if arg.alias_info and arg.alias_info.is_write]
    assert writes == ["kv_cache"]
    assert all(result.alias_info is None for result in schema.returns)
    assert torch._C._dispatch_has_kernel_for_dispatch_key("vllm_ascend::quant_mla_prolog", "PrivateUse1")


@pytest.mark.parametrize("tokens", [0, 1, 7])
@pytest.mark.parametrize("mode", [0, 3])
@pytest.mark.parametrize("return_qr", [False, True])
def test_meta_outputs_without_external_import(packaged, monkeypatch, tokens, mode, return_qr):
    loader = Mock(side_effect=AssertionError("fake tracing must not import DSL"))
    monkeypatch.setattr(packaged, "_get_dsl_op", loader)
    inputs = _inputs(tokens, mode, "meta")
    inputs["return_qr"] = return_qr
    outputs = packaged.quant_mla_prolog(**inputs)
    assert [tuple(t.shape) for t in outputs] == [
        (tokens, 2, 512),
        (0,),
        (tokens, 640) if return_qr else (0,),
        (tokens, 20) if return_qr else (0,),
    ]
    assert [t.dtype for t in outputs] == [torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.uint8]
    assert all(t.device.type == "meta" for t in outputs)
    loader.assert_not_called()


def test_fake_tensor_symbolic_tokens(packaged, monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    monkeypatch.setattr(packaged, "_get_dsl_op", Mock(side_effect=AssertionError("unexpected DSL import")))
    with FakeTensorMode(shape_env=ShapeEnv()) as mode:
        tokens = mode.shape_env.create_unbacked_symint()
        inputs = _inputs(0)
        inputs["x"] = torch.empty((tokens, 1024), dtype=torch.float8_e4m3fn)
        inputs["descale_x"] = torch.empty((tokens, 32), dtype=torch.uint8)
        inputs["cache_index"] = torch.empty((tokens,), dtype=torch.int64)
        outputs = packaged.quant_mla_prolog(**inputs)
        assert str(outputs[0].shape[0]) == str(tokens)
        assert str(outputs[2].shape[0]) == str(tokens)
        assert outputs[3].dtype == torch.uint8


@pytest.mark.parametrize("mode", [0, 3])
def test_zero_tokens_never_load_vendor(packaged, monkeypatch, mode):
    loader = Mock(side_effect=AssertionError("T0 must not load vendor"))
    monkeypatch.setattr(packaged, "_get_dsl_op", loader)
    outputs = packaged._quant_mla_prolog_impl(**_inputs(0, mode))
    assert outputs[0].shape == (0, 2, 512)
    assert outputs[3].shape == (0, 20)
    loader.assert_not_called()


@pytest.mark.parametrize("mode", [0, 3])
def test_mutating_failure_propagates_without_retry(packaged, monkeypatch, mode):
    inputs = _inputs(mode=mode)
    inputs["kv_cache"].view(torch.uint8).zero_()
    storage = inputs["kv_cache"].untyped_storage().data_ptr()

    def vendor(**kwargs):
        assert kwargs["kv_cache"].untyped_storage().data_ptr() == storage
        kwargs["kv_cache"].view(torch.uint8).fill_(17)
        raise RuntimeError("failed after cache write")

    kernel = Mock(side_effect=vendor)
    monkeypatch.setattr(packaged, "_get_dsl_op", lambda: kernel)
    with pytest.raises(RuntimeError, match="failed after cache write"):
        packaged._quant_mla_prolog_impl(**inputs)
    kernel.assert_called_once()
    assert torch.all(inputs["kv_cache"].view(torch.uint8) == 17)


def test_noncontiguous_cache_rejected_before_vendor(packaged, monkeypatch):
    inputs = _inputs()
    inputs["kv_cache"] = inputs["kv_cache"].transpose(0, 1)
    loader = Mock(side_effect=AssertionError("invalid cache must fail first"))
    monkeypatch.setattr(packaged, "_get_dsl_op", loader)
    with pytest.raises(ValueError, match="contiguous"):
        packaged._quant_mla_prolog_impl(**inputs)
    loader.assert_not_called()


def test_functionalization_rejected_without_cache_copy(packaged):
    inputs = _inputs()
    cache = inputs.pop("kv_cache")

    def forward(cache):
        return packaged.quant_mla_prolog(kv_cache=cache, **inputs)[0]

    with pytest.raises(RuntimeError, match="functionalization with cache clone/copy-back"):
        torch.func.functionalize(forward)(cache)


def test_compile_eager_preserves_cache_write_when_outputs_unused(packaged):
    # A CPU-only stand-in exercises the actual schema/dispatcher, not the vendor.
    # backend=eager tests FX side effects only; it does not establish ACLGraph support.
    inputs = _inputs()
    inputs["kv_cache"].zero_()
    cache = inputs.pop("kv_cache")
    slots = inputs.pop("cache_index")
    pointer = cache.untyped_storage().data_ptr()
    calls = []

    def cpu_impl(*args, **kwargs):
        names = [arg.name for arg in packaged.quant_mla_prolog._schema.arguments]
        values = dict(zip(names, args), **kwargs)
        target = values["kv_cache"]
        calls.append(target.untyped_storage().data_ptr())
        target.view(-1, 512)[values["cache_index"]] = len(calls)
        return packaged._quant_mla_prolog_fake(**values)

    def forward(cache, slots):
        packaged.quant_mla_prolog(kv_cache=cache, cache_index=slots, **inputs)
        return cache.sum()

    library = torch.library.Library("vllm_ascend", "IMPL", "CPU")
    try:
        library.impl("quant_mla_prolog", cpu_impl)
        compiled = torch.compile(forward, backend="eager", fullgraph=True)
        assert compiled(cache, slots).item() == 3 * 512
        assert compiled(cache, slots + 3).item() == 9 * 512
    finally:
        library._destroy()
    assert calls == [pointer, pointer]


@pytest.mark.parametrize("capturing", [False, True])
def test_public_entry_owns_capture_selection(packaged, monkeypatch, capturing):
    # This is an API/dispatch contract stub, not vendor or ACLGraph execution.
    inputs = _inputs()
    routes = []
    monkeypatch.setattr(torch, "npu", Mock(is_current_stream_capturing=Mock(return_value=capturing)), raising=False)
    monkeypatch.setattr(inputs["cache_index"], "cpu", Mock(side_effect=AssertionError("wrapper must not read slots")))

    def public(**kwargs):
        routes.append("graph" if torch.npu.is_current_stream_capturing() else "checked")
        assert kwargs["cache_index"] is inputs["cache_index"]
        assert kwargs["kv_cache"].untyped_storage().data_ptr() == inputs["kv_cache"].untyped_storage().data_ptr()
        assert kwargs["descale_x"].dtype == torch.float8_e8m0fnu
        q, qs, qr, qrs = packaged._quant_mla_prolog_fake(**inputs)
        return q, qs, qr, qrs.view(torch.float8_e8m0fnu)

    loader = Mock(return_value=public)
    monkeypatch.setattr(packaged, "_get_dsl_op", loader)
    result = packaged._quant_mla_prolog_impl(**inputs)
    assert routes == ["graph" if capturing else "checked"]
    assert result[3].dtype == torch.uint8
    loader.assert_called_once()
