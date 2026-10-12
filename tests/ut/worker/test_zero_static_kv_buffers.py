from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.worker.model_runner_v1 import _zero_static_kv_buffers


class _FakeAttnModule(torch.nn.Module):
    def __init__(self, kv_cache):
        super().__init__()
        self.kv_cache = kv_cache


def _make_runner(static_forward_context):
    runner = MagicMock()
    runner.compilation_config.static_forward_context = static_forward_context
    return runner


def _walk(obj):
    if isinstance(obj, torch.Tensor):
        assert torch.count_nonzero(obj).item() == 0
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v)


def test_zero_static_kv_buffers_zeroes_everything():
    ctx = {
        "layer0": _FakeAttnModule([torch.full((4, 4), float("nan")), torch.ones(8)]),
        "layer1": _FakeAttnModule({"k_cache": torch.tensor([[1.0, 2.0]]), "v_cache": torch.full((3,), float("inf"))}),
        "layer2": _FakeAttnModule(torch.ones(2, 2, dtype=torch.bfloat16)),
        "layer3": _FakeAttnModule(None),
        "layer4": _FakeAttnModule([]),
        "layer5": _FakeAttnModule(torch.empty(0)),
    }
    _zero_static_kv_buffers(_make_runner(ctx))
    for mod in ctx.values():
        _walk(mod.kv_cache)


def test_zero_static_kv_buffers_tolerates_empty_context():
    _zero_static_kv_buffers(_make_runner({}))
    _zero_static_kv_buffers(_make_runner(None))


def test_zero_static_kv_buffers_fp8_via_int8_view():
    # fp8 storages may not implement zero_(); the runner zeroes them through
    # an int8 reinterpretation of the same bytes.
    t = torch.ones(4, dtype=torch.uint8).view(torch.float8_e4m3fn)
    ctx = {"layer0": _FakeAttnModule([t])}
    _zero_static_kv_buffers(_make_runner(ctx))
    assert torch.count_nonzero(t.view(torch.int8)).item() == 0


def test_zero_static_kv_buffers_non_contiguous_never_calls_strided_zero():
    # Regression test for the DeepSeek-V4-Flash nightly failure that got
    # #17659 reverted (#18307): on a non-contiguous view the strided zero
    # kernel fails tiling on NPU (aclnnInplaceZero AsStridedAiCore, error
    # 561103), so zero_() must not be attempted on it at all.
    class _NoZeroTensor(torch.Tensor):
        def zero_(self):
            raise RuntimeError("zero_ must not be called on a strided view")

    storage = torch.ones(64).as_subclass(_NoZeroTensor)
    # Padded-page view: shape (4, 8), stride (16, 1) -> 8-float hole after
    # every 8-float row, exactly like the page_size_padded as_strided views.
    t = torch.as_strided(storage, (4, 8), (16, 1))
    assert not t.is_contiguous()
    ctx = {"layer0": _FakeAttnModule([t])}
    _zero_static_kv_buffers(_make_runner(ctx))
    # The view's byte span, interior padding holes included, must be wiped.
    # Bytes past the last element (the final row's trailing hole) are outside
    # the span and are never read as data, so they are left alone.
    assert torch.count_nonzero(t).item() == 0
    assert torch.count_nonzero(storage[:56]).item() == 0


def test_zero_static_kv_buffers_non_contiguous_with_storage_offset():
    # The span fill must honor storage_offset instead of restarting at 0.
    storage = torch.ones(64)
    t = torch.as_strided(storage, (2, 8), (16, 1), storage_offset=16)
    assert not t.is_contiguous()
    ctx = {"layer0": _FakeAttnModule([t])}
    _zero_static_kv_buffers(_make_runner(ctx))
    assert torch.count_nonzero(t).item() == 0
    # span = bytes 16..39 (both rows plus the interior hole between them)
    assert torch.count_nonzero(storage[16:40]).item() == 0
    # Bytes outside the view's span are not this tensor's to wipe.
    assert torch.count_nonzero(storage[:16]).item() == 16
    assert torch.count_nonzero(storage[40:]).item() == 24


def test_zero_static_kv_buffers_non_contiguous_fp8_span():
    storage = torch.ones(4, 16, dtype=torch.uint8)
    t = storage.view(torch.float8_e4m3fn).as_strided((4, 8), (16, 1))
    assert not t.is_contiguous()
    ctx = {"layer0": _FakeAttnModule([t])}
    _zero_static_kv_buffers(_make_runner(ctx))
    assert torch.count_nonzero(t.view(torch.uint8)).item() == 0
    assert torch.count_nonzero(storage.flatten()[:56]).item() == 0


def test_zero_static_kv_buffers_chunks_large_non_contiguous(monkeypatch):
    # Mirrors the a2 CI OOM: if the storage span cannot be re-viewed, the
    # fallback must re-zero in dim-0 chunks so each chunk's kernel-side
    # temporary stays bounded (full-size copy_ wanted 6.13 GiB with only
    # 4.33 GiB free).
    monkeypatch.setattr("vllm_ascend.worker.model_runner_v1._ZERO_CHUNK_BYTES", 64)

    def _broken_span(t):
        raise RuntimeError("cannot re-view storage")

    monkeypatch.setattr("vllm_ascend.worker.model_runner_v1._zero_storage_span", _broken_span)
    copy_src_shapes = []

    class _CopyRecorder(torch.Tensor):
        def copy_(self, src):
            copy_src_shapes.append(tuple(src.shape))
            return super().copy_(src)

    t = torch.ones(8, 8).as_subclass(_CopyRecorder).t()
    assert not t.is_contiguous()
    ctx = {"layer0": _FakeAttnModule([t])}
    _zero_static_kv_buffers(_make_runner(ctx))
    # 64 bytes / 32-byte row = 2 rows per chunk -> 4 chunks, scalar each
    assert copy_src_shapes == [(), (), (), ()]
    assert torch.count_nonzero(t).item() == 0


def test_zero_static_kv_buffers_oom_falls_back_to_int8_view():
    # Regression test for the a2 CI OOM: plain zero_() can materialize a
    # same-size temporary on NPU (6-12 GiB extra on a nearly-full card).
    # Only OutOfMemoryError may take the in-place int8-view fallback.
    class _OomTensor(torch.Tensor):
        def zero_(self):
            if self.dtype != torch.int8:
                raise torch.OutOfMemoryError("NPU out of memory")
            return super().zero_()

    t = torch.ones(4).as_subclass(_OomTensor)
    ctx = {"layer0": _FakeAttnModule([t])}
    _zero_static_kv_buffers(_make_runner(ctx))
    assert torch.count_nonzero(t).item() == 0


def test_zero_static_kv_buffers_propagates_unexpected_errors():
    # RuntimeError from zero_() on ordinary contiguous dtypes is not a
    # recoverable dtype/layout issue: it must abort startup instead of
    # leaving a poisoned buffer behind.
    class _NoZeroTensor(torch.Tensor):
        def zero_(self):
            raise RuntimeError("zero_ not implemented for this dtype")

    t = torch.ones(4).as_subclass(_NoZeroTensor)
    ctx = {"layer0": _FakeAttnModule([t])}
    with pytest.raises(RuntimeError, match="zero_ not implemented"):
        _zero_static_kv_buffers(_make_runner(ctx))
