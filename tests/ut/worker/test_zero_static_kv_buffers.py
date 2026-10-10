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


def test_zero_static_kv_buffers_fp8_non_contiguous_copies_scalar_not_full_buffer():
    # Regression test for the CI OOM on a nearly-full card: the fallback for
    # non-contiguous fp8 buffers must broadcast a scalar zero and must never
    # allocate a zeros_like() buffer the size of the resident KV pool.
    copy_src_shapes = []

    class _CopyRecorder(torch.Tensor):
        def copy_(self, src):
            copy_src_shapes.append(tuple(src.shape))
            return super().copy_(src)

    t = torch.ones(4, 4, dtype=torch.uint8).view(torch.float8_e4m3fn).as_subclass(_CopyRecorder).t()
    assert not t.is_contiguous()
    ctx = {"layer0": _FakeAttnModule([t])}
    _zero_static_kv_buffers(_make_runner(ctx))
    assert copy_src_shapes == [()]
    assert torch.count_nonzero(t.contiguous().view(torch.int8)).item() == 0


def test_zero_static_kv_buffers_chunks_large_non_contiguous(monkeypatch):
    # Mirrors the a2 CI OOM: a non-contiguous buffer whose zero_() wants a
    # same-size temporary must be re-zeroed in dim-0 chunks so each chunk's
    # kernel-side temporary stays bounded (full-size copy_ wanted 6.13 GiB
    # with only 4.33 GiB free).
    monkeypatch.setattr("vllm_ascend.worker.model_runner_v1._ZERO_CHUNK_BYTES", 64)
    copy_src_shapes = []

    class _OomRecorder(torch.Tensor):
        def zero_(self):
            raise torch.OutOfMemoryError("NPU out of memory")

        def copy_(self, src):
            copy_src_shapes.append(tuple(src.shape))
            return super().copy_(src)

    t = torch.ones(8, 8).as_subclass(_OomRecorder).t()
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
    # RuntimeError from zero_() on ordinary dtypes is not a recoverable
    # dtype/layout issue: it must abort startup instead of leaving a
    # poisoned buffer behind.
    class _NoZeroTensor(torch.Tensor):
        def zero_(self):
            raise RuntimeError("zero_ not implemented for this dtype")

    t = torch.ones(4).as_subclass(_NoZeroTensor)
    ctx = {"layer0": _FakeAttnModule([t])}
    with pytest.raises(RuntimeError, match="zero_ not implemented"):
        _zero_static_kv_buffers(_make_runner(ctx))
