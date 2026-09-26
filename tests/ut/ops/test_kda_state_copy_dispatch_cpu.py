# SPDX-License-Identifier: Apache-2.0
"""Exercise KDA's native copy dispatch and fallback without an NPU runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_prefill(scope):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/kimi_kda.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_run_prefill")
    namespace = {"torch": torch, **scope}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_run_prefill"]


@pytest.mark.parametrize("native,strided", [(True, False), (True, True), (False, False), (False, True)])
@pytest.mark.parametrize("keep_selected,kernel_cu", [(False, False), (True, True)])
def test_prefill_native_copy_and_legacy_routes(native, strided, keep_selected, kernel_cu):
    dtype = torch.bfloat16
    payload, stride, offset = 24, 64 if strided else 24, 16
    backing = torch.full((offset + 3 * stride + payload + 16,), -17, dtype=dtype)
    cache = backing.as_strided((4, 2, 3, 4), (stride, 12, 4, 1), offset)
    for row in range(4):
        cache[row].fill_(row + 1)
    original = backing.clone()
    indices = torch.tensor([3, 1, 2], dtype=torch.int32)
    flags = torch.tensor([True, True, False])
    keep = torch.tensor([True, False, True]) if keep_selected else None
    effective_indices = indices if keep is None else indices[keep]
    effective_flags = flags if keep is None else flags[keep]
    count = effective_indices.numel()
    expected_initial = cache[effective_indices].clone()
    expected_initial[~effective_flags] = 0
    # Return a noncontiguous FP32 final state so the integration must convert
    # to the cache dtype and pack the scatter buffer.
    final_state = torch.arange(count * payload, dtype=torch.float32).reshape(count, 2, 4, 3).transpose(-1, -2) / 7
    expected_backing = original.clone()
    expected_cache = expected_backing.as_strided(cache.shape, cache.stride(), offset)
    expected_cache[effective_indices] = final_state.to(dtype)
    output = torch.tensor([123])
    calls = []
    host_cu = torch.tensor([0, 2, 5])
    device_cu = torch.tensor([0, 3, 5]) if kernel_cu else None
    chunk_indices = object()
    metadata = SimpleNamespace(
        cu_seqlens_host=host_cu,
        cu_seqlens_kern=device_cu,
        keep_meta=keep,
        chunk_indices_chunk64_host=chunk_indices,
    )
    model = SimpleNamespace(A_log=object(), dt_bias=object(), gate_lower_bound=-5.0)
    operands = [torch.tensor([index]) for index in range(5)]

    def copy(state, packed, selected, *, to_cache=False, has_initial_state=None):
        calls.append("scatter" if to_cache else "gather")
        assert state is cache
        assert packed.is_contiguous()
        assert packed.dtype == dtype
        torch.testing.assert_close(selected, effective_indices)
        if to_cache:
            assert has_initial_state is None
            state[selected] = packed
        else:
            packed.copy_(state[selected])
            if native:
                torch.testing.assert_close(has_initial_state, effective_flags)
                packed[~has_initial_state] = 0
            else:
                assert has_initial_state is None

    def clear(packed, selected_flags):
        assert not native, "Native gather already clears rows without an initial state"
        calls.append("clear")
        torch.testing.assert_close(selected_flags, effective_flags)
        packed[~selected_flags] = 0

    def chunk(*args, **kwargs):
        calls.append("chunk")
        for actual, expected in zip(args[:5], operands):
            assert actual is expected
        torch.testing.assert_close(args[5], expected_initial, rtol=0, atol=0)
        assert args[6] is (device_cu if kernel_cu else host_cu)
        assert args[7] is chunk_indices
        assert args[8] is model.A_log and args[9] is model.dt_bias
        assert kwargs == {"lower_bound": -5.0}
        return output, final_state

    run = _load_prefill(
        {
            "supports_kda_state_copy": lambda state: native,
            "copy_kda_states": copy,
            "_copy_strided_recurrent_states": copy,
            "clear_ssm_states": clear,
            "run_chunk_kda": chunk,
        }
    )
    assert run(model, *operands, cache, indices, flags, metadata) is output
    torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)
    assert calls == (
        ["gather", "chunk", "scatter"]
        if native
        else ["gather", "clear", "chunk", "scatter"]
        if strided
        else ["clear", "chunk"]
    )


@pytest.mark.parametrize(
    "device,is_950,registered,expected",
    [("cpu", True, True, False), ("npu", False, True, False), ("npu", True, False, False), ("npu", True, True, True)],
)
def test_copy_support_requires_npu_950_and_registered_extension(device, is_950, registered, expected):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/kda_state_copy.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    operation = object()
    extension = SimpleNamespace(**({"kda_state_copy": operation} if registered else {}))
    fake_torch = SimpleNamespace(
        Tensor=torch.Tensor, float32=torch.float32, bfloat16=torch.bfloat16, ops=SimpleNamespace(_C_ascend=extension)
    )
    scope = {"torch": fake_torch, "is_950": lambda: is_950}
    exec(compile(tree, str(path), "exec"), scope)
    state = SimpleNamespace(
        device=SimpleNamespace(type=device),
        dtype=torch.float32,
        ndim=4,
        shape=(3, 2, 3, 4),
        stride=lambda dim=None: (64, 12, 4, 1) if dim is None else (64, 12, 4, 1)[dim],
    )
    assert scope["supports_kda_state_copy"](state) is expected
    if expected:
        # Cache layouts and dtypes outside the native specialization retain
        # their preceding path instead of reaching a new operator guard error.
        state.dtype = torch.float16
        assert not scope["supports_kda_state_copy"](state)
        state.dtype = torch.bfloat16
        assert scope["supports_kda_state_copy"](state)
        state.ndim = 3
        assert not scope["supports_kda_state_copy"](state)
        state.ndim, state.shape = 4, (0, 2, 3, 4)
        assert not scope["supports_kda_state_copy"](state)
        state.shape = (3, 2, 3, 4)
        state.stride = lambda dim=None: (64, 12, 1, 3) if dim is None else (64, 12, 1, 3)[dim]
        assert not scope["supports_kda_state_copy"](state)
        state.stride = lambda dim=None: (12, 12, 4, 1) if dim is None else (12, 12, 4, 1)[dim]
        assert not scope["supports_kda_state_copy"](state)


@pytest.mark.parametrize("convert_metadata", [False, True])
def test_copy_wrapper_preserves_clear_metadata_contract(convert_metadata):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/kda_state_copy.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    calls = []
    fake_torch = SimpleNamespace(
        Tensor=torch.Tensor,
        bool=torch.bool,
        ops=SimpleNamespace(_C_ascend=SimpleNamespace(kda_state_copy=lambda *args: calls.append(args))),
    )
    scope = {"torch": fake_torch, "is_950": lambda: True}
    exec(compile(tree, str(path), "exec"), scope)
    state = torch.empty((4, 2, 3, 4))
    packed = torch.empty((2, 2, 3, 4))
    indices = torch.tensor([3, 1], dtype=torch.int32)
    flags = torch.tensor([[1, 0]]) if convert_metadata else torch.tensor([True, False])
    scope["copy_kda_states"](state, packed, indices, has_initial_state=flags)
    assert len(calls) == 1
    assert calls[0][0] is state and calls[0][1] is packed and calls[0][2] is indices
    assert calls[0][4] is False
    torch.testing.assert_close(calls[0][3], torch.tensor([True, False]))
    if not convert_metadata:
        assert calls[0][3] is flags


@pytest.mark.parametrize("speculative", [False, True])
def test_decode_recurrent_keeps_original_cache_and_operator(speculative):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/kimi_kda.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_run_recurrent"
    )
    calls = []
    output = object()

    def recurrent(*args, **kwargs):
        calls.append((args, kwargs))
        return output

    def forbidden_copy(*args, **kwargs):
        pytest.fail("Decode must not enter prefill state copy or its dispatch check")

    scope = {
        "torch": torch,
        "run_recurrent_kda": recurrent,
        "copy_kda_states": forbidden_copy,
        "supports_kda_state_copy": forbidden_copy,
        "_copy_strided_recurrent_states": forbidden_copy,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    cache = torch.zeros(6, 2, 3, 4)[1::2]
    inputs = [torch.tensor([index]) for index in range(5)]
    starts, slots = torch.tensor([0, 1], dtype=torch.int32), torch.tensor([2], dtype=torch.int64)
    accepted = torch.tensor([1]) if speculative else None
    model = SimpleNamespace(A_log=object(), dt_bias=object(), gate_lower_bound=-5.0)
    assert scope["_run_recurrent"](model, *inputs, cache, starts, slots, num_accepted_tokens=accepted) is output
    assert len(calls) == 1
    args, kwargs = calls[0]
    for actual, expected in zip(args, [*inputs, cache, starts, slots, model.A_log, model.dt_bias]):
        assert actual is expected
    assert kwargs["lower_bound"] == -5.0
    assert kwargs["num_accepted_tokens"] is accepted
