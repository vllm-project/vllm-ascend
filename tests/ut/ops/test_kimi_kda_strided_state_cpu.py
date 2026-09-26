# SPDX-License-Identifier: Apache-2.0
"""Check selected-state byte copies without loading the NPU/vLLM runtime."""

import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class _Memcpy:
    def __init__(self, *, copy=True):
        self.copy = copy
        self.calls = []

    def __getitem__(self, grid):
        def run(src_ptrs, dst_ptrs, sizes, **kwargs):
            assert src_ptrs.dtype == dst_ptrs.dtype == sizes.dtype == torch.int64
            assert grid == (sizes.numel(),)
            self.calls.append((src_ptrs.clone(), dst_ptrs.clone(), sizes.clone()))
            if self.copy:
                for src, dst, size in zip(src_ptrs.tolist(), dst_ptrs.tolist(), sizes.tolist()):
                    if size:
                        ctypes.memmove(dst, src, size)

        return run


def load_copy(kernel):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/kimi_kda.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if getattr(node, "name", None) == "_copy_strided_recurrent_states"]
    scope = {"torch": torch, "batch_memcpy_kernel": kernel}
    exec(compile(tree, str(path), "exec"), scope)
    return scope["_copy_strided_recurrent_states"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gather_scatter_preserve_page_gaps_and_storage_offset(dtype):
    kernel = _Memcpy()
    copy = load_copy(kernel)
    backing = torch.full((7 * 64 + 16,), -23, dtype=dtype)
    state = backing.as_strided((7, 2, 3, 4), (64, 12, 4, 1), storage_offset=16)
    for idx in range(7):
        state[idx].fill_(idx + 1)
    original = backing.clone()
    indices = torch.tensor([6, 1, -1, 3, 7], dtype=torch.int32)
    gathered = torch.full((5, 2, 3, 4), 99, dtype=dtype)

    copy(state, gathered, indices, to_cache=False)
    expected = torch.zeros_like(gathered)
    expected[0], expected[1], expected[3] = state[6], state[1], state[3]
    torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
    torch.testing.assert_close(backing, original, rtol=0, atol=0)
    src, _, sizes = kernel.calls[-1]
    assert src[0] == state.data_ptr() + 6 * 64 * state.element_size()
    assert sizes.tolist() == [24 * state.element_size(), 24 * state.element_size(), 0, 24 * state.element_size(), 0]

    final_state = torch.arange(5, dtype=dtype).view(5, 1, 1, 1).expand_as(gathered).contiguous() + 10
    expected_backing = original.clone()
    expected_state = expected_backing.as_strided(state.shape, state.stride(), storage_offset=16)
    expected_state[6], expected_state[1], expected_state[3] = final_state[0], final_state[1], final_state[3]
    copy(state, final_state, indices, to_cache=True)
    # Compare the entire allocation, including every adjacent layer/page gap.
    torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)


def test_state_pointer_multiply_does_not_wrap_at_four_gib():
    kernel = _Memcpy(copy=False)
    copy = load_copy(kernel)
    state = SimpleNamespace(
        ndim=4,
        shape=(1607, 12, 128, 128),
        dtype=torch.float32,
        stride=lambda dim=None: (5308416, 16384, 128, 1) if dim is None else (5308416, 16384, 128, 1)[dim],
        element_size=lambda: 4,
        data_ptr=lambda: 0x100000000 + 19519488,
    )
    packed = torch.empty(3, 12, 128, 128)
    copy(state, packed, torch.tensor([1606, 0, -1], dtype=torch.int32), to_cache=False)
    src, _, sizes = kernel.calls[-1]
    assert src[0].item() == state.data_ptr() + 1606 * 21233664
    assert sizes.tolist() == [786432, 786432, 0]
    assert torch.count_nonzero(packed[2]) == 0


def test_copy_rejects_inner_strides_without_touching_memory():
    kernel = _Memcpy()
    copy = load_copy(kernel)
    state = torch.empty(4, 2, 3, 4).transpose(-1, -2)
    packed = torch.empty(1, *state.shape[1:])
    with pytest.raises(ValueError, match="first-axis"):
        copy(state, packed, torch.tensor([0]), to_cache=False)
    assert kernel.calls == []


def test_empty_copy_does_not_launch_kernel():
    kernel = _Memcpy()
    copy = load_copy(kernel)
    state = torch.empty(4, 2, 3, 4)
    packed = torch.empty(0, 2, 3, 4)
    copy(state, packed, torch.empty(0, dtype=torch.int64), to_cache=False)
    assert kernel.calls == []
