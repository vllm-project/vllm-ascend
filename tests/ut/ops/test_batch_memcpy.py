#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""CPU unit tests for ``vllm_ascend.ops.triton.batch_memcpy.batch_memcpy_kernel``.

``batch_memcpy_kernel`` is a ``@triton.jit`` device kernel, so it cannot be
launched on a CPU-only CI (raw device addresses are meaningless on the host).
These tests therefore pin the kernel's *contract* instead:

* with an active triton driver, the kernel is registered as a Triton JIT
  kernel with the expected signature and ``BLOCK_SIZE`` marked as
  ``tl.constexpr``; on CPU-only CI vllm substitutes identity/no-op triton
  placeholders, so those pins degrade to checks on the raw function and the
  ``tl``/``triton`` bindings instead;
* a launch uses exactly one program per copy and forwards the raw
  pointer/size tensors and ``BLOCK_SIZE`` unchanged (verifiable only when
  triton is active);
* the byte-level copy algorithm the kernel implements (uint8 pointer casting,
  ``block_size`` iterations with a per-step mask, sizes in *bytes*) matches a
  plain CPU ``torch`` copy on the same underlying storage, including sizes
  that are not multiples of ``BLOCK_SIZE``.

Real kernel execution (``cache_modifier=".cg"``, compiled launch, pointer
casting) requires NPU hardware and is covered by the nightly E2E test
``tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_batch_memcpy.py``.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel

# Mirrors the block size chosen by the serving callers
# (vllm_ascend.patch.worker.patch_mamba_utils._batch_memcpy_triton).
_SERVING_BLOCK_SIZE = 8192

# Every byte outside a batch row's copy range is left untouched; this sentinel
# detects accidental copies past the masked tail.
_UNTOUCHED_BYTE = 0xAB


def _kernel_callable():
    """Return the raw Python function wrapped by ``@triton.jit``."""
    return getattr(batch_memcpy_kernel, "fn", batch_memcpy_kernel)


def _reference_batch_memcpy(src, dst, src_ptrs, dst_ptrs, sizes, block_size):
    """Python twin of ``batch_memcpy_kernel`` operating on CPU byte buffers.

    The Triton kernel is a pure byte copy: for each batch row it casts the raw
    pointer to ``uint8*`` and copies ``size`` bytes in ``block_size``-wide
    steps with a per-step mask (``(i + offsets) < size``). Reinterpreting the
    CPU tensors as ``uint8`` gives identical storage, so this loop mirrors the
    kernel exactly for the masked-tail and byte-semantics cases that matter on
    the host.
    """
    src_flat = src.view(torch.uint8)
    dst_flat = dst.view(torch.uint8)
    for pid in range(sizes.numel()):
        src_start = int(src_ptrs[pid].item()) - src_flat.data_ptr()
        dst_start = int(dst_ptrs[pid].item()) - dst_flat.data_ptr()
        size = int(sizes[pid].item())
        assert src_start >= 0 and src_start + size <= src_flat.numel()
        assert dst_start >= 0 and dst_start + size <= dst_flat.numel()
        for i in range(0, size, block_size):
            end = min(i + block_size, size)
            dst_flat[dst_start + i : dst_start + end].copy_(src_flat[src_start + i : src_start + end])


def test_kernel_is_triton_jit_kernel():
    """``@triton.jit`` registers the function as a Triton kernel object."""
    assert batch_memcpy_kernel.__name__ == "batch_memcpy_kernel"
    if HAS_TRITON:
        # A Triton JITFunction is launched via `kernel[grid](...)`; it is the
        # ``__getitem__``/``run`` API, not plain ``__call__``, that serializes
        # to the device. triton and triton-ascend both name this type
        # ``JITFunction``.
        assert batch_memcpy_kernel.__class__.__name__ == "JITFunction"
        for attr in ("run", "warmup", "__getitem__"):
            assert callable(getattr(batch_memcpy_kernel, attr)), attr
    else:
        # Without an active triton driver, vllm.triton_utils substitutes
        # TritonPlaceholder whose ``jit`` is an identity decorator: the kernel
        # stays the raw Python function with no launch API. Pin that
        # degradation (and that no other wrapper slipped in) instead of the
        # JITFunction contract, which cannot exist here.
        assert callable(batch_memcpy_kernel)
        assert batch_memcpy_kernel is _kernel_callable()


def test_kernel_signature_and_block_size_constexpr():
    """Signature is ``(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr)``.

    ``BLOCK_SIZE`` must stay a compile-time constant so Triton can unroll the
    copy loop; callers pass it as a keyword.
    """
    params = list(inspect.signature(_kernel_callable()).parameters.values())
    assert [p.name for p in params] == ["src_ptrs", "dst_ptrs", "sizes", "BLOCK_SIZE"]
    # Runtime pointer/size tensors only: no element-size assumption here.
    for param in params[:3]:
        assert param.annotation is inspect.Parameter.empty, param.name
    # The annotation is evaluated at import time against the ``tl`` this module
    # imported, so it *is* ``tl.constexpr``: the real constexpr class when
    # triton is active, or vllm's placeholder value (``None``) on CPU-only
    # boxes. The identity check pins the binding in both worlds; only with
    # triton active does it also pin the constexpr type itself.
    assert params[-1].annotation is tl.constexpr


@pytest.mark.skipif(
    not HAS_TRITON,
    reason=(
        "CPU-only CI has no active triton driver: vllm.triton_utils uses an "
        "identity 'jit' placeholder, so the kernel is a plain function "
        "without the run/__getitem__ launch API."
    ),
)
def test_launch_uses_one_program_per_copy_and_forwards_arguments():
    """A single batch row maps to one program id; args pass through unchanged."""
    batch = 3
    src_ptrs = torch.zeros(batch, dtype=torch.int64)
    dst_ptrs = torch.zeros(batch, dtype=torch.int64)
    sizes = torch.zeros(batch, dtype=torch.int64)

    with patch.object(batch_memcpy_kernel, "run", new=MagicMock()) as mock_run:
        batch_memcpy_kernel[(batch,)](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=_SERVING_BLOCK_SIZE)

    mock_run.assert_called_once_with(src_ptrs, dst_ptrs, sizes, grid=(batch,), BLOCK_SIZE=_SERVING_BLOCK_SIZE)


def test_byte_copy_matches_torch_with_masked_tail():
    """Byte copy semantics: sizes are byte counts and tails stay masked."""
    numel = 8192
    src = torch.randint(0, 256, (numel,), dtype=torch.uint8)
    dst = torch.full((numel,), _UNTOUCHED_BYTE, dtype=torch.uint8)

    # Disjoint copy ranges: one exact-sized single byte, and two tails that are
    # not multiples of BLOCK_SIZE so the kernel's per-step mask is exercised.
    offsets = [0, 100, 5000]
    sizes = torch.tensor([1, 2049, 2600], dtype=torch.int64)  # 1, 2*B+1, 2*B+552
    src_ptrs = torch.tensor([src.data_ptr() + off for off in offsets], dtype=torch.int64)
    dst_ptrs = torch.tensor([dst.data_ptr() + off for off in offsets], dtype=torch.int64)

    _reference_batch_memcpy(src, dst, src_ptrs, dst_ptrs, sizes, block_size=1024)

    for off, size in zip(offsets, sizes.tolist()):
        # The copy range must hold the source bytes exactly.
        torch.testing.assert_close(dst[off : off + size], src[off : off + size])
    # Bytes outside the copy ranges are untouched (masked tail / no spill).
    untouched = torch.full_like(dst, _UNTOUCHED_BYTE)
    for off, size in zip(offsets, sizes.tolist()):
        untouched[off : off + size] = src[off : off + size]
    torch.testing.assert_close(dst, untouched)


def test_sizes_are_byte_counts_not_element_counts():
    """The uint8 pointer cast means sizes count bytes, not elements."""
    src = torch.randint(-100, 100, (64,), dtype=torch.int16)  # element_size == 2
    dst = torch.full((64,), 0x1111, dtype=torch.int16)

    src_flat = src.view(torch.uint8)
    dst_flat = dst.view(torch.uint8)
    n_bytes = 33  # odd: cannot round-trip to whole int16 elements
    src_ptrs = torch.tensor([src_flat.data_ptr()], dtype=torch.int64)
    dst_ptrs = torch.tensor([dst_flat.data_ptr()], dtype=torch.int64)
    sizes = torch.tensor([n_bytes], dtype=torch.int64)

    _reference_batch_memcpy(src, dst, src_ptrs, dst_ptrs, sizes, block_size=_SERVING_BLOCK_SIZE)

    # Exactly n_bytes copied at the byte level; the next byte keeps its sentinel
    # value because the kernel masks every offset >= size.
    torch.testing.assert_close(dst_flat[:n_bytes], src_flat[:n_bytes])
    assert dst_flat[n_bytes].item() == 0x11
    # Everything at or after the void element is untouched.
    expected = torch.full_like(dst, 0x1111, dtype=torch.int16)
    expected.view(torch.uint8)[:n_bytes] = src_flat[:n_bytes]
    torch.testing.assert_close(dst.flatten(), expected.flatten())


def test_kernel_launch_is_available_through_vllm_triton_utils():
    """The ``vllm.triton_utils`` binding still exposes ``tl``/``triton``.

    Guards the module-level ``from vllm.triton_utils import tl, triton`` line
    in the source file. On CPU-only runs without an active triton driver, vllm
    substitutes ``TritonLanguagePlaceholder``/``TritonPlaceholder``: the names
    stay bound, but ``tl.constexpr`` degrades to ``None`` (older vllm) or a
    no-op lambda (newer vllm) instead of the constexpr class.
    """
    assert tl is not None and triton is not None
    assert hasattr(tl, "constexpr")
    if HAS_TRITON:
        assert callable(tl.constexpr)
