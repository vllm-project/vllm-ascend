# SPDX-License-Identifier: Apache-2.0
"""Host pools must request pinned storage while preserving size and ownership."""

import ast
import math
import mmap
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/hisparse_optimized.py"


@pytest.mark.parametrize(
    "shape,dtype", [((1,), torch.uint8), ((128, 656), torch.uint8), ((3, 128, 128), torch.float32)]
)
def test_pinned_pool_alignment_dtype_and_live_allocation_owner(shape, dtype):
    allocations = []

    def empty(*args, **kwargs):
        if "pin_memory" in kwargs:
            assert kwargs.pop("pin_memory") is True
            owner = torch.empty(*args, **kwargs)
            allocations.append(weakref.ref(owner))
            return owner
        return torch.empty(*args, **kwargs)

    def mapping(raw):
        assert raw.is_contiguous() and raw.data_ptr() % mmap.PAGESIZE == 0
        return SimpleNamespace(tensor=raw)

    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "allocate_mapped_host")
    namespace = {
        "torch": SimpleNamespace(empty=empty, uint8=torch.uint8, Tensor=torch.Tensor, dtype=torch.dtype),
        "math": math,
        "mmap": mmap,
        "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        "MappedHostTensor": mapping,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    tensor = namespace[node.name](shape, dtype)
    assert tensor.shape == shape and tensor.dtype == dtype
    assert len(allocations) == 1 and allocations[0]() is not None
    assert tensor.data_ptr() % mmap.PAGESIZE == 0
    assert tensor.untyped_storage().data_ptr() == allocations[0]().untyped_storage().data_ptr()
    assert tensor._hisparse_registration.tensor.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
