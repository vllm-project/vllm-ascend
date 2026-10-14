# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops import kda_state_copy_plan as production


@pytest.mark.parametrize("retains_constexpr", [False, True])
@pytest.mark.parametrize("to_cache", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_fallback_matches_compiled_launcher_signature(monkeypatch, retains_constexpr, to_cache, index_dtype):
    """Model both Triton ABIs and check pointer direction and invalid-row masks."""
    signature = {"src_ptrs": "*i64", "dst_ptrs": "*i64", "sizes": "*i64"}
    if retains_constexpr:
        signature["BLOCK_SIZE"] = "constexpr"
    launches: list[tuple[tuple[int, int, int], tuple[torch.Tensor, ...]]] = []

    class CompiledKernel:
        src = SimpleNamespace(signature=signature)

        def __getitem__(self, grid):
            def launch(*args):
                assert len(args) == len(signature)
                if retains_constexpr:
                    assert args[-1] == warmup.call_args.kwargs["BLOCK_SIZE"]
                launches.append((grid, args[:3]))

            return launch

    warmup = Mock(return_value=CompiledKernel())
    kernel = Mock(pre_run_hooks=[])
    kernel.__getitem__ = Mock(return_value=warmup)
    monkeypatch.setattr(production, "batch_memcpy_kernel", kernel)
    monkeypatch.setattr(production, "_configuration", lambda: ())
    monkeypatch.setattr(production, "cache_signature", lambda state: ())
    monkeypatch.setattr(torch.npu, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
    zeros, tensor = torch.zeros, torch.tensor
    monkeypatch.setattr(torch, "zeros", lambda *args, device=None, **kwargs: zeros(*args, **kwargs))
    monkeypatch.setattr(torch, "tensor", lambda *args, device=None, **kwargs: tensor(*args, **kwargs))
    metadata = Mock(ndim=4, shape=(8, 1, 2, 3), device=SimpleNamespace(type="npu"))
    metadata.stride.side_effect = lambda dim=None: (6, 6, 3, 1) if dim is None else (6, 6, 3, 1)[dim]
    metadata.element_size.return_value = 4
    plan = production.StridedKDAFallbackPlan.prepare(metadata, 4)
    plan.seal()
    warmup.assert_called_once()
    kernel.__getitem__.side_effect = AssertionError("serving must not enter JIT")

    state = torch.empty((8, 1, 2, 3))
    packed = torch.ones((4, 1, 2, 3))
    indices = tensor([3, -1, 8, 0], dtype=index_dtype)
    plan._copy(state, packed, indices, to_cache=to_cache)
    ((grid, (src, dst, sizes)),) = launches
    assert grid == (4, 1, 1)
    torch.testing.assert_close(sizes, tensor([24, 0, 0, 24]))
    cache_ptrs = state.data_ptr() + tensor([3, 0, 7, 0]) * 24
    packed_ptrs = packed.data_ptr() + torch.arange(4) * 24
    torch.testing.assert_close(src, packed_ptrs if to_cache else cache_ptrs)
    torch.testing.assert_close(dst, cache_ptrs if to_cache else packed_ptrs)
    torch.testing.assert_close(packed, torch.ones_like(packed) if to_cache else torch.zeros_like(packed))
