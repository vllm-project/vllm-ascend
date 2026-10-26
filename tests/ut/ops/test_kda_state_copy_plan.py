# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops import kda_state_copy_plan as production


@pytest.mark.parametrize("retain_constexpr", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("to_cache", [False, True])
def test_prepared_launch_matches_compiled_signature(monkeypatch, retain_constexpr, index_dtype, to_cache):
    launches = []

    class CompiledKernel:
        def __init__(self, constants):
            self.constants = constants
            signature = dict.fromkeys(range(9), "i64")
            if retain_constexpr:
                signature.update(dict.fromkeys(range(9, 12), "constexpr"))
            self.src = SimpleNamespace(signature=signature)

        def __getitem__(self, grid):
            def launch(*args):
                assert len(args) == (12 if retain_constexpr else 9)
                if retain_constexpr:
                    assert args[9:] == tuple(self.constants[k] for k in ("TO_CACHE", "HAS_FLAGS", "BLOCK_SIZE"))
                launches.append((grid, args))

            return launch

    class Compiler:
        pre_run_hooks = ()

        def __getitem__(self, grid):
            return lambda *args, **constants: CompiledKernel(constants)

    monkeypatch.setattr(production, "_kda_state_copy_kernel", Compiler())
    monkeypatch.setattr(production, "_validate_cache_layout", lambda state: (6, 4, 6))
    monkeypatch.setattr(production, "_configuration", lambda: ())
    monkeypatch.setattr(
        production.torch,
        "npu",
        SimpleNamespace(device=lambda device: nullcontext(), synchronize=lambda: None, current_device=lambda: None),
        raising=False,
    )
    state = torch.zeros((4, 1, 2, 3))
    packed = torch.empty((2, 1, 2, 3))
    indices = torch.zeros(2, dtype=index_dtype)
    flags = torch.ones(2, dtype=torch.bool)
    plan = production.KDAStateCopyPlan.prepare(state, 2)
    plan.seal()
    plan._launch(state, packed, indices, flags, to_cache=to_cache)

    assert len(launches) == 1
    grid, args = launches[0]
    assert grid == (2, 1, 1)
    assert args[0] is state and args[1] is packed and args[2] is indices
    assert args[3] is (indices if to_cache else flags)
    assert args[4:9] == (4, 6, 6, 1, 0 if to_cache else 1)
