#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""Unit tests for the V2 DFlash2 speculator (worker/v2/spec_decode/dflash2)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
    AscendDFlash2Speculator,
    _selector_walk_kernel_ascend,
)


def test_patch_swaps_upstream_selector_walk_kernel():
    import vllm.v1.worker.gpu.spec_decode.dflash2.speculator as upstream

    import vllm_ascend.patch.worker.patch_v2.patch_dflash_speculator  # noqa: F401

    assert upstream._selector_walk_kernel is _selector_walk_kernel_ascend


def _spec_config(arch: str) -> SimpleNamespace:
    return SimpleNamespace(
        method="dflash",
        use_dspark=lambda: False,
        use_dflash=lambda: True,
        draft_model_config=SimpleNamespace(architectures=[arch]),
    )


def test_init_speculator_routes_dflash2_draft_model():
    cfg = SimpleNamespace(speculative_config=_spec_config("DFlash2DraftModel"))
    with patch("vllm_ascend.worker.v2.spec_decode.dflash2.speculator.AscendDFlash2Speculator") as d2:
        assert init_speculator(cfg, torch.device("cpu")) is d2.return_value
        d2.assert_called_once_with(cfg, torch.device("cpu"))

    cfg = SimpleNamespace(speculative_config=_spec_config("DFlashDraftModel"))
    with (
        patch("vllm_ascend.worker.v2.spec_decode.dflash2.speculator.AscendDFlash2Speculator") as d2,
        patch("vllm_ascend.worker.v2.spec_decode.dflash.speculator.AscendDFlashSpeculator") as d1,
    ):
        assert init_speculator(cfg, torch.device("cpu")) is d1.return_value
        d2.assert_not_called()


def test_init_cudagraph_manager_forces_none_when_enforce_eager(monkeypatch):
    calls: list[CUDAGraphMode] = []
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dflash.speculator.AscendDFlashSpeculator.init_cudagraph_manager",
        lambda self, mode: calls.append(mode),
    )
    speculator = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)

    speculator.speculative_config = SimpleNamespace(enforce_eager=True)
    speculator.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert calls == [CUDAGraphMode.NONE]

    speculator.speculative_config = SimpleNamespace(enforce_eager=False)
    speculator.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)
    assert calls == [CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY]


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an NPU device")
@pytest.mark.parametrize("sample_probabilistic", [False, True])
def test_selector_walk_kernel_ascend_greedy_walk(sample_probabilistic: bool):
    """Greedy walk: argmax per step, lowest index wins ties, chained via the
    previous step's winner. Row 1 is padding (req_state < 0) and must not
    write. With temperature 0 the probabilistic variant must stay greedy."""
    device = torch.device("npu")
    num_steps, top_k = 2, 3
    candidates = torch.tensor(
        [[100, 200, 300], [400, 500, 600], [7, 8, 9], [70, 80, 90]],
        dtype=torch.int64,
        device=device,
    )
    scores = torch.full((4, top_k, top_k), 7.0, dtype=torch.float32, device=device)
    scores[0, 0] = torch.tensor([0.5, 0.9, 0.9])  # tie between 1 and 2
    scores[1, 1] = torch.tensor([-1.0, -2.0, 3.0])  # continues from candidate 1
    sample_pos = torch.arange(1, 5, dtype=torch.int64, device=device)
    req_state = torch.tensor([0, 0, -1, -1], dtype=torch.int32, device=device)
    temperature = torch.tensor([0.0], dtype=torch.float32, device=device)
    seeds = torch.tensor([0], dtype=torch.int64, device=device)
    tokens = torch.full((4,), -123, dtype=torch.int64, device=device)
    realized = torch.full((4, top_k), -777.0, dtype=torch.float32, device=device)

    _selector_walk_kernel_ascend[(2,)](
        scores,
        candidates,
        sample_pos,
        req_state,
        temperature,
        seeds,
        tokens,
        realized,
        num_steps=num_steps,
        top_k=top_k,
        BLOCK_K=4,
        SAMPLE_PROBABILISTIC=sample_probabilistic,
        USE_FP64=False,
    )

    torch.testing.assert_close(tokens.cpu(), torch.tensor([200, 600, -123, -123]))
    torch.testing.assert_close(realized[0].cpu(), torch.tensor([0.5, 0.9, 0.9]))
    torch.testing.assert_close(realized[1].cpu(), torch.tensor([-1.0, -2.0, 3.0]))
    torch.testing.assert_close(realized[2:].cpu(), torch.full((2, top_k), -777.0))
