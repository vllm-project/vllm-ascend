# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.models import qwen3_dflash
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
    AscendDFlash2Speculator,
)


def _config(architecture: str | None):
    draft_model_config = None if architecture is None else SimpleNamespace(architectures=[architecture])
    speculative_config = SimpleNamespace(
        method="dflash",
        draft_model_config=draft_model_config,
        use_dflash=lambda: True,
        use_dspark=lambda: False,
    )
    return SimpleNamespace(speculative_config=speculative_config)


def test_v2_factory_routes_dflash2_without_changing_dflash1():
    with patch(
        "vllm_ascend.worker.v2.spec_decode.dflash2.speculator.AscendDFlash2Speculator",
        return_value="dflash2",
    ) as dflash2:
        assert init_speculator(_config("DFlash2DraftModel"), torch.device("cpu")) == "dflash2"
        dflash2.assert_called_once()

    with patch(
        "vllm_ascend.worker.v2.spec_decode.dflash.speculator.AscendDFlashSpeculator",
        return_value="dflash1",
    ) as dflash1:
        assert init_speculator(_config("DFlashDraftModel"), torch.device("cpu")) == "dflash1"
        dflash1.assert_called_once()


def test_v2_factory_handles_missing_draft_model_defensively():
    with patch(
        "vllm_ascend.worker.v2.spec_decode.dflash.speculator.AscendDFlashSpeculator",
        return_value="dflash1",
    ):
        assert init_speculator(_config(None), torch.device("cpu")) == "dflash1"


def test_dflash2_composes_upstream_selector_with_ascend_runtime():
    assert issubclass(AscendDFlash2Speculator, DFlash2Speculator)
    assert issubclass(AscendDFlash2Speculator, AscendDFlashSpeculator)
    mro = AscendDFlash2Speculator.__mro__
    assert mro.index(DFlash2Speculator) < mro.index(AscendDFlashSpeculator)


def test_sample_path_dispatches_the_ascend_kernel(monkeypatch):
    speculator = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)
    speculator.selector_top_k = 4
    speculator.num_speculative_steps = 3
    speculator.sample_pos = torch.zeros(6, dtype=torch.int32)
    speculator.sample_idx_mapping = torch.zeros(6, dtype=torch.int32)
    speculator.temperature = torch.zeros(2)
    speculator.seeds = torch.zeros(2, dtype=torch.int64)
    speculator.draft_tokens = torch.zeros(2, 3, dtype=torch.int64)
    speculator._selector_scores = torch.zeros(2, 3, 4)
    speculator.draft_logits = None
    speculator.use_fp64_gumbel = False

    kernel = MagicMock()
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dflash2.speculator._selector_walk_kernel_ascend",
        kernel,
    )
    candidates = torch.zeros(2, 3, 4, dtype=torch.int64)
    scores = torch.zeros(2, 3, 4, 4)

    speculator._sample_path(candidates, scores, num_reqs=2)

    kernel.__getitem__.assert_called_once_with((2,))
    call = kernel.__getitem__.return_value
    call.assert_called_once()
    assert call.call_args.kwargs == {
        "num_steps": 3,
        "top_k": 4,
        "BLOCK_K": 4,
        "SAMPLE_PROBABILISTIC": False,
        "USE_FP64": False,
        "num_warps": 1,
    }


def test_old_vllm_uses_draft_owned_rope_loader(monkeypatch):
    speculator = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)
    speculator.vllm_config = object()
    target = object()
    own_loader = MagicMock(return_value="draft")
    upstream_loader = MagicMock(side_effect=AssertionError("old loader must not run"))
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dflash2.speculator._load_dflash_model_with_draft_rope",
        own_loader,
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dflash2.speculator.upstream_load_dflash_model",
        upstream_loader,
    )
    monkeypatch.setattr(
        qwen3_dflash,
        "dflash_target_rope_is_neox_style",
        lambda _: False,
        raising=False,
    )

    assert speculator.load_draft_model(target, set()) == "draft"
    own_loader.assert_called_once_with(target, speculator.vllm_config)


def test_fixed_vllm_uses_upstream_loader(monkeypatch):
    speculator = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)
    speculator.vllm_config = object()
    target = object()
    upstream_loader = MagicMock(return_value="draft")
    own_loader = MagicMock(side_effect=AssertionError("compat loader must not run"))
    monkeypatch.delattr(
        qwen3_dflash,
        "dflash_target_rope_is_neox_style",
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dflash2.speculator.upstream_load_dflash_model",
        upstream_loader,
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dflash2.speculator._load_dflash_model_with_draft_rope",
        own_loader,
    )

    assert speculator.load_draft_model(target, set()) == "draft"
    upstream_loader.assert_called_once_with(target, speculator.vllm_config)
