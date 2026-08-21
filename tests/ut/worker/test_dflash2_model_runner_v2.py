# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.patch.platform import patch_use_v2_model_runner
from vllm_ascend.worker.v2.spec_decode import init_speculator


def _speculative_config(architecture: str):
    return SimpleNamespace(
        draft_model_config=SimpleNamespace(architectures=[architecture]),
        use_dspark=lambda: False,
        use_dflash=lambda: True,
    )


def test_dflash2_forces_v2_when_runner_env_is_unset(monkeypatch):
    monkeypatch.setattr(patch_use_v2_model_runner.envs, "VLLM_USE_V2_MODEL_RUNNER", None)
    config = SimpleNamespace(_is_dflash2_draft=lambda: True)

    assert patch_use_v2_model_runner._patched_use_v2_model_runner(config)


def test_explicit_runner_env_takes_precedence(monkeypatch):
    monkeypatch.setattr(patch_use_v2_model_runner.envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    config = SimpleNamespace(_is_dflash2_draft=lambda: True)

    assert not patch_use_v2_model_runner._patched_use_v2_model_runner(config)


def test_non_dflash2_keeps_v2_disabled_by_default(monkeypatch):
    monkeypatch.setattr(patch_use_v2_model_runner.envs, "VLLM_USE_V2_MODEL_RUNNER", None)
    config = SimpleNamespace(_is_dflash2_draft=lambda: False)

    assert not patch_use_v2_model_runner._patched_use_v2_model_runner(config)


def test_init_speculator_selects_ascend_dflash2(monkeypatch):
    dflash2 = pytest.importorskip("vllm_ascend.worker.v2.spec_decode.dflash2.speculator")
    sentinel = object()
    monkeypatch.setattr(dflash2, "AscendDFlash2Speculator", lambda *_args: sentinel)
    config = SimpleNamespace(speculative_config=_speculative_config("DFlash2DraftModel"))

    assert init_speculator(config, torch.device("cpu")) is sentinel


def test_dflash2_composes_upstream_and_ascend_speculators():
    dflash2 = pytest.importorskip("vllm_ascend.worker.v2.spec_decode.dflash2.speculator")

    assert issubclass(dflash2.AscendDFlash2Speculator, dflash2.DFlash2Speculator)
    assert issubclass(dflash2.AscendDFlash2Speculator, dflash2.AscendDFlashSpeculator)


def test_dflash2_rejects_unsupported_fp64_sampling():
    dflash2 = pytest.importorskip("vllm_ascend.worker.v2.spec_decode.dflash2.speculator")
    speculator = SimpleNamespace(use_fp64_gumbel=True)

    with pytest.raises(NotImplementedError, match="FP64 DFlash2"):
        dflash2.AscendDFlash2Speculator._sample_path(speculator, None, None, 1)


@pytest.mark.parametrize(
    ("draft_logits", "sample_probabilistic"),
    [(None, False), (object(), True)],
)
def test_dflash2_sample_path_matches_upstream_contract(
    monkeypatch,
    draft_logits,
    sample_probabilistic,
):
    dflash2 = pytest.importorskip("vllm_ascend.worker.v2.spec_decode.dflash2.speculator")
    launch = {}

    class FakeKernel:
        def __getitem__(self, grid):
            launch["grid"] = grid

            def run(*args, **kwargs):
                launch["args"] = args
                launch["kwargs"] = kwargs

            return run

    monkeypatch.setattr(dflash2, "_selector_walk_kernel_ascend", FakeKernel())
    draft_tokens = object()
    speculator = SimpleNamespace(
        use_fp64_gumbel=False,
        selector_top_k=3,
        sample_pos=object(),
        sample_idx_mapping=object(),
        temperature=object(),
        seeds=object(),
        draft_tokens=draft_tokens,
        _selector_scores=object(),
        num_speculative_steps=4,
        draft_logits=draft_logits,
    )
    candidate_ids = torch.empty(2, 4, 3, dtype=torch.int64)
    scores = torch.empty(2, 4, 3, 3)

    dflash2.AscendDFlash2Speculator._sample_path(
        speculator,
        candidate_ids,
        scores,
        num_reqs=2,
    )

    assert launch["grid"] == (2,)
    assert launch["args"][6] is draft_tokens
    assert launch["kwargs"]["SAMPLE_PROBABILISTIC"] is sample_probabilistic
