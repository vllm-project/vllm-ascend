# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode import init_speculator


def _speculative_config(architecture: str):
    return SimpleNamespace(
        draft_model_config=SimpleNamespace(architectures=[architecture]),
        use_dspark=lambda: False,
        use_dflash=lambda: True,
    )


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


@pytest.mark.parametrize(
    ("enforce_eager", "expected_mode"),
    [(True, "NONE"), (False, "FULL")],
)
def test_dflash2_honors_draft_enforce_eager(
    monkeypatch,
    enforce_eager,
    expected_mode,
):
    dflash2 = pytest.importorskip("vllm_ascend.worker.v2.spec_decode.dflash2.speculator")
    modes = []

    def fake_init_cudagraph_manager(self, cudagraph_mode):
        modes.append(cudagraph_mode.name)

    monkeypatch.setattr(
        dflash2.AscendDFlashSpeculator,
        "init_cudagraph_manager",
        fake_init_cudagraph_manager,
    )
    speculator = object.__new__(dflash2.AscendDFlash2Speculator)
    speculator.speculative_config = SimpleNamespace(enforce_eager=enforce_eager)

    dflash2.AscendDFlash2Speculator.init_cudagraph_manager(
        speculator,
        dflash2.CUDAGraphMode.FULL,
    )

    assert modes == [expected_mode]


def test_dflash2_disables_compile_on_draft_instance_only(monkeypatch):
    dflash2 = pytest.importorskip("vllm_ascend.worker.v2.spec_decode.dflash2.speculator")
    compile_module = SimpleNamespace(do_not_compile=False)
    shared_target_module = SimpleNamespace(do_not_compile=False)
    ordinary_module = SimpleNamespace()
    draft_model = SimpleNamespace(modules=lambda: [compile_module, shared_target_module, ordinary_module])
    target_model = SimpleNamespace(modules=lambda: [shared_target_module])

    monkeypatch.setattr(
        dflash2.AscendDFlashSpeculator,
        "load_draft_model",
        lambda *_args: draft_model,
        raising=False,
    )
    speculator = object.__new__(dflash2.AscendDFlash2Speculator)

    result = dflash2.AscendDFlash2Speculator.load_draft_model(speculator, target_model, set())

    assert result is draft_model
    assert compile_module.do_not_compile
    assert not shared_target_module.do_not_compile
    assert not hasattr(ordinary_module, "do_not_compile")


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
