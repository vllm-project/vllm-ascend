# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

from vllm_ascend.worker.v2 import model_runner as model_runner_module
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _fake_runner(pp_size: int, speculator) -> SimpleNamespace:
    return SimpleNamespace(
        speculator=speculator,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=pp_size)
        ),
    )


def test_clears_draft_mm_flag_on_release_lines_with_pp(monkeypatch):
    for release in ("0.28.0", "0.29.0"):
        monkeypatch.setattr(model_runner_module, "vllm_version_is", lambda v, r=release: v == r)
        speculator = SimpleNamespace(supports_mm_inputs=True)

        NPUModelRunner._clear_draft_mm_inputs_flag_for_pp_releases(_fake_runner(2, speculator))

        assert speculator.supports_mm_inputs is False, release


def test_keeps_draft_mm_flag_on_newer_vllm(monkeypatch):
    monkeypatch.setattr(model_runner_module, "vllm_version_is", lambda v: v == "0.30.0")
    speculator = SimpleNamespace(supports_mm_inputs=True)

    NPUModelRunner._clear_draft_mm_inputs_flag_for_pp_releases(_fake_runner(2, speculator))

    assert speculator.supports_mm_inputs is True


def test_keeps_draft_mm_flag_without_pipeline_parallel(monkeypatch):
    monkeypatch.setattr(model_runner_module, "vllm_version_is", lambda v: v == "0.28.0")
    speculator = SimpleNamespace(supports_mm_inputs=True)

    NPUModelRunner._clear_draft_mm_inputs_flag_for_pp_releases(_fake_runner(1, speculator))

    assert speculator.supports_mm_inputs is True


def test_tolerates_missing_speculator(monkeypatch):
    monkeypatch.setattr(model_runner_module, "vllm_version_is", lambda v: v == "0.28.0")

    NPUModelRunner._clear_draft_mm_inputs_flag_for_pp_releases(_fake_runner(2, None))
