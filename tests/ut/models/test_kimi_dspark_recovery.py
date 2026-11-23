# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.kimi_k3.dspark_speculator import KimiDSparkSpeculator
from vllm_ascend.models.kimi_k3.recoverssm import KimiDSparkRecoverSSMState, KimiRecoverSSMModelState
from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


@pytest.mark.parametrize(
    ("arch", "expected"),
    [("K3DSparkModel", KimiDSparkSpeculator), ("Qwen3DSparkModel", AscendDSparkSpeculator)],
)
def test_draft_model_selects_proposer(monkeypatch, arch, expected):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            enforce_eager=False,
            use_dspark=lambda: True,
            draft_model_config=SimpleNamespace(architectures=[arch]),
        )
    )
    monkeypatch.setattr(AscendDSparkSpeculator, "__init__", lambda *args: None)
    assert type(init_speculator(config, torch.device("cpu"))) is expected


@pytest.mark.parametrize("failure", [None, "prepare", "proposal"])
def test_kimi_draft_joins_before_publication_or_on_error(monkeypatch, failure):
    recovery = object.__new__(KimiDSparkRecoverSSMState)
    recovery.device = torch.device("cpu")
    recovery._active = recovery._submitted = False
    recovery._done = object()
    monkeypatch.setattr(recovery, "_can_merge_step", lambda: True)
    calls = []
    stream = SimpleNamespace(wait_event=lambda event: calls.append("join"))
    monkeypatch.setattr(torch.npu, "current_stream", lambda device: stream)
    state = object.__new__(KimiRecoverSSMModelState)
    state.recoverssm = recovery
    spec = object.__new__(KimiDSparkSpeculator)

    def propose(*args):
        calls.append("draft")
        if failure == "proposal":
            raise RuntimeError(failure)
        return 42

    monkeypatch.setattr(AscendDSparkSpeculator, "propose", propose)

    def sample():
        with spec.sampling_scope(state):
            assert recovery._active
            recovery._submitted = True
            if failure == "prepare":
                raise RuntimeError(failure)
            assert spec.propose() == 42
            calls.append("publish")

    if failure is None:
        sample()
        assert calls == ["draft", "join", "publish"]
    else:
        with pytest.raises(RuntimeError, match=failure):
            sample()
        assert calls == (["join"] if failure == "prepare" else ["draft", "join"])
    assert spec._recovery is None
    assert not recovery._active and not recovery._submitted


def test_kimi_draft_with_ordinary_model_state(monkeypatch):
    spec = object.__new__(KimiDSparkSpeculator)
    monkeypatch.setattr(AscendDSparkSpeculator, "propose", lambda self: 42)
    with spec.sampling_scope(object()):
        assert spec.propose() == 42
        assert spec._recovery is None
