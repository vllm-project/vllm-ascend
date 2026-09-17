# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend.models import dspark


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("previous", [None, "previous"])
@pytest.mark.parametrize("fail", [False, True])
def test_rotation_context_restores_config(monkeypatch, existing, previous, fail):
    hf_config = SimpleNamespace()
    if existing:
        hf_config._ascend_target_rotation_path = previous
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=hf_config))
    )
    monkeypatch.setattr(dspark, "get_rotation_path", lambda _: Path("target"))
    with (
        pytest.raises(RuntimeError, match="load failed") if fail else nullcontext(),
        dspark.draft_model_load_context(config),
    ):
        assert dspark.get_target_rotation_path(config) == Path("target")
        if fail:
            raise RuntimeError("load failed")
    assert hasattr(hf_config, "_ascend_target_rotation_path") == existing
    if existing:
        assert hf_config._ascend_target_rotation_path == previous


def test_rotation_resolver_falls_back_without_saved_path(monkeypatch):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=SimpleNamespace()))
    )
    monkeypatch.setattr(dspark, "get_rotation_path", lambda _: Path("target"))
    assert dspark.get_target_rotation_path(config) == Path("target")
