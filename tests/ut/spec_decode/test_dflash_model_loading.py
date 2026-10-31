# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator


@pytest.mark.parametrize("speculator_cls", [AscendDFlashSpeculator, AscendDFlash2Speculator])
@pytest.mark.parametrize("outcome", ["success", "no_hook", "load_error", "hook_error"])
def test_draft_post_process_uses_restored_target_config(monkeypatch, speculator_cls, outcome):
    additional_config = {"scheduler_config": {"profiling_chunk_config": {"enabled": True}}}
    config = SimpleNamespace(
        additional_config=additional_config,
        quant_config=object(),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )
    spec = speculator_cls.__new__(speculator_cls)
    spec.vllm_config = config
    target = object()
    names = {"target.attention"}
    draft = SimpleNamespace()

    def post_process(received):
        assert received is config
        assert received.additional_config is additional_config
        assert additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] is True
        if outcome == "hook_error":
            raise ValueError("post process failed")

    hook = Mock(side_effect=post_process)
    if outcome != "no_hook":
        draft.post_process = hook

    def load(self, received_target, received_names):
        assert self is spec
        assert received_target is target
        assert received_names is names
        assert config.additional_config is not additional_config
        assert config.additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] is False
        if outcome == "load_error":
            raise ValueError("load failed")
        return draft

    monkeypatch.setattr(DFlashSpeculator, "load_draft_model", load)
    if outcome in ("load_error", "hook_error"):
        message = "load failed" if outcome == "load_error" else "post process failed"
        with pytest.raises(ValueError, match=message):
            spec.load_draft_model(target, names)
    else:
        assert spec.load_draft_model(target, names) is draft

    assert config.additional_config is additional_config
    assert additional_config["scheduler_config"]["profiling_chunk_config"]["enabled"] is True
    if outcome in ("success", "hook_error"):
        hook.assert_called_once_with(config)
    else:
        hook.assert_not_called()
