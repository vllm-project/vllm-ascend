# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend import ascend_config
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator


@pytest.mark.parametrize("speculator_cls", [AscendDFlashSpeculator, AscendDFlash2Speculator])
@pytest.mark.parametrize("outcome", ["success", "no_hook", "load_error", "hook_error"])
def test_draft_post_process_uses_restored_target_config(monkeypatch, speculator_cls, outcome):
    config = SimpleNamespace(
        quant_config=object(),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )
    spec = speculator_cls.__new__(speculator_cls)
    spec.vllm_config = config
    target = object()
    names = {"target.attention"}
    draft = SimpleNamespace()
    target_ascend_config = SimpleNamespace(ascend_compilation_config=object(), eplb_config=object())
    draft_ascend_config = SimpleNamespace(ascend_compilation_config=object(), eplb_config=object())
    monkeypatch.setattr(ascend_config, "_ASCEND_CONFIG", target_ascend_config)
    monkeypatch.setattr(ascend_config, "_INIT_VLLM_CONFIG", config)
    monkeypatch.setattr(ascend_config, "_MEGA_MOE_SUPPORTED", True)

    def assert_target_config_restored():
        assert ascend_config.get_ascend_config() is target_ascend_config
        assert ascend_config._INIT_VLLM_CONFIG is config
        assert ascend_config.is_mega_moe_supported()

    def post_process(received):
        assert received is config
        assert_target_config_restored()
        if outcome == "hook_error":
            raise ValueError("post process failed")

    hook = Mock(side_effect=post_process)
    if outcome != "no_hook":
        draft.post_process = hook

    def load(self, received_target, received_names):
        assert self is spec
        assert received_target is target
        assert received_names is names
        # Draft platform validation can replace the singleton and disable MegaMoE.
        monkeypatch.setattr(ascend_config, "_ASCEND_CONFIG", draft_ascend_config)
        monkeypatch.setattr(ascend_config, "_INIT_VLLM_CONFIG", object())
        monkeypatch.setattr(ascend_config, "_MEGA_MOE_SUPPORTED", False)
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

    assert_target_config_restored()
    if outcome in ("success", "hook_error"):
        hook.assert_called_once_with(config)
    else:
        hook.assert_not_called()
