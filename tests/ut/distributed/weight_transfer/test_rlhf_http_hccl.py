# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest


@pytest.fixture
def example(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "examples/rl/rlhf_http_hccl.py"
    spec = importlib.util.spec_from_file_location("rlhf_http_hccl_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = MagicMock()
    model.parameters.return_value = [SimpleNamespace(numel=lambda: 1, element_size=lambda: 2)]
    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", MagicMock(return_value=model))
    monkeypatch.setattr(module.torch.accelerator, "set_device_index", MagicMock())
    monkeypatch.setattr(module, "get_world_size", MagicMock(return_value=1))
    monkeypatch.setattr(module, "get_ip", MagicMock(return_value="127.0.0.1"))
    monkeypatch.setattr(module, "get_open_port", MagicMock(return_value=1234))
    for name in ("OpenAI", "ModuleSource", "HTTPVLLMWeightSyncClient", "register_engine"):
        monkeypatch.setattr(module, name, MagicMock())
    monkeypatch.setattr(module, "generate_completions", MagicMock(return_value=["generated"] * 4))
    return module


@pytest.mark.parametrize("send_fails", [False, True])
def test_main_resumes_only_after_successful_weight_transfer(example, monkeypatch, send_fails):
    events = MagicMock()
    engine = MagicMock()
    events.attach_mock(engine.send_weights, "send")
    events.attach_mock(engine.shutdown, "shutdown")
    monkeypatch.setattr(example, "pause_generation", events.pause)
    monkeypatch.setattr(example, "resume_generation", events.resume)
    monkeypatch.setattr(example.WeightTransferTrainerFactory, "trainer_init", MagicMock(return_value=engine))
    if send_fails:
        error = RuntimeError("send failed")
        engine.send_weights.side_effect = error
        events.resume.side_effect = RuntimeError("resume must not mask a failed transfer")
        with pytest.raises(RuntimeError) as caught:
            example.main()
        assert caught.value is error
    else:
        example.main()

    expected = [call.pause(example.BASE_URL), call.send()]
    if not send_fails:
        expected.append(call.resume(example.BASE_URL))
    expected.append(call.shutdown())
    assert events.mock_calls == expected
    assert example.generate_completions.call_count == (1 if send_fails else 2)
