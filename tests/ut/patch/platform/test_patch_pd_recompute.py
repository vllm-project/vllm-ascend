# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the native preemption boundary, with vLLM imports mocked."""

import asyncio
import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from openai.types.responses import ResponseCreatedEvent, ResponseErrorEvent
from pydantic import TypeAdapter


@pytest.fixture
def adapter(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    constants = runpy.run_path(str(root / "vllm_ascend/core/pd_recompute.py"))

    class GenerationError(Exception):
        pass

    class Engine:
        async def generate(self, prompt, sampling_params):
            yield prompt

    class Serving:
        def _convert_generation_error_to_streaming_response(self, error):
            return "original error"

    serving = SimpleNamespace(StreamingResponsesResponse=ResponseCreatedEvent, OpenAIServingResponses=Serving)
    modules = {
        "vllm.entrypoints.openai.responses": {"serving": serving},
        "vllm.exceptions": {"GenerationError": GenerationError},
        "vllm.v1.engine.async_llm": {"AsyncLLM": Engine},
        "vllm_ascend.core.pd_recompute": constants,
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "pd_adapter_under_test", root / "vllm_ascend/patch/platform/patch_pd_recompute.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, Engine, Serving, serving


@pytest.mark.parametrize("before_output", [True, False])
def test_preemption_raises_before_partial_completion_is_serialized(adapter, before_output):
    module, _, _, _ = adapter
    closed = []
    normal = SimpleNamespace(finished=False, outputs=[])
    interrupted = SimpleNamespace(
        finished=True,
        outputs=[SimpleNamespace(finish_reason="error", stop_reason="recomputed", text='{"unfinished":')],
    )

    async def generate(self, prompt, params):
        try:
            if not before_output:
                yield normal
            yield interrupted
            pytest.fail("The interrupted attempt must not continue")
        finally:
            closed.append(True)

    async def exercise():
        outputs = []
        params = SimpleNamespace(extra_args={module.PD_FAIL_ON_RECOMPUTE: True})
        with pytest.raises(module.PDRecomputeError, match="preempted"):
            async for item in module._wrap_generate(generate)(None, "input", params):
                outputs.append(item)
        assert outputs == ([] if before_output else [normal])

    asyncio.run(exercise())
    assert closed == [True]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("reason", ["stop", "error"])
def test_opt_in_does_not_mistake_a_user_stop_string_for_preemption(adapter, enabled, reason):
    module, engine, _, _ = adapter
    output = SimpleNamespace(finished=True, outputs=[SimpleNamespace(finish_reason=reason, stop_reason="recomputed")])
    params = SimpleNamespace(extra_args={module.PD_FAIL_ON_RECOMPUTE: enabled})

    async def exercise():
        if enabled and reason == "error":
            with pytest.raises(module.PDRecomputeError):
                await anext(engine().generate(output, params))
        else:
            assert [item async for item in engine().generate(output, params)] == [output]

    asyncio.run(exercise())


def test_responses_preemption_is_a_valid_native_error_event(adapter):
    module, _, serving, namespace = adapter
    converter = serving()._convert_generation_error_to_streaming_response
    event = TypeAdapter(namespace.StreamingResponsesResponse).validate_json(converter(module.PDRecomputeError("ended")))
    assert isinstance(event, ResponseErrorEvent)
    assert event.type == "error"
    assert event.code == "pd_preempted"
    assert event.message == "ended"
    assert converter(ValueError("other")) == "original error"
