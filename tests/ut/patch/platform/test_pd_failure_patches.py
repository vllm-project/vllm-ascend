import ast
import asyncio
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from vllm.exceptions import VLLMValidationError
from vllm.v1.engine import FinishReason


def _load_patch_definitions(filename):
    """Load production definitions without installing process-wide wrappers."""
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform" / filename
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign))
        and not (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "_WRAPPED_CLASSES" for target in node.targets)
        )
    ]
    module = ModuleType(filename)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


@pytest.fixture
def prefill_patch():
    return _load_patch_definitions("patch_prefill_length_validation.py")


@pytest.fixture
def failure_patch():
    return _load_patch_definitions("patch_kv_load_failure_reason.py")


def _engine_request(budget, *, role="kv_producer", prompt_len=120):
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_role=role),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="deepseek_v41"), model="/models/flash", max_model_len=128
            ),
        )
    )
    request = SimpleNamespace(
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SimpleNamespace(extra_args={"kv_transfer_params": {"original_max_tokens": budget}}),
    )
    return engine, request


@pytest.mark.parametrize("budget", [9, 0, -1, True, "8"])
def test_prefill_rejects_invalid_or_oversized_original_budget(prefill_patch, budget):
    engine, request = _engine_request(budget)
    with pytest.raises(VLLMValidationError):
        prefill_patch._validate_prefill_request(engine, request)


@pytest.mark.parametrize(("budget", "role"), [(8, "kv_producer"), (None, "kv_producer"), (999, "kv_consumer")])
def test_prefill_preserves_boundary_default_and_decode_requests(prefill_patch, budget, role):
    engine, request = _engine_request(budget, role=role)
    prefill_patch._validate_prefill_request(engine, request)


def test_prefill_wrapper_preserves_v032_add_request_arguments(prefill_patch):
    calls = []

    class Engine:
        async def _add_request(self, request, prompt, parent_req, index, queue):
            calls.append((request, prompt, parent_req, index, queue))
            return queue

    prefill_patch.AsyncLLM = Engine
    prefill_patch._install_patch()
    installed = Engine._add_request
    prefill_patch._install_patch()
    assert Engine._add_request is installed
    config, request = _engine_request(8)
    engine = Engine()
    engine.vllm_config = config.vllm_config
    queue = object()
    result = asyncio.run(engine._add_request(request, "prompt", None, 0, queue))
    assert result is queue
    assert calls == [(request, "prompt", None, 0, queue)]


def test_failure_attribution_does_not_relabel_unrelated_requests(failure_patch):
    class Connector:
        def get_kv_load_failure_reason(self, request_id, invalid_blocks):
            if request_id == "failed" and 7 in invalid_blocks:
                return "kv_load_failed; please retry"
            return None

    def output(request_id):
        return SimpleNamespace(
            request_id=request_id, finish_reason=FinishReason.ERROR, new_token_ids=[], stop_reason=None
        )

    failed, unrelated = output("failed"), output("other")
    scheduler = SimpleNamespace(connector=SimpleNamespace(_connectors=[Connector()]))
    runner = SimpleNamespace(kv_connector_output=SimpleNamespace(invalid_block_ids={7}))
    failure_patch._inject_kv_load_failure_reason(scheduler, {0: SimpleNamespace(outputs=[failed, unrelated])}, runner)
    assert failed.finish_reason == FinishReason.STOP
    assert failed.stop_reason == "kv_load_failed; please retry"
    assert unrelated.finish_reason == FinishReason.ERROR
    assert unrelated.stop_reason is None


@pytest.mark.parametrize("already_forwarded", [False, True])
def test_streaming_rescue_emits_each_failure_once_before_done(failure_patch, already_forwarded):
    reason = "kv_handle_expired; please retry"

    async def source():
        if already_forwarded:
            yield f'data: {{"stop_reason": "{reason}"}}\n\n'
        yield "data: [DONE]\n\n"

    async def collect():
        return [line async for line in failure_patch._relay_with_error_rescue(source(), {0: reason}, "req-1")]

    lines = asyncio.run(collect())
    assert lines[-1] == "data: [DONE]\n\n"
    assert sum('"error"' in line for line in lines) == (0 if already_forwarded else 1)
    if not already_forwarded:
        assert '"code": "kv_handle_expired"' in lines[0]


def test_nonstreaming_failure_uses_retryable_503(failure_patch):
    serving = SimpleNamespace(create_error_response=lambda message, **kwargs: (message, kwargs))
    message, kwargs = failure_patch._kv_load_failure_error_response(serving, "kv_load_failed; please retry", "req-1")
    assert "req-1" in message
    assert kwargs == {"err_type": "ServiceUnavailableError", "status_code": 503}
