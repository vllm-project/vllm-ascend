# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise production worker ordering with CPU communication/stream doubles."""

import __future__

import ast
import importlib.util
from pathlib import Path
from types import NoneType, SimpleNamespace

import pytest


@pytest.fixture
def worker_class():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("ascend_envs_for_pp_test", root / "vllm_ascend/envs.py")
    envs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(envs)
    tree = ast.parse((root / "vllm_ascend/worker/worker.py").read_text())
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
    methods = [node for node in worker.body if getattr(node, "name", None) in {"execute_model", "sample_tokens"}]
    # Execute the production methods, following the CPU-only runner-test pattern;
    # importing the worker itself requires the full NPU/vLLM runtime.
    module = ast.Module(body=methods, type_ignores=[])
    namespace = {"envs_ascend": envs, "NoneType": NoneType}
    namespace["torch"] = SimpleNamespace(inference_mode=lambda: lambda function: function)
    exec(compile(module, "worker_ordering", "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


@pytest.mark.parametrize("mode", [None, "AI_CPU", "AIV"])
@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("last_stage", [False, True])
def test_forward_send_completes_before_feedback(worker_class, monkeypatch, mode, use_v2, last_stage):
    if mode is None:
        monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    else:
        monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", mode)
    events = []
    handles = [SimpleNamespace(wait=lambda index=index: events.append(f"send{index}")) for index in range(2)]
    runner = SimpleNamespace(
        is_last_pp_rank=last_stage,
        sample_tokens=lambda grammar: events.append("feedback") or grammar,
    )
    worker = SimpleNamespace(use_v2_model_runner=use_v2, model_runner=runner, _pp_send_work=handles)
    assert worker_class["sample_tokens"](worker, "grammar") == "grammar"
    needs_fence = use_v2 and mode == "AIV" and not last_stage
    assert events == (["send0", "send1", "feedback"] if needs_fence else ["feedback"])
    assert worker._pp_send_work == ([] if needs_fence else handles)


def test_send_failure_does_not_submit_feedback(worker_class, monkeypatch):
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    events = []

    def fail_send():
        raise RuntimeError("send failed")

    worker = SimpleNamespace(
        use_v2_model_runner=True,
        model_runner=SimpleNamespace(is_last_pp_rank=False, sample_tokens=lambda grammar: events.append("feedback")),
        _pp_send_work=[SimpleNamespace(wait=fail_send)],
    )
    with pytest.raises(RuntimeError, match="send failed"):
        worker_class["sample_tokens"](worker, None)
    assert not events
    assert worker._pp_send_work


def test_runner_without_pp_does_not_require_rank_metadata(worker_class, monkeypatch):
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    worker = SimpleNamespace(
        use_v2_model_runner=True,
        model_runner=SimpleNamespace(sample_tokens=lambda grammar: "encoded"),
        _pp_send_work=[],
    )
    assert worker_class["sample_tokens"](worker, None) == "encoded"


@pytest.mark.parametrize("pending", [[], [SimpleNamespace(wait=lambda: None)]])
def test_sampling_without_pending_sends(worker_class, monkeypatch, pending):
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    worker = SimpleNamespace(
        use_v2_model_runner=True,
        model_runner=SimpleNamespace(is_last_pp_rank=False, sample_tokens=lambda grammar: "sampled"),
        _pp_send_work=pending,
    )
    assert worker_class["sample_tokens"](worker, None) == "sampled"
    assert worker._pp_send_work == []


@pytest.mark.parametrize(
    "use_v2,async_scheduling,mode,first_stage,tokens,needs_fence",
    [
        (True, True, "AIV", False, 1, True),
        (True, True, "AIV", False, 0, False),
        (True, True, "AIV", True, 1, False),
        (True, False, "AIV", False, 1, False),
        (False, True, "AIV", False, 1, False),
        (True, True, "AI_CPU", False, 1, False),
        (True, True, None, False, 1, False),
    ],
)
def test_feedback_precedes_next_hidden_receive(
    worker_class, monkeypatch, use_v2, async_scheduling, mode, first_stage, tokens, needs_fence
):
    if mode is None:
        monkeypatch.delenv("HCCL_OP_EXPANSION_MODE", raising=False)
    else:
        monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", mode)
    events = []
    feedback_stream = object()

    def wait_feedback(stream):
        assert stream is feedback_stream
        events.append("feedback_dependency")

    def receive(**kwargs):
        events.append("hidden_receive")
        return {}, [], None

    worker_class.update(
        get_ascend_config=lambda: SimpleNamespace(msmonitor_use_daemon=False),
        get_pp_group=lambda: SimpleNamespace(is_first_rank=first_stage, irecv_tensor_dict=receive),
        get_tp_group=lambda: object(),
        enable_sp=lambda config: False,
        AsyncIntermediateTensors=lambda *args, **kwargs: object(),
        ModelRunnerOutput=type("ModelRunnerOutput", (), {}),
        AsyncModelRunnerOutput=type("AsyncModelRunnerOutput", (), {}),
    )
    worker_class["torch"].npu = SimpleNamespace(
        current_stream=lambda device: SimpleNamespace(wait_stream=wait_feedback)
    )
    runner = SimpleNamespace(
        pp_handler=SimpleNamespace(broadcast_stream=feedback_stream),
        is_pooling_model=False,
        execute_model=lambda *args: events.append("forward") or None,
    )
    worker = SimpleNamespace(
        use_v2_model_runner=use_v2,
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=async_scheduling)),
        model_runner=runner,
        _pp_send_work=[],
        profiler=None,
        device="npu:4",
        log_memory_stats=lambda: None,
    )
    assert worker_class["execute_model"](worker, SimpleNamespace(total_num_scheduled_tokens=tokens)) is None
    expected = ["feedback_dependency"] if needs_fence else []
    if tokens and not first_stage:
        expected.append("hidden_receive")
    assert events == expected + ["forward"]
