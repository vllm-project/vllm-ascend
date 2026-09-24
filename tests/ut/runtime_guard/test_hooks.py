# mypy: ignore-errors
"""Decorator-level tests for runtime_guard step / sample-phase / idle hooks."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.observability.runtime_guard.hooks import (
    runtime_guard_idle_step,
    runtime_guard_sample_tokens,
    runtime_guard_step,
)
from vllm_ascend.observability.runtime_guard.runner_bridge import (
    get_postprocess_sampled,
    note_postprocess_sampled,
)

_WORKER_ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend" / "worker"
_PENDING_SO_ATTR = "_pending_scheduler_output"


def _decorator_names(fn_node: ast.FunctionDef) -> list[str]:
    names = []
    for dec in fn_node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        names.append(target.attr if isinstance(target, ast.Attribute) else target.id)
    return names


def test_guard_step_sits_below_inference_mode():
    """runtime_guard_step must be INNER: the wave sync runs in inference mode."""
    for rel in ("model_runner_v1.py", "v2/model_runner.py"):
        tree = ast.parse((_WORKER_ROOT / rel).read_text(encoding="utf-8"))
        fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute_model"]
        assert len(fns) == 1, rel
        assert _decorator_names(fns[0]) == ["inference_mode", "runtime_guard_step"], rel


class _StepRunner:
    """Minimal stand-in exercising runtime_guard_step on a plain class."""

    def __init__(self, guard):
        self.runtime_guard = guard
        self.execute_model_state = None
        self.seen_inference_mode = None
        self.body_ran = False

    @torch.inference_mode()
    @runtime_guard_step
    def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False):
        self.body_ran = True
        self.seen_inference_mode = torch.is_inference_mode_enabled()
        return "output"


class _RaisingStepRunner:
    def __init__(self, guard):
        self.runtime_guard = guard
        self.execute_model_state = None

    @torch.inference_mode()
    @runtime_guard_step
    def execute_model(self, scheduler_output):
        raise RuntimeError("boom")


def _scheduler_output(total: int = 4) -> SimpleNamespace:
    return SimpleNamespace(total_num_scheduled_tokens=total)


def test_step_runs_sync_inside_inference_mode():
    guard = MagicMock()
    runner = _StepRunner(guard)
    so = _scheduler_output()

    assert runner.execute_model(so) == "output"
    assert runner.body_ran
    # Inner placement: the wave sync executes inside the inference-mode context.
    assert runner.seen_inference_mode is True
    assert getattr(runner, _PENDING_SO_ATTR) is so
    guard.sync_for_step.assert_called_once_with(scheduler_output=so, allow_arm=True)
    # No-sample step (execute_model_state stays None) flushes end-of-wave.
    guard.end_of_wave_sync.assert_called_once_with(allow_arm=False)


def test_step_idle_scheduler_never_arms():
    guard = MagicMock()
    runner = _StepRunner(guard)
    so = _scheduler_output(total=0)

    runner.execute_model(so)
    guard.sync_for_step.assert_called_once_with(scheduler_output=so, allow_arm=False)


def test_step_skips_flush_when_sample_phase_will_run():
    guard = MagicMock()
    runner = _StepRunner(guard)
    runner.execute_model_state = object()  # sample_tokens will pop it

    runner.execute_model(_scheduler_output())
    guard.end_of_wave_sync.assert_not_called()


def test_step_flushes_dummy_wave_even_with_state():
    guard = MagicMock()
    runner = _StepRunner(guard)
    runner.execute_model_state = object()  # dummy waves never reach sample_tokens

    runner.execute_model(_scheduler_output(), dummy_run=True)
    guard.end_of_wave_sync.assert_called_once_with(allow_arm=False)


def test_step_reads_positional_dummy_run():
    guard = MagicMock()
    runner = _StepRunner(guard)
    runner.execute_model_state = object()

    # MRV2 signature: (scheduler_output, intermediate_tensors, dummy_run, ...)
    runner.execute_model(_scheduler_output(), None, True)
    guard.end_of_wave_sync.assert_called_once_with(allow_arm=False)


def test_step_flushes_on_body_exception():
    guard = MagicMock()
    runner = _RaisingStepRunner(guard)

    with pytest.raises(RuntimeError):
        runner.execute_model(_scheduler_output())
    guard.end_of_wave_sync.assert_called_once_with(allow_arm=False)


def test_step_guardless_keeps_bare_path():
    runner = _StepRunner(None)
    runner.execute_model_state = object()
    so = _scheduler_output()

    assert runner.execute_model(so) == "output"
    assert runner.body_ran
    assert getattr(runner, _PENDING_SO_ATTR) is so


class _IdleWorker:
    """Minimal stand-in exercising runtime_guard_idle_step on the worker."""

    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.body_ran = False

    @runtime_guard_idle_step
    def execute_dummy_batch(self):
        self.body_ran = True
        return "done"


def test_idle_step_syncs_without_arming():
    guard = MagicMock()
    worker = _IdleWorker(SimpleNamespace(runtime_guard=guard))

    assert worker.execute_dummy_batch() == "done"
    assert worker.body_ran
    # Dummy waves never burn manual_dump and carry no scheduler_output.
    guard.sync_for_step.assert_called_once_with(allow_arm=False)


def test_idle_step_soft_fails_and_still_runs_body():
    guard = MagicMock()
    guard.sync_for_step.side_effect = RuntimeError("boom")
    worker = _IdleWorker(SimpleNamespace(runtime_guard=guard))

    assert worker.execute_dummy_batch() == "done"
    assert worker.body_ran


def test_idle_step_guardless_keeps_bare_path():
    worker = _IdleWorker(SimpleNamespace())  # no runtime_guard attr

    assert worker.execute_dummy_batch() == "done"
    assert worker.body_ran


def test_worker_dummy_batch_uses_idle_decorator():
    """NPUWorker must route the idle wave sync through the decorator."""
    src = (_WORKER_ROOT / "worker.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "execute_dummy_batch"]
    assert len(fns) == 1
    assert _decorator_names(fns[0]) == ["runtime_guard_idle_step"]
    # The lockstep sync itself lives in the hook, not in the worker body.
    assert "rg.sync_for_step" not in src


class _SampleRunner:
    """Minimal stand-in exercising runtime_guard_sample_tokens."""

    def __init__(self, guard):
        self.runtime_guard = guard
        self.order: list[str] = []
        self.speculative_config = None
        self.use_async_scheduling = False
        self.execute_model_state = SimpleNamespace(
            input_batch=SimpleNamespace(req_ids=["r1"]),
            finished_req_ids=["r1"],
        )
        self.run_phase_kwargs = None

    @runtime_guard_sample_tokens
    def sample_tokens(self, grammar_output):
        # Mirrors the runner: original functional body — the decorator only
        # adds guard orchestration around it.
        self.order.append("body")
        return SimpleNamespace(sampled_token_ids=[[7]])


def test_sample_tokens_orchestrates_guard_around_body():
    guard = MagicMock()
    guard.runtime_config = None  # need_pre_sample_hook -> False
    guard.needs_sample_phase_hooks.return_value = False
    runner = _SampleRunner(guard)

    def _run(sample_fn, **kwargs):
        runner.run_phase_kwargs = kwargs
        return sample_fn(), None

    guard.run_sample_phase.side_effect = _run

    out = runner.sample_tokens("grammar")
    assert out.sampled_token_ids == [[7]]
    # Body ran exactly once, inside the guard's sample_fn; result assembled
    # in hooks, not on the runner.
    assert runner.order == ["body"]
    assert runner.run_phase_kwargs == {
        "speculative_config": None,
        "need_accepted_tokens": False,
        "use_async": False,
        "accepted_token_nums_fn": None,
    }


def test_sample_tokens_passes_accepted_token_nums_fn_for_spec():
    guard = MagicMock()
    guard.runtime_config = None
    guard.needs_sample_phase_hooks.return_value = False
    runner = _SampleRunner(guard)
    runner.speculative_config = object()

    def _run(sample_fn, **kwargs):
        result = sample_fn()
        # postprocess_sampled would have noted during the sample body.
        note_postprocess_sampled(runner, [[1]], [3])
        nums_fn = kwargs["accepted_token_nums_fn"]
        assert nums_fn is not None
        assert nums_fn(result) == [3]
        return result, None

    guard.run_sample_phase.side_effect = _run
    runner.sample_tokens(None)
    assert get_postprocess_sampled(runner) == ([[1]], [3])


def test_sample_tokens_guardless_is_bare_method_call():
    runner = _SampleRunner(None)

    out = runner.sample_tokens("grammar")
    assert out.sampled_token_ids == [[7]]
    # Guardless path is a bare method call — zero guard work.
    assert runner.order == ["body"]
