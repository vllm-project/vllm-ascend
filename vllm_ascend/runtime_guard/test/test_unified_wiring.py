"""Unified v1/v2 pre-sample wrap + v2 output proxy + sample-phase wiring tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.runtime_guard.runner_bridge import (
    check_before_sample_from_batch,
    wrap_compute_logits_for_pre_sample,
)
from vllm_ascend.runtime_guard.processor import RuntimeGuardProcessor, SamplePhaseResult
from vllm_ascend.runtime_guard.test._helpers import bare_processor


class _FakeDfx:
    def __init__(self, runner=None):
        self.runner = runner
        self.calls = []

    def check_before_sample(self, **kwargs):
        self.calls.append(kwargs)


class _FakeModel:
    def __init__(self, logits):
        self.logits = logits
        self.calls = 0

    def compute_logits(self, hidden_states):
        self.calls += 1
        return self.logits


def _make_runner(logits, dfx, scheduler_output=None):
    model = _FakeModel(logits)
    return SimpleNamespace(model=model, runtime_guard=dfx, _rg_scheduler_output=scheduler_output), model


def test_wrap_fires_once_for_two_compute_logits_calls():
    dfx = _FakeDfx()
    runner, model = _make_runner("logits-tensor", dfx, scheduler_output="so")
    with wrap_compute_logits_for_pre_sample(runner, "batch"):
        assert model.compute_logits("h") == "logits-tensor"
        assert model.compute_logits("h") == "logits-tensor"
    assert model.calls == 2
    assert len(dfx.calls) == 1  # once-fire: v2 samples + prompt logprobs both call it
    assert dfx.calls[0]["logits"] == "logits-tensor"
    assert dfx.calls[0]["scheduler_output"] == "so"


def test_wrap_restores_compute_logits():
    dfx = _FakeDfx()
    runner, model = _make_runner("L", dfx)
    original = model.compute_logits
    with wrap_compute_logits_for_pre_sample(runner, "batch"):
        pass
    assert model.compute_logits == original
    assert not hasattr(model, "__dict__") or "compute_logits" not in model.__dict__ or model.__dict__["compute_logits"] is original


def test_from_batch_falls_back_to_runner_logits_indices():
    runner = SimpleNamespace(logits_indices=[1, 2])
    dfx = _FakeDfx(runner=runner)
    check_before_sample_from_batch(dfx, "L", SimpleNamespace(), scheduler_output=None)
    call = dfx.calls[0]
    assert call["logits_indices"] == [1, 2]
    assert "positions" not in call


def test_from_batch_prefers_input_batch_logits_indices():
    runner = SimpleNamespace(logits_indices=[9])
    dfx = _FakeDfx(runner=runner)
    batch = SimpleNamespace(logits_indices=[7], num_tokens=5)
    check_before_sample_from_batch(dfx, "L", batch, scheduler_output=None)
    call = dfx.calls[0]
    assert call["logits_indices"] == [7]


def test_from_batch_forwards_scheduler_output():
    dfx = _FakeDfx()
    so = SimpleNamespace(total_num_scheduled_tokens=17)
    check_before_sample_from_batch(dfx, "L", SimpleNamespace(), scheduler_output=so)
    assert dfx.calls[0]["scheduler_output"] is so


def test_run_sample_phase_invokes_spec_hooks():
    """Production golden path: check_after_spec runs via orchestrator."""
    RuntimeGuardProcessor.reset_for_tests()
    p = bare_processor()
    calls: list[str] = []

    def after_spec(*_a, **_k):
        calls.append("check_after_spec")

    p.check_after_spec = after_spec  # type: ignore[method-assign]
    p.should_check_after_spec = lambda: True  # type: ignore[method-assign]
    p.mark_finished = lambda *a, **k: calls.append("mark_finished")  # type: ignore[method-assign]
    p.record_sample_waves = lambda *a, **k: calls.append("waves")  # type: ignore[method-assign]
    p.check_after_sample = lambda *a, **k: calls.append("after_sample")  # type: ignore[method-assign]

    def sample_fn():
        calls.append("sample")
        return SamplePhaseResult(
            scheduler_output=None,
            input_batch=None,
            model_runner_output=None,
            sampler_output=SimpleNamespace(sampled_token_ids=[1]),
            valid_sampled_token_ids=[1],
            req_ids_output_copy=["r1"],
            invalid_req_indices=None,
            finished_req_ids=None,
        )

    p.run_sample_phase(
        sample_fn=sample_fn,
        speculative_config=object(),
        need_accepted_tokens=False,
        use_async=False,
        accepted_token_nums_fn=lambda _r: [1],
    )
    assert calls[0] == "sample"
    assert "sample" in calls
    assert "check_after_spec" in calls
    assert "after_sample" in calls
    # Spec check runs before wave stamp / after_sample in the orchestrator.
    assert calls.index("check_after_spec") < calls.index("after_sample")


def test_runners_call_run_sample_phase():
    """Source contract: v1/v2 sample_tokens must invoke the orchestrator."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "vllm_ascend" / "worker"
    v1 = (root / "model_runner_v1.py").read_text(encoding="utf-8")
    v2 = (root / "v2" / "model_runner.py").read_text(encoding="utf-8")
    assert "run_sample_phase(" in v1
    assert "run_sample_phase(" in v2
    assert "_rg_spec_num_sampled" in v2


def test_d11_async_model_runner_output_defers_after_sample():
    """W2-3/D-11: sync scheduling + AsyncModelRunnerOutput must not after-sample early.

    v2 always returns AsyncOutput; padded sampled_token_ids inflate IO until
    get_output trims with num_sampled.
    """
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

    from vllm_ascend.runtime_guard.processor import SamplePhaseResult

    class _FakeAsync(AsyncModelRunnerOutput):
        def get_output(self) -> ModelRunnerOutput:
            raise AssertionError("run_sample_phase must not call get_output")

    RuntimeGuardProcessor.reset_for_tests()
    p = bare_processor()
    calls: list[str] = []
    p.mark_finished = lambda *a, **k: calls.append("mark_finished")  # type: ignore[method-assign]
    p.record_sample_waves = lambda *a, **k: calls.append("waves")  # type: ignore[method-assign]
    p.check_after_sample = lambda *a, **k: calls.append("after_sample")  # type: ignore[method-assign]
    p.should_check_after_spec = lambda: False  # type: ignore[method-assign]
    p.needs_sample_phase_hooks = lambda: True  # type: ignore[method-assign]

    padded = [[7, 0, 0, 0]]  # would over-append if checked here

    def sample_fn():
        return SamplePhaseResult(
            scheduler_output=None,
            input_batch=None,
            model_runner_output=_FakeAsync(),
            sampler_output=SimpleNamespace(sampled_token_ids=padded),
            valid_sampled_token_ids=padded,
            req_ids_output_copy=["r1"],
            invalid_req_indices=None,
            finished_req_ids=None,
        )

    p.run_sample_phase(
        sample_fn=sample_fn,
        speculative_config=None,
        need_accepted_tokens=False,
        use_async=False,
    )
    assert "waves" in calls
    assert "after_sample" not in calls


def test_d11_trim_sampled_rows_drops_pad_beyond_num_sampled():
    """Padded v2 rows must keep only num_sampled tokens (AsyncOutput.get_output)."""
    import numpy as np

    from vllm_ascend.runtime_guard.token_utils import trim_sampled_rows

    padded = np.array([[42, 0, 0, 7], [9, -1, -1, -1]], dtype=np.int64)
    trimmed = trim_sampled_rows(padded, num_sampled=np.array([1, 1], dtype=np.int32))
    assert trimmed == [[42], [9]]


def test_d11_ascend_async_output_appends_trimmed_once():
    """AscendAsyncOutput.get_output is the sole after-sample append for v2."""
    from vllm_ascend.runtime_guard.io_snapshot import RequestIoSnapshotManager
    from vllm_ascend.runtime_guard.request_state import RequestGuardStore
    from vllm_ascend.runtime_guard.runner_bridge import AscendAsyncOutput

    RequestGuardStore.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()

    out = SimpleNamespace(sampled_token_ids=[[42]], req_ids=["poem"])
    inner = MagicMock()
    inner.get_output.return_value = out

    appended: list[tuple] = []

    class _Guard:
        def check_after_sample(self, sampled_token_ids, req_ids=None):
            RequestIoSnapshotManager.get().append_batch(req_ids, sampled_token_ids)
            appended.append((list(sampled_token_ids), list(req_ids or [])))

    runner = SimpleNamespace(runtime_guard=_Guard())
    assert AscendAsyncOutput(inner, runner).get_output() is out
    assert appended == [([[42]], ["poem"])]
    st = RequestGuardStore.get().get_state("poem")
    assert st is not None
    assert st.output_token_ids == [42]


class _FakeUvaBuf:
    def __init__(self, host):
        self.np = host


class _FakeStaged:
    """Staged-tensor stand-in: host mirror via ``_uva_buf.np`` + device view."""

    def __init__(self, host, gpu):
        self._uva_buf = _FakeUvaBuf(host)
        self.gpu = gpu


def _fake_req_states(rows):
    """MRV2-like ``req_states`` from ``(prompt_ids, output_ids)`` pairs.

    Mirrors the NPU reality: decode appends land only on ``.gpu``; the host
    mirror inside ``_uva_buf.np`` stays zero forever.
    """
    import numpy as np
    import torch

    n = len(rows)
    max_len = max(len(p) + len(o) for p, o in rows)
    host_all = np.zeros((n, max_len), dtype=np.int64)
    gpu_all = torch.zeros((n, max_len), dtype=torch.int64)
    for i, (p, o) in enumerate(rows):
        seq = list(p) + list(o)
        gpu_all[i, : len(seq)] = torch.tensor(seq, dtype=torch.int64)
    return SimpleNamespace(
        req_id_to_index={f"req-{i}": i for i in range(n)},
        prompt_len=SimpleNamespace(np=np.array([len(p) for p, _ in rows], dtype=np.int64)),
        # 1-D, shape [max_num_reqs] — matches vllm gpu states.py total_len.
        total_len=_FakeStaged(np.zeros(n, dtype=np.int64), torch.tensor([len(p) + len(o) for p, o in rows])),
        all_token_ids=_FakeStaged(host_all, gpu_all),
    )


def test_read_staged_row_prefers_fresh_host_mirror():
    import numpy as np

    from vllm_ascend.runtime_guard.io_snapshot import _read_staged_row

    staged = _FakeStaged(np.array([[5, 6, 7, 0]], dtype=np.int64), gpu=None)
    assert _read_staged_row(staged, 0, 3) == [5, 6, 7]
    assert _read_staged_row(staged, 0, 0) == []


def test_read_staged_row_falls_to_gpu_when_host_mirror_stale():
    import numpy as np
    import torch

    from vllm_ascend.runtime_guard.io_snapshot import _read_staged_row

    staged = _FakeStaged(np.zeros((1, 4), dtype=np.int64), torch.tensor([[1, 2, 3, 4]]))
    assert _read_staged_row(staged, 0, 4) == [1, 2, 3, 4]


def test_req_state_index_prefers_explicit_idx_then_id_map():
    from vllm_ascend.runtime_guard.io_snapshot import _req_state_index

    states = _fake_req_states([([1], [2]), ([3], [4])])
    assert _req_state_index(states, "req-1", None) == 1
    assert _req_state_index(states, "req-0", 0) == 0
    assert _req_state_index(states, "missing", None) is None


def test_output_from_req_states_reads_device_side_appends():
    from vllm_ascend.runtime_guard.io_snapshot import _output_from_req_states

    runner = SimpleNamespace(req_states=_fake_req_states([([10, 11, 12, 13], [90, 91, 92])]))
    assert _output_from_req_states(runner, "req-0", 0) == (3, [90, 91, 92])


def test_output_token_count_uses_v2_staged_tensors_when_v1_paths_empty():
    from vllm_ascend.runtime_guard.io_snapshot import output_token_count_for_request
    from vllm_ascend.runtime_guard.request_state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    runner = SimpleNamespace(req_states=_fake_req_states([([10, 11], [42, 43, 44])]))
    assert output_token_count_for_request(runner, "req-0", 0) == 3


def test_snapshot_reports_prompt_and_output_for_v2_runner():
    from vllm_ascend.runtime_guard.io_snapshot import RequestIoSnapshotManager
    from vllm_ascend.runtime_guard.request_state import RequestGuardStore

    RequestGuardStore.reset_for_tests()
    RequestIoSnapshotManager.reset_for_tests()

    runner = SimpleNamespace(req_states=_fake_req_states([([10, 11], [42, 43])]))
    snap = RequestIoSnapshotManager.get().snapshot(runner, "req-0", 0, include_token_ids=True)
    assert snap.prompt_token_ids == [10, 11]
    assert snap.prompt_token_count == 2
    assert snap.output_token_ids == [42, 43]
    assert snap.output_token_count == 2
