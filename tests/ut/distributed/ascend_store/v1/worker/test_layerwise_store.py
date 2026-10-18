"""Store admission, failure precedence, and source ownership across layer callbacks."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future

import pytest

from tests.ut.distributed.ascend_store.v1.helpers import FakeBackend, begin_step, make_topology, make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.coordinates import TokenRange
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.protocol.transfer import RangeStoreCommand
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.transfer import LayerStoreResult
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.v1.worker.layerwise.worker import (
    LayerwiseWorker,
    _StoreState,
)


def test_layer_results_update_only_current_keys_and_keep_scalar_success_constant_time():
    class NoScanDict(dict):
        def __iter__(self):
            pytest.fail("Layer results must not scan accumulated state")

        def items(self):
            pytest.fail("Layer results must not scan accumulated state")

        def keys(self):
            pytest.fail("Layer results must not scan accumulated state")

        def values(self):
            pytest.fail("Layer results must not scan accumulated state")

    class NoScanKeys(list):
        def __iter__(self):
            pytest.fail("Scalar success must not expand into per-key results")

    state = _StoreState(())
    state.failure_codes = NoScanDict({"older": -1})
    state.errors_by_key = NoScanDict()
    LayerwiseWorker._record_store_result(state, NoScanKeys(["first", "second"]), LayerStoreResult(0, True))
    LayerwiseWorker._record_store_result(state, ["first", "second"], LayerStoreResult((-9, 0), (True, True)))
    error = RuntimeError("copy completion unknown")
    LayerwiseWorker._record_store_result(state, ["first", "second"], LayerStoreResult(None, False, error))
    assert state.failure_codes == {"older": -1, "first": -9, "second": None}
    assert state.unsafe_keys == {"first", "second"}
    assert state.errors_by_key == {"first": error, "second": error}


def test_store_without_selected_objects_still_observes_cancelled_layer_tasks():
    worker, resources, _ = make_worker(layerwise=True)
    opening: Future[None] = Future()
    opening.set_result(None)
    layer: Future[None] = Future()
    layer.cancel()
    try:
        with pytest.raises(CancelledError):
            worker._finish_store(opening, {2: layer}, _StoreState(()))
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize("start_codes", ((0, 0), (0, -7), (-6, -7)))
@pytest.mark.parametrize("failure_stage", (None, "start", "prepare_layers"))
def test_store_admission_keeps_failed_requests_separate_from_copy_sources(start_codes, failure_stage, monkeypatch):
    backend = FakeBackend()
    backend.store_session_start_result = list(start_codes)
    worker, resources, _ = make_worker(backend, layerwise=True, topology=make_topology(physical_layers=(2, 5, 9)))
    commands = (
        RangeStoreCommand("first", TokenRange(0, 4), ((1,),), (b"a",), 4, 17),
        RangeStoreCommand("second", TokenRange(0, 4), ((3,),), (b"b",), 4, 18),
        RangeStoreCommand("duplicate", TokenRange(0, 4), ((5,),), (b"a",), 4, 19),
    )
    error = RuntimeError(f"failed at {failure_stage}")
    if failure_stage == "start":
        start = worker._backend_sessions.start_store

        def start_then_raise(*args):
            start(*args)
            raise error

        monkeypatch.setattr(worker._backend_sessions, "start_store", start_then_raise)
    elif failure_stage == "prepare_layers":

        def fail_preparation(batch):
            raise error

        monkeypatch.setattr(worker, "_prepare_store_layers", fail_preparation)
    try:
        begin_step(worker, store=commands)
        for layer_id in (2, 2, 5, 5, 9, 9):
            worker.save_layer(f"layers.{layer_id}.group.0")
        worker._finalize_layerwise_store(worker._active_step)
        completions = worker.fence_previous_store()
        assert [completion.request_id for completion in completions] == ["first", "second", "duplicate"]
        for index, completion in enumerate(completions[:2]):
            copied = start_codes[index] == 0 and failure_stage is None
            assert completion.evidence.succeeded == copied
            assert completion.evidence.source_release_confirmed
            evidence = completion.evidence.transfer_evidence
            assert len(evidence) == int(copied)
            if copied:
                assert evidence[0].source.block_id == commands[index].block_ids_by_group[0][0]
                assert evidence[0].source.physical_layer_ids == (2, 5, 9)
                assert evidence[0].result_code == 0
            elif failure_stage == "start" or start_codes[index] == 0:
                assert completion.evidence.error is error
            else:
                assert f"result code {start_codes[index]}" in str(completion.evidence.error)
        assert completions[2].evidence.succeeded
        assert completions[2].evidence.source_release_confirmed
        assert completions[2].evidence.transfer_evidence == ()
        started_keys = next(call[1] for call in backend.calls if call[0] == "batch_put_start")
        copied_keys = tuple(key for key, code in zip(started_keys, start_codes, strict=True) if code == 0)
        copies = [call[1] for call in backend.calls if call[0] == "batch_copy_put"]
        assert copies == ([copied_keys] * 3 if copied_keys and failure_stage is None else [])
        assert worker.take_released_store_job_ids() == {17, 18, 19}
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed


@pytest.mark.parametrize(
    ("copy_codes", "commit_code", "revoke_code", "expected_code"),
    (
        ((-9, -10), 0, 0, -9),
        ((-9, 0), 0, -8, -8),
        ((0, 0), -7, -8, -7),
        ((0, 0), None, -8, -8),
        ((0, 0), None, 0, None),
        ((0, 0), 0, 0, 0),
    ),
)
def test_store_result_precedence_does_not_hide_copy_source_release(
    copy_codes, commit_code, revoke_code, expected_code, monkeypatch
):
    backend = FakeBackend()
    backend.store_session_commit_result = [commit_code] if commit_code is not None else None
    backend.store_session_revoke_result = [revoke_code]
    worker, resources, _ = make_worker(backend, layerwise=True)
    copy = backend.batch_copy_put
    codes = iter(copy_codes)

    def copy_with_next_result(*args):
        backend.store_session_copy_result = [next(codes)]
        return copy(*args)

    monkeypatch.setattr(backend, "batch_copy_put", copy_with_next_result)
    commit_error = RuntimeError("commit failed")
    if commit_code is None:
        commit = backend.batch_commit

        def commit_then_raise(keys):
            commit(keys)
            raise commit_error

        monkeypatch.setattr(backend, "batch_commit", commit_then_raise)
    try:
        command = RangeStoreCommand("request", TokenRange(0, 4), ((1,),), (b"a",), 4, 17)
        begin_step(worker, store=(command,))
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
        worker._finalize_layerwise_store(worker._active_step)
        (completion,) = worker.fence_previous_store()
        evidence = completion.evidence
        assert evidence.succeeded == (copy_codes == (0, 0) and commit_code == 0)
        assert evidence.source_release_confirmed
        assert len(evidence.transfer_evidence) == 1
        assert evidence.transfer_evidence[0].result_code == expected_code
        assert evidence.transfer_evidence[0].source.physical_layer_ids == (0, 1)
        if commit_code is None:
            assert evidence.error is commit_error
        calls = [call[0] for call in backend.calls]
        assert calls.count("batch_commit") == int(copy_codes == (0, 0))
        assert calls.count("batch_revoke") == int(not evidence.succeeded)
        assert worker.take_released_store_job_ids() == {17}
        worker.end_step()
    finally:
        if revoke_code != 0:
            with pytest.raises(RuntimeError, match="cleanup is unconfirmed"):
                worker.close()
            assert not resources.closed
            assert worker._backend_sessions.pending_store_keys
            backend.store_session_revoke_result = [0]
        worker.close()
    assert resources.closed


def test_store_candidate_failure_completes_every_command_without_handing_off_sources(monkeypatch):
    worker, resources, backend = make_worker(layerwise=True)
    error = RuntimeError("candidate preparation failed")

    def fail_candidates(commands):
        raise error

    monkeypatch.setattr(worker, "_build_store_candidates", fail_candidates)
    try:
        commands = (
            RangeStoreCommand("first", TokenRange(0, 4), ((1,),), (b"a",), 4, 17),
            RangeStoreCommand("second", TokenRange(0, 4), ((3,),), (b"b",), 4, 18),
        )
        begin_step(worker, store=commands)
        worker.save_layer("layers.0.group.0")
        worker.save_layer("layers.1.group.0")
        worker._finalize_layerwise_store(worker._active_step)
        completions = worker.fence_previous_store()
        assert len(completions) == 2
        for completion in completions:
            assert not completion.evidence.succeeded
            assert completion.evidence.error is error
            assert completion.evidence.source_release_confirmed
            assert completion.evidence.transfer_evidence == ()
        assert not any(call[0] in ("batch_put_start", "batch_copy_put", "batch_commit") for call in backend.calls)
        assert worker.take_released_store_job_ids() == {17, 18}
        worker.end_step()
    finally:
        worker.close()
    assert resources.closed
