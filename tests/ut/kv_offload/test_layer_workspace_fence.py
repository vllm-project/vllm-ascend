# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future

import pytest

from vllm_ascend.distributed.kv_transfer.layer_workspace_fence import (
    LayerWorkspaceFence,
    RequestChunkKey,
    WorkspaceIdentityError,
    WorkspaceKey,
    WorkspaceTransferError,
)


class FakeEvent:
    def __init__(self, error: BaseException | None = None):
        self.error = error
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1
        if self.error is not None:
            raise self.error


def test_request_and_workspace_identity_validation():
    assert RequestChunkKey("request", 1, 2).chunk_id == 2
    with pytest.raises(ValueError):
        RequestChunkKey("", 0, 0)
    with pytest.raises(ValueError):
        WorkspaceKey(0, -1, 0, 0)


def test_fence_waits_for_every_registered_source():
    key = WorkspaceKey(1, 2, 3, 4)
    fence = LayerWorkspaceFence(key)
    first: Future[None] = Future()
    second: Future[None] = Future()
    fence.add_source_future(first, key)
    fence.add_source_future(second, key)
    fence.close_registration()
    first.set_result(None)
    assert fence.pending == 1
    with pytest.raises(TimeoutError):
        fence.wait_source_safe(timeout=0)
    second.set_result(None)
    fence.wait_source_safe(timeout=0)


def test_fence_rejects_wrong_generation_before_counting():
    key = WorkspaceKey(1, 2, 3, 4)
    fence = LayerWorkspaceFence(key)
    with pytest.raises(WorkspaceIdentityError):
        fence.add_source_future(Future(), WorkspaceKey(1, 2, 3, 5))
    assert fence.pending == 0


def test_fence_propagates_failure_and_cancel():
    key = WorkspaceKey(1, 2, 3, 4)
    for complete in (
        lambda future: future.set_exception(ValueError("transfer failed")),
        lambda future: future.cancel(),
    ):
        fence = LayerWorkspaceFence(key)
        future: Future[None] = Future()
        fence.add_source_future(future, key)
        fence.close_registration()
        complete(future)
        with pytest.raises(WorkspaceTransferError):
            fence.wait_source_safe(timeout=0)


def test_empty_fence_completes_after_registration_closes():
    fence = LayerWorkspaceFence(WorkspaceKey(1, 2, 3, 4))
    fence.close_registration()
    fence.wait_source_safe(timeout=0)


def test_workspace_reuse_waits_for_source_and_compute_release():
    key = WorkspaceKey(1, 2, 3, 4)
    fence = LayerWorkspaceFence(key)
    source: Future[None] = Future()
    event = FakeEvent()
    fence.add_source_future(source, key)
    fence.set_compute_release_event(event, key)
    fence.close_registration()

    with pytest.raises(TimeoutError):
        fence.wait_workspace_reusable(timeout=0)
    assert event.synchronize_calls == 1

    source.set_result(None)
    fence.wait_workspace_reusable(timeout=0)
    assert event.synchronize_calls == 2


def test_workspace_reuse_drains_compute_after_source_failure():
    key = WorkspaceKey(1, 2, 3, 4)
    fence = LayerWorkspaceFence(key)
    source: Future[None] = Future()
    event = FakeEvent()
    fence.add_source_future(source, key)
    fence.set_compute_release_event(event, key)
    fence.close_registration()
    source.set_exception(ValueError("transfer failed"))

    with pytest.raises(WorkspaceTransferError):
        fence.wait_workspace_reusable(timeout=0)
    assert event.synchronize_calls == 1


def test_workspace_reuse_requires_compute_release_event():
    fence = LayerWorkspaceFence(WorkspaceKey(1, 2, 3, 4))
    fence.close_registration()

    with pytest.raises(RuntimeError, match="Compute release event is missing"):
        fence.wait_workspace_reusable(timeout=0)


def test_compute_release_event_validates_identity_and_lifecycle():
    key = WorkspaceKey(1, 2, 3, 4)
    fence = LayerWorkspaceFence(key)
    with pytest.raises(WorkspaceIdentityError):
        fence.set_compute_release_event(FakeEvent(), WorkspaceKey(1, 2, 3, 5))

    fence.set_compute_release_event(FakeEvent(), key)
    with pytest.raises(RuntimeError, match="already set"):
        fence.set_compute_release_event(FakeEvent(), key)

    closed_fence = LayerWorkspaceFence(key)
    closed_fence.close_registration()
    with pytest.raises(RuntimeError, match="after registration closes"):
        closed_fence.set_compute_release_event(FakeEvent(), key)
