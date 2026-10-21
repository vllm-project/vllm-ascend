# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Ownership failures after a real native enqueue, without engine mocking.

These tests prove the owner protocol at a device boundary. They do not prove
that an engine/backend joins all of its secondary-stream readers.
"""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.worker.v2.spec_decode.draft_window_lifetime import (  # noqa: E402
    DraftWindowOwnerPool,
    DraftWindowOwnerState,
)

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires an Ascend NPU"
)


class NativeCompletion:
    def __init__(self, stream):
        self.event = torch.npu.Event()
        self.event.record(stream)

    def is_complete(self):
        return self.event.query()

    def wait(self):
        self.event.synchronize()


@pytest.mark.parametrize("failure", ["host_after_enqueue", "event_create", "event_record", "submit_result"])
def test_native_enqueue_failure_quarantines_backing(failure):
    pool = DraftWindowOwnerPool(1)
    stream = torch.npu.Stream()
    payload = torch.arange(1024, dtype=torch.float32, device="npu")
    stream.wait_stream(torch.npu.current_stream())

    def completion():
        if failure == "event_create":
            raise RuntimeError("injected event creation failure")
        event = torch.npu.Event()
        if failure == "event_record":
            raise RuntimeError("injected event recording failure")
        event.record(stream)
        if failure == "submit_result":
            # Real work and a real recorded event exist, but the owner did
            # not receive a valid completion token.
            return event
        return NativeCompletion(stream)

    with pytest.raises((RuntimeError, TypeError)), torch.npu.stream(stream), pool.operation(completion) as owner:
        pool.retain(owner, payload)
        pool.begin_native(owner)
        observed = payload.square()
        pool.retain(owner, observed)
        if failure == "host_after_enqueue":
            raise RuntimeError("injected host failure after native enqueue")
    assert owner.state is DraftWindowOwnerState.QUARANTINED
    assert owner.retained[0] is payload
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()
    # Only test cleanup waits for the real stream. Production failure does
    # not infer completion or release a quarantined context.
    stream.synchronize()
    torch.testing.assert_close(observed.cpu(), torch.arange(1024, dtype=torch.float32).square())
    with pytest.raises(RuntimeError, match="quarantined"):
        pool.acquire()


def test_native_completion_covers_an_explicit_late_reader_join():
    pool = DraftWindowOwnerPool(1)
    producer, reader = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.stream(producer), pool.operation(lambda: NativeCompletion(producer)) as owner:
        pool.begin_native(owner)
        payload = torch.arange(1024, dtype=torch.float32, device="npu")
        pool.retain(owner, payload)
        reader.wait_stream(producer)
        with torch.npu.stream(reader):
            result = payload * 3
        pool.retain(owner, result)
        # The actual executor must provide an equivalent join for every
        # secondary reader before recording its proposal completion.
        producer.wait_stream(reader)
    replacement = pool.acquire()
    assert replacement.lane == owner.lane
    assert owner.state is DraftWindowOwnerState.RELEASED
    torch.testing.assert_close(result.cpu(), torch.arange(1024, dtype=torch.float32) * 3)
