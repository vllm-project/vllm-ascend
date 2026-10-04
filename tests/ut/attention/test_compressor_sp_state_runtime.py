# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Compressor SP single-stream state runtime.

Covers the items that are checkable on CPU with mocks:
- gather-handle join uses Work.wait() and nothing else
- state runtime registry keying / mismatch assert / reset
- drain() physical completion and bookkeeping reset
- STATE_PG strict parsing and rank-uniform group creation loop
- the cross-layer edge that keeps the shared state buffers from being reused
  while the previous layer's chain is still in flight
"""

from dataclasses import dataclass, field

import pytest

import vllm_ascend.attention.context_parallel.dsa_cp as dsa_cp
import vllm_ascend.attention.dsa_compressor as dsa_compressor
import vllm_ascend.envs as envs
from vllm_ascend.attention.dsa_compressor import CompressorExecutor, CompressorSPGatherHandle


@dataclass
class _FakeEvent:
    synchronized: int = 0

    def synchronize(self) -> None:
        self.synchronized += 1


@dataclass
class _FakeGroup:
    ranks: list
    device_group: object = None


@dataclass
class _FakeWork:
    waited: int = 0

    def wait(self) -> None:
        self.waited += 1


@dataclass
class _FakeCompressor:
    """Minimal surface CompressorExecutor reads during construction."""

    compress_ratio: int = 4
    coff: int = 2


@dataclass
class _FakeSpMetadata:
    """The two cross-layer shared state buffers, as opaque handles."""

    state_send_buffer: object = field(default_factory=object)
    gathered_state_buffer: object = field(default_factory=object)


@dataclass
class _FakeRuntime:
    """Stand-in for CompressorSPStateRuntime's gather-side contract."""

    last_done_event: object = None
    submitted: list = field(default_factory=list)

    def submit_deferred(self, executor, deferred) -> None:
        self.submitted.append((executor, deferred))


class _RecordingStream:
    """Fake state stream that logs its ordering edges into a shared trace.

    Waited events are kept separately so assertions compare them by identity:
    ``_FakeEvent`` is a dataclass, so two distinct events would compare equal.
    Doubles as its own ``torch_npu.npu.stream`` context manager so the test
    needs no stream-context stub.
    """

    def __init__(self, trace: list) -> None:
        self.trace = trace
        self.waited_events: list = []

    def wait_event(self, event) -> None:
        self.trace.append("wait_event")
        self.waited_events.append(event)

    def __enter__(self) -> "_RecordingStream":
        return self

    def __exit__(self, *exc) -> bool:
        return False


class TestGatherHandleJoin:
    """Work.wait() is the only join. It covers completion on any stream.

    torch_npu runs the collective on the process group's internal stream, so an
    event recorded on the issuing stream captures the launch point, not
    completion -- there is deliberately no event edge left to assert.
    """

    def test_async_handle_joins_the_work_and_returns_rows(self):
        handle = CompressorSPGatherHandle(
            recv_buffer="rows", send_buffer=object(), work=_FakeWork()
        )
        assert handle.wait() == "rows"
        assert handle.work.waited == 1

    def test_inline_handle_short_circuits(self):
        handle = CompressorSPGatherHandle(recv_buffer="rows", send_buffer=object())
        assert handle.wait() == "rows"

    def test_handle_carries_no_stream_or_event_fields(self):
        # Regression guard: reintroducing a side stream or a done event here
        # would bring back the launch-point-vs-completion confusion.
        fields = CompressorSPGatherHandle.__dataclass_fields__
        assert set(fields) == {"recv_buffer", "send_buffer", "work"}


class TestStateRuntimeRegistry:
    def _patch_stream(self, monkeypatch):
        monkeypatch.setattr(
            dsa_cp.torch_npu.npu, "Stream", lambda: object(), raising=False
        )

    def test_keyed_by_ranks_and_reset(self, monkeypatch):
        self._patch_stream(monkeypatch)
        dsa_cp.reset_compressor_sp_state_runtimes()
        g1 = _FakeGroup(ranks=[0, 1, 2, 3], device_group=object())
        g2 = _FakeGroup(ranks=[4, 5, 6, 7], device_group=object())
        rt1 = dsa_cp.compressor_sp_state_runtime(g1)
        assert dsa_cp.compressor_sp_state_runtime(g1) is rt1
        assert dsa_cp.compressor_sp_state_runtime(g2) is not rt1
        assert len(dsa_cp._COMPRESSOR_SP_STATE_RUNTIMES) == 2
        dsa_cp.reset_compressor_sp_state_runtimes()
        assert len(dsa_cp._COMPRESSOR_SP_STATE_RUNTIMES) == 0

    def test_same_ranks_different_group_raises(self, monkeypatch):
        self._patch_stream(monkeypatch)
        dsa_cp.reset_compressor_sp_state_runtimes()
        g1 = _FakeGroup(ranks=[0, 1], device_group=object())
        dsa_cp.compressor_sp_state_runtime(g1)
        g1_bis = _FakeGroup(ranks=[0, 1], device_group=object())
        with pytest.raises(RuntimeError, match="different TP process group"):
            dsa_cp.compressor_sp_state_runtime(g1_bis)


class TestDrainLifecycle:
    def _runtime(self, monkeypatch):
        monkeypatch.setattr(
            dsa_cp.torch_npu.npu, "Stream", lambda: object(), raising=False
        )
        dsa_cp.reset_compressor_sp_state_runtimes()
        return dsa_cp.compressor_sp_state_runtime(
            _FakeGroup(ranks=[0], device_group=object())
        )

    def test_drain_synchronizes_and_resets(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_ASCEND_COMPRESSOR_SP_STATE_DEBUG", 0, raising=False)
        rt = self._runtime(monkeypatch)
        event = _FakeEvent()
        rt.submit(state_done_event=event, work=None, sp_metadata=object(), state_cache=object())
        assert rt.layers_submitted == 1 and rt.last_done_event is event
        rt.drain()
        assert event.synchronized == 1
        assert rt.pending_refs == [] and rt.last_done_event is None
        assert rt.layers_submitted == 0

    def test_deferred_apply_attaches_on_demand(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_ASCEND_COMPRESSOR_SP_STATE_DEBUG", 0, raising=False)
        rt = self._runtime(monkeypatch)

        attached = []

        class _FakeExecutor:
            def attach_sp_state(self, deferred, apply_stream):
                ev = _FakeEvent()
                attached.append((deferred, apply_stream, ev))
                return ev

        @dataclass
        class _Deferred:
            work: object
            sp_metadata: object
            state_cache: object

        d1, d2 = _Deferred(object(), object(), object()), _Deferred(object(), object(), object())
        rt.submit_deferred(_FakeExecutor(), d1)
        rt.submit_deferred(_FakeExecutor(), d2)
        assert rt.layers_submitted == 2 and rt.pending_applies
        rt.attach_pending()
        assert len(attached) == 2 and not rt.pending_applies
        assert rt.last_done_event is attached[-1][2]
        # Single stream: the apply half runs on the same stream as the gather.
        assert not hasattr(rt, "state_apply_stream")
        assert all(stream is rt.state_stream for _, stream, _ in attached)
        rt.drain()
        assert attached[-1][2].synchronized == 1
        assert rt.pending_refs == [] and rt.last_done_event is None

    def test_drain_noop_without_submissions(self, monkeypatch):
        rt = self._runtime(monkeypatch)
        rt.drain()  # must not raise

    def test_debug_flag_fail_closed(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_ASCEND_COMPRESSOR_SP_STATE_DEBUG", 2, raising=False)
        rt = self._runtime(monkeypatch)
        rt.submit(
            state_done_event=_FakeEvent(), work=None,
            sp_metadata=object(), state_cache=object(),
        )
        with pytest.raises(ValueError, match="STATE_DEBUG"):
            rt.drain()


class TestStatePgCreation:
    def test_all_subgroups_created_before_return(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_ASCEND_COMPRESSOR_SP_STATE_PG", 1, raising=False)
        calls: list[list[int]] = []
        sentinel = object()

        def fake_new_group(ranks):
            calls.append(list(ranks))
            return sentinel if list(ranks) == [4, 5, 6, 7] else None

        monkeypatch.setattr(dsa_cp.dist, "new_group", fake_new_group, raising=False)
        monkeypatch.setattr(dsa_cp.dist, "get_world_size", lambda: 16, raising=False)
        group = dsa_cp._create_compressor_sp_state_group(
            _FakeGroup(ranks=[4, 5, 6, 7], device_group=object())
        )
        # Every world rank executes this loop; early return after the own
        # subgroup would deadlock the remaining ranks inside new_group.
        assert calls == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        assert group is sentinel

    def test_non_divisible_world_raises(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_ASCEND_COMPRESSOR_SP_STATE_PG", 1, raising=False)
        monkeypatch.setattr(dsa_cp.dist, "get_world_size", lambda: 10, raising=False)
        with pytest.raises(RuntimeError, match="not divisible"):
            dsa_cp._create_compressor_sp_state_group(
                _FakeGroup(ranks=[0, 1, 2, 3], device_group=object())
            )

    def test_invalid_flag_fails_closed(self, monkeypatch):
        monkeypatch.setattr(envs, "VLLM_ASCEND_COMPRESSOR_SP_STATE_PG", "yes", raising=False)
        with pytest.raises(ValueError, match="STATE_PG"):
            dsa_cp._create_compressor_sp_state_group(
                _FakeGroup(ranks=[0], device_group=object())
            )


class TestCrossLayerStateBufferReuse:
    """Layer N's gather must not touch the shared buffers before layer N-1 applies.

    ``state_send_buffer`` / ``gathered_state_buffer`` live on the metadata object
    that every layer in the cache group shares, so the gather half has to be
    ordered behind the previous chain's apply-done event. Losing that edge is
    silent: it only corrupts state once HCCL backlogs a layer.
    """

    @staticmethod
    def _launch(monkeypatch, prev_done_event):
        trace: list = []
        stream = _RecordingStream(trace)
        ready_event = _FakeEvent()
        work = _FakeWork()

        executor = CompressorExecutor(
            compressor=_FakeCompressor(),
            rope_head_dim=None,
            tp_group=_FakeGroup(ranks=[0, 1], device_group=object()),
        )
        monkeypatch.setattr(
            CompressorExecutor,
            "_state_read_send_buffer",
            lambda self, cache, meta: trace.append("read_send_buffer"),
        )
        monkeypatch.setattr(
            dsa_compressor.torch_npu.npu, "stream", lambda s: s, raising=False
        )

        def _fake_all_gather(output, input_, group=None, async_op=False):
            trace.append("all_gather")
            return work

        monkeypatch.setattr(
            dsa_compressor.dist, "all_gather_into_tensor", _fake_all_gather, raising=False
        )

        runtime = _FakeRuntime(last_done_event=prev_done_event)
        executor.launch_sp_state(
            object(),
            _FakeSpMetadata(),
            state_stream=stream,
            state_ready_event=ready_event,
            runtime=runtime,
        )
        return trace, stream, ready_event, runtime

    def test_gather_is_ordered_behind_previous_layer_apply(self, monkeypatch):
        prev_done_event = _FakeEvent()
        trace, stream, ready_event, runtime = self._launch(monkeypatch, prev_done_event)
        # Both edges land BEFORE the read overwrites the shared send buffer and
        # before the all-gather overwrites the shared recv buffer.
        assert trace == ["wait_event", "wait_event", "read_send_buffer", "all_gather"]
        # Identity, not dataclass equality: the previous chain's apply-done
        # event must come first, the producer event second.
        assert stream.waited_events[0] is prev_done_event
        assert stream.waited_events[1] is ready_event
        assert len(runtime.submitted) == 1

    def test_first_layer_of_a_step_waits_only_the_producer(self, monkeypatch):
        # drain() clears last_done_event, so the first chain of a step has no
        # predecessor and must not fault on the missing event.
        trace, stream, ready_event, _ = self._launch(monkeypatch, None)
        assert trace == ["wait_event", "read_send_buffer", "all_gather"]
        assert stream.waited_events == [ready_event]


class TestStatePgDefault:
    def test_dedicated_state_pg_is_on_by_default(self):
        # A single communicator serializes the ms-scale state all-gathers ahead
        # of the next layer's latency-critical row gathers, and no arrangement
        # of user streams can fix that -- only a second process group can.
        assert envs.VLLM_ASCEND_COMPRESSOR_SP_STATE_PG == 1
