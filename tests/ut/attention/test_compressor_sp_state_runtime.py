# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Compressor SP dual-stream state runtime additions.

Covers the review P0 items that are checkable on CPU with mocks:
- AGG_WAIT_MODE fail-closed validation
- state runtime registry keying / mismatch assert / reset
- drain() physical completion and bookkeeping reset
- STATE_PG strict parsing and rank-uniform group creation loop
"""

from dataclasses import dataclass, field

import pytest

import vllm_ascend.envs as envs
import vllm_ascend.attention.context_parallel.dsa_cp as dsa_cp
from vllm_ascend.attention.dsa_compressor import CompressorSPGatherHandle


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


def _make_handle(mode: str, monkeypatch) -> CompressorSPGatherHandle:
    monkeypatch.setattr(
        envs, "VLLM_ASCEND_COMPRESSOR_SP_AGG_WAIT_MODE", mode, raising=False
    )
    return CompressorSPGatherHandle(
        recv_buffer=object(),
        send_buffer=object(),
        work=_FakeWork(),
        comm_stream=object(),
        done_event=_FakeEvent(),
    )


class TestAggWaitModeFailClosed:
    def test_invalid_mode_raises(self, monkeypatch):
        for bad in ("Both", "boht", "", "EVENT", "1"):
            handle = _make_handle(bad, monkeypatch)
            with pytest.raises(ValueError, match="AGG_WAIT_MODE"):
                handle.wait()

    def test_valid_modes_dispatch(self, monkeypatch):
        handle = _make_handle("both", monkeypatch)
        monkeypatch.setattr(
            "torch.npu.current_stream",
            lambda: type("S", (), {"wait_event": staticmethod(lambda e: None)})(),
            raising=False,
        )
        handle.wait()
        assert handle.work.waited == 1
        handle = _make_handle("event", monkeypatch)
        handle.wait()
        assert handle.work.waited == 0

    def test_inline_handle_short_circuits(self):
        handle = CompressorSPGatherHandle(recv_buffer="rows", send_buffer=object())
        assert handle.wait() == "rows"


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
