#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for the unified stream registry and event ledger.

All logic tests run on CPU via the torch/torch_npu mocks installed by
``tests/ut/conftest.py``. Tests that observe real NPU stream objects are
marked NPU-only and skipped when ``torch.npu.is_available()`` is False.
"""

from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.utils as utils_mod
from tests.ut.base import PytestBase
from vllm_ascend.overlap.events import (
    get_event_ledger,
    reset_event_ledger,
)
from vllm_ascend.overlap.streams import (
    _NEW_STREAM_NAMES,
    StreamRegistry,
    get_stream_registry,
    reset_stream_registry,
)


def _npu_available() -> bool:
    return torch.npu.is_available() is True


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Give every test a pristine registry; also clear the legacy globals."""
    reset_stream_registry()
    reset_event_ledger()
    yield
    reset_stream_registry()
    reset_event_ledger()


class TestStreamRegistry(PytestBase):
    def test_get_stream_registry_returns_same_instance(self):
        assert get_stream_registry() is get_stream_registry()

    def test_registry_reset_preserves_singleton(self):
        registry = get_stream_registry()
        registry.get_stream("global_computation")
        reset_stream_registry()
        assert get_stream_registry() is registry
        assert registry.peek_stream("global_computation") is None

    @pytest.mark.parametrize("name", sorted(_NEW_STREAM_NAMES))
    def test_same_name_returns_same_stream(self, name):
        registry = get_stream_registry()
        first = registry.get_stream(name)
        second = registry.get_stream(name)
        assert first is second

    @pytest.mark.parametrize("name", sorted(_NEW_STREAM_NAMES))
    def test_lazy_creation_no_stream_before_access(self, name):
        registry = get_stream_registry()
        assert registry.peek_stream(name) is None
        registry.get_stream(name)
        assert registry.peek_stream(name) is not None

    def test_lazy_creation_does_not_create_other_streams(self):
        registry = get_stream_registry()
        registry.get_stream("global_computation")
        for name in _NEW_STREAM_NAMES - {"global_computation"}:
            assert registry.peek_stream(name) is None

    def test_current_computation_snapshots_ambient_stream(self):
        registry = get_stream_registry()
        ambient = MagicMock(name="ambient_stream")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(torch.npu, "current_stream", MagicMock(return_value=ambient))
            snapshot = registry.get_stream("current_computation")
        assert snapshot is ambient
        # Subsequent accesses must return the cached snapshot even after the
        # ambient stream changes (legacy utils.current_stream semantics).
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(torch.npu, "current_stream", MagicMock(return_value=MagicMock()))
            assert registry.get_stream("current_computation") is ambient

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            get_stream_registry().get_stream("does_not_exist")

    def test_registered_names_are_expected(self):
        assert set(get_stream_registry().registered_names()) == _NEW_STREAM_NAMES | {"current_computation"}

    def test_attention_calculation_not_registered(self):
        # attention_calculation_stream has no in-tree consumer; its stream
        # must therefore not be managed by the registry.
        assert "attention_calculation" not in get_stream_registry().registered_names()


class TestLegacyAccessorsDelegateToRegistry(PytestBase):
    """The legacy module-level accessors must serve registry-owned streams."""

    def test_current_stream_matches_registry(self):
        legacy = utils_mod.current_stream()
        assert legacy is get_stream_registry().get_stream("current_computation")

    def test_global_stream_matches_registry(self):
        legacy = utils_mod.global_stream()
        assert legacy is get_stream_registry().get_stream("global_computation")

    def test_shared_experts_stream_matches_registry(self):
        legacy = utils_mod.shared_experts_calculation_stream()
        assert legacy is get_stream_registry().get_stream("shared_experts")

    def test_cp_chunkedprefill_stream_matches_registry(self):
        legacy = utils_mod.cp_chunkedprefill_comm_stream()
        assert legacy is get_stream_registry().get_stream("cp_chunked_prefill")

    def test_legacy_global_reset_forces_new_lookup_not_new_object(self):
        # Resetting only the module global (as tests/ut/conftest.py does)
        # must still resolve to the same registry-owned stream.
        first = utils_mod.global_stream()
        utils_mod._GLOBAL_STREAM = None
        second = utils_mod.global_stream()
        assert first is second

    def test_moe_comm_stream_mirror(self):
        import vllm_ascend.ops.fused_moe.moe_utils as moe_utils_mod

        stream = moe_utils_mod._moe_comm_stream()
        assert stream is get_stream_registry().get_stream("moe_comm")
        assert moe_utils_mod.COMM_STREAM is stream

    def test_dsa_overlap_stream_matches_registry(self):
        from vllm_ascend.attention.dsa_v1 import dsv4_dsa_overlap_stream

        assert dsv4_dsa_overlap_stream() is get_stream_registry().get_stream("dsa_overlap")

    def test_attention_calculation_stream_stays_outside_registry(self):
        # The accessor must keep working (no deletion) but without touching
        # the registry.
        stream = utils_mod.attention_calculation_stream()
        assert stream is not None
        assert get_stream_registry().peek_stream("attention_calculation") is None
        utils_mod._ATNN_CALCULATION_STREAM = None


class TestRegistryWithRealStreams(PytestBase):
    """Sanity checks against genuine torch_npu streams; NPU-only."""

    @pytest.mark.skipif(not _npu_available(), reason="requires NPU hardware")
    def test_real_streams_are_npu_stream_objects(self):
        registry = get_stream_registry()
        for name in sorted(_NEW_STREAM_NAMES):
            stream = registry.get_stream(name)
            assert isinstance(stream, torch.npu.Stream)

    @pytest.mark.skipif(not _npu_available(), reason="requires NPU hardware")
    def test_real_streams_are_distinct(self):
        registry = get_stream_registry()
        streams = [registry.get_stream(name) for name in sorted(_NEW_STREAM_NAMES)]
        for i, stream in enumerate(streams):
            for other in streams[i + 1 :]:
                assert stream is not other


class TestEventLedger(PytestBase):
    def test_record_returns_event_and_reuses_it(self):
        ledger = get_event_ledger()
        first = ledger.record("before_dispatch")
        assert first is not None
        second = ledger.record("before_dispatch")
        assert first is second

    def test_record_uses_current_stream_by_default(self):
        ledger = get_event_ledger()
        event = ledger.record("before_gmm2")
        assert event is not None
        # Default stream resolution calls torch.npu.current_stream(); the
        # event is retrievable and was recorded once.
        assert ledger.get_event("before_gmm2") is event
        event.record.assert_called_once()

    def test_record_with_explicit_stream(self):
        ledger = get_event_ledger()
        stream = MagicMock(name="stream")
        event = ledger.record("before_combine", stream=stream)
        assert event is not None
        # The recorded event must be told about the explicit stream.
        event.record.assert_called_once_with(stream)
        assert ledger.get_event("before_combine") is event

    def test_wait_before_record_returns_false(self):
        ledger = get_event_ledger()
        assert ledger.wait("never_recorded") is False

    def test_wait_after_record_returns_true(self):
        ledger = get_event_ledger()
        ledger.record("before_combine")
        assert ledger.wait("before_combine") is True

    def test_clear_single_and_all(self):
        ledger = get_event_ledger()
        ledger.record("a")
        ledger.record("b")
        ledger.clear("a")
        assert ledger.get_event("a") is None
        assert ledger.get_event("b") is not None
        ledger.clear()
        assert ledger.get_event("b") is None

    def test_graph_capture_disables_record(self):
        ledger = get_event_ledger()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(torch.npu, "is_current_stream_capturing", MagicMock(return_value=True), raising=False)
            assert ledger.record("captured_slot") is None
        # Slot must not have been populated during capture.
        assert ledger.get_event("captured_slot") is None

    def test_graph_capture_disables_wait(self):
        ledger = get_event_ledger()
        ledger.record("eager_slot")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(torch.npu, "is_current_stream_capturing", MagicMock(return_value=True), raising=False)
            assert ledger.wait("eager_slot") is False

    def test_missing_capture_probe_treated_as_eager(self):
        from vllm_ascend.overlap import events as events_mod

        ledger = get_event_ledger()
        with pytest.MonkeyPatch.context() as mp:
            mp.delattr(torch.npu, "is_current_stream_capturing", raising=False)
            assert events_mod._is_graph_capturing() is False
            event = ledger.record("no_probe_slot")
        assert event is not None

    def test_ledger_singleton(self):
        assert get_event_ledger() is get_event_ledger()

    def test_event_creation_disables_timing(self):
        ledger = get_event_ledger()
        with pytest.MonkeyPatch.context() as mp:
            event_factory = MagicMock(return_value=MagicMock())
            mp.setattr(torch.npu, "Event", event_factory)
            ledger.record("timing_disabled")
        event_factory.assert_called_once_with(enable_timing=False)

    def test_ledger_reset_preserves_singleton(self):
        ledger = get_event_ledger()
        ledger.record("slot")
        reset_event_ledger()
        assert get_event_ledger() is ledger
        assert ledger.get_event("slot") is None


class TestRegistryConcurrency(PytestBase):
    def test_concurrent_first_access_yields_one_stream(self):
        import threading

        registry = StreamRegistry()
        results: list = []
        barrier = threading.Barrier(8)

        def access():
            barrier.wait()
            results.append(registry.get_stream("shared_experts"))

        threads = [threading.Thread(target=access) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 8
        for stream in results[1:]:
            assert results[0] is stream
