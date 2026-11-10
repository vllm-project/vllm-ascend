# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native confidence-buffer provenance tests; all device operations use CPU fixtures."""

from contextlib import nullcontext
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
from vllm.v1.worker.gpu.spec_decode import adaptive_verification as native_av

from vllm_ascend.worker.v2.spec_decode.hardware_aware import physical_k_scope
from vllm_ascend.worker.v2.spec_decode.physical_k_profile import configure_physical_k_profiling


class DeviceArray:
    def __init__(self, values):
        self.values = np.array(values, dtype=np.float64)

    def __setitem__(self, key, values):
        self.values[key] = values

    def copy_(self, other):
        np.copyto(self.values, other.values)


class Event:
    def __init__(self):
        self.sync_calls = self.record_calls = 0

    def synchronize(self):
        self.sync_calls += 1

    def record(self):
        self.record_calls += 1


class Buffer:
    def __init__(self):
        self.gpu = DeviceArray(np.ones((2, 7)))
        self.np = np.ones((2, 7))

    def copy_to_cpu(self):
        np.copyto(self.np, self.gpu.values)


def config():
    return NS(
        speculative_config=NS(num_speculative_tokens=7),
        additional_config={"dynamic_spec_config": {"method": "dspark", "physical_k": {"min_k": 3, "auto_tune": True}}},
    )


def record_native(manager, values, batch):
    # Execute the actual pinned native record method, including its global
    # confidence-table copy and stale-index flip, without a device context.
    with (
        patch.object(native_av.torch.cuda, "current_stream", return_value=None),
        patch.object(native_av, "stream", lambda *args: nullcontext()),
    ):
        native_av.AdaptiveVerificationManager.record_confidences(manager, values, batch)


def make_manager():
    manager = NS(
        num_speculative_steps=7,
        num_bonus_tokens=1,
        _max_total_logits=128,
        req_states=NS(
            max_num_reqs=2,
            req_id_to_index={"r": 0, "s": 1},
            num_computed_tokens_np=np.full(2, 100),
            prefill_len=NS(np=np.zeros(2)),
            device="cpu-fixture",
        ),
        _stale_idx=0,
        _stale_confidences=[Buffer(), Buffer()],
        _copy_events=[Event(), Event()],
        _copy_stream=NS(wait_stream=lambda stream: None),
        _pending_resets=[],
        _confidence_probs=DeviceArray(np.ones((2, 7))),
        cost_tables=(np.ones(3), np.ones(128)),
        batches_to_profile=lambda sizes: [],
        set_initial_cost_curves=lambda samples: None,
        get_num_tokens=lambda per_req, drafts: sum(per_req.values()),
    )
    manager.record_confidences = MethodType(record_native, manager)
    manager = configure_physical_k_profiling(manager, config())
    manager._physical_k_draft_costs = {k: {1: 1.0 if k == 3 else 10.0, 2: 1.0 if k == 3 else 10.0} for k in range(3, 8)}
    return manager


def record(manager, width, slots=(0,), known=True):
    batch = NS(num_reqs=len(slots), idx_mapping=np.array(slots), idx_mapping_np=np.array(slots))
    if known:
        batch._vllm_ascend_produced_confidence_k = width
    values = np.full((len(slots), 7), 0.99)
    values[:, :width] = 0.9
    manager.record_confidences(values, batch)


def get(manager, width, request="r"):
    manager.get_num_tokens({request: width + 1}, {request: list(range(width))})
    return manager._physical_k_recommendation


def full_then_narrow_then_probe():
    manager = make_manager()
    record(manager, 7)
    record(manager, 7)
    assert get(manager, 7) == (1, 3)
    record(manager, 3)
    assert get(manager, 3) is None
    record(manager, 7)
    return manager


def test_published_full_drafts_do_not_make_narrow_cpu_snapshot_full():
    manager = full_then_narrow_then_probe()
    assert get(manager, 7) is None


def test_following_narrow_draft_scores_newly_readable_full_probe_immediately():
    manager = full_then_narrow_then_probe()
    get(manager, 7)
    record(manager, 3)
    assert get(manager, 3) == (1, 3)


def test_initial_unrecorded_confidences_do_not_get_priced_as_full_width():
    assert get(make_manager(), 7) is None


def test_unknown_producer_width_remains_untrusted_after_native_copies():
    manager = make_manager()
    record(manager, 7, known=False)
    record(manager, 7, known=False)
    assert get(manager, 7) is None


def test_request_mapping_uses_the_readable_snapshot_slot():
    manager = make_manager()
    record(manager, 7, slots=(0, 1))
    record(manager, 7, slots=(0, 1))
    record(manager, 3, slots=(0,))
    record(manager, 7, slots=(0,))
    assert get(manager, 3, request="s") == (1, 3)


@pytest.mark.parametrize("initial,last,expected", [(7, 3, None), (3, 7, (1, 3))])
def test_full_table_copy_keeps_inactive_request_provenance_when_it_rejoins(initial, last, expected):
    manager = make_manager()
    record(manager, initial, slots=(0,))
    record(manager, initial, slots=(0,))
    record(manager, last, slots=(0,))
    # Native copy snapshots the entire global confidence table, including r,
    # even though only s participates in these two producer steps.
    record(manager, 7, slots=(1,))
    record(manager, 7, slots=(1,))
    assert get(manager, 3, request="r") == expected


def test_recycled_request_slot_does_not_inherit_old_full_width_labels():
    manager = make_manager()
    record(manager, 7)
    record(manager, 7)
    assert get(manager, 7) == (1, 3)
    manager.req_states.req_id_to_index["new"] = 0
    manager._pending_resets.append(0)
    record(manager, 7)
    assert get(manager, 7, request="new") is None
    record(manager, 7)
    assert get(manager, 7, request="new") == (1, 3)


def test_provenance_recording_adds_no_native_copy_event_synchronizations():
    manager = make_manager()
    original_tables = tuple(table.copy() for table in manager.cost_tables)
    for width in (7, 3, 7, 3):
        record(manager, width)
    assert sum(event.sync_calls for event in manager._copy_events) == 4
    assert sum(event.record_calls for event in manager._copy_events) == 4
    for actual, expected in zip(manager.cost_tables, original_tables):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("planned,counts,expected", [(3, None, 3), (7, None, 7), (None, [3, 4], 7)])
def test_propose_scope_reports_actual_resolved_confidence_width(planned, counts, expected):
    speculator = NS(
        num_speculative_steps=7,
        num_query_per_req=7,
        sample_from_anchor=True,
        vllm_config=config(),
        _vllm_ascend_physical_k_buffers_initialized=True,
    )
    batch = NS(num_draft_tokens_per_req=counts)
    if planned is not None:
        batch._vllm_ascend_physical_draft_k = planned
    with physical_k_scope(speculator, batch) as actual:
        assert actual == expected
    assert getattr(batch, "_vllm_ascend_produced_confidence_k", None) == expected
