# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.nova_kv import NovaKVRuntime


def batch(ids, computed, scheduled, *, dummy=False):
    return SimpleNamespace(
        req_ids=ids,
        num_reqs=len(ids),
        num_computed_tokens_np=np.array(computed),
        num_scheduled_tokens=np.array(scheduled),
        is_dummy=dummy,
    )


def test_reorder_and_padding_keep_persistent_window_slots():
    runtime = NovaKVRuntime(4, torch.device("cpu"))
    pointer = runtime.request_slots.data_ptr()
    first = runtime.prepare(batch(["a", "b"], [0, 0], [3, 4]), 4)
    assert first["nova_request_slots_cpu"] == [0, 1, -1, -1]
    runtime.finish_execution(failed=False)
    next_step = runtime.prepare(batch(["b", "a"], [4, 3], [1, 1]), 2)
    assert next_step["nova_request_slots_cpu"] == [1, 0]
    assert next_step["nova_request_slots"].tolist() == [1, 0]
    assert runtime.request_slots.tolist() == [1, 0, -1, -1]
    assert runtime.request_slots.data_ptr() == pointer
    assert runtime.state.missing(["a", "b"], [4, 5]) == set()
    # Host snapshots are not views of the next step's input buffer.
    assert first["nova_request_slots_cpu"] == [0, 1, -1, -1]


@pytest.mark.parametrize("for_capture,dummy", [(True, False), (False, True)])
def test_dummy_does_not_claim_or_advance_windows(for_capture, dummy):
    runtime = NovaKVRuntime(3, torch.device("cpu"))
    runtime.prepare(batch(["a"], [0], [7]), 2)
    runtime.finish_execution(failed=False)
    metadata = runtime.prepare(batch(["dummy"], [31], [1], dummy=dummy), 3, for_capture=for_capture)
    assert metadata["nova_request_slots_cpu"] == [-1, -1, -1]
    assert metadata["nova_request_slots"].tolist() == [-1, -1, -1]
    assert runtime.state.missing(["a"], [7]) == set()
    assert runtime.prepare(batch(["a"], [7], [1]), 1)["nova_request_slots_cpu"] == [0]


def test_preempted_slot_is_reused_only_after_recompute():
    runtime = NovaKVRuntime(2, torch.device("cpu"))
    runtime.prepare(batch(["a", "b"], [0, 0], [4, 5]), 2)
    runtime.finish_execution(failed=False)
    runtime.state.release(["a"])
    with pytest.raises(RuntimeError, match="recomputed from zero"):
        runtime.prepare(batch(["a"], [4], [1]), 1)
    assert runtime.prepare(batch(["c"], [0], [2]), 1)["nova_request_slots_cpu"] == [0]
    runtime.finish_execution(failed=False)
    assert runtime.prepare(batch(["b"], [5], [1]), 1)["nova_request_slots_cpu"] == [1]


def test_failed_forward_invalidates_only_scheduled_windows():
    runtime = NovaKVRuntime(2, torch.device("cpu"))
    runtime.prepare(batch(["a", "b"], [0, 0], [4, 5]), 2)
    runtime.finish_execution(failed=False)
    runtime.prepare(batch(["b"], [5], [1]), 1)
    runtime.finish_execution(failed=True)
    assert runtime.state.missing(["a", "b"], [4, 5]) == {"b"}
    assert runtime.prepare(batch(["b"], [0], [5]), 1)["nova_request_slots_cpu"] == [1]


def test_window_capacity_checked_before_metadata_write():
    runtime = NovaKVRuntime(2, torch.device("cpu"))
    with pytest.raises(ValueError, match="capacity"):
        runtime.prepare(batch(["a"], [0], [1]), 3)
    assert runtime.request_slots.tolist() == [-1, -1]


def test_failed_batch_preparation_invalidates_partially_advanced_windows():
    runtime = NovaKVRuntime(3, torch.device("cpu"))
    runtime.prepare(batch(["a", "b", "c"], [0, 0, 0], [4, 5, 6]), 3)
    runtime.finish_execution(failed=False)
    with pytest.raises(RuntimeError, match="recomputed from zero"):
        runtime.prepare(batch(["a", "b"], [4, 99], [1, 1]), 2)
    runtime.finish_execution(failed=True)
    assert runtime.state.missing(["a", "b", "c"], [4, 5, 6]) == {"a", "b"}
    assert runtime.prepare(batch(["a", "b"], [0, 0], [4, 5]), 2)["nova_request_slots_cpu"] == [0, 1]
