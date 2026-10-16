# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.attention.dsa_offload_state import (
    DsaLayerWorkspace,
    DsaResidentState,
)


def _workspace(layer_id=0):
    return DsaLayerWorkspace(
        layer_id=layer_id,
        resident_kv_cache=torch.empty((3, 128, 1, 4)),
        resident_k_rope=torch.empty((3, 128, 1, 2)),
        selection_kv_cache=torch.empty((2, 2048, 1, 4)),
        selection_k_rope=torch.empty((2, 2048, 1, 2)),
        selection_block_table=torch.full((2, 16), -1, dtype=torch.int32),
    )


def test_resident_state_waits_for_previous_step_event_once():
    class Event:
        def __init__(self):
            self.wait_count = 0

        def wait(self):
            self.wait_count += 1

    event = Event()
    state = DsaResidentState([_workspace()])
    state.record_final_install_event(event)

    state.wait_previous_install()
    state.wait_previous_install()

    assert event.wait_count == 1


def test_resident_state_waits_for_every_recorded_step_event():
    class Event:
        def __init__(self):
            self.wait_count = 0

        def wait(self):
            self.wait_count += 1

    events = (Event(), Event())
    state = DsaResidentState([_workspace()])
    for event in events:
        state.record_final_install_event(event)

    state.wait_previous_install()

    assert [event.wait_count for event in events] == [1, 1]
