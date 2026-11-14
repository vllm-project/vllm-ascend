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
"""Unit tests for the recovery-unit graph used by frontier checkpoints."""

from vllm_ascend.hybrid_checkpoint.units import CheckpointGraph, RecoveryUnit

BLOCK = 16


class RecordingObserver:
    def __init__(self) -> None:
        self.added: list[bytes] = []
        self.removed: list[bytes] = []
        self.recached: list[bytes] = []
        self.linked: list[bytes] = []
        self.used: list[bytes] = []

    def unit_added(self, unit: RecoveryUnit) -> None:
        self.added.append(unit.key)

    def unit_recached(self, unit: RecoveryUnit) -> None:
        self.recached.append(unit.key)

    def unit_removed(self, unit: RecoveryUnit) -> None:
        self.removed.append(unit.key)

    def unit_used(self, unit: RecoveryUnit) -> None:
        self.used.append(unit.key)

    def unit_linked(self, unit: RecoveryUnit) -> None:
        self.linked.append(unit.key)


def test_insert_marks_block_boundary_as_non_frontier():
    graph = CheckpointGraph(BLOCK)
    graph.inserted(b"b0", BLOCK)
    graph.inserted(b"b1", BLOCK + 5)

    assert graph.units[b"b0"].frontier is False
    assert graph.units[b"b1"].frontier is True


def test_insert_is_idempotent_and_reports_recache():
    graph = CheckpointGraph(BLOCK)
    observer = RecordingObserver()
    graph.observer = observer

    graph.inserted(b"b0", BLOCK)
    graph.inserted(b"b0", BLOCK)

    assert len(graph.units) == 1
    assert observer.added == [b"b0"]
    assert observer.recached == [b"b0"]


def test_link_sets_parent_and_child_once():
    graph = CheckpointGraph(BLOCK)
    graph.inserted(b"p", BLOCK)
    graph.inserted(b"c", 2 * BLOCK)
    graph.admitted("req", b"p")

    graph.link("req", b"p")
    graph.link("req", b"c")
    graph.link("req", b"c")

    assert graph.units[b"c"].parent == b"p"
    assert graph.units[b"p"].children == {b"c"}


def test_admitted_ignores_resume_that_is_not_a_unit():
    graph = CheckpointGraph(BLOCK)
    graph.admitted("req", b"missing")

    assert graph.last["req"] is None
    assert graph.resumed["req"] is None


def test_removed_reparents_children():
    graph = CheckpointGraph(BLOCK)
    for key in (b"p", b"m", b"c"):
        graph.inserted(key, BLOCK)
    graph.admitted("req", b"p")
    graph.link("req", b"m")
    graph.link("req", b"c")
    graph.leaves.add(b"m")

    graph.removed(b"m")

    assert b"m" not in graph.units
    assert graph.units[b"c"].parent == b"p"
    assert graph.units[b"p"].children == {b"c"}
    assert b"m" not in graph.leaves
    # The request's last unit was the child, so it stays where it was.
    assert graph.last["req"] == b"c"
    assert graph.resumed["req"] == b"p"


def test_removed_rewinds_a_request_whose_last_unit_is_gone():
    graph = CheckpointGraph(BLOCK)
    for key in (b"p", b"m"):
        graph.inserted(key, BLOCK)
    graph.admitted("req", b"p")
    graph.link("req", b"m")

    graph.removed(b"m")

    assert graph.last["req"] == b"p"


def test_removed_on_unknown_key_is_a_no_op():
    graph = CheckpointGraph(BLOCK)
    graph.removed(b"absent")

    assert graph.units == {}


def test_touched_counts_hits_and_branch_point():
    graph = CheckpointGraph(BLOCK)
    graph.inserted(b"u", 3 * BLOCK)

    graph.touched(b"u")
    assert graph.units[b"u"].is_branch_point is False
    graph.touched(b"u")
    assert graph.units[b"u"].is_branch_point is True


def test_clear_drops_every_index():
    graph = CheckpointGraph(BLOCK)
    graph.inserted(b"u", BLOCK)
    graph.admitted("req", b"u")
    graph.leaves.add(b"u")

    graph.clear()

    assert graph.units == {}
    assert graph.last == {}
    assert graph.leaves == set()
    assert graph.resumed == {}
