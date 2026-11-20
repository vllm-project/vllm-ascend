# mypy: ignore-errors
"""CP rank resolution for the runtime_guard ``dp*_tp*_pp*_cp*`` dump tag.

Regression UTs for issue #22028: the PCP rank lives on the PCP manager
(``runner.pcp_manager.pcp_rank``), the model runner carries no ``pcp_rank``
attribute. Reading it from the runner resolved ``0`` on every PCP rank, so all
PCP ranks of a stage wrote into the same ``..._cp0`` dump directory.

    cp = pcp_rank * dcp_size + dcp_rank
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_ascend.observability.runtime_guard import rank_gate
from vllm_ascend.observability.runtime_guard.rank_gate import (
    dump_rank_tag,
    runner_cp_rank,
)


@pytest.fixture(autouse=True)
def _stub_parallel_state(monkeypatch):
    """Pin the process-group lookups to a single TP0 / last-PP rank."""
    monkeypatch.setattr(
        rank_gate,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1),
    )
    monkeypatch.setattr(
        rank_gate,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=0, is_last_rank=True),
    )


def _runner(**kwargs) -> SimpleNamespace:
    """Runner stand-in without any CP attribute unless asked for."""
    runner = SimpleNamespace(dp_rank=0, tp_rank=0)
    runner.__dict__.update(kwargs)
    return runner


def test_cp_rank_reads_pcp_rank_from_pcp_manager():
    """The PCP manager owns the PCP rank; the runner does not."""
    runner = _runner(
        dcp_size=2,
        dcp_rank=1,
        pcp_manager=SimpleNamespace(pcp_rank=2, dcp_world_size=2, dcp_rank=1),
    )
    assert runner_cp_rank(runner) == 2 * 2 + 1


def test_cp_rank_falls_back_to_runner_attribute():
    """A runner-level ``pcp_rank`` still wins when there is no manager."""
    runner = _runner(pcp_rank=3, dcp_size=1, dcp_rank=0)
    assert runner_cp_rank(runner) == 3


def test_cp_rank_reads_dcp_fields_from_pcp_manager_when_runner_lacks_them():
    runner = _runner(pcp_manager=SimpleNamespace(pcp_rank=1, dcp_world_size=4, dcp_rank=2))
    assert runner_cp_rank(runner) == 1 * 4 + 2


def test_cp_rank_without_cp_is_zero():
    assert runner_cp_rank(_runner()) == 0


def test_cp_rank_accepts_string_rank_values():
    """Rank attributes parsed from configs may reach the runner as strings."""
    runner = _runner(
        dcp_size="2",
        dcp_rank="1",
        pcp_manager=SimpleNamespace(pcp_rank="2", dcp_world_size="2", dcp_rank="1"),
    )
    assert runner_cp_rank(runner) == 2 * 2 + 1


def test_cp_rank_ignores_unparsable_rank_values():
    runner = _runner(
        dcp_size="not-a-number",
        dcp_rank="not-a-number",
        pcp_manager=SimpleNamespace(pcp_rank="not-a-number"),
    )
    assert runner_cp_rank(runner) == 0


def test_cp_rank_with_disabled_pcp_manager_is_zero():
    """PCP disabled -> ``runner.pcp_manager`` is ``None``."""
    runner = _runner(pcp_manager=None, dcp_size=1, dcp_rank=0)
    assert runner_cp_rank(runner) == 0


def test_dump_rank_tag_distinguishes_pcp_ranks():
    """Two PCP ranks of the same stage must not share a dump directory."""
    tags = []
    for pcp_rank in range(2):
        runner = _runner(
            dcp_size=1,
            dcp_rank=0,
            pcp_manager=SimpleNamespace(pcp_rank=pcp_rank, dcp_world_size=1, dcp_rank=0),
        )
        tags.append(dump_rank_tag(runner))
    assert tags == ["dp0_tp0_pp0_cp0", "dp0_tp0_pp0_cp1"]
    assert len(set(tags)) == 2


def test_dump_rank_tag_keeps_dcp_offset():
    """``cp`` keeps the ``pcp_rank * dcp_size + dcp_rank`` layout."""
    runner = _runner(
        dcp_size=2,
        dcp_rank=1,
        pcp_manager=SimpleNamespace(pcp_rank=1, dcp_world_size=2, dcp_rank=1),
    )
    assert dump_rank_tag(runner).endswith("_cp3")
