import os
from types import SimpleNamespace

import pytest

from vllm_ascend import cpu_binding


def test_explicit_rank_cpu_binding_sets_current_and_existing_threads(monkeypatch):
    applied = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1, 4, 5}, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", lambda tid, cpus: applied.append((tid, cpus)), raising=False)
    monkeypatch.setattr(
        cpu_binding.psutil,
        "Process",
        lambda: SimpleNamespace(threads=lambda: [SimpleNamespace(id=os.getpid()), SimpleNamespace(id=os.getpid() + 1)]),
    )

    cpu_binding.bind_rank_cpus(1, ["0-1", "4-5"])

    assert applied == [(0, {4, 5}), (os.getpid() + 1, {4, 5})]


@pytest.mark.parametrize(
    ("rank", "bindings", "message"),
    [
        (1, ["0-2", "2-4"], "overlap"),
        (2, ["0-1", "2-3"], "no CPU list"),
        (0, ["2-1"], "invalid"),
        (0, ["0,5"], "outside"),
    ],
)
def test_explicit_rank_cpu_binding_rejects_bad_sets(monkeypatch, rank, bindings, message):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {0, 1, 2, 3}, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", lambda *_: pytest.fail("invalid binding applied"), raising=False)

    with pytest.raises(ValueError, match=message):
        cpu_binding.bind_rank_cpus(rank, bindings)
