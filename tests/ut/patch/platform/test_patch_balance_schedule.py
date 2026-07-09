# SPDX-License-Identifier: Apache-2.0
"""Upstream-drift guards for the balance-scheduling platform patch.

These tests watch the upstream vLLM surfaces that ``patch_balance_schedule.py``
depends on. If upstream changes any of them in a way that would silently break
the patch (or stop it from taking effect), CI turns red here so we notice and
sync. They are NOT behavior/equivalence tests -- those need a running DP+MoE
engine on NPU and live under e2e/nightly.

What is guarded here (everything reachable from CPU UT):

* the ``schedule`` override signature stays callable by BOTH vllm versions CI
  runs (v0.23.0 calls ``schedule()``; 1f486d96 calls ``schedule(throttle_prefills)``)
  and carries every parameter the installed ``schedule`` exposes -- NOT an
  exact-match to the installed signature, which differs per lane;
* the ``BalanceScheduler.__init__`` signature stays drop-in compatible with
  upstream's ``Scheduler.__init__`` (upstream constructs ``Scheduler(...)``
  with kwargs, which after the swap constructs our subclass);
* upstream ``run_engine_core`` still instantiates ``DPEngineCoreProc`` by
  module-global name -- the whole reason we can swap the module-level symbol
  instead of copying ``run_engine_core``;
* the module-level class swaps actually took effect;
* the upstream Scheduler/DPEngineCoreProc methods the patch calls/super-calls
  still exist;
* the 3 balance deltas remain present in ``schedule()`` (intent lock);
* the copied ``schedule()`` body stays a verbatim copy of the ``schedule()``
  at vllm-ascend's pinned vLLM release tag (read from
  ``.github/vllm-release-tag.commit`` -- the same file CI uses), modulo exactly
  those 3 deltas. Reading the tag from the pin file means a pin advance
  auto-flips this guard to the new tag until the copy is re-synced.

What is NOT guarded here (structurally unreachable without a real engine):

* instance-attribute renames (``self.running``, ``self.dp_group``,
  ``self.kv_cache_manager`` ...) -- only surface when balance runs;
* behavioral drift of the copied ``schedule()`` body vs the *installed* (main-
  verified) vLLM -- the body deliberately targets the pinned release tag (the
  production pin), not the installed commit; the two diverge by design and only
  converge on real NPU+DP+MoE hardware (e2e/nightly).
"""

import ast
import inspect
import subprocess
import textwrap
from pathlib import Path

import pytest

# Capture the upstream originals BEFORE importing the patch: importing the patch
# mutates the module-level ``Scheduler`` / ``DPEngineCoreProc`` symbols, so grab
# the pristine classes/file paths first.
import vllm.v1.core.sched.scheduler as _upstream_sched_mod
import vllm.v1.engine.core as _upstream_engine_mod
from vllm.v1.core.sched.scheduler import Scheduler as _UpstreamScheduler
from vllm.v1.engine.core import DPEngineCoreProc as _UpstreamDPEngineCoreProc
from vllm.v1.engine.core import EngineCoreProc as _UpstreamEngineCoreProc

_UPSTREAM_SCHED_FILE = _upstream_sched_mod.__file__

# NOTE: vllm-ascend applies ALL platform patches at ``vllm_ascend`` import
# time (the platform-patch registry), which runs BEFORE this test module's
# body. So by the time we can execute anything, ``EngineCoreProc.run_engine_core``
# is ALREADY the patched ``_balance_run_engine_core`` wrapper -- we cannot
# recover the upstream original from the live object. Instead we read the
# original the patch itself stashed: ``_OriginalRunEngineCore`` is bound to
# ``EngineCoreProc.run_engine_core`` at the patch module's FIRST import, i.e.
# before the overwrite line runs (modules are imported once), so it is the
# genuine upstream original regardless of import ordering.

# Importing this module applies the production monkeypatches:
#   vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler            (eager)
#   EngineCoreProc.run_engine_core = _balance_run_engine_core            (eager)
#   vllm.v1.engine.core.DPEngineCoreProc = BalanceDPEngineCoreProc       (DEFERRED:
#       swapped inside _balance_run_engine_core only when balance is enabled)
from vllm_ascend.patch.platform.patch_balance_schedule import (  # noqa: E402
    BalanceScheduler,
    _balance_run_engine_core,
    _OriginalRunEngineCore,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _schedule_body_ast(source: str) -> str:
    """Canonical AST dump of a ``schedule`` method body with the 3 balance
    deltas stripped, so the remainder can be compared verbatim against the
    pinned release tag's ``schedule()``. AST-based on purpose: it is blind to
    comments and whitespace, so the only differences that surface are real code
    drift (not the escape-quoting of a comment or reformatting).

    The 3 deltas removed:
      * delta 1 -- the disabled-path early return (``if not
        self._balance_enabled: ... super().schedule(...)``);
      * delta 2 -- the ``balance_flag`` gate (``max(t.item() for t in
        self.balance_queue) == self.max_num_running_reqs``);
      * delta 3 -- the ``request_queue is None`` check, which exists in our
        copy as ``if request_queue is None: break`` and in upstream as
        ``assert request_queue is not None``. Both are stripped so the two
        bodies align.
    """
    tree = ast.parse(textwrap.dedent(source))
    func = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "schedule"),
        None,
    )
    assert func is not None, "no schedule() in source"

    filtered = []
    for stmt in func.body:
        dump = ast.dump(stmt)
        # delta 1: disabled-path early return. ``_balance_enabled`` is touched
        # nowhere else, so its presence alone identifies this delta.
        if "_balance_enabled" in dump:
            continue
        # delta 2: the balance_flag admission gate.
        if "balance_queue" in dump and "max_num_running_reqs" in dump:
            continue
        # delta 3: request_queue None-check (ours: if-is-None-break;
        # upstream: assert is-not-None). "Is" is a substring of "IsNot".
        if "request_queue" in dump and "None" in dump and "Is" in dump:
            continue
        filtered.append(stmt)
    return ast.dump(ast.Module(body=filtered, type_ignores=[]))


def _vllm_ascend_repo_root() -> Path | None:
    """Walk up from this test file to find the vllm-ascend repo root -- the dir
    holding ``.github/vllm-release-tag.commit``. Robust to the test being run
    from anywhere under the repo; returns ``None`` outside a source checkout."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".github" / "vllm-release-tag.commit").is_file():
            return parent
    return None


def _pinned_release_tag() -> str | None:
    """The vLLM release tag vllm-ascend pins to, read from
    ``.github/vllm-release-tag.commit`` -- the SAME file CI reads (via
    ``tr -d '[:space:]'``) to pick the tag. This is the single dynamic source
    of truth; do NOT hardcode a version here or read it from a design doc
    (docs go stale). Returns ``None`` when the pin file is absent."""
    root = _vllm_ascend_repo_root()
    if root is None:
        return None
    return (root / ".github" / "vllm-release-tag.commit").read_text(encoding="utf-8").strip() or None


def _pinned_release_schedule_source() -> tuple[str, str] | None:
    """Return ``(tag, source)`` of the pinned release tag's
    ``Scheduler.schedule()``, or ``None`` if anything is unreachable: no pin
    file, vllm not a git checkout, the tag absent from the repo, git not on
    PATH. Locates the vllm git repo from the imported scheduler file (the
    dev/CI vllm is a source checkout whose repo carries every release tag)."""
    tag = _pinned_release_tag()
    if not tag:
        return None
    sched_file = Path(_UPSTREAM_SCHED_FILE).resolve()
    # <repo>/vllm/v1/core/sched/scheduler.py -> parents[4] is the repo root.
    if len(sched_file.parents) < 5:
        return None
    repo = sched_file.parents[4]
    try:
        rel = sched_file.relative_to(repo).as_posix()
        proc = subprocess.run(
            ["git", "-C", str(repo), "show", f"{tag}:{rel}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None
    return (tag, proc.stdout)


# ---------------------------------------------------------------------------
# 1. schedule() signature is callable by BOTH engine versions (dual-version CI)
# ---------------------------------------------------------------------------


def test_schedule_signature_covers_both_engine_versions():
    """vllm-ascend CI runs against TWO vllm versions at once: the release tag
    v0.23.0 (whose engine calls ``schedule()`` with no args) and the
    main-verified commit 1f486d96 (whose engine calls
    ``schedule(throttle_prefills)``). A single override signature must be
    callable by BOTH engines, so it carries ``throttle_prefills`` with a
    default -- a deliberate superset of v0.23.0's ``schedule(self)``.

    Asserting exact equality with the *installed* signature would be wrong:
    on the v0.23.0 lane the installed signature is ``schedule(self)`` while
    ours is ``schedule(self, throttle_prefills=False)``, so an equality check
    can only ever pass on ONE of the two lanes. Instead we assert the two
    things that actually matter on both lanes:

    * both engines' call shapes bind cleanly to our signature (callable); and
    * our signature carries every parameter the installed ``schedule`` exposes
      (so an upstream parameter addition is caught here regardless of lane)."""
    sig = inspect.signature(BalanceScheduler.schedule)
    # The engine invokes schedule() on an instance, so ``self`` is implicitly
    # bound; strip it before simulating the engine's call shapes, otherwise
    # sig.bind() complains about the missing ``self`` argument.
    sig = sig.replace(parameters=[p for p in sig.parameters.values() if p.name != "self"])
    # v0.23.0 engine call shape, then 1f486d96 engine call shape.
    sig.bind()
    sig.bind(throttle_prefills=True)

    up = {k for k in inspect.signature(_UpstreamScheduler.schedule).parameters if k != "self"}
    ours = {k for k in sig.parameters if k != "self"}
    assert up <= ours, (
        f"BalanceScheduler.schedule is missing parameters the installed "
        f"vLLM Scheduler.schedule exposes ({up - ours}); the engine's call "
        f"would raise TypeError."
    )


# ---------------------------------------------------------------------------
# 1b. the 3 balance deltas remain present in schedule() (intent lock)
# ---------------------------------------------------------------------------


def test_balance_deltas_present_in_schedule():
    """The whole point of copying schedule() is to inject the balance logic.
    If a future re-sync against the pinned tag drops any of the 3 deltas,
    balance silently stops working -- this locks their presence in the source."""
    src = inspect.getsource(BalanceScheduler.schedule)

    # delta 1: disabled-path early return delegates to super().schedule().
    # Whether throttle_prefills is forwarded is decided by signature
    # introspection (_SUPER_SCHEDULE_HAS_THROTTLE), NOT a version string, so
    # the disabled path works on BOTH the v0.23.0 and 1f486d96 CI lanes.
    assert "if not self._balance_enabled:" in src
    assert "_SUPER_SCHEDULE_HAS_THROTTLE" in src
    assert "super().schedule(throttle_prefills)" in src
    assert "super().schedule()" in src

    # delta 2: the balance_flag admission gate (leader-at-cap => global freeze).
    assert "max(t.item() for t in self.balance_queue)" in src
    assert "self.max_num_running_reqs" in src

    # delta 3: `if request_queue is None: break` replaces upstream's assert.
    assert "if request_queue is None:" in src


# ---------------------------------------------------------------------------
# 1c. copied schedule() body stays verbatim with the pinned release tag
# ---------------------------------------------------------------------------


def test_schedule_body_matches_pinned_release_tag():
    """The copied ``schedule()`` body must stay a verbatim copy of the
    ``schedule()`` at vllm-ascend's pinned vLLM release tag, modulo exactly the
    3 balance deltas.

    The tag is read dynamically from ``.github/vllm-release-tag.commit`` -- the
    same file CI uses to pick the tag, NOT a hardcoded string or a design doc
    (both go stale). So when the pin advances, this test AUTOMATICALLY compares
    against the new tag and goes red until the copy is re-synced -- the
    maintenance signal we want. Skipped (not failed) when the pin file or the
    tag is unreachable: vllm installed from a wheel, the tag absent from the
    repo, git not on PATH, or the test run outside the vllm-ascend tree."""
    ref = _pinned_release_schedule_source()
    if ref is None:
        pytest.skip(
            "pinned vLLM release tag or its schedule() not retrievable "
            "(no .github/vllm-release-tag.commit, vllm not a git checkout, "
            "or tag absent)"
        )
    assert ref is not None
    tag, pinned_src = ref

    ours = _schedule_body_ast(inspect.getsource(BalanceScheduler.schedule))
    theirs = _schedule_body_ast(pinned_src)
    assert ours == theirs, (
        f"BalanceScheduler.schedule body drifted from the pinned release tag "
        f"({tag}) beyond the 3 balance deltas. Re-sync the copy against "
        f"{tag} and re-apply only: (1) disabled-path early return, "
        f"(2) balance_flag gate, (3) if request_queue is None: break."
    )


# ---------------------------------------------------------------------------
# 2. BalanceScheduler.__init__ stays drop-in compatible with upstream's
# ---------------------------------------------------------------------------


def test_balance_scheduler_init_signature_matches_upstream():
    """Upstream constructs ``Scheduler(...)`` by keyword (engine/core.py), which
    after the swap constructs ``BalanceScheduler(...)`` with the same kwargs.
    Our ``__init__`` parameter set must therefore track upstream's exactly,
    including defaults -- a divergence (added/removed/renamed param, or a
    shifted default) breaks construction at engine startup."""
    up = {k: v for k, v in inspect.signature(_UpstreamScheduler.__init__).parameters.items() if k != "self"}
    ours = {k: v for k, v in inspect.signature(BalanceScheduler.__init__).parameters.items() if k != "self"}
    assert list(up.keys()) == list(ours.keys()), (
        f"BalanceScheduler.__init__ params diverged from upstream.\n"
        f"  upstream: {list(up.keys())}\n  ours    : {list(ours.keys())}\n"
    )
    for name in up:
        assert up[name].default == ours[name].default, (
            f"default for __init__ param '{name}' diverged: upstream={up[name].default!r} ours={ours[name].default!r}"
        )


# ---------------------------------------------------------------------------
# 3. upstream run_engine_core still instantiates DPEngineCoreProc by name
# ---------------------------------------------------------------------------


def test_upstream_run_engine_core_instantiates_dp_proc_by_name():
    """The refactor deletes the copied ``run_engine_core`` and instead swaps the
    module-level ``DPEngineCoreProc`` symbol. That only works while upstream's
    ``run_engine_core`` resolves ``DPEngineCoreProc`` by module-global name at
    call time. If upstream switches to ``self.__class__(...)`` or a factory, the
    swap silently stops instantiating our subclass (balance off, no error)."""
    # Read the upstream ORIGINAL run_engine_core (the live
    # EngineCoreProc.run_engine_core is already our _balance_run_engine_core
    # wrapper by test time -- see _OriginalRunEngineCore import note above).
    src = inspect.getsource(_OriginalRunEngineCore)
    assert "DPEngineCoreProc(" in src, (
        "upstream run_engine_core no longer instantiates DPEngineCoreProc by "
        "module-global name; the _engine_core_mod.DPEngineCoreProc swap in "
        "patch_balance_schedule.py would silently break."
    )


# ---------------------------------------------------------------------------
# 4. the module-level class swaps actually took effect
# ---------------------------------------------------------------------------


def test_module_level_swaps_take_effect():
    """The patch rebinds ``Scheduler`` eagerly and installs the
    ``run_engine_core`` wrapper eagerly, but DEFERS the ``DPEngineCoreProc``
    swap to wrapper-call-time (conditional on balance being enabled), so that
    balance does not touch configs that don't use it (e.g. PD-disaggregated
    recompute). At import time the engine-core class must therefore still be
    the pristine upstream one, and the wrapper must be installed.
    (``Scheduler`` propagating into ``vllm.v1.engine.core.Scheduler``
    additionally depends on the platform patch loading before engine.core is
    imported -- that ordering is enforced by the platform patch system and is
    integration-level, not asserted here.)
    """
    assert _upstream_sched_mod.Scheduler is BalanceScheduler, (
        "patch did not rebind vllm.v1.core.sched.scheduler.Scheduler"
    )
    # DPEngineCoreProc is NOT swapped at import -- it stays pristine and is
    # swapped inside _balance_run_engine_core only when balance is enabled.
    assert _upstream_engine_mod.DPEngineCoreProc is _UpstreamDPEngineCoreProc, (
        "patch swapped vllm.v1.engine.core.DPEngineCoreProc at import time; "
        "the swap must be deferred to run_engine_core entry (conditional)."
    )
    assert _UpstreamEngineCoreProc.run_engine_core is _balance_run_engine_core, (  # type: ignore[comparison-overlap]
        "patch did not install the _balance_run_engine_core wrapper on EngineCoreProc.run_engine_core"
    )


# ---------------------------------------------------------------------------
# 5. upstream method seams the patch super-calls / the copied body calls
# ---------------------------------------------------------------------------

# Scheduler-level methods the copied schedule() body invokes on ``self``, plus
# the ones we super()-call. A rename/removal upstream breaks balance at runtime.
_SCHEDULER_METHOD_SEAMS = [
    "schedule",  # super().schedule() on the disabled path
    "_preempt_request",
    "_try_schedule_encoder_inputs",
    "_mamba_block_aligned_split",
    "_select_waiting_queue_for_scheduling",
    "_is_blocked_waiting_status",
    "_try_promote_blocked_waiting_request",
    "_make_cached_request_data",
    "_update_after_schedule",
    "_build_kv_connector_meta",
    "_inflight_prefill_reserved_blocks",
]


def test_upstream_scheduler_seams_still_exist():
    """Guard the upstream method names the patch depends on. The copied
    ``schedule()`` body calls a fixed set of Scheduler internals by name; if
    upstream renames/removes any, the body breaks when balance runs."""
    missing = [n for n in _SCHEDULER_METHOD_SEAMS if not hasattr(_UpstreamScheduler, n)]
    assert not missing, "upstream Scheduler lost methods the patch depends on: " + ", ".join(missing)
    assert hasattr(_UpstreamDPEngineCoreProc, "run_busy_loop"), (
        "upstream DPEngineCoreProc lost run_busy_loop (BalanceDPEngineCoreProc inherits it)"
    )
    assert hasattr(_UpstreamDPEngineCoreProc, "_has_global_unfinished_reqs"), (
        "upstream DPEngineCoreProc lost _has_global_unfinished_reqs; "
        "BalanceDPEngineCoreProc._has_global_unfinished_reqs super-calls it to "
        "hook the per-step balance_gather immediately after the cross-rank "
        "all-reduce. It MUST be called every non-idle iteration by run_busy_loop "
        "(incl. dummy-batch) or the all_gather deadlocks."
    )
