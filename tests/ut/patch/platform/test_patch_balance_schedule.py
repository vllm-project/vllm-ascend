# SPDX-License-Identifier: Apache-2.0
"""Upstream-drift guards for the balance-scheduling platform patch.

These tests watch the upstream vLLM surfaces that ``patch_balance_schedule.py``
depends on. If upstream changes any of them in a way that would silently break
the patch (or stop it from taking effect), CI turns red here so we notice and
sync. They are NOT behavior/equivalence tests -- those need a running DP+MoE
engine on NPU and live under e2e/nightly.

What is guarded here (everything reachable from CPU UT):

* the ``Scheduler.schedule`` signature we delegate to on the disabled path;
* the ``BalanceScheduler.__init__`` signature stays drop-in compatible with
  upstream's ``Scheduler.__init__`` (upstream constructs ``Scheduler(...)``
  with kwargs, which after the swap constructs our subclass);
* upstream ``run_engine_core`` still instantiates ``DPEngineCoreProc`` by
  module-global name -- the whole reason we can swap the module-level symbol
  instead of copying ``run_engine_core``;
* the module-level class swaps actually took effect;
* the upstream Scheduler/DPEngineCoreProc methods the patch calls/super-calls
  still exist.

What is NOT guarded here (structurally unreachable without a real engine):

* instance-attribute renames (``self.running``, ``self.dp_group``,
  ``self.kv_cache_manager`` ...) -- only surface when balance runs;
* behavioral drift of the copied ``schedule()`` body vs upstream -- impossible
  while a frozen copy exists; resolved only by Phase 2B (delete the copy,
  override an upstream hook).
"""

import ast
import inspect

# Capture the upstream originals BEFORE importing the patch: importing the patch
# mutates the module-level ``Scheduler`` / ``DPEngineCoreProc`` symbols, so grab
# the pristine classes/file paths first.
import vllm.v1.core.sched.scheduler as _upstream_sched_mod
import vllm.v1.engine.core as _upstream_engine_mod
from vllm.v1.core.sched.scheduler import Scheduler as _UpstreamScheduler
from vllm.v1.engine.core import DPEngineCoreProc as _UpstreamDPEngineCoreProc
from vllm.v1.engine.core import EngineCoreProc as _UpstreamEngineCoreProc

_UPSTREAM_SCHED_FILE = _upstream_sched_mod.__file__

# Importing this module applies the production monkeypatches:
#   vllm.v1.core.sched.scheduler.Scheduler = BalanceScheduler
#   vllm.v1.engine.core.DPEngineCoreProc = BalanceDPEngineCoreProc
from vllm_ascend.patch.platform.patch_balance_schedule import (  # noqa: E402
    BalanceDPEngineCoreProc,
    BalanceScheduler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _method_source_lines(path: str, class_name: str, method_name: str) -> list[str]:
    """Return the raw source lines of ``class_name.method_name`` in ``path``."""
    with open(path, encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return src.splitlines()[item.lineno - 1 : item.end_lineno]
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


# ---------------------------------------------------------------------------
# 1. schedule() signature matches installed vLLM (so super().schedule() works)
# ---------------------------------------------------------------------------


def test_schedule_signature_matches_upstream():
    """The override's signature must match the installed vLLM's ``schedule`` so
    that ``super().schedule()`` is callable on the disabled path."""
    up = _method_source_lines(_UPSTREAM_SCHED_FILE, "Scheduler", "schedule")[0]
    patched = inspect.getsource(BalanceScheduler.schedule).splitlines()[0]
    assert up.strip() == patched.strip(), (
        f"schedule() signature drifted from installed vLLM.\n"
        f"  installed: {up.strip()}\n  patched   : {patched.strip()}\n"
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
    src = inspect.getsource(_UpstreamEngineCoreProc.run_engine_core)
    assert "DPEngineCoreProc(" in src, (
        "upstream run_engine_core no longer instantiates DPEngineCoreProc by "
        "module-global name; the _engine_core_mod.DPEngineCoreProc swap in "
        "patch_balance_schedule.py would silently break."
    )


# ---------------------------------------------------------------------------
# 4. the module-level class swaps actually took effect
# ---------------------------------------------------------------------------


def test_module_level_swaps_take_effect():
    """The patch must have rebound the two module-level symbols upstream
    resolves at call/construction time. (Note: ``Scheduler`` propagating into
    ``vllm.v1.engine.core.Scheduler`` additionally depends on the platform
    patch loading before engine.core is imported -- that ordering is enforced
    by the platform patch system and is integration-level, not asserted here.)
    """
    assert _upstream_sched_mod.Scheduler is BalanceScheduler, (
        "patch did not rebind vllm.v1.core.sched.scheduler.Scheduler"
    )
    assert _upstream_engine_mod.DPEngineCoreProc is BalanceDPEngineCoreProc, (
        "patch did not rebind vllm.v1.engine.core.DPEngineCoreProc"
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
]


def test_upstream_scheduler_seams_still_exist():
    """Guard the upstream method names the patch depends on. The copied
    ``schedule()`` body calls a fixed set of Scheduler internals by name; if
    upstream renames/removes any, the body breaks when balance runs."""
    missing = [n for n in _SCHEDULER_METHOD_SEAMS if not hasattr(_UpstreamScheduler, n)]
    assert not missing, "upstream Scheduler lost methods the patch depends on: " + ", ".join(missing)
    assert hasattr(_UpstreamDPEngineCoreProc, "run_busy_loop"), (
        "upstream DPEngineCoreProc lost run_busy_loop (BalanceDPEngineCoreProc delegates to it via super())"
    )
