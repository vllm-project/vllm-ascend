# SPDX-License-Identifier: Apache-2.0
"""Upstream-drift guards for the consolidated engine-core platform patch.

``patch_engine_core.py`` is the single entry-point patch for
``EngineCoreProc.run_engine_core``: it replaces the per-feature wrappers that
used to live in ``patch_balance_schedule``, ``patch_dyntra_lb_core`` and
``patch_profiling_chunk`` (each re-wrapping ``run_engine_core`` in import
order). What is guarded here (everything reachable from CPU UT):

* the consolidated patch is applied eagerly at import: the live
  ``EngineCoreProc.run_engine_core`` is ``_run_engine_core_patch_func``;
* ``_OriginalRunEngineCore`` stashes the pristine upstream ``run_engine_core``
  -- the genuine original the wrapper must delegate to, regardless of import
  ordering;
* ``_patch_dp_engine_core_proc`` selects ``DyntraLBDPEngineCoreProc`` when
  dyntra-lb is enabled (and does not consult balance), else
  ``BalanceDPEngineCoreProc`` when balance is enabled, else leaves the
  module-global ``DPEngineCoreProc`` untouched (the deferred-swap invariant
  the balance patch depends on);
* ``_run_engine_core_patch_func`` initializes the ascend config, re-applies
  the profiling patches only when profiling-based chunk sizing is enabled,
  calls ``patch_watchdog_for_engine_core()`` (installs the
  ``_process_engine_step`` / ``_process_input_queue`` / signal-callback
  patches and configures + starts the watchdog), and delegates to the
  upstream entry point with ``dp_rank`` / ``local_dp_rank`` passed through;
* ``_apply_patch`` is idempotent and installs the pp-mtp ``post_step``
  patch that previously ran from ``patch_pp_mtp``'s own ``_apply_patch``;
  the watchdog patches are DEFERRED to engine-core entry
  (``patch_watchdog_for_engine_core``) -- at import time the class methods
  must still be pristine upstream;
* importing ``patch_engine_core`` pulls in ``patch_profiling_chunk`` and
  ``patch_engine_watchdog`` (the engine-core child applies both patches
  through this import chain when unpickling the ``run_engine_core``
  wrapper).
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import vllm.v1.engine.core as _engine_core_mod
from vllm.v1.engine.core import DPEngineCoreProc as _UpstreamDPEngineCoreProc
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core import EngineCoreProc as _UpstreamEngineCoreProc
from vllm.v1.engine.utils import SignalCallback as _UpstreamSignalCallback

import vllm_ascend.patch.platform.patch_engine_core as _engine_core_patch
import vllm_ascend.patch.platform.patch_engine_watchdog as _watchdog_patch
import vllm_ascend.patch.platform.patch_profiling_chunk as _profiling_patch
from vllm_ascend.patch.platform.patch_balance_schedule import BalanceDPEngineCoreProc
from vllm_ascend.patch.platform.patch_dyntra_lb_core import DyntraLBDPEngineCoreProc


def _ascend_config(profiling_enabled: bool):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            profiling_chunk_config=SimpleNamespace(enabled=profiling_enabled),
        )
    )


def _dyntra_config(enabled: bool, enable_diagnostics: bool = True):
    return SimpleNamespace(enabled=enabled, enable_diagnostics=enable_diagnostics)


# ---------------------------------------------------------------------------
# 1. the consolidated patch took effect at import
# ---------------------------------------------------------------------------


def test_engine_core_patch_applied_at_import():
    assert _engine_core_patch._PATCHED is True
    assert _UpstreamEngineCoreProc.run_engine_core is _engine_core_patch._run_engine_core_patch_func
    # The watchdog patches are deferred to engine-core entry: at import time
    # the engine methods and the signal callback must still be pristine.
    assert _UpstreamEngineCoreProc._process_engine_step is _watchdog_patch._original_process_engine_step
    assert _UpstreamEngineCoreProc._process_input_queue is not _watchdog_patch._patched_process_input_queue
    assert _UpstreamSignalCallback.trigger is _watchdog_patch._original_signal_callback


def test_original_run_engine_core_stashes_pristine_upstream():
    """The wrapper must delegate to the genuine upstream ``run_engine_core``.

    ``_OriginalRunEngineCore`` is bound to ``EngineCoreProc.run_engine_core``
    at the patch module's first import -- before the staticmethod rebind line
    runs -- so it is the pristine upstream original regardless of import
    ordering. The live class attribute is already the patch wrapper by test
    time and cannot be used for this check."""
    assert _engine_core_patch._OriginalRunEngineCore is not _engine_core_patch._run_engine_core_patch_func
    src = inspect.getsource(_engine_core_patch._OriginalRunEngineCore)
    assert "DPEngineCoreProc(" in src, (
        "upstream run_engine_core no longer instantiates DPEngineCoreProc by "
        "module-global name; the _engine_core_mod.DPEngineCoreProc swap in "
        "patch_engine_core.py would silently break."
    )


def test_apply_patch_installs_pp_mtp_post_step_patch():
    """The pp-mtp post_step patch previously ran from patch_pp_mtp's own
    _apply_patch; the refactor moved the call into patch_engine_core."""
    assert getattr(EngineCore.post_step, "_vllm_ascend_pp_mtp_patched", False) is True


def test_apply_patch_is_idempotent(monkeypatch):
    post_step_patch = MagicMock()
    monkeypatch.setattr(_engine_core_patch, "pp_mtp_patch_post_step", post_step_patch)

    _engine_core_patch._apply_patch()

    post_step_patch.assert_not_called()
    assert _UpstreamEngineCoreProc.run_engine_core is _engine_core_patch._run_engine_core_patch_func
    # _apply_patch must NOT install the watchdog patches: they are deferred
    # to engine-core entry via patch_watchdog_for_engine_core().
    assert _UpstreamEngineCoreProc._process_engine_step is _watchdog_patch._original_process_engine_step
    assert _UpstreamSignalCallback.trigger is _watchdog_patch._original_signal_callback


def test_profiling_patch_loaded_through_engine_core_patch():
    """The spawned child re-applies the profiling patch by importing this
    module when unpickling the run_engine_core wrapper; importing
    patch_engine_core must therefore pull in patch_profiling_chunk and run
    its module-level _apply_profiling_patches()."""
    assert _profiling_patch._profiling_patches_applied is True


def test_watchdog_patch_loaded_through_engine_core_patch():
    """The engine-core child applies the watchdog patches by importing this
    module when unpickling the run_engine_core wrapper; importing
    patch_engine_core must therefore pull in patch_engine_watchdog, and the
    installer the wrapper calls must be that module's function."""
    assert _engine_core_patch.patch_watchdog_for_engine_core is _watchdog_patch.patch_watchdog_for_engine_core


def test_original_process_engine_step_stashes_pristine_upstream():
    """The watchdog feed patch must delegate to the genuine upstream
    ``_process_engine_step``.

    ``patch_engine_watchdog._original_process_engine_step`` is bound to
    ``EngineCoreProc._process_engine_step`` at the watchdog module's first
    import -- before anything rebinds the class attribute (the install is
    deferred to engine-core entry) -- so it is the pristine upstream
    original regardless of import ordering."""
    assert _watchdog_patch._original_process_engine_step is not _watchdog_patch._patched_process_engine_step
    src = inspect.getsource(_watchdog_patch._original_process_engine_step)
    assert "self.step_fn()" in src, (
        "upstream _process_engine_step no longer calls self.step_fn(); the "
        "watchdog feed patch in patch_engine_watchdog.py would silently break."
    )


def test_original_signal_callback_stashes_pristine_upstream():
    """The signal-callback patch must delegate to the genuine upstream
    ``SignalCallback.trigger``, captured at the watchdog module's first
    import before the deferred install rebinds it."""
    assert _watchdog_patch._original_signal_callback is not _watchdog_patch.patched_signal_callback
    src = inspect.getsource(_watchdog_patch._original_signal_callback)
    assert "self._event.set()" in src, (
        "upstream SignalCallback.trigger no longer sets the callback event; "
        "the signal-callback patch in patch_engine_watchdog.py would silently break."
    )


# ---------------------------------------------------------------------------
# 2. DPEngineCoreProc selection
# ---------------------------------------------------------------------------


def test_dp_proc_swap_prefers_dyntra_over_balance(monkeypatch):
    monkeypatch.setattr(_engine_core_patch, "_get_dyntra_lb_config", lambda _c: _dyntra_config(True))
    monkeypatch.setattr(
        _engine_core_patch,
        "_balance_scheduling_enabled",
        MagicMock(side_effect=AssertionError("balance must not be consulted when dyntra is enabled")),
    )
    print_mock = MagicMock()
    monkeypatch.setattr(_engine_core_patch, "dyntra_print_rank_0", print_mock)
    # Register restore of the module-global swap the code under test performs.
    monkeypatch.setattr(_engine_core_mod, "DPEngineCoreProc", _engine_core_mod.DPEngineCoreProc)

    _engine_core_patch._patch_dp_engine_core_proc(vllm_config=object(), dp_rank=2)

    assert _engine_core_mod.DPEngineCoreProc is DyntraLBDPEngineCoreProc
    print_mock.assert_called_once_with("Enable DyntraLB DP load balancing.", 2, True)


def test_dp_proc_swap_uses_balance_when_dyntra_disabled(monkeypatch):
    monkeypatch.setattr(_engine_core_patch, "_get_dyntra_lb_config", lambda _c: _dyntra_config(False))
    monkeypatch.setattr(_engine_core_patch, "_balance_scheduling_enabled", lambda _c: True)
    monkeypatch.setattr(_engine_core_mod, "DPEngineCoreProc", _engine_core_mod.DPEngineCoreProc)

    _engine_core_patch._patch_dp_engine_core_proc(vllm_config=object(), dp_rank=0)

    assert _engine_core_mod.DPEngineCoreProc is BalanceDPEngineCoreProc


def test_dp_proc_swap_leaves_pristine_when_both_disabled(monkeypatch):
    """Balance off must mean no involvement: the module-global class stays
    untouched so PD-disaggregated recompute etc. keeps the upstream proc."""
    monkeypatch.setattr(_engine_core_patch, "_get_dyntra_lb_config", lambda _c: _dyntra_config(False))
    monkeypatch.setattr(_engine_core_patch, "_balance_scheduling_enabled", lambda _c: False)
    monkeypatch.setattr(_engine_core_mod, "DPEngineCoreProc", _engine_core_mod.DPEngineCoreProc)

    _engine_core_patch._patch_dp_engine_core_proc(vllm_config=object(), dp_rank=0)

    assert _engine_core_mod.DPEngineCoreProc is _UpstreamDPEngineCoreProc


# ---------------------------------------------------------------------------
# 3. the run_engine_core wrapper
# ---------------------------------------------------------------------------


def test_run_engine_core_patch_delegates_and_passes_rank_args(monkeypatch):
    expected = object()
    calls = []

    def original(*args, dp_rank=0, local_dp_rank=0, **kwargs):
        calls.append((args, dp_rank, local_dp_rank, kwargs))
        return expected

    monkeypatch.setattr(_engine_core_patch, "_OriginalRunEngineCore", original)
    init_ascend = MagicMock(return_value=_ascend_config(profiling_enabled=False))
    monkeypatch.setattr(_engine_core_patch, "init_ascend_config", init_ascend)
    profiling = MagicMock()
    monkeypatch.setattr(_engine_core_patch, "_apply_profiling_patches", profiling)
    dp_proc_patch = MagicMock()
    monkeypatch.setattr(_engine_core_patch, "_patch_dp_engine_core_proc", dp_proc_patch)
    watchdog_installer = MagicMock()
    monkeypatch.setattr(_engine_core_patch, "patch_watchdog_for_engine_core", watchdog_installer)

    vllm_config = object()
    result = _engine_core_patch._run_engine_core_patch_func(
        "engine-arg",
        vllm_config=vllm_config,
        dp_rank=1,
        local_dp_rank=2,
    )

    assert result is expected
    assert calls == [
        (
            ("engine-arg",),
            1,
            2,
            {"vllm_config": vllm_config},
        )
    ]
    init_ascend.assert_called_once_with(vllm_config)
    dp_proc_patch.assert_called_once_with(vllm_config, 1)
    profiling.assert_not_called()
    watchdog_installer.assert_called_once_with()


def test_run_engine_core_patch_applies_profiling_when_enabled(monkeypatch):
    monkeypatch.setattr(_engine_core_patch, "_OriginalRunEngineCore", MagicMock())
    monkeypatch.setattr(
        _engine_core_patch,
        "init_ascend_config",
        MagicMock(return_value=_ascend_config(profiling_enabled=True)),
    )
    profiling = MagicMock()
    monkeypatch.setattr(_engine_core_patch, "_apply_profiling_patches", profiling)
    monkeypatch.setattr(_engine_core_patch, "_patch_dp_engine_core_proc", MagicMock())
    watchdog_installer = MagicMock()
    monkeypatch.setattr(_engine_core_patch, "patch_watchdog_for_engine_core", watchdog_installer)

    _engine_core_patch._run_engine_core_patch_func(vllm_config=object())

    profiling.assert_called_once_with()
    watchdog_installer.assert_called_once_with()
