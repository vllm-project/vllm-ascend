"""Contract guards for the NPUWorker -> GPUWorker migration (RFC #18920).

Phase 0 of https://github.com/vllm-project/vllm-ascend/issues/18920 has to make
the inheritance contract machine-checkable before any override is deleted. These
tests are that mechanism:

* every downstream override must be classified in ``worker_contract.py``;
* an override classified ``INHERIT`` must really resolve to
  ``vllm.v1.worker.gpu_worker.Worker``;
* a quarantined override must still accept every call its parent accepts and
  must not lose a parent decorator.

The guard is intentionally fail-closed in both directions: fixing a known gap
without updating the registry fails, exactly like introducing a new one.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass

import pytest
from vllm.utils.gpu_sync_debug import with_gpu_sync_check
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from tests.ut.worker.worker_contract import (
    DECORATOR_GUARDED_METHODS,
    KNOWN_DECORATOR_GAPS,
    KNOWN_SIGNATURE_GAPS,
    NPU_WORKER_310_INTERFACES,
    NPU_WORKER_INTERFACES,
    PHASES,
    REMOVAL_PHASES,
    XLITE_WORKER_INTERFACES,
    Disposition,
    WorkerInterface,
)
from vllm_ascend.worker.worker import NPUWorker


@dataclass(frozen=True)
class GuardedClass:
    """One worker class, its inheritance parent and its registry."""

    label: str
    cls: type | None
    parent: type
    interfaces: tuple[WorkerInterface, ...]
    skip_reason: str | None = None


def _load_optional(
    label: str,
    module: str,
    attribute: str,
    parent: type,
    interfaces: tuple[WorkerInterface, ...],
) -> GuardedClass:
    """Import a subclass, tolerating an uninstalled optional backend."""
    try:
        loaded = importlib.import_module(module)
    except ModuleNotFoundError as exc:  # e.g. xlite is an optional accelerator
        return GuardedClass(label, None, parent, interfaces, f"{type(exc).__name__}: {exc}")
    return GuardedClass(label, getattr(loaded, attribute), parent, interfaces)


GUARDED_CLASSES = (
    GuardedClass("NPUWorker", NPUWorker, GPUWorker, NPU_WORKER_INTERFACES),
    _load_optional(
        "NPUWorker310",
        "vllm_ascend._310p.worker_310p",
        "NPUWorker310",
        NPUWorker,
        NPU_WORKER_310_INTERFACES,
    ),
    _load_optional(
        "XliteWorker",
        "vllm_ascend.xlite.xlite_worker",
        "XliteWorker",
        NPUWorker,
        XLITE_WORKER_INTERFACES,
    ),
)

_CASES = [pytest.param(guard, id=guard.label) for guard in GUARDED_CLASSES]


def _skip_if_unavailable(guard: GuardedClass) -> None:
    if guard.cls is None:
        pytest.skip(f"{guard.label} is not importable in this environment ({guard.skip_reason})")


def _declared_functions(cls: type) -> dict[str, object]:
    """Methods the class itself defines (not inherited ones)."""
    return {name: value for name, value in vars(cls).items() if inspect.isfunction(value)}


def _signature_gaps(local_fn: object, parent_fn: object) -> list[str]:
    """Ways ``local_fn`` accepts less than ``parent_fn``.

    A parent-valid call is any call the parent accepts. Every parameter the
    parent exposes must therefore be accepted here as well (by position count,
    by keyword name, or through ``**kwargs``), with no stricter default.
    """
    local = inspect.signature(local_fn)  # type: ignore[arg-type]
    parent = inspect.signature(parent_fn)  # type: ignore[arg-type]
    gaps: list[str] = []

    positional_kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    local_positional = [p for p in local.parameters.values() if p.kind in positional_kinds]
    parent_positional = [p for p in parent.parameters.values() if p.kind in positional_kinds]
    if len(local_positional) < len(parent_positional):
        gaps.append(
            f"accepts {len(local_positional)} positional parameters, the parent accepts {len(parent_positional)}"
        )
    for local_param, parent_param in zip(local_positional, parent_positional):
        if parent_param.default is not inspect.Parameter.empty and local_param.default is inspect.Parameter.empty:
            gaps.append(f"parameter {local_param.name!r} is required here but optional in the parent")

    accepts_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in local.parameters.values())
    if not accepts_var_keyword:
        local_keyword_only = {p.name for p in local.parameters.values() if p.kind is inspect.Parameter.KEYWORD_ONLY}
        parent_keyword_only = {p.name for p in parent.parameters.values() if p.kind is inspect.Parameter.KEYWORD_ONLY}
        missing = sorted(parent_keyword_only - local_keyword_only)
        if missing:
            gaps.append(f"missing keyword-only parameters {missing}")

    parent_var_positional = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parent.parameters.values())
    local_var_positional = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in local.parameters.values())
    if parent_var_positional and not local_var_positional:
        gaps.append("the parent accepts varargs but this override does not")

    return gaps


def _wrapper_depth(fn: object) -> int:
    """Number of decorator layers that preserved ``__wrapped__``."""
    depth = 0
    seen: set[int] = set()
    while (wrapped := getattr(fn, "__wrapped__", None)) is not None and id(fn) not in seen:
        seen.add(id(fn))
        fn = wrapped
        depth += 1
    return depth


def _sync_check_is_enforced() -> bool:
    """Whether ``with_gpu_sync_check`` wraps at all on this platform.

    Off CUDA-like platforms it returns the function unchanged, so the parent's
    decorator stack is one layer shallower than on a GPU runner.
    """

    def probe() -> None:
        return None

    return with_gpu_sync_check(probe) is not probe


SYNC_CHECK_ENFORCED = _sync_check_is_enforced()

#: ``sample_tokens`` keeps inference_mode but not the sync-check layer, which
#: only exists on CUDA-like platforms.
EXPECTED_DECORATOR_GAPS = frozenset(KNOWN_DECORATOR_GAPS) | (
    frozenset({"sample_tokens"}) if SYNC_CHECK_ENFORCED else frozenset()
)


@pytest.mark.parametrize("guard", _CASES)
def test_every_override_is_registered(guard):
    """A new override must be classified before it can land."""
    _skip_if_unavailable(guard)
    declared = set(_declared_functions(guard.cls))
    registered = {entry.name for entry in guard.interfaces}

    missing = sorted(declared - registered)
    stale = sorted(registered - declared)

    assert not missing, (
        f"{guard.label} defines overrides that are not classified in tests/ut/worker/worker_contract.py: {missing}. "
        "Add a WorkerInterface entry with a disposition, the RFC phase and the concrete reason for the override."
    )
    assert not stale, (
        f"{guard.label} registry lists methods the class no longer defines: {stale}. "
        "Move them to a parent registry (or delete the entry) in the same change that removed the override."
    )


@pytest.mark.parametrize("guard", _CASES)
def test_registry_has_no_duplicate_entries(guard):
    _skip_if_unavailable(guard)
    names = [entry.name for entry in guard.interfaces]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"{guard.label} registry classifies the same method twice: {duplicates}"


@pytest.mark.parametrize("guard", _CASES)
def test_dispositions_and_phases_are_consistent(guard):
    _skip_if_unavailable(guard)
    for entry in guard.interfaces:
        assert entry.phase in PHASES, f"{guard.label}.{entry.name}: unknown RFC phase {entry.phase!r}"
        assert len(entry.reason.strip()) >= 20, (
            f"{guard.label}.{entry.name}: every override needs a concrete reason, got {entry.reason!r}"
        )
        if entry.disposition in (Disposition.INHERIT, Disposition.PENDING_REMOVAL):
            assert entry.phase in REMOVAL_PHASES, (
                f"{guard.label}.{entry.name}: disposition {entry.disposition.value!r} needs a removal phase, "
                f"got {entry.phase!r}"
            )


@pytest.mark.parametrize("guard", _CASES)
def test_inherit_entries_resolve_to_the_parent(guard):
    """The list that grows phase by phase must actually resolve upstream."""
    _skip_if_unavailable(guard)
    unresolved = [
        entry.name
        for entry in guard.interfaces
        if entry.disposition is Disposition.INHERIT
        and getattr(guard.cls, entry.name, None) is not getattr(guard.parent, entry.name, None)
    ]
    assert not unresolved, (
        f"{guard.label}: registered as INHERIT but still defined on the class (or missing upstream): {unresolved}. "
        "Either delete the override or give it a quarantined disposition with the phase that removes it."
    )


@pytest.mark.parametrize("guard", _CASES)
def test_quarantined_overrides_are_still_declared(guard):
    """Phase 0 keeps behaviour; nothing may disappear before its removal phase."""
    _skip_if_unavailable(guard)
    not_declared = []
    for entry in guard.interfaces:
        if entry.disposition is Disposition.INHERIT:
            continue
        if entry.name not in vars(guard.cls):
            not_declared.append(entry.name)
            continue
        if hasattr(guard.parent, entry.name):
            assert getattr(guard.cls, entry.name) is not getattr(guard.parent, entry.name), (
                f"{guard.label}.{entry.name} is registered as {entry.disposition.value!r} but already resolves to "
                "the parent; flip the disposition to INHERIT."
            )
    assert not not_declared, (
        f"{guard.label}: quarantined overrides missing from the class: {not_declared}. "
        "Move them to INHERIT once they are removed."
    )


def test_alignment_targets_exist_on_the_parent():
    """ALIGN_FIRST and PENDING_REMOVAL only make sense for upstream methods."""
    not_upstream = [
        entry.name
        for entry in NPU_WORKER_INTERFACES
        if entry.disposition in (Disposition.ALIGN_FIRST, Disposition.PENDING_REMOVAL)
        and not hasattr(GPUWorker, entry.name)
    ]
    assert not not_upstream, (
        f"Classified as alignable but the parent Worker has no such method: {not_upstream}. "
        "Use ASCEND_SPECIFIC with the NPU behaviour it provides."
    )


def test_quarantined_signatures_accept_the_parent_call_surface():
    """Every call the parent accepts must still be a valid call downstream."""
    observed: dict[str, list[str]] = {}
    for entry in NPU_WORKER_INTERFACES:
        if not hasattr(GPUWorker, entry.name) or entry.name not in vars(NPUWorker):
            continue
        gaps = _signature_gaps(vars(NPUWorker)[entry.name], getattr(GPUWorker, entry.name))
        if gaps:
            observed[entry.name] = gaps

    assert set(observed) == set(KNOWN_SIGNATURE_GAPS), (
        "Signature incompatibilities with the parent Worker changed.\n"
        f"  expected: {sorted(KNOWN_SIGNATURE_GAPS)}\n"
        f"  observed: {sorted(observed)}\n"
        f"  details: {observed}\n"
        "If a gap was fixed, drop it from KNOWN_SIGNATURE_GAPS; if a new one appeared, see RFC Phase 0.5c."
    )


def test_guarded_methods_keep_parent_decorators():
    """Inheriting must not silently change inference-mode or sync-check behaviour."""
    observed = {
        name
        for name in DECORATOR_GUARDED_METHODS
        if _wrapper_depth(getattr(NPUWorker, name)) < _wrapper_depth(getattr(GPUWorker, name))
    }
    assert observed == set(EXPECTED_DECORATOR_GAPS), (
        "Decorator parity with the parent Worker changed.\n"
        f"  expected gaps: {sorted(EXPECTED_DECORATOR_GAPS)} (sync check enforced: {SYNC_CHECK_ENFORCED})\n"
        f"  observed gaps: {sorted(observed)}\n"
        "Phase 0.5a adds @torch.inference_mode to execute_model and @with_gpu_sync_check to both hotspots; "
        "update KNOWN_DECORATOR_GAPS together with the code."
    )
