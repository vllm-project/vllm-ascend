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
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from types import FunctionType, SimpleNamespace
from unittest.mock import patch

import pytest

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
    except ModuleNotFoundError as exc:
        # Only the explicitly optional Xlite backend may suppress an import.
        if exc.name != "xlite" and not (exc.name or "").startswith("xlite."):
            raise
        if label != "XliteWorker":
            raise
        return GuardedClass(label, None, parent, interfaces, f"{type(exc).__name__}: {exc}")
    return GuardedClass(label, getattr(loaded, attribute), parent, interfaces)


@pytest.fixture(scope="module")
def npu_guard():
    # Lazy imports keep the guard's own regression tests hardware-independent.
    from vllm.v1.worker.gpu_worker import Worker as GPUWorker

    from vllm_ascend.worker.worker import NPUWorker

    return GuardedClass("NPUWorker", NPUWorker, GPUWorker, NPU_WORKER_INTERFACES)


@pytest.fixture(params=("NPUWorker", "NPUWorker310", "XliteWorker"), scope="module")
def guard(request, npu_guard):
    if request.param == "NPUWorker":
        return npu_guard
    module, attribute, interfaces = {
        "NPUWorker310": ("vllm_ascend._310p.worker_310p", "NPUWorker310", NPU_WORKER_310_INTERFACES),
        "XliteWorker": ("vllm_ascend.xlite.xlite_worker", "XliteWorker", XLITE_WORKER_INTERFACES),
    }[request.param]
    return _load_optional(request.param, module, attribute, npu_guard.cls, interfaces)


def _skip_if_unavailable(guard: GuardedClass) -> None:
    if guard.cls is None:
        pytest.skip(f"{guard.label} is not importable in this environment ({guard.skip_reason})")


def _declared_functions(cls: type) -> dict[str, object]:
    """Methods the class itself defines (not inherited ones)."""
    return {
        name: value
        for name, value in vars(cls).items()
        if inspect.isfunction(value) or isinstance(value, (classmethod, staticmethod, property))
    }


def _signature_gaps(local_fn: object, parent_fn: object) -> list[str]:
    """Check inclusion of the parent's call shapes using Signature.bind.

    For each positional arity, the minimal keywords detect new requirements;
    the maximal keywords detect rejected names and duplicate bindings. Extra
    **kwargs names include every local parameter and a fresh name. Beyond both
    signatures' fixed positional slots, one extra argument covers *args.
    Values do not affect binding, so a single sentinel represents each value.
    """
    local = inspect.signature(local_fn)
    parent = inspect.signature(parent_fn)
    positional_kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    local_pos = [p for p in local.parameters.values() if p.kind in positional_kinds]
    parent_pos = [p for p in parent.parameters.values() if p.kind in positional_kinds]
    keyword_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    parent_keywords = {p.name for p in parent.parameters.values() if p.kind in keyword_kinds}
    has_args = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parent.parameters.values())
    has_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parent.parameters.values())
    limit = max(len(local_pos), len(parent_pos)) + 1 if has_args else len(parent_pos)
    fresh = "__contract_extra"
    while fresh in local.parameters or fresh in parent.parameters:
        fresh += "_"
    extras = (set(local.parameters) | set(parent.parameters) | {fresh}) - parent_keywords
    sentinel = object()
    gaps = []
    for count in range(limit + 1):
        consumed = {p.name for p in parent_pos[:count]}
        remaining = [p for p in parent.parameters.values() if p.kind in keyword_kinds and p.name not in consumed]
        required = {p.name: sentinel for p in remaining if p.default is inspect.Parameter.empty}
        optional = {p.name: sentinel for p in remaining}
        for keywords in (required, optional):
            candidates = [keywords]
            if has_kwargs:
                candidates += [dict(keywords, **{name: sentinel}) for name in sorted(extras)]
            for candidate in candidates:
                args = (sentinel,) * count
                try:
                    parent.bind(*args, **candidate)
                except TypeError:
                    continue
                try:
                    local.bind(*args, **candidate)
                except TypeError as exc:
                    gaps.append(
                        f"parent call with {count} positional arguments and keywords {sorted(candidate)}: {exc}"
                    )
    return gaps


def _probe_decorators(fn, probe):
    """Execute the real wrapper chain around a harmless replacement body.

    Preserve each wrapper's code, globals and context factories, replacing only
    the closure reference to its wrapped function. No Worker body is executed.
    Unknown wrapper layouts fail closed instead of claiming semantic parity.
    """
    wrapped = getattr(fn, "__wrapped__", None)
    if wrapped is None:
        return probe
    replacement = _probe_decorators(wrapped, probe)
    closure = fn.__closure__ or ()
    assert any(cell.cell_contents is wrapped for cell in closure), "Decorator does not capture its wrapped function"
    rebound = FunctionType(
        fn.__code__,
        fn.__globals__,
        fn.__name__,
        fn.__defaults__,
        tuple(_make_cell(replacement) if cell.cell_contents is wrapped else cell for cell in closure),
    )
    rebound.__kwdefaults__ = fn.__kwdefaults__
    return rebound


def _make_cell(value):
    return (lambda: value).__closure__[0]


def _decorator_effects(fn):
    import torch
    from vllm.utils import gpu_sync_debug

    sync_regions = []

    @contextmanager
    def checked_region(mode):
        sync_regions.append(mode)
        yield

    with (
        patch.object(gpu_sync_debug, "_sync_check_enabled", True),
        patch.object(gpu_sync_debug, "_checked_region", checked_region),
    ):
        inference_mode = _probe_decorators(fn, torch.is_inference_mode_enabled)()
    return inference_mode, tuple(sync_regions)


def _sync_check_is_enforced() -> bool:
    """Whether ``with_gpu_sync_check`` wraps at all on this platform.

    Off CUDA-like platforms it returns the function unchanged, so the parent's
    decorator stack is one layer shallower than on a GPU runner.
    """

    from vllm.utils.gpu_sync_debug import with_gpu_sync_check

    def probe() -> None:
        return None

    return with_gpu_sync_check(probe) is not probe


def test_every_override_is_registered(guard):
    """A new override must be classified before it can land."""
    _skip_if_unavailable(guard)
    declared = set(_declared_functions(guard.cls))
    registered = {entry.name for entry in guard.interfaces}
    declared_entries = {entry.name for entry in guard.interfaces if entry.disposition is not Disposition.INHERIT}

    missing = sorted(declared - registered)
    stale = sorted(declared_entries - declared)

    assert not missing, (
        f"{guard.label} defines overrides that are not classified in tests/ut/worker/worker_contract.py: {missing}. "
        "Add a WorkerInterface entry with a disposition, the RFC phase and the concrete reason for the override."
    )
    assert not stale, (
        f"{guard.label} registry lists methods the class no longer defines: {stale}. "
        "Mark inherited methods INHERIT; delete obsolete entries in the same change that removes the method."
    )


def test_registry_has_no_duplicate_entries(guard):
    _skip_if_unavailable(guard)
    names = [entry.name for entry in guard.interfaces]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"{guard.label} registry classifies the same method twice: {duplicates}"


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


def test_inherit_entries_resolve_to_the_parent(guard):
    """The list that grows phase by phase must actually resolve upstream."""
    _skip_if_unavailable(guard)
    unresolved = [
        entry.name
        for entry in guard.interfaces
        if entry.disposition is Disposition.INHERIT
        and (
            entry.name in vars(guard.cls)
            or not hasattr(guard.parent, entry.name)
            or inspect.getattr_static(guard.cls, entry.name, None)
            is not inspect.getattr_static(guard.parent, entry.name, None)
        )
    ]
    assert not unresolved, (
        f"{guard.label}: registered as INHERIT but still defined on the class (or missing upstream): {unresolved}. "
        "Either delete the override or give it a quarantined disposition with the phase that removes it."
    )


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
            assert inspect.getattr_static(guard.cls, entry.name) is not inspect.getattr_static(
                guard.parent, entry.name
            ), (
                f"{guard.label}.{entry.name} is registered as {entry.disposition.value!r} but already resolves to "
                "the parent; flip the disposition to INHERIT."
            )
    assert not not_declared, (
        f"{guard.label}: quarantined overrides missing from the class: {not_declared}. "
        "Move them to INHERIT once they are removed."
    )


def test_alignment_targets_exist_on_the_parent(npu_guard):
    """ALIGN_FIRST and PENDING_REMOVAL only make sense for upstream methods."""
    GPUWorker = npu_guard.parent
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


def test_quarantined_signatures_accept_the_parent_call_surface(npu_guard):
    """Every call the parent accepts must still be a valid call downstream."""
    NPUWorker, GPUWorker = npu_guard.cls, npu_guard.parent
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


def test_guarded_methods_keep_parent_decorators(npu_guard):
    """Compare inference-mode and sync-region effects, not wrapper counts."""
    expected = set(KNOWN_DECORATOR_GAPS)
    if _sync_check_is_enforced():
        expected.add("sample_tokens")
    observed = set()
    for name in DECORATOR_GUARDED_METHODS:
        local = _decorator_effects(getattr(npu_guard.cls, name))
        parent = _decorator_effects(getattr(npu_guard.parent, name))
        if local != parent:
            observed.add(name)
    assert observed == expected, f"Decorator effects changed: expected gaps {expected}, observed {observed}"


class TestContractGuards:
    """Regression tests for the gates themselves; no Worker imports required."""

    @pytest.mark.parametrize(
        "parent, local, compatible",
        [
            ("x=1", "renamed=1", False),
            ("", "extra", False),
            ("*, x=1", "*, x", False),
            ("x=1", "x", False),
            ("x", "*, x", False),
            ("*, x", "x", True),
            ("x", "x, /", False),
            ("x, /", "renamed, /", True),
            ("x=1", "*args, **kwargs", True),
            ("x, y=1", "x, *args, **kwargs", True),
            ("*args", "", False),
            ("**kwargs", "", False),
            ("*args, **kwargs", "*args, **kwargs", True),
            ("x=1, **kwargs", "x=1, y=2, **kwargs", True),
            ("x, /, **kwargs", "x, **kwargs", False),
            ("x, **kwargs", "x, /, **kwargs", False),
            ("x=1, *args", "x=1, y=2, *args", True),
            ("x=1", "x=1, *, extra=None", True),
            ("x=1, **kwargs", "x=1, *, extra=None", False),
            ("x=1", "y=1, **kwargs", True),
            ("x=1", "x=1, extra=None", True),
            ("x=1", "y, **kwargs", False),
            ("*, x=1", "x, **kwargs", False),
        ],
    )
    def test_signature_call_shapes(self, parent, local, compatible):
        parent_fn = eval(f"lambda {parent}: None")
        local_fn = eval(f"lambda {local}: None")
        assert (not _signature_gaps(local_fn, parent_fn)) is compatible

    @pytest.mark.parametrize("kind", [classmethod, staticmethod, property])
    def test_descriptors_cannot_bypass_registry(self, kind):
        cls = type("Child", (), {"method": kind(lambda self: None)})
        guard = GuardedClass("Child", cls, object, ())
        assert "method" in _declared_functions(cls)
        with pytest.raises(AssertionError, match="not classified"):
            test_every_override_is_registered(guard)

    @pytest.mark.parametrize("mode", ["inherited", "reintroduced", "alias", "missing"])
    @pytest.mark.parametrize("kind", [lambda fn: fn, classmethod, staticmethod, property])
    def test_inherited_registry_resolution(self, mode, kind):
        class Parent:
            def method(self):
                pass

        Parent.method = kind(Parent.method)

        class Child(Parent):
            pass

        if mode == "reintroduced":
            Child.method = lambda self: None
        elif mode == "alias":
            Child.method = vars(Parent)["method"]
        elif mode == "missing":
            del Parent.method
        entry = WorkerInterface("method", Disposition.INHERIT, "2", "Forwarding now resolves from the parent.")
        guard = GuardedClass("Child", Child, Parent, (entry,))
        test_every_override_is_registered(guard)
        if mode == "inherited":
            test_inherit_entries_resolve_to_the_parent(guard)
        else:
            with pytest.raises(AssertionError, match="registered as INHERIT"):
                test_inherit_entries_resolve_to_the_parent(guard)

    @pytest.mark.parametrize(
        "label, missing, optional",
        [
            ("XliteWorker", "xlite", True),
            ("XliteWorker", "xlite.backend", True),
            ("XliteWorker", "vllm_ascend.xlite.broken", False),
            ("XliteWorker", "vllm_ascend.xlite.xlite_worker", False),
            ("XliteWorker", "unrelated_dependency", False),
            ("NPUWorker310", "xlite", False),
            ("NPUWorker310", "vllm_ascend._310p.broken", False),
        ],
    )
    def test_optional_import_only_skips_xlite_dependency(self, label, missing, optional):
        error = ModuleNotFoundError(f"No module named {missing}", name=missing)
        with patch.object(importlib, "import_module", side_effect=error):
            if optional:
                result = _load_optional(label, "worker_module", "Worker", object, ())
                assert result.cls is None
                assert missing in result.skip_reason
            else:
                with pytest.raises(ModuleNotFoundError) as caught:
                    _load_optional(label, "worker_module", "Worker", object, ())
                assert caught.value is error

    def test_optional_import_success(self):
        cls = type("Worker", (), {})
        with patch.object(importlib, "import_module", return_value=SimpleNamespace(Worker=cls)):
            assert _load_optional("XliteWorker", "worker_module", "Worker", object, ()).cls is cls

    @pytest.mark.parametrize("enabled", [False, True])
    def test_decorator_probe_distinguishes_inference_mode(self, enabled):
        import torch

        def body():
            raise AssertionError("Worker body must never run")

        @wraps(body)
        def unrelated():
            return body()

        wrapper = torch.inference_mode()(body) if enabled else unrelated
        # Both wrappers have depth one, but only one enables inference mode.
        with torch.inference_mode(False):
            assert _probe_decorators(wrapper, torch.is_inference_mode_enabled)() is enabled

    def test_decorator_probe_preserves_nested_effects(self):
        events = []

        def decorate(fn):
            @wraps(fn)
            def wrapped():
                events.append("enter")
                result = fn()
                events.append("exit")
                return result

            return wrapped

        def body():
            raise AssertionError("Worker body must never run")

        assert _probe_decorators(decorate(decorate(body)), lambda: "probe")() == "probe"
        assert events == ["enter", "enter", "exit", "exit"]

    def test_decorator_probe_rejects_unknown_wrapper_layout(self):
        def body():
            pass

        @wraps(body)
        def opaque():
            return None

        with pytest.raises(AssertionError, match="does not capture"):
            _probe_decorators(opaque, lambda: None)
