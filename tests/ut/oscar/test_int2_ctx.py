# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the INT2 shared import hub (ctx).

Pins the facade surface: every exported name resolves, the config instance is
a per-process singleton, the cross-module diag registry exists, and the
backend modules contain NO star imports (explicit-interface invariant).

Registry invariants (the split-share contract):
  * the eleven handoff names are accessed ONLY through ``_R2XX_G`` — never
    via bare module ``globals()`` (that is the silent-isolation bug class
    the registry exists to prevent);
  * every remaining bare-``globals()`` name in the package is MODULE-LOCAL
    (used in exactly one module), so no other cross-module handoff exists;
  * every backend module's ``_R2XX_G`` IS the ctx dict (one shared object).
"""

import importlib
import pathlib

import regex as re

import vllm_ascend.attention.int2 as pkg
import vllm_ascend.attention.int2.ctx as ctx

REGISTRY_NAMES = [
    "_R109_STATS",
    "_R164_PASSBUF",
    "_R164_PASSN",
    "_R164_WCHKBUF",
    "_R164_WCHKN",
    "_R166_WV_BLOCKS",
    "_R197_PROBE_N",
    "_R218_DISC",
    "_R218_FIRES",
    "_R218_MIDC_N",
    "_R218_ROT_SEEN",
    "_R218_XAS",
]

_GLOB = re.compile(
    r"""globals\(\)\s*(?:\.\s*(?:get|setdefault)\s*\(\s*["']([^"']+)["']"""
    r"""|\[\s*["']([^"']+)["']\s*\])"""
)


def _backend_modules():
    root = pathlib.Path(pkg.__file__).parent
    return sorted(root.glob("*.py"))


def test_all_names_resolve():
    for name in ctx.__all__:
        assert hasattr(ctx, name), f"__all__ names {name!r} missing on module"


def test_oscar_cfg_is_singleton():
    import vllm_ascend.attention.int2.ctx as ctx2

    assert ctx is ctx2
    assert ctx._OSCAR_CFG is not None


def test_diag_registry_shared_dict():
    assert isinstance(ctx._R2XX_G, dict)
    ctx._R2XX_G.setdefault("_TEST_K", [0])
    ctx._R2XX_G["_TEST_K"][0] += 1
    assert ctx._R2XX_G["_TEST_K"] == [1]
    del ctx._R2XX_G["_TEST_K"]


def test_backend_modules_have_no_star_imports():
    for py in _backend_modules():
        src = py.read_text()
        assert "import *" not in src, f"{py.name} contains a star import"


def test_registry_names_never_bare_globals():
    # the handoff names must ride the shared dict, not module globals()
    for py in _backend_modules():
        for m in _GLOB.finditer(py.read_text()):
            name = next(g for g in m.groups() if g)
            assert name not in REGISTRY_NAMES, f"{py.name} accesses {name} via bare globals() — use _R2XX_G"


def test_remaining_bare_globals_names_are_module_local():
    # any OTHER name kept in module globals() must be used in exactly one
    # module; a name appearing in two modules is a silent-isolation bug
    where: dict[str, set[str]] = {}
    for py in _backend_modules():
        for m in _GLOB.finditer(py.read_text()):
            name = next(g for g in m.groups() if g)
            where.setdefault(name, set()).add(py.stem)
    cross = {n: sorted(v) for n, v in where.items() if len(v) > 1}
    assert not cross, f"cross-module globals() names not on the registry: {cross}"


def test_every_module_registry_binding_is_the_ctx_object():
    for py in _backend_modules():
        mod = importlib.import_module(f"vllm_ascend.attention.int2.{py.stem}")
        bound = getattr(mod, "_R2XX_G", None)
        if bound is not None:
            assert bound is ctx._R2XX_G, f"{py.stem} re-bound _R2XX_G"


def test_registry_holds_no_tensors_when_diagnostics_off():
    import torch

    for v in ctx._R2XX_G.values():
        assert not isinstance(v, torch.Tensor)
        if isinstance(v, (list, tuple)):
            assert not any(isinstance(e, torch.Tensor) for e in v)
        if isinstance(v, dict):
            assert not any(isinstance(e, torch.Tensor) for e in v.values())
