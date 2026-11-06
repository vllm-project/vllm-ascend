# SPDX-License-Identifier: Apache-2.0
"""External-path resolution for the OSCAR INT2 path.

The frozen overlay hard-coded the experiment workspace layout under the
author's home directory. This module is the single resolution point for the
paths the INT2 backend actually needs at runtime:

  - ``opp_path()``     vendor operator package (op_api/op_impl tree).
  - ``launcher_src()`` C++ launcher extension source (.cpp inside this
                       repository by default: ``csrc/oscar/``).
  - ``runner_tmp()``  scratch dirs for torch extension builds.

Each resolves from the environment first (the variables the RC53 serve
chain already sets), then falls back to repository-relative defaults.
Nothing in here may reference a user's home directory.
"""

from __future__ import annotations

import os

# vllm_ascend/oscar/paths.py is three levels below the repo root
# (file -> oscar -> vllm_ascend -> root); three dirname calls resolve it.
# Out-of-tree installs should set the OSCAR_* env overrides instead of
# relying on this default.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def opp_path() -> str:
    """Vendor OPP tree (contains op_api/lib, op_impl, ...).

    Env: ``R144_OPP_PATH`` (RC53 serve chain) or ``OSCAR_OPP_PATH``.
    Raises FileNotFoundError when unset: loading a stale hard-coded vendor
    silently was the failure mode this replaces.
    """
    for var in ("R144_OPP_PATH", "OSCAR_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH"):
        v = os.environ.get(var, "")
        if v:
            return v.split(os.pathsep)[0]
    raise FileNotFoundError(
        "OSCAR vendor OPP path not set (expected R144_OPP_PATH / OSCAR_OPP_PATH / ASCEND_CUSTOM_OPP_PATH)"
    )


def launcher_src() -> str:
    """Launcher extension source file.

    Env: ``R144_LAUNCHER_SRC`` (full .cpp path, RC53 serve chain) or
    ``OSCAR_LAUNCHER_SRC``; default = in-repo ``csrc/oscar`` launcher.
    """
    for var in ("R144_LAUNCHER_SRC", "OSCAR_LAUNCHER_SRC"):
        v = os.environ.get(var, "")
        if v:
            return v
    return os.path.join(_REPO_ROOT, "csrc", "oscar", "int2_op_launch_lse_fast_r152.cpp")


def launcher_name() -> str:
    return os.environ.get("R144_LAUNCHER_NAME", os.environ.get("OSCAR_LAUNCHER_NAME", "int2_lse_fast_r152"))


def runner_tmp(kind: str) -> str:
    """Scratch dir for torch extension builds / kernel meta caches."""
    # default suffixes the scratch dir with the uid: /tmp is world-shared on
    # multi-user hosts and extension-build caches must not collide
    base = os.environ.get("OSCAR_RUNNER_TMP", f"/tmp/oscar_runner-{os.getuid()}")
    d = os.path.join(base, kind)
    os.makedirs(d, exist_ok=True)
    return d


def lse_bridge_src() -> str:
    """Source of the R98 LSE bridge extension (int2_op_launch_lse_r98.cpp).

    OSCAR_LSE_BRIDGE_SRC overrides (e.g. a frozen out-of-tree copy); the
    default is the in-repo csrc copy. Raises if neither exists.
    """
    import os

    env = os.environ.get("OSCAR_LSE_BRIDGE_SRC")
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(f"OSCAR_LSE_BRIDGE_SRC={env} does not exist")
        return env
    cand = os.path.join(_REPO_ROOT, "csrc", "oscar", "int2_op_launch_lse_r98.cpp")
    if os.path.isfile(cand):
        return cand
    raise FileNotFoundError("R98 LSE bridge source not found (set OSCAR_LSE_BRIDGE_SRC)")
