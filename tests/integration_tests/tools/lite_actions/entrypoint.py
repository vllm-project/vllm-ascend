#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lite Actions adapter: trusted case definitions and fixed SHA selectors."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from tests.integration_tests import OverrideDefinitions  # noqa: TC001 - retain runtime annotation resolution
from tests.integration_tests.training_cases.dispatch_smoke import SMOKE_DEFAULTS, SMOKE_PROFILES, build_smoke_list

SAFE = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")


def catalog() -> dict[str, OverrideDefinitions]:
    found = {}
    for test in build_smoke_list():
        if not SAFE.fullmatch(test.test_name) or test.test_name in found or test.disabled:
            raise ValueError("duplicate/invalid/disabled CI case")
        if test.ci_kind == "worker_probe":
            valid_topology = test.nnodes == test.ngpu == 0
        else:
            valid_topology = (
                isinstance(test.nnodes, int) and test.nnodes >= 1 and isinstance(test.ngpu, int) and test.ngpu >= 1
            )
        if not valid_topology:
            raise ValueError("invalid topology")
        found[test.test_name] = test
    if not found:
        raise ValueError("empty test catalog")
    return found


def select(catalogue: dict, *, test_id: str | None = None, suite: str | None = None):
    if (test_id is None) == (suite is None):
        raise ValueError("select exactly one ID or suite")
    if suite is not None:
        if suite not in SMOKE_PROFILES:
            raise ValueError("unknown CI suite")
        return [select(catalogue, test_id=name)[0] for name in SMOKE_PROFILES[suite]]
    if test_id not in catalogue:
        raise ValueError("unknown CI test")
    return [catalogue[test_id]]


def inspect_overrides(test: OverrideDefinitions) -> dict:
    if test.ci_kind == "worker_probe":
        return {"worker": SMOKE_DEFAULTS["worker"], **test.ci_overrides}
    if test.ci_kind == "npu_smoke":
        return {**SMOKE_DEFAULTS, **test.ci_overrides}
    return dict(test.ci_overrides)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("inspect", "launch", "verify"))
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--test-id")
    sel.add_argument("--suite")
    p.add_argument("--output-dir", type=Path)
    a = p.parse_args()
    if a.phase != "inspect" and (a.test_id is None or a.output_dir is None):
        p.error("launch/verify require an explicit test ID and output directory")
    try:
        selected = select(catalog(), test_id=a.test_id, suite=a.suite)
    except ValueError as exc:
        p.error(str(exc))
    if a.phase == "inspect":
        print(
            json.dumps(
                [
                    {
                        "test_id": t.test_name,
                        "ngpu": t.ngpu,
                        "nnodes": t.nnodes,
                        "env_vars": dict(t.env_vars or {}),
                        "kind": t.ci_kind,
                        "ci_overrides": inspect_overrides(t),
                        "label": t.test_descr,
                        "timeout": t.timeout,
                        "device": t.device,
                        "required_env": list(t.required_env),
                        "phases": len(t.override_args),
                    }
                    for t in selected
                ],
                separators=(",", ":"),
            )
        )
        return
    test = selected[0]
    if test.ci_kind == "worker_probe":
        p.error("worker probes execute only in the trusted Lite Actions worker")
    if test.ci_kind != "npu_smoke":
        p.error(f"unknown CI case kind: {test.ci_kind}")
    if a.phase == "verify":
        # The Worker requires every node's launch to exit successfully first.
        return
    # Also support direct, manual invocation of the fixed model-owned entrypoint.
    # Remote SSH has already sourced the case's toolkit before Python starts.
    for name, value in (test.env_vars or {}).items():
        os.environ.setdefault(name, value)
    from tests.integration_tests.tools.lite_actions.runner import run_dispatch_smoke

    run_dispatch_smoke(test)


if __name__ == "__main__":
    main()
