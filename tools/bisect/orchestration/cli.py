# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""Decode CLI inputs and assemble the bisect application."""

import argparse
import logging
import math
import os
from pathlib import Path

from tools.bisect.build_analyzer.build_sources import normalize_build_soc
from tools.bisect.execution.participants import resolve_node_count
from tools.bisect.execution.targets import ExecutionTargets
from tools.bisect.orchestration.config import (
    DEFAULT_COORD_DIR,
    DEFAULT_GOOD_TABLE,
    DEFAULT_WORK_DIR,
    REPO_ROOT,
    SCENE_MULTI,
    SCENES,
    BisectInput,
    BisectOptions,
)
from tools.bisect.orchestration.coordinator import BisectCoordinator
from tools.bisect.orchestration.errors import BisectFatalError
from tools.bisect.testing.definition import TestEntry

logger = logging.getLogger("auto_bisect")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Decode CLI inputs and reject invalid combinations before preparing environments."""
    p = argparse.ArgumentParser(
        description="Bisect a command entry or nightly case across prepared commit environments."
    )
    p.add_argument("--scene", required=True, choices=list(SCENES))
    entry = p.add_mutually_exclusive_group(required=True)
    entry.add_argument("--test-entry", type=Path, help="JSON file containing the fixed test command and environment")
    entry.add_argument(
        "--config-yaml",
        help="CONFIG_YAML_PATH of the failing case file; the whole file (all test_cases) is run each trial",
    )
    p.add_argument("--execution-file", type=Path, help="Explicit participants and existing execution channels")
    p.add_argument(
        "--name",
        default=None,
        help="case name to match the good-table 'name' column (optional; falls back to matching by yaml/path)",
    )
    p.add_argument(
        "--soc",
        required=True,
        type=str.strip,
        help="hardware family used to select the build chain and nightly good-table row",
    )
    p.add_argument("--bad-commit", default=os.getenv("VLLM_ASCEND_REF", "HEAD"))
    p.add_argument("--good-commit", default=None, help="override; else read from good table")
    p.add_argument("--config-base-path", default=os.getenv("CONFIG_BASE_PATH"))
    p.add_argument("--good-table", default=DEFAULT_GOOD_TABLE)
    p.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    p.add_argument("--repo-dir", default=str(REPO_ROOT))
    p.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="cluster node count; if omitted for multi-node it is read from the config yaml's 'num_nodes' field",
    )
    p.add_argument(
        "--node-index",
        type=int,
        default=None,
        help="this node's index; shared-directory workers default to $LWS_WORKER_INDEX, otherwise 0",
    )
    p.add_argument("--coord-dir", default=DEFAULT_COORD_DIR, help="shared barrier dir (multi-node)")
    p.add_argument("--run-scoped-coord", action="store_true", help="coord and release paths are unique to this run")
    p.add_argument(
        "--release-file",
        default=None,
        help="multi-node worker: also exit when this file appears (the AOP "
        "leader's 'done' file), so workers don't hang if the leader skips",
    )
    p.add_argument(
        "--barrier-timeout-s",
        type=float,
        default=3600.0,
        help="multi-node: how long the master waits for all workers to "
        "signal ready each round before failing (default 3600). Lower "
        "it (e.g. 300) to fail fast when debugging worker join issues",
    )
    p.add_argument("--fail-confirm-retries", type=int, default=1)
    p.add_argument("--no-verify-good", action="store_true")
    p.add_argument("--no-verify-bad", action="store_true")
    p.add_argument(
        "--mock-tests",
        action="store_true",
        help=(
            "prepare each tested candidate's real environment, but simulate PASS/FAIL without running the test command"
        ),
    )
    p.add_argument("--trial-timeout-s", type=float, default=7200.0)
    p.add_argument(
        "--force-initial-build",
        action="store_true",
        help="replay the target base image and full build chain before the first actual trial",
    )
    p.add_argument(
        "--no-assume-built-head",
        action="store_true",
        help="initialize from the target base image and full build chain instead of trusting HEAD",
    )
    p.add_argument(
        "--native-check",
        choices=["per-commit", "since-build"],
        default="per-commit",
        help="'per-commit' (default): decide rebuild from each bisected "
        "commit's own diff (compile iff it touched native/cpp files); "
        "'since-build': from all changes since the last build",
    )
    args = p.parse_args(argv)
    try:
        normalize_build_soc(args.soc)
    except ValueError as exc:
        p.error(str(exc))
    if args.node_index is None:
        if args.scene == SCENE_MULTI and not args.execution_file:
            try:
                args.node_index = int(os.getenv("LWS_WORKER_INDEX", "0"))
            except ValueError:
                p.error("LWS_WORKER_INDEX must be an integer for shared-directory workers")
        else:
            args.node_index = 0
    if args.execution_file and not args.test_entry:
        p.error("--execution-file requires --test-entry")
    if args.execution_file and args.node_index != 0:
        p.error("--execution-file uses one controller; do not start shared-directory workers")
    if args.test_entry and not args.good_commit:
        p.error("--test-entry requires --good-commit; nightly good-table lookup does not apply")
    if args.num_nodes is not None and args.num_nodes < 1:
        p.error("--num-nodes must be positive")
    if args.node_index < 0:
        p.error("--node-index must be nonnegative")
    if args.fail_confirm_retries < 0:
        p.error("--fail-confirm-retries must be nonnegative")
    for name in ("trial_timeout_s", "barrier_timeout_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    if args.mock_tests and args.scene == SCENE_MULTI and not args.execution_file:
        raise SystemExit("Multi-node --mock-tests requires --execution-file")
    try:
        test_entry = TestEntry.load(args.test_entry) if args.test_entry else None
        execution_targets = ExecutionTargets.load(args.execution_file) if args.execution_file else None
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid test entry: {exc}") from exc
    repo_dir = Path(args.repo_dir)
    try:
        num_nodes = resolve_node_count(args.scene, args.num_nodes, execution_targets)
        if num_nodes is None:
            if test_entry is not None:
                raise ValueError("--test-entry multi-node execution requires explicit --num-nodes")
            from tools.bisect.testing.nightly import resolve_nightly_node_count

            nightly_count = resolve_nightly_node_count(repo_dir, args.config_yaml, args.config_base_path)
            num_nodes = resolve_node_count(args.scene, nightly_count)
        if not 0 <= args.node_index < num_nodes:
            raise ValueError("--node-index must be within the configured participant count")
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    inp = BisectInput(
        scene=args.scene,
        config_yaml=args.config_yaml or "",
        test_entry=test_entry,
        name=args.name,
        soc=args.soc,
        bad_commit=args.bad_commit,
        config_base_path=args.config_base_path,
        good_commit=args.good_commit,
    )
    opt = BisectOptions(
        execution_targets=execution_targets,
        repo_dir=repo_dir,
        soc=args.soc,
        mock_tests=args.mock_tests,
        work_dir=args.work_dir,
        coord_dir=args.coord_dir,
        good_table_path=args.good_table,
        fail_confirm_retries=args.fail_confirm_retries,
        verify_good=not args.no_verify_good,
        verify_bad=not args.no_verify_bad,
        num_nodes=num_nodes,
        node_index=args.node_index,
        release_file=args.release_file,
        run_scoped_coord=args.run_scoped_coord,
        barrier_timeout_s=args.barrier_timeout_s,
        trial_timeout_s=args.trial_timeout_s,
        assume_built_head=not args.no_assume_built_head,
        force_initial_build=args.force_initial_build,
        native_check=args.native_check,
    )

    # On a multi-node worker, drive the worker loop instead of the search.
    if execution_targets is None and inp.scene == SCENE_MULTI and not opt.is_master:
        from tools.bisect.orchestration.shared_worker import run_worker

        return run_worker(inp, opt)

    try:
        first_bad = BisectCoordinator(inp, opt).run()
    except BisectFatalError as exc:
        logger.error("Bisect aborted: %s", exc)
        return 2
    return 0 if first_bad is not None else 2
