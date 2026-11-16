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
"""Non-master node agent for multi-node bisect.

Mirrors the master's deploy sequence round-by-round: wait for the command,
checkout+build the requested commit, report ready (barrier), then run the
non-master ``test_multi_node`` (which ``hang_until_terminated`` on the master's
health endpoint and returns when the master's trial ends). Loops until the
master publishes the DONE sentinel.

This module is invoked automatically by ``auto_bisect.main`` when it detects a
multi-node, non-master node (``LWS_WORKER_INDEX != 0``).
"""

import logging
import time
from pathlib import Path

from tools.bisect.deployment import git_ops
from tools.bisect.deployment.manager import DEPLOY_ERRORS, BuildManager
from tools.bisect.execution.shared_directory import Coordinator
from tools.bisect.orchestration.config import BisectInput, BisectOptions
from tools.bisect.testing.nightly import kill_stray_servers, run_worker_test

logger = logging.getLogger("bisect.worker")


def _await_start(coord: Coordinator, rnd: int, opt: BisectOptions) -> bool:
    """Wait for the master's start/abort decision for round ``rnd``.

    A timeout is survivable (the master may still be rebuilding, or the round
    was abandoned after a desync): return False so the caller moves on to the
    next round instead of dying -- a dead agent leaves the master waiting out
    the barrier timeout on every remaining round.
    """
    try:
        return coord.wait_start(rnd, opt.barrier_timeout_s)
    except TimeoutError:
        logger.error("[worker] round %d: no start/abort decision within %ss; continuing", rnd, opt.barrier_timeout_s)
        return False


def run_worker(inp: BisectInput, opt: BisectOptions) -> int:
    coord = Coordinator(opt.coord_dir, opt.num_nodes, opt.node_index)
    builder = BuildManager(opt)
    log_dir = Path(opt.work_dir) / "worker_logs" / f"node{opt.node_index}"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[worker] node %d started; waiting for master commands", opt.node_index)
    # Record the legacy cutoff before fetching: completion may arrive while
    # history is downloading. A run-scoped directory needs no mtime filter.
    start_ts = None if opt.run_scoped_coord else time.time()
    # Recover the full history up front: the nightly clone is depth-1 and only
    # the tip exists locally, while every commanded commit is an ancestor.
    # Doing this before the first round keeps the slow, network-bound unshallow
    # out of the barrier window (prepare() re-resolves as a no-op backstop).
    git_ops.ensure_full_history(opt.repo_dir)
    rnd = 0
    while True:
        rnd += 1
        cmd = coord.wait_command(rnd, opt.barrier_timeout_s, release_file=opt.release_file, since_ts=start_ts)
        if cmd is None:
            logger.info("[worker] stop signal received; exiting after %d rounds", rnd - 1)
            return 0

        commit = cmd["commit"]
        # A SKIP command (e.g. vLLM mismatch decided by the leader) is consumed
        # to keep rounds in lockstep, but the worker neither deploys nor runs.
        if cmd.get("action") == "SKIP":
            logger.info("[worker] round %d: SKIP %s (no deploy/run)", rnd, commit[:12])
            continue

        log_path = log_dir / f"round{rnd}_{commit[:12]}.log"
        logger.info("[worker] round %d: deploying %s", rnd, commit[:12])
        try:
            resolved_commit = git_ops.resolve_commit(opt.repo_dir, commit)
            builder.decide(resolved_commit)
            builder.prepare(resolved_commit, log_path)
        except DEPLOY_ERRORS as exc:
            builder.invalidate_baseline()
            # Publish a deliberately inconsistent ready marker so the master
            # aborts the round if it managed to deploy successfully. The worker
            # waits for the master's start/abort decision and never launches a
            # partial distributed test. GitError is included: the worker's repo
            # is often a shallow clone and an unrecoverable commit must not
            # kill the agent (a dead agent leaves the master waiting out the
            # barrier timeout on every remaining round).
            logger.error("[worker] deploy failed for %s: %s", commit[:12], exc)
            coord.signal_ready(rnd, "worker-deploy-failed")
            _await_start(coord, rnd, opt)
            continue

        coord.signal_ready(
            rnd,
            git_ops.current_commit(opt.repo_dir),
            inp.test_entry.identity if inp.test_entry else None,
        )
        if not _await_start(coord, rnd, opt):
            logger.info("[worker] round %d aborted before test execution", rnd)
            continue
        # Launch the worker test; it returns when the master's trial completes.
        rc = run_worker_test(inp, opt, log_path, builder.test_environment)
        logger.info("[worker] round %d pytest finished rc=%d", rnd, rc)
        if inp.test_entry is None:
            kill_stray_servers()
        verdict = None
        if inp.test_entry is not None:
            verdict = "SKIP" if rc == 125 else ("PASS" if rc == 0 else "FAIL")
        coord.signal_finished(rnd, verdict)
