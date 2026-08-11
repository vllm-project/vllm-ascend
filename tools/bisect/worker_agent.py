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
import os
import subprocess
import time
from pathlib import Path

from tools.bisect import git_ops, runner
from tools.bisect.build_manager import BuildError, BuildManager
from tools.bisect.config import BisectInput, BisectOptions
from tools.bisect.coordinator import Coordinator

logger = logging.getLogger("bisect.worker")


def _launch_pytest(inp: BisectInput, opt: BisectOptions, log_path: Path) -> int:
    """Run the non-master multi-node pytest (same env recipe as the master)."""
    env = dict(os.environ)
    env["CONFIG_YAML_PATH"] = inp.config_yaml
    if inp.config_base_path:
        env["CONFIG_BASE_PATH"] = inp.config_base_path
    env["BENCHMARK_JOB_NAME"] = runner._safe_name(inp.config_yaml)
    env.setdefault("BENCHMARK_HOME", str(opt.repo_dir / "benchmark"))
    env["LWS_WORKER_INDEX"] = str(opt.node_index)

    base = inp.config_base_path or ""
    test_path = (
        runner._EXTERNAL_DP_TEST
        if "external_dp/config" in base or "external_dp/config" in inp.config_yaml
        else runner._INTERNAL_DP_TEST
    )
    sources = " ; ".join(f"source {f} 2>/dev/null || true" for f in runner._ENV_SOURCE_FILES)
    bash_cmd = f"set -e ; {sources} ; exec python -m pytest -sv --show-capture=no {test_path}"
    with open(log_path, "a", encoding="utf-8") as out:
        out.write(f"\n$ {bash_cmd}\n")
        out.flush()
        try:
            proc = subprocess.run(
                ["bash", "-c", bash_cmd],
                cwd=str(opt.repo_dir),
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=opt.trial_timeout_s,
            )
            return proc.returncode
        except subprocess.TimeoutExpired:
            logger.error("[worker] pytest timed out after %ss", opt.trial_timeout_s)
            return 124


def run_worker(inp: BisectInput, opt: BisectOptions) -> int:
    coord = Coordinator(opt.coord_dir, opt.num_nodes, opt.node_index)
    builder = BuildManager(opt)
    log_dir = Path(opt.work_dir) / "worker_logs" / f"node{opt.node_index}"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[worker] node %d started; waiting for master commands", opt.node_index)
    # Only DONE/release files created AFTER this moment count as stop signals, so
    # a stale sentinel from a previous run on the persistent PVC is ignored.
    start_ts = time.time()
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
            builder.prepare(commit, log_path)
        except BuildError as exc:
            # Don't block the barrier; report our (built) HEAD so the master can
            # detect inconsistency. The master will likely hit the same build
            # failure and record SKIP.
            logger.error("[worker] build failed for %s: %s", commit[:12], exc)

        coord.signal_ready(rnd, git_ops.current_commit(opt.repo_dir))
        # Launch the worker test; it returns when the master's trial completes.
        rc = _launch_pytest(inp, opt, log_path)
        logger.info("[worker] round %d pytest finished rc=%d", rnd, rc)
        runner.kill_stray_servers()
