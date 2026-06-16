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
"""Launch abstraction: deploy a commit and run the failing case.

Both runners reuse the *existing* nightly entries so the bisect reproduces the
real nightly environment:

* single-node -> ``pytest test_single_node.py -k <case>`` (CONFIG_YAML_PATH).
* multi-node  -> ``pytest test_multi_node.py`` after sourcing the Ascend env,
  coordinated across nodes by ``Coordinator``.

The orchestrator only calls ``validate()`` and ``teardown()``; it does not know
which scene it is driving.
"""

import contextlib
import logging
import os
import subprocess
from pathlib import Path

import psutil

from tests.e2e.nightly.bisect import git_ops
from tests.e2e.nightly.bisect.build_manager import BuildError, BuildManager
from tests.e2e.nightly.bisect.config import (
    MULTI_NODE_RUN_SH,
    SINGLE_NODE_TEST_PATH,
    BisectInput,
    BisectOptions,
    Candidate,
)
from tests.e2e.nightly.bisect.coordinator import Coordinator
from tests.e2e.nightly.bisect.verdict import RunOutcome

logger = logging.getLogger(__name__)

# Internal vs external DP pytest entries (mirrors run.sh selection logic).
_INTERNAL_DP_TEST = "tests/e2e/nightly/multi_node/internal_dp/scripts/test_multi_node.py"
_EXTERNAL_DP_TEST = "tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py"

# Ascend toolkit env files sourced before launching multi-node pytest.
_ENV_SOURCE_FILES = (
    "/usr/local/Ascend/ascend-toolkit/set_env.sh",
    "/usr/local/Ascend/nnal/atb/set_env.sh",
)


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def kill_stray_servers() -> None:
    """Kill leftover vLLM server processes *outside our own process tree*.

    Deliberately narrower than run.sh's ``pgrep python3 | kill`` so we never
    kill the long-lived bisect driver itself.
    """
    own = {os.getpid()}
    with contextlib.suppress(psutil.Error):
        own |= {c.pid for c in psutil.Process(os.getpid()).children(recursive=True)}
    for proc in psutil.process_iter(attrs=["pid", "cmdline"], ad_value=None):
        if proc.info["pid"] in own:
            continue
        cmd = proc.info["cmdline"] or []
        joined = " ".join(cmd)
        if ("vllm" in joined and "serve" in joined) or "EngineCore" in joined or "VLLM" in joined:
            try:
                proc.kill()
                logger.info("[teardown] killed stray server pid=%s", proc.info["pid"])
            except psutil.Error:
                pass


class BaseRunner:
    def __init__(self, inp: BisectInput, opt: BisectOptions, builder: BuildManager):
        self.inp = inp
        self.opt = opt
        self.builder = builder
        self.repo = opt.repo_dir

    def _base_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["CONFIG_YAML_PATH"] = self.inp.config_yaml
        if self.inp.config_base_path:
            env["CONFIG_BASE_PATH"] = self.inp.config_base_path
        env["BENCHMARK_JOB_NAME"] = _safe_name(self.inp.case_name or self.inp.config_yaml)
        return env

    def validate(self, candidate: Candidate, round_idx: int, log_dir: Path) -> RunOutcome:
        raise NotImplementedError

    def teardown(self) -> None:
        kill_stray_servers()

    def finish(self) -> None:  # multi-node overrides to release workers
        pass


class SingleNodeRunner(BaseRunner):
    def validate(self, candidate: Candidate, round_idx: int, log_dir: Path) -> RunOutcome:
        log_path = log_dir / f"round{round_idx}_{candidate.short}.log"
        decision = self.builder.prepare(candidate.commit, log_path)  # may raise BuildError

        env = self._base_env()
        cmd = [
            "python", "-m", "pytest", "-sv", "--show-capture=no", SINGLE_NODE_TEST_PATH,
        ]
        if self.inp.case_name:
            cmd += ["-k", self.inp.case_name]

        rc = self._run_pytest(cmd, env, log_path)
        job = env["BENCHMARK_JOB_NAME"]
        results_json = self.repo / "benchmark_results" / f"{job}.json"
        outcome = RunOutcome(exit_code=rc, results_json=results_json if results_json.exists() else None)
        outcome.rebuilt = decision.rebuild  # type: ignore[attr-defined]
        return outcome

    def _run_pytest(self, cmd: list[str], env: dict[str, str], log_path: Path) -> int:
        logger.info("[single] running: %s", " ".join(cmd))
        with open(log_path, "a", encoding="utf-8") as out:
            out.write(f"\n$ {' '.join(cmd)}\n")
            out.flush()
            try:
                proc = subprocess.run(
                    cmd, cwd=str(self.repo), env=env,
                    stdout=out, stderr=subprocess.STDOUT,
                    timeout=self.opt.trial_timeout_s,
                )
                return proc.returncode
            except subprocess.TimeoutExpired:
                logger.error("[single] trial timed out after %ss", self.opt.trial_timeout_s)
                return 124


class MultiNodeRunner(BaseRunner):
    """Master-side runner. Worker nodes run ``worker_agent.py`` instead."""

    def __init__(self, inp, opt, builder, coordinator: Coordinator):
        super().__init__(inp, opt, builder)
        self.coord = coordinator

    def validate(self, candidate: Candidate, round_idx: int, log_dir: Path) -> RunOutcome:
        log_path = log_dir / f"round{round_idx}_{candidate.short}.log"

        # 1) tell every node which commit to deploy (rebuild flag included)
        decision = self.builder.decide(candidate.commit)
        self.coord.publish_command(round_idx, candidate.commit, decision.rebuild)

        # 2) deploy locally (master is also a node)
        try:
            self.builder.prepare(candidate.commit, log_path)
        except BuildError:
            self.coord.publish_verdict(round_idx, "SKIP")
            raise

        # 3) barrier: every node deployed the same commit before any test starts
        self.coord.signal_ready(round_idx, git_ops.current_commit(self.repo))
        self.coord.wait_all_ready(round_idx, candidate.commit, self.opt.barrier_timeout_s)

        # 4) launch the multi-node pytest on master and read the verdict
        rc = self._run_multi_pytest(log_path)
        job = self._base_env()["BENCHMARK_JOB_NAME"]
        results_json = Path("/root/.cache/benchmark_results") / job / f"{job}.json"
        outcome = RunOutcome(exit_code=rc, results_json=results_json if results_json.exists() else None)
        outcome.rebuilt = decision.rebuild  # type: ignore[attr-defined]
        return outcome

    def finish(self) -> None:
        self.coord.publish_done()

    def _test_path(self) -> str:
        base = self.inp.config_base_path or ""
        if "external_dp/config" in base or "external_dp/config" in self.inp.config_yaml:
            return _EXTERNAL_DP_TEST
        return _INTERNAL_DP_TEST

    def _run_multi_pytest(self, log_path: Path) -> int:
        env = self._base_env()
        env.setdefault("BENCHMARK_HOME", str(self.repo / "benchmark"))
        env.setdefault("LWS_WORKER_INDEX", str(self.opt.node_index))
        # Source the Ascend toolkit env then exec pytest (mirrors run.sh, but
        # without its broad python3 kill which would take down this driver).
        sources = " ; ".join(f"source {f} 2>/dev/null || true" for f in _ENV_SOURCE_FILES)
        pytest_cmd = f"python -m pytest -sv --show-capture=no {self._test_path()}"
        bash_cmd = f"set -e ; {sources} ; exec {pytest_cmd}"
        logger.info("[multi] running: %s", pytest_cmd)
        with open(log_path, "a", encoding="utf-8") as out:
            out.write(f"\n$ {bash_cmd}\n")
            out.flush()
            try:
                proc = subprocess.run(
                    ["bash", "-c", bash_cmd], cwd=str(self.repo), env=env,
                    stdout=out, stderr=subprocess.STDOUT,
                    timeout=self.opt.trial_timeout_s,
                )
                return proc.returncode
            except subprocess.TimeoutExpired:
                logger.error("[multi] trial timed out after %ss", self.opt.trial_timeout_s)
                return 124


def build_runner(inp: BisectInput, opt: BisectOptions, builder: BuildManager):
    """Factory: pick the runner for the scene."""
    if inp.scene == "multi_node":
        coord = Coordinator(opt.coord_dir, opt.num_nodes, opt.node_index)
        return MultiNodeRunner(inp, opt, builder, coord)
    return SingleNodeRunner(inp, opt, builder)


# Keep MULTI_NODE_RUN_SH referenced for docs/tools that grep for the entry.
_NIGHTLY_ENTRIES = (SINGLE_NODE_TEST_PATH, MULTI_NODE_RUN_SH)
