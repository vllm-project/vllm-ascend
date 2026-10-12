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

* single-node -> ``pytest test_single_node.py`` over the whole CONFIG_YAML_PATH
  file (all test_cases; nightly cannot select a single case).
* multi-node  -> ``pytest test_multi_node.py`` after sourcing the Ascend env,
  coordinated across nodes by ``Coordinator``.

The orchestrator only calls ``validate()`` and ``teardown()``; it does not know
which scene it is driving.
"""

import contextlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

import psutil  # type: ignore[import-untyped]

from tools.bisect import git_ops
from tools.bisect.build_manager import DEPLOY_ERRORS, BuildManager
from tools.bisect.config import (
    MULTI_NODE_RUN_SH,
    SCENE_SINGLE,
    SINGLE_NODE_TEST_PATH,
    BisectInput,
    BisectOptions,
    Candidate,
)
from tools.bisect.coordinator import Coordinator
from tools.bisect.verdict import RunOutcome
from tools.bisect.version_compat import VersionAdaptationError, VersionAdapter, VersionPolicy

logger = logging.getLogger(__name__)

# Common entries for current configs. The legacy entries let bisect test
# commits created before the framework moved under tests/e2e/common or before
# the common multi-node dispatcher existed.
_LEGACY_SINGLE_NODE_TEST = "tests/e2e/nightly/single_node/models/scripts/test_single_node.py"
_LEGACY_SINGLE_NODE_CONFIG_DIR = "tests/e2e/nightly/single_node/models/configs"
_MULTI_NODE_TEST = "tests/e2e/common/multi_node/test_multi_node.py"
_LEGACY_MULTI_NODE_TEST = "tests/e2e/nightly/multi_node/scripts/test_multi_node.py"
_INTERNAL_DP_TEST = "tests/e2e/nightly/multi_node/internal_dp/scripts/test_multi_node.py"
_EXTERNAL_DP_TEST = "tests/e2e/nightly/multi_node/external_dp/scripts/test_external_dp.py"
_EXTERNAL_DP_CONFIG_DIR = "tests/e2e/nightly/multi_node/external_dp/config"

# Ascend toolkit env files sourced before launching multi-node pytest.
_ENV_SOURCE_FILES = (
    "/usr/local/Ascend/ascend-toolkit/set_env.sh",
    "/usr/local/Ascend/nnal/atb/set_env.sh",
)


def _single_node_test_path(repo: Path) -> str:
    """Select the current entry, with a fallback for pre-migration commits."""
    if (repo / SINGLE_NODE_TEST_PATH).is_file():
        return SINGLE_NODE_TEST_PATH
    return _LEGACY_SINGLE_NODE_TEST


def _multi_node_test_path(repo: Path, inp: BisectInput) -> str:
    """Select the common dispatcher, with a fallback for legacy commits."""
    if (repo / _MULTI_NODE_TEST).is_file():
        return _MULTI_NODE_TEST
    if (repo / _LEGACY_MULTI_NODE_TEST).is_file():
        return _LEGACY_MULTI_NODE_TEST

    base = inp.config_base_path or ""
    config_name = Path(inp.config_yaml).name
    legacy_external_config = repo / _EXTERNAL_DP_CONFIG_DIR / config_name
    if "external_dp/config" in base or "external_dp/config" in inp.config_yaml or legacy_external_config.is_file():
        return _EXTERNAL_DP_TEST
    return _INTERNAL_DP_TEST


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


class BisectFatalError(RuntimeError):
    """Unrecoverable environment problem; abort the whole search.

    Distinct from per-trial failures (which become SKIP for one commit): raised
    when a multi-node barrier times out and no worker node has ever signalled
    ready -- continuing would burn the full barrier timeout on every remaining
    round, so the bisect aborts with exit code 2 instead.
    """


class BaseRunner:
    def __init__(self, inp: BisectInput, opt: BisectOptions, builder: BuildManager):
        self.inp = inp
        self.opt = opt
        self.builder = builder
        self.repo = opt.repo_dir
        self.version_adapter = VersionAdapter(opt)
        self.version_policy = VersionPolicy()

    def configure_version_policy(self, policy: VersionPolicy) -> None:
        self.version_policy = policy

    def _ensure_versions(self, commit: str, log_path: Path) -> None:
        if not self.version_policy.enabled:
            return
        self.version_adapter.ensure_at_commit(self.repo, commit, self.version_policy, log_path)

    def _base_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["CONFIG_YAML_PATH"] = self.inp.config_yaml
        config_base_path = self.inp.config_base_path
        if self.inp.scene == SCENE_SINGLE and config_base_path:
            requested = self.repo / config_base_path / self.inp.config_yaml
            legacy_base = self.repo / _LEGACY_SINGLE_NODE_CONFIG_DIR
            legacy_config = legacy_base / Path(self.inp.config_yaml).name
            if not requested.is_file() and legacy_config.is_file():
                config_base_path = str(legacy_base)
                env["CONFIG_YAML_PATH"] = legacy_config.name
                logger.info("[single] config moved in candidate; using %s", legacy_config)
        if config_base_path:
            env["CONFIG_BASE_PATH"] = config_base_path
        return env

    @staticmethod
    def _reset_dir(path: Path) -> Path:
        """Empty ``path`` so a trial only sees its own benchmark outputs."""
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return path

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
        self._ensure_versions(candidate.commit, log_path)

        # Run the WHOLE yaml (all test_cases): nightly cannot select a single
        # case, so we don't pass -k. Each case writes its own benchmark JSON
        # under benchmark_results/; reset the dir first so the verdict only sees
        # this trial's files.
        results_dir = self._reset_dir(self.repo / "benchmark_results")
        env = self._base_env()
        cmd = ["python", "-m", "pytest", "-sv", "--show-capture=no", _single_node_test_path(self.repo)]

        rc = self._run_pytest(cmd, env, log_path)
        outcome = RunOutcome(exit_code=rc, results_dir=results_dir if results_dir.exists() else None)
        outcome.rebuilt = decision.rebuild  # type: ignore[attr-defined]
        return outcome

    def _run_pytest(self, cmd: list[str], env: dict[str, str], log_path: Path) -> int:
        logger.info("[single] running: %s", " ".join(cmd))
        with open(log_path, "a", encoding="utf-8") as out:
            out.write(f"\n$ {' '.join(cmd)}\n")
            out.flush()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(self.repo),
                    env=env,
                    stdout=out,
                    stderr=subprocess.STDOUT,
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

    def _abort_round(self, round_idx: int) -> None:
        """Publish the round's SKIP verdict so workers leave ``wait_start``
        promptly instead of sitting out the full barrier timeout."""
        self.coord.publish_verdict(round_idx, "SKIP")

    def validate(self, candidate: Candidate, round_idx: int, log_dir: Path) -> RunOutcome:
        log_path = log_dir / f"round{round_idx}_{candidate.short}.log"
        # Pre-publish deploy decision: failing here would mean the round's
        # command never appears, stranding the worker in wait_command. Publish
        # a SKIP command so both sides stay in lockstep instead.
        try:
            decision = self.builder.decide(candidate.commit)
            version_targets = self.version_adapter.targets_at(self.repo, candidate.commit, self.version_policy)
        except DEPLOY_ERRORS:
            self.coord.publish_command(round_idx, candidate.commit, rebuild=False, action="SKIP")
            self._abort_round(round_idx)
            raise
        self.coord.publish_command(
            round_idx,
            candidate.commit,
            decision.rebuild,
            action="RUN",
            version_targets=version_targets,
            version_checks=self.version_policy.checked_packages,
        )
        try:
            self.builder.prepare(candidate.commit, log_path)
            self.version_adapter.ensure_targets(
                version_targets,
                self.version_policy.checked_packages,
                log_path,
            )
        except DEPLOY_ERRORS:
            self._abort_round(round_idx)
            raise

        # 3) barrier: every node deployed the same commit before any test starts
        self.coord.signal_ready(round_idx, git_ops.current_commit(self.repo))
        try:
            self.coord.wait_all_ready(round_idx, candidate.commit, self.opt.barrier_timeout_s)
        except (TimeoutError, RuntimeError) as exc:
            self._abort_round(round_idx)
            # No worker EVER signalled ready (any round): they are gone or never
            # joined. SKIP-scanning the remaining range would burn the full
            # barrier timeout per round for nothing -> abort the whole search.
            # A worker that only joined late (its marker landed in an earlier
            # round) still counts as present and keeps the per-round SKIP.
            # (RuntimeError = split-commit / deliberately failed marker: a
            # worker is provably present, so that stays a per-round SKIP.)
            if isinstance(exc, TimeoutError) and not self.coord.any_foreign_ready():
                raise BisectFatalError(
                    f"Barrier timeout on round {round_idx} and no worker node has "
                    "ever signalled ready (in any round); the bisect cannot "
                    "proceed without all nodes. Ensure 'auto_bisect.py --scene "
                    "multi_node' runs on EVERY node against the same --coord-dir, "
                    "or raise --barrier-timeout-s if worker builds are slow."
                ) from exc
            raise VersionAdaptationError(f"Barrier failed: {exc}") from exc
        self.coord.publish_start(round_idx)

        # 4) launch the multi-node pytest on master and read the verdict
        job = _safe_name(self.inp.config_yaml)
        results_dir = self._reset_dir(Path("/root/.cache/benchmark_results") / job)
        rc = self._run_multi_pytest(log_path, job)
        outcome = RunOutcome(exit_code=rc, results_dir=results_dir if results_dir.exists() else None)
        outcome.rebuilt = decision.rebuild  # type: ignore[attr-defined]
        return outcome

    def finish(self) -> None:
        self.coord.publish_done()

    def _test_path(self) -> str:
        return _multi_node_test_path(self.repo, self.inp)

    def _run_multi_pytest(self, log_path: Path, job: str) -> int:
        env = self._base_env()
        env["BENCHMARK_JOB_NAME"] = job
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
                    ["bash", "-c", bash_cmd],
                    cwd=str(self.repo),
                    env=env,
                    stdout=out,
                    stderr=subprocess.STDOUT,
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
        if opt.is_master:
            # Clear stale DONE/round_* from a previous run before driving (the
            # coord dir is usually on a persistent PVC). Safe: workers only write
            # ready files after round 1 is published, which happens after this.
            coord.reset()
        return MultiNodeRunner(inp, opt, builder, coord)
    return SingleNodeRunner(inp, opt, builder)


# Keep MULTI_NODE_RUN_SH referenced for docs/tools that grep for the entry.
_NIGHTLY_ENTRIES = (SINGLE_NODE_TEST_PATH, MULTI_NODE_RUN_SH)
