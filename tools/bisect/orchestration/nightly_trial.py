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

import logging
from pathlib import Path

from tools.bisect.deployment import git_ops
from tools.bisect.deployment.manager import DEPLOY_ERRORS, BuildManager
from tools.bisect.execution.shared_directory import CoordinationError, Coordinator
from tools.bisect.orchestration.config import (
    BisectInput,
    BisectOptions,
    Candidate,
)
from tools.bisect.orchestration.errors import BisectFatalError
from tools.bisect.testing.executor import evaluate_test, execute_test
from tools.bisect.testing.nightly import (
    _safe_name,
    kill_stray_servers,
    prepare_nightly_environment,
    reset_results_directory,
    run_multi_test,
    run_single_test,
)
from tools.bisect.testing.outcome import RunOutcome

logger = logging.getLogger(__name__)


class BaseRunner:
    def __init__(self, inp: BisectInput, opt: BisectOptions, builder: BuildManager):
        self.inp = inp
        self.opt = opt
        self.builder = builder
        self.repo = opt.repo_dir

    def _base_env(self) -> dict[str, str]:
        return prepare_nightly_environment(self.inp, self.repo, self.builder.test_environment)

    def validate(self, candidate: Candidate, round_idx: int, log_dir: Path) -> RunOutcome:
        raise NotImplementedError

    def teardown(self) -> None:
        if self.opt.mock_tests or self.inp.test_entry is not None:
            return
        kill_stray_servers()

    def finish(self) -> None:  # multi-node overrides to release workers
        pass


class SingleNodeRunner(BaseRunner):
    def validate(self, candidate: Candidate, round_idx: int, log_dir: Path) -> RunOutcome:
        log_path = log_dir / f"round{round_idx}_{candidate.short}.log"
        self.builder.decide(candidate.commit)  # fail before changing dependencies when the image cannot match
        try:
            decision = self.builder.prepare(candidate.commit, log_path)  # may raise BuildError
        except DEPLOY_ERRORS:
            self.builder.invalidate_baseline()
            raise

        if self.opt.mock_tests:
            with open(log_path, "a", encoding="utf-8") as out:
                out.write("\n[mock-tests] Environment prepared; pytest and model serving were not run.\n")
            outcome = RunOutcome(exit_code=0)
            outcome.rebuilt = decision.rebuild
            return outcome

        if self.inp.test_entry is not None:
            outcome = execute_test(self.inp.test_entry, self.repo, self.builder.test_environment, log_path)
            outcome.rebuilt = decision.rebuild
            return outcome

        # Run the WHOLE yaml (all test_cases): nightly cannot select a single
        # case, so we don't pass -k. Each case writes its own benchmark JSON
        # under benchmark_results/; reset the dir first so the verdict only sees
        # this trial's files.
        results_dir = reset_results_directory(self.repo / "benchmark_results")
        env = self._base_env()
        rc = run_single_test(self.repo, env, log_path, self.opt.trial_timeout_s)
        outcome = RunOutcome(exit_code=rc, results_dir=results_dir if results_dir.exists() else None)
        outcome.rebuilt = decision.rebuild
        return outcome


class MultiNodeRunner(BaseRunner):
    """Master-side runner. Worker nodes run ``orchestration/shared_worker.py`` instead."""

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
        except DEPLOY_ERRORS:
            self.coord.publish_command(round_idx, candidate.commit, rebuild=False, action="SKIP")
            self._abort_round(round_idx)
            raise
        self.coord.publish_command(
            round_idx,
            candidate.commit,
            decision.rebuild,
            action="RUN",
        )
        try:
            decision = self.builder.prepare(candidate.commit, log_path)
        except DEPLOY_ERRORS:
            self.builder.invalidate_baseline()
            self._abort_round(round_idx)
            raise

        # 3) barrier: every node deployed the same commit before any test starts
        self.coord.signal_ready(
            round_idx,
            git_ops.current_commit(self.repo),
            self.inp.test_entry.identity if self.inp.test_entry else None,
        )
        try:
            self.coord.wait_all_ready(round_idx, candidate.commit, self.opt.barrier_timeout_s)
            if self.inp.test_entry is not None:
                self.coord.verify_test_entry(round_idx, self.inp.test_entry.identity)
        except (TimeoutError, RuntimeError) as exc:
            self._abort_round(round_idx)
            if "worker-deploy-failed" in str(exc):
                raise BisectFatalError(
                    f"Worker environment transition failed on round {round_idx}; "
                    f"inspect worker_logs for the failed node: {exc}"
                ) from exc
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
            raise CoordinationError(f"Barrier failed: {exc}") from exc
        self.coord.publish_start(round_idx)

        # 4) launch the multi-node pytest on master and read the verdict
        job = _safe_name(self.inp.config_yaml)
        results_dir = (
            None
            if self.inp.test_entry is not None
            else reset_results_directory(Path("/root/.cache/benchmark_results") / job)
        )
        rc = self._run_multi_pytest(log_path, job)
        if self.inp.test_entry is None:
            kill_stray_servers()
        verdict = evaluate_test(self._command_outcome)[0] if self.inp.test_entry else None
        self.coord.signal_finished(round_idx, verdict)
        try:
            self.coord.wait_all_finished(
                round_idx,
                max(
                    self.opt.barrier_timeout_s,
                    self.opt.trial_timeout_s,
                    self.inp.test_entry.timeout_seconds if self.inp.test_entry else 0,
                ),
            )
        except TimeoutError as exc:
            self._abort_round(round_idx)
            raise CoordinationError(f"Round finish barrier failed: {exc}") from exc
        outcome = (
            self._command_outcome
            if self.inp.test_entry is not None and self.inp.scene == "multi_node"
            else RunOutcome(exit_code=rc, results_dir=results_dir if results_dir.exists() else None)
        )
        if self.inp.test_entry is not None:
            verdicts = self.coord.trial_verdicts(round_idx)
            if "SKIP" in verdicts:
                outcome = RunOutcome(125, infra_error=True, skip_reason="a participating node could not run the test")
            elif "FAIL" in verdicts:
                outcome = RunOutcome(1)
        outcome.rebuilt = decision.rebuild
        return outcome

    def finish(self) -> None:
        self.coord.publish_done()

    def _run_multi_pytest(self, log_path: Path, job: str) -> int:
        if self.inp.test_entry is not None:
            self._command_outcome = execute_test(
                self.inp.test_entry, self.repo, self.builder.test_environment, log_path
            )
            return self._command_outcome.exit_code
        return run_multi_test(self.inp, self.opt, self._base_env(), log_path, job)


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
