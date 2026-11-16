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
"""Coordinate preparation and opaque test execution around the bisect search."""

import logging
import time
from pathlib import Path

from tools.bisect.algorithm.search import BisectSearch
from tools.bisect.deployment import git_ops
from tools.bisect.deployment.manager import DEPLOY_ERRORS, BuildManager, BuildTransitionError
from tools.bisect.execution.shared_directory import CoordinationError
from tools.bisect.orchestration import report
from tools.bisect.orchestration.config import (
    SCENE_MULTI,
    BisectInput,
    BisectOptions,
    Candidate,
    TrialResult,
    Verdict,
)
from tools.bisect.orchestration.errors import BisectFatalError
from tools.bisect.orchestration.state import BisectState
from tools.bisect.testing.executor import evaluate_test
from tools.bisect.testing.mock import MockTestResults
from tools.bisect.testing.nightly import resolve_good
from tools.bisect.testing.nightly_verdict import evaluate

logger = logging.getLogger("auto_bisect")


class BisectCoordinator:
    def __init__(self, inp: BisectInput, opt: BisectOptions):
        self.inp = inp
        self.opt = opt
        self.repo = opt.repo_dir
        self.trials: list[TrialResult] = []
        self._mock_results: MockTestResults | None = None
        # Monotonic deploy counter. Each actual deploy (endpoint check, bisect
        # step, or flaky-confirm retry) consumes exactly one round so that
        # multi-node workers can mirror the sequence in lockstep.
        self._round = 0
        # Verdict caching speeds up single-node resume; disabled for multi-node
        # because a cache hit would skip a deploy and desync the workers.
        self.use_cache = (
            inp.scene != SCENE_MULTI
            and not opt.mock_tests
            and opt.execution_targets is None
            and opt.assume_built_head
            and not opt.force_initial_build
        )

        run_id = inp.case_key.replace("::", "__").replace("/", "_")
        if opt.execution_targets is not None:
            run_id += "__targets-" + opt.execution_targets.identity
        self.work_dir = Path(opt.work_dir) / (f"{run_id}__mock" if opt.mock_tests else run_id)
        self.log_dir = self.work_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work_dir / "state.json"
        self.report_path = self.work_dir / "report.json"
        self.builder = None
        if opt.execution_targets is not None:
            from tools.bisect.orchestration.target_trial import TargetTrialRunner

            self.runner = TargetTrialRunner(inp, opt)
        else:
            self.builder = BuildManager(opt)
            if inp.test_entry is not None and inp.scene != SCENE_MULTI:
                from tools.bisect.orchestration.local_trial import LocalCommandTrial

                self.runner = LocalCommandTrial(inp.test_entry, opt, self.builder)
            else:
                from tools.bisect.orchestration.nightly_trial import build_runner

                self.runner = build_runner(inp, opt, self.builder)

    # ------------------------------------------------------------ one trial
    def _run_trial(self, candidate: Candidate) -> TrialResult:
        start = time.time()
        self._round += 1
        round_idx = self._round
        log_path = self.log_dir / f"round{round_idx}_{candidate.short}.log"
        try:
            outcome = self.runner.validate(candidate, round_idx, self.log_dir)
            if self.opt.mock_tests:
                assert self._mock_results is not None
                v, note = self._mock_results.evaluate(candidate.commit)
                with open(log_path, "a", encoding="utf-8") as out:
                    out.write(f"[mock-tests] {note}\n")
            else:
                v, note = evaluate_test(outcome) if self.inp.test_entry else evaluate(outcome)
            result = TrialResult(
                candidate=candidate,
                verdict=v,
                duration_s=time.time() - start,
                rebuilt=getattr(outcome, "rebuilt", False),
                exit_code=None if self.opt.mock_tests else outcome.exit_code,
                log_path=str(log_path),
                note=note,
            )
        except BuildTransitionError as exc:
            raise BisectFatalError(
                f"Environment transition failed at {candidate.short}; log: {log_path}\n{exc}"
            ) from exc
        except (*DEPLOY_ERRORS, CoordinationError) as exc:
            logger.error("Candidate %s deployment failed: %s", candidate.short, exc)
            result = TrialResult(
                candidate=candidate,
                verdict="SKIP",
                duration_s=time.time() - start,
                rebuilt=True,
                log_path=str(log_path),
                note=f"build failed -> SKIP: {exc}".splitlines()[0],
            )
        finally:
            self.runner.teardown()
        return result

    def _judge(self, candidate: Candidate, state: BisectState) -> Verdict:
        """Run + flaky-confirm a candidate, caching the verdict for resume.

        Only a cached PASS/FAIL is reused on resume. A cached SKIP is never
        honoured: SKIP means "could not be judged" (build break, flaky, transient
        environment issue), which may no longer hold, so it must be re-attempted.
        """
        if self.use_cache:
            cached = state.verdicts.get(candidate.commit)
            if cached in ("PASS", "FAIL"):
                logger.info("Using cached verdict %s for %s", cached, candidate.short)
                return cached  # type: ignore[return-value]

        result = self._run_trial(candidate)

        # Flaky guard: re-confirm a FAIL; if a retry passes, the commit is
        # unreliable and must not be used as a bisect boundary -> SKIP.
        if result.verdict == "FAIL" and self.opt.fail_confirm_retries > 0 and not self.opt.mock_tests:
            for i in range(self.opt.fail_confirm_retries):
                logger.info("Confirming FAIL for %s (%d/%d)", candidate.short, i + 1, self.opt.fail_confirm_retries)
                retry = self._run_trial(candidate)
                if retry.verdict != "FAIL":
                    result.verdict = "SKIP"
                    result.note = f"flaky: first FAIL then {retry.verdict} -> SKIP"
                    break

        report.print_verdict(result)
        self.trials.append(result)
        state.verdicts[candidate.commit] = result.verdict
        state.save(self.state_path)
        return result.verdict

    # ------------------------------------------------------- endpoint checks
    def _verify_endpoints(self, good: Candidate, candidates: list[Candidate], state: BisectState) -> bool:
        bad = candidates[-1]
        if self.opt.verify_bad:
            v = self._judge(bad, state=state)
            report.print_endpoint_check("bad", bad, v, ok=v == "FAIL")
            if v == "SKIP":
                logger.error(
                    "Bad commit could not even run the test (environment error, "
                    "e.g. dependency adaptation failed). Fix the environment "
                    "before bisecting; aborting."
                )
                return False
            if v != "FAIL":
                logger.error("Bad commit did not reproduce the failure (%s); aborting.", v)
                return False
        else:
            state.verdicts[bad.commit] = "FAIL"

        if self.opt.verify_good:
            v = self._judge(good, state=state)
            report.print_endpoint_check("good", good, v, ok=v == "PASS")
            if v == "SKIP":
                logger.error(
                    "Good baseline could not even run the test (environment error, "
                    "e.g. dependency adaptation failed). The whole range is "
                    "likely unrunnable against the installed vllm; fix the "
                    "environment before bisecting; aborting."
                )
                return False
            if v != "PASS":
                logger.error("Good baseline is not actually good (%s); range invalid.", v)
                return False
        return True

    # --------------------------------------------------------------- search
    def _bisect(self, candidates: list[Candidate], state: BisectState) -> Candidate | None:
        lo = state.lo
        hi = state.hi
        # SKIP is tracked in-run only; never pre-seeded from a (possibly stale)
        # cache, so previously-skipped commits get a fresh attempt on resume.
        search = BisectSearch(lo, hi)

        while lo < hi:
            mid = search.next_index()
            if mid is None:
                logger.warning("Entire window [%d,%d) skipped; cannot narrow further.", lo, hi)
                return None
            report.print_progress(self._round + 1, lo, hi, candidates[mid])
            v = self._judge(candidates[mid], state)
            search.record(mid, v)
            lo, hi = search.lo, search.hi
            state.lo, state.hi, state.round_idx = lo, hi, self._round
            state.save(self.state_path)

        return candidates[lo] if lo < len(candidates) else None

    # ----------------------------------------------------------------- main
    def run(self) -> Candidate | None:
        try:
            bad = git_ops.describe(self.repo, self.inp.bad_commit)
            good_sha = self._resolve_good()
            good = git_ops.describe(self.repo, good_sha)
            logger.info("Bisecting %s: good=%s bad=%s", self.inp.case_key, good.short, bad.short)

            candidates = git_ops.candidate_list(self.repo, good.commit, bad.commit)
            logger.info("Search space: %d commits", len(candidates))
            if self.opt.mock_tests:
                self._mock_results = MockTestResults(tuple(candidate.commit for candidate in candidates))
                logger.warning(
                    "MOCK TESTS: preparing real environments without the test command; "
                    "synthetic FAIL starts at candidate %d. "
                    "No regression conclusion will be reported.",
                    self._mock_results.bad_index,
                )
            state = (
                None if self.opt.mock_tests else BisectState.load(self.state_path, good=good.commit, bad=bad.commit)
            ) or BisectState(good=good.commit, bad=bad.commit, hi=len(candidates) - 1)

            # finish() publishes the DONE sentinel that releases multi-node workers;
            # it must run on every exit path (including a barrier timeout or any
            # error) so workers never hang waiting for the next round.
            try:
                if not self._verify_endpoints(good, candidates, state):
                    self._write_report(good, bad, first_bad=None)
                    return None
                first_bad = self._bisect(candidates, state)
            except BisectFatalError:
                # Unrecoverable environment (e.g. no multi-node worker ever joined):
                # still leave a report so the run is auditable, then propagate so
                # main() can exit with code 2.
                self._write_report(good, bad, first_bad=None)
                raise
            if self.opt.mock_tests:
                failed_preparations = [trial for trial in self.trials if trial.verdict == "SKIP"]
                if failed_preparations:
                    logger.error("MOCK TESTS: %d candidate environment preparations failed", len(failed_preparations))
                else:
                    logger.info("MOCK TESTS: environment preparation completed; no model tests ran")
                self._write_report(good, bad, first_bad=None)
                return None if failed_preparations else first_bad

            if first_bad is not None:
                # If the culprit fell inside a skipped region it is only a *suspect*
                # (its parent is good, but it could not be judged). Surface that
                # ambiguity instead of presenting it as a confirmed first-bad.
                if state.verdicts.get(first_bad.commit) == "SKIP":
                    logger.warning(
                        "First-bad %s could not be judged (SKIP); it is the earliest "
                        "suspect but the true culprit may be it or a later skipped "
                        "commit up to the first confirmed FAIL. Inspect manually.",
                        first_bad.short,
                    )
                report.print_conclusion(first_bad, self.trials)
            self._write_report(good, bad, first_bad=first_bad)
            return first_bad
        finally:
            self.runner.finish()

    def _write_report(self, good: Candidate, bad: Candidate, first_bad: Candidate | None) -> None:
        """Write report.json (single writer for every exit path of run())."""
        report.write_report_json(
            self.report_path,
            inp=self.inp,
            good=good,
            bad=bad,
            first_bad=first_bad,
            trials=self.trials,
            mock_tests=self.opt.mock_tests,
        )

    def _resolve_good(self) -> str:
        return resolve_good(self.inp, self.opt)
