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
"""Auto-Bisect orchestrator: find the first bad commit for a nightly failure.

Usage (single-node)::

    python -m tests.e2e.nightly.bisect.auto_bisect \
        --scene single_node \
        --config-yaml DeepSeek-R1-0528-W8A8.yaml \
        --case-name DeepSeek-R1-0528-W8A8-aclgraph \
        --bad-commit HEAD

The good commit is read from the good table unless ``--good-commit`` is given.
"""

import argparse
import logging
import os
import time
from pathlib import Path

from tests.e2e.nightly.bisect import git_ops, report
from tests.e2e.nightly.bisect.build_manager import BuildError, BuildManager
from tests.e2e.nightly.bisect.config import (
    DEFAULT_COORD_DIR,
    DEFAULT_GOOD_TABLE,
    DEFAULT_WORK_DIR,
    REPO_ROOT,
    SCENE_MULTI,
    SCENES,
    BisectInput,
    BisectOptions,
    Candidate,
    TrialResult,
    Verdict,
)
from tests.e2e.nightly.bisect.good_table import GoodTable
from tests.e2e.nightly.bisect.runner import build_runner
from tests.e2e.nightly.bisect.state import BisectState
from tests.e2e.nightly.bisect.verdict import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("auto_bisect")


class Bisector:
    def __init__(self, inp: BisectInput, opt: BisectOptions):
        self.inp = inp
        self.opt = opt
        self.repo = opt.repo_dir
        self.builder = BuildManager(opt)
        self.runner = build_runner(inp, opt, self.builder)
        self.trials: list[TrialResult] = []
        # Monotonic deploy counter. Each actual deploy (endpoint check, bisect
        # step, or flaky-confirm retry) consumes exactly one round so that
        # multi-node workers can mirror the sequence in lockstep.
        self._round = 0
        # Verdict caching speeds up single-node resume; disabled for multi-node
        # because a cache hit would skip a deploy and desync the workers.
        self.use_cache = inp.scene != SCENE_MULTI

        run_id = inp.case_key.replace("::", "__").replace("/", "_")
        self.work_dir = Path(opt.work_dir) / run_id
        self.log_dir = self.work_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work_dir / "state.json"
        self.report_path = self.work_dir / "report.json"

    # ------------------------------------------------------------ one trial
    def _run_trial(self, candidate: Candidate) -> TrialResult:
        start = time.time()
        self._round += 1
        round_idx = self._round
        log_path = self.log_dir / f"round{round_idx}_{candidate.short}.log"
        try:
            outcome = self.runner.validate(candidate, round_idx, self.log_dir)
            v, note = evaluate(outcome)
            result = TrialResult(
                candidate=candidate,
                verdict=v,
                duration_s=time.time() - start,
                rebuilt=getattr(outcome, "rebuilt", False),
                exit_code=outcome.exit_code,
                log_path=str(log_path),
                note=note,
            )
        except BuildError as exc:
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
        """Run + flaky-confirm a candidate, caching the verdict for resume."""
        if self.use_cache:
            cached = state.verdicts.get(candidate.commit)
            if cached:
                logger.info("Using cached verdict %s for %s", cached, candidate.short)
                return cached  # type: ignore[return-value]

        result = self._run_trial(candidate)

        # Flaky guard: re-confirm a FAIL; if a retry passes, the commit is
        # unreliable and must not be used as a bisect boundary -> SKIP.
        if result.verdict == "FAIL" and self.opt.fail_confirm_retries > 0:
            for i in range(self.opt.fail_confirm_retries):
                logger.info("Confirming FAIL for %s (%d/%d)", candidate.short, i + 1,
                            self.opt.fail_confirm_retries)
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
    def _verify_endpoints(self, good: Candidate, candidates: list[Candidate],
                          state: BisectState) -> bool:
        bad = candidates[-1]
        if self.opt.verify_bad:
            v = self._judge(bad, state=state)
            report.print_endpoint_check("bad", bad, v, ok=v == "FAIL")
            if v != "FAIL":
                logger.error("Bad commit did not reproduce the failure (%s); aborting.", v)
                return False
        else:
            state.verdicts[bad.commit] = "FAIL"

        if self.opt.verify_good:
            v = self._judge(good, state=state)
            report.print_endpoint_check("good", good, v, ok=v == "PASS")
            if v != "PASS":
                logger.error("Good baseline is not actually good (%s); range invalid.", v)
                return False
        return True

    # --------------------------------------------------------------- search
    @staticmethod
    def _pick_mid(lo: int, hi: int, skipped: set[int]) -> int | None:
        """Pick a testable index in [lo, hi), nearest to the midpoint."""
        mid = (lo + hi) // 2
        for offset in range(0, hi - lo):
            for cand in (mid + offset, mid - offset):
                if lo <= cand < hi and cand not in skipped:
                    return cand
        return None

    def _bisect(self, candidates: list[Candidate], state: BisectState) -> Candidate | None:
        lo = state.lo
        hi = state.hi if state.hi else len(candidates) - 1
        skipped = {i for i, c in enumerate(candidates) if state.verdicts.get(c.commit) == "SKIP"}

        while lo < hi:
            mid = self._pick_mid(lo, hi, skipped)
            if mid is None:
                logger.warning("Entire window [%d,%d) skipped; cannot narrow further.", lo, hi)
                break
            report.print_progress(self._round + 1, lo, hi, candidates[mid])
            v = self._judge(candidates[mid], state)
            if v == "FAIL":
                hi = mid
            elif v == "PASS":
                lo = mid + 1
            else:  # SKIP
                skipped.add(mid)
            state.lo, state.hi, state.round_idx = lo, hi, self._round
            state.save(self.state_path)

        return candidates[lo] if lo < len(candidates) else None

    # ----------------------------------------------------------------- main
    def run(self) -> Candidate | None:
        bad = git_ops.describe(self.repo, self.inp.bad_commit)
        good_sha = self._resolve_good()
        good = git_ops.describe(self.repo, good_sha)
        logger.info("Bisecting %s: good=%s bad=%s", self.inp.case_key, good.short, bad.short)

        candidates = git_ops.candidate_list(self.repo, good.commit, bad.commit)
        logger.info("Search space: %d commits", len(candidates))

        state = BisectState.load(self.state_path) or BisectState(hi=len(candidates) - 1)

        if not self._verify_endpoints(good, candidates, state):
            self.runner.finish()
            report.write_report_json(self.report_path, inp=self.inp, good=good, bad=bad,
                                     first_bad=None, trials=self.trials)
            return None

        first_bad = self._bisect(candidates, state)
        self.runner.finish()

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
            self._maybe_update_good_table(candidates, first_bad, good)
        report.write_report_json(self.report_path, inp=self.inp, good=good, bad=bad,
                                 first_bad=first_bad, trials=self.trials)
        return first_bad

    def _resolve_good(self) -> str:
        if self.inp.good_commit:
            return self.inp.good_commit
        table = GoodTable(self.opt.good_table_path)
        entry = table.lookup(self.inp.case_key)
        if entry is None:
            raise SystemExit(
                f"No good commit for case {self.inp.case_key!r}: not in table "
                f"{self.opt.good_table_path} and --good-commit not given."
            )
        return entry.last_good_commit

    def _maybe_update_good_table(self, candidates, first_bad, good) -> None:
        if not os.getenv("BISECT_UPDATE_GOOD_TABLE"):
            return
        idx = candidates.index(first_bad)
        new_good = candidates[idx - 1] if idx > 0 else good
        GoodTable(self.opt.good_table_path).update(
            case_key=self.inp.case_key,
            scene=self.inp.scene,
            config_yaml=self.inp.config_yaml,
            case_name=self.inp.case_name,
            last_good_commit=new_good.commit,
            last_good_pr=new_good.pr_number or "",
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-bisect a nightly test failure to the first bad commit.")
    p.add_argument("--scene", required=True, choices=list(SCENES))
    p.add_argument("--config-yaml", required=True, help="CONFIG_YAML_PATH of the failing case")
    p.add_argument("--case-name", default="", help="pytest -k filter (single-node case id)")
    p.add_argument("--bad-commit", default=os.getenv("VLLM_ASCEND_REF", "HEAD"))
    p.add_argument("--good-commit", default=None, help="override; else read from good table")
    p.add_argument("--config-base-path", default=os.getenv("CONFIG_BASE_PATH"))
    p.add_argument("--good-table", default=DEFAULT_GOOD_TABLE)
    p.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    p.add_argument("--repo-dir", default=str(REPO_ROOT))
    p.add_argument("--num-nodes", type=int, default=int(os.getenv("LWS_GROUP_SIZE", "1")))
    p.add_argument("--node-index", type=int, default=int(os.getenv("LWS_WORKER_INDEX", "0")))
    p.add_argument("--coord-dir", default=DEFAULT_COORD_DIR, help="shared barrier dir (multi-node)")
    p.add_argument("--fail-confirm-retries", type=int, default=1)
    p.add_argument("--no-verify-good", action="store_true")
    p.add_argument("--no-verify-bad", action="store_true")
    p.add_argument("--trial-timeout-s", type=float, default=5400.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    inp = BisectInput(
        scene=args.scene,
        config_yaml=args.config_yaml,
        case_name=args.case_name,
        bad_commit=args.bad_commit,
        config_base_path=args.config_base_path,
        good_commit=args.good_commit,
    )
    opt = BisectOptions(
        repo_dir=Path(args.repo_dir),
        work_dir=args.work_dir,
        coord_dir=args.coord_dir,
        good_table_path=args.good_table,
        fail_confirm_retries=args.fail_confirm_retries,
        verify_good=not args.no_verify_good,
        verify_bad=not args.no_verify_bad,
        num_nodes=args.num_nodes,
        node_index=args.node_index,
        trial_timeout_s=args.trial_timeout_s,
    )

    # On a multi-node worker, drive the worker loop instead of the search.
    if inp.scene == SCENE_MULTI and not opt.is_master:
        from tests.e2e.nightly.bisect.worker_agent import run_worker

        return run_worker(inp, opt)

    first_bad = Bisector(inp, opt).run()
    return 0 if first_bad is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
