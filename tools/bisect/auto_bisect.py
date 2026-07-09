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

    python -m tools.bisect.auto_bisect \
        --scene single_node \
        --config-yaml DeepSeek-R1-0528-W8A8.yaml \
        --name DeepSeek-R1-0528-W8A8 \
        --bad-commit HEAD

Each trial runs the whole CONFIG_YAML_PATH file (all test_cases). The good
commit is read from the nightly status table unless ``--good-commit`` is given.
"""

import argparse
import logging
import os
import time
from pathlib import Path

from tools.bisect import git_ops, report
from tools.bisect.build_manager import BuildError, BuildManager
from tools.bisect.config import (
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
from tools.bisect.good_table import GoodTable
from tools.bisect.state import BisectState
from tools.bisect.verdict import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("auto_bisect")


class Bisector:
    def __init__(self, inp: BisectInput, opt: BisectOptions):
        self.inp = inp
        self.opt = opt
        self.repo = opt.repo_dir
        self.builder = BuildManager(opt)
        from tools.bisect.runner import build_runner

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
        if result.verdict == "FAIL" and self.opt.fail_confirm_retries > 0:
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
                    "e.g. vllm/vllm-ascend version mismatch). Fix the environment "
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
                    "e.g. vllm/vllm-ascend version mismatch). The whole range is "
                    "likely unrunnable against the installed vllm; fix the "
                    "environment before bisecting; aborting."
                )
                return False
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
        # SKIP is tracked in-run only; never pre-seeded from a (possibly stale)
        # cache, so previously-skipped commits get a fresh attempt on resume.
        skipped: set[int] = set()

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

        state = BisectState.load(self.state_path, good=good.commit, bad=bad.commit) or BisectState(
            good=good.commit, bad=bad.commit, hi=len(candidates) - 1
        )

        # finish() publishes the DONE sentinel that releases multi-node workers;
        # it must run on every exit path (including a barrier timeout or any
        # error) so workers never hang waiting for the next round.
        try:
            if not self._verify_endpoints(good, candidates, state):
                report.write_report_json(
                    self.report_path, inp=self.inp, good=good, bad=bad, first_bad=None, trials=self.trials
                )
                return None
            first_bad = self._bisect(candidates, state)
        finally:
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
        report.write_report_json(
            self.report_path, inp=self.inp, good=good, bad=bad, first_bad=first_bad, trials=self.trials
        )
        return first_bad

    def _resolve_good(self) -> str:
        if self.inp.good_commit:
            return self.inp.good_commit
        entry = GoodTable(self.opt.good_table_path).lookup_last_good(
            name=self.inp.name, config_yaml=self.inp.config_yaml
        )
        if entry is None:
            raise SystemExit(
                f"No successful good-table row for name={self.inp.name!r} / "
                f"config_yaml={self.inp.config_yaml!r} in {self.opt.good_table_path}, "
                "and --good-commit was not given."
            )
        # Record the paired vLLM commit so the env can be kept in sync (the good
        # row knows which vLLM that vllm-ascend commit was validated against).
        self.inp.good_vllm_commit = entry.vllm_commit or None
        if entry.vllm_commit:
            logger.info(
                "Good row pairs vllm-ascend %s with vLLM %s", entry.vllm_ascend_commit[:12], entry.vllm_commit[:12]
            )
        return entry.vllm_ascend_commit


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-bisect a nightly test failure to the first bad commit.")
    p.add_argument("--scene", required=True, choices=list(SCENES))
    p.add_argument(
        "--config-yaml",
        required=True,
        help="CONFIG_YAML_PATH of the failing case file; the whole file (all test_cases) is run each trial",
    )
    p.add_argument(
        "--name",
        default=None,
        help="nightly case name to match the good-table 'name' column (optional; falls back to matching by yaml/path)",
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
        default=int(os.getenv("LWS_WORKER_INDEX", "0")),
        help="this node's index; defaults to $LWS_WORKER_INDEX (set by LWS), else 0",
    )
    p.add_argument("--coord-dir", default=DEFAULT_COORD_DIR, help="shared barrier dir (multi-node)")
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
    p.add_argument("--trial-timeout-s", type=float, default=5400.0)
    p.add_argument(
        "--force-initial-build",
        action="store_true",
        help="clean-rebuild on the first trial instead of trusting the container's existing build at HEAD",
    )
    p.add_argument(
        "--no-assume-built-head", action="store_true", help="do not treat the container's HEAD as already built"
    )
    p.add_argument(
        "--native-check",
        choices=["per-commit", "since-build"],
        default="per-commit",
        help="'per-commit' (default): decide rebuild from each bisected "
        "commit's own diff (compile iff it touched native/cpp files); "
        "'since-build': from all changes since the last build",
    )
    return p.parse_args(argv)


# Default config dirs for multi-node, used to locate the yaml when deriving
# num_nodes (mirrors the internal/external DP loaders).
_MULTI_CONFIG_BASES = (
    "tests/e2e/nightly/multi_node/internal_dp/config",
    "tests/e2e/nightly/multi_node/external_dp/config",
)


def _resolve_num_nodes(args: argparse.Namespace, repo_dir: Path) -> int:
    """Node count: explicit --num-nodes, else the yaml's num_nodes (multi), else 1."""
    if args.num_nodes is not None:
        return args.num_nodes
    if args.scene != SCENE_MULTI:
        return 1
    import yaml  # local import: only needed for multi-node

    bases = [args.config_base_path] if args.config_base_path else list(_MULTI_CONFIG_BASES)
    for base in bases:
        path = repo_dir / base / args.config_yaml
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            n = int(data.get("num_nodes", 0))
            if n:
                logger.info("Resolved num_nodes=%d from %s", n, path)
                return n
    raise SystemExit(
        "Could not determine --num-nodes: not provided and no 'num_nodes' found in "
        f"the multi-node config yaml {args.config_yaml!r}. Pass --num-nodes explicitly."
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_dir = Path(args.repo_dir)
    num_nodes = _resolve_num_nodes(args, repo_dir)
    inp = BisectInput(
        scene=args.scene,
        config_yaml=args.config_yaml,
        name=args.name,
        bad_commit=args.bad_commit,
        config_base_path=args.config_base_path,
        good_commit=args.good_commit,
    )
    opt = BisectOptions(
        repo_dir=repo_dir,
        work_dir=args.work_dir,
        coord_dir=args.coord_dir,
        good_table_path=args.good_table,
        fail_confirm_retries=args.fail_confirm_retries,
        verify_good=not args.no_verify_good,
        verify_bad=not args.no_verify_bad,
        num_nodes=num_nodes,
        node_index=args.node_index,
        release_file=args.release_file,
        barrier_timeout_s=args.barrier_timeout_s,
        trial_timeout_s=args.trial_timeout_s,
        assume_built_head=not args.no_assume_built_head,
        force_initial_build=args.force_initial_build,
        native_check=args.native_check,
    )

    # On a multi-node worker, drive the worker loop instead of the search.
    if inp.scene == SCENE_MULTI and not opt.is_master:
        from tools.bisect.worker_agent import run_worker

        return run_worker(inp, opt)

    first_bad = Bisector(inp, opt).run()
    return 0 if first_bad is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
