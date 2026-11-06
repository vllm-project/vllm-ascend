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
"""Decide whether a checkout needs recompilation and perform the deploy.

Decision rule (the compile-optimisation core):

* Compare the target commit against the *last successfully built* commit, not
  just the candidate's own diff -- bisect jumps around, so we must catch any
  native change anywhere in the skipped range.
* If the delta touches native sources / build definitions  -> rebuild.
* Analyze the Docker build chain to decide dependency transition actions.
* Otherwise (pure ``.py`` / yaml / md)                     -> plain checkout;
  the editable install picks the change up live.
"""

import contextlib
import logging
import os
import platform
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tools.bisect import git_ops
from tools.bisect.build_analyzer import analyze_commit, plan_transition
from tools.bisect.build_analyzer.diff import package_state_changes
from tools.bisect.build_analyzer.ir import BuildAnalysis, BuildSyntaxError
from tools.bisect.build_analyzer.planner import TransitionPlan
from tools.bisect.config import (
    BUILD_DEF_GLOBS,
    NATIVE_GLOBS,
    BisectOptions,
)
from tools.bisect.git_ops import GitError
from tools.bisect.version_compat import TORCH_NPU_PIP_INDEX_URL, VersionAdaptationError

logger = logging.getLogger(__name__)

# Build artifacts removed before a rebuild so stale ``.so`` files can never be
# loaded against a new checkout.
# The custom-op build script preserves csrc/build, including CMakeCache.txt.
# That cache can point HI_PYTHON at a short-lived uv build-environment path,
# so discard it before rebuilding a candidate in a new isolated environment.
_STALE_ARTIFACTS = ("build", "csrc/build", "*.egg-info", "dist")


@dataclass
class BuildDecision:
    rebuild: bool
    reinstall_reqs: bool
    native_hits: list[str]
    reason: str


class BuildManager:
    """Owns checkout + (conditional) recompile, tracking the built commit."""

    def __init__(self, options: BisectOptions):
        self.opt = options
        self.repo = options.repo_dir
        # The commit whose binary is currently installed. By default we trust
        # that the nightly container is already built at its current HEAD, so we
        # seed the baseline with it -> the bad endpoint becomes checkout-only and
        # we avoid a slow/network-bound first-prepare reinstall. None means the
        # first prepare() must do a conservative clean rebuild.
        self.last_built_commit: str | None = None
        if options.assume_built_head and not options.force_initial_build:
            try:
                self.last_built_commit = git_ops.current_commit(self.repo)
                logger.info("[build] assuming container is built at HEAD %s", self.last_built_commit[:12])
            except Exception:  # noqa: BLE001 - best effort; fall back to rebuild
                self.last_built_commit = None
        self.installed_commit = self.last_built_commit
        self._baseline_untrusted = False
        self._plan_cache: dict[tuple[str, str], TransitionPlan] = {}
        self._analysis_cache: dict[str, BuildAnalysis] = {}

    def _analysis_at(self, commit: str) -> BuildAnalysis:
        if commit in self._analysis_cache:
            return self._analysis_cache[commit]
        architecture = "aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"
        os_variant = "ubuntu"
        os_release = Path("/etc/os-release")
        if os_release.is_file() and "openeuler" in os_release.read_text(encoding="utf-8").lower():
            os_variant = "openEuler"
        try:
            analysis = analyze_commit(self.repo, commit, self.opt.soc, architecture, os_variant)
        except (BuildSyntaxError, ValueError, OSError) as exc:
            raise BuildError(f"build-chain analysis failed: {exc}") from exc
        self._analysis_cache[commit] = analysis
        return analysis

    def _transition_plan(self, target_commit: str) -> TransitionPlan | None:
        if self.installed_commit is None or not self.opt.soc:
            return None
        key = self.installed_commit, target_commit
        if key in self._plan_cache:
            return self._plan_cache[key]
        before = self._analysis_at(self.installed_commit)
        after = self._analysis_at(target_commit)
        before_base = next((op.value for op in before.operations if op.kind == "from"), None)
        after_base = next((op.value for op in after.operations if op.kind == "from"), None)
        if before_base != after_base:
            raise BuildError(
                f"CANN base image changed: {before_base} -> {after_base}; use a matching image for this candidate"
            )
        plan = plan_transition(before, after)
        self._plan_cache[key] = plan
        return plan

    # ------------------------------------------------------------ decision
    def decide(self, target_commit: str) -> BuildDecision:
        if self._baseline_untrusted:
            raise BuildError("a previous environment transition failed; restart from a matching container")
        self._transition_plan(target_commit)
        if self.last_built_commit is None:
            # No trusted baseline: rebuild the extension from the target checkout.
            return BuildDecision(
                rebuild=True,
                reinstall_reqs=False,
                native_hits=[],
                reason="no established build baseline (clean rebuild)",
            )
        # Choose which set of changed files drives the decision.
        if self.last_built_commit == target_commit:
            files = []
            scope = "already built this exact commit"
        elif self.opt.native_check == "per-commit":
            files = git_ops.commit_changed_files(self.repo, target_commit)
            scope = "this commit"
        else:
            files = git_ops.changed_files(self.repo, self.last_built_commit, target_commit)
            scope = "since last build"

        build_def_globs = (
            tuple(pattern for pattern in BUILD_DEF_GLOBS if pattern != "pyproject.toml")
            if self.opt.soc
            else BUILD_DEF_GLOBS
        )
        native = git_ops.matches_any(files, NATIVE_GLOBS + build_def_globs)
        if native:
            reason = f"native/build-def files changed ({scope}): {native[:5]}"
            return BuildDecision(True, False, native, reason)
        return BuildDecision(
            rebuild=False,
            reinstall_reqs=False,
            native_hits=[],
            reason=(
                "already built this exact commit"
                if self.last_built_commit == target_commit
                else f"no native/cpp changes ({scope}) -> editable install, no compile"
            ),
        )

    # ------------------------------------------------------------- prepare
    def prepare(self, target_commit: str, log_file: Path | None = None) -> BuildDecision:
        """Checkout ``target_commit`` and rebuild/reinstall as needed.

        The target is resolved first: on a multi-node worker (or any node whose
        repo is a shallow clone, e.g. the nightly ``--depth 1`` container) the
        commanded commit -- an ancestor in the bisect range -- is usually NOT
        present locally, so it must be recovered from origin exactly like the
        master's ``describe()`` does. When the commit is already local this is
        a single no-op rev-parse.

        Raises ``BuildError`` on a failed install and ``GitError`` when the
        commit cannot be recovered, so the caller can record a SKIP (rather
        than a misleading FAIL) for this commit.
        """
        target_commit = git_ops.resolve_commit(self.repo, target_commit)
        decision = self.decide(target_commit)
        plan = self._transition_plan(target_commit)
        logger.info("[build] %s -> %s", target_commit[:12], decision.reason)
        if self.opt.soc and log_file is not None:
            target_analysis = self._analysis_at(target_commit)
            source_commit = self.installed_commit
            sources = ", ".join(dict.fromkeys(instruction.source for instruction in target_analysis.instructions))
            with open(log_file, "a", encoding="utf-8") as out:
                out.write(
                    f"[build-chain:before] installed_commit={source_commit[:12] if source_commit else 'unknown'} "
                    f"target_commit={target_commit[:12]} SoC={self.opt.soc}\n"
                    f"[build-chain:target] sources={sources}\n"
                )
                if source_commit is not None:
                    changes = package_state_changes(self._analysis_at(source_commit), target_analysis)
                    out.write(f"[build-chain:comparison] package_changes={len(changes)}\n")
                    for change in changes:
                        before = f"{change.before.action} {change.before.specification}" if change.before else "absent"
                        after = f"{change.after.action} {change.after.specification}" if change.after else "absent"
                        out.write(f"[build-chain:package] {change.name}: {before} -> {after}\n")
                for outcome in target_analysis.package_outcomes:
                    out.write(
                        f"[build-chain:target-package] {outcome.source}:{outcome.line} "
                        f"{outcome.name} {outcome.action} {outcome.specification}\n"
                    )
                for diagnostic in target_analysis.diagnostics:
                    out.write(f"[build-chain:diagnostic] {diagnostic}\n")
        if plan is not None:
            logger.info("[build-chain] transition=%s actions=%d", plan.status, len(plan.actions))
            for blocker in plan.blockers:
                logger.warning("[build-chain] %s", blocker)
            if log_file is not None:
                with open(log_file, "a", encoding="utf-8") as out:
                    out.write(f"\n[build-chain] transition={plan.status} actions={len(plan.actions)}\n")
                    for blocker in plan.blockers:
                        out.write(f"[build-chain] WARNING: {blocker}\n")
                    for action in plan.actions:
                        out.write(f"[build-chain] action {action.id} {action.kind}: {action.command}\n")

        git_ops.checkout(self.repo, target_commit)

        root_env = None
        if self.opt.soc:
            target_analysis = self._analysis_at(target_commit)
            root_source = target_analysis.instructions[0].source
            root_args = {
                name: value
                for operation in target_analysis.operations
                if operation.source == root_source and operation.kind == "arg"
                for name, _, value in (operation.value.partition("="),)
            }
            root_env = dict(os.environ)
            root_env.setdefault("PIP_INDEX_URL", TORCH_NPU_PIP_INDEX_URL)
            root_env["VLLM_BATCH_INVARIANT"] = "1"
            root_env["PIP_EXTRA_INDEX_URL"] = " ".join(
                url for url in (root_args.get("ASCEND_INDEX_URL"), root_args.get("PYTORCH_INDEX_URL")) if url
            )

        if (decision.reinstall_reqs or decision.rebuild) and log_file is not None:
            logger.info(
                "[build] compiling/installing (this can take a while); follow progress with: tail -f %s", log_file
            )
        if plan is not None:
            self._execute_transition(plan, log_file, root_env)
        if decision.rebuild:
            self._clean_artifacts()
            install_cmd = self.opt.pip_install_cmd
            if self.opt.soc:
                install_cmd = [
                    "bash",
                    "-lc",
                    "source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
                    "source /usr/local/Ascend/nnal/atb/set_env.sh && "
                    f"exec {shlex.join(install_cmd)}",
                ]
            self._run(install_cmd, log_file, "pip install -e .", env=root_env)
        # Only advance the baseline once the binary actually matches the source.
        if decision.rebuild or self.last_built_commit is None:
            self.last_built_commit = target_commit
        self.installed_commit = target_commit
        self._baseline_untrusted = False
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as out:
                out.write(
                    f"[build-chain:after] checked_out={git_ops.current_commit(self.repo)[:12]} "
                    f"installed_commit={self.installed_commit[:12]} "
                    f"actions_executed={len(plan.actions) if plan else 0} rebuilt={decision.rebuild}\n"
                )
        self._plan_cache.clear()
        self._analysis_cache.clear()
        return decision

    def _execute_transition(
        self,
        plan: TransitionPlan,
        log_file: Path | None,
        env: dict[str, str] | None,
    ) -> None:
        if not plan.actions:
            return
        local_root = shlex.quote(str(self.repo))
        for action in plan.actions:
            command = action.command.replace("-e /vllm-workspace/vllm-ascend/", f"-e {local_root}/")
            context = [
                f"cd {local_root}" if item == "cd /vllm-workspace/vllm-ascend" else item
                for item in action.shell_context
            ]
            script = " && ".join(["set -e", *context, command])
            self._run(["bash", "-lc", script], log_file, f"build-chain action {action.id}", env=env)

    def invalidate_baseline(self) -> None:
        """A partially applied transition cannot be compared with the old environment."""
        self.installed_commit = None
        self.last_built_commit = None
        self._baseline_untrusted = True
        self._plan_cache.clear()
        self._analysis_cache.clear()

    # ------------------------------------------------------------- helpers
    def _clean_artifacts(self) -> None:
        for pattern in _STALE_ARTIFACTS:
            for path in self.repo.glob(pattern):
                logger.info("[build] removing stale artifact %s", path.name)
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
        # Drop compiled python caches; cheap and avoids import surprises.
        for cache in self.repo.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)

    def _run(
        self,
        cmd: list[str],
        log_file: Path | None,
        label: str,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        logger.info("[build] running: %s", " ".join(cmd))
        workdir = cwd or self.repo
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as out:
                out.write(self._cmake_python_diagnostics("before install"))
                out.flush()
                proc = subprocess.run(cmd, cwd=str(workdir), stdout=out, stderr=subprocess.STDOUT, text=True, env=env)
            tail = "(see build log)"
            if proc.returncode != 0:
                with open(log_file, "a", encoding="utf-8") as out:
                    out.write(self._cmake_python_diagnostics("after failed install"))
                    out.write(self._cmake_failure_logs())
                # Surface the tail of the build log in the job log: the log
                # file lives in the runner container and is lost when the job
                # ends, so "(see build log)" alone leaves nothing to debug.
                with contextlib.suppress(OSError):
                    tail = log_file.read_text(encoding="utf-8", errors="replace")[-4000:]
        else:
            proc = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True, env=env)
            tail = (proc.stdout or "")[-2000:]
        if proc.returncode != 0:
            self.invalidate_baseline()
            raise BuildTransitionError(f"{label} failed (rc={proc.returncode}):\n{tail}")

    def _cmake_python_diagnostics(self, phase: str) -> str:
        """Report only the CMake Python paths, not the full potentially sensitive cache."""
        cache = self.repo / "csrc" / "build" / "CMakeCache.txt"
        lines = [f"\n[build] CMake diagnostics ({phase}): {cache}\n"]
        if not cache.is_file():
            lines.append("[build] CMakeCache.txt does not exist\n")
            return "".join(lines)
        try:
            entries = cache.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            lines.append(f"[build] cannot read CMakeCache.txt: {exc}\n")
            return "".join(lines)
        for key in ("HI_PYTHON", "Python3_EXECUTABLE", "_Python3_EXECUTABLE"):
            matches = [entry.partition("=")[2] for entry in entries if entry.startswith(f"{key}:")]
            if not matches:
                lines.append(f"[build] {key}: not cached\n")
                continue
            value = matches[-1]
            lines.append(f"[build] {key}: {value} (exists={Path(value).is_file()})\n")
        if phase == "after failed install":
            lines.append("[build] uv may remove temporary build Python on exit; compare the before-install paths\n")
        return "".join(lines)

    def _cmake_failure_logs(self) -> str:
        cmake_files = self.repo / "csrc" / "build" / "CMakeFiles"
        lines = []
        for name in ("CMakeConfigureLog.yaml", "CMakeError.log"):
            path = cmake_files / name
            if not path.is_file():
                continue
            try:
                tail = path.read_text(encoding="utf-8", errors="replace")[-4000:]
            except OSError as exc:
                lines.append(f"[build] cannot read {path}: {exc}\n")
            else:
                lines.append(f"\n[build] tail of {path}:\n{tail}\n")
        return "".join(lines)


class BuildError(RuntimeError):
    pass


class BuildTransitionError(BuildError):
    """An environment installation failed; continuing the bisect is unsafe."""


# Everything a deploy (resolve + checkout + build transition) can
# raise that must surface as a per-round SKIP instead of killing the driving
# agent. One shared tuple so every catch site (master runner, worker agent,
# orchestrator trial loop) stays in lockstep -- a missed copy leaves the other
# side waiting out the full barrier timeout on every remaining round.
DEPLOY_ERRORS = (BuildError, GitError, VersionAdaptationError)
