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
import json
import logging
import os
import platform
import shlex
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import regex as re

from tools.bisect import git_ops
from tools.bisect.build_analyzer import analyze_commit, plan_transition
from tools.bisect.build_analyzer.base_apply import apply_base_delta, target_environment
from tools.bisect.build_analyzer.base_image import (
    BaseImageDelta,
    BaseImageError,
    from_reference,
)
from tools.bisect.build_analyzer.diff import package_state_changes
from tools.bisect.build_analyzer.environment import (
    build_environment,
    environment_assignments,
    environment_references,
    recover_inherited_environment,
)
from tools.bisect.build_analyzer.image_executor import ImageReplayExecutor
from tools.bisect.build_analyzer.ir import BuildAnalysis, BuildSyntaxError
from tools.bisect.build_analyzer.planner import TransitionPlan, plan_target_image_transition
from tools.bisect.build_analyzer.registry_image import RegistryImageResolver
from tools.bisect.build_analyzer.replay_cleanup import protect_replay_cleanup
from tools.bisect.config import (
    BUILD_DEF_GLOBS,
    NATIVE_GLOBS,
    BisectOptions,
)
from tools.bisect.git_ops import GitError
from tools.bisect.process_output import run_streaming
from tools.bisect.version_compat import VersionAdaptationError

logger = logging.getLogger(__name__)

# Build artifacts removed before a rebuild so stale ``.so`` files can never be
# loaded against a new checkout.
# The custom-op build script preserves csrc/build, including CMakeCache.txt.
# That cache can point HI_PYTHON at a short-lived uv build-environment path,
# so discard it before rebuilding a candidate in a new isolated environment.
_STALE_ARTIFACTS = ("build", "csrc/build", "csrc/torch_ops_extension/*/build", "*.egg-info", "dist")


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
        self._transition_marker = Path(options.work_dir) / f"base-transition-node{options.node_index}.pending"
        self._baseline_untrusted = self._transition_marker.exists()
        if self._baseline_untrusted:
            if options.force_initial_build:
                # Allow one full initialization, without trusting or deleting
                # the pending state. A new failure must block later trials.
                logger.warning(
                    "[build-chain] recovering incomplete transition with full target replay: %s",
                    self._transition_marker,
                )
                self._baseline_untrusted = False
            else:
                logger.error("[base-image] incomplete prior transition: %s", self._transition_marker)
        self._plan_cache: dict[tuple[str | None, str], TransitionPlan] = {}
        self._analysis_cache: dict[str, BuildAnalysis] = {}
        self._base_delta_cache: dict[tuple[str, str], BaseImageDelta | None] = {}
        self._image_resolver = RegistryImageResolver(Path.home() / ".cache" / "nightly_bisect" / "images")
        self.test_environment: dict[str, str] | None = None
        self._build_env_baseline: dict[str, str | None] = {}
        self._unknown_build_env: set[str] = set()

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
        if not self.opt.soc:
            return None
        key = self.installed_commit, target_commit
        if key in self._plan_cache:
            return self._plan_cache[key]
        before = (
            self._analysis_at(self.installed_commit) if self.installed_commit is not None else BuildAnalysis((), (), ())
        )
        after = self._analysis_at(target_commit)
        before_base = next((op.value for op in before.operations if op.kind == "from"), None)
        after_base = next((op.value for op in after.operations if op.kind == "from"), None)
        base_delta = None
        if self.installed_commit is None or before_base != after_base:
            if after_base is None or (self.installed_commit is not None and before_base is None):
                raise BuildError("Target image replay requires a base image reference")
            logger.info(
                "[build-chain] FROM changed: %s -> %s; replay target image including Python "
                "and the full target build chain",
                before_base,
                after_base,
            )
            architecture = "aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"
            try:
                reference, history = self._image_resolver.resolve_history(from_reference(after_base), architecture)
                plan = plan_target_image_transition(
                    before, after, reference, history, rebuild_project=self.installed_commit != target_commit
                )
            except (BaseImageError, ValueError, OSError) as exc:
                raise BuildError(f"target image history analysis failed: {exc}") from exc
            destination = Path(self.opt.work_dir) / f"target-image-plan-node{self.opt.node_index}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")
            self._plan_cache[key] = plan
            return plan
        plan = plan_transition(before, after, base_delta, rebuild_project=self.installed_commit != target_commit)
        self._base_delta_cache[key] = base_delta
        self._plan_cache[key] = plan
        return plan

    # ------------------------------------------------------------ decision
    def decide(self, target_commit: str) -> BuildDecision:
        if self._baseline_untrusted:
            raise BuildError(
                "a previous environment transition failed; restart with --force-initial-build "
                "for full target initialization, or use a matching container"
            )
        transition = self._transition_plan(target_commit)
        if transition and any(action.kind.startswith("image-") for action in transition.actions):
            return BuildDecision(True, False, [], "target image replay requires rebuilding native extensions")
        if self.installed_commit is not None:
            delta = self._base_delta_cache.get((self.installed_commit, target_commit))
            if delta is not None and (delta.changed_files or delta.changed_config):
                return BuildDecision(True, False, [], "base image changed; rebuild dependent native extensions")
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
        base_delta = (
            self._base_delta_cache.get((self.installed_commit, target_commit)) if self.installed_commit else None
        )
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
                    if any(action.kind.startswith("image-") for action in plan.actions):
                        out.write("[build-chain:mode] target image including Python, then full target build replay\n")
                    for blocker in plan.blockers:
                        out.write(f"[build-chain] WARNING: {blocker}\n")
                    for action in plan.actions:
                        out.write(f"[build-chain] action {action.id} {action.kind}: {action.command}\n")

        if plan is not None and (plan.actions or plan.target_environment is not None):
            self._transition_marker.parent.mkdir(parents=True, exist_ok=True)
            self._transition_marker.write_text(
                f"environment transition to {target_commit} in progress\n", encoding="utf-8"
            )
            self._baseline_untrusted = True

        git_ops.checkout(self.repo, target_commit)

        root_env = None
        if self.opt.soc:
            root_env = dict(os.environ)
            if base_delta is not None:
                root_env = target_environment(base_delta, root_env)

        if (decision.reinstall_reqs or decision.rebuild) and log_file is not None:
            logger.info(
                "[build] compiling/installing (this can take a while); follow progress with: tail -f %s", log_file
            )
        project_rebuilt = plan is not None and any(
            action.kind == "install"
            and any(word.rstrip("/") == "/vllm-workspace/vllm-ascend" for word in shlex.split(action.command))
            for action in plan.actions
        )
        if plan is not None:
            if project_rebuilt:
                # Project installs in the plan must not inherit the previous
                # image's native build cache.
                self._clean_artifacts()
            self._execute_transition(plan, log_file, root_env, base_delta)
        # Only advance the baseline once the binary actually matches the source.
        if project_rebuilt:
            self.last_built_commit = target_commit
        self.installed_commit = target_commit
        self.test_environment = root_env
        self._transition_marker.unlink(missing_ok=True)
        self._baseline_untrusted = False
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as out:
                out.write(
                    f"[build-chain:after] checked_out={git_ops.current_commit(self.repo)[:12]} "
                    f"installed_commit={self.installed_commit[:12]} "
                    f"actions_executed={len(plan.actions) if plan else 0} rebuilt={project_rebuilt}\n"
                )
        self._plan_cache.clear()
        self._base_delta_cache.clear()
        self._analysis_cache.clear()
        return BuildDecision(project_rebuilt, decision.reinstall_reqs, decision.native_hits, decision.reason)

    def _inherited_build_environment(
        self, plan: TransitionPlan, environment: dict[str, str], log_file: Path | None
    ) -> dict[str, str]:
        recovered, unknown = recover_inherited_environment(plan.previous_environment, environment)
        managed = set(dict(plan.previous_environment)) | set(dict(plan.target_environment or ()))
        managed.update(name for action in plan.actions for name, _ in action.environment)
        for name in managed:
            if name not in self._build_env_baseline:
                self._build_env_baseline[name] = recovered.get(name)
                if name in unknown:
                    self._unknown_build_env.add(name)
        needed = managed - set(dict(plan.target_environment or ()))
        needed.update(name for _, value in (plan.target_environment or ()) for name in environment_references(value))
        needed.update(
            name for action in plan.actions for _, value in action.environment for name in environment_references(value)
        )
        for name in sorted(needed & self._unknown_build_env):
            warning = f"Cannot recover inherited ENV {name}; retaining its ambient baseline value"
            logger.warning("[build-chain] %s", warning)
            if log_file is not None:
                with open(log_file, "a", encoding="utf-8") as out:
                    out.write(f"[build-chain:warning] {warning}\n")
        inherited = dict(environment)
        for name, value in self._build_env_baseline.items():
            if value is None:
                inherited.pop(name, None)
            else:
                inherited[name] = value
        return inherited

    def _execute_transition(
        self,
        plan: TransitionPlan,
        log_file: Path | None,
        env: dict[str, str] | None,
        base_delta: BaseImageDelta | None = None,
    ) -> None:
        if not plan.actions and plan.target_environment is None:
            return
        if env is None:
            env = dict(os.environ)
        image_actions = tuple(action for action in plan.actions if action.kind.startswith("image-"))
        image_executor = None
        if image_actions:
            architecture = "aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"
            try:
                image_executor = ImageReplayExecutor(
                    image_actions,
                    self._image_resolver,
                    architecture,
                    env,
                    report=lambda message: self._report_image_progress(message, log_file),
                )
            except (OSError, RuntimeError, ValueError) as exc:
                raise BuildError(f"target image replay preflight failed: {exc}") from exc
            self._transition_marker.parent.mkdir(parents=True, exist_ok=True)
            self._transition_marker.write_text("target-image replay in progress\n", encoding="utf-8")
            self._baseline_untrusted = True
        local_root = shlex.quote(str(self.repo))
        unavailable_artifacts: set[str] = set()
        inherited_environment = None
        for action in plan.actions:
            if action.kind.startswith("image-"):
                try:
                    image_executor.execute(
                        action,
                        lambda command, environment, identifier=action.id: self._run(
                            command, log_file, f"image replay action {identifier}", env=environment
                        ),
                    )
                    self._build_env_baseline.clear()
                    self._unknown_build_env.clear()
                except (OSError, RuntimeError, ValueError) as exc:
                    self.invalidate_baseline()
                    raise BuildTransitionError(f"target image replay failed at action {action.id}: {exc}") from exc
                continue
            if action.kind == "apply-base-image":
                if base_delta is None:
                    raise BuildTransitionError("base image action has no resolved snapshot")
                logger.info(
                    "[base-image] applying %d changed files from %s",
                    len(base_delta.changed_files),
                    base_delta.after.digest,
                )
                if log_file is not None:
                    with open(log_file, "a", encoding="utf-8") as out:
                        out.write(
                            f"[base-image] applying digest={base_delta.after.digest} "
                            f"files={len(base_delta.changed_files)} config={len(base_delta.changed_config)}\n"
                        )
                self._transition_marker.parent.mkdir(parents=True, exist_ok=True)
                self._transition_marker.write_text(
                    f"{base_delta.before.digest} -> {base_delta.after.digest}\n", encoding="utf-8"
                )
                self._baseline_untrusted = True
                try:
                    apply_base_delta(base_delta, Path("/"))
                except (OSError, RuntimeError) as exc:
                    self.invalidate_baseline()
                    raise BuildTransitionError(f"base image application failed: {exc}") from exc
                if log_file is not None:
                    with open(log_file, "a", encoding="utf-8") as out:
                        out.write("[base-image] applied\n")
                continue
            if inherited_environment is None and plan.target_environment is not None:
                inherited_environment = self._inherited_build_environment(plan, env, log_file)
            if action.kind in {"set-build-environment", "restore-build-environment"}:
                if action.kind == "set-build-environment":
                    env.update(build_environment(environment_assignments(action.command), inherited_environment or env))
                if log_file is not None:
                    with open(log_file, "a", encoding="utf-8") as out:
                        out.write(f"[build-environment] action {action.id} {action.kind}: {action.command}\n")
                continue
            command = action.command.replace("-e /vllm-workspace/vllm-ascend/", f"-e {local_root}/")
            inputs = "\n".join((command, *action.shell_context))
            unavailable = next(
                (
                    path
                    for path in sorted(unavailable_artifacts)
                    if re.search(r"(?<![\w./-])" + re.escape(path.rstrip("/")) + r"(?=$|[/\s\"'\[\];|&)])", inputs)
                ),
                None,
            )
            if unavailable and action.kind != "git-sync":
                warning = f"action {action.id} omitted: optional input unavailable: {unavailable}"
                logger.warning("[build-chain] %s", warning)
                if log_file is not None:
                    with open(log_file, "a", encoding="utf-8") as out:
                        out.write(f"[build-chain:warning] {warning}\n")
                if action.kind in {"extract", "prepare-directory"} and action.artifact is not None:
                    unavailable_artifacts.add(action.artifact)
                continue
            context = [
                f"cd {local_root}" if item == "cd /vllm-workspace/vllm-ascend" else item
                for item in action.shell_context
            ]
            script = " && ".join(["set -e", *context, command])
            warning_only = action.optional
            try:
                self._run(
                    ["/bin/bash", "-c", script],
                    log_file,
                    f"build-chain action {action.id}",
                    env=build_environment(action.environment, inherited_environment)
                    if inherited_environment is not None
                    else env,
                    invalidate_on_failure=not warning_only,
                )
            except BuildTransitionError as exc:
                if not warning_only:
                    raise
                if action.kind in ("download", "git-sync") and action.artifact is not None:
                    unavailable_artifacts.add(action.artifact)
                warning = f"optional {action.kind} action {action.id} failed; continuing: {exc}"
                if action.kind == "git-sync":
                    warning += f"\nTarget repository was not prepared: {action.artifact}; existing version retained."
                logger.warning("[build-chain] %s", warning)
                if log_file is not None:
                    with open(log_file, "a", encoding="utf-8") as out:
                        out.write(f"[build-chain:warning] {warning}\n")
            else:
                if (
                    action.kind in {"download", "git-sync", "extract", "prepare-directory"}
                    and action.artifact is not None
                ):
                    unavailable_artifacts.difference_update(
                        path
                        for path in tuple(unavailable_artifacts)
                        if path == action.artifact or path.startswith(action.artifact.rstrip("/") + "/")
                    )
        if plan.target_environment is not None:
            if inherited_environment is None:
                inherited_environment = self._inherited_build_environment(plan, env, log_file)
            final_environment = build_environment(plan.target_environment, inherited_environment)
            env.clear()
            env.update(final_environment)

    def _report_image_progress(self, message: str, log_file: Path | None) -> None:
        logger.info("%s", message)
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as out:
                out.write(message + "\n")
                out.flush()

    def invalidate_baseline(self) -> None:
        """A partially applied transition cannot be compared with the old environment."""
        self.installed_commit = None
        self.last_built_commit = None
        self._baseline_untrusted = True
        self._plan_cache.clear()
        self._base_delta_cache.clear()
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
        invalidate_on_failure: bool = True,
    ) -> None:
        if len(cmd) >= 3 and Path(cmd[0]).name in {"bash", "sh", "dash"} and cmd[1] in {"-c", "-lc", "-cl"}:
            protected = [self.repo.resolve().as_posix(), (Path.home() / ".cache" / "nightly_bisect").as_posix()]
            protected.append(self._transition_marker.parent.resolve().as_posix())
            if log_file is not None:
                protected.append(log_file.resolve().parent.as_posix())
            script, _ = protect_replay_cleanup(cmd[2], tuple(protected))
            cmd = [*cmd[:2], script, *cmd[3:]]
        logger.info("[build] running: %s", " ".join(cmd))
        workdir = cwd or self.repo
        if log_file is not None:
            install_command = "pip install" in shlex.join(cmd)
            with open(log_file, "a", encoding="utf-8") as out:
                out.write(f"\n[build-command] {label}: {shlex.join(cmd)}\n")
                if install_command:
                    out.write(self._cmake_python_diagnostics("before install"))
                out.flush()
                command_offset = log_file.stat().st_size
                proc = run_streaming(cmd, cwd=str(workdir), output=out, env=env)
            tail = "(see build log)"
            if proc.returncode != 0:
                # Capture this command's output before appending diagnostics;
                # stale CMake logs must not hide a download or shell failure.
                with contextlib.suppress(OSError), open(log_file, "rb") as output:
                    output.seek(max(command_offset, log_file.stat().st_size - 4000))
                    tail = output.read().decode("utf-8", errors="replace") or "(command produced no output)"
                with open(log_file, "a", encoding="utf-8") as out:
                    out.write(f"\n[build-command] {label}: exit_code={proc.returncode}\n")
                    if install_command:
                        out.write(self._cmake_python_diagnostics("after failed install"))
                        out.write(self._cmake_failure_logs())
        else:
            proc = run_streaming(cmd, cwd=str(workdir), env=env)
            tail = (proc.stdout or "")[-2000:]
        if proc.returncode != 0:
            if invalidate_on_failure:
                self.invalidate_baseline()
            raise BuildTransitionError(f"{label} failed (rc={proc.returncode}): {shlex.join(cmd)}\n{tail}")

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
