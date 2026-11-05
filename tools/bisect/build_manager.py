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
* If it only touches ``requirements*``                     -> reinstall deps.
* Otherwise (pure ``.py`` / yaml / md)                     -> plain checkout;
  the editable install picks the change up live.
"""

import contextlib
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tools.bisect import git_ops
from tools.bisect.build_diff import command_fragments, compare_build_chains
from tools.bisect.build_ir import changed_effects
from tools.bisect.build_recipe import BuildRecipe, recipe_at_commit
from tools.bisect.config import (
    BUILD_DEF_GLOBS,
    NATIVE_GLOBS,
    REQUIREMENTS_GLOBS,
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
_ARG_ACTIONS = {
    "switch vLLM", "use for pip installs", "use for nightly pip installs",
    "reinstall runtime wheels", "switch AISBench",
}

def _chain_changes(previous: BuildRecipe, target: BuildRecipe) -> list[str]:
    """Reject changed Docker operations unless a known adapter handles the input."""
    unsupported = []
    effects = changed_effects(previous, target)
    for layer, old_steps, new_steps in (
        ("base", previous.root_chain, target.root_chain),
        ("nightly", previous.nightly_chain, target.nightly_chain),
    ):
        for change in compare_build_chains(old_steps, new_steps):
            step = change.new or change.old
            index = (change.new_index if change.new_index is not None else change.old_index) + 1
            if change.kind == "input":
                old_value = dict(change.old.inputs).get(change.argument, "")
                new_value = dict(change.new.inputs).get(change.argument, "")
                covered = any(
                    after.action in _ARG_ACTIONS
                    and (not old_value or old_value in before.value)
                    and (not new_value or new_value in after.value)
                    for before, after in effects
                )
                if not covered:
                    unsupported.append(f"{layer} step {index}: ARG {change.argument} changed without a replayable effect")
            elif change.kind == "command":
                fragments = command_fragments(change.old.command, change.new.command)
                unsupported.append(f"{layer} step {index} {step.instruction} changed: {'; '.join(fragments[:3])}")
            else:
                unsupported.append(f"{layer} step {index} {change.kind}: {step.instruction} {step.command[:120]}")
    return unsupported


def _chain_delta(previous: BuildRecipe, target: BuildRecipe) -> list[str]:
    lines = []
    for layer, old_steps, new_steps in (
        ("base", previous.root_chain, target.root_chain),
        ("nightly", previous.nightly_chain, target.nightly_chain),
    ):
        for change in compare_build_chains(old_steps, new_steps):
            if change.kind == "input":
                before = dict(change.old.inputs).get(change.argument) or "<unset>"
                after = dict(change.new.inputs).get(change.argument) or "<unset>"
                lines.append(f"[build-chain] {layer} step {change.new_index + 1} {change.argument}: {before} -> {after}")
    for old, new in changed_effects(previous, target):
        lines.append(f"[build-chain] effect {new.name}: action={new.action}")
    return lines or ["[build-chain] no effective Docker build input changed"]


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
        self.installed_recipe: BuildRecipe | None = None
        if self.last_built_commit and options.soc:
            self.installed_recipe = recipe_at_commit(self.repo, self.last_built_commit, options.soc)

    def _candidate_recipe(self, commit: str) -> BuildRecipe | None:
        if not self.opt.soc:
            return None
        try:
            return recipe_at_commit(self.repo, commit, self.opt.soc)
        except ValueError as exc:
            raise BuildError(str(exc)) from exc

    def _check_recipe(self, recipe: BuildRecipe) -> None:
        installed = self.installed_recipe
        if installed is None:
            return
        if recipe.cann_image != installed.cann_image:
            raise BuildError(
                f"CANN base image changed: {installed.cann_image} -> {recipe.cann_image}; "
                "use a matching image for this candidate"
            )
        unsupported = _chain_changes(installed, recipe)
        if unsupported:
            raise BuildError("Docker build chain cannot be replayed in-place: " + "; ".join(unsupported[:5]))
        if recipe.ais_bench_url != installed.ais_bench_url:
            raise BuildError("AISBench repository URL changed; use a matching nightly image")
        if len(recipe.runtime_wheels) != len(installed.runtime_wheels):
            raise BuildError("the set of Dockerfile-installed wheels changed; use a matching image")
        if recipe.flash_attention_url != installed.flash_attention_url:
            raise BuildError("flash-attention-npu repository changed; use a matching image")

    # ------------------------------------------------------------ decision
    def decide(self, target_commit: str) -> BuildDecision:
        recipe = self._candidate_recipe(target_commit)
        if recipe is not None:
            self._check_recipe(recipe)
        if self.last_built_commit is None:
            # No trusted baseline: rebuild, but only reinstall requirements when
            # they are actually known-stale (we cannot diff, so be safe=False
            # and rely on the container already having dev deps).
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
            if recipe is not None else BUILD_DEF_GLOBS
        )
        native = git_ops.matches_any(files, NATIVE_GLOBS + build_def_globs)
        reqs = [] if recipe is not None else git_ops.matches_any(files, REQUIREMENTS_GLOBS)
        if recipe is not None and self.installed_recipe is not None:
            effects = [new for _, new in changed_effects(self.installed_recipe, recipe)]
            build_inputs = [
                effect.name for effect in effects if effect.action == "rebuild vLLM Ascend"
            ]
            native = native or build_inputs
            reqs = reqs or (
                ["requirements-nightly.txt"]
                if any(effect.action == "reinstall nightly requirements" for effect in effects)
                else []
            )

        if native:
            reason = f"native/build-def files changed ({scope}): {native[:5]}"
            return BuildDecision(True, bool(reqs), native, reason)
        if reqs:
            return BuildDecision(False, True, [], f"requirements changed ({scope}): {reqs}")
        if recipe is not None and self.installed_recipe is not None:
            if any(
                effect.action in ("reinstall runtime wheels", "rebuild extension", "switch AISBench")
                for effect in effects
            ):
                return BuildDecision(False, False, [], "Dockerfile wheel or source dependency changed")
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
        logger.info("[build] %s -> %s", target_commit[:12], decision.reason)

        target_recipe = self._candidate_recipe(target_commit)
        if target_recipe is not None and self.installed_recipe is not None and log_file is not None:
            with open(log_file, "a", encoding="utf-8") as out:
                out.write("\n".join(_chain_delta(self.installed_recipe, target_recipe)) + "\n")

        git_ops.checkout(self.repo, target_commit)

        recipe = target_recipe
        root_env = None
        nightly_env = None
        if recipe is not None:
            root_env = dict(os.environ)
            root_env.setdefault("PIP_INDEX_URL", TORCH_NPU_PIP_INDEX_URL)
            root_env["VLLM_BATCH_INVARIANT"] = "1"
            root_env["PIP_EXTRA_INDEX_URL"] = " ".join(
                url for url in (recipe.ascend_index_url, recipe.pytorch_index_url) if url
            )
            nightly_env = dict(os.environ)
            nightly_env.setdefault("PIP_INDEX_URL", TORCH_NPU_PIP_INDEX_URL)
            if recipe.nightly_index_url:
                nightly_env["PIP_EXTRA_INDEX_URL"] = recipe.nightly_index_url

        if (decision.reinstall_reqs or decision.rebuild) and log_file is not None:
            logger.info(
                "[build] compiling/installing (this can take a while); follow progress with: tail -f %s", log_file
            )
        if decision.rebuild:
            self._clean_artifacts()
            install_cmd = self.opt.pip_install_cmd
            if recipe is not None:
                install_cmd = [
                    "bash",
                    "-lc",
                    "source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
                    "source /usr/local/Ascend/nnal/atb/set_env.sh && "
                    f"exec {shlex.join(install_cmd)}",
                ]
            self._run(install_cmd, log_file, "pip install -e .", env=root_env)
        if recipe is not None and self.installed_recipe is not None:
            if recipe.runtime_wheels != self.installed_recipe.runtime_wheels:
                for wheel in recipe.runtime_wheels:
                    self._run(
                        ["python3", "-m", "pip", "install", "--force-reinstall", "--no-deps", wheel],
                        log_file,
                        f"install Dockerfile wheel {wheel.rsplit('/', 1)[-1]}",
                        env=root_env,
                    )
            if recipe.flash_attention_ref != self.installed_recipe.flash_attention_ref:
                self._switch_flash_attention(recipe, log_file, root_env)
        if decision.reinstall_reqs:
            self._run(self.opt.pip_requirements_cmd, log_file, "pip install nightly requirements", env=nightly_env)
        if recipe is not None and self.installed_recipe is not None:
            if recipe.ais_bench_tag != self.installed_recipe.ais_bench_tag:
                self._switch_ais_bench(recipe, log_file, nightly_env)

        # Only advance the baseline once the binary actually matches the source.
        if decision.rebuild or self.last_built_commit is None:
            self.last_built_commit = target_commit
        if recipe is not None:
            self.installed_recipe = recipe
        return decision

    def _switch_flash_attention(
        self, recipe: BuildRecipe, log_file: Path | None, env: dict[str, str] | None
    ) -> None:
        if not recipe.flash_attention_url or not recipe.flash_attention_ref:
            raise BuildError("flash-attention-npu build was removed; use a matching image")
        with tempfile.TemporaryDirectory(prefix="bisect-flash-attn-") as tmp:
            source = Path(tmp) / "flash-attention-npu"
            self._run(
                ["git", "clone", recipe.flash_attention_url, str(source)],
                log_file,
                "clone flash-attention-npu",
            )
            self._run(["git", "checkout", recipe.flash_attention_ref], log_file, "checkout flash-attention-npu", cwd=source)
            self._run(
                ["git", "submodule", "update", "--init", "--recursive"],
                log_file,
                "update flash-attention-npu submodules",
                cwd=source,
            )
            build_env = dict(os.environ if env is None else env)
            build_env["FLASH_ATTN_BUILD_NPU"] = "910"
            self._run(
                [
                    "bash", "-lc",
                    "source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
                    "source /usr/local/Ascend/nnal/atb/set_env.sh && "
                    "source /usr/local/Ascend/cann/set_env.sh && "
                    "exec python3 setup.py install",
                ],
                log_file,
                "build flash-attention-npu",
                env=build_env,
                cwd=source,
            )

    def _switch_ais_bench(self, recipe: BuildRecipe, log_file: Path | None, env: dict[str, str] | None) -> None:
        benchmark = self.repo / "benchmark"
        if not (benchmark / ".git").exists():
            raise BuildError(f"AISBench checkout missing at {benchmark}")
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=benchmark, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise BuildError(f"cannot inspect AISBench checkout at {benchmark}: {exc}") from exc
        if status.stdout.strip():
            raise BuildError(f"AISBench checkout has local changes at {benchmark}")
        try:
            target = git_ops.resolve_commit(benchmark, recipe.ais_bench_tag)
            git_ops.checkout(benchmark, target)
        except GitError as exc:
            raise BuildError(f"cannot check out AISBench {recipe.ais_bench_tag}: {exc}") from exc
        self._run(
            ["python3", "-m", "pip", "install", "-e", ".", "-r", "requirements/api.txt", "-r", "requirements/extra.txt"],
            log_file,
            f"install AISBench {recipe.ais_bench_tag}",
            env=env,
            cwd=benchmark,
        )

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
            raise BuildError(f"{label} failed (rc={proc.returncode}):\n{tail}")

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


# Everything a deploy (resolve + checkout + build + version adaptation) can
# raise that must surface as a per-round SKIP instead of killing the driving
# agent. One shared tuple so every catch site (master runner, worker agent,
# orchestrator trial loop) stays in lockstep -- a missed copy leaves the other
# side waiting out the full barrier timeout on every remaining round.
DEPLOY_ERRORS = (BuildError, GitError, VersionAdaptationError)
