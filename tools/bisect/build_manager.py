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

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tools.bisect import git_ops
from tools.bisect.config import (
    BUILD_DEF_GLOBS,
    NATIVE_GLOBS,
    REQUIREMENTS_GLOBS,
    BisectOptions,
)

logger = logging.getLogger(__name__)

# Build artifacts removed before a rebuild so stale ``.so`` files can never be
# loaded against a new checkout.
_STALE_ARTIFACTS = ("build", "*.egg-info", "dist")


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

    # ------------------------------------------------------------ decision
    def decide(self, target_commit: str) -> BuildDecision:
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
        if self.last_built_commit == target_commit:
            return BuildDecision(
                rebuild=False,
                reinstall_reqs=False,
                native_hits=[],
                reason="already built this exact commit",
            )

        # Choose which set of changed files drives the decision.
        if self.opt.native_check == "per-commit":
            files = git_ops.commit_changed_files(self.repo, target_commit)
            scope = "this commit"
        else:
            files = git_ops.changed_files(self.repo, self.last_built_commit, target_commit)
            scope = "since last build"

        native = git_ops.matches_any(files, NATIVE_GLOBS + BUILD_DEF_GLOBS)
        reqs = git_ops.matches_any(files, REQUIREMENTS_GLOBS)

        if native:
            reason = f"native/build-def files changed ({scope}): {native[:5]}"
            return BuildDecision(True, bool(reqs), native, reason)
        if reqs:
            return BuildDecision(False, True, [], f"requirements changed ({scope}): {reqs}")
        return BuildDecision(
            rebuild=False,
            reinstall_reqs=False,
            native_hits=[],
            reason=f"no native/cpp changes ({scope}) -> editable install, no compile",
        )

    # ------------------------------------------------------------- prepare
    def prepare(self, target_commit: str, log_file: Path | None = None) -> BuildDecision:
        """Checkout ``target_commit`` and rebuild/reinstall as needed.

        Raises ``BuildError`` on a failed install so the caller can record a
        SKIP (rather than a misleading FAIL) for this commit.
        """
        decision = self.decide(target_commit)
        logger.info("[build] %s -> %s", target_commit[:12], decision.reason)

        git_ops.checkout(self.repo, target_commit)

        if (decision.reinstall_reqs or decision.rebuild) and log_file is not None:
            logger.info(
                "[build] compiling/installing (this can take a while); follow progress with: tail -f %s", log_file
            )
        if decision.reinstall_reqs:
            self._run(self.opt.pip_requirements_cmd, log_file, "pip install requirements")
        if decision.rebuild:
            self._clean_artifacts()
            self._run(self.opt.pip_install_cmd, log_file, "pip install -e .")

        # Only advance the baseline once the binary actually matches the source.
        if decision.rebuild or self.last_built_commit is None:
            self.last_built_commit = target_commit
        return decision

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

    def _run(self, cmd: list[str], log_file: Path | None, label: str) -> None:
        logger.info("[build] running: %s", " ".join(cmd))
        if log_file is not None:
            with open(log_file, "a", encoding="utf-8") as out:
                proc = subprocess.run(cmd, cwd=str(self.repo), stdout=out, stderr=subprocess.STDOUT, text=True)
            tail = "(see build log)"
        else:
            proc = subprocess.run(cmd, cwd=str(self.repo), capture_output=True, text=True)
            tail = (proc.stdout or "")[-2000:]
        if proc.returncode != 0:
            raise BuildError(f"{label} failed (rc={proc.returncode}):\n{tail}")


class BuildError(RuntimeError):
    pass
