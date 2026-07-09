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
"""Constants and data models shared across the bisect tool."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# --------------------------------------------------------------------------- #
# Scene / verdict vocabulary
# --------------------------------------------------------------------------- #
SCENE_SINGLE = "single_node"
SCENE_MULTI = "multi_node"
SCENES = (SCENE_SINGLE, SCENE_MULTI)

# PASS  -> commit is good for this case
# FAIL  -> commit reproduces the failure
# SKIP  -> commit could not be judged (build broken, infra error, ...).
#          Treated like ``git bisect skip`` and excluded from the search.
Verdict = Literal["PASS", "FAIL", "SKIP"]

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Repository root (……/vllm-ascend). This file lives at
# tools/bisect/config.py -> 2 parents up is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Nightly launch entries we reuse instead of re-implementing.
SINGLE_NODE_TEST_PATH = "tests/e2e/nightly/single_node/models/scripts/test_single_node.py"
MULTI_NODE_RUN_SH = "tests/e2e/nightly/multi_node/scripts/run.sh"

# The "good table" lives on a fixed nightly path (PVC / shared cache). It is a
# plain CSV so it can be eyeballed and edited by hand. Override with
# ``BISECT_GOOD_TABLE``. Falls back to a repo-local sample for dev runs.
DEFAULT_GOOD_TABLE = os.getenv(
    "BISECT_GOOD_TABLE",
    "/root/.cache/nightly_bisect/good_table.csv",
)
SAMPLE_GOOD_TABLE = str(Path(__file__).resolve().parent / "good_table.sample.csv")

# Where per-run artifacts (logs, state, final report) are written.
DEFAULT_WORK_DIR = os.getenv("BISECT_WORK_DIR", "/root/.cache/nightly_bisect/runs")

# Shared directory used as the cross-node barrier in multi-node bisects. It must
# point at the same (network) filesystem on every node.
DEFAULT_COORD_DIR = os.getenv("BISECT_COORD_DIR", "/root/.cache/nightly_bisect/coord")

# --------------------------------------------------------------------------- #
# Native-code detection
# --------------------------------------------------------------------------- #
# A commit touching any of these (relative to the last successfully built
# commit) forces a ``pip install`` rebuild. Everything else (``.py`` / yaml /
# md) is picked up live by the editable install, so a plain checkout suffices.
NATIVE_GLOBS = (
    "*.cpp",
    "*.cc",
    "*.cxx",
    "*.c",
    "*.cu",
    "*.h",
    "*.hpp",
    "*.hxx",
    "*.cuh",
    "*.inc",
    "*.cmake",
    "CMakeLists.txt",
    "*/CMakeLists.txt",
    "csrc/*",
    "csrc/**",
    "**/*.so",
)

# Build-definition files: a change here may alter compile flags or build deps,
# so we also force a rebuild.
BUILD_DEF_GLOBS = (
    "setup.py",
    "pyproject.toml",
    "setup.cfg",
    "CMakeLists.txt",
)

# A change to any of these triggers a dependency reinstall (separate from the
# native rebuild, see BuildManager).
REQUIREMENTS_GLOBS = (
    "requirements*.txt",
    "requirements/*.txt",
)


@dataclass(frozen=True)
class Candidate:
    """A single commit in the bisect search space."""

    commit: str  # full 40-char sha
    pr_number: str | None  # parsed from "(#NNNN)" in the subject, if any
    subject: str  # commit subject line

    @property
    def short(self) -> str:
        return self.commit[:12]

    @property
    def label(self) -> str:
        """Human-facing identifier used in [PASS]/[FAIL] markers."""
        if self.pr_number:
            return f"PR-{self.pr_number}"
        return f"commit-{self.short}"


@dataclass
class TrialResult:
    """Outcome of validating one candidate."""

    candidate: Candidate
    verdict: Verdict
    duration_s: float = 0.0
    rebuilt: bool = False
    exit_code: int | None = None
    log_path: str | None = None
    note: str = ""


@dataclass
class BisectInput:
    """Everything needed to start a bisect run, assembled by the CLI.

    A nightly trial runs at *whole-YAML* granularity (all ``test_cases`` in the
    file) -- nightly cannot select a single case -- so there is no per-case
    field. ``config_yaml`` both selects the failing file (CONFIG_YAML_PATH) and
    is the unit a trial runs and judges.
    """

    scene: str  # SCENE_SINGLE | SCENE_MULTI
    config_yaml: str  # CONFIG_YAML_PATH value: the whole failing case file
    bad_commit: str  # current failing commit (resolved to full sha later)
    name: str | None = None  # nightly case name, to match the good-table 'name'
    config_base_path: str | None = None  # CONFIG_BASE_PATH override (multi/ext dp)
    good_commit: str | None = None  # explicit good; else looked up in the table
    good_vllm_commit: str | None = None  # paired vLLM commit of the good row

    @property
    def case_key(self) -> str:
        """Stable key (scene + whole yaml) for state/work-dir/report."""
        return f"{self.scene}::{self.config_yaml}"


@dataclass
class BisectOptions:
    """Tunables that change how the search runs (not what it searches)."""

    repo_dir: Path = REPO_ROOT
    work_dir: str = DEFAULT_WORK_DIR
    coord_dir: str = DEFAULT_COORD_DIR
    good_table_path: str = DEFAULT_GOOD_TABLE
    # Re-confirm a FAIL this many extra times before trusting it (flaky guard).
    fail_confirm_retries: int = 1
    # Verify the endpoints before searching (good must PASS, bad must FAIL).
    verify_good: bool = True
    verify_bad: bool = True
    # Editable reinstall command pieces. --no-input avoids any interactive
    # prompt that could hang the (silent, log-redirected) build step.
    pip_install_cmd: list[str] = field(
        default_factory=lambda: [
            "pip",
            "install",
            "-e",
            ".",
            "--no-input",
            "--disable-pip-version-check",
        ]
    )
    pip_requirements_cmd: list[str] = field(
        default_factory=lambda: [
            "pip",
            "install",
            "-r",
            "requirements-dev.txt",
            "--no-input",
            "--disable-pip-version-check",
        ]
    )
    # The nightly container is already built+installed at its current HEAD (the
    # bad commit that just failed). Treating HEAD as the established build
    # baseline makes the bad endpoint a checkout-only (no rebuild / no slow
    # requirements reinstall) trial. Set force_initial_build=True to opt back
    # into a conservative clean rebuild on the first trial.
    assume_built_head: bool = True
    force_initial_build: bool = False
    # How to decide whether a rebuild is needed for each bisected commit:
    #   "per-commit" (default): inspect only that commit's own diff (the files
    #       that commit changed) -> compile iff it touched native/cpp files.
    #   "since-build": inspect every file changed between the last successfully
    #       built commit and the target -> safe against bisect jumps (a native
    #       change in a jumped-over commit still triggers a rebuild).
    native_check: str = "per-commit"
    # Multi-node coordination.
    num_nodes: int = 1
    node_index: int = 0
    barrier_timeout_s: float = 3600.0
    # Optional external "leader finished" sentinel file. When the AOP pipeline's
    # leader decides NOT to bisect (env/age skip) it never starts the coordinator,
    # so a worker would otherwise wait out the full barrier timeout. If set, the
    # worker also exits promptly once this file appears (run.sh touches its
    # ``done`` file on every leader exit path).
    release_file: str | None = None
    # Per-trial pytest timeout (seconds).
    trial_timeout_s: float = 5400.0

    @property
    def is_master(self) -> bool:
        return self.node_index == 0
