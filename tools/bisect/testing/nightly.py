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
"""Compatibility with nightly's good table; not part of candidate selection."""

import contextlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

import psutil

from tools.bisect.execution.process_output import run_streaming
from tools.bisect.orchestration.config import SCENE_SINGLE, SINGLE_NODE_TEST_PATH, BisectInput, BisectOptions
from tools.bisect.testing.executor import execute_test
from tools.bisect.testing.good_table import GoodTable

logger = logging.getLogger(__name__)


def resolve_good(inp, options) -> str:
    if inp.good_commit:
        return inp.good_commit
    if inp.test_entry is not None:
        raise ValueError("Command entries require an explicit good commit")
    entry = GoodTable(options.good_table_path).lookup_last_good(
        name=inp.name,
        config_yaml=inp.config_yaml,
        soc=inp.soc,
        scene=inp.scene,
    )
    if entry is None:
        raise SystemExit(
            f"No successful good-table row for name={inp.name!r} / "
            f"config_yaml={inp.config_yaml!r} in {options.good_table_path}, and --good-commit was not given."
        )
    inp.good_vllm_commit = entry.vllm_commit or None
    if entry.vllm_commit:
        logger.info("Good row pairs vllm-ascend %s with vLLM %s", entry.vllm_ascend_commit[:12], entry.vllm_commit[:12])
    return entry.vllm_ascend_commit


_MULTI_CONFIG_BASES = (
    "tests/e2e/nightly/multi_node/internal_dp/config",
    "tests/e2e/nightly/multi_node/external_dp/config",
)


def resolve_nightly_node_count(repo: Path, config_yaml: str, config_base_path: str | None) -> int:
    """Read participant metadata only for the existing nightly YAML adapter."""
    import yaml  # Load YAML support only when the nightly adapter needs it.

    bases = [config_base_path] if config_base_path else _MULTI_CONFIG_BASES
    for directory in bases:
        path = repo / directory / config_yaml
        if not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: invalid nightly YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path}: nightly configuration must be a mapping")
        value = data.get("num_nodes")
        if value is None:
            continue
        if isinstance(value, str) and value.isdecimal():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{path}: num_nodes must be a positive integer")
        logger.info("Resolved num_nodes=%d from %s", value, path)
        return value
    raise ValueError(f"Could not determine --num-nodes from {config_yaml!r}; pass --num-nodes explicitly")


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


def prepare_nightly_environment(
    inp: BisectInput, repo: Path, test_environment: dict[str, str] | None
) -> dict[str, str]:
    env = dict(test_environment or os.environ)
    env["CONFIG_YAML_PATH"] = inp.config_yaml
    config_base_path = inp.config_base_path
    if inp.scene == SCENE_SINGLE and config_base_path:
        requested = repo / config_base_path / inp.config_yaml
        legacy_base = repo / _LEGACY_SINGLE_NODE_CONFIG_DIR
        legacy_config = legacy_base / Path(inp.config_yaml).name
        if not requested.is_file() and legacy_config.is_file():
            config_base_path = str(legacy_base)
            env["CONFIG_YAML_PATH"] = legacy_config.name
            logger.info("[single] config moved in candidate; using %s", legacy_config)
    if config_base_path:
        env["CONFIG_BASE_PATH"] = config_base_path
    return env


def reset_results_directory(path: Path) -> Path:
    """Keep benchmark evidence scoped to the current test execution."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_nightly_command(command: list[str], repo: Path, env: dict[str, str], log_path: Path, timeout: float) -> int:
    logger.info("[nightly] running: %s", " ".join(command))
    with log_path.open("a", encoding="utf-8") as output:
        output.write("\n$ " + " ".join(command) + "\n")
        output.flush()
        try:
            return run_streaming(command, cwd=repo, env=env, output=output, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            logger.error("[nightly] test timed out after %ss", timeout)
            return 124


def run_single_test(repo: Path, env: dict[str, str], log_path: Path, timeout: float) -> int:
    command = ["python", "-u", "-m", "pytest", "-sv", "--show-capture=no", _single_node_test_path(repo)]
    return run_nightly_command(command, repo, env, log_path, timeout)


def run_multi_test(inp: BisectInput, opt: BisectOptions, env: dict[str, str], log_path: Path, job: str) -> int:
    env = dict(env)
    env["BENCHMARK_JOB_NAME"] = job
    env.setdefault("BENCHMARK_HOME", str(opt.repo_dir / "benchmark"))
    env.setdefault("LWS_WORKER_INDEX", str(opt.node_index))
    sources = " ; ".join(f"source {path} 2>/dev/null || true" for path in _ENV_SOURCE_FILES)
    command = (
        f"set -e ; {sources} ; exec python -u -m pytest -sv --show-capture=no "
        f"{_multi_node_test_path(opt.repo_dir, inp)}"
    )
    return run_nightly_command(["bash", "-c", command], opt.repo_dir, env, log_path, opt.trial_timeout_s)


def run_worker_test(
    inp: BisectInput, opt: BisectOptions, log_path: Path, test_environment: dict[str, str] | None
) -> int:
    if inp.test_entry is not None:
        outcome = execute_test(inp.test_entry, opt.repo_dir, test_environment, log_path)
        return 125 if outcome.infra_error else outcome.exit_code
    env = prepare_nightly_environment(inp, opt.repo_dir, test_environment)
    env["LWS_WORKER_INDEX"] = str(opt.node_index)
    return run_multi_test(inp, opt, env, log_path, _safe_name(inp.config_yaml))
