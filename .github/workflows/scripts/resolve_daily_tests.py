#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#
"""Resolve daily nightly test cases for daytime regression runs.

Reads a daily config YAML (either the repo's ``daily_config.yaml`` or an
inline YAML string materialized by the workflow) and ``nightly_config.yaml``,
then resolves the ``daily_test_cases`` workflow input into per-SOC,
comma-separated test-name lists that are dispatched to the
``schedule_nightly_test_<soc>.yaml`` workflows.

Every selected ``name`` is validated against ``nightly_config.yaml`` so a daily
entry referencing a removed test is surfaced as a warning instead of silently
dispatching a no-op run.

Supported ``--test-cases`` formats:
  - JSON dict:      {"a2": ["multi-node-qwen3-235b-dp"], "a3": ["daily-all"]}
                    Exact per-SOC control; absent SOCs are not dispatched.
  - daily-<soc> token: daily-all / daily-a2 / daily-a3 / ...
                    Expands to every daily name configured for that SOC.
  - comma-separated test names:
                    multi-node-qwen3-235b-dp,deepseek-r1-0528-w8a8
                    Matched by test name across all SOCs.

Writes GITHUB_OUTPUT (or stdout when unset):
  - has_daily_<soc>=true|false
  - daily_<soc>_tests=<comma-separated test names>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

try:
    import yaml
except ImportError:
    subprocess.check_call(["pip3", "install", "pyyaml", "-q"])
    import yaml

# SOCs that map 1:1 to the schedule_nightly_test_<soc>.yaml workflows.
SUPPORTED_SOCS = ("a2", "a3", "a3-560t", "a5")

DAILY_TOKEN_PREFIX = "daily-"
DAILY_ALL_TOKEN = "daily-all"


def _load_yaml(path):
    """Load a YAML file, tolerating an empty/whitespace file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _collect_ordered_names(node, ordered, seen):
    """Recursively collect ``name`` fields, preserving declaration order."""
    if isinstance(node, dict):
        for value in node.values():
            _collect_ordered_names(value, ordered, seen)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                name = item["name"]
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)
            elif isinstance(item, (dict, list)):
                _collect_ordered_names(item, ordered, seen)


def _names_by_soc(config):
    """Return {soc: [ordered names]} for every SOC present in the config."""
    result = {}
    for soc in SUPPORTED_SOCS:
        soc_block = config.get(soc)
        if not isinstance(soc_block, dict):
            continue
        ordered: list[str] = []
        _collect_ordered_names(soc_block, ordered, set())
        if ordered:
            result[soc] = ordered
    return result


def _daily_names_by_soc(daily_config):
    """Return {soc: [ordered names]} from the daily config."""
    return _names_by_soc(daily_config)


def _nightly_ordered_by_soc(nightly_config):
    """Return {soc: [ordered names]} from nightly_config.yaml (ordered)."""
    result = {}
    for soc in SUPPORTED_SOCS:
        soc_block = nightly_config.get(soc)
        if not isinstance(soc_block, dict):
            continue
        ordered: list[str] = []
        _collect_ordered_names(soc_block, ordered, set())
        if ordered:
            result[soc] = ordered
    return result


def _nightly_names_by_soc(nightly_config):
    """Return {soc: set(names)} from nightly_config.yaml (all sections)."""
    ordered = _nightly_ordered_by_soc(nightly_config)
    return {soc: set(names) for soc, names in ordered.items()}


def _expand_spec(raw, daily_by_soc):
    """Parse the ``daily_test_cases`` input into {soc: [requested tokens]}.

    Returns spec where spec maps each SOC to the list of requested tokens
    (test names or the ``daily-all`` shortcut).
    """
    raw = (raw or "").strip()
    spec = {}
    if not raw or raw == DAILY_ALL_TOKEN:
        for soc in daily_by_soc:
            spec[soc] = [DAILY_ALL_TOKEN]
        return spec

    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"::error::Invalid JSON for daily_test_cases: {exc}")
        if not isinstance(parsed, dict):
            raise SystemExit("::error::daily_test_cases JSON must be an object mapping SOC to an array of test names.")
        for soc, requested in parsed.items():
            if not isinstance(requested, (str, list)):
                print(f"::warning::daily_test_cases value for '{soc}' is not a string or list; skipped.")
                continue
            spec[str(soc)] = [requested] if isinstance(requested, str) else requested
        return spec

    # Comma-separated tokens: daily-<soc> shortcuts and/or plain test names.
    for token in (t.strip() for t in raw.split(",") if t.strip()):
        if token == DAILY_ALL_TOKEN or token.startswith(DAILY_TOKEN_PREFIX):
            if token == DAILY_ALL_TOKEN:
                for soc in daily_by_soc:
                    spec.setdefault(soc, []).append(DAILY_ALL_TOKEN)
            else:
                soc = token[len(DAILY_TOKEN_PREFIX) :]
                if soc in daily_by_soc:
                    spec.setdefault(soc, []).append(DAILY_ALL_TOKEN)
                else:
                    print(f"::warning::Unknown daily soc '{soc}' in '{token}'; skipped.")
        else:
            matched = False
            for soc, names in daily_by_soc.items():
                if token in names:
                    spec.setdefault(soc, []).append(token)
                    matched = True
            if not matched:
                print(f"::warning::Daily test name '{token}' is not configured for any SOC; skipped.")
    return spec


def _resolve(raw, daily_by_soc, nightly_by_soc):
    """Return {soc: [validated, ordered test names]} to dispatch."""
    spec = _expand_spec(raw, daily_by_soc)
    known_all = set()
    for nightly_names in nightly_by_soc.values():
        known_all |= nightly_names

    result = {}
    for soc, requested in spec.items():
        if soc not in daily_by_soc:
            print(f"::warning::SOC '{soc}' has no daily tests configured; skipped.")
            continue
        daily_ordered = daily_by_soc[soc]
        daily_names_set = set(daily_ordered)
        want = set()
        for item in requested:
            if item == DAILY_ALL_TOKEN:
                want |= daily_names_set
            elif item in daily_names_set:
                want.add(item)
            else:
                print(f"::warning::Daily test name '{item}' is not configured for SOC '{soc}'; skipped.")
        missing = want - known_all
        for name in sorted(missing):
            print(f"::warning::Daily test name '{name}' not found in nightly_config.yaml; skipped.")
        want -= missing
        if want:
            result[soc] = [name for name in daily_ordered if name in want]
    return result


def _resolve_direct(raw, nightly_ordered):
    """Resolve ``daily_test_cases`` directly against nightly_config.yaml.

    Used when ``daily_test_cases`` is non-empty: the input IS the test spec and
    ``daily_config_yaml`` is not consulted. Names are validated against
    ``nightly_config.yaml`` and routed to the SOC where they are defined.

    ``nightly_ordered``: {soc: [ordered names]}.
    """
    nightly_sets = {soc: set(names) for soc, names in nightly_ordered.items()}

    if raw.startswith("{"):
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"::error::Invalid JSON for daily_test_cases: {exc}")
        if not isinstance(spec, dict):
            raise SystemExit("::error::daily_test_cases JSON must be an object mapping SOC to an array of test names.")
        result = {}
        for soc, requested in spec.items():
            if soc not in nightly_sets:
                print(f"::warning::SOC '{soc}' has no tests in nightly_config.yaml; skipped.")
                continue
            all_names = nightly_sets[soc]
            if isinstance(requested, str):
                requested = [requested]
            want = set()
            for item in requested:
                if item == DAILY_ALL_TOKEN:
                    want |= all_names
                elif item in all_names:
                    want.add(item)
                else:
                    print(f"::warning::Test name '{item}' not found for SOC '{soc}' in nightly_config.yaml; skipped.")
            if want:
                result[soc] = [name for name in nightly_ordered[soc] if name in want]
        return result

    # Comma-separated tokens: daily-<soc> shortcuts and/or plain test names.
    spec = {}
    for token in (t.strip() for t in raw.split(",") if t.strip()):
        if token == DAILY_ALL_TOKEN or token.startswith(DAILY_TOKEN_PREFIX):
            if token == DAILY_ALL_TOKEN:
                for soc in nightly_sets:
                    spec[soc] = DAILY_ALL_TOKEN
            else:
                soc = token[len(DAILY_TOKEN_PREFIX) :]
                if soc in nightly_sets:
                    spec[soc] = DAILY_ALL_TOKEN
                else:
                    print(f"::warning::Unknown SOC '{soc}' in '{token}'; skipped.")
        else:
            matched = False
            for soc, names in nightly_sets.items():
                if token in names:
                    spec.setdefault(soc, set()).add(token)
                    matched = True
            if not matched:
                print(f"::warning::Test name '{token}' not found in nightly_config.yaml for any SOC; skipped.")

    result = {}
    for soc, requested in spec.items():
        if requested == DAILY_ALL_TOKEN:
            result[soc] = list(nightly_ordered[soc])
        else:
            valid = [name for name in nightly_ordered[soc] if name in requested]
            if valid:
                result[soc] = valid
    return result


def _emit_output(resolved):
    """Write per-SOC outputs to GITHUB_OUTPUT, or stdout when unset."""
    lines = []
    for soc in SUPPORTED_SOCS:
        key = soc.replace("-", "_")
        names = resolved.get(soc, [])
        lines.append(f"has_daily_{key}={str(bool(names)).lower()}")
        lines.append(f"daily_{key}_tests={','.join(names)}")
    text = "\n".join(lines) + "\n"
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


def main():
    parser = argparse.ArgumentParser(
        description="Resolve daily nightly test cases for dispatch.",
    )
    parser.add_argument(
        "--daily-config",
        required=True,
        help="Path to the daily config YAML (repo default or inline materialized).",
    )
    parser.add_argument(
        "--nightly-config",
        required=True,
        help="Path to nightly_config.yaml used to validate test names.",
    )
    parser.add_argument(
        "--test-cases",
        default="",
        help="daily_test_cases input. Empty = use all tests from --daily-config. "
        "Non-empty = direct spec: JSON dict, daily-<soc> tokens, or names.",
    )
    args = parser.parse_args()

    raw = (args.test_cases or "").strip()
    nightly_ordered = _nightly_ordered_by_soc(_load_yaml(args.nightly_config))
    nightly_by_soc = {soc: set(names) for soc, names in nightly_ordered.items()}

    if not raw:
        # Empty input: use every daily test declared in daily_config_yaml.
        daily_by_soc = _daily_names_by_soc(_load_yaml(args.daily_config))
        resolved = _resolve(DAILY_ALL_TOKEN, daily_by_soc, nightly_by_soc)
        print("::notice::daily_test_cases is empty; using all tests from daily_config_yaml")
    else:
        # Non-empty input: daily_test_cases is the spec itself (no daily_config).
        resolved = _resolve_direct(raw, nightly_ordered)
        print("::notice::daily_test_cases is set; resolved directly against nightly_config.yaml")

    for soc, names in sorted(resolved.items()):
        print(f"[{soc}] dispatch {len(names)} test(s): {', '.join(names)}")
    _emit_output(resolved)


if __name__ == "__main__":
    main()
