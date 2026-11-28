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
"""Coordinate one trial over explicit heterogeneous execution targets."""

import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tools.bisect.execution.bundle import tool_bundle
from tools.bisect.execution.channels import ChannelError, KubernetesChannel, LocalChannel
from tools.bisect.orchestration.errors import BisectFatalError
from tools.bisect.testing.outcome import RunOutcome

_TRANSPORT_GRACE_SECONDS = 60.0


class TargetTrialRunner:
    def __init__(self, inp, options):
        self.inp = inp
        self.options = options
        self.targets = options.execution_targets
        self.bundle_digest, self.bundle = tool_bundle()
        self.channels = {}
        self.session_key = hashlib.sha256(
            json.dumps(
                (str(options.repo_dir.resolve()), inp.case_key, inp.good_commit, inp.bad_commit, self.targets.identity)
            ).encode()
        ).hexdigest()
        ray_addresses = {node.option("address") for node in self.targets.nodes if node.channel == "ray"}
        if len(ray_addresses) > 1:
            raise BisectFatalError("One controller can attach to only one explicit Ray cluster")
        try:
            for node in self.targets.nodes:
                if node.channel not in self.channels:
                    if node.channel == "local":
                        self.channels[node.channel] = LocalChannel()
                    elif node.channel == "kubernetes":
                        self.channels[node.channel] = KubernetesChannel()
                    else:
                        from tools.bisect.execution.ray_channel import RayChannel

                        self.channels[node.channel] = RayChannel(node.option("address"))
        except Exception as exc:
            self.finish()
            raise BisectFatalError(f"Could not initialize explicit execution channels: {exc}") from exc

    def _invoke(self, node, commit: str, trial: str, action: str, log_dir: Path) -> dict:
        request = {
            "nonce": uuid.uuid4().hex,
            "bundle": self.bundle,
            "bundle_digest": self.bundle_digest,
            "action": action,
            "channel": node.channel,
            "repository": node.repository,
            "session_key": self.session_key,
            "target_name": node.name,
            "commit": commit,
            "trial": trial,
            "soc": self.options.soc,
            "force_initial_build": self.options.force_initial_build and trial.startswith("round1_"),
            "ignore_baseline": trial.startswith("round1_")
            and (self.options.force_initial_build or not self.options.assume_built_head),
            "assume_built_head": self.options.assume_built_head,
            "native_check": self.options.native_check,
            "test": asdict(self.inp.test_entry),
        }
        timeout = (
            self.targets.prepare_timeout_seconds if action != "test" else self.inp.test_entry.timeout_seconds
        ) + _TRANSPORT_GRACE_SECONDS
        path = log_dir / f"{trial}_{node.name}_{action}.log"
        with path.open("a", encoding="utf-8") as output:
            response = self.channels[node.channel].invoke(node, request, output, timeout)
        if response.get("status") != "ok":
            raise ChannelError(f"Node {node.name} {action} failed: {response.get('error')}; log: {path}")
        if action == "prepare" and response.get("commit") != commit:
            raise ChannelError(f"Node {node.name} prepared a different commit")
        return response

    def _phase(self, commit: str, trial: str, action: str, log_dir: Path) -> list[dict]:
        results = []
        errors = []
        with ThreadPoolExecutor(max_workers=len(self.targets.nodes)) as executor:
            futures = {
                executor.submit(self._invoke, node, commit, trial, action, log_dir): node for node in self.targets.nodes
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append(f"{futures[future].name}: {exc}")
        if errors:
            raise BisectFatalError(
                f"{action} did not complete on all targets; no next round is safe: " + "; ".join(errors)
            )
        return results

    def validate(self, candidate, round_idx: int, log_dir: Path) -> RunOutcome:
        trial = f"round{round_idx}_{candidate.short}"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Verify every target before mutating any target. No discovery fallback.
        try:
            for node in self.targets.nodes:
                self.channels[node.channel].verify(node)
        except Exception as exc:
            raise BisectFatalError(f"Execution target preflight failed: {exc}") from exc
        probed = self._phase(candidate.commit, trial, "probe", log_dir)
        identities = [item.get("execution_identity") for item in probed]
        if None in identities or len(set(identities)) != len(identities):
            raise BisectFatalError("Explicit targets resolve to duplicate or unidentified execution environments")
        prepared = self._phase(candidate.commit, trial, "prepare", log_dir)
        if self.options.mock_tests:
            outcome = RunOutcome(0)
        else:
            completed = self._phase(candidate.commit, trial, "test", log_dir)
            verdicts = [item.get("verdict") for item in completed]
            if any(verdict not in {"PASS", "FAIL", "SKIP"} for verdict in verdicts):
                raise BisectFatalError("A target returned an invalid verdict")
            outcome = (
                RunOutcome(125, infra_error=True, skip_reason="a target could not judge the test")
                if "SKIP" in verdicts
                else RunOutcome(1 if "FAIL" in verdicts else 0)
            )
        outcome.rebuilt = any(item.get("rebuilt") for item in prepared)
        with (log_dir / f"{trial}.log").open("a", encoding="utf-8") as summary:
            summary.write(
                json.dumps(
                    {
                        "mode": "explicit-targets",
                        "commit": candidate.commit,
                        "nodes": [node.name for node in self.targets.nodes],
                        "exit_code": None if self.options.mock_tests else outcome.exit_code,
                        "mock_tests": self.options.mock_tests,
                    }
                )
                + "\n"
            )
        return outcome

    def teardown(self) -> None:
        # Entries own service lifecycle. No global process-name scan.
        pass

    def finish(self) -> None:
        for channel in self.channels.values():
            channel.close()
