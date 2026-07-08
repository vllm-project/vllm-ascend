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
"""Filesystem barrier used to keep multi-node bisect rounds in lockstep.

Every node must see the same shared directory (PVC / NFS). Per round the master
publishes a *command* (which commit to deploy); every node deploys, reports its
HEAD as *ready*; the master verifies all nodes agree on the commit before any
test launches; after judging, the master publishes the *verdict* / a *done*
sentinel so workers know whether to continue or exit.

Protocol files under ``<coord_dir>/round_<N>/``::

    command.json                {round, commit, rebuild}
    ready_<idx>.json            {node, head}
    verdict.json                {verdict}            (master, after the trial)
    <coord_dir>/DONE            sentinel -> workers exit
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 3.0


class Coordinator:
    def __init__(self, coord_dir: str, num_nodes: int, node_index: int):
        self.root = Path(coord_dir)
        self.num_nodes = num_nodes
        self.node_index = node_index
        self.root.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Master-only: clear stale coordination state from a previous run.

        The coord dir often lives on a persistent PVC, so a leftover DONE flag or
        round_*/ from an earlier run would make this run's workers exit instantly
        (seeing a stale DONE) or read a stale command. Called once by the master
        BEFORE publishing round 1 -- workers only write ready files after they see
        round 1, so nothing of this run is destroyed.
        """
        import shutil

        removed = []
        if self._done_flag.exists():
            self._done_flag.unlink()
            removed.append("DONE")
        for d in self.root.glob("round_*"):
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d.name)
        if removed:
            logger.info("[coord] reset stale coordination state: %s", removed)

    # --------------------------------------------------------------- paths
    def _round_dir(self, rnd: int) -> Path:
        d = self.root / f"round_{rnd}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def _done_flag(self) -> Path:
        return self.root / "DONE"

    # ------------------------------------------------------- master writes
    def publish_command(self, rnd: int, commit: str, rebuild: bool, action: str = "RUN") -> None:
        """Publish the per-round command. ``action`` is "RUN" (deploy + run the
        distributed test) or "SKIP" (this commit is pre-skipped, e.g. vLLM
        mismatch -- workers consume the command but do not deploy/run, keeping
        leader/worker rounds in lockstep)."""
        path = self._round_dir(rnd) / "command.json"
        path.write_text(json.dumps({"round": rnd, "commit": commit, "rebuild": rebuild, "action": action}))
        logger.info("[coord] published command round=%d commit=%s action=%s", rnd, commit[:12], action)

    def publish_verdict(self, rnd: int, verdict: str) -> None:
        (self._round_dir(rnd) / "verdict.json").write_text(json.dumps({"verdict": verdict}))

    def publish_done(self) -> None:
        self._done_flag.write_text("done")
        logger.info("[coord] published DONE")

    def is_done(self) -> bool:
        return self._done_flag.exists()

    # ------------------------------------------------------- worker writes
    def signal_ready(self, rnd: int, head: str) -> None:
        path = self._round_dir(rnd) / f"ready_{self.node_index}.json"
        path.write_text(json.dumps({"node": self.node_index, "head": head}))
        logger.info("[coord] node %d ready for round %d at %s", self.node_index, rnd, head[:12])

    @staticmethod
    def _fresh(path: Path, since_ts: float | None) -> bool:
        """True if ``path`` exists and (when since_ts given) was modified after it.

        Guards against stale sentinels on a persistent PVC: a DONE / release file
        left over from a previous run has an mtime older than this worker's start,
        so it is ignored rather than causing an immediate (wrong) exit.
        """
        if not path.exists():
            return False
        if since_ts is None:
            return True
        try:
            return path.stat().st_mtime >= since_ts
        except OSError:
            return False

    # -------------------------------------------------------------- reads
    def wait_command(
        self, rnd: int, timeout_s: float, release_file: str | None = None, since_ts: float | None = None
    ) -> dict | None:
        """Worker: block until the round command appears, or a stop signal.

        Returns the command dict, or None when the bisect is over -- signalled by
        either the coordinator DONE sentinel or an external ``release_file`` (the
        AOP leader's ``done`` file, touched even when it decides not to bisect).
        ``since_ts`` (the worker's start time) makes a *stale* DONE/release file
        from a previous run on a persistent PVC be ignored.
        """
        path = self._round_dir(rnd) / "command.json"
        release = Path(release_file) if release_file else None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._fresh(self._done_flag, since_ts):
                return None
            if release is not None and self._fresh(release, since_ts):
                logger.info("[coord] external release file %s present; worker exiting", release)
                return None
            if path.exists():
                return json.loads(path.read_text())
            time.sleep(_POLL_INTERVAL_S)
        raise TimeoutError(f"Timed out waiting for command of round {rnd}")

    def wait_all_ready(self, rnd: int, expected_commit: str, timeout_s: float) -> None:
        """Master: barrier until every node reports ready on ``expected_commit``."""
        rdir = self._round_dir(rnd)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            readies = sorted(rdir.glob("ready_*.json"))
            if len(readies) >= self.num_nodes:
                self._assert_consistent(readies, expected_commit)
                logger.info("[coord] all %d nodes ready for round %d", self.num_nodes, rnd)
                return
            time.sleep(_POLL_INTERVAL_S)
        ready_now = sorted(p.name for p in rdir.glob("ready_*.json"))
        raise TimeoutError(
            f"Barrier timeout: only {len(ready_now)}/{self.num_nodes} nodes ready "
            f"for round {rnd} (ready: {ready_now}). The missing node(s) never "
            "signalled ready -- ensure 'auto_bisect.py --scene multi_node' is "
            "launched on EVERY node (workers auto-enter the worker loop), all "
            f"pointing at the same shared --coord-dir ({self.root}). A worker that "
            "only ran the test (not the bisect agent) will never join this barrier."
        )

    def wait_verdict(self, rnd: int, timeout_s: float) -> str | None:
        """Worker: block until the master publishes this round's verdict / DONE."""
        path = self._round_dir(rnd) / "verdict.json"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if path.exists():
                return json.loads(path.read_text()).get("verdict")
            if self.is_done():
                return None
            time.sleep(_POLL_INTERVAL_S)
        raise TimeoutError(f"Timed out waiting for verdict of round {rnd}")

    def _assert_consistent(self, readies: list[Path], expected_commit: str) -> None:
        for r in readies:
            head = json.loads(r.read_text()).get("head", "")
            if not head.startswith(expected_commit[:12]) and not expected_commit.startswith(head[:12]):
                raise RuntimeError(
                    f"Node {r.name} is on {head[:12]}, expected {expected_commit[:12]}; "
                    "aborting round to avoid a split-commit run"
                )
