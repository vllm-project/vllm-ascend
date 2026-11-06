# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Translate the common connector lifecycle into logical KV readiness events.

Submission snapshots come from the scheduler, never from backend-specific
transport offsets. No connector calls, waits or result polling are added.
"""

from __future__ import annotations

import functools
from typing import Any


class ConnectorGuard:
    def __init__(self, runtime: Any, connector: Any):
        self.runtime = runtime
        self.state = runtime.state
        self.staged: list[dict] = []
        self.inflight: dict[str, dict] = {}
        self.sync: dict[str, dict] = {}
        self.drained: list[dict] = []
        self.completed: list[tuple[dict, set[int]]] = []
        self.ready: dict[int, set[str]] = {}
        self.ready_groups: dict[int, set[int]] = {}
        self.failed: set[int] = set()
        self.failed_blocks: dict[tuple[str, int], set[int]] = {}
        self.post_forward_complete = False
        self._wrap(connector, "start_load_kv", self.start)
        self._wrap(connector, "wait_for_layer_load", self.wait)
        self._wrap(connector, "get_transfer_results", self.results)
        self._wrap(connector, "get_block_ids_with_load_errors", self.errors)

    @staticmethod
    def _wrap(connector, name, callback):
        original = getattr(connector, name)

        @functools.wraps(original)
        def observed(*args, **kwargs):
            return callback(original, *args, **kwargs)

        setattr(connector, name, observed)

    def stage(self, loads: list[dict]) -> None:
        if self.staged or self.sync:
            self.finish_step()
        self.staged.extend(loads)

    def start(self, original, *args, **kwargs):
        submitted, self.staged = self.staged, []
        accepted = []
        for meta in submitted:
            request = meta["request_id"]
            if request in self.inflight:
                # Request-ID-only completions cannot distinguish overlapping incarnations.
                self.runtime.gap("overlapping_load_request_id", request_id=request)
                self.state.finish_load(meta)
                continue
            accepted.append(meta)
            if meta["is_async"]:
                self.inflight[request] = meta
            else:
                self.sync[request] = meta
        self.runtime._check_failure()
        try:
            return original(*args, **kwargs)
        except Exception:
            for meta in accepted:
                self.state.fail_load(meta)
                self.inflight.pop(meta["request_id"], None)
            raise

    def wait(self, original, layer_name, *args, **kwargs):
        result = original(layer_name, *args, **kwargs)
        group = self.runtime.layer_groups.get(layer_name)
        if group is None:
            return result
        required = {name for name, index in self.runtime.layer_groups.items() if index == group}
        for meta in self.sync.values():
            if meta["load_id"] in self.failed:
                continue
            ready = self.ready.setdefault(meta["load_id"], set())
            ready.add(layer_name)
            groups = self.ready_groups.setdefault(meta["load_id"], set())
            if required <= ready and group not in groups:
                self.state.load(meta, group_ids={group}, finish=False)
                groups.add(group)
        self.runtime._check_failure()
        return result

    def check_ready(self, layer: str) -> None:
        group = self.runtime.layer_groups.get(layer)
        if group is None:
            return
        for meta in self.sync.values():
            if layer not in self.ready.get(meta["load_id"], ()):
                self.runtime.gap("load_layer_not_ready", request_id=meta["request_id"], layer=layer)
                self.runtime._missing_groups.add(group)
                self.runtime._unready_load_groups.add(group)

    def before_audit(self) -> None:
        for layer in self.runtime.layer_groups:
            self.check_ready(layer)

    def results(self, original, *args, **kwargs):
        result = original(*args, **kwargs)
        failed = set(result.failed_recving or ())
        for request in set(result.finished_recving or ()) | failed:
            meta = self.inflight.pop(request, None)
            if meta is None:
                meta = self.sync.get(request)
            if meta is None:
                continue
            if request in failed:
                self.failed.add(meta["load_id"])
                if not meta["is_async"]:
                    self.state.fail_load(meta, finish=False)
            self.drained.append(meta)
        return result

    def errors(self, original, *args, **kwargs):
        invalid = original(*args, **kwargs)
        if invalid:
            # Core queries invalid blocks after transfer results; keep the snapshot until then.
            for meta in (*self.inflight.values(), *self.sync.values()):
                if self.state.loads.get(meta["request_id"]) != meta:
                    continue
                affected = invalid & self._target_blocks(meta)
                if affected:
                    self.failed_blocks.setdefault((meta["pool_id"], meta["load_id"]), set()).update(affected)
                    self.state.fail_load(meta, block_ids=affected, finish=False)
        for meta in self.drained:
            self.completed.append((meta, set(invalid or ())))
        self.drained.clear()
        self.post_forward_complete = True
        if self.runtime._phase is None:
            self.finish_step()
            self.runtime._check_failure()
        return invalid

    def _target_blocks(self, meta: dict) -> set[int]:
        targets: set[int] = set()
        for group, refs in enumerate(meta["block_table"]):
            size = self.state.group_sizes[group]
            first, last = meta["token_start"] // size, (meta["token_end"] + size - 1) // size
            targets.update(block for block, _ in refs[first:last])
        return targets

    def commit_completions(self) -> None:
        # MRV2 polls after forward but before its audit. These completions may
        # authorize the next step, never an earlier read in the current step.
        for meta, invalid in self.completed:
            # Legacy connectors can report failed blocks in an earlier poll,
            # then drain only a request-ID completion in a later step.
            invalid |= self.failed_blocks.pop((meta["pool_id"], meta["load_id"]), set())
            if self.state.loads.get(meta["request_id"]) != meta:
                continue  # Cancellation may release ownership before the known old receive drains.
            if meta["is_async"]:
                if meta["load_id"] in self.failed:
                    self.state.fail_load(meta, finish=False)
                else:
                    self.state.load(meta, block_ids=self._target_blocks(meta) - invalid, finish=False)
            if invalid:
                self.state.fail_load(meta, block_ids=invalid, finish=False)
            if meta["is_async"] or meta["load_id"] in self.failed:
                self.state.finish_load(meta)
        self.completed.clear()

    def finish_step(self) -> None:
        self.commit_completions()
        for meta in self.staged:
            self.runtime.gap("load_not_submitted", request_id=meta["request_id"])
            self.state.finish_load(meta)
        for meta in self.sync.values():
            if meta["load_id"] not in self.failed:
                required = set(self.runtime.layer_groups.values())
                if not required <= self.ready_groups.get(meta["load_id"], set()):
                    self.runtime.gap("load_completion_missing", request_id=meta["request_id"])
            self.state.finish_load(meta)
            self.failed_blocks.pop((meta["pool_id"], meta["load_id"]), None)
        self.staged.clear()
        self.sync.clear()
        self.ready.clear()
        self.ready_groups.clear()
        self.failed.clear()
        self.post_forward_complete = False
