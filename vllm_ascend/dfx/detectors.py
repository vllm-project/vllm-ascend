# SPDX-License-Identifier: Apache-2.0
"""Checks over already resolved host results; never transfers device tensors.

Repetition and configured sequences are symptoms, not proof of incorrect math.
Finite-but-wrong computation requires independent reference evidence.
"""

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm_ascend.dfx.config import DfxConfig


@dataclass(frozen=True)
class OutputContext:
    execution_id: int
    requests: tuple[tuple[str, int], ...]
    vocab_size: int
    proposed_tokens: tuple[tuple[str, int], ...] = ()


class OutputDetectors:
    def __init__(self, config: DfxConfig):
        self.config = config
        self.histories: OrderedDict[tuple[str, int], tuple[int, deque]] = OrderedDict()
        self.acceptance: OrderedDict[tuple[str, int], deque] = OrderedDict()
        self.evictions = 0
        self.out_of_order = 0
        self.missing_nan_counts = 0
        self.missing_logprobs = 0

    def reset_history(self):
        self.histories.clear()
        self.acceptance.clear()

    def check(self, output: Any, context: OutputContext) -> list[dict]:
        findings = []

        def report(code, req_id=None, severity="invariant", **details):
            findings.append(dict(code=code, req_id=req_id, severity=severity, **details))

        expected = tuple(req_id for req_id, _ in context.requests)
        actual = tuple(output.req_ids)
        if actual != expected or output.req_id_to_index != {req_id: i for i, req_id in enumerate(expected)}:
            report("output_request_mapping")
            return findings  # Never attribute suspect rows to requests.
        if len(output.sampled_token_ids) != len(expected):
            report("output_row_count")
            return findings
        if sum(map(len, output.sampled_token_ids)) > self.config.max_checked_tokens:
            report("output_check_budget_exceeded", severity="coverage_gap")
            self.reset_history()
            return findings
        nan_counts = output.num_nans_in_logits
        if nan_counts is None:
            self.missing_nan_counts += 1
        else:
            for req_id, count in nan_counts.items():
                if req_id not in expected:
                    report("nan_count_request_mapping", req_id)
                elif count > 0:
                    report("logits_nan", req_id, count=int(count))
        if output.logprobs is None:
            self.missing_logprobs += 1
        proposed_by_request = dict(context.proposed_tokens)
        for row, key in enumerate(context.requests):
            req_id, generation = key
            tokens = output.sampled_token_ids[row]
            if any(type(token) is not int or not 0 <= token < context.vocab_size for token in tokens):
                report("sampled_token_out_of_range", req_id)
                continue
            previous = self.histories.pop(key, None)
            if previous is not None and context.execution_id <= previous[0]:
                self.out_of_order += 1
                self.histories[key] = previous
                continue
            history = previous[1] if previous else deque(maxlen=self.config.token_history_size)
            proposed = proposed_by_request.get(req_id, 0)
            if proposed > 0 and tokens:
                accepted = len(tokens) - 1  # Verification emits accepted drafts plus one target/bonus token.
                if accepted > proposed:
                    report("spec_output_length_exceeds_proposals", req_id)
                else:
                    window = self.acceptance.pop(key, deque(maxlen=self.config.token_history_size))
                    window.append((accepted, proposed))
                    self.acceptance[key] = window
                    accepted_total = sum(item[0] for item in window)
                    proposed_total = sum(item[1] for item in window)
                    if proposed_total >= self.config.spec_min_proposals:
                        rate = accepted_total / proposed_total
                        if rate < self.config.spec_acceptance_floor:
                            report(
                                "spec_acceptance_low",
                                req_id,
                                "symptom",
                                acceptance_rate=rate,
                                proposed=proposed_total,
                                accepted=accepted_total,
                            )
                    while len(self.acceptance) > self.config.max_tracked_requests:
                        self.acceptance.popitem(last=False)
            seen = set()
            for token in tokens:
                history.append(token)
                recent = tuple(history)
                for period in range(1, self.config.repeat_max_period + 1):
                    width = period * self.config.repeat_min_count
                    if len(recent) >= width and recent[-width:] == recent[-period:] * self.config.repeat_min_count:
                        if "token_repeat" not in seen:
                            report("token_repeat", req_id, "symptom", period=period, generation=generation)
                            seen.add("token_repeat")
                        break
                for index, pattern in enumerate(self.config.token_patterns):
                    if recent[-len(pattern) :] == tuple(pattern) and index not in seen:
                        report("output_token_pattern", req_id, "symptom", pattern_index=index, generation=generation)
                        seen.add(index)
            self.histories[key] = (context.execution_id, history)
            while len(self.histories) > self.config.max_tracked_requests:
                evicted, _ = self.histories.popitem(last=False)
                self.acceptance.pop(evicted, None)
                self.evictions += 1
            if output.logprobs is not None and tokens:
                values = output.logprobs.slice_request(row, len(tokens))
                probabilities = values.logprobs
                ids = values.logprob_token_ids
                # Consume only the existing host numpy contract. Never call .cpu().
                if not isinstance(probabilities, np.ndarray) or not isinstance(ids, np.ndarray):
                    report("unsupported_logprob_layout", req_id, "coverage_gap")
                elif probabilities.shape != ids.shape or probabilities.ndim != 2 or len(probabilities) != len(tokens):
                    report("logprob_shape", req_id)
                elif np.isnan(probabilities).any() or np.isposinf(probabilities).any():
                    report("logprob_nonfinite", req_id)
                # -inf is valid for masked alternatives; do not flag it globally.
        return findings


def check_positions(positions, counts, computed):
    if len(positions) != sum(counts) or len(counts) != len(computed):
        return ["position_shape"]
    offset = 0
    for count, start in zip(counts, computed):
        if count < 0 or start < 0:
            return ["negative_position_or_length"]
        row = positions[offset : offset + count]
        if count and (row[0] != start or (np.diff(row) != 1).any()):
            return ["position_alignment"]
        offset += count
    return []


def check_block_plan(req_ids, counts, positions, table, lengths, block_size, expected_blocks):
    """Plain non-circular, non-sharded attention only; table is host numpy."""
    if block_size <= 0 or table.ndim != 2 or len(table) < len(req_ids) or len(lengths) < len(req_ids):
        return ["block_table_shape"]
    findings = set()
    written = set()
    offset = 0
    for row, (req_id, count) in enumerate(zip(req_ids, counts)):
        length = int(lengths[row])
        expected = expected_blocks[req_id]
        if length < 0 or length > table.shape[1] or length != len(expected):
            findings.add("block_table_length")
            offset += count
            continue
        if not np.array_equal(table[row, :length], expected):
            findings.add("block_table_request_mapping")
        query_positions = positions[offset : offset + count]
        indices = query_positions // block_size
        offset += count
        if (indices < 0).any() or (indices >= length).any():
            findings.add("kv_write_out_of_allocated_range")
            continue
        blocks = table[row, indices]
        if (blocks < 0).any():
            findings.add("negative_physical_block")
            continue
        slots = blocks.astype(np.int64) * block_size + query_positions % block_size
        unique = set(slots.tolist())
        if len(unique) != len(slots) or written.intersection(unique):
            findings.add("kv_planned_write_collision")
        written.update(unique)
    return sorted(findings)


class BlockOwnership:
    """Worker-observed associations, NOT authoritative allocator generations."""

    def __init__(self, config):
        self.config = config
        self.previous = {}
        self.computed = {}
        self.epoch = 0

    def reset(self):
        self.previous.clear()
        self.computed.clear()

    def observe(self, requests, *, allow_rollback=(), check_progress=False):
        if len(requests) > self.config.max_tracked_requests:
            self.reset()
            return {"complete": False, "reason": "request_budget"}, []
        current = {}
        progress = {}
        total = 0
        for req_id, request in requests.items():
            progress[req_id] = int(request.num_computed_tokens)
            for group, blocks in enumerate(request.block_ids):
                total += len(blocks)
                if total > self.config.max_tracked_blocks:
                    self.reset()
                    return {"complete": False, "reason": "block_budget"}, []
                for block in blocks:
                    current.setdefault((group, int(block)), set()).add(req_id)
        changes = []
        for key in sorted(self.previous.keys() | current.keys()):
            before = self.previous.get(key, set())
            after = current.get(key, set())
            if before != after:
                self.epoch += 1
                changes.append(
                    dict(
                        group=key[0],
                        block=key[1],
                        association_epoch=self.epoch,
                        previous_owners=sorted(before),
                        owners=sorted(after),
                        event="attach" if not before else "detach" if not after else "owner_change",
                    )
                )
        violations = []
        for req_id, value in progress.items():
            if value < 0:
                violations.append("negative_computed_tokens")
            elif check_progress and req_id not in allow_rollback and value < self.computed.get(req_id, 0):
                violations.append("computed_tokens_rollback_without_resume")
        self.previous = current
        self.computed = progress
        return dict(complete=True, changes=changes, authority="worker_observed_not_allocator"), sorted(set(violations))


def check_transfer_registration(entries, views, num_blocks):
    """Validate local registration arithmetic, not remote DMA completion."""
    findings = set()
    if num_blocks <= 0:
        return ["transfer_invalid_block_count"]
    registered = {entry["base"] for entry in entries}
    if views.keys() - registered:
        findings.add("transfer_unregistered_cache_view")
    for entry in entries:
        view = views.get(entry["base"])
        if view is None:
            findings.add("transfer_unknown_cache_base")
            continue
        scale, remainder = divmod(view["blocks"], num_blocks)
        if remainder or scale < 1:
            findings.add("transfer_block_count_mapping")
            continue
        expected_stride = view["stride_bytes"] * scale
        length, stride = entry["length"], entry["stride"]
        if length <= 0 or length > expected_stride:
            findings.add("transfer_block_length_exceeds_stride")
        if stride != expected_stride:
            findings.add("transfer_stride_mismatch")
        end = entry["base"] + (num_blocks - 1) * stride + length
        if end > view["storage_end"] or entry["base"] < view["storage_begin"]:
            findings.add("transfer_registration_out_of_allocation")
        if entry.get("groups") is not None and not set(view["groups"]).issubset(entry["groups"]):
            findings.add("transfer_missing_alias_group")
    return sorted(findings)
