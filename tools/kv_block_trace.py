#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read KV JSONL records without importing vLLM or torch."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_events(paths, issues=None):
    events = []
    seen_files = set()

    def malformed(file, number, reason):
        message = f"{file}:{number}: {reason}"
        print(message, file=sys.stderr)
        if issues is not None:
            issues.append({"reason": reason, "source": f"{file}:{number}"})

    for path in paths:
        files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for file in files:
            if file.resolve() in seen_files:
                continue
            seen_files.add(file.resolve())
            with file.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        malformed(file, number, "invalid_or_incomplete_json")
                        continue
                    if not isinstance(event, dict) or event.get("schema_version") not in (1, 2):
                        malformed(file, number, "unsupported_schema")
                        continue
                    try:
                        if not all(isinstance(event[k], str) for k in ("run_id", "trace_id", "event")):
                            raise ValueError
                        if int(event["sequence"]) < 1 or int(event["wall_time_ns"]) < 0:
                            raise ValueError
                    except (KeyError, ValueError, TypeError):
                        malformed(file, number, "invalid_envelope")
                        continue
                    event["source"] = f"{file}:{number}"
                    events.append(event)
    # Cross-process ordering is approximate; original writer sequence is retained.
    return sorted(events, key=lambda e: (int(e["wall_time_ns"]), e["trace_id"], int(e["sequence"])))


def request_ids(event):
    ids = set(event.get("request_ids", ()))
    ids.update(r["request_id"] for r in event.get("requests", ()))
    ids.update(event[key] for key in ("request_id", "remote_request_id") if event.get(key))
    return ids


def request_keys(event):
    run = event["run_id"]
    local_engine = event.get("engine_id")
    local_ids = set(event.get("request_ids", ()))
    local_ids.update(r["request_id"] for r in event.get("requests", ()))
    if event.get("request_id"):
        local_ids.add(event["request_id"])
    keys = {(run, local_engine, rid) for rid in local_ids}
    if event.get("remote_request_id"):
        keys.add((run, event.get("remote_engine_id"), event["remote_request_id"]))
    return keys


def block_refs(event):
    yield from event.get("block_refs", ())
    for request in event.get("requests", ()):
        yield from request.get("block_refs", ())
    if "block_id" in event and "pool_id" in event:
        yield event


def block_keys(event):
    for ref in block_refs(event):
        yield ref.get("group_id"), ref["block_id"]
    if "block_id" in event:
        yield event.get("group_id", None if "pool_id" in event else 0), event["block_id"]
    for name in ("block_ids", "local_block_ids"):
        for group, blocks in enumerate(event.get(name, ())):
            yield from ((group, block) for block in blocks)
    for group in event.get("groups", ()):
        yield from (
            (group["group_id"], slot // group["block_size"])
            for slot in group["slots"]
            if isinstance(slot, int) and slot >= 0
        )
        for row in group.get("block_table", ()):
            kernel_size = group.get("kernel_block_size", group["block_size"])
            yield from (
                (group["group_id"], block * kernel_size // group["block_size"])
                for block in row
                if block * kernel_size // group["block_size"] != group.get("null_block_id", 0)
            )


def select_events(events, request=None, block=None, group=None, engine=None, run_id=None, pool_id=None, epoch=None):
    selected = [e for e in events if run_id is None or e["run_id"] == run_id]
    if request:
        # Follow explicit P/D ID links, without guessing UUID suffix conventions.
        aliases = {key for e in selected for key in request_keys(e) if key[2].startswith(request)}
        # Co-batched requests are context, not aliases. Only an explicit
        # local/remote transfer pair may expand the identity closure.
        pairs = [
            request_keys({**e, "request_ids": [], "requests": []})
            for e in selected
            if e.get("request_id") and e.get("remote_request_id") and e["event"].startswith("transfer.")
        ]
        while True:
            expanded = aliases | {key for pair in pairs if pair & aliases for key in pair}
            if expanded == aliases:
                break
            aliases = expanded
        spans = {(e["trace_id"], e["span_id"]) for e in selected if request_keys(e) & aliases and "span_id" in e}
        selected = [
            e
            for e in selected
            if request_keys(e) & aliases
            or (e["trace_id"], e.get("span_id")) in spans
            or e["event"].startswith("trace.")
        ]
    if engine is not None:
        selected = [e for e in selected if e.get("engine_id") == engine]
    if block is not None or pool_id is not None or epoch is not None:

        def matches(event):
            if pool_id is not None or epoch is not None:
                return any(
                    (pool_id is None or ref.get("pool_id") == pool_id)
                    and (epoch is None or str(ref.get("alloc_epoch")) == str(epoch))
                    and (block is None or ref["block_id"] == block)
                    and (group is None or ref.get("group_id") == group)
                    for ref in block_refs(event)
                )
            return any(b == block and (group is None or g == group) for g, b in block_keys(event))

        related_requests = {rid for e in selected if matches(e) for rid in request_ids(e)}
        spans = {(e["trace_id"], e["span_id"]) for e in selected if "span_id" in e and matches(e)}
        selected = [
            e
            for e in selected
            if matches(e)
            or (e["trace_id"], e.get("span_id")) in spans
            or e["event"].startswith("trace.")
            or (not list(block_keys(e)) and request_ids(e) & related_requests)
        ]
    return selected


def check_events(events):
    """Check recording completeness, never infer content correctness."""
    issues = []
    writers = defaultdict(list)
    ids = {(e["run_id"], e.get("event_id")) for e in events if e.get("event_id")}
    for event in events:
        writers[event["run_id"], event["trace_id"]].append(event)
        if event["event"] in (
            "trace.gap",
            "trace.truncated",
            "trace.observation_error",
            "snapshot.truncated",
            "snapshot.skipped",
        ):
            issues.append({"reason": event["event"], "source": event.get("source"), "detail": event.get("reason")})
        for parent in event.get("parent_event_ids", ()):
            if (event["run_id"], parent) not in ids:
                issues.append({"reason": "missing_parent", "event_id": event.get("event_id"), "parent": parent})
    for (run, writer), rows in writers.items():
        rows.sort(key=lambda e: int(e["sequence"]))
        previous = 0
        for row in rows:
            seq = int(row["sequence"])
            if seq != previous + 1:
                issues.append(
                    {
                        "reason": "sequence_gap_or_duplicate",
                        "run_id": run,
                        "writer_id": writer,
                        "expected": previous + 1,
                        "actual": seq,
                    }
                )
            previous = seq
        if rows[0]["event"] != "trace.manifest":
            issues.append({"reason": "unverified_start", "run_id": run, "writer_id": writer})
        if rows[-1]["event"] != "trace.stop":
            issues.append({"reason": "open_or_incomplete_writer", "run_id": run, "writer_id": writer})
    if not events:
        issues.append({"reason": "no_events"})
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--request", help="Request ID or prefix; follows explicit P/D request links")
    parser.add_argument("--block", type=int, help="Local physical block ID; pair with --engine and --group")
    parser.add_argument("--group", type=int)
    parser.add_argument("--engine")
    parser.add_argument("--run-id")
    parser.add_argument("--pool-id", help="Allocator pool instance; scopes block IDs across restarts")
    parser.add_argument("--epoch", help="Allocation generation; requires --pool-id and --block")
    parser.add_argument(
        "--check", action="store_true", help="Check whole input completeness; exit 2 on gaps (not a KV integrity check)"
    )
    parser.add_argument("--json", action="store_true", help="Emit complete selected JSONL records")
    args = parser.parse_args()
    if args.epoch is not None and (args.pool_id is None or args.block is None):
        parser.error("--epoch requires --pool-id and --block")
    issues = []
    all_events = load_events(args.paths, issues)
    if args.check:
        checked = [e for e in all_events if args.run_id is None or e["run_id"] == args.run_id]
        issues.extend(check_events(checked))
        print(
            json.dumps(
                {"recording_complete": not issues, "kv_integrity": "not_checked", "issues": issues}, ensure_ascii=True
            )
        )
        return 2 if issues else 0
    events = select_events(
        all_events, args.request, args.block, args.group, args.engine, args.run_id, args.pool_id, args.epoch
    )
    for event in events:
        if args.json:
            print(json.dumps(event, ensure_ascii=True))
            continue
        blocks = ",".join(
            f"g{g}:b{b}" for g, b in sorted(set(block_keys(event)), key=lambda pair: (str(pair[0]), pair[1]))
        )
        warning = (
            " DUMMY_CACHE_CHANGED"
            if event["event"] == "cache.diff" and event.get("phase") == "dummy" and event["changed_elements"]
            else ""
        )
        print(
            f"{event['wall_time_ns']} {event['host']} pid={event['pid']} "
            f"engine={event.get('engine_id', '-')} dp={event.get('dp_rank', '-')} "
            f"tp={event.get('tp_rank', '-')} {event['event']} "
            f"req={','.join(sorted(request_ids(event))) or '-'} blocks={blocks or '-'} "
            f"step={event.get('step_id', '-')} epoch={event.get('alloc_epoch', '-')} "
            f"pool={event.get('pool_id', '-')}{warning}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
