#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read KV JSONL records without importing vLLM or torch."""

import argparse
import json
import sys
from pathlib import Path


def load_events(paths):
    events = []
    for path in paths:
        files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for file in files:
            with file.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid/incomplete record: {file}:{number}", file=sys.stderr)
                        continue
                    if event.get("schema_version") != 1 or "trace_id" not in event:
                        continue
                    event["source"] = f"{file}:{number}"
                    events.append(event)
    # Cross-process ordering is approximate; original writer sequence is retained.
    return sorted(events, key=lambda e: (e["wall_time_ns"], e["trace_id"], e["sequence"]))


def request_ids(event):
    ids = set(event.get("request_ids", ()))
    ids.update(event[key] for key in ("request_id", "remote_request_id") if event.get(key))
    return ids


def block_keys(event):
    if "block_id" in event:
        yield event.get("group_id", 0), event["block_id"]
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
            yield from ((group["group_id"], block * kernel_size // group["block_size"]) for block in row if block != 0)


def select_events(events, request=None, block=None, group=None, engine=None, run_id=None):
    selected = [e for e in events if run_id is None or e["run_id"] == run_id]
    if request:
        # Follow explicit P/D ID links, without guessing UUID suffix conventions.
        aliases = {rid for e in selected for rid in request_ids(e) if rid.startswith(request)}
        while True:
            expanded = aliases | {rid for e in selected if request_ids(e) & aliases for rid in request_ids(e)}
            if expanded == aliases:
                break
            aliases = expanded
        spans = {(e["trace_id"], e["span_id"]) for e in selected if request_ids(e) & aliases and "span_id" in e}
        selected = [
            e
            for e in selected
            if request_ids(e) & aliases or (e["trace_id"], e.get("span_id")) in spans or e["event"].startswith("trace.")
        ]
    if engine is not None:
        selected = [e for e in selected if e.get("engine_id") == engine]
    if block is not None:
        related_requests = {
            rid
            for e in selected
            if any(b == block and (group is None or g == group) for g, b in block_keys(e))
            for rid in request_ids(e)
        }
        spans = {
            (e["trace_id"], e["span_id"])
            for e in selected
            if "span_id" in e and any(b == block and (group is None or g == group) for g, b in block_keys(e))
        }
        selected = [
            e
            for e in selected
            if any(b == block and (group is None or g == group) for g, b in block_keys(e))
            or (e["trace_id"], e.get("span_id")) in spans
            or e["event"].startswith("trace.")
            or (not list(block_keys(e)) and request_ids(e) & related_requests)
        ]
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--request", help="Request ID or prefix; follows explicit P/D request links")
    parser.add_argument("--block", type=int, help="Local physical block ID; pair with --engine and --group")
    parser.add_argument("--group", type=int)
    parser.add_argument("--engine")
    parser.add_argument("--run-id")
    parser.add_argument("--json", action="store_true", help="Emit complete selected JSONL records")
    args = parser.parse_args()
    events = select_events(load_events(args.paths), args.request, args.block, args.group, args.engine, args.run_id)
    for event in events:
        if args.json:
            print(json.dumps(event, ensure_ascii=True))
            continue
        blocks = ",".join(f"g{g}:b{b}" for g, b in sorted(set(block_keys(event))))
        warning = (
            " DUMMY_CACHE_CHANGED"
            if event["event"] == "cache.diff" and event.get("phase") == "dummy" and event["changed_elements"]
            else ""
        )
        print(
            f"{event['wall_time_ns']} {event['host']} pid={event['pid']} "
            f"engine={event.get('engine_id', '-')} dp={event.get('dp_rank', '-')} "
            f"tp={event.get('tp_rank', '-')} {event['event']} "
            f"req={','.join(sorted(request_ids(event))) or '-'} blocks={blocks or '-'}{warning}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
