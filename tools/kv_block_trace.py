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
        # cache.config describes groups without execution slots or block tables.
        yield from (
            (group["group_id"], slot // group["block_size"])
            for slot in group.get("slots", ())
            if isinstance(slot, int) and slot >= 0
        )
        for row in group.get("block_table", ()):
            kernel_size = group.get("kernel_block_size", group["block_size"])
            yield from (
                (group["group_id"], block * kernel_size // group["block_size"])
                for block in row
                if block * kernel_size // group["block_size"] != group.get("null_block_id", 0)
            )


def observation_scope(event):
    return event["run_id"], event.get("host"), event.get("host_boot_id"), event.get("engine_id"), event.get("dp_rank")


def process_scope(event):
    return (
        *observation_scope(event),
        event.get("pid"),
        event.get("process_instance_id"),
        event.get("tp_rank"),
        event.get("pp_rank"),
    )


def writer_scope(event):
    return event["run_id"], event["trace_id"]


def observation_time(event):
    try:
        return int(event["monotonic_ns"])
    except (KeyError, TypeError, ValueError):
        return None


def event_reference(event):
    return event.get("event_id", f"{event['trace_id']}:{event['sequence']}")


def correlate_accesses(events):
    """Derive host-observed block context for dummy spans without inventing owners.

    Relations are query context, not device generation or causality assertions.
    Reconstruct only complete writer sequences on one host/boot; never compare
    monotonic timestamps across hosts. Missing/ambiguous evidence stays unknown.
    """
    # Query exports can be read again. Recompute derived fields from source
    # records; a filtered export must not retain now-unverifiable associations.
    events = [
        {k: v for k, v in e.items() if k not in ("access_relations", "query_context")}
        if "access_relations" in e or "query_context" in e
        else e
        for e in events
    ]
    writers = defaultdict(list)
    process_workers = defaultdict(set)
    lifecycle_times = defaultdict(set)
    transfer_writers = defaultdict(set)
    for event in events:
        writers[writer_scope(event)].append(event)
        if event["event"] in ("schedule.received", "schedule.apply"):
            process_workers[process_scope(event)].add(event["trace_id"])
        if event["event"] in ("block.alloc", "block.acquire", "block.release"):
            key = (*observation_scope(event), event.get("pool_id"), event["trace_id"], event["block_id"])
            lifecycle_times[key].add(observation_time(event))
        if event["event"].startswith("transfer."):
            transfer_writers[process_scope(event)].add(writer_scope(event))
    usable = set()
    for writer, rows in writers.items():
        rows = sorted(rows, key=lambda e: int(e["sequence"]))
        times = [observation_time(e) for e in rows]
        if (
            [int(e["sequence"]) for e in rows] == list(range(1, len(rows) + 1))
            and all(t is not None for t in times)
            and times == sorted(times)
            and not any(e["event"] in ("trace.gap", "trace.observation_error", "trace.truncated") for e in rows)
        ):
            usable.add(writer)

    pools, bindings, transfers, spans, derived = {}, {}, {}, {}, {}

    def relations(event):
        result = []
        process = process_scope(event)
        binding = bindings.get(process)
        now = observation_time(event)
        for group in event.get("groups", ()):
            blocks = sorted(
                {
                    slot // group["block_size"]
                    for slot in group["slots"]
                    if isinstance(slot, int)
                    and slot >= 0
                    and slot // group["block_size"] != group.get("null_block_id", 0)
                }
            )
            for block in blocks:
                relation = {
                    "group_id": group["group_id"],
                    "block_id": block,
                    "association": "unknown",
                    "owner_request_ids": [],
                    "device_epoch_verified": False,
                    "observed_at": "span_begin",
                }
                result.append(relation)
                if writer_scope(event) not in usable or event.get("host") is None or now is None:
                    relation["reason"] = "incomplete_access_writer"
                    continue
                if not binding or len(process_workers[process]) != 1 or observation_time(binding) >= now:
                    relation["reason"] = "missing_or_ambiguous_process_pool_binding"
                    continue
                pool_key = (*observation_scope(event), binding.get("pool_id"), binding.get("scheduler_id"))
                state = pools.get(pool_key, {}).get(block)
                if (
                    not state
                    or state["time"] >= now
                    or not state["epoch"]
                    or now in lifecycle_times[(*pool_key, block)]
                ):
                    relation["reason"] = "missing_or_unordered_allocator_evidence"
                    continue
                owners = state["owners"].get(group["group_id"], {})
                relation.update(
                    association="host_observed_lifecycle",
                    pool_id=binding["pool_id"],
                    alloc_epoch=state["epoch"],
                    owner_request_ids=sorted(owners),
                    process_identity="boot_and_start_time" if event.get("process_instance_id") else "legacy_host_pid",
                    evidence_event_ids=[event_reference(binding), state["allocation"], *sorted(owners.values())],
                )
                transfer = transfers.get((process, group["group_id"], block))
                if (
                    transfer
                    and state["allocated_at"] < observation_time(transfer) < now
                    and transfer_writers[process].issubset(usable)
                ):
                    relation["latest_transfer"] = {
                        "event_id": event_reference(transfer),
                        "event": transfer["event"],
                        "request_id": transfer.get("request_id"),
                        "remote_request_id": transfer.get("remote_request_id"),
                        "remote_engine_id": transfer.get("remote_engine_id"),
                        "matches_recorded_owner": transfer.get("request_id") in owners,
                    }
        return result

    ordered = sorted(
        (e for e in events if observation_time(e) is not None),
        key=lambda e: (observation_time(e), e["trace_id"], int(e["sequence"])),
    )
    for event in ordered:
        name, process, now = event["event"], process_scope(event), observation_time(event)
        if writer_scope(event) in usable:
            if name == "schedule.received" and event.get("pool_id") and event.get("scheduler_id"):
                bindings[process] = event
            if name in ("block.alloc", "block.acquire", "block.release") and event.get("pool_id"):
                key = (*observation_scope(event), event["pool_id"], event["trace_id"])
                pool = pools.setdefault(key, {})
                block = event["block_id"]
                if name == "block.alloc":
                    pool[block] = {
                        "epoch": event.get("alloc_epoch"),
                        "time": now,
                        "allocated_at": now,
                        "allocation": event_reference(event),
                        "owners": {},
                    }
                elif block in pool:
                    state = pool[block]
                    if str(state["epoch"]) != str(event.get("alloc_epoch")):
                        pool.pop(block)  # Do not repair inconsistent state from an ownership record.
                        continue
                    state["time"] = now
                    owners = state["owners"].setdefault(event["group_id"], {})
                    if name == "block.acquire":
                        owners[event["request_id"]] = event_reference(event)
                    else:
                        owners.pop(event["request_id"], None)
            if name in ("transfer.start", "transfer.complete", "transfer.error", "transfer.skipped"):
                for group_id, blocks in enumerate(event.get("local_block_ids", ())):
                    for block in blocks:
                        transfers[process, group_id, block] = event
        span = (writer_scope(event), event.get("span_id"))
        if name == "forward.begin" and event.get("phase") == "dummy" and event.get("span_id"):
            context = relations(event)
            spans[span] = (event, context)
            derived[id(event)] = context
        elif name == "cache.diff" and event.get("phase") == "dummy" and span in spans:
            begin, context = spans[span]
            current = relations({**begin, "monotonic_ns": event["monotonic_ns"]})
            current = {(r["group_id"], r["block_id"]): r for r in current}
            observations = []
            for relation in context:
                key = relation["group_id"], relation["block_id"]
                if key != (event["group_id"], event["block_id"]):
                    continue
                after = current.get(key)
                changed = after != relation
                observations.append({**relation, "context_changed_during_span": changed})
                if changed and after is not None:
                    observations.append({**after, "observed_at": "span_end", "context_changed_during_span": True})
            derived[id(event)] = observations
    return [{**e, "access_relations": derived[id(e)]} if id(e) in derived else e for e in events]


def related_request_keys(event):
    return {
        (event["run_id"], event.get("engine_id"), request_id)
        for relation in event.get("access_relations", ())
        for request_id in relation["owner_request_ids"]
    }


def include_relation_evidence(selected, available):
    """Keep the records supporting a derived relation, not unrelated request spans."""
    needed = set()
    for event in selected:
        for relation in event.get("access_relations", ()):
            ids = list(relation.get("evidence_event_ids", ()))
            if relation.get("latest_transfer"):
                ids.append(relation["latest_transfer"]["event_id"])
            needed.update((event["run_id"], event_id) for event_id in ids)
    chosen = {(writer_scope(e), str(e["sequence"])) for e in selected}
    return [
        e if (writer_scope(e), str(e["sequence"])) in chosen else {**e, "query_context": "access_relation_evidence"}
        for e in available
        if (writer_scope(e), str(e["sequence"])) in chosen or (e["run_id"], event_reference(e)) in needed
    ]


def select_events(
    events,
    request=None,
    block=None,
    group=None,
    engine=None,
    run_id=None,
    pool_id=None,
    epoch=None,
    host=None,
    dp_rank=None,
    tp_rank=None,
    pp_rank=None,
):
    selected = correlate_accesses([e for e in events if run_id is None or e["run_id"] == run_id])
    available = selected
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
        spans = {
            (writer_scope(e), e["span_id"])
            for e in selected
            if (request_keys(e) | related_request_keys(e)) & aliases and "span_id" in e
        }
        selected = [
            e
            for e in selected
            if (request_keys(e) | related_request_keys(e)) & aliases
            or (writer_scope(e), e.get("span_id")) in spans
            or e["event"].startswith("trace.")
        ]
    if engine is not None:
        selected = [e for e in selected if e.get("engine_id") == engine]
    for field, value in (("host", host), ("dp_rank", dp_rank), ("tp_rank", tp_rank), ("pp_rank", pp_rank)):
        if value is not None:
            selected = [e for e in selected if e.get(field) == value]
    if block is not None or pool_id is not None or epoch is not None:

        def matches(event):
            if pool_id is not None or epoch is not None:
                return any(
                    (pool_id is None or ref.get("pool_id") == pool_id)
                    and (epoch is None or str(ref.get("alloc_epoch")) == str(epoch))
                    and (block is None or ref["block_id"] == block)
                    and (group is None or ref.get("group_id") == group)
                    for ref in (*block_refs(event), *event.get("access_relations", ()))
                )
            return any(b == block and (group is None or g == group) for g, b in block_keys(event))

        related_requests = {key for e in selected if matches(e) for key in request_keys(e)}
        spans = {(writer_scope(e), e["span_id"]) for e in selected if "span_id" in e and matches(e)}
        selected = [
            e
            for e in selected
            if matches(e)
            or (writer_scope(e), e.get("span_id")) in spans
            or e["event"].startswith("trace.")
            or (not list(block_keys(e)) and request_keys(e) & related_requests)
        ]
    return include_relation_evidence(selected, available)


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
    parser.add_argument("--host", help="Host scope for a physical block query")
    parser.add_argument("--dp-rank", type=int)
    parser.add_argument("--tp-rank", type=int)
    parser.add_argument("--pp-rank", type=int)
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
        all_events,
        args.request,
        args.block,
        args.group,
        args.engine,
        args.run_id,
        args.pool_id,
        args.epoch,
        args.host,
        args.dp_rank,
        args.tp_rank,
        args.pp_rank,
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
        related = []
        for relation in event.get("access_relations", ()):
            if relation["association"] == "unknown":
                related.append(f"g{relation['group_id']}:b{relation['block_id']}=unknown({relation['reason']})")
            else:
                owners = ",".join(relation["owner_request_ids"]) or "none"
                transfer = relation.get("latest_transfer", {})
                related.append(
                    f"g{relation['group_id']}:b{relation['block_id']}@{relation['pool_id']}/e{relation['alloc_epoch']}"
                    f" owners={owners} basis=host-observed transfer={transfer.get('event_id', '-')}"
                    f" at={relation['observed_at']}"
                    f" window_changed={relation.get('context_changed_during_span', 'n/a')}"
                )
        print(
            f"{event['wall_time_ns']} {event['host']} pid={event['pid']} "
            f"engine={event.get('engine_id', '-')} dp={event.get('dp_rank', '-')} "
            f"tp={event.get('tp_rank', '-')} {event['event']} "
            f"req={','.join(sorted(request_ids(event))) or '-'} blocks={blocks or '-'} "
            f"step={event.get('step_id', '-')} epoch={event.get('alloc_epoch', '-')} "
            f"pool={event.get('pool_id', '-')}{warning}"
        )
        for relation in related:
            print(f"  related: {relation}")
        if event.get("query_context"):
            print(f"  context: {event['query_context']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
