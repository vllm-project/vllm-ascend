#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stream server/client logs into at most 280 lines; never infer output accuracy."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def summarize(paths):
    counts = Counter()
    layouts = defaultdict(dict)
    allocations = defaultdict(dict)
    configs = {}
    errors = []
    evidence = Counter()
    sources = {}
    waiting = {}
    released = {}
    comparisons = []
    ranges = Counter()
    for path in paths:
        with Path(path).open(errors="replace") as log:
            for number, line in enumerate(log, 1):
                if any(word in line for word in ("Traceback", "ERROR", "save_failed")) and len(errors) < 50:
                    errors.append(f"{Path(path).name}:{number} {line.strip()[:600]}")
                marker = next(
                    (m for m in ("DSV4_OFFLOAD ", "[KVPOOL_RANGE_DEBUG] ", "DSV4_VALIDATION ") if m in line), None
                )
                if marker is None:
                    continue
                raw = line.split(marker, 1)[1].strip()
                if not raw.startswith("{"):
                    continue
                try:
                    event = json.loads(raw)
                    kind = event["event"]
                except (ValueError, KeyError):
                    evidence["malformed"] += 1
                    continue
                # A file is the host scope; never compare host monotonic clocks.
                process = (str(path), event.get("pid"))
                counts[kind] += 1
                layer = event.get("layer")
                if kind == "step_begin":
                    for table in (sources, waiting, released):
                        for key in [key for key in table if key[0] == process]:
                            del table[key]
                elif kind == "config":
                    configs[process] = event
                elif kind == "component":
                    layouts[process][(event["component"], event["segment"])] = event
                elif kind == "allocation":
                    allocations[process][event["address"]] = event
                elif kind == "source_done":
                    sources[process, layer] = event["t_ns"]
                elif kind == "slot_wait":
                    waiting[process, layer] = event
                    evidence["blocked_waits"] += bool(event["blocked"])
                elif kind == "slot_release":
                    wait = waiting.pop((process, layer), None)
                    done = sources.get((process, event["predecessor"]))
                    if wait is None or done is None or done > event["t_ns"] or wait["t_ns"] > event["t_ns"]:
                        evidence["bad_fences"] += 1
                    else:
                        evidence["ordered_fences"] += 1
                    released[process, layer] = event["t_ns"]
                elif kind == "compute_release":
                    if event.get("predecessor") is not None:
                        release = released.pop((process, layer), None)
                        if release is None or release > event["t_ns"]:
                            evidence["bad_compute_gates"] += 1
                elif kind == "load_result":
                    evidence["failed_loads"] += event["failed"]
                elif kind == "restore":
                    evidence[f"restore_{event['source']}"] += event["keys"]
                    if event["source"] == "previous_step" and not event["shared"] and event["keys"]:
                        evidence["resident_group_restore"] += event["keys"]
                elif kind == "comparison":
                    comparisons.append(event)
                elif kind == "range":
                    ranges[event["direction"] + "_bytes"] += sum(event["requested_bytes"])
                    ranges[event["direction"] + "_failed"] += sum(code < 0 for code in event["results"])
                elif kind == "push_prefix":
                    evidence["d_local_prefix_requests"] += event["local_tokens"] > 0
                    evidence["d_local_prefix_tokens"] += event["local_tokens"]
                    evidence["d_push_prefix_tokens"] += event["external_tokens"]
                elif kind == "p2p_receive":
                    evidence["p2p_receive_failed"] += bool(event["failed"])
                    evidence["p2p_receive_complete"] += not event["failed"]
                elif kind == "p2p_send":
                    ranges["p2p_bytes"] += event["bytes"]
                    ranges["p2p_failed"] += bool(event["failed"])
                elif kind == "commit":
                    evidence["commit_failed"] += sum(code != 0 for code in event["results"])
    output = ["DSV4 offload acceptance evidence (PASS / FAIL / INSUFFICIENT)"]
    output.extend("ERROR " + error for error in errors)
    output.append("EVENTS " + json.dumps(dict(counts), sort_keys=True))
    output.append("TRANSFER " + json.dumps(dict(ranges), sort_keys=True))
    output.append("EVIDENCE " + json.dumps(dict(evidence), sort_keys=True))
    bad_layout = 0
    budget_checks = 0
    complete_layouts = set()
    incomplete_layouts = 0
    for process, config in configs.items():
        entries = layouts[process]
        # Ordinary consumer allocations are outside the shared-ring probe.
        if not entries and not config["predecessors"] and config.get("buffers") is None:
            continue
        seen = defaultdict(list)
        for entry in entries.values():
            seen[entry["address"]].append(entry)
        conflicts = [
            members
            for members in seen.values()
            if len({m["component"] for m in members}) > 1 and any(not m["shared"] for m in members)
        ]
        bad_layout += len(conflicts)
        by_layer = defaultdict(dict)
        for entry in entries.values():
            if entry["component"].endswith((".self_attn", ".attn")):
                match = re.search(r"layers\.(\d+)", entry["component"])
                if match:
                    by_layer[int(match.group(1))][entry["segment"]] = (
                        entry["address"],
                        entry["stride"],
                        entry["page_bytes"],
                    )
        checked_edges = 0
        for layer, predecessor in config["predecessors"].items():
            left, right = by_layer[int(layer)], by_layer[int(predecessor)]
            if left and right:
                checked_edges += 1
                bad_layout += left != right
        component_count = len({e["component"] for e in entries.values()})
        if component_count == config.get("expected_components") and checked_edges == len(config["predecessors"]):
            complete_layouts.add(config.get("buffers"))
        else:
            incomplete_layouts += 1
        actual = sum(event["bytes"] for event in allocations[process].values())
        if actual:
            budget_checks += 1
            evidence["budget_mismatch"] += actual != config["planned_bytes"]
        output.append(
            "CONFIG "
            + json.dumps(
                dict(
                    file=Path(process[0]).name,
                    pid=process[1],
                    blocks=config["blocks"],
                    predecessors=config["predecessors"],
                    components=len(entries),
                    resident_alias_conflicts=len(conflicts),
                    planned_bytes=config["planned_bytes"],
                    allocation_bytes=actual,
                    alignment_reserve=sum(a["alignment_reserve"] for a in allocations[process].values()),
                ),
                sort_keys=True,
            )
        )
    failed = bool(
        errors
        or bad_layout
        or evidence["bad_fences"]
        or evidence["bad_compute_gates"]
        or evidence["failed_loads"]
        or evidence["p2p_receive_failed"]
        or evidence["commit_failed"]
        or evidence["budget_mismatch"]
        or any(ranges[key] for key in ranges if key.endswith("_failed"))
    )
    layout_status = (
        "FAIL"
        if bad_layout
        else ("PASS" if {1, 2, 3, 4} <= complete_layouts and not incomplete_layouts else "INSUFFICIENT")
    )
    output.append(
        f"LAYOUT {layout_status} complete_buffers={sorted(b for b in complete_layouts if isinstance(b, int))} "
        f"conflicts={bad_layout} incomplete={incomplete_layouts}"
    )
    output.append(
        f"BUDGET {'FAIL' if evidence['budget_mismatch'] else 'PASS' if budget_checks else 'INSUFFICIENT'} "
        f"checks={budget_checks}"
    )
    comparison_failed = any(not e.get("equal", False) for e in comparisons)
    output.append(
        f"CONTINUATION {'FAIL' if comparison_failed else 'PASS' if comparisons else 'INSUFFICIENT'} "
        f"comparisons={len(comparisons)}"
    )
    output.append("PREFIX_CONCURRENCY INSUFFICIENT (requires endpoint/miss/fault workload review)")
    fence_status = (
        "FAIL"
        if evidence["bad_fences"] or evidence["bad_compute_gates"]
        else ("PASS" if evidence["ordered_fences"] and evidence["blocked_waits"] else "INSUFFICIENT")
    )
    output.append(f"SLOT_FENCE {fence_status} blocked={evidence['blocked_waits']} ordered={evidence['ordered_fences']}")
    output.append("LIFECYCLE INSUFFICIENT (requires completion/cancellation/GET/PUT/capacity fault cases)")
    output.append(
        "RESULT "
        + ("FAIL" if failed or comparison_failed else "INSUFFICIENT: server scenario coverage requires review")
    )
    return output if len(output) <= 280 else output[:270] + output[-10:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()
    print("\n".join(summarize(args.logs)))


if __name__ == "__main__":
    main()
