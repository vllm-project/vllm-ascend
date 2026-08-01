#!/usr/bin/env python3
"""Summarize expert offload cache hit rates from vllm-ascend logs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, TextIO


DETAIL_RE = re.compile(
    r"\[UPDATE-W\]\s+l=(?P<layer>\d+)\s+call=(?P<call>\d+).*?\|needed\|=(?P<needed_count>\d+).*?needed=(?P<needed>\[[^\]]*\])"
)
HIT_RE = re.compile(
    r"\[UPDATE-W\]\s+l=(?P<layer>\d+)\s+cache_hit=(?P<hit>\[[^\]]*\])\s+cache_miss=(?P<miss>\[[^\]]*\])"
)


@dataclass
class Event:
    layer: int
    call: int | None
    hits: int
    misses: int
    requests: int
    hit_experts: list[int]
    miss_experts: list[int]
    needed_experts: list[int] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0


@dataclass
class Bucket:
    calls: int = 0
    hits: int = 0
    misses: int = 0
    requests: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    def add(self, event: Event) -> None:
        self.calls += 1
        self.hits += event.hits
        self.misses += event.misses
        self.requests += event.requests


@dataclass
class Summary:
    layers: dict[int, Bucket]
    event_steps: dict[int, Bucket]
    decode_steps: dict[int, Bucket]
    events: list[Event]
    global_hits: int
    global_misses: int
    global_requests: int
    decode_step_layers: int | None = None

    @property
    def global_hit_rate(self) -> float:
        return self.global_hits / self.global_requests if self.global_requests else 0.0


def _parse_list(value: str) -> list[int]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"expected list, got {value}")
    return [int(item) for item in parsed]


def parse_lines(lines: Iterable[str]) -> list[Event]:
    pending: dict[int, tuple[int, list[int]]] = {}
    events: list[Event] = []

    for line in lines:
        detail_match = DETAIL_RE.search(line)
        if detail_match:
            layer = int(detail_match.group("layer"))
            call = int(detail_match.group("call"))
            needed = _parse_list(detail_match.group("needed"))
            pending[layer] = (call, needed)
            continue

        hit_match = HIT_RE.search(line)
        if not hit_match:
            continue

        layer = int(hit_match.group("layer"))
        hit_experts = _parse_list(hit_match.group("hit"))
        miss_experts = _parse_list(hit_match.group("miss"))
        call = None
        needed_experts: list[int] = []
        if layer in pending:
            call, needed_experts = pending.pop(layer)

        requests = len(needed_experts) if needed_experts else len(hit_experts) + len(miss_experts)
        events.append(
            Event(
                layer=layer,
                call=call,
                hits=len(hit_experts),
                misses=len(miss_experts),
                requests=requests,
                hit_experts=hit_experts,
                miss_experts=miss_experts,
                needed_experts=needed_experts,
            )
        )

    return events


def _infer_decode_step_layers(events: list[Event]) -> int | None:
    layers = {event.layer for event in events}
    if not layers:
        return None
    return max(layers) + 1


def summarize(events: list[Event], decode_step_layers: int | None = None) -> Summary:
    layers: dict[int, Bucket] = defaultdict(Bucket)
    event_steps: dict[int, Bucket] = defaultdict(Bucket)
    decode_steps: dict[int, Bucket] = defaultdict(Bucket)
    global_bucket = Bucket()
    if decode_step_layers is None:
        decode_step_layers = _infer_decode_step_layers(events)

    for event in events:
        layers[event.layer].add(event)
        global_bucket.add(event)
        if event.call is not None:
            event_steps[event.call].add(event)
            if decode_step_layers:
                decode_steps[event.call // decode_step_layers].add(event)

    return Summary(
        layers=dict(layers),
        event_steps=dict(event_steps),
        decode_steps=dict(decode_steps),
        events=events,
        global_hits=global_bucket.hits,
        global_misses=global_bucket.misses,
        global_requests=global_bucket.requests,
        decode_step_layers=decode_step_layers,
    )


def _open_log(path: str) -> TextIO:
    if path == "-":
        return sys.stdin
    return Path(path).open("r", encoding="utf-8", errors="replace")


def _print_table(summary: Summary, top_steps: int) -> None:
    print("GLOBAL")
    print(
        f"  requests={summary.global_requests} hits={summary.global_hits} "
        f"misses={summary.global_misses} hit_rate={summary.global_hit_rate:.6f}"
    )

    print("\nPER_LAYER")
    print("layer,calls,requests,hits,misses,hit_rate")
    for layer, bucket in sorted(summary.layers.items()):
        print(
            f"{layer},{bucket.calls},{bucket.requests},{bucket.hits},"
            f"{bucket.misses},{bucket.hit_rate:.6f}"
        )

    print("\nPER_EVENT_STEP")
    print("step,events,requests,hits,misses,hit_rate")
    for step, bucket in sorted(summary.event_steps.items())[:top_steps]:
        print(
            f"{step},{bucket.calls},{bucket.requests},{bucket.hits},"
            f"{bucket.misses},{bucket.hit_rate:.6f}"
        )
    if len(summary.event_steps) > top_steps:
        print(f"... truncated {len(summary.event_steps) - top_steps} event steps; use --top-steps to show more")

    print(f"\nPER_DECODE_STEP decode_step_layers={summary.decode_step_layers}")
    print("decode_step,layer_events,requests,hits,misses,hit_rate")
    for step, bucket in sorted(summary.decode_steps.items())[:top_steps]:
        print(
            f"{step},{bucket.calls},{bucket.requests},{bucket.hits},"
            f"{bucket.misses},{bucket.hit_rate:.6f}"
        )
    if len(summary.decode_steps) > top_steps:
        print(f"... truncated {len(summary.decode_steps) - top_steps} decode steps; use --top-steps to show more")


def _write_csv(path: str, rows: Iterable[dict[str, int | float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_summary(summary: Summary) -> dict[str, object]:
    return {
        "global": {
            "requests": summary.global_requests,
            "hits": summary.global_hits,
            "misses": summary.global_misses,
            "hit_rate": summary.global_hit_rate,
        },
        "decode_step_layers": summary.decode_step_layers,
        "layers": {
            str(layer): {
                "calls": bucket.calls,
                "requests": bucket.requests,
                "hits": bucket.hits,
                "misses": bucket.misses,
                "hit_rate": bucket.hit_rate,
            }
            for layer, bucket in sorted(summary.layers.items())
        },
        "event_steps": {
            str(step): {
                "events": bucket.calls,
                "requests": bucket.requests,
                "hits": bucket.hits,
                "misses": bucket.misses,
                "hit_rate": bucket.hit_rate,
            }
            for step, bucket in sorted(summary.event_steps.items())
        },
        "decode_steps": {
            str(step): {
                "layer_events": bucket.calls,
                "requests": bucket.requests,
                "hits": bucket.hits,
                "misses": bucket.misses,
                "hit_rate": bucket.hit_rate,
            }
            for step, bucket in sorted(summary.decode_steps.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="log path, or '-' for stdin")
    parser.add_argument("--json", action="store_true", help="print JSON instead of tables")
    parser.add_argument("--top-steps", type=int, default=30, help="number of per-step rows to print")
    parser.add_argument(
        "--decode-step-layers",
        default="auto",
        help="MoE layer count used to group event calls into decode steps; use 'auto' or an integer",
    )
    parser.add_argument("--layer-csv", help="write per-layer stats as CSV")
    parser.add_argument("--event-step-csv", help="write per-event-step stats as CSV")
    parser.add_argument("--step-csv", help="write per-decode-step stats as CSV")
    args = parser.parse_args()

    with _open_log(args.log) as log_file:
        events = parse_lines(log_file)

    if args.decode_step_layers == "auto":
        decode_step_layers = None
    else:
        decode_step_layers = int(args.decode_step_layers)
        if decode_step_layers < 1:
            raise ValueError("--decode-step-layers must be >= 1")
    summary = summarize(events, decode_step_layers=decode_step_layers)

    if args.layer_csv:
        _write_csv(
            args.layer_csv,
            (
                {
                    "layer": layer,
                    "calls": bucket.calls,
                    "requests": bucket.requests,
                    "hits": bucket.hits,
                    "misses": bucket.misses,
                    "hit_rate": bucket.hit_rate,
                }
                for layer, bucket in sorted(summary.layers.items())
            ),
        )
    if args.event_step_csv:
        _write_csv(
            args.event_step_csv,
            (
                {
                    "step": step,
                    "events": bucket.calls,
                    "requests": bucket.requests,
                    "hits": bucket.hits,
                    "misses": bucket.misses,
                    "hit_rate": bucket.hit_rate,
                }
                for step, bucket in sorted(summary.event_steps.items())
            ),
        )
    if args.step_csv:
        _write_csv(
            args.step_csv,
            (
                {
                    "decode_step": step,
                    "layer_events": bucket.calls,
                    "requests": bucket.requests,
                    "hits": bucket.hits,
                    "misses": bucket.misses,
                    "hit_rate": bucket.hit_rate,
                }
                for step, bucket in sorted(summary.decode_steps.items())
            ),
        )

    if args.json:
        print(json.dumps(_json_summary(summary), indent=2, sort_keys=True))
    else:
        _print_table(summary, args.top_steps)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
