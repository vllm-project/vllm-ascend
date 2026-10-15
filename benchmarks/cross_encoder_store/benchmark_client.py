#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed-loop client for the cross-Encoder Store benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiohttp


@dataclass(frozen=True)
class RequestResult:
    ordinal: int
    request_id: str
    source_idx: str
    encoder: int
    pd: int
    client_process: int
    status: int
    ttft_s: float | None
    e2e_s: float
    tpot_s: float | None
    output_tokens: int
    error: str | None


def percentile(values: list[float], percent: float) -> float | None:
    """Return a linearly interpolated percentile."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def load_requests(path: Path) -> list[dict[str, Any]]:
    requests = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not requests:
        raise ValueError(f"request manifest is empty: {path}")
    return requests


def request_owner(request: dict[str, Any], phase: str, ordinal: int) -> int:
    if phase == "population":
        owner = request.get("population_encoder")
    elif phase == "measurement":
        owner = request.get("measurement_encoder")
    elif phase == "warmup":
        owner = ordinal % 2
    else:
        raise ValueError(f"unknown phase: {phase}")
    if owner not in (0, 1):
        raise ValueError(f"request {request.get('request_id')} has invalid owner {owner!r}")
    return owner


def build_messages(request: dict[str, Any], image_base_url: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in request["content_parts"]:
        if part["type"] == "text":
            content.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image":
            relative_path = quote(part["path"], safe="/")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"{image_base_url.rstrip('/')}/{relative_path}"},
                }
            )
        else:
            raise ValueError(f"unsupported content part: {part!r}")
    return [{"role": "user", "content": content}]


def route_urls(args: argparse.Namespace) -> dict[tuple[int, int], str]:
    return {
        (0, 0): args.e0_pd0_url.rstrip("/"),
        (0, 1): args.e0_pd1_url.rstrip("/"),
        (1, 0): args.e1_pd0_url.rstrip("/"),
        (1, 1): args.e1_pd1_url.rstrip("/"),
    }


async def run_one(
    session: aiohttp.ClientSession,
    request: dict[str, Any],
    ordinal: int,
    args: argparse.Namespace,
    urls: dict[tuple[int, int], str],
    client_process: int = 0,
) -> RequestResult:
    encoder = request_owner(request, args.phase, ordinal)
    pd = ordinal % 2
    request_id = f"ces-{args.phase}-{ordinal:04d}-{request['source_idx']}"
    body = {
        "model": args.served_model_name,
        "messages": build_messages(request, args.image_base_url),
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    first_output_at: float | None = None
    output_tokens = 0
    status = 0
    error: str | None = None
    try:
        async with session.post(
            f"{urls[(encoder, pd)]}/v1/chat/completions",
            json=body,
            headers={"x-request-id": request_id},
        ) as response:
            status = response.status
            if response.status != 200:
                error = (await response.text())[:2000]
            else:
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    event = json.loads(payload)
                    usage = event.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        output_tokens = int(usage["completion_tokens"])
                    choices = event.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        generated = delta.get("content") or delta.get("reasoning_content")
                        if generated and first_output_at is None:
                            first_output_at = time.perf_counter()
    except Exception as exc:  # Keep all request records for failed benchmark runs.
        error = f"{type(exc).__name__}: {exc}"

    finished = time.perf_counter()
    ttft_s = None if first_output_at is None else first_output_at - started
    tpot_s = None
    if ttft_s is not None and output_tokens > 1:
        tpot_s = (finished - first_output_at) / (output_tokens - 1)
    if status == 200 and first_output_at is None and error is None:
        error = "stream completed without nonempty generated content"
    return RequestResult(
        ordinal=ordinal,
        request_id=request_id,
        source_idx=str(request["source_idx"]),
        encoder=encoder,
        pd=pd,
        client_process=client_process,
        status=status,
        ttft_s=ttft_s,
        e2e_s=finished - started,
        tpot_s=tpot_s,
        output_tokens=output_tokens,
        error=error,
    )


async def run_closed_loop(
    indexed_requests: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
    *,
    concurrency: int | None = None,
    client_process: int = 0,
) -> tuple[list[RequestResult], float, float]:
    queue: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
    for ordinal, request in indexed_requests:
        queue.put_nowait((ordinal, request))
    results: list[RequestResult] = []
    urls = route_urls(args)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=0)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def worker() -> None:
            while True:
                try:
                    ordinal, request = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    results.append(
                        await run_one(
                            session,
                            request,
                            ordinal,
                            args,
                            urls,
                            client_process,
                        )
                    )
                finally:
                    queue.task_done()

        started = time.perf_counter()
        worker_count = min(concurrency or args.concurrency, len(indexed_requests))
        await asyncio.gather(*(worker() for _ in range(worker_count)))
        finished = time.perf_counter()
    return sorted(results, key=lambda result: result.ordinal), started, finished


def partition_requests(requests: list[dict[str, Any]], process_count: int) -> list[list[tuple[int, dict[str, Any]]]]:
    groups: list[list[tuple[int, dict[str, Any]]]] = [[] for _ in range(process_count)]
    for ordinal, request in enumerate(requests):
        groups[ordinal % process_count].append((ordinal, request))
    return groups


def client_process_main(
    indexed_requests: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
    process_index: int,
    concurrency: int,
    start_event: Any,
    ready_queue: Any,
    result_queue: Any,
) -> None:
    ready_sent = False
    try:
        ready_queue.put({"status": "ready", "process_index": process_index})
        ready_sent = True
        start_event.wait()
        results, started, finished = asyncio.run(
            run_closed_loop(
                indexed_requests,
                args,
                concurrency=concurrency,
                client_process=process_index,
            )
        )
        result_queue.put(
            {
                "status": "ok",
                "process_index": process_index,
                "started": started,
                "finished": finished,
                "results": [asdict(result) for result in results],
            }
        )
    except BaseException as exc:
        message = {
            "status": "error",
            "process_index": process_index,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        (result_queue if ready_sent else ready_queue).put(message)


def run_client_processes(requests: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[RequestResult], float]:
    process_count = min(args.client_processes, len(requests))
    groups = partition_requests(requests, process_count)
    base_concurrency, extra = divmod(args.concurrency, process_count)
    if base_concurrency == 0:
        raise ValueError("concurrency must be at least client-processes")

    context = mp.get_context("spawn")
    start_event = context.Event()
    ready_queue = context.Queue()
    result_queue = context.Queue()
    processes = []
    for process_index, group in enumerate(groups):
        process_concurrency = base_concurrency + (process_index < extra)
        process = context.Process(
            target=client_process_main,
            args=(
                group,
                args,
                process_index,
                process_concurrency,
                start_event,
                ready_queue,
                result_queue,
            ),
            name=f"cross-encoder-store-client-{args.phase}-{process_index}",
        )
        process.start()
        processes.append(process)

    outputs = []
    try:
        for _ in processes:
            message = ready_queue.get(timeout=args.request_timeout)
            if message["status"] != "ready":
                raise RuntimeError(f"client preparation failed: {message['error']}\n{message['traceback']}")
        print(
            f"{args.phase}: starting {len(requests)} requests across "
            f"{process_count} client processes (total concurrency "
            f"{args.concurrency})",
            flush=True,
        )
        start_event.set()
        for _ in processes:
            while True:
                try:
                    message = result_queue.get(timeout=1)
                    break
                except queue.Empty:
                    failed = [process for process in processes if process.exitcode not in (None, 0)]
                    if failed:
                        raise RuntimeError(
                            "client process exited before returning results: "
                            + ", ".join(f"{process.name}={process.exitcode}" for process in failed)
                        )
            if message["status"] != "ok":
                raise RuntimeError(f"client run failed: {message['error']}\n{message['traceback']}")
            outputs.append(message)
    finally:
        if not start_event.is_set():
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()

    started = min(output["started"] for output in outputs)
    finished = max(output["finished"] for output in outputs)
    results = [RequestResult(**record) for output in outputs for record in output["results"]]
    return sorted(results, key=lambda result: result.ordinal), finished - started


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarize(results: list[RequestResult], wall_s: float, args: argparse.Namespace) -> dict[str, Any]:
    successful = [result for result in results if result.status == 200 and result.error is None]
    return {
        "phase": args.phase,
        "manifest": str(args.manifest),
        "concurrency": args.concurrency,
        "client_processes": min(args.client_processes, len(results)),
        "requests": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "wall_s": wall_s,
        "request_throughput": len(successful) / wall_s if wall_s else None,
        "image_references": sum(len(request["images"]) for request in load_requests(args.manifest)),
        "output_tokens": sum(result.output_tokens for result in successful),
        "output_token_throughput": (sum(result.output_tokens for result in successful) / wall_s if wall_s else None),
        "ttft_s": distribution([result.ttft_s for result in successful if result.ttft_s is not None]),
        "e2e_s": distribution([result.e2e_s for result in successful]),
        "tpot_s": distribution([result.tpot_s for result in successful if result.tpot_s is not None]),
        "encoder_requests": {
            "E0": sum(result.encoder == 0 for result in results),
            "E1": sum(result.encoder == 1 for result in results),
        },
        "pd_requests": {
            "PD0": sum(result.pd == 0 for result in results),
            "PD1": sum(result.pd == 1 for result in results),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("warmup", "population", "measurement"),
        required=True,
    )
    parser.add_argument("--image-base-url", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--e0-pd0-url", required=True)
    parser.add_argument("--e0-pd1-url", required=True)
    parser.add_argument("--e1-pd0-url", required=True)
    parser.add_argument("--e1-pd1-url", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--client-processes", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.client_processes < 1:
        parser.error("--client-processes must be positive")
    if args.client_processes > args.concurrency:
        parser.error("--client-processes cannot exceed --concurrency")
    return args


def main() -> None:
    args = parse_args()
    requests = load_requests(args.manifest)
    results, wall_s = run_client_processes(requests, args)
    report = {
        "summary": summarize(results, wall_s, args),
        "requests": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    if report["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
