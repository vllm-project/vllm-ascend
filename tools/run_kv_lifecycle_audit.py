#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput

from tests.ut.kv_offload.utils import (
    assert_scheduler_empty,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic KV lifecycle audit using the unit-test scheduler helpers."
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-tokens", type=int, default=40)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--out", default="/tmp/kv_lifecycle_audit.json")
    return parser.parse_args()


def snapshot(scheduler, label: str) -> dict[str, object]:
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    return {
        "label": label,
        "free_blocks": scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks,
        "num_requests": len(scheduler.requests),
        "running": len(scheduler.running),
        "waiting": len(scheduler.waiting),
        "finished_req_ids": sorted(scheduler.finished_req_ids),
        "req_to_blocks": {
            req_id: len(blocks)
            for req_id, blocks in manager.req_to_blocks.items()
        },
    }


def run_one_iteration(iteration: int, args: argparse.Namespace) -> dict[str, object]:
    vllm_config = create_vllm_config(
        max_num_batched_tokens=args.max_num_batched_tokens,
        block_size=args.block_size,
    )
    scheduler = create_scheduler(vllm_config)
    start_free_blocks = scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks

    request = create_request(
        request_id=iteration + 1,
        max_tokens=1,
        num_tokens=args.num_tokens,
        do_remote_decode=True,
        block_size=args.block_size,
    )
    request_id = request.request_id

    states: list[dict[str, object]] = [snapshot(scheduler, "start")]

    scheduler.add_request(request)
    states.append(snapshot(scheduler, "after_add_request"))

    scheduler_output = scheduler.schedule()
    states.append(snapshot(scheduler, "after_schedule_step1"))

    model_runner_output = create_model_runner_output(reqs=[request])
    scheduler.update_from_output(scheduler_output, model_runner_output)
    states.append(snapshot(scheduler, "after_update_from_output_step1"))

    connector_scheduler = scheduler.connector.connector_scheduler if scheduler.connector else None
    delayed_send_marked = connector_scheduler is not None and request_id in connector_scheduler._reqs_need_send

    scheduler_output = scheduler.schedule()
    states.append(snapshot(scheduler, "after_schedule_step2"))

    scheduler.update_from_output(scheduler_output, EMPTY_MODEL_RUNNER_OUTPUT)
    states.append(snapshot(scheduler, "after_update_from_output_step2"))

    scheduler_output = scheduler.schedule()
    states.append(snapshot(scheduler, "after_schedule_step3"))

    model_runner_output = copy.deepcopy(EMPTY_MODEL_RUNNER_OUTPUT)
    model_runner_output.kv_connector_output = KVConnectorOutput(finished_sending={request_id})
    scheduler.update_from_output(scheduler_output, model_runner_output)
    states.append(snapshot(scheduler, "after_finished_sending"))

    assert_scheduler_empty(scheduler)
    end_free_blocks = scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks

    return {
        "iteration": iteration,
        "request_id": request_id,
        "start_free_blocks": start_free_blocks,
        "end_free_blocks": end_free_blocks,
        "delayed_send_marked": delayed_send_marked,
        "states": states,
    }


def main() -> int:
    args = parse_args()
    records = [run_one_iteration(i, args) for i in range(args.iterations)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"iterations": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
