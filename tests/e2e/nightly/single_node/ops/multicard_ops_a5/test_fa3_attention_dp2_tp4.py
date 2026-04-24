import multiprocessing as mp
from queue import Empty

import pytest
import torch

from tests.e2e.nightly.single_node.ops.fa3_attention_test_utils import (
    cleanup_npu,
    cosine_similarity_per_req,
    make_fa3_batch,
    make_fa3_decode_case,
    require_a5,
    run_fa3_operator,
    shard_case_by_heads,
)


WORLD_SIZE = 8
TP_SIZE = 4
DP_SIZE = 2
RESULT_TIMEOUT_S = 300


def _worker(rank: int, result_queue: mp.Queue):
    try:
        require_a5()

        torch.npu.set_device(rank)
        device = torch.device(f"npu:{rank}")
        local_tp_rank = rank % TP_SIZE
        local_dp_rank = rank // TP_SIZE

        full_case = make_fa3_decode_case(
            seq_lens=[129, 257, 193, 385],
            num_heads=128,
            latent_dim=512,
            rope_dim=64,
            block_size=128,
            device=device,
            seed=20260421,
        )

        local_heads = full_case.num_heads // TP_SIZE
        local_case = shard_case_by_heads(
            full_case,
            head_start=local_tp_rank * local_heads,
            head_end=(local_tp_rank + 1) * local_heads,
        )

        scenarios = {
            "global_2req": [local_dp_rank],
            "global_4req": [local_dp_rank * 2, local_dp_rank * 2 + 1],
        }

        for case_name, req_ids in scenarios.items():
            single_outputs = []
            for req_id in req_ids:
                single_batch = make_fa3_batch(local_case, [req_id])
                single_outputs.append(run_fa3_operator(single_batch, local_case).squeeze(0))

            batched = make_fa3_batch(local_case, req_ids)
            batched_output = run_fa3_operator(batched, local_case)
            single_outputs_tensor = torch.stack(single_outputs, dim=0)

            batched_vs_single = (batched_output.float() - single_outputs_tensor.float()).abs()
            batched_vs_golden = (batched_output.float() - batched.golden_output.float()).abs()
            cos = cosine_similarity_per_req(batched_output, batched.golden_output)

            result_queue.put(
                {
                    "rank": rank,
                    "dp_rank": local_dp_rank,
                    "tp_rank": local_tp_rank,
                    "case": case_name,
                    "max_diff_single": batched_vs_single.max().item(),
                    "max_diff_golden": batched_vs_golden.max().item(),
                    "min_cosine_golden": cos.min().item(),
                    "local_batch_size": len(req_ids),
                }
            )
    finally:
        cleanup_npu()


@torch.inference_mode()
def test_fa3_operator_emulates_dp2_tp4_local_batch_switch():
    require_a5()

    if torch.npu.device_count() < WORLD_SIZE:
        pytest.skip(f"This test needs {WORLD_SIZE} NPUs, got {torch.npu.device_count()}.")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    workers = [ctx.Process(target=_worker, args=(rank, result_queue)) for rank in range(WORLD_SIZE)]

    for worker in workers:
        worker.start()

    results = []
    try:
        for _ in range(WORLD_SIZE * 2):
            results.append(result_queue.get(timeout=RESULT_TIMEOUT_S))
    except Empty as exc:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
        raise RuntimeError("Timed out waiting for FA3 dp2 tp4 worker results.") from exc
    finally:
        for worker in workers:
            worker.join(timeout=RESULT_TIMEOUT_S)
            if worker.is_alive():
                worker.kill()

    failed_workers = [worker for worker in workers if worker.exitcode != 0]
    if failed_workers:
        exit_codes = {idx: worker.exitcode for idx, worker in enumerate(workers)}
        raise RuntimeError(f"Some FA3 dp2 tp4 workers failed: {exit_codes}")

    assert len(results) == WORLD_SIZE * 2

    results_by_case = {}
    for result in results:
        results_by_case.setdefault(result["case"], []).append(result)

    assert len(results_by_case["global_2req"]) == WORLD_SIZE
    assert len(results_by_case["global_4req"]) == WORLD_SIZE

    for result in results_by_case["global_2req"]:
        assert result["local_batch_size"] == 1, result
        assert result["max_diff_single"] <= 2e-2, result
        assert result["max_diff_golden"] <= 6e-2, result
        assert result["min_cosine_golden"] >= 0.995, result

    for result in results_by_case["global_4req"]:
        assert result["local_batch_size"] == 2, result
        assert result["max_diff_single"] <= 2e-2, result
        assert result["max_diff_golden"] <= 6e-2, result
        assert result["min_cosine_golden"] >= 0.995, result
