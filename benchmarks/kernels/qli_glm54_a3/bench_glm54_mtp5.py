# SPDX-License-Identifier: Apache-2.0
"""Sweep GLM-5.4 paged INT8 QLI for MTP5 on one Ascend A3 die."""

import argparse
import json
from pathlib import Path

import torch
from bench_qli import run_case
from qli_fixture import HEADS, PHYSICAL_LAYOUTS, TOPK, WIDTH, case_config

ROWS_PER_REQUEST = 6
PAGE_SIZE = 128
TILELANG_BACKEND = "tilelang-v29"
DEFAULT_QLI_CONFIG = {
    "index_n_heads": HEADS,
    "index_head_dim": WIDTH,
    "index_topk": TOPK,
    "max_position_embeddings": 1048576,
}


def read_qli_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))["text_config"]
    fields = ("index_n_heads", "index_head_dim", "index_topk", "max_position_embeddings")
    qli = {name: config[name] for name in fields}
    if any(type(value) is not int or value < 1 for value in qli.values()):
        raise ValueError("Expected positive integer QLI configuration dimensions")
    if tuple(qli[name] for name in fields[:3]) != (HEADS, WIDTH, TOPK):
        raise ValueError("The current AscendC/TileLang fixture requires 32 Q heads, width 128, and top-k 2048")
    return qli


def make_case(args, batch):
    case = case_config(
        f"b{batch}-mtp5",
        batch,
        ROWS_PER_REQUEST,
        args.context,
        PAGE_SIZE,
        args.seed,
        physical_layout=args.physical_layout,
    )
    rows = batch * ROWS_PER_REQUEST
    pages_per_request = (args.context + PAGE_SIZE - 1) // PAGE_SIZE
    case.update(
        input_source="synthetic_paged_glm54_a3_c8",
        query_shape=[rows, HEADS, WIDTH],
        key_shape=[batch * pages_per_request, PAGE_SIZE, 1, WIDTH],
        block_table_shape=[batch, pages_per_request],
        output_shape=[rows, 1, TOPK],
        visible_length_range=[args.context - ROWS_PER_REQUEST + 1, args.context],
    )
    return case


def checkpoint(output, report):
    pending = output / "timing.json.pending"
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(output / "timing.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional GLM-5.4 config with QLI dimensions in text_config")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-start", type=int, default=1)
    parser.add_argument("--batch-stop", type=int, default=80, help="inclusive final batch")
    parser.add_argument(
        "--context", type=int, default=131072, help="visible K length including the final verification token"
    )
    parser.add_argument("--physical-layout", choices=PHYSICAL_LAYOUTS, default="random")
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--samples", type=int, default=23)
    parser.add_argument("--forwards", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument(
        "--oracle-batch", type=int, action="append", default=[], help="also check CPU scores for this batch"
    )
    parser.set_defaults(strict_indices=True, fixture_replicas=1)
    args = parser.parse_args()
    qli = read_qli_config(args.config) if args.config else DEFAULT_QLI_CONFIG
    if not 1 <= args.batch_start <= args.batch_stop <= 80:
        parser.error("Expected 1 <= batch-start <= batch-stop <= 80")
    if not ROWS_PER_REQUEST <= args.context <= qli["max_position_embeddings"]:
        parser.error("Context must include all six MTP5 rows and fit the model position limit")
    if min(args.samples, args.forwards, args.warmup) < 1:
        parser.error("Timing counts must be positive")
    if any(not args.batch_start <= batch <= args.batch_stop for batch in args.oracle_batch):
        parser.error("Oracle batches must be inside the requested batch range")

    import torch_npu

    if not torch.npu.is_available() or torch.npu.device_count() != 1:
        raise RuntimeError("Benchmark expects exactly one visible NPU die")
    torch.npu.set_device(0)
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=False)
    backends = ("native", TILELANG_BACKEND)
    cases = [make_case(args, b) for b in range(args.batch_start, args.batch_stop + 1)]
    report = {
        "model": "GLM-5.4",
        "model_qli_config": qli,
        "native_api": "torch_npu.npu_quant_lightning_indexer",
        "tilelang_backend": TILELANG_BACKEND,
        "q_heads": HEADS,
        "head_dim": WIDTH,
        "topk": TOPK,
        "sparse_mode": 3,
        "torch": str(torch.__version__),
        "torch_npu": str(torch_npu.__version__),
        "timing_method": "acl_graph_event_abba_per_call",
        "forwards_per_graph": args.forwards,
        "warmup_replays": args.warmup,
        "fixture_replicas": args.fixture_replicas,
        "strict_indices": True,
        "cases": cases,
        "parity": {},
        "case_status": {},
        "results": [],
        "comparisons": [],
    }
    checkpoint(args.output, report)
    with (args.output / "timing_samples.jsonl").open("x", encoding="utf-8") as raw, torch.inference_mode():
        for case in cases:
            name = case["case"]
            dispatch = {}
            try:
                parity, rows, samples, comparison = run_case(
                    case,
                    backends,
                    args,
                    True,
                    check_reference=case["batch"] in args.oracle_batch,
                    allow_index_mismatch=True,
                    dispatch=dispatch,
                )
            except AssertionError as error:
                print(f"VALIDATION_FAILED {name} {error}", flush=True)
                report["case_status"][name] = {"status": "validation-failed", "dispatch": dispatch}
                checkpoint(args.output, report)
                continue
            exact = parity["cross_backend_differing_slots"] == 0
            report["parity"][name] = parity
            report["case_status"][name] = {
                "status": "ordered-indices-match" if exact else "ordered-indices-differ",
                "dispatch": dispatch,
            }
            report["results"].extend(
                dict(row, dispatch=dispatch[row["backend"]], ordered_indices_match=exact) for row in rows
            )
            if comparison is not None:
                report["comparisons"].append(
                    dict(comparison, dispatch=dispatch[TILELANG_BACKEND], ordered_indices_match=exact)
                )
            for sample in samples:
                raw.write(json.dumps({"case": name, **sample}) + "\n")
            raw.flush()
            checkpoint(args.output, report)
            print(f"DISPATCH {name} {json.dumps(report['case_status'][name])}", flush=True)


if __name__ == "__main__":
    main()
