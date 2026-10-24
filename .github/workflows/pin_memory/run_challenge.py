# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Phase 3 (challenge): after the max-seq phase, run R repeated full-protocol
rounds per experiment at the max seq already found, hunting for the best pin
ratio on the CI A5 (950) card. Per-round results go to results_challenge.jsonl.

Launch one experiment at a time (argv filter, same granularity as
run_maxseq.py: group, model-short, or group_model-short); review between
experiments. argv[2] = rounds (default 5).

Reuses run_maxseq.full_run for the arms (same script/args/protocol), but
renames each round's arm logs to per-round records so every round re-runs
fresh (full_run reuses arm logs it finds at the canonical names).
"""

import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_maxseq as rm

R_DEFAULT = 5
RESULTS = f"{rm.OUT}/results_challenge.jsonl"


def _read_jsonl(path):
    """Yield parsed JSON objects from path, skipping malformed lines."""
    with open(path) as f:
        for line in f:
            with contextlib.suppress(Exception):
                yield json.loads(line)


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else None
    nrounds = int(sys.argv[2]) if len(sys.argv) > 2 else R_DEFAULT
    rm.log(f"=== challenge start filter={key} rounds={nrounds} (loadavg {rm.loadavg()}) ===")
    rm.wait_leftovers()

    done = set()
    if os.path.exists(RESULTS):
        for s in _read_jsonl(RESULTS):
            done.add(s["experiment"])

    seqs = {}
    results_file = f"{rm.OUT}/results_maxseq.jsonl"
    if os.path.exists(results_file):
        for s in _read_jsonl(results_file):
            if s.get("status") == "ok":
                exp = s["experiment"]
                seqs[exp] = max(seqs.get(exp, 0), s["max_seq"])

    plan = [
        ("delock", f"{rm.DELOCK}/zero3_offload_bench.py", rm.M3, rm.base_env, 3600),
        ("delock", f"{rm.DELOCK}/zero3_offload_bench_qwen35.py", rm.M35, rm.base_env, 3600),
        ("act", f"{rm.ACT}/bench.py", rm.M3, rm.base_env, 3600),
        ("act", f"{rm.ACT}/bench_qwen35.py", rm.M35, rm.base_env, 3600),
        ("sao", f"{rm.SAO}/bench.py", rm.M3, rm.sao_env, 5400),
        ("sao", f"{rm.SAO}/bench.py", rm.M35, rm.sao_env, 5400),
    ]
    for group, script, model, env, timeout in plan:
        short = model.split("/")[-1]
        exp_name = f"{group}_maxseq_{short}"
        if key and key not in (group, short, f"{group}_{short}"):
            continue
        if exp_name in done:
            rm.log(f"  {exp_name}: already in results_challenge.jsonl, skip")
            continue
        seq = seqs.get(exp_name)
        if seq is None:
            rm.log(f"  {exp_name}: no max seq found, skip")
            continue
        # move maxseq-phase arm logs aside so round 1 re-runs instead of reusing
        for label in ("unpinned", "pinned-unregistered", "pinned-registered", "pageable", "pinned"):
            src = f"{rm.OUT}/{exp_name}_seq{seq}_{label}.log"
            if os.path.exists(src):
                os.rename(src, f"{rm.OUT}/{exp_name}_seq{seq}_{label}.maxseq.log")
        rm.log(f"=== challenge {exp_name} seq{seq} {nrounds} rounds ===")
        rounds = []
        for r in range(1, nrounds + 1):
            rm.log(f"  round {r}/{nrounds} (loadavg {rm.loadavg()})")
            summary = rm.full_run(group, model, seq, timeout)
            for label in ("unpinned", "pinned-unregistered", "pinned-registered", "pageable", "pinned"):
                src = f"{rm.OUT}/{exp_name}_seq{seq}_{label}.log"
                if os.path.exists(src):
                    os.rename(src, f"{rm.OUT}/challenge_{exp_name}_r{r}_{label}.log")
            summary["round"] = r
            rounds.append(summary)
            key_ratio = summary.get("speedup") or summary.get("pin_speedup")
            rm.log(f"  round {r}: status={summary.get('status')} ratio={key_ratio}")
        best = None
        for s in rounds:
            ratio = s.get("speedup") or s.get("pin_speedup")
            if ratio is None:
                continue
            if best is None or ratio > (best.get("speedup") or best.get("pin_speedup")):
                best = s
        with open(RESULTS, "a") as f:
            f.write(
                json.dumps(
                    {
                        "experiment": exp_name,
                        "seq": seq,
                        "rounds": rounds,
                        "best": best,
                        "best_round": best and best["round"],
                    }
                )
                + "\n"
            )
        rm.log(f"  BEST: {json.dumps(best)}")
    rm.log("=== CHALLENGE DONE ===")


if __name__ == "__main__":
    main()
