#!/usr/bin/env bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# This file is a part of the vllm-ascend project.
#
# Performance sweep for DSpark speculative decoding, Model Runner V2 vs V1.
#
# For every (runner, num_speculative_tokens, batch_size) combination this
# script starts an API server, runs `vllm bench serve` against it, records peak
# HBM from npu-smi, and writes one JSON result file per combination. Throughput,
# TTFT, TPOT and end-to-end latency all come from `vllm bench serve`, which is
# the only source here that measures them per request.
#
# The model runner is chosen with VLLM_USE_V2_MODEL_RUNNER, read when the config
# is built, so each combination needs its own server process. Note that leaving
# the variable unset selects V2 (the current default), never V1 - both arms are
# therefore always set explicitly.
#
# Usage:
#   MODEL=<target-model> DRAFT_MODEL=<draft-model> \
#   TP=1 DP=1 bash ./dspark_v1_v2_perf.sh
#
set -euo pipefail

# The vLLM server of the configuration currently being measured. A global EXIT trap (not a RETURN trap, which
# is not inherited by called functions and does not run when the script is interrupted) guarantees that the
# background server is stopped on every exit path: normal end, failure under `set -e`, SIGINT or SIGTERM.
server_pid=""
stop_server() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=""
    fi
}
trap stop_server EXIT

MODEL="${MODEL:?set MODEL to the target checkpoint}"
DRAFT_MODEL="${DRAFT_MODEL:?set DRAFT_MODEL to the DSpark draft checkpoint}"
OUT_DIR="${OUT_DIR:-./perf_results}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
DP="${DP:-1}"
QUANTIZATION="${QUANTIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
# issue #13745 asks for 1/3/5; 7 is the Kimi-K3 recipe default, kept for parity
# with the shipped configuration.
SPEC_TOKENS="${SPEC_TOKENS:-1 3 5 7}"
BATCH_SIZES="${BATCH_SIZES:-1 4 8 16}"
RUNNERS="${RUNNERS:-0 1}"
# Draft graph mode. V1 always drafts eager (use_cuda_graph is hardcoded False in
# dspark_proposer.py), so set this to "true" for an apples-to-apples comparison
# and run a second V2-only sweep with "false" to measure the full-graph gain.
ENFORCE_EAGER_DRAFT="${ENFORCE_EAGER_DRAFT:-true}"
DATASET_NAME="${DATASET_NAME:-random}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-3600}"

mkdir -p "$OUT_DIR"

# Peak HBM across all visible devices, sampled while the benchmark runs.
# torch's allocator counters only cover the process that queries them, which
# misses the worker processes, so read the device instead.
sample_hbm() {
    local out_file="$1"
    : > "$out_file"
    while true; do
        npu-smi info 2>/dev/null \
            | grep -oE '[0-9]+ +/ +[0-9]+' \
            | awk '{print $1}' >> "$out_file" || true
        sleep 2
    done
}

peak_hbm_mb() {
    local sample_file="$1"
    [[ -s "$sample_file" ]] && sort -n "$sample_file" | tail -1 || echo 0
}

wait_for_server() {
    local deadline=$((SECONDS + SERVER_TIMEOUT))
    while (( SECONDS < deadline )); do
        if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    echo "server did not become healthy within ${SERVER_TIMEOUT}s" >&2
    return 1
}

run_one() {
    local runner="$1" spec_tokens="$2" batch_size="$3"
    local tag="runner${runner}_spec${spec_tokens}_bs${batch_size}_eager${ENFORCE_EAGER_DRAFT}"
    local server_log="${OUT_DIR}/${tag}.server.log"
    local hbm_file="${OUT_DIR}/${tag}.hbm.samples"
    local result_file="${OUT_DIR}/${tag}.json"

    rm -f "$result_file"
    echo "=== ${tag} ==="

    local spec_config
    spec_config=$(printf '{"method":"dspark","model":"%s","num_speculative_tokens":%s,"enforce_eager":%s}' \
        "$DRAFT_MODEL" "$spec_tokens" "$ENFORCE_EAGER_DRAFT")

    local -a server_cmd=(
        vllm serve "$MODEL"
        --port "$PORT"
        --trust-remote-code
        --tensor-parallel-size "$TP"
        --data-parallel-size "$DP"
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-seqs "$batch_size"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --no-enable-prefix-caching
        --seed 42
        --compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\"}"
        --speculative-config "$spec_config"
    )
    [[ -n "$QUANTIZATION" ]] && server_cmd+=(--quantization "$QUANTIZATION")

    VLLM_USE_V2_MODEL_RUNNER="$runner" "${server_cmd[@]}" > "$server_log" 2>&1 &
    server_pid=$!

    if ! wait_for_server; then
        echo "  !! server failed to start; see ${server_log}" >&2
        stop_server
        return 1
    fi

    sample_hbm "$hbm_file" &
    local sampler_pid=$!

    local bench_rc=0
    vllm bench serve \
        --backend vllm \
        --model "$MODEL" \
        --port "$PORT" \
        --dataset-name "$DATASET_NAME" \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency "$batch_size" \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-filename "$result_file" \
        2>&1 | tee "${OUT_DIR}/${tag}.bench.log" || bench_rc=${PIPESTATUS[0]}

    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true

    if (( bench_rc != 0 )); then
        echo "  !! benchmark failed with rc=${bench_rc}; see ${OUT_DIR}/${tag}.bench.log" >&2
        stop_server
        return "$bench_rc"
    fi
    if [[ ! -s "$result_file" ]]; then
        echo "  !! benchmark exited successfully but wrote no result: ${result_file}" >&2
        stop_server
        return 1
    fi

    local peak
    peak=$(peak_hbm_mb "$hbm_file")
    echo "  peak HBM: ${peak} MB"
    # Fold the HBM reading and the run identity into the benchmark JSON so a
    # single file fully describes the data point.
    local metadata_rc=0
    python3 - "$result_file" "$peak" "$runner" "$spec_tokens" "$batch_size" "$ENFORCE_EAGER_DRAFT" <<'PY' || metadata_rc=$?
import json, sys
path, peak, runner, spec_tokens, batch_size, eager = sys.argv[1:7]
with open(path) as handle:
    data = json.load(handle)
data["ascend_run_meta"] = {
    "model_runner": "v2" if runner == "1" else "v1",
    "vllm_use_v2_model_runner": runner,
    "num_speculative_tokens": int(spec_tokens),
    "max_num_seqs": int(batch_size),
    "enforce_eager_draft": eager == "true",
    "peak_hbm_mb": int(peak),
}
with open(path, "w") as handle:
    json.dump(data, handle, indent=2)
PY
    if (( metadata_rc != 0 )); then
        echo "  !! failed to add metadata to ${result_file}" >&2
        rm -f "$result_file"
        stop_server
        return "$metadata_rc"
    fi

    stop_server
    sleep 10  # let HBM drain before the next server starts
}

overall_rc=0
for runner in $RUNNERS; do
    for spec_tokens in $SPEC_TOKENS; do
        for batch_size in $BATCH_SIZES; do
            if ! run_one "$runner" "$spec_tokens" "$batch_size"; then
                echo "  (continuing after failure)" >&2
                overall_rc=1
            fi
        done
    done
done

echo
echo "Results in ${OUT_DIR}. Summarise with:"
echo "  python3 dspark_v1_v2_perf_summary.py ${OUT_DIR} --out perf_report.md"
exit "$overall_rc"
