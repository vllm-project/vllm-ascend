#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

MODE=${MODE:?set MODE to static_partition or address_table}
RUN_LABEL=${RUN_LABEL:?set RUN_LABEL for this fresh server run}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the local model directory}
WORKLOAD_DIR=${WORKLOAD_DIR:?set WORKLOAD_DIR to generated workload JSONL files}
RESULT_ROOT=${RESULT_ROOT:-${SCRIPT_DIR}/results/jenga-precision}
PORT=${PORT:-18085}
SEED=${SEED:-20260901}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-8}
CASES=${CASES:-"512 2048 8192 32760"}

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_DIR="${RESULT_ROOT}/${RUN_LABEL}"
mkdir -p "${RUN_DIR}"
server_pid=""

cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" || true
    wait "${server_pid}" || true
  fi
}
trap cleanup EXIT

VLLM_ASCEND_ENABLE_TYPED_KV_CACHE=1 \
VLLM_ASCEND_TYPED_KV_CACHE_MODE="${MODE}" \
VLLM_BATCH_INVARIANT=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  vllm serve "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --served-model-name qwen35-27b \
    --generation-config vllm \
    --seed "${SEED}" \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --skip-mm-profiling \
    --no-enable-prefix-caching \
    --enforce-eager \
    >"${RUN_DIR}/server.log" 2>&1 &
server_pid=$!

for _ in $(seq 1 240); do
  if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    wait "${server_pid}"
  fi
  sleep 2
done
curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null

python - "${WORKLOAD_DIR}" "${RUN_DIR}/precision.json" \
  "${PORT}" "${SEED}" "${OUTPUT_TOKENS}" "${MODE}" "${RUN_LABEL}" "${CASES}" <<'PY'
import hashlib
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

workload_dir = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
port = int(sys.argv[3])
seed = int(sys.argv[4])
output_tokens = int(sys.argv[5])
mode = sys.argv[6]
run_label = sys.argv[7]
selected_cases = set(sys.argv[8].split())

cases = (
    ("512", "longbench-512.jsonl"),
    ("2048", "longbench-2048.jsonl"),
    ("8192", "longbench-8192.jsonl"),
    ("32760", "longbench-32768.jsonl"),
)
results = []
for label, filename in cases:
    if label not in selected_cases:
        continue
    row = json.loads((workload_dir / filename).read_text(encoding="utf-8").splitlines()[0])
    payload = json.dumps(
        {
            "model": "qwen35-27b",
            "prompt": row["prompt"],
            "max_tokens": output_tokens,
            "temperature": 0,
            "seed": seed,
            "ignore_eos": True,
            "return_token_ids": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"precision request {label} failed with HTTP {error.code}: {detail}"
        ) from error
    elapsed = time.perf_counter() - started
    choice = body["choices"][0]
    prompt_token_ids = choice["prompt_token_ids"]
    prompt_token_ids_hash = hashlib.sha256(
        ",".join(str(token_id) for token_id in prompt_token_ids).encode("ascii")
    ).hexdigest()
    result = {
        "case": label,
        "request_id": row["request_id"],
        "prompt_tokens_expected": row["prompt_tokens"],
        "prompt_tokens_reported": body["usage"]["prompt_tokens"],
        "completion_tokens": body["usage"]["completion_tokens"],
        "elapsed_seconds": elapsed,
        "finish_reason": choice["finish_reason"],
        "text": choice["text"],
        "prompt_token_ids_sha256": prompt_token_ids_hash,
        "token_ids": choice["token_ids"],
    }
    results.append(result)
    output_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "run_label": run_label,
                "seed": seed,
                "output_tokens": output_tokens,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case": label,
                "prompt_tokens": result["prompt_tokens_reported"],
                "completion_tokens": result["completion_tokens"],
                "elapsed_seconds": round(elapsed, 3),
            }
        ),
        flush=True,
    )
PY

cleanup
trap - EXIT
