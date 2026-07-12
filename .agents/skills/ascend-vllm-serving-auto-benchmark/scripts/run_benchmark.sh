#!/usr/bin/env bash
# run_benchmark.sh — End-to-end benchmark runner for vllm-ascend on Ascend 910C

set -euo pipefail

MODEL_PATH=""
MODEL_NAME=""
CONFIG_FILE=""
TP=1
PP=1
MAX_MODEL_LEN=8192
DTYPE="float16"
QUANTIZATION=""
PORT=8000
PARALLEL_LEVELS="1 4 8 16 32 64"
REQUESTS_PER_LEVEL="10 40 80 160 320 640"
INPUT_TOKENS=1024
OUTPUT_TOKENS=512
WARMUP_REQUESTS=20
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=""
NPU_DEVICES=""
SKIP_WARMUP=0
ACLGRAPH_OVERRIDE=""
NZ_OVERRIDE=""
TRUST_REMOTE_CODE=1
STREAM_MODE=1
DRY_RUN=0
SERVER_PID=""
EVALSCOPE_CMD=(evalscope perf)

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ENV_ASSIGNMENTS=()

log() { echo "[$(date '+%H:%M:%S')] $*"; }

shell_escape() {
    printf "%q" "$1"
}

join_command() {
    local out=""
    local token
    for token in "$@"; do
        out+="$(_escape "${token}") "
    done
    printf "%s" "${out% }"
}

_escape() {
    shell_escape "$1"
}

get_env_value() {
    local key="$1"
    local prefix="${key}="
    local entry
    for entry in "${SERVER_ENV_ASSIGNMENTS[@]}"; do
        if [[ "${entry}" == "${prefix}"* ]]; then
            printf "%s" "${entry#*=}"
            return 0
        fi
    done
    return 1
}

upsert_env_value() {
    local key="$1"
    local value="$2"
    local prefix="${key}="
    local i
    for ((i = 0; i < ${#SERVER_ENV_ASSIGNMENTS[@]}; i++)); do
        if [[ "${SERVER_ENV_ASSIGNMENTS[$i]}" == "${prefix}"* ]]; then
            SERVER_ENV_ASSIGNMENTS[$i]="${prefix}${value}"
            return
        fi
    done
    SERVER_ENV_ASSIGNMENTS+=("${prefix}${value}")
}

default_npu_devices() {
    local count="$1"
    local devices=()
    local idx

    for ((idx = 0; idx < count; idx++)); do
        devices+=("${idx}")
    done

    local joined=""
    for idx in "${!devices[@]}"; do
        if [[ "${idx}" -gt 0 ]]; then
            joined+=","
        fi
        joined+="${devices[$idx]}"
    done

    printf "%s" "${joined}"
}

load_yaml_value() {
    local key_path="$1"
    python3 - "$CONFIG_FILE" "$key_path" <<'PY'
import sys

import yaml

config_path, key_path = sys.argv[1], sys.argv[2]
with open(config_path, encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh) or {}

value = cfg
for key in key_path.split("."):
    if isinstance(value, dict) and key in value:
        value = value[key]
    else:
        value = None
        break

if value is None:
    sys.exit(0)
if isinstance(value, bool):
    print("true" if value else "false")
else:
    print(value)
PY
}

load_yaml_env_assignments() {
    python3 - "$CONFIG_FILE" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh) or {}

env = cfg.get("server", {}).get("env", {}) or {}
for key, value in env.items():
    if value is None:
        continue
    print(f"{key}={value}")
PY
}

cleanup_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "Stopping server (PID ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        log "Server stopped."
    fi
}

detect_evalscope_outputs_flag() {
    local help_output=""
    if [[ "${DRY_RUN}" -eq 1 ]] && ! command -v evalscope >/dev/null 2>&1; then
        printf "%s" "--outputs-dir"
        return 0
    fi

    if command -v evalscope >/dev/null 2>&1; then
        help_output=$(evalscope perf --help 2>/dev/null || true)
    else
        help_output=$(python3 -m evalscope perf --help 2>/dev/null || true)
    fi

    if grep -q -- "--outputs-dir" <<<"${help_output}" || grep -q -- "outputs_dir" <<<"${help_output}"; then
        printf "%s" "--outputs-dir"
        return 0
    fi

    printf "%s" ""
}

run_evalscope_with_output_dir() {
    local result_dir="$1"
    shift

    local output_flag="$1"
    shift

    if [[ -n "${output_flag}" ]]; then
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            printf '[DRY-RUN] %s\n' "$(join_command "$@" "${output_flag}" "${result_dir}")"
        else
            "$@" "${output_flag}" "${result_dir}"
        fi
        return
    fi

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '[DRY-RUN] cd %s && %s\n' \
            "$(shell_escape "${result_dir}")" \
            "$(join_command "$@")"
        return
    fi

    (
        cd "${result_dir}"
        "$@"
    )
}

trap cleanup_server EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --pp) PP="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --parallel-levels) PARALLEL_LEVELS="$2"; shift 2 ;;
        --requests-per-level) REQUESTS_PER_LEVEL="$2"; shift 2 ;;
        --input-tokens) INPUT_TOKENS="$2"; shift 2 ;;
        --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
        --warmup-requests) WARMUP_REQUESTS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --npu-devices) NPU_DEVICES="$2"; shift 2 ;;
        --skip-warmup) SKIP_WARMUP=1; shift ;;
        --no-aclgraph) ACLGRAPH_OVERRIDE="0"; shift ;;
        --no-nz) NZ_OVERRIDE="0"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -n "${CONFIG_FILE}" ]]; then
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "ERROR: config file not found: ${CONFIG_FILE}" >&2
        exit 1
    fi

    if ! python3 -c "import yaml" >/dev/null 2>&1; then
        echo "ERROR: PyYAML is required to read --config files. Run: pip install pyyaml" >&2
        exit 1
    fi

    while IFS= read -r assignment; do
        [[ -n "${assignment}" ]] || continue
        SERVER_ENV_ASSIGNMENTS+=("${assignment}")
    done < <(load_yaml_env_assignments)

    [[ -z "${MODEL_PATH}" ]] && MODEL_PATH="$(load_yaml_value model.path || true)"
    [[ -z "${MODEL_NAME}" ]] && MODEL_NAME="$(load_yaml_value model.name || true)"

    _tp="$(load_yaml_value server.tensor_parallel_size || true)"
    [[ -n "${_tp}" ]] && TP="${_tp}"
    _pp="$(load_yaml_value server.pipeline_parallel_size || true)"
    [[ -n "${_pp}" ]] && PP="${_pp}"
    _port="$(load_yaml_value server.port || true)"
    [[ -n "${_port}" ]] && PORT="${_port}"
    _mml="$(load_yaml_value model.max_model_len || true)"
    [[ -n "${_mml}" ]] && MAX_MODEL_LEN="${_mml}"
    _dtype="$(load_yaml_value model.dtype || true)"
    [[ -n "${_dtype}" ]] && DTYPE="${_dtype}"
    _quant="$(load_yaml_value model.quantization || true)"
    if [[ -n "${_quant}" && "${_quant}" != "None" && "${_quant}" != "null" ]]; then
        QUANTIZATION="${_quant}"
    fi
    _pl="$(load_yaml_value benchmark.parallel_levels || true)"
    [[ -n "${_pl}" ]] && PARALLEL_LEVELS="${_pl}"
    _rpl="$(load_yaml_value benchmark.requests_per_level || true)"
    [[ -n "${_rpl}" ]] && REQUESTS_PER_LEVEL="${_rpl}"
    _it="$(load_yaml_value workload.input_tokens || true)"
    [[ -n "${_it}" ]] && INPUT_TOKENS="${_it}"
    _ot="$(load_yaml_value workload.output_tokens || true)"
    [[ -n "${_ot}" ]] && OUTPUT_TOKENS="${_ot}"
    _wr="$(load_yaml_value benchmark.warmup_requests || true)"
    [[ -n "${_wr}" ]] && WARMUP_REQUESTS="${_wr}"
    _stream="$(load_yaml_value benchmark.stream || true)"
    [[ "${_stream}" == "false" ]] && STREAM_MODE=0
    _trust_remote_code="$(load_yaml_value model.trust_remote_code || true)"
    [[ "${_trust_remote_code}" == "false" ]] && TRUST_REMOTE_CODE=0

    if [[ -z "${NPU_DEVICES}" ]]; then
        _npu="$(load_yaml_value server.env.ASCEND_RT_VISIBLE_DEVICES || true)"
        [[ -n "${_npu}" ]] && NPU_DEVICES="${_npu}"
    fi
fi

if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: --model-path is required (or set model.path in config)" >&2
    exit 1
fi

if [[ -z "${MODEL_NAME}" ]]; then
    MODEL_NAME="$(basename "${MODEL_PATH}")"
    log "INFO: --model-name not set, using '${MODEL_NAME}'"
fi

if [[ -z "${NPU_DEVICES}" ]]; then
    NPU_DEVICES="$(default_npu_devices "$((TP * PP))")"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="./benchmark_output/${TIMESTAMP}_${MODEL_NAME}"
fi

mkdir -p "${OUTPUT_DIR}"
EVALSCOPE_RESULTS_DIR="${OUTPUT_DIR}/evalscope_results"
WARMUP_RESULTS_DIR="${OUTPUT_DIR}/evalscope_warmup"
mkdir -p "${EVALSCOPE_RESULTS_DIR}" "${WARMUP_RESULTS_DIR}"

SERVER_LOG="${OUTPUT_DIR}/server.log"
NPU_INFO_FILE="${OUTPUT_DIR}/npu_info.txt"

upsert_env_value "ASCEND_RT_VISIBLE_DEVICES" "${NPU_DEVICES}"
if [[ -n "${ACLGRAPH_OVERRIDE}" ]]; then
    upsert_env_value "VLLM_ASCEND_ENABLE_ACLGRAPH" "${ACLGRAPH_OVERRIDE}"
elif ! get_env_value "VLLM_ASCEND_ENABLE_ACLGRAPH" >/dev/null 2>&1; then
    upsert_env_value "VLLM_ASCEND_ENABLE_ACLGRAPH" "1"
fi

if [[ -n "${NZ_OVERRIDE}" ]]; then
    upsert_env_value "VLLM_ASCEND_ENABLE_NZ" "${NZ_OVERRIDE}"
elif ! get_env_value "VLLM_ASCEND_ENABLE_NZ" >/dev/null 2>&1; then
    upsert_env_value "VLLM_ASCEND_ENABLE_NZ" "0"
fi

log "=== Phase 1: Environment Validation ==="

npu-smi info > "${NPU_INFO_FILE}" 2>&1 || log "WARN: npu-smi not available"
log "NPU info saved to ${NPU_INFO_FILE}"

if ! python3 -c "import vllm_ascend" 2>/dev/null; then
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "WARN: vllm-ascend not installed; continuing because --dry-run was requested"
    else
        echo "ERROR: vllm-ascend not installed. Run: pip install vllm-ascend" >&2
        exit 1
    fi
fi

if ! command -v evalscope >/dev/null 2>&1 && ! python3 -m evalscope --version >/dev/null 2>&1; then
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        log "WARN: evalscope not installed; continuing because --dry-run was requested"
    else
        echo "ERROR: evalscope not found. Run: pip install evalscope[perf] -U" >&2
        exit 1
    fi
fi

if ! command -v evalscope >/dev/null 2>&1; then
    EVALSCOPE_CMD=(python3 -m evalscope perf)
fi

if [[ -n "${CONFIG_FILE}" ]]; then
    log "Validating config file: ${CONFIG_FILE}"
    python3 "${SKILL_DIR}/scripts/validate_configs.py" "${CONFIG_FILE}"
fi

VLLM_ASCEND_VERSION="$(python3 -c "import vllm_ascend; print(getattr(vllm_ascend, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")"
VLLM_ASCEND_COMMIT="$(python3 -c "
import importlib.util, pathlib, subprocess
spec = importlib.util.find_spec('vllm_ascend')
if not spec:
    print('unknown')
else:
    loc = pathlib.Path(spec.origin).parent
    result = subprocess.run(
        ['git', '-C', str(loc), 'rev-parse', '--short', 'HEAD'],
        capture_output=True, text=True,
    )
    print(result.stdout.strip() if result.returncode == 0 else 'unknown')
" 2>/dev/null || echo "unknown")"

log "vllm-ascend: ${VLLM_ASCEND_VERSION} (${VLLM_ASCEND_COMMIT})"
log "NPU devices: ${NPU_DEVICES}"
log "TP=${TP}  PP=${PP}  port=${PORT}"

SERVER_ARGS=(
    python3 -m vllm.entrypoints.openai.api_server
    --model "${MODEL_PATH}"
    --served-model-name "${MODEL_NAME}"
    --tensor-parallel-size "${TP}"
    --pipeline-parallel-size "${PP}"
    --max-model-len "${MAX_MODEL_LEN}"
    --dtype "${DTYPE}"
    --port "${PORT}"
)

if [[ -n "${QUANTIZATION}" ]]; then
    SERVER_ARGS+=(--quantization "${QUANTIZATION}")
fi

if [[ "${TRUST_REMOTE_CODE}" -eq 1 ]]; then
    SERVER_ARGS+=(--trust-remote-code)
fi

SERVER_CMD_PARTS=(env)
for assignment in "${SERVER_ENV_ASSIGNMENTS[@]}"; do
    SERVER_CMD_PARTS+=("${assignment}")
done
for arg in "${SERVER_ARGS[@]}"; do
    SERVER_CMD_PARTS+=("${arg}")
done
printf '%s\n' "$(join_command "${SERVER_CMD_PARTS[@]}")" > "${OUTPUT_DIR}/server_cmd.txt"

log "=== Phase 2: Launching vLLM-Ascend Server ==="
if [[ "${DRY_RUN}" -eq 0 ]]; then
    env "${SERVER_ENV_ASSIGNMENTS[@]}" "${SERVER_ARGS[@]}" > "${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    log "Server started with PID ${SERVER_PID}, log: ${SERVER_LOG}"

    log "Waiting for server to be ready (max 300s)..."
    READY=0
    for i in $(seq 1 60); do
        sleep 5
        HTTP_CODE="$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo "000")"
        if [[ "${HTTP_CODE}" == "200" ]]; then
            READY=1
            break
        fi
        log "  Waiting... (${i}/60, last HTTP ${HTTP_CODE})"
    done

    if [[ "${READY}" -eq 0 ]]; then
        echo "ERROR: Server failed to start within 300s. Check ${SERVER_LOG}" >&2
        exit 1
    fi
    log "Server is ready."
else
    log "[DRY-RUN] Would start: $(join_command "${SERVER_CMD_PARTS[@]}")"
fi

OUTPUT_FLAG="$(detect_evalscope_outputs_flag)"
EVALSCOPE_BASE_ARGS=(
    "${EVALSCOPE_CMD[@]}"
    --model "${MODEL_NAME}"
    --url "http://127.0.0.1:${PORT}/v1/chat/completions"
    --api openai
    --dataset random
    --max-tokens "${OUTPUT_TOKENS}"
    --min-tokens "${OUTPUT_TOKENS}"
    --min-prompt-length "${INPUT_TOKENS}"
    --max-prompt-length "${INPUT_TOKENS}"
    --tokenizer-path "${MODEL_PATH}"
)

if [[ "${STREAM_MODE}" -eq 1 ]]; then
    EVALSCOPE_BASE_ARGS+=(--stream)
fi

if [[ "${SKIP_WARMUP}" -eq 0 ]]; then
    log "=== Phase 3: Warm-up (${WARMUP_REQUESTS} requests) ==="
    WARMUP_ARGS=("${EVALSCOPE_BASE_ARGS[@]}" --parallel 1 --number "${WARMUP_REQUESTS}" --name warmup)
    run_evalscope_with_output_dir "${WARMUP_RESULTS_DIR}" "${OUTPUT_FLAG}" "${WARMUP_ARGS[@]}" || \
        log "WARN: Warm-up returned non-zero exit; continuing"
    log "Warm-up complete."
else
    log "=== Phase 3: Warm-up skipped ==="
fi

log "=== Phase 4: Stress Test ==="
log "Concurrency levels: ${PARALLEL_LEVELS}"
log "Requests per level: ${REQUESTS_PER_LEVEL}"

BENCHMARK_ARGS=(
    "${EVALSCOPE_BASE_ARGS[@]}"
    --parallel ${PARALLEL_LEVELS}
    --number ${REQUESTS_PER_LEVEL}
    --name benchmark
)

if [[ -n "${OUTPUT_FLAG}" ]]; then
    EVALSCOPE_CMD_PARTS=("${BENCHMARK_ARGS[@]}" "${OUTPUT_FLAG}" "${EVALSCOPE_RESULTS_DIR}")
    printf '%s\n' "$(join_command "${EVALSCOPE_CMD_PARTS[@]}")" > "${OUTPUT_DIR}/evalscope_cmd.txt"
else
    printf 'cd %s && %s\n' \
        "$(shell_escape "${EVALSCOPE_RESULTS_DIR}")" \
        "$(join_command "${BENCHMARK_ARGS[@]}")" > "${OUTPUT_DIR}/evalscope_cmd.txt"
fi

run_evalscope_with_output_dir "${EVALSCOPE_RESULTS_DIR}" "${OUTPUT_FLAG}" "${BENCHMARK_ARGS[@]}"
log "Stress test complete. Results in ${EVALSCOPE_RESULTS_DIR}"

cleanup_server
SERVER_PID=""

log "=== Phase 5: Generating Report ==="
REPORT_ARGS=(
    python3 "${SKILL_DIR}/scripts/generate_report.py"
    --results-dir "${EVALSCOPE_RESULTS_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --model-name "${MODEL_NAME}"
    --model-path "${MODEL_PATH}"
    --npu-info "${NPU_INFO_FILE}"
    --vllm-commit "${VLLM_ASCEND_COMMIT}"
    --server-cmd "$(cat "${OUTPUT_DIR}/server_cmd.txt" 2>/dev/null || true)"
    --evalscope-cmd "$(cat "${OUTPUT_DIR}/evalscope_cmd.txt" 2>/dev/null || true)"
)

if [[ -n "${CONFIG_FILE}" ]]; then
    REPORT_ARGS+=(--config "${CONFIG_FILE}")
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
    log "[DRY-RUN] Would run: $(join_command "${REPORT_ARGS[@]}")"
else
    "${REPORT_ARGS[@]}"
fi

log ""
log "==================================================================="
log "  Benchmark complete!"
log "  Markdown report : ${OUTPUT_DIR}/benchmark_report.md"
log "  CSV results     : ${OUTPUT_DIR}/benchmark_results.csv"
log "  Server log      : ${SERVER_LOG}"
log "  NPU info        : ${NPU_INFO_FILE}"
log "==================================================================="
