#!/usr/bin/env bash
# run_benchmark.sh — End-to-end benchmark runner for vllm-ascend on Ascend 910C
#
# Usage:
#   bash run_benchmark.sh [OPTIONS]
#
# Options (all override config file values):
#   --model-path PATH            Model local path or HuggingFace/ModelScope ID
#   --model-name NAME            Short name used in report and API served_model_name
#   --config FILE                YAML config file (default: none)
#   --tp INT                     Tensor parallel size (default: 1)
#   --pp INT                     Pipeline parallel size (default: 1)
#   --max-model-len INT          Max context length (default: 8192)
#   --dtype STR                  float16|bfloat16|auto (default: float16)
#   --quantization STR           w8a8|w4a16|null (default: none)
#   --port INT                   Server port (default: 8000)
#   --parallel-levels STR        Space-separated concurrency list (default: "1 4 8 16 32 64")
#   --requests-per-level STR     Space-separated request counts (default: "10 40 80 160 320 640")
#   --input-tokens INT           Input token length (default: 1024)
#   --output-tokens INT          Output token length (default: 512)
#   --warmup-requests INT        Warm-up request count (default: 20)
#   --output-dir DIR             Where to save reports (default: ./benchmark_output/<timestamp>)
#   --npu-devices STR            ASCEND_RT_VISIBLE_DEVICES value (default: auto-detect from tp)
#   --skip-warmup                Skip warm-up phase
#   --no-aclgraph                Disable ACLGraph (VLLM_ASCEND_ENABLE_ACLGRAPH=0)
#   --no-nz                      Disable NZ format (VLLM_ASCEND_ENABLE_NZ=0)
#   --dry-run                    Print commands without executing them

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
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
ENABLE_ACLGRAPH=1
ENABLE_NZ=0
DRY_RUN=0

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)      MODEL_PATH="$2";        shift 2 ;;
        --model-name)      MODEL_NAME="$2";        shift 2 ;;
        --config)          CONFIG_FILE="$2";       shift 2 ;;
        --tp)              TP="$2";                shift 2 ;;
        --pp)              PP="$2";                shift 2 ;;
        --max-model-len)   MAX_MODEL_LEN="$2";     shift 2 ;;
        --dtype)           DTYPE="$2";             shift 2 ;;
        --quantization)    QUANTIZATION="$2";      shift 2 ;;
        --port)            PORT="$2";              shift 2 ;;
        --parallel-levels) PARALLEL_LEVELS="$2";   shift 2 ;;
        --requests-per-level) REQUESTS_PER_LEVEL="$2"; shift 2 ;;
        --input-tokens)    INPUT_TOKENS="$2";      shift 2 ;;
        --output-tokens)   OUTPUT_TOKENS="$2";     shift 2 ;;
        --warmup-requests) WARMUP_REQUESTS="$2";   shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2";        shift 2 ;;
        --npu-devices)     NPU_DEVICES="$2";       shift 2 ;;
        --skip-warmup)     SKIP_WARMUP=1;          shift ;;
        --no-aclgraph)     ENABLE_ACLGRAPH=0;      shift ;;
        --no-nz)           ENABLE_NZ=0;            shift ;;
        --dry-run)         DRY_RUN=1;              shift ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ── Apply YAML config if provided ─────────────────────────────────────────────
if [[ -n "${CONFIG_FILE}" ]]; then
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "ERROR: config file not found: ${CONFIG_FILE}" >&2
        exit 1
    fi
    _yaml_get() {
        python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG_FILE}'))
keys = '$1'.split('.')
val = cfg
for k in keys:
    if isinstance(val, dict) and k in val:
        val = val[k]
    else:
        val = None
        break
if val is not None:
    print(val)
" 2>/dev/null || true
    }
    [[ -z "${MODEL_PATH}" ]]   && MODEL_PATH=$(_yaml_get model.path)
    [[ -z "${MODEL_NAME}" ]]   && MODEL_NAME=$(_yaml_get model.name)
    _tp=$(_yaml_get server.tensor_parallel_size);  [[ -n "$_tp" ]] && TP=$_tp
    _pp=$(_yaml_get server.pipeline_parallel_size); [[ -n "$_pp" ]] && PP=$_pp
    _port=$(_yaml_get server.port);                [[ -n "$_port" ]] && PORT=$_port
    _mml=$(_yaml_get model.max_model_len);         [[ -n "$_mml" ]] && MAX_MODEL_LEN=$_mml
    _dtype=$(_yaml_get model.dtype);               [[ -n "$_dtype" ]] && DTYPE=$_dtype
    _quant=$(_yaml_get model.quantization);        [[ -n "$_quant" && "$_quant" != "None" && "$_quant" != "null" ]] && QUANTIZATION=$_quant
    _pl=$(_yaml_get benchmark.parallel_levels);    [[ -n "$_pl" ]] && PARALLEL_LEVELS=$_pl
    _rpl=$(_yaml_get benchmark.requests_per_level);[[ -n "$_rpl" ]] && REQUESTS_PER_LEVEL=$_rpl
    _it=$(_yaml_get workload.input_tokens);        [[ -n "$_it" ]] && INPUT_TOKENS=$_it
    _ot=$(_yaml_get workload.output_tokens);       [[ -n "$_ot" ]] && OUTPUT_TOKENS=$_ot
    _wr=$(_yaml_get benchmark.warmup_requests);    [[ -n "$_wr" ]] && WARMUP_REQUESTS=$_wr
    _npu=$(_yaml_get server.env.ASCEND_RT_VISIBLE_DEVICES); [[ -n "$_npu" && -z "${NPU_DEVICES}" ]] && NPU_DEVICES=$_npu
fi

# ── Validation ─────────────────────────────────────────────────────────────────
if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: --model-path is required (or set model.path in config)" >&2
    exit 1
fi
if [[ -z "${MODEL_NAME}" ]]; then
    MODEL_NAME=$(basename "${MODEL_PATH}")
    echo "INFO: --model-name not set, using '${MODEL_NAME}'"
fi

# Auto-detect NPU devices
if [[ -z "${NPU_DEVICES}" ]]; then
    NPU_DEVICES=$(seq -s "," 0 $((TP * PP - 1)))
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="./benchmark_output/${TIMESTAMP}_${MODEL_NAME}"
fi
mkdir -p "${OUTPUT_DIR}"

EVALSCOPE_RESULTS_DIR="${OUTPUT_DIR}/evalscope_results"
mkdir -p "${EVALSCOPE_RESULTS_DIR}"

SERVER_LOG="${OUTPUT_DIR}/server.log"
NPU_INFO_FILE="${OUTPUT_DIR}/npu_info.txt"

# ── Logging helper ─────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
run() {
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

# ── Phase 1: Environment check ─────────────────────────────────────────────────
log "=== Phase 1: Environment Validation ==="

npu-smi info > "${NPU_INFO_FILE}" 2>&1 || log "WARN: npu-smi not available"
log "NPU info saved to ${NPU_INFO_FILE}"

if ! python3 -c "import vllm_ascend" 2>/dev/null; then
    echo "ERROR: vllm-ascend not installed. Run: pip install vllm-ascend" >&2
    exit 1
fi

if ! command -v evalscope &>/dev/null && ! python3 -m evalscope --version &>/dev/null 2>&1; then
    echo "ERROR: evalscope not found. Run: pip install evalscope[perf] -U" >&2
    exit 1
fi

if [[ -n "${CONFIG_FILE}" ]]; then
    log "Validating config file: ${CONFIG_FILE}"
    run python3 "${SKILL_DIR}/scripts/validate_configs.py" "${CONFIG_FILE}"
fi

# Get version info
VLLM_ASCEND_VERSION=$(python3 -c "import vllm_ascend; print(getattr(vllm_ascend, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
VLLM_ASCEND_COMMIT=$(python3 -c "
import importlib.util, subprocess, pathlib
spec = importlib.util.find_spec('vllm_ascend')
if spec:
    loc = str(pathlib.Path(spec.origin).parent)
    result = subprocess.run(['git', '-C', loc, 'rev-parse', '--short', 'HEAD'],
        capture_output=True, text=True)
    print(result.stdout.strip() if result.returncode == 0 else 'unknown')
else:
    print('unknown')
" 2>/dev/null || echo "unknown")

log "vllm-ascend: ${VLLM_ASCEND_VERSION} (${VLLM_ASCEND_COMMIT})"
log "NPU devices: ${NPU_DEVICES}"
log "TP=${TP}  PP=${PP}  port=${PORT}"

# ── Phase 2: Launch vLLM-Ascend server ────────────────────────────────────────
log "=== Phase 2: Launching vLLM-Ascend Server ==="

# Build server command
QUANT_FLAGS=""
if [[ -n "${QUANTIZATION}" ]]; then
    QUANT_FLAGS="--quantization ${QUANTIZATION}"
fi

SERVER_CMD="ASCEND_RT_VISIBLE_DEVICES=${NPU_DEVICES} \
VLLM_ASCEND_ENABLE_ACLGRAPH=${ENABLE_ACLGRAPH} \
VLLM_ASCEND_ENABLE_NZ=${ENABLE_NZ} \
python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_PATH} \
  --served-model-name ${MODEL_NAME} \
  --tensor-parallel-size ${TP} \
  --pipeline-parallel-size ${PP} \
  --max-model-len ${MAX_MODEL_LEN} \
  --dtype ${DTYPE} \
  --port ${PORT} \
  --trust-remote-code \
  ${QUANT_FLAGS}"

# Save server command for report
echo "${SERVER_CMD}" > "${OUTPUT_DIR}/server_cmd.txt"

if [[ "${DRY_RUN}" -eq 0 ]]; then
    export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}"
    export VLLM_ASCEND_ENABLE_ACLGRAPH="${ENABLE_ACLGRAPH}"
    export VLLM_ASCEND_ENABLE_NZ="${ENABLE_NZ}"

    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --tensor-parallel-size "${TP}" \
        --pipeline-parallel-size "${PP}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --dtype "${DTYPE}" \
        --port "${PORT}" \
        --trust-remote-code \
        ${QUANT_FLAGS:+${QUANT_FLAGS}} \
        > "${SERVER_LOG}" 2>&1 &

    SERVER_PID=$!
    log "Server started with PID ${SERVER_PID}, log: ${SERVER_LOG}"

    # Wait for server to be ready
    log "Waiting for server to be ready (max 300s)..."
    READY=0
    for i in $(seq 1 60); do
        sleep 5
        if curl -s "http://127.0.0.1:${PORT}/health" | grep -q "{}"; then
            READY=1
            break
        fi
        # Also accept 200 with any body
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo "000")
        if [[ "${HTTP_CODE}" == "200" ]]; then
            READY=1
            break
        fi
        log "  Waiting... (${i}/60, last HTTP ${HTTP_CODE})"
    done

    if [[ "${READY}" -eq 0 ]]; then
        echo "ERROR: Server failed to start within 300s. Check ${SERVER_LOG}" >&2
        kill "${SERVER_PID}" 2>/dev/null || true
        exit 1
    fi
    log "Server is ready."
else
    log "[DRY-RUN] Would start: ${SERVER_CMD}"
    SERVER_PID=0
fi

# ── Phase 3: Warm-up ──────────────────────────────────────────────────────────
EVALSCOPE_CMD_BASE="evalscope perf \
  --model ${MODEL_NAME} \
  --url http://127.0.0.1:${PORT}/v1/chat/completions \
  --api openai \
  --stream \
  --dataset random \
  --max-tokens ${OUTPUT_TOKENS} \
  --min-tokens ${OUTPUT_TOKENS} \
  --min-prompt-length ${INPUT_TOKENS} \
  --max-prompt-length ${INPUT_TOKENS} \
  --tokenizer-path ${MODEL_PATH}"

if [[ "${SKIP_WARMUP}" -eq 0 ]]; then
    log "=== Phase 3: Warm-up (${WARMUP_REQUESTS} requests) ==="
    WARMUP_CMD="${EVALSCOPE_CMD_BASE} --parallel 1 --number ${WARMUP_REQUESTS} --name warmup"
    run bash -c "${WARMUP_CMD}" || log "WARN: Warm-up returned non-zero exit; continuing"
    log "Warm-up complete."
else
    log "=== Phase 3: Warm-up skipped ==="
fi

# ── Phase 4: Stress test ──────────────────────────────────────────────────────
log "=== Phase 4: Stress Test ==="
log "Concurrency levels: ${PARALLEL_LEVELS}"
log "Requests per level: ${REQUESTS_PER_LEVEL}"

EVALSCOPE_CMD="${EVALSCOPE_CMD_BASE} \
  --parallel ${PARALLEL_LEVELS} \
  --number ${REQUESTS_PER_LEVEL} \
  --name benchmark \
  --work-dir ${EVALSCOPE_RESULTS_DIR}"

echo "${EVALSCOPE_CMD}" > "${OUTPUT_DIR}/evalscope_cmd.txt"

run bash -c "${EVALSCOPE_CMD}"
log "Stress test complete. Results in ${EVALSCOPE_RESULTS_DIR}"

# ── Shutdown server ───────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" -eq 0 && "${SERVER_PID}" -ne 0 ]]; then
    log "Stopping server (PID ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    log "Server stopped."
fi

# ── Phase 5: Report generation ────────────────────────────────────────────────
log "=== Phase 5: Generating Report ==="

REPORT_CMD="python3 ${SKILL_DIR}/scripts/generate_report.py \
  --results-dir ${EVALSCOPE_RESULTS_DIR} \
  --output-dir ${OUTPUT_DIR} \
  --model-name ${MODEL_NAME} \
  --model-path ${MODEL_PATH} \
  --npu-info ${NPU_INFO_FILE} \
  --vllm-commit ${VLLM_ASCEND_COMMIT} \
  --server-cmd \"$(cat ${OUTPUT_DIR}/server_cmd.txt 2>/dev/null || echo '')\" \
  --evalscope-cmd \"$(cat ${OUTPUT_DIR}/evalscope_cmd.txt 2>/dev/null || echo '')\""

if [[ -n "${CONFIG_FILE}" ]]; then
    REPORT_CMD="${REPORT_CMD} --config ${CONFIG_FILE}"
fi

run bash -c "${REPORT_CMD}"

log ""
log "==================================================================="
log "  Benchmark complete!"
log "  Markdown report : ${OUTPUT_DIR}/benchmark_report.md"
log "  CSV results     : ${OUTPUT_DIR}/benchmark_results.csv"
log "  Server log      : ${SERVER_LOG}"
log "  NPU info        : ${NPU_INFO_FILE}"
log "==================================================================="
