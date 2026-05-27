#!/usr/bin/env bash
# bench_apc_matrix.sh
#
# Multi-config matrix benchmark for PR8829 ALL vs ALIGN mamba cache modes.
#
# Matrix:
#   prefix_len ∈ {2048, 8192, 16384}
#   K (prompts_per_prefix) ∈ {2, 4, 6, 8}
#   mode ∈ {align, all}
#   = 3 * 4 * 2 = 24 server restarts total
#
# Fixed:
#   model:         /home/weight/Qwen3.5-27B-w8a8-org (TP=2)
#   max_num_batched_tokens: 8192    (chunk step = 8 * block_size, no waste)
#   block_size:    1024
#   max_num_seqs:  32
#   concurrency:   16
#   suffix_len:    1024
#   total_prompts: 48 (num_prefixes * K)
#   max_tokens:    1  (focus on TTFT)
#
# Robustness:
#   - Resumable: skip group if results/Gxx_<mode>/summary.json exists
#   - Per-group log: server.log, client.log
#   - FAILED marker on failure, continue to next group
#   - Health check + timeout on server boot
#   - Cleanup trap on Ctrl-C
#
# Usage:
#   ./bench_apc_matrix.sh                 # run full matrix
#   ./bench_apc_matrix.sh --dry-run       # print plan only
#   ./bench_apc_matrix.sh --only G4_all   # run single group
#   ./bench_apc_matrix.sh --force         # ignore existing summary.json
#   ./bench_apc_matrix.sh --port 8826     # custom port

set -uo pipefail
# NOTE: -e is intentionally OFF — we want to continue on group failure.

# --------------------------------------------------------------------
# Defaults (matches your live serve command)
# --------------------------------------------------------------------
MODEL="${MODEL:-/home/weight/Qwen3.5-27B-w8a8-org}"
SERVED_NAME="${SERVED_NAME:-qwen3.5}"
HOST="${HOST:-0.0.0.0}"
# Client-side address used by readiness probe + vllm bench serve.
# Must bypass http(s)_proxy or curl/httpx will round-trip through the proxy
# and fail to reach the local vllm server.
CLIENT_HOST="${CLIENT_HOST:-127.0.0.1}"
export no_proxy="${no_proxy:-},127.0.0.1,localhost,${CLIENT_HOST}"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost,${CLIENT_HOST}"
PORT="${PORT:-8826}"
TP="${TP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-133000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# Client knobs
CONCURRENCY="${CONCURRENCY:-16}"
SUFFIX_LEN="${SUFFIX_LEN:-1024}"
TOTAL_PROMPTS="${TOTAL_PROMPTS:-48}"
MAX_TOKENS="${MAX_TOKENS:-1}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
SEED="${SEED:-1024}"
WARMUP_PROMPTS="${WARMUP_PROMPTS:-3}"

# Server boot timeout (sec)
WAIT_READY_MAX="${WAIT_READY_MAX:-300}"
# Per-group client timeout (sec)
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-600}"
# Idle wait between groups (NPU release)
COOLDOWN_SEC="${COOLDOWN_SEC:-15}"

# Matrix axes (override via env)
#
# Two ways to define (prefix_len, suffix_len) pairs:
#   1. PAIRS env (preferred for non-uniform suffixes), space-separated "prefix:suffix":
#        PAIRS="6144:2048 12288:4096 14336:3000"
#   2. PREFIX_LENS env (legacy): all groups share SUFFIX_LEN.
#        PREFIX_LENS="2048 8192 16384"
#
# Default = new sub-chunk pair matrix targeting ALL-mode advantage region.
PAIRS_DEFAULT="6144:2048 12288:4096 14336:3000"
PAIRS=(${PAIRS:-$PAIRS_DEFAULT})
# If PAIRS contains no ':' tokens, fall back to legacy PREFIX_LENS mode.
PREFIX_LENS=(${PREFIX_LENS:-})
K_VALUES=(${K_VALUES:-2 4 6 8})
MODES=(${MODES:-align all})

# Control flags
DRY_RUN=0
ONLY_GROUP=""
FORCE=0

# --------------------------------------------------------------------
# Parse args
# --------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN=1; shift ;;
        --force)         FORCE=1; shift ;;
        --only)          ONLY_GROUP="$2"; shift 2 ;;
        --port)          PORT="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --concurrency)   CONCURRENCY="$2"; shift 2 ;;
        --suffix-len)    SUFFIX_LEN="$2"; shift 2 ;;
        --total-prompts) TOTAL_PROMPTS="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_ROOT="${SCRIPT_DIR}/results/matrix_$(date +%Y%m%d_%H%M%S)"
LATEST_LINK="${SCRIPT_DIR}/results/matrix_latest"
mkdir -p "$RESULTS_ROOT"
rm -f "$LATEST_LINK"
ln -s "$RESULTS_ROOT" "$LATEST_LINK"

MASTER_LOG="${RESULTS_ROOT}/master.log"
echo "[INFO] results dir: $RESULTS_ROOT" | tee -a "$MASTER_LOG"
echo "[INFO] latest link: $LATEST_LINK"  | tee -a "$MASTER_LOG"

# --------------------------------------------------------------------
# Cleanup trap
# --------------------------------------------------------------------
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[CLEANUP] killing server pid=$SERVER_PID" | tee -a "$MASTER_LOG"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 5
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

# Kill any stale vllm serve on the port (only this user's processes).
kill_stale_serve() {
    local pids
    pids=$(pgrep -u "$USER" -f "vllm serve.*--port[= ]${PORT}" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        log "killing stale vllm serve pids: $pids"
        for pid in $pids; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        sleep 5
        for pid in $pids; do
            kill -KILL "$pid" 2>/dev/null || true
        done
    fi
}

wait_server_ready() {
    local elapsed=0
    while (( elapsed < WAIT_READY_MAX )); do
        if curl --noproxy '*' -fsS "http://${CLIENT_HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
            log "  server ready after ${elapsed}s"
            return 0
        fi
        # Detect early death.
        if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "  ERROR: server pid=$SERVER_PID died during boot"
            return 1
        fi
        sleep 5
        elapsed=$(( elapsed + 5 ))
        if (( elapsed % 30 == 0 )); then
            log "  waiting server ready... ${elapsed}/${WAIT_READY_MAX}s"
        fi
    done
    log "  ERROR: server did not become ready within ${WAIT_READY_MAX}s"
    return 1
}

start_server() {
    local mode="$1"
    local log_file="$2"

    kill_stale_serve

    log "  starting server mode=${mode} port=${PORT}"
    # Use --additional-config to set mamba_cache_mode (PR8829 convention).
    # If your build still uses the older --mamba-cache-mode CLI flag, swap below.
    nohup vllm serve "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --served-model-name "$SERVED_NAME" \
        --quantization ascend \
        --gdn-prefill-backend triton \
        --enable-prefix-caching \
        --enforce-eager \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --block-size "$BLOCK_SIZE" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --trust-remote-code \
        --async-scheduling \
        --seed "$SEED" \
        --additional-config "{\"mamba_cache_mode\":\"${mode}\"}" \
        > "$log_file" 2>&1 &

    SERVER_PID=$!
    log "  server pid=$SERVER_PID"
    wait_server_ready
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "  stopping server pid=$SERVER_PID"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        local elapsed=0
        while (( elapsed < 30 )) && kill -0 "$SERVER_PID" 2>/dev/null; do
            sleep 2
            elapsed=$(( elapsed + 2 ))
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log "  force kill server pid=$SERVER_PID"
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
    fi
    SERVER_PID=""
    sleep "$COOLDOWN_SEC"
}

run_client() {
    local group_dir="$1"
    local prefix_len="$2"
    local k="$3"
    local num_prefixes="$4"
    local suffix_len="$5"

    local client_log="${group_dir}/client.log"
    local summary_json="${group_dir}/summary.json"

    log "  client: prefix_len=$prefix_len suffix_len=$suffix_len K=$k num_prefixes=$num_prefixes total=$TOTAL_PROMPTS"

    timeout "$CLIENT_TIMEOUT" vllm bench serve \
        --backend openai \
        --host "$CLIENT_HOST" \
        --port "$PORT" \
        --model "$SERVED_NAME" \
        --tokenizer "$MODEL" \
        --trust-remote-code \
        --endpoint /v1/completions \
        --dataset-name prefix_repetition \
        --prefix-repetition-prefix-len "$prefix_len" \
        --prefix-repetition-suffix-len "$suffix_len" \
        --prefix-repetition-num-prefixes "$num_prefixes" \
        --prefix-repetition-output-len "$MAX_TOKENS" \
        --num-prompts "$TOTAL_PROMPTS" \
        --max-concurrency "$CONCURRENCY" \
        --request-rate "$REQUEST_RATE" \
        --seed "$SEED" \
        --save-result \
        --result-dir "$group_dir" \
        --result-filename summary.json \
        > "$client_log" 2>&1

    local rc=$?
    if [[ $rc -ne 0 ]]; then
        log "  client FAILED rc=$rc (see $client_log)"
        echo "rc=$rc" > "${group_dir}/FAILED"
        return 1
    fi
    if [[ ! -f "$summary_json" ]]; then
        log "  client returned 0 but no summary.json produced"
        echo "missing summary.json" > "${group_dir}/FAILED"
        return 1
    fi
    return 0
}

# --------------------------------------------------------------------
# Build group list
# --------------------------------------------------------------------
# Parallel arrays (portable: works on bash 3.2+)
GROUP_IDS=()
GROUP_PREFIX_LENS=()
GROUP_SUFFIX_LENS=()
GROUP_KS=()
GROUP_NUM_PREFIXES_ARR=()

# lookup helper: get_idx <gid> -> echoes 0-based index, or empty
get_idx() {
    local target="$1" i
    for i in "${!GROUP_IDS[@]}"; do
        if [[ "${GROUP_IDS[$i]}" == "$target" ]]; then
            echo "$i"
            return
        fi
    done
}

# Build (prefix, suffix) list from PAIRS or fall back to legacy PREFIX_LENS.
PAIR_PLEN=()
PAIR_SLEN=()
if [[ ${#PAIRS[@]} -gt 0 && "${PAIRS[0]}" == *:* ]]; then
    for tok in "${PAIRS[@]}"; do
        plen="${tok%%:*}"
        slen="${tok##*:}"
        if [[ -z "$plen" || -z "$slen" || "$plen" == "$slen" && "$tok" != *:* ]]; then
            log "ERROR: bad PAIR token: $tok (expected prefix:suffix)"
            exit 2
        fi
        PAIR_PLEN+=("$plen")
        PAIR_SLEN+=("$slen")
    done
elif [[ ${#PREFIX_LENS[@]} -gt 0 ]]; then
    for plen in "${PREFIX_LENS[@]}"; do
        PAIR_PLEN+=("$plen")
        PAIR_SLEN+=("$SUFFIX_LEN")
    done
else
    log "ERROR: no PAIRS or PREFIX_LENS configured"
    exit 2
fi

idx=0
for pi in "${!PAIR_PLEN[@]}"; do
    plen="${PAIR_PLEN[$pi]}"
    slen="${PAIR_SLEN[$pi]}"
    for k in "${K_VALUES[@]}"; do
        idx=$(( idx + 1 ))
        if (( TOTAL_PROMPTS % k != 0 )); then
            log "WARN: skip K=$k because TOTAL_PROMPTS=$TOTAL_PROMPTS is not divisible by K"
            continue
        fi
        num_prefixes=$(( TOTAL_PROMPTS / k ))
        if (( num_prefixes < 1 )); then
            log "WARN: skip K=$k because TOTAL_PROMPTS=$TOTAL_PROMPTS / K < 1"
            continue
        fi
        gid="G${idx}"
        GROUP_IDS+=("$gid")
        GROUP_PREFIX_LENS+=("$plen")
        GROUP_SUFFIX_LENS+=("$slen")
        GROUP_KS+=("$k")
        GROUP_NUM_PREFIXES_ARR+=("$num_prefixes")
    done
done

# --------------------------------------------------------------------
# Print plan
# --------------------------------------------------------------------
log "==============================================================="
log "Matrix plan: ${#GROUP_IDS[@]} groups × ${#MODES[@]} modes = $((${#GROUP_IDS[@]} * ${#MODES[@]})) runs"
log "==============================================================="
printf "%-4s %-12s %-12s %-3s %-12s %-6s\n" "id" "prefix_len" "suffix_len" "K" "num_prefixes" "total" | tee -a "$MASTER_LOG"
for i in "${!GROUP_IDS[@]}"; do
    printf "%-4s %-12s %-12s %-3s %-12s %-6s\n" \
        "${GROUP_IDS[$i]}" \
        "${GROUP_PREFIX_LENS[$i]}" \
        "${GROUP_SUFFIX_LENS[$i]}" \
        "${GROUP_KS[$i]}" \
        "${GROUP_NUM_PREFIXES_ARR[$i]}" \
        "$TOTAL_PROMPTS" | tee -a "$MASTER_LOG"
done
log "==============================================================="

if [[ $DRY_RUN -eq 1 ]]; then
    log "DRY-RUN: exiting without running."
    exit 0
fi

# --------------------------------------------------------------------
# Execute matrix: outer loop mode (1 server boot per mode-group combo)
#
# We boot server fresh for EVERY (gid, mode) combination because:
# 1. mode switch requires restart anyway
# 2. prefix cache leaks across groups would skew the cold-start measurement
# --------------------------------------------------------------------
total_runs=$(( ${#GROUP_IDS[@]} * ${#MODES[@]} ))
current=0
skipped=0
failed=0
ok=0

for mode in "${MODES[@]}"; do
    for i in "${!GROUP_IDS[@]}"; do
        gid="${GROUP_IDS[$i]}"
        current=$(( current + 1 ))
        run_label="${gid}_${mode}"
        group_dir="${RESULTS_ROOT}/${run_label}"

        # --only filter
        if [[ -n "$ONLY_GROUP" && "$run_label" != "$ONLY_GROUP" ]]; then
            continue
        fi

        log ""
        log "================ [$current/$total_runs] $run_label ================"

        # Resume: skip if already done
        if [[ -f "${group_dir}/summary.json" && $FORCE -eq 0 ]]; then
            log "  SKIP: summary.json already exists (use --force to re-run)"
            skipped=$(( skipped + 1 ))
            continue
        fi

        mkdir -p "$group_dir"
        rm -f "${group_dir}/FAILED"

        plen=${GROUP_PREFIX_LENS[$i]}
        slen=${GROUP_SUFFIX_LENS[$i]}
        k=${GROUP_KS[$i]}
        nprefix=${GROUP_NUM_PREFIXES_ARR[$i]}

        # Record config for this run
        cat > "${group_dir}/config.json" <<EOF
{
  "group_id": "$gid",
  "mode": "$mode",
  "prefix_len": $plen,
  "suffix_len": $slen,
  "K": $k,
  "num_prefixes": $nprefix,
  "total_prompts": $TOTAL_PROMPTS,
  "concurrency": $CONCURRENCY,
  "max_num_batched_tokens": $MAX_NUM_BATCHED_TOKENS,
  "block_size": $BLOCK_SIZE,
  "max_tokens": $MAX_TOKENS,
  "model": "$MODEL"
}
EOF

        # Start server
        server_log="${group_dir}/server.log"
        if ! start_server "$mode" "$server_log"; then
            log "  SERVER BOOT FAILED for $run_label"
            echo "server boot failed" > "${group_dir}/FAILED"
            failed=$(( failed + 1 ))
            stop_server
            continue
        fi

        # Run client
        if run_client "$group_dir" "$plen" "$k" "$nprefix" "$slen"; then
            log "  OK: $run_label"
            ok=$(( ok + 1 ))
        else
            log "  FAIL: $run_label (continuing to next)"
            failed=$(( failed + 1 ))
        fi

        # Always stop server before next group (prevent cache leak across groups)
        stop_server
    done
done

# --------------------------------------------------------------------
# Final summary
# --------------------------------------------------------------------
log ""
log "==============================================================="
log "MATRIX DONE: ok=$ok  failed=$failed  skipped=$skipped  total=$total_runs"
log "Results: $RESULTS_ROOT"
log "==============================================================="

# Try to aggregate (best effort)
if [[ -x "${SCRIPT_DIR}/aggregate_matrix.py" ]]; then
    log "Aggregating..."
    python3 "${SCRIPT_DIR}/aggregate_matrix.py" "$RESULTS_ROOT" \
        > "${RESULTS_ROOT}/matrix_summary.md" 2>>"$MASTER_LOG" \
        && log "Summary written: ${RESULTS_ROOT}/matrix_summary.md" \
        || log "WARN: aggregate failed (results still in per-group dirs)"
fi
