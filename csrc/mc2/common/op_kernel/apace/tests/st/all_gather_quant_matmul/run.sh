#!/bin/bash
# run.sh — apace AllGatherQuantMatmul Prefill ST
#
# 用法:
#   bash run.sh                        # 运行 cases.csv 全部 case
#   bash run.sh 1                      # 运行 CSV 第 1 行
#   bash run.sh --cli m k n r          # 命令行模式
#   bash run.sh --skip-build ...       # 跳过编译
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ST_DIR}/build"

SKIP_BUILD=0
CSV_FILE="${SCRIPT_DIR}/cases.csv"

ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build)  SKIP_BUILD=1; shift ;;
        --cli)         CLI_MODE=1; shift ;;
        -h|--help)
            sed -n '1,15p' "$0"; exit 0 ;;
        *)
            ARGS+=("$1"); shift ;;
    esac
done

command -v python3 >/dev/null || { echo "ERROR: python3 not available"; exit 1; }

run_single() {
    local M=$1 K=$2 N=$3 RANK_NUM=$4
    echo ""
    echo "=========================================="
    echo "apace AllGatherQuantMatmul Prefill ST"
    echo "  M=$M K=$K N=$N rankNum=$RANK_NUM"
    echo "=========================================="

    cd "$SCRIPT_DIR"
    rm -rf input output

    echo "[1/4] Generate CPU golden + input data..."
    python3 scripts/gen_data.py "$M" "$K" "$N" "$RANK_NUM"

    if [ "$SKIP_BUILD" -eq 0 ]; then
        echo "[2/4] Build..."
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"
        cmake -S "$ST_DIR" -B "$BUILD_DIR" || { echo "ERROR: cmake failed"; return 1; }
        cmake --build "$BUILD_DIR" --target apace_ag_qmm_st --parallel 4 || { echo "ERROR: build failed"; return 1; }
    else
        [ -x "$BUILD_DIR/all_gather_quant_matmul/apace_ag_qmm_st" ] || { echo "ERROR: --skip-build but binary missing"; return 1; }
        echo "[2/4] Skip build (--skip-build)"
    fi

    echo "[3/4] NPU run ($RANK_NUM ranks)..."
    local EXE_PATH="$BUILD_DIR/all_gather_quant_matmul/apace_ag_qmm_st"
    local EXE_DIR="$(dirname "$EXE_PATH")"
    rm -rf "$EXE_DIR/input" "$EXE_DIR/output"
    cp -r input "$EXE_DIR/input"
    mkdir -p "$EXE_DIR/output"
    for r in $(seq 0 $((RANK_NUM-1))); do
        mkdir -p "$EXE_DIR/output/$r"
    done

    MAX_RETRY=3
    RETRY=0
    while [ ${RETRY} -lt ${MAX_RETRY} ]; do
        fuser -k 8998/tcp 2>/dev/null || true
        sleep 1

        KERNEL_OUT=$("$EXE_PATH" $M $K $N $RANK_NUM 2>&1)
        KERNEL_RC=$?
        echo "${KERNEL_OUT}"

        if [ ${KERNEL_RC} -eq 0 ] && echo "${KERNEL_OUT}" | grep -q "Status: SUCCESS"; then
            echo "  Kernel finished!"
            break
        fi

        if echo "${KERNEL_OUT}" | grep -q "connect peers failed"; then
            RETRY=$((RETRY + 1))
            echo "  [retry ${RETRY}/${MAX_RETRY}] connect conflict..."
            sleep 2
            continue
        fi

        echo "ERROR: Kernel failed"
        return 1
    done

    if [ ${RETRY} -ge ${MAX_RETRY} ]; then
        echo "ERROR: Max retries exceeded"
        return 1
    fi

    cd "$SCRIPT_DIR"
    for r in $(seq 0 $((RANK_NUM-1))); do
        mkdir -p "output/$r"
        cp "$EXE_DIR/output/$r/npu_out.bin" "output/$r/npu_out.bin" 2>/dev/null || true
    done

    echo "[4/4] Verify..."
    python3 scripts/verify_result.py "$M" "$N" "$RANK_NUM" "./output"
    return $?
}

if [ "${CLI_MODE:-0}" -eq 1 ]; then
    M=2048; K=3584; N=4096; RANK_NUM=4
    idx=0
    [ $idx -lt ${#ARGS[@]} ] && { M=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { K=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { N=${ARGS[$idx]}; idx=$((idx+1)); }
    [ $idx -lt ${#ARGS[@]} ] && { RANK_NUM=${ARGS[$idx]}; idx=$((idx+1)); }
    run_single "$M" "$K" "$N" "$RANK_NUM"
    exit $?
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV not found: $CSV_FILE"; exit 1
fi

DATA_LINES=$(grep -v '^#' "$CSV_FILE" | grep -v '^$')
TOTAL_LINES=$(echo "$DATA_LINES" | grep -c .)

if [ ${#ARGS[@]} -eq 0 ]; then
    echo "==== cases.csv ($TOTAL_LINES cases) ===="
    echo "Row  M      K     N     rank"
    local_idx=0
    while IFS= read -r line; do
        local_idx=$((local_idx + 1))
        printf "%-5s %s\n" "$local_idx" "$(echo "$line" | awk -F, '{printf "%-6s %-5s %-5s %s", $1,$2,$3,$4}')"
    done <<< "$DATA_LINES"
    echo ""
    echo "Running all $TOTAL_LINES cases..."
    SELECTED_ROWS=($(seq 1 $TOTAL_LINES))
else
    SELECTED_ROWS=("${ARGS[@]}")
fi

for row in "${SELECTED_ROWS[@]}"; do
    if [ "$row" -lt 1 ] || [ "$row" -gt "$TOTAL_LINES" ]; then
        echo "ERROR: row $row out of range (1-$TOTAL_LINES)"; exit 1
    fi
done

echo "==== Running ${#SELECTED_ROWS[@]} case(s): ${SELECTED_ROWS[*]} ===="

PASS_CNT=0; FAIL_CNT=0
RESULTS=()

for row in "${SELECTED_ROWS[@]}"; do
    SELECTED=$(echo "$DATA_LINES" | sed -n "${row}p")
    M=$(echo "$SELECTED" | awk -F, '{print $1}')
    K=$(echo "$SELECTED" | awk -F, '{print $2}')
    N=$(echo "$SELECTED" | awk -F, '{print $3}')
    RANK_NUM=$(echo "$SELECTED" | awk -F, '{print $4}')

    echo ""
    echo "########## CSV row ${row}/${TOTAL_LINES} ##########"

    if run_single "$M" "$K" "$N" "$RANK_NUM"; then
        PASS_CNT=$((PASS_CNT + 1))
        RESULTS+=("PASS  row$row  M=$M K=$K N=$N rank=$RANK_NUM")
    else
        FAIL_CNT=$((FAIL_CNT + 1))
        RESULTS+=("FAIL  row$row  M=$M K=$K N=$N rank=$RANK_NUM")
    fi
    SKIP_BUILD=1
done

echo ""
echo "=========================================="
echo "Summary: PASS=$PASS_CNT  FAIL=$FAIL_CNT  total=${#SELECTED_ROWS[@]}"
echo "=========================================="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done

[ "$FAIL_CNT" -eq 0 ]
