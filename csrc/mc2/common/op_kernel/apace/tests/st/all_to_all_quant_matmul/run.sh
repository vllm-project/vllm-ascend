#!/bin/bash
# run.sh — apace AllToAllQuantMatmul ST 一键脚本
#
# 流程: 生成数据 -> 编译 -> 多 rank 运行 -> 精度比对
#
# 用法:
#   bash run.sh                        # 无参数: 列出 cases.csv 全部 case，依次运行
#   bash run.sh 1                      # 运行 CSV 第 1 行
#   bash run.sh 1 3 5                  # 运行 CSV 第 1/3/5 行
#   bash run.sh 2-4                    # 运行 CSV 第 2~4 行
#   bash run.sh all                    # 运行 CSV 全部行
#   bash run.sh --csv <file> 1 3       # 指定 csv 文件 + 行号
#   bash run.sh --cli m k n r h        # 命令行模式(绕过 csv)
#   bash run.sh --skip-build ...       # 跳过编译
#   bash run.sh --gen-only ...         # 仅生成 CPU golden
#   bash run.sh --verify-only ...      # 仅精度比对
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ST_DIR}/build"

SKIP_BUILD=0; GEN_ONLY=0; VERIFY_ONLY=0; CLI_MODE=0
CSV_FILE="${SCRIPT_DIR}/cases.csv"

# ---- 解析参数 ----
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build)  SKIP_BUILD=1; shift ;;
        --gen-only)    GEN_ONLY=1; shift ;;
        --verify-only) VERIFY_ONLY=1; shift ;;
        --cli)         CLI_MODE=1; shift ;;
        --csv)         CSV_FILE="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,18p' "$0"; exit 0 ;;
        *)
            ARGS+=("$1"); shift ;;
    esac
done

# ---- 将行号参数展开为列表（支持 "1 3 5"、"2-4"、"all"）----
expand_rows() {
    local input=("$@")
    local result=()
    for item in "${input[@]}"; do
        if [ "$item" = "all" ]; then
            for ((i=1; i<=TOTAL_LINES; i++)); do result+=($i); done
        elif [[ "$item" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            local start=${BASH_REMATCH[1]} end=${BASH_REMATCH[2]}
            for ((i=start; i<=end; i++)); do result+=($i); done
        elif [[ "$item" =~ ^[0-9]+$ ]]; then
            result+=($item)
        fi
    done
    echo "${result[@]}"
}

# ---- 环境检查（提前做，避免多行重复）----
if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1 || true
else
    echo "ERROR: ASCEND_HOME_PATH 未设置"; exit 1
fi
command -v python3 >/dev/null || { echo "ERROR: python3 不可用"; exit 1; }

# ---- 单 case 执行函数 ----
run_single() {
    local M=$1 K=$2 N=$3 RANK_NUM=$4 HEAD_M_SIZE=$5

    local MODE="precision"

    echo ""
    echo "=========================================="
    echo "apace AllToAllQuantMatmul ST"
    echo "  M=$M K=$K N=$N rankNum=$RANK_NUM headMSize=$HEAD_M_SIZE"
    echo "=========================================="

    cd "$SCRIPT_DIR"
    rm -rf input output

    # ---- 1. 生成数据 ----
    echo "[1/4] 生成 CPU golden + 输入数据..."
    python3 scripts/gen_data.py "$M" "$K" "$N" "$RANK_NUM"

    if [ "$GEN_ONLY" -eq 1 ]; then return 0; fi

    # ---- 2. 编译 ----
    if [ "$VERIFY_ONLY" -eq 0 ]; then
        if [ "$SKIP_BUILD" -eq 0 ]; then
            echo "[2/4] 编译..."
            mkdir -p "$BUILD_DIR"
            cmake -S "$ST_DIR" -B "$BUILD_DIR" || { echo "ERROR: cmake 失败"; return 1; }
            cmake --build "$BUILD_DIR" --target apace_a2a_qmm_udma_st --parallel 4 || {
                echo "ERROR: 编译失败。"; return 1; }
        else
            [ -x "$BUILD_DIR/all_to_all_quant_matmul/apace_a2a_qmm_udma_st" ] || { echo "ERROR: --skip-build 但二进制不存在"; return 1; }
            echo "[2/4] 跳过编译 (--skip-build)"
        fi

        # ---- 3. 多 rank 运行 ----
        echo "[3/4] NPU 运行 ($RANK_NUM ranks)..."
        local EXE_PATH="$BUILD_DIR/all_to_all_quant_matmul/apace_a2a_qmm_udma_st"
        local EXE_DIR="$(dirname "$EXE_PATH")"
        rm -rf "$EXE_DIR/input" "$EXE_DIR/output"
        cp -r input "$EXE_DIR/input"
        mkdir -p "$EXE_DIR/output"
        for r in $(seq 0 $((RANK_NUM-1))); do
            mkdir -p "$EXE_DIR/output/$r"
        done
        cd "$EXE_DIR"
        "./apace_a2a_qmm_udma_st" "$M" "$K" "$N" "$RANK_NUM" "$MODE" "$HEAD_M_SIZE"
        # 回拷 npu_out
        cd "$SCRIPT_DIR"
        for r in $(seq 0 $((RANK_NUM-1))); do
            mkdir -p "output/$r"
            cp "$EXE_DIR/output/$r/npu_out.bin" "output/$r/npu_out.bin" 2>/dev/null || true
        done
    fi

    # ---- 4. 精度比对 ----
    echo "[4/4] 精度比对..."
    python3 scripts/verify_result.py "$M" "$N" "$RANK_NUM" "./output"
    return $?
}

# ---- 选择参数来源 ----
if [ "$CLI_MODE" -eq 1 ]; then
    # 命令行模式
    M=2048; K=3584; N=4096; RANK_NUM=2; HEAD_M_SIZE=512
    idx=0
    if [ ${#ARGS[@]} -gt 0 ]; then
        [ $idx -lt ${#ARGS[@]} ] && { M=${ARGS[$idx]}; idx=$((idx+1)); }
        [ $idx -lt ${#ARGS[@]} ] && { K=${ARGS[$idx]}; idx=$((idx+1)); }
        [ $idx -lt ${#ARGS[@]} ] && { N=${ARGS[$idx]}; idx=$((idx+1)); }
        [ $idx -lt ${#ARGS[@]} ] && { RANK_NUM=${ARGS[$idx]}; idx=$((idx+1)); }
        [ $idx -lt ${#ARGS[@]} ] && { HEAD_M_SIZE=${ARGS[$idx]}; idx=$((idx+1)); }
    fi
    run_single "$M" "$K" "$N" "$RANK_NUM" "$HEAD_M_SIZE"
    exit $?
fi

# ---- CSV 模式 ----
if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV 文件不存在: $CSV_FILE"; exit 1
fi

DATA_LINES=$(grep -v '^#' "$CSV_FILE" | grep -v '^$' | tail -n +2)
TOTAL_LINES=$(echo "$DATA_LINES" | grep -c .)

# 无参数: 列出全部 case 并全部运行
if [ ${#ARGS[@]} -eq 0 ]; then
    echo "==== cases.csv (共 $TOTAL_LINES 个 case) ===="
    echo "行号  M      K     N     rank  headMSize"
    local_idx=0
    while IFS= read -r line; do
        local_idx=$((local_idx + 1))
        printf "%-5s %s\n" "$local_idx" "$(echo "$line" | awk -F, '{printf "%-6s %-5s %-5s %-5s %s", $1,$2,$3,$4,$5}')"
    done <<< "$DATA_LINES"
    echo ""
    echo "将依次运行全部 $TOTAL_LINES 个 case..."
    SELECTED_ROWS=($(seq 1 $TOTAL_LINES))
else
    SELECTED_ROWS=($(expand_rows "${ARGS[@]}"))
fi

# 校验行号范围
for row in "${SELECTED_ROWS[@]}"; do
    if [ "$row" -lt 1 ] || [ "$row" -gt "$TOTAL_LINES" ]; then
        echo "ERROR: 行号 $row 超出范围 (1-$TOTAL_LINES)"; exit 1
    fi
done

echo "==== 将运行 ${#SELECTED_ROWS[@]} 个 case: ${SELECTED_ROWS[*]} ===="

# ---- 逐行执行 ----
PASS_CNT=0; FAIL_CNT=0
RESULTS=()

for row in "${SELECTED_ROWS[@]}"; do
    SELECTED=$(echo "$DATA_LINES" | sed -n "${row}p")
    M=$(echo "$SELECTED" | awk -F, '{print $1}')
    K=$(echo "$SELECTED" | awk -F, '{print $2}')
    N=$(echo "$SELECTED" | awk -F, '{print $3}')
    RANK_NUM=$(echo "$SELECTED" | awk -F, '{print $4}')
    HEAD_M_SIZE=$(echo "$SELECTED" | awk -F, '{print $5}')

    echo ""
    echo "########## CSV 第 ${row}/${TOTAL_LINES} 行 ##########"

    if run_single "$M" "$K" "$N" "$RANK_NUM" "$HEAD_M_SIZE"; then
        PASS_CNT=$((PASS_CNT + 1))
        RESULTS+=("PASS  行$row  M=$M K=$K N=$N rank=$RANK_NUM headM=$HEAD_M_SIZE")
    else
        FAIL_CNT=$((FAIL_CNT + 1))
        RESULTS+=("FAIL  行$row  M=$M K=$K N=$N rank=$RANK_NUM headM=$HEAD_M_SIZE")
    fi
    # 首次编译后，后续 case 跳过编译
    SKIP_BUILD=1
done

# ---- 汇总 ----
echo ""
echo "=========================================="
echo "汇总: PASS=$PASS_CNT  FAIL=$FAIL_CNT  共${#SELECTED_ROWS[@]}个case"
echo "=========================================="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done

[ "$FAIL_CNT" -eq 0 ]
