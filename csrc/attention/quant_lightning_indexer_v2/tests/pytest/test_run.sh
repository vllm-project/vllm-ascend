#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
DEFAULT_PT_PATH="$SCRIPT_DIR/pt_path"
PT_SAVE_SCRIPT="$SCRIPT_DIR/batch/quant_lightning_indexer_v2_pt_save.py"
LIST_PT_SCRIPT="$SCRIPT_DIR/batch/list_pt_from_excel.py"
BATCH_TEST_SCRIPT="$SCRIPT_DIR/test_quant_lightning_indexer_v2_batch.py"
SINGLE_TEST_SCRIPT="$SCRIPT_DIR/test_quant_lightning_indexer_v2_single.py"

show_help() {
    cat <<EOF
用法: $0 <command> [选项]

命令:
  single       执行 paramset 中的单用例，可保存本次实际输入 PT 和结果表
  batch        有 -E 时生成 PT 后执行；无 -E 时直接执行已有 PT
  batch_exec   按 Excel 中的 Testcase_Name 筛选已有 PT，仅运行 NPU 和 compare
  help         显示帮助

通用选项:
  -M, --run-mode MODE       eager|graph，默认 eager
  -O, --output FILE         结果 Excel 路径；single 默认 single_result.xlsx，batch 默认 result.xlsx

batch/batch_exec 选项:
  -C, --cases NAMES         按给定顺序执行 case 名，逗号分隔，可省略 .pt
  -I, --indexes INDEXES     按自然排序后的 1-based 序号执行，如 3,1,5-7
  -E, --excel FILE          Excel 路径；batch 不传时跳过 PT 生成
  -S, --sheet NAME          Sheet 名，默认 Sheet1
  -P, --pt-path DIR         PT 生成和读取目录，默认 $DEFAULT_PT_PATH

single 选项:
      --save-pt DIR         single 保存本次实际输入和 CPU golden 的目录

示例:
  $0 single --save-pt ./single_pt -O ./result/single.xlsx
  $0 batch -P ./pt_path
  $0 batch -E ./excel/test_cases.xlsx -P ./pt_path -O ./result/batch.xlsx
  $0 batch -P ./pt_path -I 3,1,5-7
  $0 batch_exec -E ./excel/test_cases.xlsx -P ./pt_path -M graph
EOF
}

require_value() {
    if [ -z "$2" ]; then
        echo "错误: $1 缺少参数值" >&2
        exit 2
    fi
}

validate_run_mode() {
    if [ "$RUN_MODE" != "eager" ] && [ "$RUN_MODE" != "graph" ]; then
        echo "错误: run mode 仅支持 eager/graph，当前值: $RUN_MODE" >&2
        exit 2
    fi
}

run_batch_pytest() {
    local explicit_files="$1"
    QLIV2_TESTCASE_DIR="$PT_PATH" \
    QLIV2_PT_FILE_LIST="$explicit_files" \
    QLIV2_CASE_NAMES="$CASE_NAMES" \
    QLIV2_CASE_INDEXES="$CASE_INDEXES" \
    QLIV2_RESULT_PATH="$RESULT_PATH" \
    QLIV2_RUN_MODE="$RUN_MODE" \
        python3 -m pytest -rA -s "$BATCH_TEST_SCRIPT" -v -m ci \
        -W ignore::UserWarning -W ignore::DeprecationWarning
}

run_single() {
    echo "===== QLI_V2 single: mode=$RUN_MODE result=$RESULT_PATH ====="
    QLIV2_SINGLE_SAVE_PT_DIR="$SAVE_PT_DIR" \
    QLIV2_SINGLE_RESULT_PATH="$RESULT_PATH" \
    QLIV2_RUN_MODE="$RUN_MODE" \
        python3 -m pytest -rA -s "$SINGLE_TEST_SCRIPT" -v -m ci \
        -W ignore::UserWarning -W ignore::DeprecationWarning
}

run_batch() {
    if [ -n "$EXCEL_PATH" ]; then
        if [ ! -f "$EXCEL_PATH" ]; then
            echo "错误: Excel 文件不存在: $EXCEL_PATH" >&2
            exit 1
        fi
        echo "===== 生成 PT: excel=$EXCEL_PATH sheet=$EXCEL_SHEET output=$PT_PATH ====="
        python3 "$PT_SAVE_SCRIPT" "$EXCEL_PATH" "$PT_PATH" --sheet "$EXCEL_SHEET" || exit 1
    elif [ ! -d "$PT_PATH" ]; then
        echo "错误: PT 目录不存在: $PT_PATH" >&2
        exit 1
    fi
    echo "===== 执行 PT: input=$PT_PATH mode=$RUN_MODE result=$RESULT_PATH ====="
    run_batch_pytest ""
}

run_batch_from_excel() {
    if [ -z "$EXCEL_PATH" ]; then
        echo "错误: batch_exec 必须指定 -E/--excel" >&2
        exit 2
    fi
    if [ ! -f "$EXCEL_PATH" ]; then
        echo "错误: Excel 文件不存在: $EXCEL_PATH" >&2
        exit 1
    fi
    if [ ! -d "$PT_PATH" ]; then
        echo "错误: PT 目录不存在: $PT_PATH" >&2
        exit 1
    fi
    local file_list
    file_list=$(python3 "$LIST_PT_SCRIPT" "$EXCEL_PATH" "$PT_PATH" --sheet "$EXCEL_SHEET") || exit 1
    echo "===== Excel 筛选后仅执行 NPU + compare: mode=$RUN_MODE result=$RESULT_PATH ====="
    run_batch_pytest "$file_list"
}

if [ $# -lt 1 ]; then
    show_help
    exit 2
fi

COMMAND="$1"
shift

EXCEL_PATH=""
EXCEL_SHEET="Sheet1"
PT_PATH="$DEFAULT_PT_PATH"
RUN_MODE="eager"
RESULT_PATH=""
CASE_NAMES=""
CASE_INDEXES=""
SAVE_PT_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        -E|--excel)
            require_value "$1" "$2"
            EXCEL_PATH="$2"
            shift 2
            ;;
        -S|--sheet)
            require_value "$1" "$2"
            EXCEL_SHEET="$2"
            shift 2
            ;;
        -P|--pt-path)
            require_value "$1" "$2"
            PT_PATH="$2"
            shift 2
            ;;
        -M|--run-mode)
            require_value "$1" "$2"
            RUN_MODE="$2"
            shift 2
            ;;
        -O|--output)
            require_value "$1" "$2"
            RESULT_PATH="$2"
            shift 2
            ;;
        -C|--cases)
            require_value "$1" "$2"
            CASE_NAMES="$2"
            shift 2
            ;;
        -I|--indexes)
            require_value "$1" "$2"
            CASE_INDEXES="$2"
            shift 2
            ;;
        --save-pt)
            require_value "$1" "$2"
            SAVE_PT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "错误: 未知选项 $1" >&2
            show_help
            exit 2
            ;;
    esac
done

if [ -n "$CASE_NAMES" ] && [ -n "$CASE_INDEXES" ]; then
    echo "错误: --cases 和 --indexes 不能同时使用" >&2
    exit 2
fi
if [ -z "$RESULT_PATH" ]; then
    if [ "$COMMAND" = "single" ]; then
        RESULT_PATH="$SCRIPT_DIR/single_result.xlsx"
    else
        RESULT_PATH="$SCRIPT_DIR/result.xlsx"
    fi
fi
validate_run_mode

case "$COMMAND" in
    single)
        run_single
        ;;
    batch)
        run_batch
        ;;
    batch_exec)
        run_batch_from_excel
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误: 未知命令 $COMMAND" >&2
        show_help
        exit 2
        ;;
esac
