#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# find_api_doc.sh — 结构无关的 Ascend C API 文档查找工具
#
# 用途：在 $ASC_DEVKIT_DIR/docs/zh/api/ 下搜索 API 文档，不假设子目录结构。
#       适用于 health check、ST 测试，以及作为 agent find 命令的参考实现。
#
# 用法：find_api_doc.sh <APIName> [APIName2 ...]
# 输出：每行一个匹配的文档路径（相对于 $ASC_DEVKIT_DIR）
# 退出码：0=找到，1=未找到，2=环境错误

set -euo pipefail

ASC_DEVKIT_DIR="${ASC_DEVKIT_DIR:-}"

if [ -z "$ASC_DEVKIT_DIR" ]; then
  echo "Error: ASC_DEVKIT_DIR is not set" >&2
  exit 2
fi

if [ ! -d "$ASC_DEVKIT_DIR/docs/zh/api/" ]; then
  echo "Error: $ASC_DEVKIT_DIR/docs/zh/api/ not found" >&2
  exit 2
fi

if [ $# -eq 0 ]; then
  echo "Usage: $0 <APIName> [APIName2 ...]" >&2
  echo "Searches $ASC_DEVKIT_DIR/docs/zh/api/ for matching .md files" >&2
  exit 2
fi

found=0
for api in "$@"; do
  while IFS= read -r path; do
    printf '%s\n' "${path#$ASC_DEVKIT_DIR/}"
    found=1
  done < <(find "$ASC_DEVKIT_DIR/docs/zh/api/" -name "${api}*.md" -type f 2>/dev/null | sort)
done

if [ "$found" -eq 0 ]; then
  echo "No API docs found for: $*" >&2
  exit 1
fi

exit 0
