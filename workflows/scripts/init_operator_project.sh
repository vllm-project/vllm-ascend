# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

#!/bin/bash
# init_operator_project.sh - 初始化算子项目目录结构
# 使用：bash init_operator_project.sh <operator_name>
# 示例：bash init_operator_project.sh softmax0309

set -e

# 解析参数
OPERATOR_NAME=${1:-""}
if [ -z "$OPERATOR_NAME" ]; then
    echo "用法: $0 <operator_name>"
    echo "示例: $0 softmax0309"
    exit 1
fi

# 设置项目根目录
PROJECT_ROOT="operators/${OPERATOR_NAME}"

echo "================================================================"
echo "算子项目初始化"
echo "================================================================"
echo ""
echo "算子名称: ${OPERATOR_NAME}"
echo "项目路径: ${PROJECT_ROOT}"
echo ""

# 检查目录是否已存在
if [ -d "$PROJECT_ROOT" ]; then
    echo "⚠️  警告：项目目录已存在"
    if [ -t 0 ]; then
        # 交互模式：询问用户
        read -p "是否继续？（y/n）" -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "已取消"
            exit 0
        fi
    else
        # 非交互模式（Agent 调用）：自动继续
        echo "  非交互模式，自动继续..."
    fi
fi

# 创建目录结构
echo "[1/3] 创建目录结构..."
echo "────────────────────────────────────────────────────────────────"
mkdir -p "${PROJECT_ROOT}/docs"
mkdir -p "${PROJECT_ROOT}/build"
mkdir -p "${PROJECT_ROOT}/test"

echo "✓ docs/      - 文档目录"
echo "✓ build/     - 编译输出目录"
echo "✓ test/      - 测试数据目录"
echo ""

# 创建 README
echo "[2/3] 创建 README.md..."
echo "────────────────────────────────────────────────────────────────"
cat > "${PROJECT_ROOT}/README.md" << README_EOF
# ${OPERATOR_NAME} 算子

## 基本信息

- **算子名称**：${OPERATOR_NAME}
- **创建时间**：$(date '+%Y-%m-%d %H:%M:%S')
- **开发状态**：开发中

## 目录结构

\`\`\`
${OPERATOR_NAME}/
├── docs/           # 设计、计划、环境、审查、串讲文档
├── build/          # 编译输出
├── test/           # 测试数据
├── ${OPERATOR_NAME}.asc  # Kernel 实现（待创建）
├── CMakeLists.txt  # 构建脚本（待创建）
├── gen_data.py  # Golden 数据生成（待创建）
└── run.sh          # 运行脚本（待创建）
\`\`\`

## 开发进度

- [ ] Phase 0: 环境准备
- [ ] Phase 1: 需求分析与方案设计
- [ ] Phase 2: 算子实现
- [ ] Phase 3: 全面测试
- [ ] Phase 4-6: 问题处理、总结、文档

## 参考资料

- 设计文档：docs/DESIGN.md（待生成）
- 环境检查：docs/environment.md（待生成）
README_EOF

echo "✓ README.md 已创建"
echo ""

# 显示后续步骤
echo "[3/3] 后续步骤..."
echo "────────────────────────────────────────────────────────────────"
echo ""
echo "✅ 项目初始化完成！"
echo ""
echo "下一步操作："
echo "  1. 加载 /ascendc-env-check skill，按 skill 指引完成 CANN 环境与 NPU 设备检查"
echo "  2. 按 workflows/templates/environment-template.md 填空，生成"
echo "     ${PROJECT_ROOT}/docs/environment.md"
echo "  3. 开始设计阶段（Phase 1）"
echo ""
echo "================================================================"
