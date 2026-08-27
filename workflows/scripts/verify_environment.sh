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
# verify_environment.sh - 验证 Ascend C 开发环境并保存结果（JSON 格式）
# 使用：bash verify_environment.sh <operator_name>
# 示例：bash verify_environment.sh softmax0309

set -e

# 解析参数
OPERATOR_NAME=${1:-""}
if [ -z "$OPERATOR_NAME" ]; then
    echo "用法: $0 <operator_name>"
    echo "示例: $0 softmax0309"
    exit 1
fi

# 设置保存路径
SAVE_DIR="operators/${OPERATOR_NAME}/docs"
SAVE_FILE="${SAVE_DIR}/environment.json"

# 检查项目是否已初始化
if [ ! -d "$SAVE_DIR" ]; then
    echo "❌ 错误：项目目录不存在"
    echo ""
    echo "请先运行项目初始化："
    if [ -f "workflows/scripts/init_operator_project.sh" ]; then
        echo "  bash workflows/scripts/init_operator_project.sh ${OPERATOR_NAME}"
    else
        _script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        echo "  bash ${_script_dir}/init_operator_project.sh ${OPERATOR_NAME}"
    fi
    exit 1
fi

echo "================================================================"
echo "Ascend C 开发环境验证"
echo "================================================================"
echo ""
echo "算子名称: ${OPERATOR_NAME}"
echo "保存路径: ${SAVE_FILE}"
echo ""

ERRORS=0
WARNINGS=0

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 用数组收集 JSON 数据
declare -A ENV_DATA

# 辅助函数
error() {
    echo -e "${RED}❌ $1${NC}"
    ERRORS=$((ERRORS + 1))
}

warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
    WARNINGS=$((WARNINGS + 1))
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# JSON 转义函数
json_escape() {
    local str="$1"
    str="${str//\\/\\\\}"  # 反斜杠
    str="${str//\"/\\\"}"  # 双引号
    str="${str//$'\n'/\\n}" # 换行
    str="${str//$'\r'/\\r}" # 回车
    str="${str//$'\t'/\\t}" # 制表符
    echo "$str"
}

# 收集环境信息
collect_env_info() {
    # --- 自动检测/纠正 ASCEND_HOME_PATH ---
    detect_ascend_home() {
        local _resolved=""
        _resolve_toolkit_path() {
            local base="$1"
            if [ -d "$base/compiler" ]; then
                echo "$base"
                return
            fi
            local toolkit_dir="$base/ascend-toolkit"
            if [ -d "$toolkit_dir" ]; then
                for d in $(ls -d "$toolkit_dir"/* 2>/dev/null | sort -r); do
                    if [ -d "$d/compiler" ]; then
                        echo "$d"
                        return
                    fi
                done
                if [ -L "$toolkit_dir/latest" ]; then
                    local real
                    real=$(readlink -f "$toolkit_dir/latest")
                    if [ -d "$real/compiler" ]; then
                        echo "$real"
                        return
                    fi
                fi
            fi
            for d in $(ls -d "$base"/cann-* 2>/dev/null | sort -r); do
                if [ -d "$d/compiler" ]; then
                    echo "$d"
                    return
                fi
            done
            echo ""
        }

        for var in ASCEND_HOME ASCEND_TOOLKIT_HOME ASCEND_HOME_PATH ASCEND_CANN_HOME; do
            local val
            eval "val=\$$var"
            if [ -n "$val" ] && [ -d "$val" ]; then
                _resolved=$(_resolve_toolkit_path "$val")
                if [ -n "$_resolved" ]; then
                    TOOLKIT_PATH="$_resolved"
                    return
                fi
            fi
        done

        if [ -d "/usr/local/Ascend" ]; then
            _resolved=$(_resolve_toolkit_path "/usr/local/Ascend")
            if [ -n "$_resolved" ]; then
                TOOLKIT_PATH="$_resolved"
                echo "  ✓ 自动发现: Toolkit路径=$TOOLKIT_PATH"
                return
            fi
        fi

        if [ -d "$HOME/Ascend" ]; then
            _resolved=$(_resolve_toolkit_path "$HOME/Ascend")
            if [ -n "$_resolved" ]; then
                TOOLKIT_PATH="$_resolved"
                echo "  ✓ 自动发现: Toolkit路径=$TOOLKIT_PATH"
                return
            fi
        fi
    }

    detect_ascend_home
    echo ""

    # 1. 检查环境变量
    echo "[1/7] 检查环境变量..."
    echo "────────────────────────────────────────────────────────────────"

    if [ -z "$ASCEND_HOME_PATH" ] && [ -z "$TOOLKIT_PATH" ]; then
        error "ASCEND_HOME_PATH 未设置且无法自动发现 Toolkit"
        ENV_DATA[ascend_home_path]=""
        ENV_DATA[ascend_home_path_valid]="false"

        echo ""
        echo "  解决方法："
        echo "  export ASCEND_HOME_PATH=/usr/local/Ascend"
        echo "  或"
        echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
        echo ""
    else
        if [ -n "$ASCEND_HOME_PATH" ]; then
            success "ASCEND_HOME_PATH = $ASCEND_HOME_PATH"
            ENV_DATA[ascend_home_path]="$(json_escape "$ASCEND_HOME_PATH")"
        else
            warning "ASCEND_HOME_PATH 未设置"
            ENV_DATA[ascend_home_path]=""
        fi
        if [ -n "$TOOLKIT_PATH" ]; then
            success "Toolkit 路径 = $TOOLKIT_PATH"
            ENV_DATA[toolkit_path]="$(json_escape "$TOOLKIT_PATH")"
        fi
        ENV_DATA[ascend_home_path_valid]="true"
    fi
    
    # 2. 检查 CANN 安装
    echo ""
    echo "[2/7] 检查 CANN 安装..."
    echo "────────────────────────────────────────────────────────────────"
    
    if [ -d "$TOOLKIT_PATH" ]; then
        success "CANN Toolkit 目录存在"
        ENV_DATA[cann_dir_exists]="true"

        CANN_VERSION=""
        if [ -f "$TOOLKIT_PATH/compiler/version.info" ]; then
            CANN_VERSION=$(grep '^Version=' "$TOOLKIT_PATH/compiler/version.info" | cut -d'=' -f2)
        fi
        if [ -z "$CANN_VERSION" ]; then
            CANN_VERSION=$(basename "$TOOLKIT_PATH" | sed 's/cann-//' | sed 's/-beta//')
        fi
        ENV_DATA[cann_version]="$(json_escape "$CANN_VERSION")"
    else
        error "CANN Toolkit 目录不存在"
        ENV_DATA[cann_dir_exists]="false"
    fi
    
    if [ -d "$TOOLKIT_PATH/aarch64-linux" ]; then
        success "aarch64-linux 目录存在"
        ENV_DATA[arch_dir_exists]="true"
        ENV_DATA[arch_dir]="aarch64-linux"
    elif [ -d "$TOOLKIT_PATH/arm64-linux" ]; then
        success "arm64-linux 目录存在"
        ENV_DATA[arch_dir_exists]="true"
        ENV_DATA[arch_dir]="arm64-linux"
    elif [ -d "$TOOLKIT_PATH/x86_64-linux" ]; then
        success "x86_64-linux 目录存在"
        ENV_DATA[arch_dir_exists]="true"
        ENV_DATA[arch_dir]="x86_64-linux"
    else
        error "架构目录不存在 ($TOOLKIT_PATH)"
        ENV_DATA[arch_dir_exists]="false"
    fi
    
    # 3. 检查编译器
    echo ""
    echo "[3/7] 检查 Ascend C 编译器..."
    echo "────────────────────────────────────────────────────────────────"
    
    ARCH_DIR="${ENV_DATA[arch_dir]:-aarch64-linux}"
    COMPILER_PATH="$TOOLKIT_PATH/$ARCH_DIR/ccec_compiler/bin/bisheng"
    
    if [ -f "$COMPILER_PATH" ]; then
        success "编译器存在: bisheng"
        ENV_DATA[bisheng_path]="$(json_escape "$COMPILER_PATH")"
        ENV_DATA[bisheng_exists]="true"
        
        if [ -x "$COMPILER_PATH" ]; then
            success "编译器可执行"
            ENV_DATA[bisheng_executable]="true"
        else
            error "编译器不可执行"
            ENV_DATA[bisheng_executable]="false"
        fi
    else
        error "编译器不存在: bisheng"
        ENV_DATA[bisheng_path]=""
        ENV_DATA[bisheng_exists]="false"
    fi
    
    # 4. 检查头文件
    echo ""
    echo "[4/7] 检查头文件..."
    echo "────────────────────────────────────────────────────────────────"
    
    HEADER_PATHS=(
        "$TOOLKIT_PATH/$ARCH_DIR/ascendc/include/basic_api/kernel_operator.h"
        "$TOOLKIT_PATH/$ARCH_DIR/asc/include/kernel_operator.h"
        "$TOOLKIT_PATH/$ARCH_DIR/include/ascendc/basic_api/kernel_operator.h"
        "$TOOLKIT_PATH/include/kernel_operator.h"
    )
    
    HEADER_FOUND=false
    HEADER_PATH=""
    
    for path in "${HEADER_PATHS[@]}"; do
        if [ -f "$path" ]; then
            HEADER_PATH="$path"
            HEADER_FOUND=true
            break
        fi
    done
    
    if [ "$HEADER_FOUND" = true ]; then
        success "头文件存在: kernel_operator.h"
        ENV_DATA[kernel_operator_h]="$(json_escape "$HEADER_PATH")"
        ENV_DATA[header_exists]="true"
    else
        error "头文件不存在: kernel_operator.h"
        ENV_DATA[kernel_operator_h]=""
        ENV_DATA[header_exists]="false"
    fi
    
    # 5. 检查库文件
    echo ""
    echo "[5/7] 检查库文件..."
    echo "────────────────────────────────────────────────────────────────"
    
    LIB_REGISTER="$TOOLKIT_PATH/lib64/libregister.so"
    LIB_ACL="$TOOLKIT_PATH/lib64/libascendcl.so"
    LIBS_OK=true
    
    if [ -f "$LIB_REGISTER" ]; then
        success "libregister.so 存在"
        ENV_DATA[libregister_so]="$(json_escape "$LIB_REGISTER")"
    else
        error "libregister.so 不存在"
        ENV_DATA[libregister_so]=""
        LIBS_OK=false
    fi
    
    if [ -f "$LIB_ACL" ]; then
        success "libascendcl.so 存在"
        ENV_DATA[libascendcl_so]="$(json_escape "$LIB_ACL")"
    else
        error "libascendcl.so 不存在"
        ENV_DATA[libascendcl_so]=""
        LIBS_OK=false
    fi
    
    ENV_DATA[all_libs_exist]="$LIBS_OK"
    
    # 6. 检查 Simulator 可运行性
    echo ""
    echo "[6/7] 检查 Simulator 支持情况..."
    echo "────────────────────────────────────────────────────────────────"

    KIRIN_PLATFORMS=(Kirin9030 KirinX90)
    SIM_PLATFORMS_JSON="{}"

    if [ -d "$ASCEND_HOME_PATH/x86_64-linux/simulator" ]; then
        _sim_root="$ASCEND_HOME_PATH/x86_64-linux/simulator"
    elif [ -d "$ASCEND_HOME_PATH/aarch64-linux/simulator" ]; then
        _sim_root="$ASCEND_HOME_PATH/aarch64-linux/simulator"
    else
        _sim_root=""
    fi

    if [ -n "$_sim_root" ]; then
        _entries=""
        for plat in "${KIRIN_PLATFORMS[@]}"; do
            if [ -f "$_sim_root/$plat/lib/libruntime_camodel.so" ]; then
                success "$plat: 可运行"
                _entries+="\"$plat\": true,"
            elif [ -d "$_sim_root/$plat" ]; then
                warning "$plat: 不可运行（目录存在但缺 libruntime_camodel.so）"
                _entries+="\"$plat\": false,"
            fi
        done
        SIM_PLATFORMS_JSON="{${_entries%,}}"
    else
        warning "未发现 simulator 目录"
    fi

    ENV_DATA[simulator_platforms]="$SIM_PLATFORMS_JSON"
    
    # 8. 检查 asc-devkit
    echo ""
    echo "[7/7] 检查 asc-devkit..."
    echo "────────────────────────────────────────────────────────────────"
    
    # 自动检测 asc-devkit 路径
    detect_asc_devkit() {
        # 优先级 1: 环境变量已设置且目录存在
        if [ -n "$ASC_DEVKIT_DIR" ] && [ -d "$ASC_DEVKIT_DIR" ]; then
            echo "$ASC_DEVKIT_DIR"
            return
        fi
        
        # 优先级 2: 当前工作目录下的 asc-devkit
        if [ -d "$(pwd)/asc-devkit" ]; then
            echo "$(pwd)/asc-devkit"
            return
        fi
        
        # 优先级 3: 脚本真实路径所在目录的上级目录下的 asc-devkit
        # 使用 readlink -f 解析 symlink，确保能定位到真实的插件根目录
        local script_real
        script_real="$(readlink -f "${BASH_SOURCE[0]}")"
        local script_dir
        script_dir="$(cd "$(dirname "$script_real")" && pwd)"
        local project_root
        project_root="$(dirname "$script_dir")"
        # 脚本在 workflows/scripts/，project_root 是 workflows/，再上级才是插件根目录
        local plugin_root
        plugin_root="$(dirname "$project_root")"
        if [ -d "$plugin_root/asc-devkit" ]; then
            echo "$plugin_root/asc-devkit"
            return
        fi
        
        # 优先级 4: global 模式下的 ~/.config/opencode/asc-devkit
        local global_devkit="${HOME}/.config/opencode/asc-devkit"
        if [ -d "$global_devkit" ]; then
            echo "$global_devkit"
            return
        fi
        
        # 优先级 5: ~/.claude/asc-devkit (Claude global mode)
        local claude_devkit="${HOME}/.claude/asc-devkit"
        if [ -d "$claude_devkit" ]; then
            echo "$claude_devkit"
            return
        fi
        
        # 优先级 6: 环境变量已设置但目录不存在（返回原值，后续会报错）
        if [ -n "$ASC_DEVKIT_DIR" ]; then
            echo "$ASC_DEVKIT_DIR"
            return
        fi
        
        # 未找到
        echo ""
    }
    
    ASC_DEVKIT_PATH=$(detect_asc_devkit)
    
    if [ -n "$ASC_DEVKIT_PATH" ] && [ -d "$ASC_DEVKIT_PATH" ]; then
        # 显示发现路径（如果是从自动检测得到的）
        if [ -z "$ASC_DEVKIT_DIR" ] || [ "$ASC_DEVKIT_DIR" != "$ASC_DEVKIT_PATH" ]; then
            echo "  ✓ 自动发现: ASC_DEVKIT_PATH=$ASC_DEVKIT_PATH"
        fi
        success "asc-devkit 目录存在"
        ENV_DATA[asc_devkit_path]="$(json_escape "$ASC_DEVKIT_PATH")"
        ENV_DATA[asc_devkit_exists]="true"
        
        # 检查 API 文档
        if [ -d "$ASC_DEVKIT_PATH/docs/api" ]; then
            success "API 文档目录存在"
            ENV_DATA[api_docs_exist]="true"
        else
            warning "API 文档目录不存在"
            ENV_DATA[api_docs_exist]="false"
        fi
        
        # 检查 CMake 配置
        if [ -d "$ASC_DEVKIT_PATH/cmake" ]; then
            success "CMake 配置目录存在"
            ENV_DATA[cmake_config_exists]="true"
        else
            warning "CMake 配置目录不存在"
            ENV_DATA[cmake_config_exists]="false"
        fi
        
        # 统计示例数量
        EXAMPLES_COUNT=$(find "$ASC_DEVKIT_PATH/examples" -type f -name "*.asc" 2>/dev/null | wc -l)
        if [ "$EXAMPLES_COUNT" -gt 0 ]; then
            success "找到 $EXAMPLES_COUNT 个示例"
            ENV_DATA[examples_count]="$EXAMPLES_COUNT"
        else
            warning "未找到示例文件"
            ENV_DATA[examples_count]="0"
        fi
    else
        error "asc-devkit 目录不存在"
        ENV_DATA[asc_devkit_path]=""
        ENV_DATA[asc_devkit_exists]="false"
    fi
    
}

# 生成 JSON 文件
generate_json() {
    cat > "$SAVE_FILE" << EOF
{
  "check_time": "$(date -Iseconds)",
  "operator": "${OPERATOR_NAME}",
  "environment": {
    "ascend_home_path": "${ENV_DATA[ascend_home_path]}",
    "ascend_home_path_valid": ${ENV_DATA[ascend_home_path_valid]},
    "toolkit_path": "${ENV_DATA[toolkit_path]}",
    "cann_dir_exists": ${ENV_DATA[cann_dir_exists]},
    "cann_version": "${ENV_DATA[cann_version]}",
    "simulator_platforms": ${ENV_DATA[simulator_platforms]},
    "arch_dir": "${ENV_DATA[arch_dir]}",
    "arch_dir_exists": ${ENV_DATA[arch_dir_exists]},
    "bisheng_path": "${ENV_DATA[bisheng_path]}",
    "bisheng_exists": ${ENV_DATA[bisheng_exists]},
    "bisheng_executable": ${ENV_DATA[bisheng_executable]},
    "kernel_operator_h": "${ENV_DATA[kernel_operator_h]}",
    "header_exists": ${ENV_DATA[header_exists]},
    "libregister_so": "${ENV_DATA[libregister_so]}",
    "libascendcl_so": "${ENV_DATA[libascendcl_so]}",
    "all_libs_exist": ${ENV_DATA[all_libs_exist]},
    "asc_devkit_path": "${ENV_DATA[asc_devkit_path]}",
    "asc_devkit_exists": ${ENV_DATA[asc_devkit_exists]},
    "api_docs_exist": ${ENV_DATA[api_docs_exist]},
    "cmake_config_exists": ${ENV_DATA[cmake_config_exists]},
    "examples_count": ${ENV_DATA[examples_count]}
  },
  "validation": {
    "all_passed": $([ $ERRORS -eq 0 ] && echo "true" || echo "false"),
    "error_count": $ERRORS,
    "warning_count": $WARNINGS
  }
}
EOF
}

# 主流程
collect_env_info
generate_json

# 输出总结
echo ""
echo "================================================================"
echo "验证结果"
echo "================================================================"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ 环境验证通过！${NC}"
    echo ""
    echo "环境检查结果已保存到："
    echo "  ${SAVE_FILE}"
    echo ""
    echo "后续步骤："
    echo "  1. 开始 Phase 1：需求分析与方案设计"
    echo "  2. 生成设计文档：docs/DESIGN.md"
else
    echo -e "${RED}✗ 环境验证失败${NC}"
    echo ""
    echo "错误数量：$ERRORS"
    echo "警告数量：$WARNINGS"
    echo ""
    echo "请根据上述错误信息修复环境配置后重试。"
    exit 1
fi
