#!/bin/bash
set -euo pipefail

# ==================== 参数解析 ====================
REVERSE=false
if [[ "${1:-}" == "--reverse" || "${1:-}" == "-r" ]]; then
    REVERSE=true
fi

# ==================== 基础信息 ====================
echo "🚀 Bailing v3 0day Adapter for vLLM-Ascend 0.20.2rc1"
echo "=================================================="

VLLM_PATH=$(python -c "import vllm, os; print(os.path.dirname(vllm.__path__[0]))" 2>/dev/null)
ASCEND_PATH=$(python -c "import vllm_ascend, os; print(os.path.dirname(vllm_ascend.__path__[0]))" 2>/dev/null)

if [ -z "$VLLM_PATH" ] || [ -z "$ASCEND_PATH" ]; then
    echo "❌ Error: vllm or vllm-ascend not found in current Python environment"
    exit 1
fi

echo "📍 vllm path:        $VLLM_PATH"
echo "📍 vllm-ascend path: $ASCEND_PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==================== Git 安全目录配置 ====================
echo ""
git config --global --add safe.directory "$VLLM_PATH"
git config --global --add safe.directory "$ASCEND_PATH"

# ==================== 根据模式设置变量 ====================
if [ "$REVERSE" = true ]; then
    ACTION_ICON="🔙"
    ACTION_VERB="Reversing"
    ACTION_PAST="reversed"
    GIT_APPLY_FLAG="-R"
else
    ACTION_ICON="🔧"
    ACTION_VERB="Applying"
    ACTION_PAST="applied"
    GIT_APPLY_FLAG=""
fi

# ==================== 应用/回退 vllm 补丁 ====================
VLLM_PATCH="$SCRIPT_DIR/bailing_v3_vllm.patch"
if [ ! -f "$VLLM_PATCH" ]; then
    echo "❌ Error: Patch file not found: $VLLM_PATCH"
    exit 1
fi

echo "$ACTION_ICON $ACTION_VERB vllm Bailing v3 patches..."
(
    cd "$VLLM_PATH"
    git apply $GIT_APPLY_FLAG -p1 "$VLLM_PATCH"
)
echo "✅ vllm patches $ACTION_PAST successfully"

# ==================== 应用/回退 vllm-ascend 补丁 ====================
ASCEND_PATCH="$SCRIPT_DIR/bailing_v3_vllm_ascend.patch"
if [ ! -f "$ASCEND_PATCH" ]; then
    echo "❌ Error: Patch file not found: $ASCEND_PATCH"
    exit 1
fi

echo "$ACTION_ICON $ACTION_VERB vllm-ascend Bailing v3 patches..."
(
    cd "$ASCEND_PATH"
    git apply $GIT_APPLY_FLAG -p1 "$ASCEND_PATCH"
)
echo "✅ vllm-ascend patches $ACTION_PAST successfully"

# ==================== 完成提示 ====================
echo ""
if [ "$REVERSE" = true ]; then
    echo "🎉 All done! Bailing v3 patches have been cleanly reverted."
else
    echo "🎉 All done! You can now run Bailing v3 inference."
fi