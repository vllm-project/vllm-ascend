#!/bin/bash
# setup-workspace-branch.sh
# 一次性初始化：创建独立的 workspace-outputs 分支 + 本地 worktree
#
# 使用方式: bash .workspace/scripts/setup-workspace-branch.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BRANCH="workspace-outputs"
WORKTREE_DIR="$REPO_ROOT/.workspace/sync"

echo "==> 1. 创建 orphan 分支: $BRANCH"

cd "$REPO_ROOT"

if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
    echo "    分支 $BRANCH 已存在，跳过创建"
else
    TMP_DIR="$(mktemp -d)"
    trap "rm -rf $TMP_DIR" EXIT

    git clone --no-checkout --single-branch --branch main "$REPO_ROOT" "$TMP_DIR" 2>/dev/null || \
    git clone --no-checkout "$REPO_ROOT" "$TMP_DIR"

    cd "$TMP_DIR"
    git checkout --orphan "$BRANCH"
    git rm -rf . 2>/dev/null || true
    echo "# Workspace Outputs" > README.md
    git add README.md
    git commit -m "init workspace-outputs branch"
    git push -u origin "$BRANCH"
    cd "$REPO_ROOT"
    git fetch origin "$BRANCH"

    echo "    分支 $BRANCH 已创建并推送"
fi

echo ""
echo "==> 2. 创建本地 worktree: $WORKTREE_DIR"

if [ -d "$WORKTREE_DIR" ]; then
    echo "    worktree 目录已存在，跳过"
else
    git worktree add "$WORKTREE_DIR" "$BRANCH"
    echo "    worktree 已创建"
fi

echo ""
echo "==> 3. 添加 .gitignore 规则"

GITIGNORE="$REPO_ROOT/.gitignore"
if ! grep -q "\.workspace/sync/" "$GITIGNORE" 2>/dev/null; then
    echo ".workspace/sync/" >> "$GITIGNORE"
    echo "    已添加 .workspace/sync/ 到 .gitignore"
else
    echo "    已存在，跳过"
fi

echo ""
echo "==> 初始化完成 ✅"
echo ""
echo "后续操作:"
echo "  推送本地产出到远端:  bash .workspace/scripts/sync-outputs.sh push"
echo "  拉取远端产出到本地:  bash .workspace/scripts/sync-outputs.sh pull"
echo "  查看状态:            bash .workspace/scripts/sync-outputs.sh status"
