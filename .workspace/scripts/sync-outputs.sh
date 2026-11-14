#!/bin/bash
# sync-outputs.sh
# 在 .workspace/outputs/ 和 git orphan 分支之间同步产出文件
#
# 使用方式:
#   bash .workspace/scripts/sync-outputs.sh push     # 本地 → 远端
#   bash .workspace/scripts/sync-outputs.sh pull     # 远端 → 本地
#   bash .workspace/scripts/sync-outputs.sh status   # 查看差异

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKTREE_DIR="$REPO_ROOT/.workspace/sync"
OUTPUTS_DIR="$REPO_ROOT/.workspace/outputs"
BRANCH="workspace-outputs"

die() { echo "ERROR: $*" >&2; exit 1; }

# ── 前置检查 ──────────────────────────────────────────────

check_worktree() {
    if [ ! -d "$WORKTREE_DIR" ]; then
        die "worktree 不存在，请先运行: bash .workspace/scripts/setup-workspace-branch.sh"
    fi
}

# ── push: 本地产出 → git 推送 ─────────────────────────────

cmd_push() {
    check_worktree

    # 1. 同步 outputs/ 到 worktree
    if [ -d "$OUTPUTS_DIR" ]; then
        mkdir -p "$WORKTREE_DIR/outputs"
        rsync -a --delete "$OUTPUTS_DIR/" "$WORKTREE_DIR/outputs/"
        echo "==> 已同步 outputs/ 到 worktree"
    else
        echo "==> outputs/ 目录不存在，跳过"
    fi

    cd "$WORKTREE_DIR"

    # 2. 检查是否有变更
    if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
        echo "==> 无变更，跳过推送"
        return
    fi

    # 3. 提交并推送
    git add -A
    TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
    HOSTNAME="$(hostname)"

    # 检查是否有暂存内容
    if git diff --cached --quiet; then
        echo "==> 无暂存内容，跳过提交"
        return
    fi

    git commit -m "sync outputs ($TIMESTAMP from $HOSTNAME)"
    git push origin "$BRANCH"

    echo "==> 推送完成 ✅"
}

# ── pull: git 拉取 → 本地产出 ─────────────────────────────

cmd_pull() {
    check_worktree

    cd "$WORKTREE_DIR"

    # 1. 拉取远端
    echo "==> 拉取远端 $BRANCH ..."
    git pull origin "$BRANCH" || die "git pull 失败，请检查网络或手动处理冲突"

    # 2. 同步 worktree/outputs/ 到本地 outputs/
    if [ -d "$WORKTREE_DIR/outputs" ]; then
        mkdir -p "$OUTPUTS_DIR"
        rsync -a --delete "$WORKTREE_DIR/outputs/" "$OUTPUTS_DIR/"
        echo "==> 已同步 worktree/outputs/ 到本地 outputs/"
    else
        echo "==> worktree 中无 outputs/ 目录"
    fi

    echo "==> 拉取完成 ✅"
}

# ── status: 查看同步状态 ───────────────────────────────────

cmd_status() {
    check_worktree

    echo "──────────── Git 分支状态 ────────────"
    cd "$WORKTREE_DIR"
    git status
    echo ""

    echo "──────────── 未推送的提交 ────────────"
    git fetch origin "$BRANCH" 2>/dev/null || true
    LOCAL=$(git rev-parse HEAD 2>/dev/null || echo "none")
    REMOTE=$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo "none")
    if [ "$LOCAL" != "$REMOTE" ]; then
        if [ "$REMOTE" = "none" ]; then
            echo "  远端无分支"
        else
            echo "  本地: $LOCAL"
            echo "  远端: $REMOTE"
            echo ""
            git log --oneline "origin/$BRANCH..HEAD" 2>/dev/null || echo "  (无差异)"
        fi
    else
        echo "  已同步"
    fi
    echo ""

    echo "──────────── 本地 vs Worktree 差异 ────────────"
    if [ -d "$OUTPUTS_DIR" ] && [ -d "$WORKTREE_DIR/outputs" ]; then
        diff -rq "$OUTPUTS_DIR" "$WORKTREE_DIR/outputs" 2>/dev/null || true
    elif [ ! -d "$OUTPUTS_DIR" ]; then
        echo "  本地 outputs/ 不存在"
    elif [ ! -d "$WORKTREE_DIR/outputs" ]; then
        echo "  worktree outputs/ 不存在"
    fi
}

# ── main ───────────────────────────────────────────────────

case "${1:-}" in
    push)   cmd_push ;;
    pull)   cmd_pull ;;
    status) cmd_status ;;
    *)
        echo "用法: bash .workspace/scripts/sync-outputs.sh {push|pull|status}"
        echo ""
        echo "  push    — 将本地 .workspace/outputs/ 推送到远端 workspace-outputs 分支"
        echo "  pull    — 从远端拉取并恢复到本地 .workspace/outputs/"
        echo "  status  — 查看 git 分支和文件差异"
        exit 1
        ;;
esac
