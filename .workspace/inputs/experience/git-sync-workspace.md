# Git Sync Workspace 使用指南

## 解决的问题

私有网络服务器上的分析产出（`.workspace/outputs/`）不方便直接传输到本地机器。通过一个独立的 git orphan 分支 + worktree 做中转，实现产出文件的同步。

## 架构

```
服务器（私有网络）                         本地机器
─────────────────                    ─────────────────
.workspace/outputs/                  .workspace/outputs/
        │                                    ▲
        ▼                                    │
.workspace/sync/  ←→  git push/pull  ←→  .workspace/sync/
(独立 worktree,                       (独立 worktree,
 orphan 分支)                          orphan 分支)
```

- `workspace-outputs` 分支是一个 **orphan 分支**，与主代码历史完全隔离
- `.workspace/sync/` 是 checkout 到该分支的 git worktree
- `sync-outputs.sh` 负责在 `outputs/` 和 `sync/` 之间搬运文件

## 初始化（每台机器执行一次）

```bash
# 在仓库根目录
bash .workspace/scripts/setup-workspace-branch.sh
```

这会：
1. 创建 orphan 分支 `workspace-outputs`（仅第一台机器需要推送）
2. 在 `.workspace/sync/` 创建 worktree
3. 将 `.workspace/sync/` 加入 `.gitignore`（避免主分支跟踪）

## 日常使用

### 推送产出（服务器上）

```bash
bash .workspace/scripts/sync-outputs.sh push
```

做的事：
1. `rsync` 将 `.workspace/outputs/` 复制到 `.workspace/sync/outputs/`
2. 在 worktree 中 `git commit` + `git push`

### 拉取产出（本地机器上）

```bash
bash .workspace/scripts/sync-outputs.sh pull
```

做的事：
1. 在 worktree 中 `git pull`
2. `rsync` 将 `.workspace/sync/outputs/` 复制到 `.workspace/outputs/`

### 查看状态

```bash
bash .workspace/scripts/sync-outputs.sh status
```

显示：
- 当前分支 git 状态
- 未推送的提交
- 本地 outputs/ 和 worktree 的差异

## 注意事项

- 分支 commit 历史会线性增长，定期 squash 或偶尔清理即可
- worktree 不会自动提交——每次都需要手动 `push`
- 两台机器同时编辑可能产生冲突：`pull` 时会提示，手动解决后继续 `push`
- 如果远端还没有 `workspace-outputs` 分支，第一台机器运行 setup 脚本时自动创建并推送
