# 克隆与检出

仓库克隆、PR 分支检出、base 分支确定、merge-base 计算的操作指南。

> **脚本替代**：大部分操作可通过 `python scripts/fetch_pr_context.py --repo <owner/repo> --pr <N>` 一键完成。本文档提供手动操作的详细步骤，供需要定制时参考。

---

## 1. 仓库克隆

### 标准克隆（推荐）

```bash
git clone --depth=500 https://gitcode.com/{owner}/{repo}.git /tmp/{task_prefix}_{owner}_{repo}_{timestamp}
cd /tmp/{task_prefix}_{owner}_{repo}_{timestamp}
```

> `--depth=500` 确保有足够的历史记录来找到 merge base。

### 轻量克隆

```bash
git clone --depth=200 "https://gitcode.com/${owner}/${repo}.git" "$WORK_DIR"
```

> 用于仅需查看变更概览的场景（如设计文档生成）。

### 全量克隆

```bash
git clone https://gitcode.com/{owner}/{repo}.git /tmp/{task_prefix}_{owner}_{repo}_{timestamp}
```

> 仅在 `--depth=500` 仍无法找到 merge-base 时使用。

---

## 2. 临时目录命名规范

| Skill | 目录名格式 | 示例 |
|-------|-----------|------|
| gitcode-pr-handler | `/tmp/gitcode-pr-handler_{owner}_{repo}_{timestamp}` | `/tmp/gitcode-pr-handler_cann_ops-math_20260323_200609` |
| gitcode-issue-gen | `/tmp/gitcode-issue-gen_{owner}_{repo}_{timestamp}` | `/tmp/gitcode-issue-gen_cann_ops-math_20260323_200609` |
| gitcode-issue-handler | `/tmp/gitcode-issue-handler_{owner}_{repo}_{timestamp}` | `/tmp/gitcode-issue-handler_cann_ops-math_20260323_200609` |

---

## 3. PR 分支检出

### 主方案：通过 merge-requests 引用（推荐）

GitCode 平台会自动将 fork PR 的分支同步到上游仓库的 `refs/merge-requests/` 引用下，因此无需访问 fork 仓库即可获取 PR 代码。

```bash
git fetch origin +refs/merge-requests/{pr_number}/head:pr_{pr_number}
git checkout pr_{pr_number}
```

> **注意**：GitCode 使用 `refs/merge-requests/{n}/head`，不是 GitHub 的 `refs/pull/{n}/head`。

### 降级方案：从 fork remote 获取

仅当主方案失败（如上游仓库权限限制）时使用。需要从 PR API 获取 fork 信息（`head.repo.full_name`、`head.ref`）。

```bash
git remote add fork https://gitcode.com/{fork_user}/{fork_repo}.git
git fetch fork {head_ref}:pr_{pr_number}
git checkout pr_{pr_number}
```

> **禁止**使用 API（`contents/`、`git/blobs/`、`raw.gitcode.com`）获取文件内容作为替代方案。这些方式无法获取完整仓库上下文且流程脆弱。

---

## 4. Base 分支确定

### 方式一：API 获取（推荐）

从 PR 详情 API 的 `base.ref` 字段获取目标分支名。

### 方式二：git remote 自动检测

```bash
BASE_BRANCH=$(git remote show origin | grep 'HEAD branch' | cut -d':' -f2 | tr -d ' ')
```

---

## 5. Merge-base 计算

```bash
# 获取 base 分支
git fetch origin {base_ref}:base_branch

# 计算 merge base
MERGE_BASE=$(git merge-base base_branch pr_{pr_number})
```

### merge-base 失败处理

```bash
# 如果 merge-base 失败，尝试加深历史
git fetch --deepen=500
# 然后重试
MERGE_BASE=$(git merge-base base_branch pr_{pr_number})
```

> **注意**：不要直接使用 base_branch 作为比较基准，必须使用 merge-base。

---

## 6. 错误处理

| 错误场景 | 处理方式 |
|----------|----------|
| 克隆失败（网络） | 检查网络连接，重试 |
| 克隆失败（权限） | 提示用户使用 SSH 或配置凭证 |
| PR 引用不存在 | 确认 PR 编号，等待 git fetch 完成 |
| fork remote 不可访问（403） | 改用 `refs/merge-requests/{n}/head` 从上游获取 |
| merge-base 找不到 | 加深历史 `git fetch --deepen=500`，仍失败则全量克隆 |
