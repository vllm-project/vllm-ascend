---
name: gitcode-toolkit
description: GitCode 协作通用基础参考（内部参考，不直接触发）。提供 GitCode API、Token 配置、URL 解析、日志规范、变更展示，Git 克隆/分支/diff/log/remote 通用操作，以及 PR 创建、Issue 创建和 Issue 评论工作流等共享文档与确定性脚本。供 gitcode-pr-handler、gitcode-issue-gen、gitcode-issue-handler 等 GitCode 协作类 skill 引用使用，本 skill 自身不响应用户触发。
license: CANN-2.0
---

# GitCode Toolkit

GitCode 协作类 skill 的共享基础库。遵循**渐进式披露**原则：本文档为速查索引，详细内容按需在 `references/` 中展开，确定性操作由 `scripts/` 提供。

> **定位**：内部参考，不直接触发。其他 GitCode skill（`gitcode-pr-handler`、`gitcode-issue-gen`、`gitcode-issue-handler` 等）按需引用。

---

## 常见问题速答

### Q: 如何拉取 PR 代码到本地？

**方式一（推荐）：通过 merge-requests 引用**
```bash
git fetch origin +refs/merge-requests/{pr_number}/head:pr_{pr_number}
git checkout pr_{pr_number}
```

**方式二：从 fork remote 获取（仅当方式一失败时）**
```bash
git remote add fork https://gitcode.com/{fork_user}/{fork_repo}.git
git fetch fork {head_ref}:pr_{pr_number}
git checkout pr_{pr_number}
```

### Q: 如何查看 PR 相比 master 的变更？

**方式一：merge-base 模式（推荐）**
```bash
# 先获取 base 分支
git fetch origin {base_ref}:base_branch
# 计算 merge base
MERGE_BASE=$(git merge-base base_branch pr_{pr_number})
# 查看变更统计
git diff --stat $MERGE_BASE pr_{pr_number}
# 查看具体文件 diff
git diff $MERGE_BASE pr_{pr_number} -- {file_path}
```

**方式二：triple-dot 模式（本地已有 PR 分支）**
```bash
git diff --stat origin/${BASE_BRANCH}...HEAD
git diff origin/${BASE_BRANCH}...HEAD -- path/to/file.py
```

### Q: 如何获取 GitCode API Token？

Token 获取优先级：用户直接在消息中提供 → 环境变量 `GITCODE_TOKEN` → 询问用户。

**传递方式**（二选一）：
| 方式 | 示例 |
|------|------|
| Header 方式 | `curl -H "PRIVATE-TOKEN: {token}" https://api.gitcode.com/api/v5/...` |
| 参数方式 | `curl https://api.gitcode.com/api/v5/...?access_token={token}` |

> Token 需至少 **reporter** 以上权限才能创建 Issue。详见 [token-config.md](references/token-config.md)。

---

## 速查索引

### 必选流程（MUST）

| 步骤 | 内容 | 详细文档 | 脚本支持 |
|------|------|----------|----------|
| Step 0 | 环境预检（token / git / curl / /tmp / git-author） | [env-check.md](references/env-check.md) | `bash scripts/preflight.sh` |
| Token | 获取优先级：用户消息 → `GITCODE_TOKEN` → 询问 | [token-config.md](references/token-config.md) | — |
| URL | 解析 PR/Issue 链接：`/pull/{n}`, `/issues/{n}`, `/merge_requests/{n}` | [url-parsing.md](references/url-parsing.md) | `python scripts/parse_gitcode_url.py "<url>"` |
| API | PR/Issue/仓库 API + 错误码处理 | [gitcode-api.md](references/gitcode-api.md) | — |
| Issue 评论 | 目标解析、评论 POST/GET 回查、幂等与安全 | [issue-comment-workflow.md](references/issue-comment-workflow.md) | `gitcode_client.py` |
| 写操作授权 | 写前精确确认、内容变化失效与写后回查 | [authorization-contract.md](references/authorization-contract.md) | — |

### 建议流程（SHOULD）

| 步骤 | 内容 | 详细文档 | 脚本支持 |
|------|------|----------|----------|
| 克隆+检出 | PR 分支检出、base 分支确定、merge-base | [clone-and-checkout.md](references/clone-and-checkout.md) | — |
| Diff+变更 | 变更统计（merge-base / triple-dot 模式） | [diff-and-changes.md](references/diff-and-changes.md) | — |
| Log+文件 | commit 元信息提取、文件读取 | [log-and-show.md](references/log-and-show.md) | — |
| Remote+分支 | remote 管理、分支查询、推送 | [remote-and-branch.md](references/remote-and-branch.md) | — |
| PR 上下文 | 一键获取 clone + diff + log | — | `python scripts/fetch_pr_context.py --repo <owner/repo> --pr <N>` |

### 可选参考（MAY）

| 内容 | 详细文档 |
|------|----------|
| 变更列表展示格式 | [change-table-display.md](references/change-table-display.md) |
| 日志命名与记录规范 | [logging-conventions.md](references/logging-conventions.md) |
| Git 操作易错点对照 | [pitfalls.md](references/pitfalls.md) |

---

## 工作流

### 外部写操作调用约定

评论、Issue 更新、push、PR 和 CI 触发在写操作前必须有用户确认。已有工作流的“完整
预览 → 确认”契约继续有效。需要跨子流程复用检查点时按以下步骤传递证据：

1. 输入：调用方传入覆盖当前准确目标、操作类型和内容摘要的当前会话证据。
2. 复核：按 [authorization-contract.md](references/authorization-contract.md) 校验证据；
   证据缺失、目标或内容变化时停在写操作前，不得扩大作用域。
3. 执行：只执行检查点覆盖的 API 或 Git 命令，不自动合并，不改换目标。
4. 验证：评论/PR 使用 GET 回查，push 使用 `git ls-remote` 回查；失败时按具体工作流
   重试、降级或返回 blocker，不用 HTTP 成功码代替最终验证。

在依赖它们的具体操作前发现 Token、git author 或 remote 缺失时返回对应门禁；
这类凭据恢复不能替代用户确认。
调用共享脚本不能替代上层的真实用户确认。

### PR 创建工作流

从 fork 仓库向上游创建 PR 的完整 8 步流程（Step 1~8），含 git 提交身份校验（Step 5：5.1 硬性阻断 + 5.2 建议性校验）。详见：

> **[references/pr-creation-workflow.md](references/pr-creation-workflow.md)**

### Issue 创建工作流

创建 GitCode Issue 的标准 7 步流程。Step 3/4 为业务层（由调用方 skill 实现），infra 提供 Step 1/2/5/6/7 的通用能力。详见：

> **[references/issue-creation-workflow.md](references/issue-creation-workflow.md)**

## 验证

修改本 Skill 或脚本后至少运行：

```bash
python3 -m pytest -q infra/gitcode-toolkit/tests
bash -n infra/gitcode-toolkit/scripts/trigger_pr_pipeline.sh
python3 tests/lib/skill_validator.py validate-skill infra/gitcode-toolkit/SKILL.md
```

测试失败、脚本语法错误或 validator 不通过时不得发布；缺少可选测试依赖时明确记录
未验证边界，不得声称全部通过。`trigger_pr_pipeline.sh` 的参数测试不得设置 Token
或访问网络。

---

## 脚本工具

| 脚本 | 功能 | 用法 |
|------|------|------|
| `scripts/parse_gitcode_url.py` | 确定性 URL 解析，输出 JSON | `python scripts/parse_gitcode_url.py "<url>"` |
| `scripts/preflight.sh` | 环境预检，输出结构化报告 | `bash scripts/preflight.sh [--skip-git-author]` |
| `scripts/gitcode_client.py` | 共享的 GitCode HTTP、Token、URL、限流重试工具 | Python 模块，由上层脚本导入 |
| `scripts/fetch_pr_context.py` | 一键获取 PR 上下文（clone+diff+log） | `python scripts/fetch_pr_context.py --repo <owner/repo> --pr <N>` |
| `scripts/trigger_pr_pipeline.sh` | 触发 PR CI 流水线（提交 "compile" 评论） | `bash scripts/trigger_pr_pipeline.sh --repo <owner/repo> --pr <N>` |

---

## 适用平台

本工具集**仅适用于 GitCode 平台**（`gitcode.com`）。不支持 GitHub / GitLab 等平台。

---

## 参考文档索引

| 文档 | 说明 | 约束级别 |
|------|------|:---:|
| [env-check.md](references/env-check.md) | Step 0 环境预检（token / git / 临时目录） | **MUST** |
| [gitcode-api.md](references/gitcode-api.md) | PR/Issue/仓库 API + 错误处理 | **MUST** |
| [url-parsing.md](references/url-parsing.md) | URL 格式识别与解析 | **MUST** |
| [token-config.md](references/token-config.md) | Token 获取优先级 | **MUST** |
| [pr-creation-workflow.md](references/pr-creation-workflow.md) | PR 创建 8 步完整流程（含 git 身份校验） | SHOULD |
| [issue-comment-workflow.md](references/issue-comment-workflow.md) | Issue 评论的目标解析、POST/GET、幂等和安全边界 | SHOULD |
| [authorization-contract.md](references/authorization-contract.md) | 外部写操作的通用精确确认与回查边界 | MUST |
| [issue-creation-workflow.md](references/issue-creation-workflow.md) | Issue 创建 7 步流程（含预设模板） | SHOULD |
| [clone-and-checkout.md](references/clone-and-checkout.md) | 克隆、浅克隆、PR 分支检出、merge-base | SHOULD |
| [diff-and-changes.md](references/diff-and-changes.md) | diff 变更统计（merge-base / triple-dot 模式） | SHOULD |
| [log-and-show.md](references/log-and-show.md) | git log 元信息提取、git show 文件读取 | SHOULD |
| [remote-and-branch.md](references/remote-and-branch.md) | remote 管理、分支查询、push | SHOULD |
| [logging-conventions.md](references/logging-conventions.md) | 日志命名与记录规范 | MAY |
| [change-table-display.md](references/change-table-display.md) | 变更文件列表展示格式 | MAY |
| [pitfalls.md](references/pitfalls.md) | Git 操作易错点对照表 | MAY |
