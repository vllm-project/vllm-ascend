# GitCode API 参考文档

基础地址：`https://api.gitcode.com/api/v5`

---

## 目录

1. [PR 相关 API](#1-pr-相关-api)
2. [Issue 相关 API](#2-issue-相关-api)
3. [仓库内容 API](#3-仓库内容-api)
4. [错误处理](#4-错误处理)
5. [最佳实践](#5-最佳实践)
6. [命令速查](#6-命令速查)
7. [用户账号 API](#7-用户账号-api)

---

## 1. PR 相关 API

### 获取 PR 详情

```bash
GET /repos/{owner}/{repo}/pulls/{number}

curl 'https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}?access_token={token}'
```

**返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `number` | int | PR 编号 |
| `title` | string | PR 标题 |
| `body` | string | PR 描述 |
| `state` | string | 状态 (`open`, `closed`) |
| `user.login` | string | 作者用户名 |
| `user.name` | string | 作者名称 |
| `head.ref` | string | 源分支名 |
| `head.sha` | string | 源分支最新 commit |
| `base.ref` | string | 目标分支名 |
| `merged` | bool | 是否已合并 |
| `mergeable` | bool | 是否可合并 |
| `commits` | int | 提交数 |
| `additions` | int | 新增行数 |
| `deletions` | int | 删除行数 |
| `changed_files` | int | 变更文件数 |

### 获取 PR 文件列表

```bash
GET /repos/{owner}/{repo}/pulls/{number}/files

curl 'https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/files?access_token={token}&per_page=100'
```

> **注意**：此 API 可能返回认证错误（`Invalid header parameter: private-token, required`）。
> **推荐替代方案**：使用 git 命令获取变更文件，详见 [本 skill SKILL.md 速查表](../SKILL.md#git-操作核心命令) 及 [diff-and-changes.md](diff-and-changes.md)。

**返回字段（每个文件）**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `filename` | string | 文件路径 |
| `status` | string | 状态 (`added`, `modified`, `removed`, `renamed`) |
| `additions` | int | 新增行数 |
| `deletions` | int | 删除行数 |
| `changes` | int | 总变更行数 |
| `patch` | string | diff 内容 |

### 获取 PR Commits

```bash
GET /repos/{owner}/{repo}/pulls/{number}/commits

curl 'https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/commits?access_token={token}'
```

### 获取已有评论列表

```bash
GET /repos/{owner}/{repo}/pulls/{number}/comments

curl 'https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/comments?access_token={token}'
```

**用途**：避免重复提交相同评论

### 提交行内评论

针对具体代码行提交评论：

```bash
POST /repos/{owner}/{repo}/pulls/{number}/comments

curl -X POST 'https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/comments?access_token={token}' \
-H 'Content-Type: application/json' \
-H 'Accept: application/json' \
-d '{
  "body": "评论内容",
  "path": "src/main.py",
  "position": 45
}'
```

**参数说明**：

| 参数 | 必填 | 类型 | 说明 |
|------|:----:|------|------|
| `body` | Y | string | 评论内容 |
| `path` | Y | string | 文件相对路径（如 `src/main.py`） |
| `position` | Y | int | PR 分支版本文件中的绝对行号（1-based）。通过 `git clone` + `git checkout pr_{number}` 后，用 Read 工具读取本地文件得到的行号即为该值。不是 diff 相对行号，不是 diff hunk 索引。已在多个 PR 实测验证。 |
| `commit_id` | | string | 提交 SHA（可选，默认使用最新） |
| `in_reply_to` | | int | 回复的评论 ID（用于回复） |

### 提交整体评论

不针对特定代码行的评论：

```bash
POST /repos/{owner}/{repo}/pulls/{number}/comments

curl -X POST 'https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/comments?access_token={token}' \
-H 'Content-Type: application/json' \
-d '{
  "body": "整体评论内容"
}'
```

**触发 PR CI 流水线**：提交内容为 `compile` 的整体评论可触发 GitCode PR 的 CI 编译流水线：

```bash
curl -X POST 'https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}/comments?access_token={token}' \
-H 'Content-Type: application/json' \
-d '{"body": "compile"}'
```

> 脚本封装：`bash scripts/trigger_pr_pipeline.sh --repo <owner/repo> --pr <N>`。
> 共享脚本只执行 GitCode 评论调用，不解释调用方的业务授权模式。调用方必须在执行脚本前
> 完成真实用户确认和授权证据记录。

### 更新 PR

```bash
PATCH /repos/{owner}/{repo}/pulls/{number}

curl -X PATCH "https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{number}?access_token={token}" \
  --data-urlencode "title=PR标题" \
  --data-urlencode "body=PR描述内容"
```

### 创建 PR

```bash
POST /repos/{owner}/{repo}/pulls

curl -X POST "https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "access_token=${token}" \
  -d "title=${pr_title}" \
  -d "body=${pr_body}" \
  -d "head=${username}:${branch_name}" \
  -d "base=master"
```

> **head 参数格式**：从 fork 仓库向上游创建 PR 时，必须使用 `{fork用户名}:{分支名}` 格式。

### 获取已有 PR 列表

```bash
GET /repos/{owner}/{repo}/pulls

curl "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls?state=opened&source_branch=${branch_name}&access_token=${token}"
```

### 批量提交评论脚本

```bash
#!/bin/bash
OWNER="owner"
REPO="repo"
PR_NUMBER="123"
TOKEN="$GITCODE_TOKEN"
COMMENTS_FILE="comments.json"

cat "$COMMENTS_FILE" | jq -c '.[]' | while read -r comment; do
    body=$(echo "$comment" | jq -r '.body')
    path=$(echo "$comment" | jq -r '.path')
    position=$(echo "$comment" | jq -r '.position')

    echo "提交评论: $path:$position"

    response=$(curl -s -w "\n%{http_code}" -X POST \
        "https://api.gitcode.com/api/v5/repos/$OWNER/$REPO/pulls/$PR_NUMBER/comments?access_token=$TOKEN" \
        -H 'Content-Type: application/json' \
        -d "{\"body\": \"$body\", \"path\": \"$path\", \"position\": $position}")

    http_code=$(echo "$response" | tail -n1)
    if [ "$http_code" -eq 201 ]; then
        echo "  成功"
    else
        echo "  失败: HTTP $http_code"
        echo "$response" | head -n-1
    fi

    sleep 0.5
done
```

---

## 2. Issue 相关 API

### 获取 Issue 详情

```bash
GET /repos/{owner}/{repo}/issues/{number}

curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/issues/${issue_number}?access_token=${token}" \
  --connect-timeout 60 --max-time 180
```

**响应字段**：

| 字段 | 说明 |
|------|------|
| `title` | Issue 标题 |
| `body` | Issue 正文（包含模板和待填写内容） |
| `labels` | 标签列表 |
| `state` | 状态（open/closed） |
| `issue_state` | 仓库自定义流程状态名称 |
| `issue_state_detail.id` | 自定义流程状态 ID；只作观测，不得跨仓库硬编码 |

### 创建 Issue

```bash
POST /repos/{owner}/{repo}/issues

curl -X POST 'https://api.gitcode.com/api/v5/repos/{owner}/{repo}/issues?access_token={token}' \
-H 'Content-Type: application/json' \
-H 'Accept: application/json' \
-d '{
  "title": "Issue 标题",
  "body": "Issue 描述内容",
  "labels": "requirement",
  "assignees": "username"
}'
```

**参数说明**：

| 参数 | 必填 | 类型 | 说明 |
|------|:----:|------|------|
| `title` | Y | string | Issue 标题（最大 255 字符） |
| `body` | Y | string | Issue 描述（支持 Markdown） |
| `labels` | | **string** | 标签名称，**单个字符串**（如 `"requirement"`），不支持数组格式 |
| `assignees` | | **string** | 指派用户名，**单个字符串**（如 `"username"`），不支持数组格式 |
| `milestone` | | int | 里程碑 ID |

> **重要**：GitCode API 的 `labels` 和 `assignees` 参数与 GitHub API 不同，**必须使用字符串格式**，不支持 JSON 数组。传入数组（如 `["bug", "enhancement"]`）会导致 `400 BAD_REQUEST` 错误。多个标签可用逗号分隔（如 `"bug,enhancement"`）。

### 获取 Issue 列表

```bash
GET /repos/{owner}/{repo}/issues

curl 'https://api.gitcode.com/api/v5/repos/{owner}/{repo}/issues?access_token={token}&state=open'
```

### 更新 Issue

```bash
PATCH /repos/{owner}/{repo}/issues/{number}

curl -X PATCH 'https://api.gitcode.com/api/v5/repos/{owner}/{repo}/issues/{number}?access_token={token}' \
-H 'Content-Type: application/json' \
-d '{
  "title": "更新后的标题",
  "body": "更新后的描述"
}'
```

### 关闭 Issue

```bash
PATCH /repos/{owner}/{repo}/issues/{number}

curl -X PATCH 'https://api.gitcode.com/api/v5/repos/{owner}/{repo}/issues/{number}?access_token={token}' \
-H 'Content-Type: application/json' \
-d '{"state": "closed"}'
```

### 读取与更新自定义 Issue 状态

GitCode 的核心 `state`（open/closed）和看板自定义 `issue_state` 是两层状态。切换自定义
状态时，先实时读取仓库所属组的可用状态目录，再用名称完成迁移；状态 ID 会随组配置变化，
禁止硬编码。

```text
GET https://web-api.gitcode.com/api/v2/groups/{owner}/issue-extend/issue_extend_status_list
GET https://web-api.gitcode.com/issuepr/api/v1/issue/{project_id}/issue-extend/info/{number}
PUT https://web-api.gitcode.com/issuepr/api/v1/issue/{project_id}/issue-extend/status-flow/{number}
Content-Type: application/json

{"status_before":"<current_status>","status_current":"<target_status>"}
```

`project_id` 从标准 Issue 详情的 `repository.id` 获取。写前必须把当前/目标状态名称和是否
同时更新核心 state 纳入授权预览；PUT 后重新 GET extend info，只有名称与目标一致才算
成功。若还需要更新核心 `state`，使用上面的标准 PATCH 接口并单独回查。

### 提交 Issue 评论

```bash
POST /repos/{owner}/{repo}/issues/{number}/comments

curl -X POST "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/issues/${issue_number}/comments" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -d "{
    \"access_token\": \"${token}\",
    \"body\": \"${comment_body}\"
  }" \
  --connect-timeout 60 --max-time 300
```

---

## 3. 仓库内容 API

### 获取仓库内容

用于获取模板文件：

```bash
GET /repos/{owner}/{repo}/contents/{path}

curl 'https://api.gitcode.com/api/v5/repos/{owner}/{repo}/contents/{path}?ref={branch}&access_token={token}'
```

**返回示例（目录）**：

```json
[
  {
    "name": "bug_report.md",
    "path": ".github/ISSUE_TEMPLATE/bug_report.md",
    "type": "file",
    "download_url": "https://gitcode.com/..."
  }
]
```

**返回示例（文件）**：

```json
{
  "name": "bug_report.md",
  "content": "LS0tCm5hbWU6IEJ1ZyByZXBvcnQK...",
  "encoding": "base64"
}
```

需要解码：`echo "${content}" | base64 -d`

### 常见模板位置

| 类型 | 路径 |
|------|------|
| GitHub PR 模板 | `.github/PULL_REQUEST_TEMPLATE.md` |
| GitCode PR 模板 | `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` |
| GitHub Issue 模板 | `.github/ISSUE_TEMPLATE/*.md` |
| GitCode Issue 模板 | `.gitcode/ISSUE_TEMPLATE/` |
| GitLab 合并请求模板 | `.gitlab/merge_request_templates/*.md` |

---

## 4. 错误处理

### HTTP 状态码

| 状态码 | 说明 | 处理方式 |
|--------|------|----------|
| 200 | 成功（GET/PATCH） | 正常处理响应 |
| 201 | 创建成功（POST） | 正常处理响应 |
| 401 | Token 无效或过期 | 提示用户提供新 token |
| 403 | 无权限访问 | 确认用户是否有仓库写权限 |
| 404 | 资源不存在 | 确认链接是否正确 |
| 409 | 冲突 | PR 可能已更新，重新获取代码 |
| 422 | 参数验证失败 | 检查参数格式和有效性 |
| 429 | 请求频率限制 | 按 `Retry-After`/reset 设置共享 cooldown 后重试；总计最多 3 次 attempt |

### 错误响应格式

```json
{
  "error_code": 422,
  "error_code_name": "VALIDATION_ERROR",
  "error_message": "错误描述",
  "trace_id": "abc123def456"
}
```

### 常见错误处理

#### 行号无效 (422)

```
原因：提交的行号在文件中不存在
解决：重新使用 Read 工具或 git show 获取正确的行号
```

```bash
# 重新获取行号
git show pr_{number}:{file_path} | grep -n "代码片段"
```

#### 文件路径错误 (404)

```
原因：文件路径不正确或文件不在 PR 变更中
解决：确认文件在 PR 的变更文件列表中
```

#### Token 权限不足 (403)

```
原因：Token 没有写权限
解决：提示用户生成具有 repo 权限的新 token
```

#### Title 过长 (422)

```
原因：标题超过 255 字符
解决：截断标题或重新生成更简洁的标题
```

#### Label 不存在 (422)

```
原因：指定的标签在仓库中不存在
解决：先获取可用标签列表，或跳过标签参数
```

---

## 5. 最佳实践

### 提交策略

1. **逐条提交**：避免批量提交失败导致全部丢失
2. **按优先级排序**：先提交高严重程度的问题
3. **添加延迟**：避免请求频率限制（建议 `sleep 0.5`）
4. **记录结果**：跟踪每条评论的提交状态

### 内容生成

1. **标题简洁**：控制在 50-80 字符内
2. **结构清晰**：使用 Markdown 标题分隔不同部分
3. **关联明确**：包含 PR 链接或编号
4. **可追溯**：记录变更原因和影响范围

### 安全考虑

1. **Token 保护**：不在日志中输出 token
2. **敏感信息**：不在评论/Issue 中包含敏感信息
3. **权限最小化**：只申请必要的权限

---

## 6. 命令速查

```bash
# 获取 PR 详情
curl "https://api.gitcode.com/api/v5/repos/$OWNER/$REPO/pulls/$PR?access_token=$TOKEN"

# 获取 PR 文件
curl "https://api.gitcode.com/api/v5/repos/$OWNER/$REPO/pulls/$PR/files?access_token=$TOKEN"

# 获取 PR Commits
curl "https://api.gitcode.com/api/v5/repos/$OWNER/$REPO/pulls/$PR/commits?access_token=$TOKEN"

# 提交行内评论
curl -X POST "https://api.gitcode.com/api/v5/repos/$OWNER/$REPO/pulls/$PR/comments?access_token=$TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"body":"评论内容","path":"文件路径","position":行号}'

# 获取已有评论
curl "https://api.gitcode.com/api/v5/repos/$OWNER/$REPO/pulls/$PR/comments?access_token=$TOKEN"

# 更新 PR
curl -X PATCH "https://api.gitcode.com/api/v5/repos/$OWNER/$REPO/pulls/$PR?access_token=$TOKEN" \
  --data-urlencode "title=PR标题" --data-urlencode "body=PR描述"

# 创建 PR
curl -X POST "https://api.gitcode.com/api/v5/repos/$OWNER/$REPO/pulls?access_token=$TOKEN" \
  -d "title=标题" -d "body=描述" -d "head=user:branch" -d "base=master"

# 获取 Issue 详情
curl "https://api.gitcode.com/api/v5/repos/$OWNER/$REPO/issues/$ISSUE?access_token=$TOKEN"

# 创建 Issue
curl -X POST "https://api.gitcode.com/api/v5/repos/$OWNER/$REPO/issues?access_token=$TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"title":"标题","body":"描述"}'

# 提交 Issue 评论
curl -X POST "https://api.gitcode.com/api/v5/repos/$OWNER/$REPO/issues/$ISSUE/comments?access_token=$TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"body":"评论内容"}'

# 获取仓库内容/模板
curl "https://api.gitcode.com/api/v5/repos/$OWNER/$REPO/contents/{path}?access_token=$TOKEN"

# 获取当前 token 对应账号
curl "https://api.gitcode.com/api/v5/user?access_token=$TOKEN"

# 获取账号绑定的全部邮箱
curl "https://api.gitcode.com/api/v5/emails?access_token=$TOKEN"
```

---

## 7. 用户账号 API

用于校验 token 对应账号的身份信息（如 PR 提交前比对 git `user.email` 是否与账号绑定邮箱一致）。

### 获取当前 token 对应的账号

```bash
GET /user

curl "https://api.gitcode.com/api/v5/user?access_token=${token}"
```

**关键返回字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `login` | string | 账号用户名 |
| `name` | string | 显示名 |
| `email` | string | 账号主邮箱（**仅一个**，公开邮箱） |

### 获取账号绑定的全部邮箱

```bash
GET /emails

curl "https://api.gitcode.com/api/v5/emails?access_token=${token}"
```

**返回示例**：

```json
[{"email": "user@example.com", "state": "confirmed"}]
```

| 字段 | 说明 |
|------|------|
| `email` | 绑定邮箱 |
| `state` | `confirmed`（已验证）/ 未确认 |

> **用途**：commit 关联到 GitCode 主页的依据是「commit email ∈ 账号绑定邮箱」。比对时建议大小写不敏感。
> **注意**：此端点可能依赖 token 的 user/email scope，部分 token 会返回 401/403——拿不到时应**降级跳过**（视为「无法校验」），不要阻断主流程。
