# PR 创建工作流

从 fork 仓库向上游 `cann` 组织仓库创建 Pull Request 的标准步骤。作为 `gitcode-issue-handler` 等 skill 的 PR 创建子流程被引用。

---

## 流程总览

```
1. 获取信息   → 分支名、commit历史、目标仓库
2. 获取模板   → 从目标仓库获取 PR 模板
3. 分析填充   → 分析 commit 内容，自动填充模板
4. 用户确认   → 展示填充后的模板，等待用户确认/修改
5. 校验身份   → 校验 git user.name / user.email 已配置
6. 推送分支   → 确保分支已推送到 origin
7. 创建 PR    → 调用 GitCode API 创建 PR
8. 记录日志   → 保存操作日志
```

> **前置校验**：进入 Step 6 推送之前，**必须**确认 git 提交身份（`user.name` / `user.email`）已配置——这是 PR 提交流程的硬性前置条件。缺失时立刻停下来问用户，不要带着空身份往下走。详见下方 Step 5 与 [env-check.md](env-check.md) 的「Git 提交用户信息」。

---

## Step 1: 获取信息

**必需参数**

| 参数 | 说明 | 获取方式 |
|------|------|----------|
| 分支名 | 源分支名称 | 从当前 git 分支获取或用户指定 |
| commit 历史 | 用于分析生成 PR 内容 | git log 获取 |
| 变更文件列表 | 用于推断模板字段 | git diff 获取 |

**默认配置**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| 上游仓库 | `cann` | 目标组织名称 |
| 目标分支 | `master` | 上游仓库的目标分支 |
| 用户仓库 | 当前 git 配置 | 从 git remote 获取 |

下文的 `origin` / `master` 是直接调用方的兼容默认示例。长流程编排器如果已经根据
Issue URL、manifest 或已验证的 remote 得到精确的 `fork_remote`、`base_ref` 和
`base_branch`，后续命令必须使用这些调用方值；不得为了套用示例而改回
`origin` / `master`。这项覆盖不改变未传这些字段的既有工作流。

### 1.1 检测 Remote 配置

```bash
git remote -v
```

自动识别逻辑：
- 上游仓库：URL 中包含 `cann/` 的 remote
- Fork 仓库：其他 remote（非 cann 组织）

### 1.2 如果仍无法自动识别

展示候选 remote 及 URL，请用户选择 fork 仓库。不得在无法确定时静默
选择可能的推送目标。

### 1.3 获取当前信息

```bash
current_branch=$(git branch --show-current)
username=$(git remote get-url ${fork_remote} | sed -E 's|.*[:/]([^/]+)/[^/]+\.git|\1|')
repo=$(git remote get-url ${fork_remote} | sed -E 's|.*[:/][^/]+/([^/]+)\.git|\1|')

git log master..HEAD --pretty=format:"%s" --no-merges
git diff master...HEAD --name-only
git log master..HEAD --pretty=format:"%s%n%b" --no-merges
```

---

## Step 2: 获取 PR 模板

模板文件按优先级：

1. `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`
2. `.gitcode/PULL_REQUEST_TEMPLATE.md`
3. `PULL_REQUEST_TEMPLATE.md`

```bash
git show origin/master:.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md
git show origin/master:.gitcode/PULL_REQUEST_TEMPLATE.md
```

**默认模板**（仓库无模板时使用）：

```markdown
## 描述
<!--详细描述改动-->

## 关联的Issue
<!--Issue链接或问题单单号-->

## 测试
<!--测试验证内容-->

## 文档更新
<!--文档更新说明-->

## 类型标签
- [ ] Bug修复
- [ ] 新特性
- [ ] 性能优化
- [ ] 文档更新
- [ ] 其他
```

---

## Step 3: 分析并填充模板

**信息来源映射**

| 模板字段 | 自动获取方式 | 备选方案 |
|----------|--------------|----------|
| **描述** | 从 commit messages 汇总生成 | 用户输入 |
| **关联的Issue** | 从 commit message 提取 `#数字` 或 `fix #数字` | 用户输入 |
| **测试** | 检测 tests/ 目录变更，提示用户填写 | 用户输入 |
| **文档更新** | 检测 docs/、README.md 等文件变更 | 用户输入 |
| **类型标签** | 从 PR 标题前缀推断 | 用户选择 |

**类型标签推断规则**

| 标题前缀 | 类型标签 |
|----------|----------|
| `fix:` | Bug修复 |
| `feat:` | 新特性 |
| `perf:` | 性能优化 |
| `docs:` | 文档更新 |
| `refactor:` / `test:` / `chore:` | 其他 |

**分析脚本要点**

```bash
commits=$(git log master..HEAD --pretty=format:"%s" --no-merges)
issues=$(git log master..HEAD --pretty=format:"%s %b" --no-merges | grep -oE '#[0-9]+' | sort -u)
test_files=$(git diff master...HEAD --name-only | grep -E '(tests?/|_test\.|_spec\.)')
doc_files=$(git diff master...HEAD --name-only | grep -E '(docs?/|README|\.md$)')
first_commit=$(git log master..HEAD --pretty=format:"%s" --no-merges | head -1)
```

---

## Step 4: 用户确认

用 AskUserQuestion 展示填充后的模板预览，选项：

1. **确认创建** - 使用当前模板内容创建 PR
2. **修改模板** - 用户手动编辑
3. **取消操作** - 终止流程

确认时展示：PR 标题、源分支 → 目标分支、填充后的模板内容。

如果调用方已显式采用
[authorization-contract.md](authorization-contract.md)，并提供当前会话中覆盖
当前准确目标和完整内容的合法检查点，可复用该确认而不重复询问。
不得根据调用方名称或“处理/修复 Issue”等宽泛请求推断已经授权创建 PR。调用方不传
检查点证据时，继续使用上述原有确认流程。

---

## Step 5: 校验 git 提交身份

推送前对 git 提交身份做两层校验：**5.1 是否已配置**（硬性，缺失即阻断）和 **5.2 email 是否与 GitCode 账号绑定一致**（建议，不一致仅告警）。`git push` 本身不会因身份缺失或 email 不匹配而失败，但 commit author 是公开字段：缺失/配错人，或 email 没对上账号绑定邮箱，会导致 commit 挂错身份、或在 GitCode 上显示为「未关联用户」，事后难补救。

### 5.1 校验是否已配置（硬性，缺失即阻断）

```bash
NAME=$(git config user.name 2>/dev/null)
EMAIL=$(git config user.email 2>/dev/null)
if [ -z "$NAME" ] || [ -z "$EMAIL" ]; then
  echo "MISSING: git author identity (user.name / user.email)"
fi
```

- 两项都已配置 → 展示读到的 `Name <email>` 让用户一眼确认是不是本次想用的身份，确认无误后进入 5.2。
- 任一缺失 → **立即停下来用 AskUserQuestion 询问**，不要继续 push：

```
问题: 未检测到 git 提交身份（user.name / user.email），无法安全提交 PR，请提供：
选项:
  - 用我下面提供的 name 和 email（在下一条消息中给出）
  - 已在别处配置好，让我重新读取一次
  - 取消本次操作
```

拿到用户提供的值后**只在当前工作目录写 local 配置**，禁止改全局 `~/.gitconfig`：

```bash
git -C "$WORK_DIR" config user.name  "$NAME"
git -C "$WORK_DIR" config user.email "$EMAIL"
```

> 完整规则（global 继承、反模式、禁止用 `--author=` / `-c user.name=` inline 绕过）详见 [env-check.md](env-check.md) 的「5. Git 提交用户信息」。

### 5.2 校验 email 是否与 GitCode 账号绑定一致（建议，不一致仅告警）

git 里配的 `user.email` 只是本地字符串，只有当它**等于 token 对应 GitCode 账号的某个已绑定邮箱**时，commit 才会关联到该用户主页；否则 PR 仍能提，但 commit 显示为「未关联用户」。用已有的 token 调账号绑定邮箱接口做一次比对：

```bash
# 拉取当前 token 账号的全部绑定邮箱（小写化）
BOUND=$(curl -s "https://api.gitcode.com/api/v5/emails?access_token=${token}" \
  --connect-timeout 20 --max-time 40 \
  | python3 -c "import sys,json;
try:
  print('\n'.join(e['email'].lower() for e in json.load(sys.stdin)))
except Exception:
  pass")

if [ -z "$BOUND" ]; then
  echo "SKIP: 无法获取账号绑定邮箱（token 缺 user/email scope 或接口不可用），跳过一致性校验"
elif printf '%s\n' "$BOUND" | grep -qxF "$(echo "$EMAIL" | tr 'A-Z' 'a-z')"; then
  echo "OK: git user.email 与 GitCode 账号绑定邮箱一致"
else
  echo "WARN: git user.email ($EMAIL) 不在账号绑定邮箱中，commit 将不会关联到 GitCode 主页"
fi
```

> 备用端点：若只需账号主邮箱，可用 `GET /api/v5/user` 取单个 `email` 字段比对（详见 [gitcode-api.md](gitcode-api.md)）。

处理策略（**这是建议性校验，绝不硬阻断**）：

- `OK` → 一行通过提示，进入 Step 6。
- `SKIP`（接口不可用 / token 无 scope）→ 打印一行「已跳过 email 绑定校验」，**直接继续**，不打扰用户。
- `WARN`（不一致）→ 用 AskUserQuestion 提示风险，让用户决定，**默认不阻断**：

```
问题: git user.email 与 GitCode 账号绑定邮箱不一致，commit 将显示为「未关联用户」。如何处理？
选项:
  - 继续提交（接受 commit 不关联到我的主页）
  - 我改用账号绑定的邮箱（在下一条消息给出，仅写工作目录 local 配置）
  - 取消本次操作
```

---

## Step 6: 推送分支

```bash
git push -u origin ${branch_name}
git ls-remote --heads origin ${branch_name}
```

---

## Step 7: 创建 PR

先按 `head + base + state=opened` 查询已有 PR。找到匹配项时直接复用并记录 URL，避免
重试、恢复运行或上次响应丢失造成重复创建；只有确认不存在时才 POST。

**API**

```
POST https://api.gitcode.com/api/v5/repos/{upstream_owner}/{upstream_repo}/pulls
```

| 参数 | 类型 | 说明 |
|------|------|------|
| access_token | string | GitCode API Token |
| title | string | PR 标题 |
| body | string | PR 描述内容（填充后的模板） |
| head | string | 源分支，格式: `{username}:{branch}` |
| base | string | 目标分支，通常为 `master` |

```bash
curl -X POST "https://api.gitcode.com/api/v5/repos/${upstream_owner}/${upstream_repo}/pulls" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "access_token=${token}" \
  --data-urlencode "title=${pr_title}" \
  --data-urlencode "body=${pr_body}" \
  --data-urlencode "head=${username}:${branch_name}" \
  --data-urlencode "base=master" \
  --connect-timeout 30
```

**head 参数格式**：从 fork 仓库向上游创建 PR 时，`head` 必须是 `{fork用户名}:{分支名}`，例如 `your-username:fix/xxx`。当 fork 改过名时建议用更稳的 `{fork_owner}/{fork_repo}:{branch}` 格式。

**成功响应**：接受 API 实际返回的 HTTP 200 或 201，但必须校验 JSON 中存在 PR iid
和 `web_url`，再 GET 回查源分支、目标分支和 opened 状态。HTTP 成功但字段/回查不匹配
仍视为失败；422 提示已存在时重新查询并复用已有 PR。

```json
{
  "id": 8395063,
  "iid": 1564,
  "title": "fix: 修复异构安装路径问题",
  "state": "opened",
  "web_url": "https://gitcode.com/cann/ops-math/merge_requests/1564",
  "source_branch": "fix/heterogeneous-install-path",
  "target_branch": "master"
}
```

> 错误码处理详见 [gitcode-api.md](gitcode-api.md)。

---

## Step 8: 记录日志

日志文件命名：`logs/pr-create_{YYYYMMDD}_{HHMMSS}.log`。日志格式详见 [logging-conventions.md](logging-conventions.md)。

---

## 常见问题

**Q1: PR 创建失败，提示 "head not found"**：分支未推送到 origin，先 `git push -u origin ${branch_name}`。

**Q2: PR 创建失败，提示 "Another open merge request already exists"**：该分支已有未合并 PR，从 API 返回里取已有 PR 链接。

**Q3: 模板获取失败**：仓库无模板时退回默认模板（见 Step 2）。

**Q4: 查看已有 PR**：

```bash
curl "https://api.gitcode.com/api/v5/repos/${upstream_owner}/${upstream_repo}/pulls?state=opened&source_branch=${branch_name}&access_token=${token}"
```
