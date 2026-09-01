# Issue 创建工作流

创建 GitCode Issue 的标准步骤，供其他需要创建 Issue 的工作流/skill 引用。

---

## 流程总览

```
1. 获取信息   → 目标仓库、问题数据
2. 获取模板   → 从目标仓库查询 Issue 模板
3. 选择模板   → （业务层）问题类型→模板映射
4. 填充内容   → （业务层）根据问题内容生成 Issue body
5. 用户确认   → 展示 Issue 内容，等待确认
6. 创建 Issue → 调用 GitCode API 创建 Issue
7. 记录日志   → 保存操作日志
```

> **Step 3、4 为业务层**，由调用方 skill 根据业务场景实现（如扫描报告→模板选择、问题列表→Issue内容）。infra 仅提供 Step 1、2、5、6、7 的通用能力。

---

## Step 1: 获取信息

**必需参数**

| 参数 | 说明 | 获取方式 |
|------|------|----------|
| 目标仓库 | Issue 创建目标仓库 | 用户指定或从问题数据推断 |
| 问题数据 | 问题列表、扫描报告 | 用户提供 |

**仓库路径格式**

| 格式 | 示例 |
|------|------|
| URL | `https://gitcode.com/cann/ops-math` |
| 项目路径 | `cann/ops-math` |
| 仓库名 | `ops-math`（默认组织为 `cann`） |

**解析命令**

```bash
# 从 URL 提取项目路径
project_path=$(echo "${url}" | sed -E 's|.*gitcode\.com/([^/]+/[^/]+).*|\1|')

# 从仓库名构建项目路径（默认 cann 组织）
project_path="cann/${repo_name}"
```

---

## Step 2: 获取 Issue 模板

**模板查询优先级**

| 优先级 | 路径 | 说明 |
|:---:|------|------|
| 1 | `.gitcode/ISSUE_TEMPLATE/*.zh-CN.yml` | GitCode 中文模板（优先） |
| 2 | `.gitcode/ISSUE_TEMPLATE/*.yml` | GitCode YAML 表单模板 |
| 3 | `.gitcode/ISSUE_TEMPLATE/*.md` | GitCode Markdown 模板 |
| 4 | `.github/ISSUE_TEMPLATE/*.yml` | GitHub YAML 表单模板（兼容） |
| 5 | `.github/ISSUE_TEMPLATE/*.md` | GitHub Markdown 模板（兼容） |
| 6 | **预设模板** | 仓库无模板时使用 |

**API 查询命令**

```bash
owner="cann"
repo="ops-math"
token="${GITCODE_TOKEN}"

# 查询模板目录列表（.gitcode）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.gitcode/ISSUE_TEMPLATE?access_token=${token}"

# 查询模板目录列表（.github）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.github/ISSUE_TEMPLATE?access_token=${token}"

# 获取单个模板内容（返回 Base64 编码）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.gitcode/ISSUE_TEMPLATE/bug_report.yml?access_token=${token}"

# 解码模板内容
content=$(curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.gitcode/ISSUE_TEMPLATE/bug_report.yml?access_token=${token}" | jq -r '.content')
echo "${content}" | base64 -d
```

**模板类型解析**

**YAML 表单模板**（`.yml` 文件）：

```yaml
name: Bug Report
description: 报告一个缺陷
title: "[Bug]: "
labels: ["bug-report"]
body:
  - type: textarea
    id: description
    attributes:
      label: 问题描述
```

解析方式：`yaml.safe_load(content)`，提取 `name`、`title`、`labels`、`body`。

**Markdown 模板**（`.md` 文件）：

```markdown
---
name: Bug Report
about: 报告一个缺陷
title: '[Bug]: '
labels: bug-report
---

**问题描述**
{问题描述内容}
```

解析方式：提取 `---` 之间的 YAML front-matter，获取 `name`、`title`、`labels`。

**预设模板列表**（仓库无模板时 fallback）

| 模板类型 | 标签 | 适用场景 |
|---------|------|---------|
| Bug-Report | `bug-report` | 缺陷反馈 |
| Documentation | `documentation` | 文档问题 |
| Requirement | `requirement` | 需求建议 |
| Question | `question` | 咨询讨论 |
| Blank | 无 | 通用问题 |

**Bug-Report 预设模板**

```markdown
Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

### Describe the current behavior / 问题描述

{问题描述}

### Environment / 环境信息

**软件环境**:
- CANN 版本: {版本}
- 操作系统: {OS}

**硬件环境**:
- NPU 型号: {芯片型号}

### Steps to reproduce the issue / 重现步骤

{重现步骤}

### Describe the expected behavior / 预期结果

{预期结果}

### Related log / screenshot / 日志 / 截图

{日志/截图}
```

**Documentation 预设模板**

```markdown
Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

### Document Link（文档链接）

{文档链接}

### Issues Section（问题文档片段）

{问题片段}

### Existing Issues（存在的问题）

{问题描述}

### Suggested Fix（修复建议）

{修复建议}
```

**Requirement 预设模板**

```markdown
Thanks for sending an requirement! Please fill in the following template to help quickly solve your problem.

### Background（背景信息）

{背景}

### Benefit / Necessity（价值/作用）

{价值说明}

### Design（设计方案）

{设计方案}
```

---

## Step 3: 选择模板（业务层）

> ⚠️ **此步骤为业务层**，由调用方 skill 实现。infra 不提供具体实现。

**业务层职责**：
- 根据问题类型选择模板（如 UT缺失→Bug-Report、README缺失→Documentation）
- 确定模板标签、标题前缀

**示例映射**（由调用方 skill 定义）：

| 问题类型 | 模板类型 | 标签 |
|---------|---------|------|
| UT缺失 | Bug-Report | bug-report |
| README缺失 | Documentation | documentation |
| 功能需求 | Requirement | requirement |

---

## Step 4: 填充内容（业务层）

> ⚠️ **此步骤为业务层**，由调用方 skill 实现。infra 不提供具体实现。

**业务层职责**：
- 根据问题数据生成 Issue 标题
- 根据模板字段填充 Issue body 内容
- 处理合并场景（多个问题合并为一个 Issue）

**标题格式建议**

| 模板类型 | 单问题标题 | 合并标题 |
|---------|-----------|---------|
| Bug-Report | `[Bug-Report]: {repo} {op_name} {简述}` | `[Bug-Report]: {repo} {简述}（{n}个算子）` |
| Documentation | `[Documentation]: {repo} {op_name} {简述}` | `[Documentation]: {repo} {简述}（{n}个算子）` |

---

## Step 5: 用户确认

用 AskUserQuestion 展示 Issue 内容预览，选项：
1. **确认创建** - 使用当前内容创建 Issue
2. **修改内容** - 用户手动编辑
3. **取消操作** - 终止流程

确认时展示：
- Issue 标题
- 目标仓库
- Issue body 内容预览
- 标签

---

## Step 6: 创建 Issue

**API**

```
POST https://api.gitcode.com/api/v5/repos/{owner}/{repo}/issues
```

| 参数 | 必填 | 类型 | 说明 |
|------|:----:|------|------|
| `title` | Y | string | Issue 标题（最大 255 字符） |
| `body` | Y | string | Issue 描述（支持 Markdown） |
| `labels` | | **string** | 标签名称，**单个字符串**（如 `"bug-report"`），不支持数组 |
| `assignees` | | **string** | 指派用户名，**单个字符串**，不支持数组 |

> **重要**：GitCode API 的 `labels` 和 `assignees` 必须使用字符串格式，不支持 JSON 数组。多个标签可用逗号分隔（如 `"bug,enhancement"`）。

**curl 命令**

```bash
token="${GITCODE_TOKEN}"

curl -X POST "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/issues?access_token=${token}" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "title": "${issue_title}",
    "body": "${issue_body}",
    "labels": "${labels}"
  }' \
  --connect-timeout 30
```

**使用 PRIVATE-TOKEN 方式**（更安全）

```bash
curl -X POST "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/issues" \
  -H "PRIVATE-TOKEN: ${token}" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "${issue_title}",
    "body": "${issue_body}",
    "labels": "${labels}"
  }'
```

**成功响应 (HTTP 201)**

```json
{
  "id": 123456,
  "iid": 42,
  "title": "[Bug-Report]: ops-math add CMake OPTYPE error",
  "state": "opened",
  "web_url": "https://gitcode.com/cann/ops-math/issues/42",
  "labels": [{"name": "bug-report"}]
}
```

**权限要求**

| 权限等级 | access_level | 能否创建 Issue |
|---------|:------------:|:-------------:|
| Guest | 10 | Web 可，API 不行 |
| Reporter | 20 | ✅ 可 |
| Developer | 30 | ✅ 可 |

**失败处理**

| 状态码 | 说明 | 处理方式 |
|--------|------|----------|
| 401 | Token 无效 | 提示用户提供新 token |
| 403 | 无权限 | 降级为手动提交（提供提交链接） |
| 404 | 项目不存在 | 确认仓库路径是否正确 |
| 422 | 参数验证失败 | 检查参数格式（labels 是否为字符串） |

**手动提交 fallback**

```bash
# 生成提交链接
submit_url="https://gitcode.com/${owner}/${repo}/issues/new"

# 如果有标签
submit_url="https://gitcode.com/${owner}/${repo}/issues/new?labels=${labels}"

echo "Issue 文件已生成，请手动提交："
echo "提交链接：${submit_url}"
echo "Issue 标题：${issue_title}"
echo "Issue 内容见文件：${issue_file}"
```

---

## Step 7: 记录日志

日志文件命名：`logs/issue-create_{YYYYMMDD}_{HHMMSS}.log`。日志格式详见 [logging-conventions.md](logging-conventions.md)。

---

## 常见问题

**Q1: 模板查询返回空**：仓库无 Issue 模板目录，使用预设模板（见 Step 2）。

**Q2: 创建失败，提示 403 Forbidden**：用户权限不足（Guest 级），提供手动提交链接。

**Q3: labels 参数报错**：必须使用字符串格式 `"bug-report"`，不能用数组 `["bug-report"]`。

**Q4: Issue body 过长**：GitCode 支持长内容，但如果超过限制，分多次 PATCH 更新。

**Q5: 查看已有 Issue**

```bash
curl "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/issues?state=opened&access_token=${token}"
```
