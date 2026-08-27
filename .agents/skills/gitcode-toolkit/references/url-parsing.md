# URL 格式识别与解析

识别和解析 GitCode 平台的 PR/Issue URL。

---

## PR URL 格式

支持以下三种格式：

| 格式 | 示例 |
|------|------|
| `/pull/{n}` | `https://gitcode.com/{owner}/{repo}/pull/123` |
| `/pulls/{n}` | `https://gitcode.com/{owner}/{repo}/pulls/123` |
| `/merge_requests/{n}` | `https://gitcode.com/{owner}/{repo}/merge_requests/123` |

## Issue URL 格式

| 格式 | 示例 |
|------|------|
| `/issues/{n}` | `https://gitcode.com/{owner}/{repo}/issues/456` |

---

## 解析规则

从 URL 中提取以下信息：

| 提取项 | 说明 |
|--------|------|
| `owner` | 组织或用户名 |
| `repo` | 仓库名 |
| `pr_number` / `issue_number` | PR/Issue 编号 |

### 示例

```
输入: https://gitcode.com/myorg/myproject/pulls/123
输出: owner=myorg, repo=myproject, pr_number=123

输入: https://gitcode.com/cann/opbase/issues/91
输出: owner=cann, repo=opbase, issue_number=91
```

---

## 错误处理

| 错误场景 | 处理方式 |
|----------|----------|
| URL 格式不匹配 | 提示用户提供正确的 GitCode PR/Issue 链接 |
| 链接无法访问 | 检查是否为私有仓库，提示用户确认权限 |
