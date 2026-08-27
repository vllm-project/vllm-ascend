# Remote 与分支管理

Remote 管理、分支查询、推送等 Git 操作。

---

## 1. Remote 管理

### 列出所有 remote

```bash
git remote -v
```

### 获取 remote URL

```bash
git remote get-url ${remote}
```

**自动识别 fork 和上游**（gitcode-toolkit PR 创建流程）：
- 上游仓库：URL 中包含 `cann/` 的 remote
- Fork 仓库：其他 remote（非 cann 组织）

---

## 2. 分支查询

### 获取当前分支名

```bash
git branch --show-current
```

### 检查远程分支是否存在

```bash
git ls-remote --heads origin ${branch}
```

---

## 3. 推送分支

```bash
# 推送当前分支到 origin 并设置上游
git push -u origin ${branch}
```

> 用于 gitcode-toolkit 的 PR 创建流程中确保分支已推送到 origin，否则 API 创建 PR 会提示 "head not found"。
