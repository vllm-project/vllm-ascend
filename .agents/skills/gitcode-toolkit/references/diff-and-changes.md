# Diff 变更统计

使用 `git diff` 获取 PR 变更文件列表和统计信息。

---

## 1. merge-base 模式

**适用 skill**：gitcode-pr-handler、gitcode-issue-gen

先通过 merge-base 计算出基准 commit，再与 PR 分支比较。

```bash
# 前提：已计算 MERGE_BASE
# MERGE_BASE=$(git merge-base base_branch pr_{pr_number})
```

### 变更统计

```bash
# 每个文件的新增/删除行数
git diff --numstat $MERGE_BASE pr_{pr_number}

# 文件变更类型（新增/修改/删除/重命名）
git diff --name-status $MERGE_BASE pr_{pr_number}

# 变更文件列表和统计
git diff --stat $MERGE_BASE pr_{pr_number}
```

### 按类型筛选

```bash
# 新增文件
git diff --diff-filter=A --name-only $MERGE_BASE pr_{pr_number}

# 修改文件
git diff --diff-filter=M --name-only $MERGE_BASE pr_{pr_number}

# 删除文件
git diff --diff-filter=D --name-only $MERGE_BASE pr_{pr_number}
```

### 单文件 diff

```bash
# 查看某个文件的详细变更
git diff $MERGE_BASE pr_{pr_number} -- {file_path}
```

---

## 2. triple-dot 模式

**适用 skill**：PR 创建流程（见 [pr-creation-workflow.md](pr-creation-workflow.md)）

使用 triple-dot 语法 `origin/${BASE_BRANCH}...HEAD` 比较分支差异。

```bash
# 前提：已检出 PR 分支并确定 BASE_BRANCH
```

### 变更统计

```bash
# 每个文件的新增/删除行数
git diff --numstat "origin/${BASE_BRANCH}...HEAD"

# 文件变更类型（新增/修改/删除/重命名）
git diff --name-status "origin/${BASE_BRANCH}...HEAD"

# 变更文件列表和统计
git diff --stat "origin/${BASE_BRANCH}...HEAD"
```

### 按类型筛选

```bash
# 新增文件
git diff "origin/${BASE_BRANCH}...HEAD" --diff-filter=A --name-only

# 修改文件
git diff "origin/${BASE_BRANCH}...HEAD" --diff-filter=M --name-only

# 删除文件
git diff "origin/${BASE_BRANCH}...HEAD" --diff-filter=D --name-only
```

### 单文件 diff

```bash
# 查看某个文件的详细变更
git diff "origin/${BASE_BRANCH}...HEAD" -- path/to/file.py
```

---

## 3. 两种模式对比

| 特性 | merge-base 模式 | triple-dot 模式 |
|------|----------------|----------------|
| 语法 | `$MERGE_BASE pr_{n}` | `origin/${BASE_BRANCH}...HEAD` |
| 前提条件 | 需先计算 merge-base | 需已检出 PR 分支 + 确定 BASE_BRANCH |
| 精确度 | 精确到 commit 级别 | 同样精确 |
| 适用场景 | 需要精确变更范围（review、描述生成） | 本地已有 PR 分支（文档生成、PR 创建） |
