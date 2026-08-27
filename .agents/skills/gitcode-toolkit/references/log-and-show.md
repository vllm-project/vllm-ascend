# Log 与 Show

使用 `git log` 提取 commit 元信息，使用 `git show` 读取文件内容。

---

## 1. git log — 元信息提取

### 最新 commit 信息

```bash
# 最新 commit 标题（PR 标题）
PR_TITLE=$(git log -1 --pretty=format:"%s")

# 最新 commit 作者（PR 作者）
PR_AUTHOR=$(git log -1 --pretty=format:"%an")
```

### PR commit 历史

```bash
# 相对于 base 分支的 commit 历史（oneline）
git log --oneline "origin/${BASE_BRANCH}..HEAD"
```

### commit 标题列表（PR 创建流程）

```bash
# 相对于 master 的 commit 标题列表（不含 merge commit）
git log master..HEAD --pretty=format:"%s" --no-merges
```

### commit 标题+正文（PR 创建流程）

```bash
# 相对于 master 的 commit 标题和正文（不含 merge commit）
git log master..HEAD --pretty=format:"%s%n%b" --no-merges
```

---

## 2. git show — 读取文件内容

### 读取 HEAD 版本文件

```bash
git show HEAD:path/to/file
```

### 读取 base 分支文件

```bash
git show origin/${BASE_BRANCH}:path/to/file
```

### 行号验证

```bash
# 读取 PR 分支的文件并显示行号（用于验证评论行号）
git show pr_{n}:{file_path} | cat -n
```
