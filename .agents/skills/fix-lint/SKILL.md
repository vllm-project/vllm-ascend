---
name: fix-lint
description: "自动进行 vllm-ascend 仓库的 lint 更改并提交"
---

# fix-lint Skill

这个技能用于自动执行 vllm-ascend 仓库的 lint 更改并提交。

## 执行步骤

### 1. 激活 conda base 环境

```bash
conda activate base
```

### 2. 注释 actionlint 相关内容

编辑 `.pre-commit-config.yaml` 文件，将 actionlint 相关的配置注释掉：

```yaml
# - repo: https://github.com/rhysd/actionlint
#   rev: v1.7.7
#   hooks:
#   - id: actionlint
#     exclude: '.*\.github/workflows/scripts/.*\.ya?ml$'
```

### 3. 执行 format.sh

在当前目录下执行：

```bash
bash format.sh
```

### 4. 取消注释 actionlint

将第 2 步中注释的内容恢复，取消注释。

### 5. 查看 lint 更改

根据仓库内的 git 记录，使用以下命令查看 lint 更改了哪些文件：

```bash
# 查看未暂存的更改
git diff --name-only

# 查看已暂存的更改
git diff --cached --name-only

# 查看所有更改的文件
git status
```

### 6. 提交更改

执行 git commit 提交更改：

```bash
git add -A
git commit -sm "Fix lint"
```

### 7. 报告结果

向用户报告：
- lint 执行过程中修改了哪些文件
- commit 的具体内容