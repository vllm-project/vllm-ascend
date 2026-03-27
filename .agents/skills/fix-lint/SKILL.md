---
name: fix-lint
description: "自动进行 vllm-ascend 仓库的 lint 更改并提交"
---

# fix-lint Skill

这个技能用于自动执行 vllm-ascend 仓库的 lint 更改并提交。

## 执行步骤

### 1. 激活 conda base 环境（如有需要）

### 2. 先提交当前状态

```bash
git commit -sm "Before fix lint"
```

### 3. 注释 actionlint 相关内容

编辑 `.pre-commit-config.yaml` 文件，将 actionlint 相关的配置注释掉。

### 4. 执行 format.sh

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/pre-commit run --all-files
```

### 5. 恢复 actionlint 配置（取消注释）

### 6. 检查是否有实际的 lint 更改

```bash
git diff --name-only
```

如果没有任何文件被修改，则：
- 恢复 `.pre-commit-config.yaml` 的更改
- 报告 "没有需要提交的 lint 更改"
- 结束执行

### 7. 如果有更改，提交

```bash
git add -A
git commit -sm "Fix lint"
```

### 8. 报告结果

向用户报告 lint 执行的结果。