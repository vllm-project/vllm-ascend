---
name: fix-lint
description: "自动进行 vllm-ascend 仓库的 lint 更改并提交"
---

# fix-lint Skill

这个技能用于自动执行 vllm-ascend 仓库的 lint 更改并提交。

## 执行步骤（静默执行，不输出中间状态）

1. 提交当前状态：`git add -A && git commit -sm "Before fix lint"`
2. 注释 `.pre-commit-config.yaml` 中的 actionlint
3. 执行 lint：`/opt/homebrew/Caskroom/miniconda/base/bin/pre-commit run --all-files`
4. 恢复 actionlint 配置
5. 检查更改：`git diff --name-only`
6. 如果有实际代码更改，提交：`git add -A && git commit -sm "Fix lint"`
7. 报告最终结果