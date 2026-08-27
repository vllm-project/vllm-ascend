# PR 代码获取

派发为 `general` 子 Agent 执行。

## 派发

```
Agent({
  subagent_type: "general",
  description: "PR 代码获取",
  prompt: "PR 代码获取

【输入】
- PR URL: {pr_url}

【执行步骤】

0. 清理旧缓存（同一 PR 重复检视时，防止读取陈旧代码）：
   - 从 PR URL 中提取 PR 编号
   - 若 `./operators/.pr_diff/{pr_number}.diff` 存在 → 删除该文件
   - 若 `./operators/.pr_repo/{pr_number}/` 存在 → 删除整个目录（`rm -rf`）
   - 若 `./operators/pr-{pr_number}/code_summary.md` 存在 → 删除该文件
   确保 diff、源码、概要全部来自本次获取，不混用不同次检视的数据。
1. 识别 PR 托管平台（URL 含 `gitcode.com` → GitCode）
2. 获取 diff：
   - 脚本路径 = `{skill_base}/scripts/get_gitcode_pr_diff.py`
   - `mkdir -p ./operators/.pr_diff`
   - 执行脚本，传入完整 URL，`--output` 保存到 `./operators/.pr_diff/{pr_number}.diff`
3. 克隆完整源码：
   - `mkdir -p ./operators/.pr_repo`
    - 执行 `python {skill_base}/scripts/clone_pr_source.py --repo {repo_url} --pr {pr_number} --clone-dir ./operators/.pr_repo/{pr_number}/`
    - 克隆失败则终止
4. 统计 diff 规模并路由检视模式：
   - 变更文件数 = diff 中所有变更文件路径数
   - 变更行数 = diff 新增+删除行数（排除注释空行）
   - 执行 `python3 {skill_base}/scripts/workflow.review_mode.py --lines {变更行数} --files {变更文件数}`，解析返回的 JSON 取 mode/routing/guidance

【输出】

返回以下结构化信息：
- `operator_name`: {提取的算子名}
- `diff_path`: `./operators/.pr_diff/{pr_number}.diff`
- `repo_path`: `./operators/.pr_repo/{pr_number}/`
- `mode`: {minimal|compact|standard|large}
- `routing`: {pr-review|pr-large-review}
- `guidance`: {对应档位的 plan agent 工作指导}

禁止生成报告文件。"
})
```
