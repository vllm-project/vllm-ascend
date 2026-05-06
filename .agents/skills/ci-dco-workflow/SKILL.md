---
name: ci-dco-workflow
description: |
  Standard workflow for vllm-ascend PR CI triage and DCO compliance fixes.
  Use when checks fail, workflows do not trigger, or DCO is red.
  Covers log collection, root-cause isolation, minimal patching, DCO-safe history rewrite,
  and final verification gates before merge.
---

# ci-dco-workflow

用于处理 PR 上的 CI 失败与 DCO 签名问题，目标是快速恢复“可合并状态”。

## 何时使用

- 用户反馈“有新的 CI 错误 / check fail / nightly fail”
- 用户反馈“DCO 不通过 / ACTION_REQUIRED”
- 用户反馈“为什么没触发 e2e / workflow 没跑”
- 用户要求整理提交历史并保证 DCO 合规

## 执行流程

### 1. 快速体检（先判断是触发问题还是代码问题）

1. 拉取 PR 状态与 checks：
   - `gh pr view <pr> --json headRefOid,mergeable,mergeStateStatus,statusCheckRollup,url`
2. 如果关键 workflow 缺失：
   - 查询该 `head sha` 的 runs，确认是否只有 `pull_request_target`
   - 结论标记为“触发链路问题”而非“代码失败”

### 2. 失败定位（只看首个真实错误）

1. 下载失败 job 日志：
   - `gh run view <run_id> --job <job_id> --log > /tmp/<job>.log`
2. 提取关键信号（按优先级）：
   - `Traceback`
   - `FAILED tests/...`
   - `ModuleNotFoundError|ImportError|TypeError|RuntimeError`
   - 排除纯 warning 噪声
3. 只对“第一处根因”修复，避免一次改动过大

### 3. 修复策略（最小变更）

- 原则：
  - 先修兼容导入与版本守卫，再修行为逻辑
  - 仅改动与当前根因直接相关的文件
  - 保留失败可观测性（不要吞掉真实错误）
- 提交前最少验证：
  - `python -m py_compile <changed_py_files>`
  - `pre-commit run --all-files --hook-stage manual --show-diff-on-failure`
  - 若 `ruff format` 报 `files were modified by this hook`：
    - 接受格式化改动并重新提交（常见是空行/导入分组等样式修正）
    - 不要手工对抗 formatter，直到本地 `pre-commit` 全绿
  - 相关 UT/脚本可跑则跑，不能跑要明确说明原因

### 4. DCO 合规修复

#### A. 单个新提交缺签名

- 直接用 `-s` 重新提交：
  - `git commit -s -m "<message>"`

#### B. 历史里已有无签名提交（常见于 Web merge/revert）

1. 先建备份分支：
   - `git branch backup/<name>-<date>`
2. 推荐做法：在最新 `upstream/main` 上重放有效提交（线性化）
   - `git reset --hard upstream/main`
   - `git cherry-pick <good_commit_1> <good_commit_2> ...`
3. 强推：
   - `git push --force-with-lease origin <branch>`

#### C. 验证 DCO

- `gh pr view <pr> --json commits,statusCheckRollup`
- 确认：
  - `DCO` 为 `SUCCESS`
  - PR commits 只保留语义化提交

### 5. 收尾门禁

合并前必须满足：

1. `mergeable == MERGEABLE`
2. DCO `SUCCESS`
3. 关键 CI（E2E/Nightly/Pre-commit）进入可解释状态：
   - 通过，或
   - 有明确外部依赖故障证据（网络/runner/action download）
4. 提交数合理（建议 3-5 个语义化 commits）

## 输出模板

对用户输出固定包含：

1. 本次首个根因（一句话）
2. 已修改内容（文件 + 目的）
3. commit SHA 与 push 状态
4. 当前 checks 状态（通过/排队/外部故障）
5. 下一步动作（如 rerun、等待 runner、补充日志）
