# AOP 自动二分功能说明

## 1. 改动范围

以下提交共同形成了当前 AOP 自动二分能力：

| 提交 | 功能 |
|---|---|
| `b68bd0ee` | 按 nightly/weekly、SoC、scene 隔离成功基线表，并发安全更新并兼容旧表 |
| `ec75937d` | 将频率隔离后的 good table 发布为 GitHub Actions Artifact |
| `390f4bfb` | 记录 vLLM、CANN、torch-npu 环境，并在每个二分候选执行前回放 |
| `e25ac3ab` | 将 11 个二分参数从 `/nightly`、`/weekly` 评论命令传到 AOP 和二分工具 |
| `86110a55` | 将可选参数合并为一个校验后的 `bisect_args_json`，规避 workflow dispatch input 数量限制 |

它们解决三个核心问题：成功基线可能来自不同频率和硬件；历史提交在当前软件环境中可能无法正确复现；原 AOP 入口无法控制端点、超时、重试和构建策略。

## 2. 整体能力

受支持的 nightly/weekly 用例失败后，AOP 可以：

1. 区分已知环境故障和可能的代码回归。
2. 从当前 frequency、branch、SoC、scene 对应的 good table 选择最近成功提交，或使用显式 good commit。
3. 解析候选提交对应的 vLLM、CANN、torch-npu 环境。
4. 每轮切换运行时环境和 vllm-ascend commit，必要时重新构建。
5. 单节点复用原测试入口；多节点按轮同步环境、commit 和 barrier。
6. 将 pytest 退出码与 benchmark JSON 合并为 PASS、FAIL、SKIP。
7. 找到首个 bad commit/PR，并生成日志、续跑状态和 JSON 报告。

## 3. 触发链路

```text
PR 评论 /nightly 或 /weekly
  -> pr_nightly_command.yml
       解析、校验参数，生成 bisect_args_json
  -> schedule_{nightly,weekly}_*.yaml
       解包 JSON，转发到 reusable E2E workflow
  -> _e2e_nightly_{single,multi}_node*.yaml
       注入 AOP 与 BISECT_* 参数
  -> 原始 E2E 用例失败
  -> AOP classify
       命中环境错误规则：停止
       未命中：继续
  -> AOP good-age gate
       有显式 good：跳过查表和 age gate
       无显式 good：检查 success 基线及有效期
  -> update_good_table.py 记录 failure 环境
  -> aop_process.sh 或 multi_node/scripts/run.sh
       使用 Bash 数组组装 auto_bisect argv
  -> tools.bisect.auto_bisect
       校验端点、解析候选环境、执行二分
  -> report.json
```

`--aop_enabled` 是评论命令中二分参数的父开关。二分参数必须位于它之后；未启用 AOP、参数缺值或格式非法时，不会 dispatch 下游 workflow。

## 4. Good table：隔离成功基线

nightly 和 weekly 使用不同路径：

```text
/root/.cache/vllm-ascend/<branch>/nightly/good_table.csv
/root/.cache/vllm-ascend/<branch>/weekly/good_table.csv
```

表结构：

```csv
name,yaml/path,link,status,vLLM Git information,vLLM-Ascend Git information,soc,scene,time
```

工具匹配调用方提供的 `name`、`config-yaml`、`soc`、`scene`，在 `status=success` 的行中取时间最新的一条。旧七列表仍可读取。

矩阵任务会并发写表，`update_good_table.py` 使用跨进程文件锁、临时文件和原子替换，防止部分写入和相互覆盖。正式 scheduled/manual run 发布：

```text
good-table-<frequency>-<platform>-<branch>
```

Artifact 保留 `nightly/good_table.csv` 或 `weekly/good_table.csv` 路径；PR 评论任务不发布正式基线 Artifact。

## 5. Env table：记录并回放历史环境

env table 对 success 和 failure 都记录环境：

```csv
name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,CANN Version,torch-npu Version,time
```

默认位于 good table 同目录的 `env_table.csv`，可用 `--env-table` 或 `BISECT_ENV_TABLE` 覆盖。

对每个 good/candidate/bad commit：

1. 优先选择相同 case 且 vllm-ascend commit 精确匹配的行。
2. 没有精确行时，选择 first-parent 历史中离候选最近的祖先行。
3. `N/A`、`unknown`、`None` 和空值不触发环境安装。

环境切换顺序是：

```text
CANN -> torch-npu -> vLLM -> vllm-ascend checkout/build -> pytest
```

- CANN 不同时 source 已安装目标版本的 `set_env.sh`；目标版本不存在则报 `EnvSwitchError`。
- torch-npu 不同时精确版本强制重装。
- vLLM 切换 `/vllm-workspace/vllm` 到目标 ref，并从源码 editable install。
- 任一运行时组件变化都会使 vllm-ascend build baseline 失效。

环境无法切换时，候选为 SKIP，而不是误判为代码 FAIL。

## 6. 评论命令和参数

```text
/nightly DeepSeek-R1 --aop_enabled

/nightly DeepSeek-R1 --aop_enabled \
  --good-commit abc1234 \
  --bad-commit def5678 \
  --fail-confirm-retries 3 \
  --trial-timeout 14400 \
  --no-assume-built-head \
  --native-check since-build
```

`/weekly` 使用相同格式。

| 评论命令参数 | auto-bisect 参数 | 默认行为 |
|---|---|---|
| `--good-commit SHA` | `--good-commit` | 查询 good table |
| `--bad-commit SHA\|HEAD` | `--bad-commit` | `HEAD` |
| `--fail-confirm-retries N` | `--fail-confirm-retries` | `1` |
| `--trial-timeout SEC` | `--trial-timeout-s` | `7200` |
| `--barrier-timeout SEC` | `--barrier-timeout-s` | `3600` |
| `--no-verify-good` | 同名 flag | 默认验证 good |
| `--no-verify-bad` | 同名 flag | 默认验证 bad |
| `--force-initial-build` | 同名 flag | 默认信任初始构建 |
| `--no-assume-built-head` | 同名 flag | 默认把容器 HEAD 视为已构建 |
| `--native-check MODE` | `--native-check` | `per-commit` |
| `--config-base-path PATH` | `--config-base-path` | workflow/环境默认路径 |

PR 命令先校验 SHA、数字、枚举和安全路径字符，再压缩为一个 JSON input。schedule workflow 使用 `fromJSON` 解包；单节点通过 AOP Shell 位置参数传递，多节点通过 Jinja2/Kubernetes 环境变量传给 leader 和 worker。Shell 使用数组保留参数边界。完整规则见 [BISECT_PARAMS.md](./BISECT_PARAMS.md)。

## 7. 单节点和多节点

单节点复用原 nightly 入口。模型 accuracy 配置根据 `model_type` 选择：

| `model_type` | pytest 入口 |
|---|---|
| `vllm` 或缺省 | `test_lm_eval_correctness.py` |
| `vllm-asr` | `test_asr_eval_correctness.py` |
| `vllm-rm` | `test_rm_eval_correctness.py` |

每轮清空 benchmark result 目录，防止上一轮 JSON 污染 verdict。

多节点中 node index 0 为 leader，其余为 worker：

```text
leader: resolve env -> publish command(commit, rebuild, env, action)
worker: wait -> switch env -> checkout/build -> signal ready
leader: local checkout/build -> signal ready -> wait all ready
all nodes: run distributed test
leader: evaluate -> next round or DONE
```

所有节点必须共享 `coord-dir`。leader 环境失败时发布 `action=SKIP`，worker 消费该轮但不执行，防止 round 失步。worker 部署失败时上报当前 HEAD，由 leader 的一致性检查阻止错误 barrier。`DONE` 或 release file 释放 worker，陈旧 sentinel 通过时间戳忽略。

## 8. 构建和 verdict

默认 `--native-check per-commit`：

- 纯 Python/YAML/Markdown：只 checkout；
- native/build 文件：重新 editable install vllm-ascend；
- requirements：重新安装依赖；
- runtime env 变化：强制使 build baseline 失效。

`since-build` 检查距上次 build 的累计变化，更保守。`force-initial-build` 和 `no-assume-built-head` 用于不信任容器初始二进制的情况。

| 信号 | Verdict |
|---|---|
| pytest rc=0，benchmark 无失败 | PASS |
| pytest rc=1 或 JSON 中 `pass_fail=fail` | FAIL |
| pytest rc=2/3/4/5、超时 124、环境/构建无法判断 | SKIP |

默认 bad 必须 FAIL、good 必须 PASS；端点为 SKIP 时中止。首次 FAIL 默认确认一次，重试不再 FAIL 时记为 flaky SKIP。单节点可复用 state 的 PASS/FAIL；多节点禁用 cache，避免 cache hit 跳过 deploy 导致 round 失步。

候选使用 `git log --first-parent --reverse good..bad`，一个 mainline PR 对应一个二分原子，PR 号从提交标题末尾的 `(#NNNN)` 提取。

## 9. 输出、兼容性和边界

输出位于 `$BISECT_WORK_DIR/<scene>__<config_yaml>/`：

- `logs/round<N>_<sha>.log`：环境、构建和 pytest 日志；
- `state.json`：搜索窗口和 verdict，相同命令可续跑；
- `report.json`：first bad commit/PR 和全部 trial。

找到 first bad 返回 `0`；端点不成立、区间无效或无法定位返回 `2`。

- 不传新增参数时保持原默认行为。
- 显式 good 跳过 good table 和 age gate，但 env table 仍用于环境回放。
- legacy good table 仍可读取，新写入统一包含 `soc/scene`。
- UT 已覆盖真实 AOP Shell 参数组装，以及从真实临时 Git/CSV 到报告的可控全链路。
- GitHub dispatch、真实 Kubernetes/LWS 并发、实际环境安装和 NPU 执行仍由 workflow/NPU E2E 验证。

## 10. 维护入口

| 模块 | 职责 |
|---|---|
| `update_good_table.py` | good/env table 采集、锁和原子更新 |
| `aop_process.sh` | 单节点 AOP 命令组装 |
| `multi_node/scripts/run.sh` | 多节点 AOP 触发和节点加入 |
| `auto_bisect.py` | 端点校验、二分、状态和报告 |
| `good_table.py` / `env_table.py` | 基线及候选环境解析 |
| `env_manager.py` | CANN、torch-npu、vLLM 切换 |
| `build_manager.py` | checkout、依赖和 native build |
| `runner.py` | 单/多节点执行 |
| `coordinator.py` / `worker_agent.py` | 多节点同步和 worker 循环 |

操作步骤见 [USAGE_zh.md](./USAGE_zh.md)，参数细节见 [BISECT_PARAMS.md](./BISECT_PARAMS.md)，测试设计与结果见 [UT_REPORT_zh.md](./UT_REPORT_zh.md)。
