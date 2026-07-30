# AOP 自动二分优化 UT 报告

## 1. 范围与结论

本报告覆盖分支 `codex/split-good-table-frequency_ut` 中与 AOP 调用自动二分工具直接相关的提交：

- `390f4bfb`：记录并回放 vLLM、CANN、torch-npu 运行时环境；多节点按轮同步环境。
- `e25ac3ab`：把 11 个二分参数从 `/nightly`、`/weekly` 命令透传到工作流、Pod、AOP 脚本和 `auto_bisect.py`。
- `86110a55`：把可选参数压缩成一个经过校验的 `bisect_args_json`，避免 GitHub `workflow_dispatch` 的 input 数量上限。

同时纳入其前置依赖 `b68bd0ee`：按 nightly/weekly 与 SoC 拆分 good table，使 AOP 能选择正确的成功基线；`ec75937d` 负责发布对应 artifact。

当前 bisect UT 共 70 项，本地 Windows 全部执行通过。AOP Shell 测试在 Windows 显式使用 Git for Windows Bash，在 Linux 使用系统 Bash。本次新增 14 项测试，重点补齐运行时环境管理、参数契约、多节点失败同步，以及从 AOP Shell 到 `report.json` 的全链路仿真。

## 2. 触发路径

```text
PR 评论 /nightly 或 /weekly
  -> pr_nightly_command.yml 解析并校验参数
  -> 参数编码为 bisect_args_json
  -> schedule_{nightly,weekly}_*.yaml 解包
  -> _e2e_nightly_{single,multi}_node*.yaml 注入 BISECT_* 环境变量
  -> 首次正式测试失败
  -> AOP classify：命中环境类错误则停止
  -> AOP age gate：显式 good commit 时跳过，否则查询 cadence/SoC/scene 对应 good table
  -> update_good_table.py 记录失败时的 env_table 行
  -> aop_process.sh 或 multi_node/scripts/run.sh 组装 auto_bisect CLI
  -> auto_bisect.py 解析 good/bad、候选提交和 env_table
  -> EnvTable 为每个候选选择“精确行或最近祖先行”
  -> EnvironmentManager 按 CANN -> torch-npu -> vLLM 顺序切换环境
  -> 环境变化使 BuildManager 的构建基线失效
  -> 单节点直接执行 nightly pytest；多节点 leader 广播 commit/rebuild/env/action
  -> worker 同步环境和提交，barrier 后共同执行
  -> PASS/FAIL/SKIP 驱动二分并生成 report.json
```

触发二分必须同时满足：正式测试失败、AOP 已启用、失败日志未被规则识别为环境故障，并且存在未过期的成功基线；若显式提供 `--good-commit`，则不依赖 good table 和 age gate。多节点还要求 worker 进入共享协调目录并按轮消费命令。

## 3. 风险分析与 UT 设计

| 风险 | 关键契约 | UT 覆盖 |
|---|---|---|
| 参数在长链路中丢失或类型改变 | CLI 中 good/bad、重试、超时、端点校验、构建策略、native-check、config path、env table 保持原值 | `test_parse_args_maps_extended_aop_parameters` |
| 缺省调用改变历史行为 | 未给可选参数时继续使用既有默认值 | 既有 CLI/runner 测试；工作流需 CI 集成验证 |
| env table 选错版本 | 精确 commit 优先，否则选择同一 case 的最近祖先 | `test_env_table_prefers_exact_commit_row`、`test_env_table_uses_closest_preceding_status_row` |
| 占位值触发错误安装 | 空值、N/A、unknown、None 均视为未知 | `test_known_rejects_status_table_placeholders` |
| 环境切换顺序不正确 | CANN 先于 torch-npu，vLLM 最后 | `test_ensure_applies_components_in_dependency_order` |
| 空环境仍执行破坏性切换 | `None` 或空 RuntimeEnv 必须 no-op | `test_ensure_empty_target_is_noop` |
| CANN 目标版本不可用 | 明确抛出 `EnvSwitchError`，候选应 SKIP | `test_ensure_cann_reports_unavailable_runtime` |
| torch-npu 版本不同时安装命令错误 | 指定精确版本并强制重装 | `test_ensure_torch_npu_installs_only_when_version_differs` |
| 环境变更后复用旧 native build | 任一环境组件变化都使 build baseline 失效 | `test_runtime_env_change_invalidates_build_baseline` |
| leader 环境切换失败导致 worker 卡住 | 广播同轮 `action=SKIP` 和目标 env，不进入 barrier | `test_multi_node_env_failure_broadcasts_skip` |
| 多节点环境载荷丢失 | command.json 能往返保存 env | `test_publish_command_can_include_runtime_env` |
| cadence/SoC/scene 之间污染 good table | 复合键更新且兼容旧 7 列表 | `test_update_replaces_only_same_composite_key`、`test_update_migrates_matching_legacy_row` |
| 分散组件均通过但完整二分链无法收敛 | 使用真实 Git 历史、good/env table、候选选择、端点校验、verdict、状态和报告，只替换构建/NPU 执行 | `test_aop_cli_to_first_bad_report_full_chain` |
| AOP Shell 组装时丢失或拆散参数 | Windows Git Bash/Linux Bash 下真实执行 `aop_process.sh`，使用 fake Python 捕获 update-table 与 auto-bisect 的完整 argv | `test_aop_shell_forwards_complete_bisect_contract` |

本次未用纯文本断言复制 GitHub Actions YAML。YAML、shell、`fromJSON` 和真实 `workflow_dispatch` 的组合属于集成边界，建议通过最小化 workflow smoke 覆盖，避免 UT 对缩进和实现细节高度敏感。

## 4. 新增与既有关键用例

本次新增：

- `test_env_manager.py`：10 项，覆盖占位值、CANN 版本读取、切换顺序、空目标、CANN 缺失和 torch-npu 安装命令。
- `test_runner.py`：2 项，覆盖环境变化使构建基线失效，以及 multi-node leader 环境失败后的 SKIP 广播。
- `test_auto_bisect.py`：将原参数测试扩展为 11 个 AOP 参数与 env table 的完整 CLI 映射。
- `test_full_chain.py`：使用四个真实 Git 提交和真实 CSV 表，贯通 CLI、good baseline、环境继承、端点验证、二分收敛及最终报告；仅 mock 构建与 NPU case 执行。
- `test_aop_shell_chain.py`：Windows Git Bash/Linux Bash 下真实运行 AOP Shell，验证 11 个二分控制参数及 good/env table 参数无丢失、无错误拆词地到达 Python CLI。

既有 56 项继续覆盖二分中点选择、候选提交、构建决策、good/env table、协调器、runner 入口、状态恢复、报告与 verdict。

## 5. 执行记录

执行环境：Windows，Python 3.12；为绕过仓库顶层 NPU/Torch fixture，仅把 `tests/ut/tools/bisect` 设为 conftest 边界。临时目录位于仓库内。

```powershell
python -m pytest -q `
  --confcutdir=tests/ut/tools/bisect `
  --basetemp .tmp-pytest `
  tests/ut/tools/bisect
```

结果：

```text
....................................................................     [100%]
70 passed
```

静态检查：`git diff --check` 通过，无空白错误。

## 6. 尚需 CI/NPU 验证的边界

- GitHub `/nightly` 与 `/weekly` 评论解析、JSON 打包、各调度工作流 `fromJSON` 解包的真实链路。
- single-node 与 multi-node Pod 中 `BISECT_*` 环境变量、shell 数组及带空格/特殊字符参数的传递。
- 真实 CANN 多版本挂载、torch-npu wheel 源、vLLM source checkout/install 的可用性与耗时。
- LWS 共享卷上的多 worker barrier、SKIP/DONE/release-file 时序和 leader 异常退出恢复。
- NPU 上完整的“首次失败 -> AOP -> 二分 -> first bad report”物理闭环；当前全链路 UT 已覆盖仓库内可控逻辑，但 mock 了构建和 NPU case 执行。

建议 CI 最少增加两条 smoke：一条 single-node 使用伪 runner 验证 11 参数最终 argv；一条双 worker 使用临时共享目录验证 RUN、SKIP、DONE 三种命令时序。真实 NPU nightly 再承担环境切换和模型执行验证。
