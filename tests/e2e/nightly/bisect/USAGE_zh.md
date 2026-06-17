# Nightly Auto-Bisect 工具使用指导

## 一、它做什么

当某个 nightly 用例失败时，自动在 `vllm-ascend` 的提交历史里二分查找**首个引入问题的 commit（及其 PR）**。区间为：

- **Bad**：当前失败的 commit；
- **Good**：从 Good 表里读取的“该用例最近一次通过的 commit”。

二分以 **commit 为最小单位**，并复用现有 nightly 的拉起入口（单机 `test_single_node.py` / 多机 `test_multi_node.py`），保证复现环境与 nightly 一致。

---

## 二、前置条件

1. 在 **NPU 环境**、且位于 `vllm-ascend` git 仓库根目录下运行（仓库需包含 good..bad 之间的提交历史，必要时工具会自动 `git fetch`）。
2. 依赖：`pytest`、`openai`、`aisbench`、`psutil`、`filelock`、`regex`（nightly 环境已具备）。
3. 运行前确保 `PYTHONPATH` 含仓库根目录（在根目录下直接用 `python -m` 即可）。

---

## 三、配置 Good 表（必读）

Good 表是一个**纯 CSV**，默认路径 `/root/.cache/nightly_bisect/good_table.csv`，可用 `BISECT_GOOD_TABLE` 覆盖。列定义：

```
case_key,scene,config_yaml,case_name,last_good_commit,last_good_pr,updated_at
```

- `case_key` = `scene::config_yaml::case_name`（查表主键）
- 多机用例 `case_name` 通常为空，写成 `multi_node::Qwen3-235B-W8A8.yaml::`

示例（参见同目录 `good_table.sample.csv`）：

```csv
case_key,scene,config_yaml,case_name,last_good_commit,last_good_pr,updated_at
single_node::DeepSeek-R1-0528-W8A8.yaml::DeepSeek-R1-0528-W8A8-aclgraph,single_node,DeepSeek-R1-0528-W8A8.yaml,DeepSeek-R1-0528-W8A8-aclgraph,5e45ac99...,10442,2026-06-15T02:00:00Z
multi_node::Qwen3-235B-W8A8.yaml::,multi_node,Qwen3-235B-W8A8.yaml,,a63d03fc...,10177,2026-06-15T02:00:00Z
```

> 如果不想依赖表，也可在命令行用 `--good-commit <sha>` 直接指定 good 端点。

---

## 四、单机用例

在仓库根目录执行：

```bash
python -m tests.e2e.nightly.bisect.auto_bisect \
    --scene single_node \
    --config-yaml DeepSeek-R1-0528-W8A8.yaml \
    --case-name DeepSeek-R1-0528-W8A8-aclgraph \
    --bad-commit HEAD
```

参数说明：
- `--config-yaml`：失败用例所在的 YAML（对应 `CONFIG_YAML_PATH`）；
- `--case-name`：pytest 的 `-k` 过滤值，精确锁定到失败的那个 case；
- `--bad-commit`：当前失败 commit，默认取环境变量 `VLLM_ASCEND_REF`，否则 `HEAD`；
- good 端点未指定时自动查 Good 表。

---

## 五、多机用例

**在每个节点上都要启动同一条命令**，并让所有节点指向**同一个共享目录** `--coord-dir`（必须是各节点都能读写的网络盘/PVC）。

- master 节点（`LWS_WORKER_INDEX=0`）：驱动二分；
- 其余节点：自动进入 worker 受控循环（切码→编译→就绪→跑→等下一轮）。

```bash
python -m tests.e2e.nightly.bisect.auto_bisect \
    --scene multi_node \
    --config-yaml Qwen3-235B-W8A8.yaml \
    --bad-commit "$VLLM_ASCEND_REF" \
    --num-nodes 2 \
    --coord-dir /shared/nightly_bisect/coord
```

说明：
- `--num-nodes` / `--node-index` 默认读环境变量 `LWS_GROUP_SIZE` / `LWS_WORKER_INDEX`（LWS 编排下一般无需手填）；
- internal / external DP 通过 `--config-base-path`（或 yaml 路径含 `external_dp/config`）自动区分；
- 所有节点必须切到同一 commit 后才会开跑（屏障同步），避免“半边新半边旧”。

---

## 六、常用可选参数 / 环境变量

| 参数 / 变量 | 默认 | 作用 |
|---|---|---|
| `--good-commit <sha>` | 查表 | 显式指定 good，跳过 Good 表 |
| `--config-base-path` | 环境变量 | 多机 internal/external DP 配置目录 |
| `--fail-confirm-retries N` | 1 | FAIL 复测次数；复测出现 PASS 则判为 flaky→SKIP |
| `--no-verify-good` | 关 | 跳过开跑前对 good 端点的复核 |
| `--no-verify-bad` | 关 | 跳过开跑前对 bad 端点的复核 |
| `--trial-timeout-s` | 5400 | 单轮 pytest 超时（秒）|
| `--work-dir` / `BISECT_WORK_DIR` | `/root/.cache/nightly_bisect/runs` | 日志/状态/报告输出目录 |
| `--good-table` / `BISECT_GOOD_TABLE` | `/root/.cache/nightly_bisect/good_table.csv` | Good 表路径 |
| `--coord-dir` / `BISECT_COORD_DIR` | `/root/.cache/nightly_bisect/coord` | 多机共享屏障目录 |
| `BISECT_UPDATE_GOOD_TABLE=1` | 关 | 定位完成后把“新的 last good”写回 Good 表 |

> 编译优化是自动的：只有当切换涉及 `.cpp/.h/CMakeLists.txt/setup.py` 等原生/构建文件时才 `pip install` 重编译；纯 `.py`/yaml 改动只 `git checkout`。无需手动控制。

---

## 七、输出怎么看

**控制台**：每轮都打印明显标识，例如：

```
=== Round 3: testing PR-10492 (46356897b4d8); window=[2,4] 2 left, ~1 rounds to go ===
[FAIL] PR-10492  (46356897b4d8, rebuilt, 612s) - pytest exited non-zero (rc=1)
...
================================================================
  FIRST BAD COMMIT: 46356897b4d8...
  FIRST BAD PR:     PR-10492
  Subject:          [Performance] Skip compute_slot_mapping ...
================================================================
```

`[PASS]/[FAIL]/[SKIP]` 分别表示该 commit 通过 / 复现失败 / 无法判定（编译失败或 flaky）。

**文件**（位于 `$BISECT_WORK_DIR/<case_key>/`）：
- `logs/round<N>_<sha>.log`：每轮的编译 + pytest 全量日志；
- `state.json`：二分窗口和已判定结果（被中断后**重跑同一命令即可断点续跑**）；
- `report.json`：最终结论（首个 bad commit / PR + 完整试跑历史）。

**退出码**：`0` = 成功定位首个 bad；`2` = 未定位（如端点校验失败、区间无效）。

---

## 八、典型流程小结

1. 确认 Good 表里有该用例的记录（或准备好 `--good-commit`）；
2. 在节点上跑对应 `--scene` 的命令（多机则每个节点都跑）；
3. 看控制台的 `[PASS]/[FAIL]` 收敛过程；
4. 读 `report.json` 拿到 First Bad PR；
5. 如被中断，原命令重跑即续跑。
