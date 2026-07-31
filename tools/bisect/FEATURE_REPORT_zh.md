# 功能报告：good table 按测试频率拆分（nightly / weekly）

## 1. 背景与目标

自动二分（auto-bisect）依赖 good table 提供“最近一次成功基线”（last-known-good
commit）。此前 nightly 与 weekly 两套定时测试共用同一张
`good_table.csv`，导致：

- weekly（每周一次）的成功基线会被 nightly（每天一次）的新记录覆盖/混淆，
  两个频率的基准不独立；
- 同一用例在不同 SoC（A2 / A3 / 310P）或不同 scene（single / multi node）
  上成功与否不同，但表中没有区分维度，无法选到正确的基线。

本次功能把 good table 拆分为 nightly / weekly 两套独立表，并在行内引入
`soc + scene` 维度，使自动二分在“频率、硬件、场景、用例”四个维度上都能取到
正确的成功基线。

## 2. 总体设计

### 2.1 表结构与命名空间

每个分支、每个测试频率各维护一张表：

```text
/root/.cache/vllm-ascend/<branch>/nightly/good_table.csv
/root/.cache/vllm-ascend/<branch>/weekly/good_table.csv
```

表头（新增 `soc`、`scene` 两列）：

```csv
name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,soc,scene,time
```

### 2.2 复合键与旧数据兼容

- 写入侧以 `soc + scene + yaml/path` 为稳定复合键：同一用例同一维度下只保留
  最新一次成功记录，互不覆盖；`name` 只是展示名，不参与唯一性判断。
- 读侧（`GoodTable.lookup_last_good`）对调用方提供的每个维度（`name`、
  `config_yaml`、`soc`、`scene`）都必须匹配，再取时间最新的一条成功记录。
- 兼容迁移：旧的七列行（无 `soc`/`scene`）仍可被读取；首次被同路径的新格式
  写入命中时自动迁移为九列新格式。

### 2.3 表生命周期

- 创建：表不是预置的，第一次有用例在正式运行中测试成功、执行
  `update_good_table.py` 时惰性创建（自动建目录并写表头）。
- 刷新：每次正式运行（`request_id == ''`）中某用例成功，即更新该复合键对应
  的行并刷新 `time`；失败行不会写入 good table（表内 `status` 恒为
  `success`）。PR 触发的运行（带 `request_id`）不写表。

### 2.4 读写与并发

- 写入使用跨进程文件锁（`good_table.csv.lock`）+ 临时文件原子替换
  （`os.replace`），matrix 并发作业共享同一张表时不会互相覆盖或读到半写文件。
- 读取是纯只读访问，不做缓存，实时反映最后一次成功写入。

### 2.5 AOP 年龄门禁

正式测试失败后，AOP 会检查该频率对应表中最近一次成功的时间：

- `aop_commit_age.sh` / `run.sh` 新增 `max_age_days` 参数，按频率取不同阈值：
  - nightly：3 天；
  - weekly：8 天。
- 超过阈值（`is_old=true`）则跳过自动二分；显式提供 `--good-commit` 时完全
  跳过年龄检查。
- 年龄查询按 `name` + 当前 `soc` 过滤新格式行（旧七列行在迁移期内仍可命中）。

### 2.6 触发与分流

- `/nightly`、`/weekly` slash 命令分别派发到 nightly / weekly 调度工作流；
  schedule 事件自动选择 `all` 用例、测试 `main` 并默认启用 AOP。
- 可复用工作流新增 `test_frequency`（表命名空间与年龄阈值）和 `soc_version`
  （表内 SoC 维度）两个输入；`schedule_weekly_test_*.yaml` 统一传入
  `test_frequency: weekly`。

## 3. 改动清单

| 文件 | 改动 |
| --- | --- |
| `tests/e2e/nightly/scripts/update_good_table.py` | 九列表头、复合键更新、惰性建表、文件锁 + 原子替换、legacy 行迁移、`--soc` 参数 |
| `tools/bisect/good_table.py` | `soc`/`scene` 列与多维度匹配、旧行兼容、时间归一化修复（混合时区/无法解析时间不再崩溃） |
| `tools/bisect/auto_bisect.py`、`tools/bisect/config.py` | 新增 `--soc`，透传到 good table 查询 |
| `tests/e2e/nightly/scripts/aop_commit_age.sh` | `max_age_days`、`soc` 过滤参数 |
| `tests/e2e/nightly/multi_node/scripts/run.sh` | `BISECT_SOC`、`MAX_GOOD_AGE_DAYS`，`--soc` 传给 auto-bisect |
| `tests/e2e/nightly/scripts/aop_process.sh` | `soc` 参数透传 |
| `_e2e_nightly_single_node*.yaml`、`_e2e_nightly_multi_node*.yaml` | `test_frequency`、`soc_version`、按频率取 good table 路径与年龄阈值 |
| `schedule_weekly_test_*.yaml`、`schedule_nightly_test_*.yaml` | 传入 `soc_version`/`test_frequency`，schedule 自动 all + AOP |
| `.github/workflows/pr_nightly_command.yml` | weekly 派发透传 `aop_enabled` |
| `tools/bisect/README.md`、`USAGE_zh.md`、`good_table.sample.csv` | 文档与新表结构样例 |
| `tests/ut/tools/bisect/test_good_table.py` 等 | 见 UT 报告 |

## 4. 验证

- 本地 UT：71 passed（含本次新增 19 个用例）。
- PR 分支（`codex/split-good-table-frequency`）：75 passed。
- UT 分支（`codex/split-good-table-frequency_ut`）：89 passed。
- 静态检查：`git diff --check` 无空白错误。
- 待 NPU/CI 验证：真实 nightly / weekly 运行中的表写入与 AOP 年龄门禁行为。

## 5. 关联

- PR：#13078
- UT 报告：`tools/bisect/UT_REPORT_good_table_zh.md`
