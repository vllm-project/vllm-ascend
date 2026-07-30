# Bisect 参数扩展说明

## 概述

将 `/nightly` PR 命令和 AOP 链路可传递的 bisect 参数从 **6 个扩展到 17 个**（新增 11 个），覆盖 `auto_bisect.py` 的全部主要参数。

---

## 一、使用方式

```
/nightly <用例名> --aop_enabled [bisect 可选参数...]
```

> `--aop_enabled` 是 bisect 参数的父级开关，必须写在所有 bisect 参数之前。
> 未启用 AOP 或将 bisect 参数写在 `--aop_enabled` 之前时，命令会直接报错且不会触发 workflow。

命令按从左到右的顺序解析，结构如下：

```text
/nightly
└─ <用例名>
   └─ --aop_enabled
      ├─ --good-commit <sha>
      ├─ --bad-commit <sha|HEAD>
      ├─ --fail-confirm-retries <n>
      ├─ --trial-timeout <seconds>
      ├─ --barrier-timeout <seconds>
      ├─ --no-verify-good
      ├─ --no-verify-bad
      ├─ --force-initial-build
      ├─ --no-assume-built-head
      ├─ --native-check <mode>
      └─ --config-base-path <path>
```

### 新增参数一览

| `/nightly` 参数 | 对应 `auto_bisect` 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|---|
| `--good-commit <sha>` | `--good-commit` | string | (查表) | 指定已知 good commit，跳过 good_table 查询和基于表记录的 commit-age gate |
| `--bad-commit <sha>` | `--bad-commit` | string | `HEAD` | 指定 bad commit |
| `--fail-confirm-retries <n>` | `--fail-confirm-retries` | number | 1 | FAIL 后复测次数，防 flaky 误判 |
| `--trial-timeout <n>` | `--trial-timeout-s` | number | 7200 | 单次 trial 超时秒数 |
| `--barrier-timeout <n>` | `--barrier-timeout-s` | number | 3600 | 多机节点同步超时秒数 |
| `--no-verify-good` | `--no-verify-good` | flag | false | 跳过 good 端点验证 |
| `--no-verify-bad` | `--no-verify-bad` | flag | false | 跳过 bad 端点验证 |
| `--force-initial-build` | `--force-initial-build` | flag | false | 首轮强制 clean rebuild |
| `--no-assume-built-head` | `--no-assume-built-head` | flag | false | 不信任容器 HEAD 已构建 |
| `--native-check <mode>` | `--native-check` | string | `per-commit` | 编译检测策略：`per-commit` / `since-build` |
| `--config-base-path <path>` | `--config-base-path` | string | (空) | 覆盖 `CONFIG_BASE_PATH` |

### 使用示例

```bash
# 默认行为（所有 bisect 参数使用内置默认值）
/nightly DeepSeek-R1 --aop_enabled

# 指定 commit 范围
/nightly DeepSeek-R1 --aop_enabled --good-commit abc1234 --bad-commit def5678

# 调大超时 + 增加 flaky 容忍度
/nightly DeepSeek-R1 --aop_enabled --trial-timeout 14400 --fail-confirm-retries 3

# 跳过验证 + 强制重建
/nightly DeepSeek-R1 --aop_enabled --no-verify-good --force-initial-build

# 多机场景组合
/nightly Qwen3-32B --aop_enabled --barrier-timeout 600 --native-check since-build
```

### 非法命令示例

```bash
# 未启用 AOP
/nightly DeepSeek-R1 --trial-timeout 3600

# --aop_enabled 出现在 bisect 参数之后
/nightly DeepSeek-R1 --trial-timeout 3600 --aop_enabled

# 参数缺值
/nightly DeepSeek-R1 --aop_enabled --trial-timeout --no-verify-good

# 参数格式错误
/nightly DeepSeek-R1 --aop_enabled --native-check invalid-mode
```

上述命令都会在 PR 命令解析阶段报错，不会触发下游 workflow。

### 参数校验

| 参数 | 校验规则 |
|---|---|
| `--good-commit` | 7～40 位十六进制 commit SHA |
| `--bad-commit` | `HEAD` 或 7～40 位十六进制 commit SHA |
| `--fail-confirm-retries` | 非负整数 |
| `--trial-timeout` | 大于 0 的整数或小数 |
| `--barrier-timeout` | 大于 0 的整数或小数 |
| `--native-check` | 仅支持 `per-commit`、`since-build` |
| `--config-base-path` | 仅支持字母、数字、`_`、`.`、`/`、`-` |

所有需要值的参数都会检查 next-arg；下一个 token 是其他 flag 或命令已经结束时，按缺值处理。

### 兼容性

不传任何 bisect 参数时，行为与改动前完全一致。所有参数默认值保持为空/`false`/`HEAD`，由 `auto_bisect.py` 使用其内置默认值。

---

## 二、参数传递链路

```
PR 评论: /nightly <cases> --aop_enabled --trial-timeout 3600 ...
  │
  ├─ pr_nightly_command.yml    解析 --flags → outputs → gh workflow run -f <params>
  │
  ├─ schedule_*_test_*.yaml    workflow_dispatch inputs → 转发到 E2E template
  │
  ├─ [单机] _e2e_nightly_single_node*.yaml
  │         workflow_call inputs → aop_process.sh 位置参数
  │         └─ python -m tools.bisect.auto_bisect --trial-timeout-s 3600 ...
  │
  └─ [多机] _e2e_nightly_multi_node*.yaml
            workflow_call inputs → jinja2 -D → lws.yaml 环境变量
            → run.sh → build_bisect_extra_args()
            └─ python -m tools.bisect.auto_bisect --trial-timeout-s 3600 ...
```

---

## 三、修改文件清单（19 个，含本文档）

### 1. PR 命令解析
| 文件 | 改动 |
|---|---|
| `.github/workflows/pr_nightly_command.yml` | 按严格顺序解析 11 个新 `--flags`；校验缺值和参数格式；job outputs + dispatch `-f` 转发（5 个 dispatch job） |

### 2. 调度流程（5 个）
| 文件 | 改动 |
|---|---|
| `.github/workflows/schedule_nightly_test_a2.yaml` | 新增 `bisect_*` inputs；multi_node 和 single_node 转发 |
| `.github/workflows/schedule_nightly_test_a3.yaml` | 同上；multi/double/single/multi-card 四处转发 |
| `.github/workflows/schedule_nightly_test_a3_560t.yaml` | 同上 |
| `.github/workflows/schedule_weekly_test_a3.yaml` | 新增 inputs + 转发 |
| `.github/workflows/schedule_weekly_test_310p.yaml` | 新增 inputs，并转发到单机 AOP 模板 |
| `.github/workflows/schedule_weekly_test_a2.yaml` | 新增 inputs，并转发到 accuracy AOP 模板 |

### 3. E2E 模板（4 个）
| 文件 | 改动 |
|---|---|
| `.github/workflows/_e2e_nightly_single_node.yaml` | 新增 11 个 inputs；`aop_process.sh` 调用传参替换硬编码 `HEAD` |
| `.github/workflows/_e2e_nightly_single_node_560t.yaml` | 同上 |
| `.github/workflows/_e2e_nightly_multi_node.yaml` | 新增 11 个 inputs；jinja2 `-D` 变量传入 |
| `.github/workflows/_e2e_nightly_multi_node_560t.yaml` | 同上 |
| `.github/workflows/_e2e_nightly_single_node_models.yaml` | Weekly A2 accuracy 失败时记录模型并触发对应测试入口的二分 |

### 4. K8s 模板与 Shell（4 个）
| 文件 | 改动 |
|---|---|
| `tests/e2e/nightly/multi_node/scripts/lws.yaml.jinja2` | leader + worker pod 各新增 11 个 `BISECT_*` 环境变量，并安全转义 YAML 字符串 |
| `tests/e2e/nightly/multi_node/scripts/lws_560t.yaml.jinja2` | 560T leader + worker pod 同步新增并安全转义 `BISECT_*` 环境变量 |
| `tests/e2e/nightly/scripts/aop_process.sh` | 接收 10 个新位置参数（`$13`~`$22`）；条件拼接到 `BISECT_CMD` |
| `tests/e2e/nightly/multi_node/scripts/run.sh` | 新增 `build_bisect_extra_args()` 数组构造函数；leader + worker 两条路径复用；`bad_commit` 替换硬编码 `HEAD` |

### 5. 二分执行与测试（2 个）
| 文件 | 改动 |
|---|---|
| `tools/bisect/runner.py` | 根据 Weekly A2 模型配置选择 LM/ASR/RM accuracy pytest 入口 |
| `tests/ut/tools/bisect/test_runner.py` | 覆盖 accuracy 测试入口选择逻辑 |

### 6. 说明文档（1 个）
| 文件 | 改动 |
|---|---|
| `tools/bisect/BISECT_PARAMS.md` | 记录命令结构、参数约束、传递链路和修改范围 |

---

## 四、设计原则

1. **默认行为不变**：所有新参数默认为空/`false`/`HEAD`，不指定时由 `auto_bisect.py` 使用内置默认值
2. **逐层显式传递**：每个参数在每层都有明确命名（`bisect_<name>`），可追踪、可调试
3. **严格的父子参数关系**：`--aop_enabled` 必须位于所有 bisect 参数之前，否则不触发 workflow
4. **Boolean 用 flag，值参数用 next-arg**：与 `auto_bisect.py` 的 argparse 定义保持一致
5. **安全传递**：PR dispatch 使用环境变量传值，避免把用户输入直接拼入 Shell 源码
6. **保持参数边界**：多机 leader/worker 使用 Bash 数组传参，避免 word splitting 和 pathname expansion
