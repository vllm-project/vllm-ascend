# Nightly Auto-Bisect 工具使用指导

## 一、它做什么

当某个 nightly 用例失败时,在 `vllm-ascend` 的提交历史里二分查找**首个引入问题的 commit(及其 PR)**:

- **Bad**:当前失败的 commit(默认 `HEAD`)。
- **Good**:从 nightly 状态表里读取的"该用例最近一次 `success` 的 vLLM-Ascend commit"。

二分以 **commit 为最小单位**,每一轮复用现有 nightly 入口(单机 `test_single_node.py` / 多机 `test_multi_node.py`)**完整运行整个 YAML 文件的所有用例**,保证复现环境与 nightly 一致。

> 注意:nightly 无法精确到单个 case 粒度,所以本工具**按整个 YAML 文件判定好坏**(文件内任一 case 失败即判 FAIL),不能只跑某一个 case。

---

## 二、前置条件(真实环境务必确认)

1. 在 **NPU 容器**内、`vllm-ascend` 仓库根目录下运行(例如 `/workspace/vllm-ascend`)。
2. 直接用 `python -m tools.bisect.auto_bisect ...` 运行(在仓库根目录下,包可被导入)。
3. **vllm-ascend 必须是 editable 安装**(`pip install -e`)——这样纯 `.py` 改动 checkout 后即时生效、无需重装。nightly 容器默认就是 editable。
4. **vLLM 不需要改动**,容器里现有的 vLLM 即为配套版本(本工具只切换 vllm-ascend)。
5. 依赖:`pytest`、`openai`、`aisbench`、`psutil`、`filelock`、`regex`(nightly 容器已具备)。

---

## 三、Good 表(只读,由 nightly 流水线产出)

Good 表就是 nightly 的状态表(CSV),**工具只读不写**。列结构:

```csv
name,yaml/path,link,status,vLLM Git information,vLLM-Ascend Git information,time
```

示例:

```csv
name,yaml/path,link,status,vLLM Git information,vLLM-Ascend Git information,time
DeepSeek-R1-0528-W8A8,tests/e2e/nightly/single_node/models/configs/DeepSeek-R1-0528-W8A8.yaml,https://github.com/.../job/80000000001,success,ad7125a431e1...,46356897b4d8...,2026-06-15 03:40:00 +08:00
DeepSeek-R1-0528-W8A8,tests/e2e/nightly/single_node/models/configs/DeepSeek-R1-0528-W8A8.yaml,https://github.com/.../job/81510428999,failure,ad7125a431e1...,b545857e2983...,2026-06-16 03:46:00 +08:00
```

工具的查找逻辑:

- 按 `--name`(匹配 `name` 列)或 `--config-yaml`(匹配 `yaml/path` 列,支持文件名或目录后缀)定位行;
- 在所有 `status=success` 的行里取 **`time` 最新**的一条;
- 用它的 **`vLLM-Ascend Git information`** 作为 good 端点(更新的 failure 行不会干扰)。

用 `--good-table` 指定表路径(或环境变量 `BISECT_GOOD_TABLE`,默认 `/root/.cache/nightly_bisect/good_table.csv`)。
如果不想依赖表,可用 `--good-commit <sha>` 直接指定 good。

---

## 四、单机用例

在仓库根目录执行(以 DeepSeek-R1 为例):

```bash
python -m tools.bisect.auto_bisect \
    --scene single_node \
    --config-yaml DeepSeek-R1-0528-W8A8.yaml \
    --name DeepSeek-R1-0528-W8A8 \
    --bad-commit HEAD \
    --good-table /path/to/nightly_status.csv
```

- `--config-yaml`:失败用例所在 YAML(对应 `CONFIG_YAML_PATH`);**会运行该文件里的全部 test_cases**。
- `--name`:可选,用于匹配 Good 表的 `name` 列;不传则用 `--config-yaml` 去匹配 `yaml/path`。
- `--bad-commit`:默认取环境变量 `VLLM_ASCEND_REF`,否则 `HEAD`。
- good 端点不传 `--good-commit` 时自动查表。

---

## 五、多机用例

**在每个节点上都启动同一条命令**,并让所有节点指向**同一个共享目录** `--coord-dir`(各节点都能读写的网络盘/PVC)。master(`LWS_WORKER_INDEX=0`)驱动二分,其余节点自动进入 worker 循环。

```bash
python -m tools.bisect.auto_bisect \
    --scene multi_node \
    --config-yaml Qwen3-235B-W8A8.yaml \
    --bad-commit "$VLLM_ASCEND_REF" \
    --num-nodes 2 \
    --coord-dir /shared/nightly_bisect/coord
```

- `--num-nodes` 不填则自动从配置 YAML 的 `num_nodes` 字段读取;`--node-index` 不填则读 `LWS_WORKER_INDEX`(LWS 自动注入)。LWS 编排下这两个都无需手填。
- `--coord-dir` 不填时默认 `/root/.cache/nightly_bisect/coord`;LWS 下 `/root/.cache` 是共享 PVC,各节点天然共享,可不填。非 LWS 环境需手动指定共享路径。
- internal / external DP 通过 `--config-base-path`(或 yaml 路径含 `external_dp/config`)自动区分。
- 所有节点切到同一 commit 后才会开跑(屏障同步)。

> ⚠️ **常见坑(barrier timeout)**:报错 `Barrier timeout: only 1/2 nodes ready` 表示**只有 master 跑了 bisect、worker 节点没跑**。多机 bisect 要求**每个节点都启动 `auto_bisect.py --scene multi_node`**(worker 节点会自动进入 worker 循环:接收 commit→部署→上报 ready→等 master)。如果你的流水线只在 leader 上调了 bisect、worker pod 只跑了用例,worker 永远不会加入屏障,master 就会超时。修法:让流水线在**所有节点**(含 worker)都执行同一条 bisect 命令,且共享同一个 `--coord-dir`。

---

## 六、编译判定(只有动了 C++ 才编译)

每切到一个 commit,工具**根据该 commit 改动的文件类型**决定是否重新编译 vllm-ascend:

- 改动**全是非 native**(`.py`/yaml/md 等)→ **只 `git checkout`,不编译**(editable 安装即时生效);
- 改动命中 **native/构建文件**(`*.cpp/*.cc/*.cu/*.h/*.hpp/*.cuh`、`csrc/**`、`CMakeLists.txt`、`setup.py`)→ 执行 `pip install -e .` 重新编译;
- `requirements*.txt` 变化 → 只重装依赖。

相关开关(一般用默认即可):

| 开关 | 默认 | 说明 |
|---|---|---|
| `--native-check per-commit` | **per-commit** | 只看当前 commit 自身改动判断(编译最少)。可选 `since-build`:看距上次编译的累计改动(跳跃更安全) |
| `--force-initial-build` | 关 | 首轮强制全量重编译(默认信任容器已在 HEAD 构建好) |
| `--no-assume-built-head` | 关 | 不把容器当前 HEAD 当作"已构建" |

> bad 端点(==HEAD)因为容器已构建好,默认是 `already built` 直接跳过编译。

### 6.1 vLLM 版本配套检查(自动)

每切到一个 commit,工具会读取该 commit 的 `.github/vllm-release-tag.commit`(它钉死了这个 commit 配套的 vLLM tag,如 `v0.22.1`),与容器实际 vLLM(优先 `VLLM_VERSION` 环境变量,否则 `vllm.__version__`)比对:

- **配套**(release 段一致,忽略 `v` 前缀和 dev/local 后缀)→ 正常跑;
- **不配套**(如 commit 钉 `v0.22.1`、容器是 `0.21.0`)→ 该 commit **直接判 SKIP**,日志写明 `vllm version mismatch: this commit pins ... but the container has ...`,**不再浪费一次 pytest 跑出莫名的 rc=4**;
- 容器是无法解析的 dev 构建 → 宽松放行(交给 pytest 判定)。

> 这是为了解决"二分跨过 vLLM 版本变更点时,老 commit 在当前容器里跑不起来"的问题。若某端点因 vLLM 不配套被 SKIP,二分会明确报错中止(见第七节)。

---

## 七、好坏判定(verdict)

每轮跑完整个 YAML 后:

- pytest 退出码 `1`(用例断言失败/崩溃)→ **FAIL**;
- 退出码 `0` 但 `benchmark_results/` 里**任一 case 的 json `pass_fail=fail`**(精度/性能未达基线)→ **FAIL**;
- 退出码 `0` 且无回归 → **PASS**;
- 退出码 `2/3/4/5` 或超时(收集失败、conftest ImportError、环境问题等)→ **SKIP**;
- **vLLM 版本不配套**(见 6.1)→ **SKIP**(带明确原因);
- SKIP 不作为二分信号,类似 `git bisect skip`。

端点校验:开跑前先确认 bad 复现失败、good 确实通过;若任一端点是 SKIP(环境跑不起来 / vLLM 不配套),会**明确报错并中止**而不是给错误结论。

---

## 八、输出怎么看

**控制台**:逐轮打印明显标识,例如:

```text
=== Round 3: testing PR-10492 (46356897b4d8); window=[0,3] 3 left, ~2 rounds to go ===
[FAIL] PR-10492  (46356897b4d8, checkout-only, 612s) - pytest exited non-zero (rc=1)
...
================================================================
  FIRST BAD COMMIT: 46356897b4d8...
  FIRST BAD PR:     PR-10492
================================================================
```

`checkout-only` 表示这轮没编译;`rebuilt` 表示这轮重编译了。

**文件**(位于 `$BISECT_WORK_DIR/<scene>__<config_yaml>/`,默认 `BISECT_WORK_DIR=/root/.cache/nightly_bisect/runs`):

- `logs/round<N>_<sha>.log`:每轮的**编译 + pytest 全量输出**(编译期间控制台是静默的,看实时进度就 `tail -f` 这个文件);
- `state.json`:二分窗口 + 已判定结果(**被中断后原命令重跑会断点续跑**);
- `report.json`:最终结论(首个 bad commit / PR + 完整试跑历史)。

**退出码**:`0` = 成功定位首个 bad;`2` = 未定位(端点校验失败 / 区间无效 / 环境问题)。

---

## 九、全部命令行参数详解

### 9.1 必填核心参数

#### `--scene` (必填)

- **作用**:选择运行场景,决定用哪套 nightly 入口拉起、是否做多机协调。
- **取值**:`single_node`(单机,跑 `test_single_node.py`)或 `multi_node`(多机,跑 `test_multi_node.py` 并做跨节点屏障)。
- **注意**:ops 算子用例也属于 `single_node`。多机时**每个节点都要带这个参数**。

#### `--config-yaml` (必填)

- **作用**:指定失败用例所在的 YAML 文件,会被设进环境变量 `CONFIG_YAML_PATH`;**每一轮二分都完整运行这个文件里的全部 `test_cases`**(不能只跑单个 case)。
- **取值**:相对 configs 目录的文件名,如 `DeepSeek-R1-0528-W8A8.yaml`;多机时是对应配置文件名。
- **双重用途**:① 实际运行的对象;② 不传 `--name` 时,用它去匹配 Good 表的 `yaml/path` 列(支持文件名或目录后缀匹配)。
- **注意**:它也是工作目录/状态/报告的命名键(`case_key = scene::config_yaml`)。

### 9.2 二分区间端点

#### `--bad-commit`

- **作用**:二分区间的"坏"端,即当前失败的 vllm-ascend commit。
- **默认**:先取环境变量 `VLLM_ASCEND_REF`,没有则用 `HEAD`。
- **取值**:任意 git 引用——完整 sha / 短 sha / 分支名 / `HEAD` / PR 号(本地没有时工具会自动 `git fetch` PR ref)。
- **常用**:nightly 触发时一般就是 `HEAD`(容器当前停在的失败提交)。

#### `--good-commit`

- **作用**:显式指定"好"端,**跳过 Good 表查询**。
- **默认**:不传 → 从 Good 表自动取(见 `--good-table`)。
- **何时用**:你已经知道某个确定通过的 commit,或不想/没有 Good 表时。
- **注意**:必须是 `--bad-commit` 的**祖先**,否则工具报"区间无效"。

### 9.3 Good 表相关

#### `--good-table`

- **作用**:nightly 状态表 CSV 的路径,工具**只读**它来确定 good 端点。
- **默认**:环境变量 `BISECT_GOOD_TABLE`,再没有则 `/root/.cache/nightly_bisect/good_table.csv`。
- **逻辑**:按 `--name` 或 `--config-yaml` 定位行 → 在所有 `status=success` 的行里取 `time` 最新的一条 → 用它的 `vLLM-Ascend Git information` 当 good。
- **注意**:表里该用例必须**至少有一条 success 行**,否则需改用 `--good-commit`。

#### `--name`

- **作用**:用来匹配 Good 表的 `name` 列(比 `yaml/path` 更精确)。
- **默认**:`None`(不传)→ 退回用 `--config-yaml` 匹配 `yaml/path`。
- **何时用**:一个 YAML 对应多种 nightly job 名、或 `name` 与文件名不一致时,用它锁定正确的行。

### 9.4 编译控制(决定何时 `pip install -e .`)

#### `--native-check`

- **作用**:决定"用哪些改动文件"判断是否需要重新编译 vllm-ascend。
- **取值**:
    - `per-commit`(**默认**):只看**当前 commit 自身**改的文件,命中 C++/native 才编译——编译次数最少。
    - `since-build`:看**距上次编译以来累计**改的文件,二分跳跃时也不会用到过期 `.so`,更保守。
- **判定文件类型**:`*.cpp/*.cc/*.cu/*.h/*.hpp/*.cuh`、`csrc/**`、`CMakeLists.txt`、`setup.py` → 编译;纯 `.py`/yaml → 只 checkout。

#### `--force-initial-build`

- **作用**:首轮强制做一次干净的全量重编译。
- **默认**:关(工具默认信任"容器当前 HEAD 已编译好",bad 端点直接跳过编译)。
- **何时用**:你不确定容器当前的二进制是否和 HEAD 源码一致时。

#### `--no-assume-built-head`

- **作用**:关闭"把容器当前 HEAD 当作已构建基线"的假设。
- **默认**:关(即默认**启用**该假设)。
- **何时用**:容器里装的不是 HEAD、或被改过,需要工具不要偷懒跳过首轮编译时。

### 9.5 稳健性与校验

#### `--fail-confirm-retries`

- **作用**:对判为 FAIL 的 commit 额外复测几次;若复测出现 PASS,说明用例 flaky,该 commit 判 **SKIP**(不当作二分边界)。
- **默认**:`1`(FAIL 后再复测 1 次)。
- **调整**:用例偶发性强可调大(更稳但更慢);设 `0` 关闭复测(最快但易被抖动误导)。

#### `--no-verify-good` / `--no-verify-bad`

- **作用**:跳过开跑前对 good / bad 端点的复核(默认会先确认 bad 复现失败、good 确实通过)。
- **默认**:关(即默认**都会校验**)。
- **何时用**:你完全确信端点状态、想省那 1~2 次试跑时。**首次跑强烈建议保留**——能挡住"区间不成立 / 环境跑不起来"的情况(端点若是 SKIP 会明确报错中止,而不是给错误结论)。

#### `--trial-timeout-s`

- **作用**:单轮 pytest 的超时(秒);超时记为 SKIP。
- **默认**:`5400`(90 分钟)。
- **调整**:大模型起服务 + aisbench 很慢时适当调大,避免正常用例被误杀成 SKIP。

### 9.6 多机参数(`--scene multi_node` 时)

#### `--num-nodes`

- **作用**:集群节点总数;master 用它做屏障(等齐所有节点就绪才开跑)。
- **默认**:**不填则自动从多机配置 YAML 的 `num_nodes` 字段读取**(在 `internal_dp/config` 或 `external_dp/config`,或 `--config-base-path` 指定的目录里按 `--config-yaml` 找该文件)。
- **何时手填**:配置文件里没有 `num_nodes`、或你想覆盖时。单机场景固定为 1。
- **注意**:这里**不依赖** `LWS_GROUP_SIZE` 之类的环境变量(节点数的权威来源就是配置 YAML,与现有 nightly 多机逻辑一致)。

#### `--node-index`

- **作用**:当前节点编号;`0` 为 master(驱动二分),其余为 worker(进入受控循环)。
- **默认**:**不填则读环境变量 `LWS_WORKER_INDEX`(LWS 编排自动注入),没有则 `0`**。
- **何时手填**:**非 LWS 环境**(没有 `LWS_WORKER_INDEX`)时必须在每个节点上手动指定各自的索引;LWS 下无需填。

#### `--coord-dir`

- **作用**:多机共享屏障目录,各节点通过它同步"切到哪个 commit、是否就绪、本轮结束"。
- **默认**:环境变量 `BISECT_COORD_DIR`,再没有则 `/root/.cache/nightly_bisect/coord`。
- **能不能不填**:**在 LWS 环境下可以不填**——`/root/.cache` 正是各节点挂载的同一块共享 PVC(`shared-volume`,nightly 的结果也写在这里),所以默认值天然各节点共享。
- **何时必须填**:**非 LWS、或 `/root/.cache` 不是共享挂载**时,必须显式指定为一个所有节点都能读写的同一共享路径(网络盘/PVC),否则屏障同步失效。

### 9.7 路径与输出

#### `--work-dir`

- **作用**:本次 bisect 的日志/状态/报告输出根目录。
- **默认**:环境变量 `BISECT_WORK_DIR`,再没有则 `/root/.cache/nightly_bisect/runs`。
- **产物**:`<work-dir>/<scene>__<config_yaml>/` 下有 `logs/round*.log`、`state.json`(断点续跑)、`report.json`(最终结论)。

#### `--repo-dir`

- **作用**:vllm-ascend 仓库路径(所有 `git checkout`/编译/跑测都在这里)。
- **默认**:工具自身所在的仓库根目录(通常就是你 `cd` 进的 `/workspace/vllm-ascend`)。
- **何时用**:几乎不用改;除非你想让工具操作另一份仓库副本。

#### `--config-base-path`

- **作用**:覆盖 configs 的基准目录,设进环境变量 `CONFIG_BASE_PATH`;主要用于**多机 internal/external DP** 区分配置目录。
- **默认**:环境变量 `CONFIG_BASE_PATH`。
- **注意**:路径里含 `external_dp/config` 时,多机会自动选用 external DP 的 pytest 入口。

---

## 十、真实环境快速自检清单

1. `cd /workspace/vllm-ascend`(仓库根目录);
2. `pip show vllm-ascend | grep -i location` 确认是 editable 安装;
3. `--good-table` 指向真实的 nightly 状态 CSV,且里面该用例有 `success` 行;
4. 直接跑第四节命令;第一轮(bad=HEAD)应**直接进 pytest、不 pip**;
5. 想看实时进度:`tail -f $BISECT_WORK_DIR/<scene>__<yaml>/logs/round*.log`;
6. 中断了就原命令重跑(自动续跑)。
