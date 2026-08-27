# 环境检查报告模板

> ⚠️ **CANNBot 在 Step 1 填写本模板生成 `operators/{operator_name}/docs/environment.md`**
>
> 数据采集：加载 `/ascendc-env-check` skill，按 skill 指引完成 CANN 环境与 NPU 设备的检查。
>
> **字段语义**：
>
> | 模板字段 | 取值来源 |
> |---------|---------|
> | 芯片型号 | NPU 设备检查输出的 Chip Name |
> | SocVersion | NPU 设备检查输出（从 `npu-smi info` 推得；`aclrtGetSocName()` 仅运行期可用，env-check 阶段不调用） |
> | 设备数 | NPU 设备检查输出的可用设备数量 |
> | ASCEND_HOME_PATH | CANN 环境检查输出 |
> | CANN 版本 | CANN 环境检查输出 |
> | CPU 架构目录 | `ls $ASCEND_HOME_PATH` 自查（aarch64-linux / x86_64-linux） |
> | bisheng / kernel_operator.h / libs | 在 `$ASCEND_HOME_PATH/<arch>/...` 对应位置 `ls -l` / `test -x` 自查 |
> | asc-devkit 路径 | 按优先级查：`$ASC_DEVKIT_DIR` / `./asc-devkit` / 插件根目录 / `~/.config/opencode/asc-devkit` / `~/.claude/asc-devkit` |
> | API 文档数 / 示例数 | `find <asc-devkit>/docs/api -name '*.md' \| wc -l` 和 `find <asc-devkit>/examples -name '*.asc' \| wc -l` |
>
> **状态行规范**：标题下方第一行必须是 `**算子**: <name>   **状态**: ✅ 通过` 或 `**状态**: ❌ 失败`。Step 2 门禁正则：`^\*\*算子\*\*.*\*\*状态\*\*:\s*✅\s*通过`（必须含字面 "通过"，未替换占位符会被拒）。
>
> **判定规则**：
> - 任一 ❌ 错误（不含 ⚠ 警告）→ 状态为 ❌ 失败。
> - **设备数 = 0** 必须计入「错误明细」并令状态为 ❌ 失败（即使 CANN 环境检查全通过）。

---

## 模板正文（以下从「# 环境检查报告」开始，整段复制到 environment.md 后填空）

```markdown
# 环境检查报告

**算子**: {operator_name}   **状态**: <填写「✅ 通过」或「❌ 失败」，禁止保留占位符>   **时间**: {ISO8601}

## 硬件

| 项 | 值 |
|----|----|
| 芯片型号 | Ascend {chip_name}（如未检测到填 "未识别"） |
| SocVersion | {ASCEND910B / ASCEND950 / ...，未检测到填 "未知"} |
| 设备数 | {N} |

> NPU Arch / `--npu-arch` 编译参数由 Architect 在 Step 2 通过 `/npu-arch` skill 查得后写入 DESIGN.md，本章节不填。

## CANN

- ASCEND_HOME_PATH: `{path}`
- 版本: `{version}`
- CPU 架构目录: `{aarch64-linux / x86_64-linux}`

## 编译器与库

> 标 ✅ 的前提是 `test -x <path>` 通过（对可执行文件）或 `test -f <path>` 通过（对头文件/.so）；存在但不可执行 → ❌。

- ✅ / ❌ bisheng: `{path}`
- ✅ / ❌ kernel_operator.h: `{path}`
- ✅ / ❌ libregister.so: `{path}`
- ✅ / ❌ libascendcl.so: `{path}`

## asc-devkit

- ✅ / ❌ 路径: `{path}`
- API 文档: {N} 个
- 示例: {N} 个
- CMake 配置: ✅ / ❌

## 检查汇总

- 错误: {N}
- 警告: {N}

### 错误明细（如 0 则省略本节）
- ❌ {错误描述}

### 警告明细（如 0 则省略本节）
- ⚠ {警告描述}
```

---

## 成功案例：模板该如何填

举例：CANN 环境与 NPU 设备均正常，Atlas 800T A2（910B3）开发环境：

- 状态行 → `**算子**: add   **状态**: ✅ 通过   **时间**: 2026-05-26T15:00:00+08:00`
- 「硬件」→ 芯片型号 `Ascend 910B3`，SocVersion `ASCEND910B`，设备数 `8`
- 「CANN」→ `ASCEND_HOME_PATH: /usr/local/Ascend/ascend-toolkit/latest`，版本 `8.0.RC1`，CPU 架构目录 `aarch64-linux`
- 「编译器与库」→ bisheng / kernel_operator.h / libregister.so / libascendcl.so 全部 ✅
- 「asc-devkit」→ 路径 ✅，API 文档 / 示例数填入实际数量，CMake 配置 ✅
- 「检查汇总」→ 错误 `0`，警告 `0`；错误/警告明细两节均省略
- CANNBot 通过门禁正则后进入 Step 2

## 失败案例：模板该如何填

举例：当 CANN 环境检查报 `ASCEND_HOME_PATH 未设置`：

- 状态行 → `**状态**: ❌ 失败`
- 「CANN」章节 → `ASCEND_HOME_PATH: 未设置`
- 「编译器与库」章节 → 全部标 ❌
- 「检查汇总」错误数 ≥ 1，错误明细列出 `ASCEND_HOME_PATH 未设置`
- CANNBot 读完此 environment.md 后告知用户失败原因，**禁止进入 Step 2**

另一种失败形态：CANN 环境通过但 NPU 设备数为 0：

- 状态行 → `**状态**: ❌ 失败`
- 「硬件」→ 设备数 `0`
- 「检查汇总」错误 ≥ 1，错误明细包含 `NPU 设备数为 0，无法运行算子`
- CANNBot 按「NPU 设备不可用」路径处理，**禁止进入 Step 2**
