# 310P CI 提交看护矩阵 (Watchdog Matrix)

## 1. 触发条件与测试范围
| 触发场景 | 测试范围 | 必过条件 (Blocking) | 告警条件 (Non-blocking) |
| :--- | :--- | :--- | :--- |
| **PR 提交 (常规)** | 基于 `select_tests.py` 选出的 UT + 关联 E2E | `default_cpu_ut`, 路由命中的 `310p_x1` E2E 用例 | 无 |
| **PR 提交 (代码解耦改造)** | `.github/workflows` 变更 | 所有 CI Workflow 语法校验通过 (Lint) | 无 |
| **主干同步 (main2main)** | 全量核心回归集 | `tests/ut` (CPU) + `tests/e2e/pull_request/one_card/_310p` | `four_card/_310p` |
| **定时任务 (Daily/Nightly)** | 310P 全量 E2E 回归 | 无 (仅用于发现 Flaky) | 所有 310P E2E 用例，失败自动重试 1 次（防 Flaky）|

## 2. 测试组件级独立策略
- **Build 阶段**：镜像构建失败或 `csrc` 编译失败，直接阻断，不进入测试阶段。
- **CPU UT 阶段**：物理隔离，不受 NPU 机器可用性影响。
- **NPU UT/E2E 阶段**：依赖对应的 Runner 标签（`310p_x1`, `310p_x4`）。不同模型或组件的用例独立判定，单点失败不阻塞其余流水线。

## 3. 310P 硬件特性限制
> **Note**: 310P 硬件不支持图模式（CUDAGraph），因此所有 E2E 用例必须显式开启 `enforce_eager=True` 和 `float16` 精度，以此作为 310P CI 的基础门禁条件。
