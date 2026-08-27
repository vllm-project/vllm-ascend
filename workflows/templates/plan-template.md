# {operator_name} 算子开发计划

> ⚠️ `{operator_name}` → 实际算子名称。本文档在开发流程中持续更新。

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | {operator_name} |
| 数学公式 | y = f(x) |
| 输入 | x1: shape=[...], dtype=... |
| 输出 | y: shape=[...], dtype=... |
| 算子类别 | Elementwise / Reduction / Broadcast / ... |
| 需求类型 | 特定用例 / 通用 |

---

## 2. 文件清单

| 文件 | 状态 |
|------|------|
| `kernel/{operator_name}_tiling.h` — Tiling 结构体（kernel/host 共用） | ⬜ |
| `kernel/{operator_name}_kernel.asc` — Kernel 计算逻辑 | ⬜ |
| `host/{operator_name}.asc` — Host + main 入口 | ⬜ |
| `torch_library/{operator_name}_torch.cpp` — PyTorch host（Tiling + launch） | ⬜ |
| `torch_library/register.cpp` + `torch_library/ops.h` — TORCH_LIBRARY 注册 | ⬜ |
| `CMakeLists.txt` — 双 target（可执行文件 + .so） | ⬜ |
| `run.sh` + `scripts/gen_data.py` + `scripts/verify_result.py` | ⬜ |
| `scripts/test_torch.py` — PyTorch 通路测试 | ⬜ |

---

## 3. 测试计划

精度标准：FP32 atol=1e-6, rtol=1e-6

**Golden 计算**：定义在 `scripts/golden.py` 中，gen_data.py 和 test_torch.py 共用。

**用例（T=可执行文件, P=PyTorch, 1:1 对应）**：
测试用例必须可追溯到 DESIGN.md §0.2 用户原始需求（覆盖率需 100%）。

| 编号 | 用例 | 需求追溯 | 输入 | 预期输出 |
|-----|------|---------|------|---------|
| T1/P1 | 随机数据 | （标注覆盖 DESIGN.md §0.2 第几条） | randn(...) | y=f(x) |
| T2/P2 | 零值 | （同上） | zeros(...) | y=f(0) |
| T3/P3 | 正负混合 | （同上） | randn, -randn | y=f(x,-x) |

---

## 4. 开发进度

| 阶段 | 检查项 | 状态 |
|------|--------|------|
| 框架搭建 | 工程创建 + CMake 双 target + 空 Kernel 编译通过 | ⬜ |
| Kernel 实现 | TilingData + Host Tiling + Kernel Compute + 编译通过 | ⬜ |
| 可执行文件验证 | T1-T3 全部通过 | ⬜ |
| PyTorch 验证 | TORCH_LIBRARY 注册 + `torch.ops.npu.{operator_name}()` 可调用 + P1-P3 全部通过 | ⬜ |
| 性能验收 | msprof 采集 + 数据归档 + 达标判定 | ⬜ |

---

## 5. 已知问题和决策记录

| 日期 | 问题/决策 | 说明 |
|------|----------|------|

---

## 6. 测试结果

### 6.1 可执行文件通路

**状态**: ⬜ | **脚本**: run.sh + scripts/verify_result.py

| 编号 | 用例 | 需求追溯 | 结果 | Max Diff |
|-----|------|---------|------|----------|
| T1 | 随机数据 | （标注覆盖 DESIGN.md §0.2 第几条） | ⬜ | |
| T2 | 零值 | （同上） | ⬜ | |
| T3 | 正负混合 | （同上） | ⬜ | |

### 6.2 PyTorch 通路

**状态**: ⬜ | **脚本**: scripts/test_torch.py | **约束**: 与 §6.1 逐行对应，相同输入和 golden

| 编号 | 用例 | 需求追溯 | 结果 | Max Diff |
|-----|------|---------|------|----------|
| P1 | 随机数据 | （标注覆盖 DESIGN.md §0.2 第几条） | ⬜ | |
| P2 | 零值 | （同上） | ⬜ | |
| P3 | 正负混合 | （同上） | ⬜ | |

### 6.3 产物 & 执行状态

- [ ] `build/{operator_name}` 可执行文件存在
- [ ] `build/lib{operator_name}_ops.so` 存在
- [ ] `torch.ops.load_library` + `torch.ops.npu.{operator_name}` 可调用

| 通路 | 状态 | 运行时间 | 跳过原因 |
|------|------|---------|---------|
| 可执行文件 | ⬜ | | |
| PyTorch | ⬜ | | |

---

## 7. 性能验收

**状态**: ⬜ | **数据**: docs/perf/round_NNN/

| 指标 | 值 | 判定 |
|------|------|------|
| Task Duration | | |
| Block Dim | | |
| 主导流水 | | |

**达标判定**: ⬜ | **理由**:

---

## 8. 汇总

| 通路 | 用例数 | 通过 | 失败 | 状态 |
|------|--------|------|------|------|
| 可执行文件 | | | | ⬜ |
| PyTorch | | | | ⬜ |
| 性能 | | | | ⬜ |
