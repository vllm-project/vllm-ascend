s# Ascend C Kernel 直调开发指南

> 本文档是编码规范、工程配置、代码审查和 API 验证的单一参考源（Single Source of Truth）。

---

## 1. 编码规范

### 1.1 文件与命名

**文件扩展名**：Kernel 代码文件必须为 `.asc`（不是 `.cpp`、`.cc`）。ASC 编译器只识别 `.asc` 后缀，三角括号 `<<<>>>` Kernel 启动语法也仅在 `.asc` 文件中有效。

**命名规范**：

| 类型 | 规范 | 示例 |
|-----|------|------|
| 算子名 | `{功能}_custom` | `add_custom`、`sinh_custom` |
| 类名 | 大驼峰 | `Kernel{Operator}{Branch}` |
| 函数名 | 小驼峰 | `copyIn`、`compute` |
| 变量名 | 下划线 | `row_idx`、`block_length` |
| 常量 | 全大写 | `UB_SIZE`、`BLOCK_SIZE` |

### 1.2 代码结构

参考 `/ascendc-direct-invoke-template` 的 `add_kernel/add.asc`，标准结构包含：

**代码顺序**（严格遵循）：

```
1. #include "kernel_operator.h"
2. Kernel 类定义（__aicore__ 成员函数：Init / Process / CopyIn / Compute / CopyOut）
3. 核函数入口（extern "C" __global__ __vector__ 或 __global__ __aicore__）
4. Host 端 KernelCall 封装（ACL 内存管理 + <<<>>> 调用）
5. main 函数（ACL 初始化 + Tiling 计算 + KernelCall + ACL 清理）
```

> 实现时加载 `/ascendc-direct-invoke-template`，基于 add.asc 模板修改，不要从零编写。

**关键规范**：

| 规范 | 说明 |
|------|------|
| 入口属性 | 矩阵类/矩阵向量融合类：`__global__ __aicore__`；纯向量类：`__global__ __vector__` |
| 内存管理 | `TPipe` + `TQue<VECIN/VECOUT>` |
| 双缓冲 | `TQue<VECIN/VECOUT, 1>`（depth=1），Double Buffer 由 `InitBuffer(que, 2, size)` 的 num 参数控制 |
| 前向声明 | **禁止**，Kernel 函数必须定义在调用之前 |

### 1.3 硬件适配

**核心要求**：所有硬件参数必须动态获取，**禁止写死**。

**核数获取 API**：

| 算子类型 | Host 侧 API | 说明 |
|---------|-------------|------|
| 纯向量计算（Add/Mul/Div/Reduce 等） | `ACL_DEV_ATTR_VECTOR_CORE_NUM` | Vector Core 数量 |
| 矩阵计算（MatMul/Conv 等） | `ACL_DEV_ATTR_CUBE_CORE_NUM` | Cube Core 数量 |
| 混合计算 | `ACL_DEV_ATTR_AICORE_CORE_NUM` | AI Core 数量 |

```cpp
// Host 侧（强制）
int64_t availableCoreNum = 8;
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &availableCoreNum);
uint32_t usedNumBlocks = (totalRows < availableCoreNum) ? totalRows : (uint32_t)availableCoreNum;

// Kernel 侧（强制）
blockIdx = AscendC::GetBlockIdx();
if (blockIdx >= tiling.usedCoreNum) return;  // 越界检查
```

**多核切分要求**：
- 数据量足以支撑多核
- 各核处理不同数据段（无重复计算）
- 核数 = min(数据量, 可用核数)
- 尾核数据量合理

**禁止模式**（违反 = 审查不通过）：

| 禁止模式 | 说明 | 正确做法 |
|---------|------|---------|
| `blockDim = 8;` 或 `blockDim = 4;` | 写死核数 | `aclrtGetDeviceInfo()` 动态获取 |
| `blockIdx = 0;` 或 `blockIdx = 1;` | 写死核索引 | `AscendC::GetBlockIdx()` |
| `constexpr TILE = 4096;` 或 `#define TILE 1024` | 写死分块大小 | 根据 UB 容量动态计算 |
| 任何硬编码的 UB 大小 | 写死 UB 大小 | 动态获取 |
| 纯向量算子用 `ACL_DEV_ATTR_AICORE_CORE_NUM` | 核数类型错误 | 使用 `ACL_DEV_ATTR_VECTOR_CORE_NUM` |

### 1.4 API 使用规则

**API 黑名单**（违反 = 审查不通过）：

| API | 状态 | 原因 | 替代方案 |
|-----|------|------|---------|
| `DataCopy(GM, UB)` | ❌ 禁止 | 不支持非对齐数据，易导致隐蔽 bug | `DataCopyPad` |
| `DataCopy(UB, GM)` | ❌ 禁止 | 不支持非对齐数据，易导致隐蔽 bug | `DataCopyPad` |
| `GlobalTensor::SetValue/GetValue` | ❌ 禁止（仅调试） | 效率极低 | `DataCopyPad` 批量搬运 |

```cpp
// ✅ 正确：统一使用 DataCopyPad
AscendC::DataCopyPad(xLocal, xGm, {rowsThisLoop, rLength, rLengthAlign, 0, 0});

// ❌ 错误：当数据长度不是 32 字节的倍数时会出错
AscendC::DataCopy(xLocal, xGm, dataLength);  // 危险！
```

**Host 侧禁止计算型操作**：

Host 侧代码仅包含：Tiling 参数计算、内存分配/释放、Kernel 启动（`<<<>>>`）、结果验证。

| 禁止 | 说明 |
|------|------|
| Host 侧 Transpose | CPU 三重循环交换维度 |
| Host 侧 Reshape/Pad | CPU 循环重排数据 |
| Host 侧任何对 tensor 的计算型循环 | 部署时输入/输出在 Device 上，无法搬回 Host |

**API 使用验证要求**：
- 使用任何 API 前，必须通过 `/ascendc-docs-search` skill 查阅 `{API_NAME}` API 文档
- 禁止猜测 API 用法，必须查阅文档和示例
- 优先通过 `/ascendc-docs-search` skill 参考 `$ASC_DEVKIT_DIR/docs/` 和 `$ASC_DEVKIT_DIR/examples/` 中的官方资料

### 1.5 内存与流水线

**流水线同步**（违反 = 审查不通过）：
- DataCopy 后必须使用 EnQue/DeQue 同步

**VECIN → VECOUT 数据流**（违反 = 审查不通过）：
- 最终计算结果必须写入 VECOUT buffer（outQueue），不能留在 VECIN buffer（inQueue）中
- ❌ 在 inQueue 的 xLocal 上原地计算完 → FreeTensor(xLocal) → AllocTensor(yLocal) → EnQue(yLocal)（yLocal 是空的）
- ✅ 最后一步计算直接写入 yLocal，或 DataCopy(yLocal, xLocal) 后再 EnQue

**内存安全**：
- 所有 `AllocTensor` 都有对应的 `FreeTensor`
- Queue 的 EnQue/DeQue 配对正确
- Buffer 大小计算正确（考虑对齐）
- GM 访问偏移计算正确
- UB 访问不越界

### 1.6 精度与数值稳定性

- 使用稳定的数学公式
- 处理数值溢出（exp、reduce 等）
- FP16 场景使用混合精度（如需要）
- 处理空 tensor、单元素 tensor
- 处理非对齐场景
- 多核切分的尾核处理正确
- 无除零风险

### 1.7 禁止事项

- ❌ 不要在 kernel 中使用 `std::` 命名空间函数
- ❌ 不要使用动态内存分配（`new`/`malloc`）
- ❌ 不要使用递归调用
- ❌ 不要使用未初始化的变量

---

## 2. 工程配置

### 2.1 CMakeLists.txt 配置

**强制要求**（违反 = 审查不通过）：

- 使用 `find_package(ASC REQUIRED)`
- project 包含 ASC 语言：`project(... LANGUAGES ASC CXX)`
- 使用 `add_executable`（而非自定义函数）
- 链接必需库（tiling_api, register, platform, m, dl）
- 设置 NPU 架构（`--npu-arch`）

验证命令：
```bash
python workflows/scripts/verify_cmake_config.py operators/{operator_name}/CMakeLists.txt
```

### 2.2 Tiling 计算位置

**强制要求**：Tiling 参数计算在 Host 侧完成。

- ✅ 正确：在 `xxx_common.h` 中定义 `ComputeXxxTiling()` 函数，在 `xxx.asc` 的 `main()` 中调用
- ❌ 错误：在 Kernel 的 `Init()` 或 `Process()` 中计算 Tiling 参数

### 2.3 环境配置

**关键环境变量**：

正确名称：`ASCEND_HOME_PATH`（不是 ASCEND_HOME）

```bash
# ✅ 正确
export ASCEND_HOME_PATH=/home/developer/Ascend/cann

# ❌ 错误
export ASCEND_HOME=/home/developer/Ascend/cann
```

**优先级顺序**：
1. `$ENV{ASCEND_HOME_PATH}` (环境变量)
2. `$ENV{HOME}/Ascend/cann` (默认路径)
3. `/usr/local/Ascend/cann` (备用默认)

**编译器**：`bisheng`，位于 `$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler/bin/bisheng`，CMake 通过 `find_package(ASC)` 自动发现。

**构建流程**：
```bash
mkdir build && cd build
cmake ..
make
```

**自动化验证脚本**：
```bash
bash workflows/scripts/init_operator_project.sh {operator_name}
# 环境检查：加载 /ascendc-env-check skill 完成检查，按 workflows/templates/environment-template.md
# 填写后生成 operators/{operator_name}/docs/environment.md
python workflows/scripts/verify_cmake_config.py operators/{operator_name}/CMakeLists.txt
```

---

## 3. 代码审查检查清单

### 3.1 审查通过标准

**必须满足**：
- ✅ §1 编码规范中所有"违反 = 审查不通过"项全部通过
- ✅ §2 工程配置中所有强制要求全部通过
- ✅ 总体检查勾选率 ≥ 90%
- ✅ 无明显的性能问题
- ✅ 无潜在的内存安全问题
- ✅ 自检发现的问题已当场修复（不得标记为"已知问题"跳过）

**Reviewer 提出修复建议时**，如涉及具体 API 调用，必须查阅 API 文档确认函数签名和语义，在建议中附上文档来源。禁止凭记忆推荐 API。

### 3.2 检查项

- [ ] API 使用：已查阅官方文档，参数正确（类型、对齐、取值范围）
- [ ] API 注释：每个 API 调用都有参数注释（API 名称、功能说明、参数说明）
- [ ] 代码一致性：所有分支代码风格一致、变量命名一致、文件路径与 DESIGN.md 一致
- [ ] 性能优化：使用了 Double Buffer、Buffer 大小合理、减少不必要的数据拷贝、流水线优化
- [ ] 分支数量与设计一致

---

## 4. API 语义验证方法论

### 核心原则

任何 API 调用前，必须回答 3 个问题：

```
问题 1：数据布局是什么？
    ├─ 内存如何排列？（连续 / 带 stride / 矩阵 / 多维）
    ├─ 是否对齐？（32 字节对齐 / 无对齐）
    └─ 输入输出格式？（标量 / 向量 / 矩阵）

问题 2：需要什么操作？
    ├─ 操作类型？（reduce / broadcast / elementwise / copy）
    ├─ 操作维度？
    └─ 特殊要求？（数值稳定 / 高精度）

问题 3：API 能实现吗？
    ├─ 查阅官方文档了吗？
    ├─ API 适用场景对吗？
    ├─ 满足 API 限制吗？
    └─ 有更好的选择吗？
```

**验证公式**：`正确使用 = (数据布局 ∈ API 支持范围) AND (满足所有限制条件) AND (无更好选择)`

### 验证步骤

1. **查阅官方文档**：通过 `/ascendc-docs-search` skill 查阅 `{API_NAME}` API 文档
2. **确认数据布局**：内存排列、对齐状态、维度信息
3. **验证匹配性**：数据布局与 API 能力匹配、限制条件满足
4. **确认选择**：无更好的 API 选择
5. **记录结果**：在设计文档中记录验证过程

### 常见错误模式

| 场景 | ❌ 错误 API | ✅ 正确 API | 错误原因 |
|------|-----------|-----------|---------|
| Reduce（带 stride） | `ReduceMax(dst, src, tmp, count)` | `ReduceMax<T, Pattern::RA>(...)` | Level 2 只能处理连续数据 |
| GM ↔ UB 搬运 | `DataCopy(dst, src, size)` | `DataCopyPad(dst, src, padParams)` | DataCopy 无法处理非对齐 |
| GM 单元素访问 | `xGm.SetValue/GetValue(idx, val)` | `DataCopyPad` 批量搬运 | SetValue/GetValue 效率极低 |
| 标量广播 | `Duplicate + Sub` | `Adds(dst, src, -scalar, count)` | 性能低，浪费 buffer |

---

## 5. 参考资源

| 资源类型 | 路径 | 说明 |
|---------|------|------|
| API 文档 | `$ASC_DEVKIT_DIR/docs/api/` | Ascend C API |
| 高性能模板 | `find "$ASC_DEVKIT_DIR"/examples/ -path "*/01_add/basic_api_memory_allocator_add"` | 双缓冲+流水线 |
| 各类示例 | `find "$ASC_DEVKIT_DIR"/examples/ -type d -name "00_introduction"` | 加法、减法、多输入等 |
| 调试示例 | `find "$ASC_DEVKIT_DIR"/examples/ -name "printf.asc"` | printf 调试方法 |
| 设计模板 | `workflows/templates/design-template.md` | 设计文档模板（Architect 用） |
| 工程模板 | `/ascendc-direct-invoke-template` skill | Kernel 直调工程模板（Developer 用） |
| 算子族方法论 | `/ascendc-tiling-design` 路由表 | 各算子族 methodology(Tiling 策略、特定 API 用法等),算子开发前必读对应族 |
