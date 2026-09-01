# RegBase 路线专项检查

当代码或设计文档明确选择 RegBase 路线时触发。5 条检查规则，覆盖路线一致性、API 合规、寄存器级边界、参考实现约束和架构来源。

<适用>
语言: C++
侧别: Kernel
领域: true
触发: RegTensor, MaskReg, asc_vf_call, __simd_vf__, __simd_callee__, __VEC_SCOPE__, AscendC::Reg::, AscendC::MicroAPI::, UpdateMask, LoadAlign, StoreAlign, LoadDist, StoreDist, CastTrait, __ubuf__, arch35/, arch36/, DAV_3510, ArchVersion::V3510, CURRENT_ARCH_VERSION, __NPU_ARCH__
触发(直接): RegTensor, MaskReg, asc_vf_call, __simd_vf__, __simd_callee__, __VEC_SCOPE__, AscendC::Reg::, AscendC::MicroAPI::
  说明: 代码明确使用 RegBase 编程模型，命中即可确认
触发(架构): DAV_3510, arch35/, arch36/, ArchVersion::V3510, CURRENT_ARCH_VERSION, __NPU_ARCH__
  说明: 目标架构支持 RegBase，需结合代码判断是否实际使用
触发(间接): UpdateMask, LoadAlign, StoreAlign, LoadDist, StoreDist, CastTrait, __ubuf__
  说明: RegBase 上下文中常见，标准代码中也可能出现，需与其他信号联合判断
默认启用: true

适用架构: arch35 及以上（Ascend 950 系列及后续架构）
适用场景: Kernel 侧 arch35 及以上 RegBase 路线代码
不适用场景: Tiling 侧不涉及 RegBase 编程模型
排除场景: 仅使用标准 AscendC vector API (DataCopy/Add/Mul 等) 且无 RegTensor/VF 相关代码, DESIGN.md 明确选择 MemBase/SIMD 路线
介绍: RegBase 是基于寄存器的编程模型，与传统 MemBase（基于 UB 的 LocalTensor 计算）不同，使用 RegTensor 在寄存器上直接计算，通过 VF function（__simd_vf__）和 AscendC::Reg::* API 实现高性能计算
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 5
</检视负载>

**快速索引**

| 条例ID | 标题 | 类别 | 适用侧别 |
|--------|------|------|---------|
| RB-1 | RegBase 路线一致性 | 架构决策 | All |
| RB-2 | RegBase API 白名单 | API 合规 | Kernel |
| RB-3 | 寄存器级计算边界 | 边界安全 | Kernel |
| RB-4 | 参考实现约束一致性 | 实现合规 | Kernel |
| RB-5 | 架构来源标注 | 可追溯性 | All |

---

#### RB-1 RegBase 路线一致性

**问题描述**：技术路线必须与 DESIGN.md 的方案决策一致，禁止在同一算子内混用 RegBase 与 MemBase/SIMD 路线。

**检查方法**：
1. Read DESIGN.md，提取方案决策段落中声明的技术路线（RegBase / MemBase / SIMD）
2. Grep 代码中的路线信号：
   - RegBase 信号：`RegTensor`、`MaskReg`、`asc_vf_call`、`__simd_vf__`
   - MemBase 信号：`LocalTensor`、`DataCopy`（非 RegBase 封装）、`pipe.InitBuffer`
   - SIMD 信号：`simd_`前缀函数、`__builtin_neon_`
3. 交叉比对：
   - DESIGN 声明 RegBase 但代码含 MemBase/SIMD 信号 → FAIL
   - DESIGN 声明 MemBase 但代码含 RegBase 信号 → FAIL
   - 代码中出现两条路线的信号且无明确隔离（如不同函数/不同编译单元）→ FAIL

**错误示例**：
DESIGN.md 声明使用 RegBase 路线，但 `op_kernel/xxx.cpp` 中同时出现 `RegTensor` 和原生 `DataCopy`（非 RegBase 封装），路线混用。

**正确示例**：
DESIGN.md 声明 RegBase，代码中所有数据搬运通过 RegBase 封装的 API 完成，无原生 MemBase API 调用。

---

#### RB-2 RegBase API 白名单

**问题描述**：RegBase 路线下使用的 API 和调用结构必须来自 RegBase 文档或已验证参考实现，禁止凭函数名猜测 API 用法。

**检查方法**：
1. 加载 `ascendc-regbase-best-practice` skill，获取 API 白名单和参考实现文档
2. 提取代码中所有函数调用（Grep 函数名）
3. 逐个核对：
   - API 在白名单中 → PASS
   - API 不在白名单中但有参考实现使用 → SUSPICIOUS，标注来源
   - API 不在白名单且无参考实现 → FAIL
4. 特别关注：RegTensor 的构造方法、生命周期管理、mask 参数传递方式

**注意**：引用 API 前必须检查 API 白名单、API reference 或官方文档，不要凭函数名猜测。

---

#### RB-3 寄存器级计算边界

**问题描述**：寄存器级计算的边界处理（mask/tail）、数据搬运边界必须清晰可追溯，不能有隐含的越界或未处理尾部。

**检查方法**：
1. 追踪每个 `RegTensor` 的生命周期：创建 → 计算 → 写回
2. 检查 mask 处理：
   - 每个计算步骤是否传递了正确的 mask 参数
   - tail 元素（不满一个完整寄存器宽度的尾部）是否有显式处理
3. 检查数据搬运边界：
   - 从 GM 到寄存器的搬运长度是否与 Tiling 计算的当前块大小一致
   - 写回 GM 时是否只写入有效数据（非整个寄存器宽度）
4. Grep `repeatTimes`、`mask`、`tail` 关键词，确认每个使用点有边界计算

**错误示例**：
```cpp
auto reg_a = RegTensor::Load(gm_ptr, block_size);  // block_size 可能不满足对齐
reg_a.Compute();                                     // 未传递 mask，尾部元素计算无效数据
RegTensor::Store(reg_a, gm_out, reg_a.Size());       // 写回整个寄存器宽度，含无效尾部
```

**正确示例**：
```cpp
auto reg_a = RegTensor::Load(gm_ptr, block_size);
reg_a.Compute(mask);                                  // 显式传递 mask
RegTensor::Store(reg_a, gm_out, valid_count);         // 只写回有效元素数
```

---

#### RB-4 参考实现约束一致性

**问题描述**：代码实现必须与已选 RegBase 参考实现的约束一致，不能只照搬设计伪代码而忽略真实工程模板和 API 签名。

**检查方法**：
1. 从 DESIGN.md 提取伪代码或算法描述
2. 从 `ascendc-regbase-best-practice` skill 获取对应参考实现的约束
3. 逐项比对：
   - 伪代码中的数据布局假设是否与 RegBase API 的实际要求一致
   - 伪代码中的循环结构是否满足 RegBase 的流水线约束
   - 伪代码中的内存分配是否与 RegBase 的寄存器分配模型匹配
4. 代码偏离伪代码时，偏离是否有合理理由并记录

---

#### RB-5 架构来源标注

**问题描述**：架构判断必须显式说明来源。如果某条经验或约束来自兼容路径而不是主路径 `DAV_3510 / RegBase`，需要说清楚。

**检查方法**：
1. 检查代码注释和 DESIGN.md 中的架构相关描述
2. 确认每个架构约束标注了来源：
   - 主路径（DAV_3510 + RegBase 原生）→ 标注 `[RegBase-native]`
   - 兼容路径（通过兼容层支持的旧架构）→ 标注 `[compat]` 并说明兼容的具体架构
   - 通用约束（不限架构）→ 标注 `[general]`
3. 未标注来源的架构约束 → SUSPICIOUS

**注意**：兼容路径的约束可能与主路径不同（如寄存器数量、支持的 dtype），混用会导致运行时错误。
