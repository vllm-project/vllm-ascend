# Ascend C 算子注册与 TilingKey 模板约定

<适用>
语言: C++
侧别: Tiling, Kernel, Host
领域: true
触发: ASCENDC_TPL_ARGS_DECL, ASCENDC_TPL_DTYPE_DECL, ASCENDC_TPL_DATATYPE_DECL, TILING_KEY_IS, REG_OP, IMPL_OP_INFERSHAPE, InferDataType, _apt
默认启用: true
适用场景: op_kernel/**/*.cpp, op_kernel/**/*tiling_key*.h, op_kernel/**/*struct*.h, op_host/*.cpp, op_graph/*.cpp
介绍: 四条代码检视规则：1) TilingKey 注册禁止包含 dtype；2) Kernel 禁止 TILING_KEY_IS；3) REG_OP 必须实现 InferDataType；4) Kernel 文件禁止 _apt 后缀
</适用>

<检视负载>
通用检视子 agent 检视条款容量上限: 4
</检视负载>

> **适用场景**：涉及 TilingKey 模板注册、Kernel 派发宏、算子 REG_OP 注册、Kernel 文件命名的代码
>
> **说明**：本规范覆盖算子注册与 TilingKey 模板系统的工程约定，条款标注适用范围：`[适用: Tiling]` / `[适用: Kernel]` / `[适用: Host]`

## 快速索引

| 规范编号 | 规范名称 | 类别 | 侧别 | 严重级别 |
|---------|---------|------|------|---------|
| 1 | ASCENDC_TPL_ARGS_DECL 禁止包含算子数据类型 | TilingKey注册 | Tiling | 高 |
| 2 | Kernel 代码禁止使用 TILING_KEY_IS 宏 | Kernel派发 | Kernel | 高 |
| 3 | REG_OP 注册的算子必须实现 InferDataType | 算子注册 | Host | 高 |
| 4 | Kernel 文件禁止 _apt 后缀 | 命名规范 | Kernel | 高 |

---

## 详细规范

### 1. ASCENDC_TPL_ARGS_DECL 禁止包含算子数据类型 `[适用: Tiling]`

> **注意**：`ASCENDC_TPL_DTYPE_DECL` 定义的 **INT 索引类型**（如 INDEX_DTYPE: `TPL_INT32` / `TPL_INT64`）仍允许出现，本条只关注算子**运算数据类型**（fp16/bf16/fp32/int8 等）。

**严重级别**：高

**问题描述**

`ASCENDC_TPL_ARGS_DECL` 宏中直接使用 `ASCENDC_TPL_DTYPE_DECL` 或 `ASCENDC_TPL_DATATYPE_DECL` 声明算子数据类型（fp16、bf16、fp32 等），会使得数据类型的枚举值直接暴露为 TilingKey 枚举组合。这违反 TilingKey 不耦合算子语义类型的设计原则。正确做法是将 dtype 差异编码为整型 TilingKey（如 `schMode`），由 Host 侧 `ChooseTilingKey` 统一决策和派发。

**检查点**

| 序号 | 检查项 | 判断依据 |
|------|--------|---------|
| 1 | `ASCENDC_TPL_ARGS_DECL` 内是否出现 `ASCENDC_TPL_DTYPE_DECL` | 查找是否声明了 fp16/bf16/fp32 等运算数据类型枚举 |
| 2 | `ASCENDC_TPL_ARGS_DECL` 内是否出现 `ASCENDC_TPL_DATATYPE_DECL` | 查找是否通过 `C_DT_*` / `ASCENDC_TPL_INPUT(n)` 绑定了输入 dtype |
| 3 | `ASCENDC_TPL_SEL` 内是否使用 `ASCENDC_TPL_DTYPE_SEL` / `ASCENDC_TPL_DATATYPE_SEL` | 确认选择器未展开数据类型的枚举组合 |

**注意**：以下不视为违规：
- `ASCENDC_TPL_DTYPE_DECL` 用于 **索引/中间类型**（如 `INDEX_DTYPE: TPL_INT32, TPL_INT64`），属于合理用途

**反例**

```
optim/apply_rms_prop/op_kernel/arch35/apply_rms_prop_tiling_key.h
activation/relu6/op_kernel/arch35/relu6_tiling_key.h
activation/gelu_v2/op_kernel/arch35/gelu_v2_struct.h
optim/apply_adamax/op_kernel/arch35/apply_adamax_tiling_key.h
loss/sigmoid_cross_entropy_with_logits/op_kernel/arch35/sigmoid_cross_entropy_with_logits_struct.h
```

```cpp
// 反例1: ASCENDC_TPL_DATATYPE_DECL 直接声明 fp16/fp32/bf16 作为 TilingKey（relu6_tiling_key.h）
ASCENDC_TPL_ARGS_DECL(Relu6,
    ASCENDC_TPL_DATATYPE_DECL(D_T, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_BF16, ASCENDC_TPL_INPUT(0))
);

// 反例2: ASCENDC_TPL_DTYPE_DECL 直接声明 fp16/bf16/fp32 作为 TilingKey（sigmoid_cross_entropy_with_logits_struct.h）
ASCENDC_TPL_ARGS_DECL(SigmoidCrossEntropyWithLogits,
    ASCENDC_TPL_DTYPE_DECL(dType, TPL_FP16, TPL_BF16, TPL_FP32)
);

// 反例3: ASCENDC_TPL_DATATYPE_DECL 绑定输入0的 dtype（apply_rms_prop_tiling_key.h）
ASCENDC_TPL_ARGS_DECL(ApplyRmsProp,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_UINT_DECL(BUFFER_MODE, 8, ASCENDC_TPL_UI_LIST, 0, 1)
);
```

**正例**

```
pooling/adaptive_avg_pool2d/op_kernel/arch35/adaptive_avg_pool2d_struct.h
pooling/max_pool_grad/op_kernel/arch35/max_pool_grad_struct.h
pooling/avg_pool_v2_grad/op_kernel/arch35/avg_pool_v2_grad_tiling_key.h
pooling/adaptive_pool3d_common/op_kernel/arch35/adaptive_pool3d_tiling_struct.h
```

```cpp
// 正例1: 全部用 ASCENDC_TPL_UINT_DECL 整型编码，无 dtype 声明
// （pooling/adaptive_avg_pool2d/op_kernel/arch35/adaptive_avg_pool2d_struct.h）
#define TPL_INT32_UINT32 0
#define TPL_INT64_UINT64 1
#define TPL_SMALL_KERNEL 0
#define TPL_BIG_KERNEL 1
#define TPL_SIMT_KERNEL 2
ASCENDC_TPL_ARGS_DECL(AdaptiveAvgPool2d,
    ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, 2, ASCENDC_TPL_UI_LIST, TPL_SMALL_KERNEL, TPL_BIG_KERNEL, TPL_SIMT_KERNEL),
    ASCENDC_TPL_UINT_DECL(DTYPE_MODE, 3, ASCENDC_TPL_UI_LIST, TPL_INT32_UINT32, TPL_INT64_UINT64),
    ASCENDC_TPL_UINT_DECL(NC_FACTOR, 1, ASCENDC_TPL_UI_LIST, TPL_NC_FACTOR_64, TPL_NC_FACTOR_128),
    ASCENDC_TPL_UINT_DECL(BIG_KERNEL_COPY_MODE, 1, ASCENDC_TPL_UI_LIST, TPL_BIG_KERNEL_NDDMA, TPL_BIG_KERNEL_COPYPAD),
);

// 正例2: 仅有 ASCENDC_TPL_UINT_DECL，无任何 dtype 声明（max_pool_grad_struct.h）
// op_kernel/arch35/max_pool_grad_struct.h ✅ 全部用 UINT
```

**检视方法**

```bash
# 1. 在 tiling_key 文件中查找 ASCENDC_TPL_DTYPE_DECL 或 ASCENDC_TPL_DATATYPE_DECL
grep -n "ASCENDC_TPL_DTYPE_DECL\|ASCENDC_TPL_DATATYPE_DECL" op_kernel/**/*tiling_key*.h op_kernel/**/*struct*.h 2>/dev/null

# 2. 对命中条目逐一判断是否属于算子运算数据类型（排除 INDEX_DTYPE 等索引类型）
# 3. 同时检查 ASCENDC_TPL_SEL 中是否有对应的 ASCENDC_TPL_DTYPE_SEL / ASCENDC_TPL_DATATYPE_SEL
```

**逐步检视**：
1. 在 `tiling_key.h` / `struct.h` 中找到所有 `ASCENDC_TPL_ARGS_DECL` 宏
2. 检查其参数列表中是否有 `ASCENDC_TPL_DTYPE_DECL(...)` 或 `ASCENDC_TPL_DATATYPE_DECL(...)`
3. 对命中的条目判断：声明的类型值是否包含 fp16/bf16/fp32/int8 等**算子运算数据类型**——若是则为违规
4. 排除 `INDEX_DTYPE: TPL_INT32, TPL_INT64` 等索引/中间类型定义（非算子运算数据类型）

---

### 2. Kernel 代码禁止使用 TILING_KEY_IS 宏 `[适用: Kernel]`

**严重级别**：高

**问题描述**

Kernel 入口函数（`op_kernel/*.cpp`）中使用 `TILING_KEY_IS(N)` 宏按硬编码的 TilingKey 数值分支选择不同实现路径，直接耦合了 Kernel 实现与具体 TilingKey 值。正确做法是通过模板参数（如 `uint32_t schMode`）利用 `if constexpr` 实现编译期分支派发，Kernel 逻辑不感知 TilingKey 具体数值。

**检查点**

| 序号 | 检查项 | 判断依据 |
|------|--------|---------|
| 1 | Kernel `.cpp` 中是否出现 `TILING_KEY_IS(` | 搜索 Kernel 入口函数中的 `TILING_KEY_IS` 调用 |
| 2 | 是否使用硬编码数值（如 `TILING_KEY_IS(0)`）或宏常量数值（如 `TILING_KEY_IS(800001)`） | 判断是否直接引用了具体 TilingKey 值 |

**反例**

```
pooling/max_pool_with_argmax/op_kernel/max_pool_with_argmax.cpp
pooling/max_pool_v3/op_kernel/max_pool_v3_apt.cpp
pooling/avg_pool_v2/op_kernel/avg_pool_v2_apt.cpp
quant/dynamic_quant_v2/op_kernel/dynamic_quant_v2.cpp
quant/flat_quant/op_kernel/flat_quant.cpp
rnn/single_layer_lstm_grad/op_kernel/single_layer_lstm_grad.cpp
```

```cpp
// 反例1: 大量硬编码宏常量的 TILING_KEY_IS 分支（max_pool_with_argmax.cpp）
#define MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C 800001
#define MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C_PAD 800002
// ...
__global__ __aicore__ void max_pool_with_argmax(GM_ADDR x, GM_ADDR y, GM_ADDR argmax,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C)) {        // ❌
        // ...
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C_PAD)) {  // ❌
        // ...
    } else if (TILING_KEY_IS(MAX_POOL_WITH_ARGMAX_TILING_KEY_NHWC_BIG_C_NANPROP)) {  // ❌
        // ...
    }
    // ... 更多 TILING_KEY_IS 分支
}

// 反例2: 在 Kernel 中使用 TILING_KEY_IS 按硬编码数值分支（quant/flat_quant.cpp）
__global__ __aicore__ void flat_quant(...)
{
    if (TILING_KEY_IS(1)) {                    // ❌
        // ...
    } else if (TILING_KEY_IS(2)) {             // ❌
        // ...
    } else if (TILING_KEY_IS(3)) {             // ❌
        // ...
    }
}

// 反例3: LSTM grad 中大量 TILING_KEY_IS 硬编码组合（rnn/single_layer_lstm_grad.cpp）
__global__ __aicore__ void single_layer_lstm_grad(...)
{
    if (TILING_KEY_IS(0)) {                    // ❌
    } else if (TILING_KEY_IS(1)) {             // ❌
    } else if (TILING_KEY_IS(10)) {            // ❌
    } // ... 更多分支
}
```

**正例**

```
pooling/max_pool_grad/op_kernel/max_pool_grad.cpp
conv/conv3d_transpose_v2/op_kernel/conv3d_transpose_v2.cpp
conv/quant_conv3d/op_kernel/quant_conv3d.cpp
pooling/max_pool3d_grad/op_kernel/max_pool3d_grad.cpp
conv/conv3d_backprop_input_v2/op_kernel/conv3d_backprop_input_v2.cpp
```

```cpp
// 正例1: 使用模板参数 + if constexpr 编译期分支（max_pool_grad.cpp）
// Kernel 通过模板参数 KERNEL_MODE/FORMAT/INDICES_DTYPE 做编译期分支，无 TILING_KEY_IS
template <
    uint64_t KERNEL_MODE = TPL_SIMT_KERNEL, uint64_t FORMAT = TPL_NCHW_FORMAT,
    uint64_t INDICES_DTYPE = TPL_INT32, uint64_t IS_CHECK_RANGE = TPL_NO_CHECK_RANGE>
__global__ __aicore__ void max_pool_grad(
    GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grads, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MaxPoolGradWithArgmaxSimtTilingCommonData);

    if constexpr (KERNEL_MODE == TPL_SIMT_KERNEL) {            // ✅ 模板参数编译期分支
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxSimtTilingCommonData, tilingData, tiling);
        // ...
    }
}

// 正例2: 模板参数分支，无 TILING_KEY_IS（conv3d_transpose_v2.cpp）
__global__ __aicore__ void conv3d_transpose_v2(GM_ADDR x, GM_ADDR filter, GM_ADDR y, ...)
{
    // ...
    if constexpr (FORMAT_Y == FORMAT_NCDHW) {                 // ✅ 模板参数编译期分支
        // ...
    }
}

// 正例3: 模板参数直接传入，无 TILING_KEY_IS（quant_conv3d.cpp）
__global__ __aicore__ void quant_conv3d(GM_ADDR x, GM_ADDR filter, ...)
{
    // ...
    if constexpr (GroupType == CONV_GROUP_TYPE_NORMAL_CONV) {  // ✅
        // ...
    }
}
```

**检视方法**

```bash
# 1. 在 Kernel 源文件中查找 TILING_KEY_IS
grep -rn "TILING_KEY_IS(" op_kernel/*.cpp op_kernel/**/*.cpp 2>/dev/null

# 2. 确认命中文件为 Kernel 入口函数（包含 __global__ __aicore__ 的 .cpp）
# 3. 排除 op_host 目录下的文件（Tiling 侧使用 TILING_KEY_IS 是合理的）
```

**逐步检视**：
1. 在 `op_kernel/` 目录下搜索 `TILING_KEY_IS(` 的所有出现
2. 排除 `op_host/` 目录下的文件（Host/Tiling 侧允许使用）
3. 检查 Kernel 入口函数是否改用 `if constexpr (schMode == ...)` 基于模板参数编译期分支
4. 若 Kernel 仅做简单透传无分支，检查模板参数是否为 `uint32_t schMode` 或 `ASCENDC_TPL_KERNEL_TYPE`

---

### 3. REG_OP 注册的算子必须实现 InferDataType `[适用: Host]`

**严重级别**：高

**问题描述**

当算子通过 `REG_OP(OP_NAME)` 在 `op_graph/xxx_proto.h` 中注册后，必须实现 `InferDataType` 函数。
- **优先位置**：`op_graph/xxx_graph_infer.cpp`（独立 infer 文件，使用 `IMPL_OP(OpName).InferDataType(...)` 注册）
- **兼容位置**：`op_host/xxx_infershape.cpp`（与 InferShape 同文件，使用 `IMPL_OP_INFERSHAPE(OpName).InferShape(...).InferDataType(...)` 注册）

缺少 `InferDataType` 会导致框架无法正确推导输出 dtype，引发运行时 dtype 不匹配错误。

**检查点**

| 序号 | 检查项 | 判断依据 |
|------|--------|---------|
| 1 | `op_graph/*_proto.h` 中是否有 `REG_OP(OpName)` | 确认算子已注册 |
| 2 | `op_host/*_infershape.cpp` / `op_graph/*_graph_infer.cpp` 中是否有 `InferDataType` 函数定义 | 确认函数实现存在 |
| 3 | `IMPL_OP_INFERSHAPE` 或 `IMPL_OP` 是否链式调用了 `.InferDataType(...)` | 确认函数已注册到框架 |

**正例**

**优先位置**（`op_graph/xxx_graph_infer.cpp`）：

```
activation/celu_v2/op_graph/celu_v2_graph_infer.cpp
```

**兼容位置**（`op_host/xxx_infershape.cpp`）：

```
pooling/adaptive_avg_pool2d/op_host/adaptive_avg_pool2d_infershape.cpp
pooling/max_pool_grad_with_argmax/op_host/max_pool_grad_with_argmax_infershape.cpp
pooling/max_pool_with_argmax/op_host/max_pool_with_argmax_infershape.cpp
pooling/adaptive_max_pool3d/op_host/adaptiva_max_pool3d_infershape.cpp
pooling/max_pool_grad_with_argmax_v3/op_host/max_pool_grad_with_argmax_v3_infershape.cpp
```

```cpp
// 正例1【优先】: 独立 infer 文件，IMPL_OP 模式（celu_v2_graph_infer.cpp）
static ge::graphStatus InferDataTypeCeluV2(gert::InferDataTypeContext* context)
{
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    context->SetOutputDataType(IDX_0, sizeDtype);
    return GRAPH_SUCCESS;
}
IMPL_OP(CeluV2).InferDataType(InferDataTypeCeluV2);
//      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 有 InferDataType ✅

// 正例2【兼容】: IMPL_OP_INFERSHAPE 链式调用（adaptive_avg_pool2d_infershape.cpp）
static ge::graphStatus InferDtype4AdaptiveAvgPool2d(gert::InferDataTypeContext* context)
{
    const auto input_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, input_dtype);
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(AdaptiveAvgPool2d)
    .InferShape(InferShape4AdaptiveAvgPool2d)
    .InferDataType(InferDtype4AdaptiveAvgPool2d);
//  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 有 InferDataType ✅

// 正例3【兼容】: IMPL_OP_INFERSHAPE 链式调用（max_pool_grad_with_argmax_infershape.cpp）
static ge::graphStatus InferDataTypeForMaxPoolGradWithArgmax(gert::InferDataTypeContext* context)
{
    const auto xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, xDtype);
    context->SetOutputDataType(1, DT_INT32);
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MaxPoolGradWithArgmax)
    .InferShape(InferShapeForMaxPoolGradWithArgmax)
    .InferDataType(InferDataTypeForMaxPoolGradWithArgmax);
```

**反例**

> 以下算子在 `op_graph/*_proto.h` 中有 `REG_OP`，但**整个算子目录下（含 `op_host/` 和 `op_graph/`）均无 `InferDataType` 实现**。

```
pooling/avg_pool_v2/op_host/avg_pool_v2_infershape.cpp      ← 仅 .InferShape()，无 .InferDataType()
pooling/avg_pool/op_host/avg_pool_infershape.cpp             ← 仅 .InferShape()，无 .InferDataType()
pooling/avg_pool3_d/op_host/avg_pool3_d_infershape.cpp       ← 仅 .InferShape()，无 .InferDataType()
pooling/avg_pool3_d_grad/op_host/avg_pool3_d_grad_infershape.cpp  ← 仅 .InferShape()，无 .InferDataType()
```

```cpp
// 反例1: 仅有 InferShape，整个算子目录无 InferDataType（avg_pool_v2_infershape.cpp）
// op_graph/avg_pool_v2_proto.h: REG_OP(AvgPoolV2) ✅ 已注册
// 检查 range: 整个 pooling/avg_pool_v2/ 目录 — 无任何 InferDataType ❌
IMPL_OP_INFERSHAPE(AvgPoolV2).InferShape(InferShapeForAvgPoolV2);
//                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                            只有 InferShape，缺少 .InferDataType(...) ❌

// 反例2: 仅有 InferShape，缺失 InferDataType（avg_pool_infershape.cpp）
// op_graph/avg_pool_proto.h: REG_OP(AvgPool) ✅ 已注册
// 检查 range: 整个 pooling/avg_pool/ 目录 — 无任何 InferDataType ❌
IMPL_OP_INFERSHAPE(AvgPool).InferShape(InferShapeForAvgPool);
//                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                         缺少 .InferDataType(...) ❌

// 反例3: 仅有 InferShape，缺失 InferDataType（avg_pool3_d_infershape.cpp）
// op_graph/avg_pool3_d_proto.h: REG_OP(AvgPool3D) ✅ 已注册
// 检查 range: 整个 pooling/avg_pool3_d/ 目录 — 无任何 InferDataType ❌
IMPL_OP_INFERSHAPE(AvgPool3D).InferShape(InferShapeForAvgPool3D);
//                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                           缺少 .InferDataType(...) ❌
```

**检视方法**

```bash
# 1. 查找所有 REG_OP 注册
grep -rn "REG_OP(" --include='*proto*.h' op_graph/ 2>/dev/null

# 2. 对每个已注册的算子，搜索整个算子目录下是否有 InferDataType
grep -rn "InferDataType" <op_dir>/ --include='*.cpp' --include='*.h' 2>/dev/null | grep -v test

# 3. 无任何匹配即为违规（需搜索算子目录下的所有文件，不只看 infershape.cpp）
```

**逐步检视**：
1. 在 `op_graph/*_proto.h` 中找到 `REG_OP(OpName)`
2. **优先**在 `op_graph/xxx_graph_infer.cpp` 中检查是否定义了 `InferDataType` 函数并用 `IMPL_OP(OpName).InferDataType(...)` 注册
3. **兼容**在 `op_host/xxx_infershape.cpp` 中检查 `IMPL_OP_INFERSHAPE(OpName)` 是否链式调用了 `.InferDataType(函数名)`
4. 仅 `.InferShape(...)` 无 `.InferDataType(...)` 则判为违规

---

### 4. Kernel 文件禁止 _apt 后缀 `[适用: Kernel]`

> **排除**：ops-transformer 仓暂不遵守此规则。

**严重级别**：高

**问题描述**

`op_kernel/` 下的 Kernel 入口文件应使用标准命名 `xxx.cpp`，禁止添加 `_apt` 后缀（如 `xxx_apt.cpp`）。`_apt` 后缀源自 Adaptive Tiling Pattern 的旧写法，现已不推荐使用。Kernel 入口文件命名为 `{op_name}.cpp` 即可，无需额外后缀。

**检查点**

| 序号 | 检查项 | 判断依据 |
|------|--------|---------|
| 1 | `op_kernel/` 目录下是否存在 `*_apt.cpp` 文件 | 文件名为违规 |

**正例**

```
pooling/max_pool_grad/op_kernel/max_pool_grad.cpp
pooling/avg_pool3_d/op_kernel/avg_pool3_d.cpp
pooling/max_pool_with_argmax/op_kernel/max_pool_with_argmax.cpp
pooling/max_pool_grad_with_argmax/op_kernel/max_pool_grad_with_argmax.cpp
pooling/adaptive_max_pool3d/op_kernel/adaptive_max_pool3d.cpp
conv/quant_conv3d/op_kernel/quant_conv3d.cpp
```

```cpp
// 正例: 标准命名，无需 _apt 后缀
// pooling/avg_pool3_d/op_kernel/avg_pool3_d.cpp ✅
// pooling/max_pool_grad/op_kernel/max_pool_grad.cpp ✅
// pooling/max_pool_with_argmax/op_kernel/max_pool_with_argmax.cpp ✅
```

**反例**

```
pooling/avg_pool_v2/op_kernel/avg_pool_v2_apt.cpp
pooling/max_pool_v2/op_kernel/max_pool_v2_apt.cpp
pooling/max_pool_v3/op_kernel/max_pool_v3_apt.cpp
pooling/avg_pool/op_kernel/avg_pool_apt.cpp
pooling/avg_pool3_d/op_kernel/avg_pool3_d_apt.cpp
pooling/avg_pool_grad/op_kernel/avg_pool_grad_apt.cpp
pooling/adaptive_avg_pool2d/op_kernel/adaptive_avg_pool2d_apt.cpp
pooling/adaptive_max_pool3d/op_kernel/adaptive_max_pool3d_apt.cpp
conv/deformable_offsets/op_kernel/deformable_offsets_apt.cpp
quant/dynamic_quant_v2/op_kernel/dynamic_quant_v2_apt.cpp
```

```cpp
// 反例1: 文件名带 _apt 后缀（avg_pool_v2_apt.cpp）❌
// 文件路径: pooling/avg_pool_v2/op_kernel/avg_pool_v2_apt.cpp

// 反例2: 文件名带 _apt 后缀（avg_pool_apt.cpp）❌
// 文件路径: pooling/avg_pool/op_kernel/avg_pool_apt.cpp

// 反例3: 文件名带 _apt 后缀（avg_pool3_d_apt.cpp）❌
// 文件路径: pooling/avg_pool3_d/op_kernel/avg_pool3_d_apt.cpp
// 注意: 同目录下 avg_pool3_d.cpp 是正例 ✅
```

**检视方法**

```bash
# 1. 在 op_kernel 目录下搜索 _apt 后缀文件
find . -path '*/op_kernel/*_apt.cpp' ! -path '*/tests/*' 2>/dev/null

# 2. 排除 ops-transformer 仓的文件
```

**逐步检视**：
1. 在 `op_kernel/` 目录下搜索文件名含 `_apt.cpp` 的文件
2. 检查文件是否在 ops-transformer 仓中（ops-transformer 暂不适用此规则）
3. 对非 ops-transformer 的 `_apt.cpp` 文件标记为违规
