# DequantSituQuant

## 功能说明

DequantSituQuant 算子将反量化 (Dequant)、Situ 激活函数和量化 (Quant) 三个操作融合为一个算子，减少中间数据的读写开销。

### 计算公式

```text
dequantOut = cast_to_float(x) * dequant_scale + dequant_bias
situOut = Situ(dequantOut)
out = Quant(situOut, quant_scale, quant_offset)
```

其中 Situ 激活函数为：

```python
d = last_dim(x) // 2
if activate_left:
    gate = dequantOut[..., :d]
    up   = dequantOut[..., d:]
else:
    gate = dequantOut[..., d:]
    up   = dequantOut[..., :d]
situ_a = beta * tanh(gate / beta) * sigmoid(gate)
if linear_beta > 0:
    up = linear_beta * tanh(up / linear_beta)
output = situ_a * up
```

## 支持规格

| 产品 | 支持 |
|------|------|
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 (Ascend 910B) | ✓ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 (Ascend 910_93) | ✓ |

### 数据类型支持

| 输入/输出 | 数据类型 |
|-----------|----------|
| x | INT8 |
| dequant_scale | FLOAT32 |
| dequant_bias | FLOAT32 |
| quant_scale | FLOAT32 |
| quant_offset | FLOAT32 |
| y | INT8 |
| scale | FLOAT32 |

## 输入输出

### 输入

| 参数 | 类型 | Shape | 说明 |
|------|------|-------|------|
| x | INT8 | [N..., 2H] | 输入数据，最后一维必须为偶数 |
| dequant_scale | FLOAT32 | [2H] 或 [1] | 反量化 scale |
| dequant_bias (可选) | FLOAT32 | [2H] 或 [1] | 反量化 bias |
| quant_scale (可选) | FLOAT32 | [H] 或 [1] | 量化 scale |
| quant_offset (可选) | FLOAT32 | [H] 或 [1] | 量化 offset |

### 输出

| 参数 | 类型 | Shape | 说明 |
|------|------|-------|------|
| y | INT8 | [N..., H] | 量化输出 |
| scale | FLOAT32 | [N...] | 动态量化 scale (static 模式下为无意义值) |

### 属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| beta | Float | 1.0 | Situ beta 参数 |
| linear_beta | Float | 0.0 | Situ linear_beta 参数 (≤0 不启用) |
| activate_left | Bool | false | gate 在左半 (true) 还是右半 (false) |
| quant_mode | String | "static" | 量化模式: "static" 或 "dynamic" |

## 目录结构

```text
dequant_situ_quant/
├── op_host/
│   ├── dequant_situ_quant_def.cpp
│   ├── dequant_situ_quant_tiling.h
│   ├── dequant_situ_quant_tiling.cpp
│   ├── dequant_situ_quant_infershape.cpp
│   └── CMakeLists.txt
├── op_kernel/
│   ├── dequant_situ_quant.cpp
│   └── dequant_situ_quant.h
├── op_graph/
│   └── dequant_situ_quant_proto.h
├── docs/
│   └── aclnnDequantSituQuant.md
├── dequant_situ_quant_torch_adpt.h
├── CMakeLists.txt
└── README.md
```
