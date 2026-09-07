# GMMSituA5 Operator Interface and Usage Guide

> 更新：2026-09-07。代码基准：`vllm-ascend/szy_situ_debug@08b4bcbb1a4a0d4f2ae5334f0d3501caa17a4577`。
>
> 面向：模型接入、调用端开发、接口评审及 UT 编写人员。本文说明当前实现，不将未完成的硬件验证写成支持保证。
>
> 本文侧重接口与使用方式，优先回答“做什么、传什么、怎么调、如何检查”，不展开内部流水和性能详设。

## 1. 算子功能与使用范围

`grouped_matmul_situ_quant` 将专家分组矩阵乘 GMM1、SiTU 激活和输出动态 MX 量化融合，输入已量化的 FP8 activation 与 packed FP4 专家权重，返回 FP8 payload 及 E8M0 scale。

```text
已按 expert 排好的 X + XScale + W13 + W13Scale + group_list
                           ↓
               GMM1 → BF16 → SiTU → BF16 → MX Quant
                           ↓
                  Y: FP8 + YScale: E8M0
                           ↓
                     下游 GMM2（未融合）
```

适用于 Ascend A5 / Ascend950（arch35）的 MX A8W4 SiTU 推理路径。不负责输入量化、top-k 路由、token dispatch、GMM2 或 combine；不提供训练反向接口。

### 1.1 功能表达

对专家 `e` 的有效输入行，概念上的运算是：

```text
A = decode_fp8(X) × XScale             # scale 沿 K 每 32 个元素一组
B = decode_fp4(W13[e]) × W13Scale[e]   # 每个输出通道沿 K 每 32 个元素一组
GU = round_bf16(A @ B.T)               # 宽度 N=2I
G, U = GU[:, :I], GU[:, I:]            # 固定左 gate、右 up

Z = round_bf16(
    beta * tanh(G / beta) * sigmoid(G)
    * linear_beta * tanh(U / linear_beta)
)
Y, YScale = dynamic_mx_quant(Z)         # 每 32 个元素一个 scale
```

这是接口语义，不是可用于 bit-exact 验收的 CPU 数学实现。实际累加、近似、舍入和 MX 指数选取有固定计算顺序；MX scale 不是简单的 `amax/448`。

**当前固定启用 up 分支的 tanh。** 不支持通过 `linear_beta=None` 或 `0` 关闭它，也没有 `activate_left` 参数。对照独立 `situ_mx_quant` 时必须显式传 `activate_left=True`，并使用同一组 `beta/linear_beta`。不能沿用独立接口的默认值。

### 1.2 支持边界

| 项目 | 当前范围 |
|---|---|
| 硬件 | Ascend A5 / ascend950，需对应 native 扩展和 custom OPP |
| 输入 activation | FP8 E4M3FN + E8M0 MX scale；不是 BF16 原始 hidden states |
| 权重 | FP4 E2M1，两个数打包成一字节；生产推荐预转换 NZ |
| 输出 | FP8 E4M3FN payload + E8M0 scale，固定组合 |
| 分组 | counts / cumsum；允许零 token expert 和容量大于有效行数 |
| 权重组织 | stacked Tensor；独立 NZ TensorList；ND 兼容入口见第 4 节 |
| 激活 | SiTU，固定左 gate、右 up，固定启用 linear_beta |
| 不支持 | 非空 bias、smooth_scale、任意输出 dtype、SwiGLU 替代、自动 padding 任意 K/N |

## 2. 推荐 Python API

```python
from vllm_ascend.ops.grouped_matmul_situ_quant import (
    grouped_matmul_situ_quant,
    is_available,
    to_weight_nz,
    to_weight_nz_list,
)

def grouped_matmul_situ_quant(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    weight_scale: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    group_list: torch.Tensor,
    *,
    beta: float,
    linear_beta: float,
    group_list_type: int = 1,
    weight_format: str = "nz",
) -> tuple[torch.Tensor, torch.Tensor]:
    ...
```

以上为签名示意，不要复制这段定义覆盖实际导入的函数。关键字参数从 `beta` 开始；返回两个新分配的 Tensor，没有 `out=` 或 workspace 参数。

### 2.1 形状符号

| 符号 | 含义 | 合法约束 |
|---|---|---|
| E | 本次调用的本地专家数 | `E > 0`，不是全模型专家数 |
| C | activation 容量行数 `x.shape[0]` | `C >= 0` |
| M | 路由后的有效总行数 | `0 <= M <= C` |
| K | GMM1 输入宽度 | `K > 0`，`K % 64 == 0` |
| N | GMM1 输出宽度，gate/up 拼接 | `N > 0`，`N % 128 == 0` |
| I | SiTU 输出宽度 `N/2` | `I % 64 == 0` |
| kb | `K/64` | 一个 kb 包含两个独立的 32 元素量化组 |

### 2.2 输入参数

所有输入 Tensor 应位于同一张 NPU，调用前选择对应当前 device；本接口不自动搬设备、排序 token 或重新量化输入。

| 参数 | 规范 dtype / shape | 布局与含义 |
|---|---|---|
| `x` | `torch.float8_e4m3fn`，`[C,K]` | ND、连续；前 M 行按 expert 顺序紧凑排列 |
| `x_scale` | `torch.float8_e8m0fnu`，`[C,kb,2]` | ND、连续、M-major；最后一维是相邻两组 scale |
| `weight`（转换前） | `torch.float4_e2m1fn_x2`，`[E,N,K/2]` | 连续 ND packed 字节；NZ 调用须先转换 |
| `weight`（NZ） | `to_weight_nz` 或 `to_weight_nz_list` 的返回值 | 复用真实 NZ storage，不自行 reshape、改 tag 或连续化 |
| `weight_scale`（stacked） | E8M0，`[E,N,kb,2]` | 连续 N-major；也接受该 storage 的 loader transpose view，见第 3 节 |
| `weight_scale`（list） | E 个 E8M0 Tensor，各 `[N,kb,2]` | 与独立权重一一对应；可传对应 loader view |
| `group_list` | `torch.int64`，`[E]` | NPU、ND、连续；含义由 `group_list_type` 决定 |
| `beta` | 模型给定的有限正 float | 必填；不要把 demo 常量视为模型默认值 |
| `linear_beta` | 模型给定的有限正 float | 必填；当前不能关闭 up tanh |
| `group_list_type` | `1` 或 `0`；默认 `1` | `1=counts`，`0=cumsum` |
| `weight_format` | `"nz"` 或 `"nd"`；默认 `"nz"` | 选择入口，不是转换请求；传 ND 到 NZ 入口会报错 |

这些是**调用者必须遵守的合法输入约束**，不是所有条件均已有完备报错。例如当前仅检查 beta/linear_beta 非零，未完整拒绝负数和 NaN；group_list 值也没有全面的设备安全校验。不要把非法输入作为普通负向 demo 下发 NPU。

### 2.3 输出参数与消费规则

| 返回值 | dtype | shape | 消费方式 |
|---|---|---|---|
| `y` | `torch.float8_e4m3fn` | `[C,I]` | ND、连续，供 GMM2 使用 |
| `y_scale` | `torch.float8_e8m0fnu` | `[C,I/64,2]` | ND、连续，作为 GMM2 per-token scale |

普通有限 scale 的语义为：`decode_e8m0(b) = 2 ** (int(b)-127)`；字节 `255` 是特殊值，不应用这个公式解码。

对有效输出行 `r`，反量化索引为：

```text
j 是 Y 的列号
scale = YScale[r, j // 64, (j % 64) // 32]
approx_Z[r,j] = float(Y[r,j]) * decode_e8m0(scale)
```

`YScale` 的末维 2 不代表同一组的 scale/zero-point，也不是两份副本；MX 不在这里使用 zero-point。

结果只定义在 `y[:M]` 和 `y_scale[:M]`。容量尾部 `[M:C)` 不保证为零，不能参与全矩阵精度比较、finite 检查或无 group_list 的下游计算。零 token expert 不在中间插入空白行。

- `C=0`：返回 `[0,I]`、`[0,I/64,2]`，不启动核心 kernel，但 ND 适配仍可能先发生。
- `C>0, M=0`：有容量但无有效输出；不检查其数值。
- 输出为推理结果；本接口未定义 autograd backward。

## 3. 最容易用错的两个数据契约

### 3.1 group_list 是分段描述，不是 token→expert 索引数组

例如 `C=16`，4 个专家实际接收 `[3,0,5,4]` 行：

| 模式 | group_list 内容 | M 的定义 |
|---|---|---|
| `group_list_type=1` | `[3,0,5,4]` | 各项求和，M=12 |
| `group_list_type=0` | `[3,3,8,12]` | 最后一项，M=12；不带额外的前导 0 |

两者都表示：expert0 使用 `x[0:3]`；expert1 不计算；expert2 使用 `x[3:8]`；expert3 使用 `x[8:12]`。`x[12:16]` 是容量尾部。

约束：counts 非负；cumsum 非负且单调不减；长度为 E；有效总行数不超过 C；所有 expert 权重与 scale 的顺序一致。若 token 尚未排好，调用者先完成 dispatch，不能只改 group_list。

### 3.2 weight_scale 必须是 N-major 物理字节

```python
# 推荐：先形成连续 N-major base。
ws_base = ...  # [E,N,kb,2], E8M0, NPU, contiguous

# 生产 loader 可能交付这个 view；底层仍然是 N-major。
ws_loader = ws_base.transpose(-3, -2)  # [E,kb,N,2], non-contiguous

# 两者都可以直接传给公开 wrapper。
weight_scale = ws_loader
```

wrapper 通过反转置恢复 canonical view，通常不复制 scale。**不能先对 `ws_loader` 调用 `.contiguous()`**：那会按 `[E,kb,N,2]` 的逻辑顺序重排字节，使 kernel 把别的通道 scale 当成本通道 scale。当前 wrapper 不一定识别这种“已连续但错序”的输入。

不要把 `.to(dtype)` 和 `.view(dtype)` 混用：

- checkpoint 中 UINT8 packed FP4 / E8M0 是编码字节，恢复语义要用适用的 `view(torch.dtype)`，不做数值转换。
- 原始浮点数转换为 FP8 是数值转换，但完整输入量化仍需配套计算 scale；单次 `.to(FP8)` 不会替你生成 MX scale。
- 已经 NZ format-cast 的权重可能有特殊 dtype/format 描述，直接保留；不能为了“统一 dtype”再对任意 NZ storage 强行 `view(dtype)`。
- `torch_npu` 的 dtype 常量在部分版本是整数 API 标识；用于 `view(dtype)` 应使用对应的原生 `torch.float4_e2m1fn_x2` / `torch.float8_e8m0fnu`。

## 4. 权重准备与四种入口

### 4.1 生产推荐：加载一次 NZ，forward 直接使用

```python
# 已有正确的 packed FP4 编码，不是随机的浮点权重。
w_nd = packed_weight_bytes.view(torch.float4_e2m1fn_x2)
# w_nd: NPU ND contiguous [E,N,K/2]
w_nz = to_weight_nz(w_nd)  # 加载期执行一次，保存返回值

y, y_scale = grouped_matmul_situ_quant(
    x=x,
    x_scale=x_scale,
    weight=w_nz,
    weight_scale=ws_loader,  # 或 canonical ws_base
    group_list=counts,
    beta=model_beta,
    linear_beta=model_linear_beta,
    group_list_type=1,
    weight_format="nz",
)
```

此片段演示接入结构，变量由 loader/上游量化提供；完整可运行输入构造见第 7 节。`to_weight_nz` 是格式转换 helper，不负责把 BF16 权重量化到 FP4。

### 4.2 独立 NZ TensorList

```python
w_nz_list = to_weight_nz_list(w_nd)    # 每专家独立 allocation，加载期执行
ws_list = [ws_loader[e] for e in range(E)]

y, y_scale = grouped_matmul_situ_quant(
    x, x_scale, w_nz_list, ws_list, counts,
    beta=model_beta,
    linear_beta=model_linear_beta,
    group_list_type=1,
    weight_format="nz",
)
```

`to_weight_nz_list` 为各 expert 保留 singleton expert 维进行转换。优先使用 helper 的返回形式，特别是 `E=1`；不要自行把每个 NZ tensor squeeze 成任意二维形式。

NZ list 核心通过动态输入地址表访问各 expert，不要求权重 allocation 连续。权重和 scale 必须一起采用 list，长度均为 E；不支持“权重 list + scale stacked”混搭。

| Python 入口组合 | forward 权重 payload 行为 | 使用建议 |
|---|---|---|
| NZ + stacked | 原 NZ storage，不 cat、不 format cast | 首选 |
| NZ + TensorList | 原 expert storages + 动态地址表 | 独立专家权重首选 |
| ND + stacked | 每次调用转换到 NZ | 兼容/接口对照；含转换开销 |
| ND + TensorList | 拼接权重/scale，再转换 | 实现保留；非空路径覆盖不足，不作为本文推荐 demo |

“零拷贝”仅指 NZ 入口的权重 payload；不表示无 output 分配、无 workspace、无 TensorList descriptor 构建，也不表示加载期转换免费。不要在 forward 循环内反复调用 `to_weight_nz*`。

## 5. 辅助接口、注册与运行前提

| 接口 | 用途 | 限制 |
|---|---|---|
| `is_available() -> bool` | 检查 NPU/A5 和扩展符号可见性 | 不是一次 device kernel 执行验证；不能证明 custom OPP 可用或版本匹配 |
| `to_weight_nz(w_nd)` | stacked packed ND → NZ | 输入已是 FP4 编码；真实格式转换会分配 |
| `to_weight_nz_list(w_nd)` | stacked packed ND → 独立 NZ list | 加载期 helper，会 clone/cast，各 expert 独立 storage |

在配好 CANN、torch-npu、vLLM 和当前 vllm-ascend 的 **Linux A5 环境**中使用。torch 必须包含本接口使用的原生 FP4/E8M0 dtype，扩展必须包含 GMSQ 的 ascend950 构建和对应 custom OPP。

```python
import torch
import torch_npu  # 注册 torch.npu

from vllm_ascend.ops.grouped_matmul_situ_quant import is_available

torch.npu.set_device(0)  # 调用者选择的可见卡；不要占用其他任务的卡
assert is_available(), "需要 A5 和包含 GMSQ 的 vllm_ascend_C 扩展"
```

wrapper 会导入 `vllm_ascend.vllm_ascend_C` 完成 native 注册。不要用 `dir(torch.ops._C_ascend)` 判断算子是否存在：namespace 是惰性的。真实扩展导入失败或 OPP 缺失应该修正环境，不应通过 monkeypatch 伪装通过。

独立调用 wrapper/本 demo **不需要设置** `VLLM_ASCEND_ENABLE_GMM_SITU_QUANT=1`；该变量控制的是 vLLM MoE 的路径选择，不是公开函数的开关。开启变量也不会编译缺失的 native 算子。

## 6. 在 vLLM 中接入与 GMM2 衔接

服务选择融合路径时，在启动新进程前设置：

```bash
export VLLM_ASCEND_ENABLE_GMM_SITU_QUANT=1
```

默认 `0`。相关模块在 import 时读取开关；不要在已经加载模型或捕获图的进程中动态切换它。当前 `08b4bcbb1` 将 native FP4 loader 调整限定在开启融合的 SiTU **W13**，W2 保持原 GMM2 loader 路径，不能把 W13 的处理照搬到 W2。

实际融合 MoE 入口还要求 MXFP、`QuantType.W4A8MXFP`、非空 `dynamic_scale`，且没有 scale bias / per-channel weight。`beta` 和 `linear_beta` 来自模型 activation 配置，demo 的 `4.0/25.0` 只是用例值。若模型关闭 linear_beta，当前融合路径不满足语义，应使用 split 或另行扩展，不能私自把 0 改成 25。

与 GMM2 的边界映射：

| GMSQ / 调用者数据 | GMM2 入参 | 规则 |
|---|---|---|
| `y: [C,I] FP8` | `hidden_states` | 不先转成 BF16，也不当作未量化输入 |
| `y_scale: [C,I/64,2] E8M0` | `per_token_scale` | 同 payload 一起传递，不是 W2 scale |
| W2 / W2Scale | `weight` / `weight_scale` | 遵守 GMM2 自己的格式和 dtype 契约 |
| 原 group_list / type | `group_list` / `group_list_type` | 保持分组一致，只消费有效区 |

本文 demo 截止于 SiTU 量化输出，不执行 GMM2，不验证 W2 TensorList 支持、服务显存或端到端性能。

图相关注意：Meta 推导与 runtime 输出需要一致；NPUGraph replay 与 torch.compile 是不同能力。已有固定地址 group_list 原位更新的测试不代表可以随意替换权重对象、TensorList 地址或 Python 标量。跨 stream 使用也须由调用者维护依赖与存储生命周期。

## 7. 教学型 UT demo

文件：[test_grouped_matmul_situ_quant_api_demo.py](../../../tests/ut/ops/test_grouped_matmul_situ_quant_api_demo.py)。

它是小型硬件接口测试，不加载 checkpoint、不启动 vLLM 服务、不需要 HCCL。设计目标是展示“合法输入 → 加载期转换 → 调用 → 检查”的完整过程。

### 7.1 用例形状与逐步讲解

主测试：`test_nz_stacked_and_list_match_for_counts_and_cumsum`。

| 项目 | Demo 数值 / shape |
|---|---|
| E / C / M | 3 / 8 / 5 |
| K / N / I / kb | 128 / 128 / 64 / 2 |
| x / x_scale | `[8,128]` / `[8,2,2]` |
| packed ND weight | `[3,128,64]`，每字节编码两份 FP4 |
| canonical weight_scale | `[3,128,2,2]`，N-major |
| loader view | `[3,2,128,2]`，原 N-major storage 的非连续转置 view |
| counts / cumsum | `[3,0,2]` / `[3,3,5]` |
| y / y_scale | `[8,64]` / `[8,1,2]`，仅前 5 行有定义 |

按代码阅读顺序：

1. **构造输入编码。** CPU 上用确定性的小值生成 activation payload；用有效 nibble 组合 packed FP4，两半都填；MX scale 用受控的指数编码。再将这些小 Tensor 放到 NPU。这里是人工合法测试输入，不是 checkpoint 量化算法。
2. **构造带位置差异的 scale。** 先生成连续 `[E,N,kb,2]`，再 `transpose(-3,-2)` 模拟 loader。可反转置检查 `data_ptr` 相同；不把 loader view 连续化。
3. **做加载期转换。** 分别调用 `to_weight_nz`、`to_weight_nz_list`。后者产生独立专家权重，演示 list 不要求共享一块大存储。测试同时保留两套权重只是为了对照，生产不必保留重复副本。
4. **调用相同公开 API。** 改变 stacked/list 组织和 counts/cumsum 表示，但不改变 token 分组、输入字节和 SiTU 参数。`beta=4.0, linear_beta=25.0` 仅为示例值。
5. **检查结构和有效区。** 先断言输出 shape/dtype，再同步并将前 5 行的小结果转成 UINT8 字节拷到 CPU 比较。FP8 finite 检查可以在这份小 CPU 副本上转 float；scale 检查特殊编码。不要读取尾部 3 行的数值来判断结果。

这里 expert0 对应输入行 `[0:3)`，expert1 没有输入，expert2 对应 `[3:5)`。每次调用都返回 8 行容量，而不是自动缩成 5 行。最后一维 2 的两个 scale 分别作用于 Y 的 `[0:32)` 和 `[32:64)`。

### 7.2 测试目的与证据边界

- 构造有限 FP8 payload 和有限 E8M0 scale，不使用随机 0～255 的 scale 字节。
- packed FP4 每字节的高低两个 nibble 都构造有效值，不让一半 K 元素意外恒为零。
- weight scale 随 expert/channel/K-group 变化，并构造真实 loader transpose view。
- 对照 NZ stacked 与独立 NZ TensorList；对照 counts 与 cumsum。
- 检查输出 shape、dtype、有效区有限性及字节一致性；不检查未定义容量尾部。

**这些对照共享同一 GMSQ 核心，不是独立的 GMM1 + SituMxQuant 精度真值。** 即使通过，也不证明所有 scale 排列都正确、任意模型 bit-exact、forward 完全零分配或性能达标。split golden、极值、Meta/compile、生产 shape 和性能验收仍由各自测试负责。

### 7.3 运行方式

在服务器已构建并安装当前代码的 vllm-ascend 仓库根目录，先由操作者选定空闲可见卡，再运行：

```bash
python -m pytest --noconftest tests/ut/ops/test_grouped_matmul_situ_quant_api_demo.py -v -s -rA
```

`--noconftest` 用于这个自包含教学 demo：仓内共享 UT conftest 会引入大量模型 patch/mock，并可能注入 A2 build 信息，不适合作为本例的硬件环境准备。此选项不是建议其他 UT 全部绕过 conftest。测试使用进程的逻辑 `npu:0`，物理卡可见性由运行者提前设置。

无 NPU / 非 A5 的跳过不算验证通过；A5 环境的 native 扩展或自定义算子缺失需要明确修复。查看 pytest 的 PASSED/SKIPPED/FAILED，不要只依据进程没有 traceback。

### 7.4 本次交付验证范围

本次已检查 demo 的 Python 语法、120 列限制和尾随空白，以及本文 Python 示例语法、UTF-8 编码和文件链接。当前 Windows 本地没有 torch/torch-npu/pytest/NPU，未执行硬件测试，Ruff 也未运行。

这些静态检查不等于 A5 执行验收。本次不改生产代码、不重新构建、不启动远端服务。服务器运行后应记录 commit、实际扩展路径、CANN 内部版本和 pytest 结果，再将硬件结论补入评审材料。

## 8. 常见错误与处理方式

| 现象 | 优先检查 | 不建议的处理 |
|---|---|---|
| 算子未注册 / `is_available=False` | A5 检测、native 扩展构建、实际导入路径 | 仅打开服务开关或 monkeypatch 检测 |
| NZ format 错误 | 是否传了真实 `to_weight_nz*` 结果，入口是否匹配 | 给 ND 字节改 format tag 冒充 NZ |
| `EZ1009` dtype/非连续错误 | packed 数据语义 dtype、scale base/view、加载的 OPP | 给所有 Tensor 无差别 `.contiguous()` |
| shape 正确但数值异常 | 首先 dump 有效区的原始 payload/scale、stride 和 N-major 字节 | 仅比较 ND/NZ 同核心就宣布精度正确 |
| `linear_beta=0` tiling 失败 | 模型是否关闭 up tanh；当前融合能力是否匹配 | 偷换成 25 或极大数改变模型语义 |
| counts/cumsum 输出不一致 | type、cumsum 是否多放前导 0、token 是否按 expert 排列 | 修改 group_list 但不改变 token 对应关系 |
| 尾部有随机值或 NaN | 是否错误检查了 `[M:C)` 未定义区域 | 强求整个 capacity Tensor 全零/有限 |
| 下游 GMM2 报错/OOM | Y/scale descriptor、有效区、W2 契约及实际显存时间线 | 仅按报错位置推断 GMSQ 精度或 workspace 根因 |

常规调试可打印 shape、dtype、stride、format；需要数值时只 dump 小样本或分块有效区。NPU→CPU、`.item()` 和同步是诊断开销，不放进正式热路径；避免对完整大输出 `.float()` 自造 FP32 峰值显存。

## 9. 低层接口附录

推荐调用 Python wrapper，以下仅用于绑定层评审或排查注册问题。

```text
torch.ops._C_ascend.grouped_matmul_situ_quant                 # ND stacked
torch.ops._C_ascend.grouped_matmul_situ_quant.list            # ND list
torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz       # NZ stacked
torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz.list  # NZ list
```

共同位置参数顺序（与公开 wrapper 顺序不同）：

```text
x, weight, weight_scale, weight_assist_matrix,
bias, x_scale, smooth_scale, group_list,
dequant_mode, dequant_dtype, quant_mode, group_list_type,
tuning_config, beta, linear_beta
```

| 参数 | 当前设定 |
|---|---|
| `dequant_mode / dequant_dtype / quant_mode` | 固定 `1 / 0 / 1`，不支持其他组合 |
| `weight_assist_matrix` | 推荐 `None`；非空接收但忽略并 warning |
| `bias / smooth_scale` | 传 `None`；非空拒绝 |
| `tuning_config` | 推荐 `None`；当前接收但忽略 |
| weight / weight_scale | stacked 重载为 Tensor，`.list` 重载为 Tensor[] |

底层自定义 ACLNN `GroupedMatmulSituQuant` 是另一层接口：输入 `x,x_scale,weight(dynamic),weight_scale(dynamic),group_list`，属性 `group_list_type,beta,linear_beta`，输出 `y,y_scale`。OpDef 的 `group_list_type` 默认 **0**，Python wrapper 默认 **1**；直接使用低层接口应显式传值，不能混用默认值。ND 的 format cast 是 Torch adapter 行为，不是 OpDef 支持 ND 权重 kernel。

## 10. 当前代码索引

以下相对链接在本工作区可直接定位；行号易随版本变化，评审以函数名为准。

| 文件 | 重点符号 |
|---|---|
| [Python wrapper](../../../vllm_ascend/ops/grouped_matmul_situ_quant.py) | `grouped_matmul_situ_quant`、`to_weight_nz*`、`_align_weight_scale` |
| [Torch 注册](../../../csrc/torch_binding.cpp) | 四个 `grouped_matmul_situ_quant*` schema |
| [Torch adapter](../../../csrc/moe/grouped_matmul_situ_quant/grouped_matmul_situ_quant_torch_adpt.h) | `ValidateV2Params`、`RunV2Core`、ND/NZ 四入口 |
| [OpDef](../../../csrc/moe/grouped_matmul_situ_quant/op_host/grouped_matmul_situ_quant_def.cpp) | `GroupedMatmulSituQuant` 输入、属性、SoC 注册 |
| [Host tiling](../../../csrc/moe/grouped_matmul_situ_quant/op_host/grouped_matmul_situ_quant_tiling.cpp) | K/N、属性合法范围 |
| [Meta](../../../csrc/torch_binding_meta.cpp) | `grouped_matmul_situ_quant_meta` / list meta |
| [MoE 接入](../../../vllm_ascend/ops/fused_moe/moe_mlp.py) | `_w4a8_situ_apply_mlp`，SiTU 输出与 GMM2 传参 |
| [W4A8 loader](../../../vllm_ascend/quantization/methods/w4a8_mxfp4.py) | `process_weights_after_loading`，W13/W2 分别处理 |
| [已有 UT](../../../tests/ut/ops/test_grouped_matmul_situ_quant.py) | Meta、empty rank、graph replay 等 |
| [本次 API demo](../../../tests/ut/ops/test_grouped_matmul_situ_quant_api_demo.py) | 小规模公开接口调用示例 |
