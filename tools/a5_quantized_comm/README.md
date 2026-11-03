# A5 原生 QuantAllReduce：PyTorch 封装与精度验证

此工具为已安装的 `aclnnQuantAllReduce` 增加独立的 PyTorch 包装，不需要重编 vLLM。
目标环境为 A5 / Ascend 950、CANN 9.2.0、PyTorch 2.10.0、torch_npu 2.10.0.post7。
只有在目标环境编译并执行多卡测试后，才能确认该环境的原生算子可用且精度符合预期。

当前封装只验证 Eager 下的 MXFP8 E4M3FN AllReduce，输入为 FP8 数据和 E8M0 scale，
原生算子完成通信、反量化和求和，输出 BF16 或 FP32。
封装会同步流，以保证输入、workspace 和 ACL tensor 描述符在执行完成前存活。
因此它是精度诊断工具，不能用其耗时判断生产性能，也不用于入图。

## 1. 单进程编译

先在已有 CANN / PyTorch 环境中执行。需要 CANN 开发头文件、C++ 编译器和 ninja。

```bash
cd /path/to/vllm-ascend
python3 tools/a5_quantized_comm/build.py \
    --cann-home /usr/local/Ascend/cann-9.2.0 \
    --build-dir /tmp/a5_quant_comm_probe
```

构建脚本直接包含本机 `aclnn_quant_all_reduce.h`，并链接本机算子库，避免手写 ABI。
成功后生成 `/tmp/a5_quant_comm_probe/manifest.json`。先编译一次，再启动多 rank。
若找不到头文件，定位后将**包含此头文件的目录**传给 `--op-include`：

```bash
find /usr/local/Ascend/cann-9.2.0 -name aclnn_quant_all_reduce.h
python3 tools/a5_quantized_comm/build.py \
    --cann-home /usr/local/Ascend/cann-9.2.0 \
    --op-include /actual/path/to/include/aclnnop
```

有 `.so` 符号不代表已安装开发头文件；若未找到，需要补齐同版本 CANN SDK。
如编译器内存不足，可以在构建命令前设置 PyTorch 标准参数 `MAX_JOBS=1`。

## 2. 双卡精度测试

```bash
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    tools/a5_quantized_comm/validate.py \
    --manifest /tmp/a5_quant_comm_probe/manifest.json \
    --rows 128 --hidden-sizes 5120
```

默认对两种输出类型各测试七组数据，并更换输入重复两轮：手工编码的 FP8/scale、
全零、不同 rank 的常数、随机值、各 rank 不同幅度、正负抵消、含离群值和不同 block 幅度。
所有 rank 的结果都会参与检查。手工编码用例让 scale 在 rank、token、相邻 block 间变化，
可独立于动态量化算子检查 scale 排列和每个 rank 的反量化语义。

扩大形状覆盖（通信域大小可选 2、4、8）：

```bash
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    tools/a5_quantized_comm/validate.py \
    --rows 2 128 1024 --hidden-sizes 1024 5120 8192
```

首轮保守要求 `T` 能被 rank 数整除；`H` 范围为 `[1024,8192]` 且 128 对齐。
大形状需要按 CANN 要求增大 `HCCL_BUFFSIZE`，脚本会检查所需最小空间。
运行需满足本机 CANN 对 A5 驱动、固件和 UB-Memory 通信引擎的要求。

## 3. 如何判断精度

每个用例输出一行 JSON，区分两条基准：

1. **通信基准**：CPU 按字节解析 E4M3FN 和 E8M0，逐 rank 反量化，再以 FP32 求和。
   `comm_pass` 检查原生输出是否接近这一结果，不依赖 `npu_anti_mx_quant`。
   全零与常数用例要求精确一致；其他用例默认允许 BF16 的有限舍入误差。
2. **量化误差**：`quantization_vs_original` 比较上述结果与原始 BF16 输入的 FP32 求和。
   `end_to_end_rank0` 还包含原生输出舍入。记录 `max_abs`、RMSE、NRMSE。

普通用例的通信判据是 `abs(actual - rounded_reference) <= atol + rtol * sum(abs(rank_dequant))`，
以免正负抵消放大相对误差。默认 `atol=1e-6`、BF16 `rtol=2^-7`、FP32 `rtol=2e-5`；
这些是诊断容差，不代表厂商的精度保证。可用 `--comm-atol`、`--bf16-comm-rtol`、
`--fp32-comm-rtol` 调整。`comm_bad_elements_all_ranks` 非零会导致进程失败。
`rank_difference_elements` 仅作观察，因为原生归约默认非确定性。

量化误差默认报告，是否可接受要结合业务数据判断。需要强制检查时，例如：

```bash
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
    tools/a5_quantized_comm/validate.py --max-quant-nrmse 0.05
```

`0.05` 只是示例质量门槛。若原始输入之和为零而量化误差非零，NRMSE 为无穷大。
先确认所有 `comm_pass=true`，再评估量化误差；这一步不能替代模型输出精度验证。

## 4. Python 调用入口

```python
import json

import torch
import torch_npu

manifest = json.load(open("/tmp/a5_quant_comm_probe/manifest.json"))
torch.ops.load_library(manifest["library"])

# x: contiguous BF16 [T,H] on NPU
q, scales = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
# view 保留 E8M0 的原始位模式，不能用数值类型转换 to(torch.uint8)。
scale_bytes = scales.view(torch.uint8).reshape(x.shape[0], x.shape[1] // 64, 2).contiguous()
y = torch.ops.a5_quant_comm_probe.quant_all_reduce(q.contiguous(), scale_bytes, dedicated_hcom, False)
```

`dedicated_hcom` 的初始化见 `validate.py`：默认 Gloo 组负责收集精度数据和控制，
单独建立 HCCL 组并获取其通信域名称。**不能**在这个 HCCL 组上插入普通
`all_reduce`、`all_gather` 或 `barrier`。所有 rank 必须以相同顺序调用原生算子。
不要在 graph capture 中调用此同步工具，也不要在原生调用失败后继续复用通信域。

## 5. 不依赖 NPU 的基准解码检查

```bash
python3 -m unittest discover -s tools/a5_quantized_comm -p test_reference.py -v
```

这只检查 CPU 字节解析的已知编码和边界值，不能证明 C++ 编译或 NPU 执行正确。

## 6. API 返回成功但精度完全错误

`GetWorkspaceSize` 成功后还必须生成非空 executor；封装现在会检查这一点。
若出现 `ACL_SUCCESS but left executor=nullptr`，先检查实际加载的算子库，
不能将“导出符号存在且返回码为 0”视为已有可执行的算子实现。
错误中的 `symbol_library` 会给出 GetWorkspaceSize 符号实际绑定到的库路径，
它可能因动态库搜索顺序与构建日志中的候选路径不同。
`workspace_size=0` 单独出现不代表错误，关键是是否生成 executor。

例如对构建日志中选中的库执行以下命令。若改用其他路径，也需检查那个库。

```bash
LIB=/usr/local/Ascend/cann-9.2.0/lib64/libopapi_transformer.so
objdump -d --disassemble=aclnnQuantAllReduceGetWorkspaceSize "$LIB"
objdump -d --disassemble=aclnnQuantAllReduce "$LIB"
```

如果实际函数只有 `mov w0, #0; ret`（或等价的直接返回 0 指令），
则该函数是返回成功的空实现，并没有创建执行计划或发起通信。
仅仅符号间隔小不能证明是空实现；若是跳转指令，还需继续查看跳转目标。
此时应定位、安装包含真实实现的对应算子包，再重新构建并验证。

输出现在预填 NaN；若 API 没有写入全部输出，会通过非有限值检查暴露。
精度失败时还会打印各 rank 的前八个实际值和基准值，以及非零/非有限元素数，
以区分全零、未写入输出与数值计算偏差。非有限结果对应的误差指标记为 `null`。
不要通过放宽容差掩盖此类失败。

接口及约束依据 [CANN QuantAllReduce 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnQuantAllReduce.md)。
若该文档与本机 9.2.0 SDK 有差异，以本机 SDK 头文件和对应版本的接口约束为准。
