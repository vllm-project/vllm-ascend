# _num_nans_kernel

## 功能说明

- 算子功能：`_num_nans_kernel` 逐请求（逐行）统计 logits 中的 NaN 数量，
  输出 INT32 计数张量，供采样链路上报 `num_nans_in_logits` 指标（如
  `recompute_scheduler` 依据该指标判断 logits 质量并触发重算）。生产环境经
  `vllm_ascend/patch/worker/patch_v2/patch_triton.py` 将上游
  `vllm.v1.worker.gpu.metrics.logits.get_num_nans` 整体替换为本实现（包装
  函数为 `get_num_nans`）。

- 计算公式（逐行独立处理，`r` 为请求行索引）：

    $$
    num\_nans[r] = \sum_{i=0}^{vocab\_size-1} \mathbb{1}\left[isnan(logits[r, i])\right]
    $$

- 算法流程（逐行并行，`tl.program_id(0)` 即请求行索引）：

    1. 按 `BLOCK_SIZE` 分块遍历该行 vocab 维度（`vocab_size` 不是
       `BLOCK_SIZE` 整数倍时多一轮迭代）。
    2. 掩码加载 `block < vocab_size`，越界位置以 `other=0` 填充（0 非
       NaN，不会污染计数）。
    3. 加载数据转为 FLOAT32，调用 CANN libdevice 的 `isnan` 判 NaN，结果转
       INT32。
    4. `tl.sum` 块内求和并累加到标量 `num_nans`。
    5. 循环结束后将该行总计数写入 `num_nans_ptr[r]`。

## 与上游实现的差异

- 上游 vLLM 版本（`vllm/v1/worker/gpu/metrics/logits.py`）的 libdevice 从
  `torch._inductor.runtime.triton_helpers` 导入，会解析到 CUDA 符号，在
  Ascend 上内核编译失败；且其启动路径依赖 `triton.experimental.gluon.nvidia`，
  triton-ascend 不提供该模块。
- vLLM Ascend 版本直接导入 `triton.language.extra.cann.libdevice`，内核逻辑
  与上游一致，仅 libdevice 来源不同。
- 通过 patch 整体替换 `sampler.get_num_nans` 与
  `rejection_sampler.get_num_nans`；待编译器与 Triton Ascend toolkit 支持上游
  实现后可回退（见 `patch_triton.py` 内注释）。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:--------|:----------------|:------|:---------|:---------|
| logits_ptr | 输入 | 二维 `[num_reqs, vocab_size]` logits 张量。 | FLOAT32 | ND |
| logits_stride | 属性 | `logits` 的 stride(0)，即行跨度。 | INT | - |
| num_nans_ptr | 输出 | 一维 `[num_reqs]`，每行的 NaN 计数。 | INT32 | ND |
| vocab_size | 属性 | vocab 维度长度。 | INT | - |
| BLOCK_SIZE | 属性 | 常量，分块遍历宽度，生产固定为 8192。 | constexpr INT | - |

## 约束说明

- `logits` 必须为二维 FLOAT32；按 `logits_stride + 偏移` 访问行内元素，
  支持行跨度大于 `vocab_size` 的排布。
- `num_nans` 必须为一维 INT32，长度等于 `num_reqs`。
- `BLOCK_SIZE` 生产固定 8192；`vocab_size` 小于 `BLOCK_SIZE` 时靠掩码保护，
  大于时多轮迭代，两者均正确。
- 每个 program 只处理一个请求行，grid 尺寸应为 `(num_reqs,)`。
- 输出为精确整数计数，无浮点累加误差，对比应逐位相等
  （`rtol=0, atol=0`）。

## 调用示例

```python
import torch

from vllm_ascend.ops.triton.v2.metrics.num_nans import (
    _num_nans_kernel,
    get_num_nans,
)

device = "npu"
num_reqs, vocab_size = 4, 4096

logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32, device=device)
logits[0, :10] = float("nan")  # 请求 0 注入 10 个 NaN

# 方式一：生产包装函数（patch 后 sampler 实际调用的入口）
num_nans = get_num_nans(logits)

# 方式二：直接启动内核（与精度 UT 一致）
num_nans = torch.empty(num_reqs, dtype=torch.int32, device=device)
_num_nans_kernel[(num_reqs,)](
    logits,
    logits.stride(0),
    num_nans,
    vocab_size,
    BLOCK_SIZE=8192,
)
torch.npu.synchronize()
print(num_nans.cpu())  # 期望 [10, 0, 0, 0]
```

## test ut

精度 UT 位于
`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_num_nans.py`，
直接测试本 Ascend 内核，与 CPU 参考实现
（`torch.isnan(logits).sum(dim=-1).to(torch.int32)`）逐位比对
（`rtol=0, atol=0`）：

- 参数组合：`dtype ∈ {fp32, bf16, fp16}` × NaN 布局（行头连续 / 逐行
  随机散布） × 形状（`(num_reqs, vocab_size)` 覆盖 128/5000/8192/10000/
  16384，含尾部掩码、非 2 幂、单块/多块 + 部分尾部掩码组合） ×
  `frac_nan ∈ {0.0, 0.1, 0.5, 1.0}`，共 120 组；`frac_nan=0.0/1.0` 同时
  覆盖全无 NaN 与全 NaN 边界。数据在 CPU 上以 fp32 构造并注入 NaN 后
  再转目标 dtype，参考值按转换后的张量计算，验证低精度输入经内核
  `.to(tl.float32)` 上转后 NaN 不丢失、不误产生。
- 特殊值专项：`test_num_nans_special_values_not_counted`，将 ±Inf/±0.0/
  max-finite/min-normal/subnormal 平铺满行，要求计数为 0；另一行在其上
  叠加散布 NaN，要求恰好计数，防止 CANN `isnan` 退化为无穷/指数类判断。

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_num_nans.py
```
