# _num_nans_kernel

## 功能说明

- 算子功能：`_num_nans_kernel` 是 vLLM 上游的 Triton 内核（源码位于
  `vllm/v1/worker/gpu/metrics/logits.py`），按行统计 logits 矩阵中 NaN 元素的
  个数，供 `get_num_nans` 做指标统计使用。本仓库对该内核不做逻辑改写，仅通过
  单测验证其在 Ascend NPU 上的精度，并处理 Ascend 下的 libdevice 编译问题。

- 计算公式（逐行独立处理）：

    $$
    num\_nans_i = \sum_{j=0}^{vocab\_size-1} \mathbb{1}\left[isnan(logits_{i,j})\right]
    $$

- 算法流程：

    1. 以 `BLOCK_SIZE` 为块大小，沿 vocab 维度分块遍历。
    2. 对每个 block，用 `tl.load(..., mask=block < vocab_size, other=0)` 读取一行
       中的一段 logits，越界位置用 0 填充。
    3. 将 block 转为 FLOAT32，调用 `libdevice.isnan` 判断 NaN，得到 int1 掩码。
    4. 用 `tl.sum` 累加当前 block 的 NaN 个数到 `num_nans`（int32）。
    5. 遍历完成后，将 `num_nans` 写入 `num_nans_ptr + req_idx`。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:--------|:----------------|:------|:---------|:---------|
| logits_ptr | 输入 | 二维 `[num_reqs, vocab_size]` logits 数据的首地址。 | FLOAT32 | ND |
| logits_stride | 属性 | logits 第 0 维的 stride（行间步长）。 | INT | - |
| num_nans_ptr | 输出 | 一维 `[num_reqs]`，每行 NaN 元素个数。 | INT32 | ND |
| vocab_size | 属性 | 词表大小，即每行的元素个数。 | INT | - |
| BLOCK_SIZE | 属性（constexpr） | vocab 维度分块遍历的块大小，调用时固定为 8192。 | constexpr | - |

## 约束说明

- `logits` 必须为二维且数据类型为 FLOAT32。
- `vocab_size` 必须等于 logits 的实际列数，否则统计结果失真。
- `num_nans` 输出张量必须为一维、长度等于 `num_reqs`、数据类型 INT32。
- 内核沿 vocab 维度按 `BLOCK_SIZE` 分块，`vocab_size` 无固定上限（非 2 的幂也可），
  `mask=block < vocab_size` 负责边界截断。

### Ascend NPU 上的 libdevice 编译问题

- 上游内核从 `torch._inductor.runtime.triton_helpers` 导入 `libdevice`，在 Ascend 下
  `libdevice.isnan` 会解析为不支持的 CUDA 符号并在编译期返回 `None`（表现为
  `AttributeError: 'NoneType' object has no attribute 'to'`）。
- vLLM Ascend 已在 `vllm_ascend/patch/worker/patch_v2/patch_triton.py` 中统一将
  `vllm.v1.worker.gpu.metrics.logits.libdevice` 重绑定为
  `triton.language.extra.cann.libdevice`，使 sampler / rejection sampler 中的
  `get_num_nans` 在编译内核时使用 CANN libdevice。
- 单测文件为了可独立运行（不依赖 patch 加载顺序），在 import 后同样做一次模块级
  libdevice 重绑定，再启动内核。

### 单测中的 `vllm.triton_utils` shim

- 测试环境安装的 triton-ascend 3.2.x 早于 `triton.experimental.gluon`（及
  `triton.experimental.gluon.nvidia`）的引入时间，而 `vllm.triton_utils` 会无条件
  导入 gluon，直接导入真实包会抛 `ModuleNotFoundError`。
- 单测在任何 `vllm.*` 导入之前，向 `sys.modules` 安装一个 package 形状的
  `vllm.triton_utils` shim：`__path__` 指向真实的 `triton_utils` 目录（子模块
  `allocation` / `libdevice` / `importing` 仍从磁盘加载），`tl` / `triton` 直接取自
  已安装的 triton，gluon 相关属性用占位符代替。真实包的 `__init__.py` 因此不会
  执行，内核测试得以独立运行。

## 调用示例

```python
import torch

from vllm.v1.worker.gpu.metrics.logits import get_num_nans

device = "npu"
logits = torch.randn(4, 4096, dtype=torch.float32, device=device)
logits[0, :2] = float("nan")  # 第 0 行注入 2 个 NaN

num_nans = get_num_nans(logits)  # shape [4], dtype int32, 首元素为 2
```

## test ut

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_num_nans_kernel.py #--noconftest
```