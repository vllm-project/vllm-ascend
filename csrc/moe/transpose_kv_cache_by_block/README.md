# TransposeKvCacheByBlock

## 产品支持情况

| 产品 | 算子注册的 SoC |
| --- | --- |
| Atlas A2 系列产品 | `ascend910b` |
| Atlas A3 系列产品 | `ascend910_93` |

支持 FP16、BF16，数据格式为 ND。

## 功能说明

`TransposeKvCacheByBlock` 在原有存储上重排指定 block 的 K/V Cache。
它用于 PD 分离中 P/D 的 TP 配置不同、D 端需要拼接多个 P 端 KV 分片的场景，
将按分片存放的数据恢复为按 token 存放的数据，供后续 Attention 使用。
同一次调用可处理多层 K/V Cache；算子只搬运和重排数据，不进行浮点计算。

设 `B = blockSize`、`H = headNum`、`D = headDim`、`S = splitNum`。
每个 cache 的 Tensor shape 均为 `[num_blocks, B, H, D]`，
但待重排 block 中的数据按 `[S, B, H / S, D]` 的顺序存放。
算子交换其中的分片轴和 token 轴，再合并头维度，得到 `[B, H, D]`。

对每层 K/V 和每个选中的 `block_id`，等价操作为：

```python
before = cache[block_id].clone()
after = (
    before.reshape(split_num, block_size, head_num // split_num, head_dim)
    .transpose(0, 1)
    .contiguous()
    .reshape(block_size, head_num, head_dim)
)
cache[block_id].copy_(after)
```

该变换会写回原 Tensor，但不改变其 shape、stride、storage offset 或数据指针。
未选中的 block 和 block 之间的间隙保持不变。`splitNum = 1` 时数据顺序不变；
一般情况下，不应对已经重排过的 block 再次执行此操作。

## 接口与参数

PyTorch 接口在 [torch_binding.cpp](../../torch_binding.cpp) 中注册：

```python
torch.ops._C_ascend.transpose_kv_cache_by_block(
    kCache, vCache, blockIDs, blockSize, headNum, headDim, splitNum, layerNum
)
```

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `kCache` | `Tensor[]` | 各层 K Cache，列表长度为 `layerNum`；每个 Tensor 为 `[num_blocks, blockSize, headNum, headDim]`，FP16 或 BF16。 |
| `vCache` | `Tensor[]` | 各层 V Cache，与对应 K Cache 的 shape 相同；所有 K/V Tensor 的 dtype 必须相同。 |
| `blockIDs` | `Tensor` | 连续的一维 INT64 Tensor，shape 为 `[num_selected_blocks]`；所有层使用同一组 block ID，与所有 cache 位于同一 NPU。 |
| `blockSize` | `int` | 每个 block 的 token 数，必须大于 0。 |
| `headNum` | `int` | 当前 cache 的 KV 头数，必须大于 0，并能被 `splitNum` 整除。 |
| `headDim` | `int` | 每个头的维度，必须大于 0；当前 FP16/BF16 实现要求为 16 的倍数。 |
| `splitNum` | `int` | 每个 block 中按头切分的输入分片数，必须大于 0；connector 调用时对应 `tp_num_need_pulls`。 |
| `layerNum` | `int` | 本次处理的层数，必须大于等于 0，并与两个 cache 列表的长度一致。 |

接口返回 `None`，结果原地写入 `kCache` 和 `vCache`。

### 底层 stride 属性

C++ binding 从各 Tensor 推导以下 ACLNN 属性，PyTorch 调用方不需要传入：

| ACLNN 属性 | 单位 | 说明 |
| --- | --- | --- |
| `kBlockStride` | 元素 | 相邻 K block 起点之间的物理间隔；默认值 `0` 表示 `B * H * D`。 |
| `vBlockStride` | 元素 | 相邻 V block 起点之间的物理间隔；默认值 `0` 表示 `B * H * D`。 |

K/V stride 独立传递，kernel 使用 64 位 block 偏移定位各 Tensor 中的数据。
连续相邻、K/V stride 对相同的层合并为一次 ACLNN 调用；stride 对不同的层分组调用。
单 block Tensor 的第 0 维 stride 不影响寻址，binding 将其归一化为 `B * H * D` 后分组。
直接使用 ACLNN 时，应显式提供非连续 cache 的实际 stride；默认值不会自动推导非连续布局。

## 约束说明

### 支持的存储布局

每个 block 内部的 `[B, H, D]` 存储必须连续，只有第 0 维允许存在间隙。
当对应维度大小大于 1 时，要求 `stride(3) = 1`、`stride(2) = D`、
`stride(1) = H * D`。维度大小为 1 时，该维度的 stride 不作此限制。

当 `num_blocks > 1` 时，要求 `stride(0) >= B * H * D`，确保同一 Tensor 的 block 不重叠。
K/V 可以共享一块底层存储，但各自访问的有效数据区域必须互不重叠。

| 布局 | 示例 | 是否支持 |
| --- | --- | --- |
| 连续 cache | 独立的 `[num_blocks, B, H, D]` Tensor | 支持 |
| K/V 按 block 交错 | `backing` 为 `[num_blocks, 2, B, H, D]`，K/V 分别取 `backing[:, 0]`、`backing[:, 1]` | 支持，block stride 为 `2 * B * H * D` |
| block 间有额外间隙 | 每个 block 内连续，第 0 维 stride 大于 block 元素数 | 支持，K/V stride 可不同 |
| 非零 storage offset | 从更大的 backing 中切片得到的合法 cache view | 支持 |
| block 内不连续 | `backing[..., ::2]` | 不支持，binding 报错 |
| 同一 Tensor 的 block 重叠 | `num_blocks > 1` 且 `stride(0) < B * H * D` | 不支持，binding 报错 |

`IgnoreContiguous()` 允许上述 block-strided Tensor 进入 ACLNN；它不表示支持任意非连续布局。
调用时保留原 cache view，不需要先将整个 cache 转为连续 Tensor。

### 调用方责任与边界

- `blockIDs` 中的 ID 必须互不重复，且对每层 cache 都满足 `0 <= block_id < num_blocks`。
  当前 binding 只检查其 shape、dtype 和连续性，不读取设备端 ID 来检查范围或重复值。
- 不同层以及 K/V 之间被处理的数据区域必须互不重叠；binding 不检查跨 Tensor 的别名重叠。
- 调用前应保证 KV 传输完成，调用后应通过正确的 stream 依赖保证 Attention 读取到重排结果。
  调用方不得同时读写同一批待重排 block。
- K/V 每层的 block 数必须相同；各层均使用同一组 `blockSize`、`headNum`、`headDim` 和 `splitNum`。
- 空 `blockIDs` 在完成 cache 元数据校验后直接返回，不发射 kernel；
  `num_blocks = 0` 的 cache 只允许配合空 `blockIDs`。
- `layerNum = 0` 时，两个 cache 列表必须为空；基础参数校验后直接返回。
- 当前 tiling 只沿 `blockSize` 切分，不切分 `headNum` 或 `headDim`。
  若无法在设备 UB 容量及可用核数下找到合法切分，tiling 会报错。

## 调用示例

以下示例覆盖共享 backing 的非连续 K/V view，并按整个 backing 校验结果。
需要在已编译安装本算子的 A2/A3 NPU 环境中运行。

```python
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

assert enable_custom_op(), "Custom operators must be available"

num_blocks, block_size, head_num, head_dim, split_num = 5, 16, 4, 128, 2
backing = torch.randn(
    num_blocks, 2, block_size, head_num, head_dim,
    dtype=torch.float16, device="npu",
)
k_cache, v_cache = backing[:, 0], backing[:, 1]
selected_ids = [4, 1]
block_ids = torch.tensor(selected_ids, dtype=torch.int64, device="npu")
expected = backing.cpu()

for cache in (expected[:, 0], expected[:, 1]):
    for block_id in selected_ids:
        before = cache[block_id].clone()
        cache[block_id].copy_(
            before.reshape(split_num, block_size, -1)
            .transpose(0, 1)
            .reshape_as(before)
        )

torch.ops._C_ascend.transpose_kv_cache_by_block(
    [k_cache], [v_cache], block_ids,
    block_size, head_num, head_dim, split_num, 1,
)
torch.npu.synchronize()
torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
```

## 框架接入与验证

[Mooncake connector](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py)
和 [Mooncake hybrid connector](../../../vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py)
中的 `reformat_kv_cache_with_fused_op` 调用此算子。
`enable_transpose_kv_cache_by_block` 默认为 `true`，可在已有的 `--additional-config`
中设置为 `false` 来关闭 connector 的该融合路径，详见
[配置文档](../../../docs/source/user_guide/configuration/additional_config.md)。
此开关不影响直接调用 `torch.ops._C_ascend.transpose_kv_cache_by_block`。

开启开关不意味着所有 PD 重排都走此算子：connector 还会判断是否需要多分片拼接、
自定义算子是否可用等条件。`mooncake_connector.py` 的 hybrid linear 分支使用独立的
Torch 重排路径；NZ 格式转换也不由此算子完成。

编译时需保证 C++ binding 与自定义算子库来自同一版本；新增 stride 属性改变了生成的
ACLNN 接口和 tiling 数据结构，更新后应同时重新编译二者。

在仓库根目录执行回归测试：

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_transpose_kv_cache_by_block.py
```

测试覆盖 FP16/BF16、连续/交错/带间隙布局、K/V 独立 stride、混合层布局、
非零 storage offset、单 block 的任意 stride、空选择和非法布局。
存储保持用例对整个 backing 做精确比较，同时检查 view 的数据指针、stride 和 storage offset。
实现背景见 [PR #6366](https://github.com/vllm-project/vllm-ascend/pull/6366)。
