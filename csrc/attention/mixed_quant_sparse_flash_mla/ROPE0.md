# A5 C8 RoPE0 接口契约

本功能面向 A5（Ascend950）的 `MixedQuantSparseFlashMla` 及其 Metadata 算子。`rope_head_dim=0` 表示输入中没有 RoPE 分量，保留 448 维 NOPE 内容分量。C8 对应本算子的 FP8 KV 量化路径。

## 输入与输出

| 项目 | `rope_head_dim=0` 的约束 |
| --- | --- |
| Q / attention output | 最后一维为 448；Q 支持 BSND、TND，输出形状与 Q 相同 |
| Metadata | `head_dim=448`、`rope_head_dim=0` |
| `quant_mode=1` KV | 最后一维 480 字节：NOPE FP8 448B + BF16 scales 14B + padding 18B |
| `quant_mode=2` KV | 仅支持 PA_BBND，最后一维 456 字节：每个 token 的 feature 448B、scales 7B、padding 1B |

Mode 1 支持 PA_BBND，以及与 Q 布局相同的 BSND/TND KV。Mode 2 的物理 page 先存该 page 全部 token 的 feature，再存各 token 的 scales 与 padding；不能按逐 token 的 456B 交错记录填充。

`ori_kv` 与提供的 `cmp_kv` 使用相同的量化模式和上述宽度。其余序列长度、索引、mask、量化及布局限制沿用原算子。

内核在 Q 的 L1、KV 的 UB 中补入 64 维零 RoPE，复用原有 512 维计算及 Flash Decode 缓冲，最终写回 448 维输出。该实现以功能正确为目标，保留内部缓冲占用。原 `rope_head_dim=64` 路径保留，其 Q/output512、mode 1 KV608B、mode 2 KV584B 契约不变。

## Native 调用入口

在已编译并安装本工作树 native 扩展和 custom OPP 的 A5 环境中加载：

```python
import torch
import torch_npu
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C

metadata_op = torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla_metadata
attention_op = torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla
```

先调用 Metadata 生成任务划分，再把结果传入 attention 的 `metadata` 参数。两次调用均传 `rope_head_dim=0`；Metadata 还需传 `head_dim=448`。其余实参参照测试中的 [调用实现](../../../tests/e2e/nightly/single_node/ops/mixed_quant_sparse_flash_mla/batch/mixed_quant_sparse_flash_mla_process.py)。

## 精度测试入口

在配置好 CANN、PyTorch 与 torch_npu 的 A5 环境中，从仓库根目录执行：

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
cd tests/e2e/nightly/single_node/ops/mixed_quant_sparse_flash_mla

# 原始 D512 / RoPE64 精度用例
python -m pytest -c pytest.ini --confcutdir=. -m ci -v -s test_mixed_quant_sparse_flash_mla_single.py

# D448 / RoPE0 精度用例
python -m pytest -c pytest.ini --confcutdir=. -m ci -v -s test_mixed_quant_sparse_flash_mla_rope0.py
```

RoPE0 用例使用独立 CPU attention golden 和原有误差阈值，覆盖两种量化模式、CSA/SWA/ORI_SPARSE、尾块、LSE 开关，并逐例记录和检查实际 Metadata 的 Flash Decode（FD）使能状态。普通执行要求 FD 关闭；FD 仅在 batch consistency 的运行时确定性级别 3 下可达，需由同级别的 Metadata 和 attention 调用共同验证。当前 104 环境的 torch_npu 和 CANN 接口均拒绝级别 3，因此尚未验证 FD 精度。Host/API UT 的运行入口为同目录的 `run_cpp_ut.sh`，与 NPU 精度测试分开记录。
