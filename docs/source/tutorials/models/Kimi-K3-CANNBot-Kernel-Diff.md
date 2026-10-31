# CANNBot KDA 算子源码差异说明

本文记录 recipes 算子的来源、源码差异，以及 `kimi_kda.py` 中必要的 vLLM 调用适配。

## 合并基线与替换范围

本分支保留已有历史，并合并官方 `releases/v0.26.0rc` 的提交
`2ed6cbbaf481d84cfd3f2d01d47bcf79ee064c1f`。在此基础上，模型计算的替换范围仅为
Kimi K3 的 ShortConv 和 KDA；MoE、MLA、调度和其他模型使用合并后的上游实现。

ShortConv 参考 recipes 的 `models/kimi_k3/models/modeling_kimi_k3.py` 中
`KimiShortConvolution`：Prefill 调用 `cann_ops_transformer.causal_conv1d_fn`，
普通 Decode 使用 `[B, 1, D]` 调用 `causal_conv1d_update`，多 token Verify
使用二维输入并传入接受数。vLLM 的初始缓存标记仍由 attention metadata 提供，
以保留 chunked prefill 和缓存续算；不能照抄 recipes 冷启动 Prefill 的全零标记。

KDA Prefill 参考 recipes 的 `_prefill_flash_kda`，逐请求补齐至 64 token 的倍数，
以 `B=1` 调用 `flash_kda`，再还原有效 token 并写回对应请求的状态。
Decode/Verify 使用 `fused_recurrent_kda_op`。两个内核接收原始 gate 和 beta，
由内核完成激活；vLLM 的缓存索引和混合 batch 元数据适配予以保留。

为隔离本仓自定义卷积与官方卷积，本分支将自定义实现的底层注册名从
`CausalConv1d` 改为 `VllmCausalConv1d`，并同步修改生成的 ACLNN 入口、
kernel 入口、tiling 注册及构建目录。Python 接口
`torch.ops._C_ascend.npu_causal_conv1d_custom` 保持不变，供 GDN 等原有调用者使用；
Kimi 仍调用官方 `cann_ops_transformer.causal_conv1d_fn/causal_conv1d_update`。
这项改名需要重新编译安装本仓自定义算子包和 C++ 扩展，具体步骤见下节。
依赖仍包括现有 CANN 中的 `cann_ops_transformer`、`cannbot-dsl` 和 `ninja==1.13.0`；
CANNBot DSL 首次使用某个内核配置时仍会进行自身的 JIT 编译。

**验证范围**：仅完成源码核对，未在本地运行测试或执行 NPU 验证。
源码确认两套实现原先共用底层注册名，但尚未证明报错进程实际选中了哪套定义。
改名消除本仓与官方卷积之间的这处冲突；是否解决服务器的 `output 1` 内核匹配错误，
仍须在重新安装后验证。服务器须检查首请求、连续 Decode、
chunked prefill 和启用投机解码时的状态续算。

## 自定义卷积改名后的部署

| 调用方 | Python 接口 | ACLNN 入口 | 底层注册名 |
| --- | --- | --- | --- |
| Kimi ShortConv Prefill | `cann_ops_transformer.causal_conv1d_fn` | `aclnnCausalConv1dFn` | `CausalConv1d` |
| Kimi ShortConv Decode | `cann_ops_transformer.causal_conv1d_update` | `aclnnCausalConv1dUpdate` | `CausalConv1d` |
| 本仓自定义 GDN 卷积 | `_C_ascend.npu_causal_conv1d_custom` | `aclnnVllmCausalConv1d` | `VllmCausalConv1d` |

自定义算子的单输出定义、参数和计算逻辑保持不变。Ascend 310P 的独立
`CausalConv1dV310` 不参与本次改名。KDA recipes 内核与调用适配也保持不变。

先停止服务和 Ray workers，再在每个节点原来的 Python/CANN 环境中更新并编译安装。
以下示例使用服务器已有的 `wjcfork` remote：

```bash
cd /data/w50063966/vllm-ascend
git pull --ff-only wjcfork v0.26.0rc
COMPILE_CUSTOM_KERNELS=1 /usr/local/python3.11.10/bin/python3 -m pip install -v -e . \
    --no-build-isolation --no-deps
```

安装会重新生成 ACLNN、算子定义、tiling 和 kernel 产物，替换本 checkout 中的
`vllm_ascend/_cann_ops_custom`，并重新编译 C++ 扩展。仅拉取源码、仅修改 JSON、
仅复制新扩展或者仅安装算子 `.run` 包都不足以完成这次更新。
构建成功后，新进程启动时会检查本仓 metadata；如果仍含旧 `CausalConv1d` 注册，
会提示重新编译。该检查只覆盖本仓安装目录，不能排除其他目录中另有旧算子包。

在空闲 NPU 上运行共存回归，验证 FP16/BF16 下自定义卷积与官方 Prefill、续算、
Decode 的输出和缓存更新：

```bash
/usr/local/python3.11.10/bin/python3 -m pytest -v \
    tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_causal_conv1d_official.py
```

确认所有节点安装成功后，重新启动 Ray workers 和服务，并发送真实请求。
旧进程会缓存算子库和入口地址，因此更新后必须重启。

## 内核文件对比

以下比较两个算子文件：

1. `flash_kda.py`：Prefill KDA
2. `fused_recurrent_kda.py`：Decode/Verify KDA

对比基线：

- recipe：`cann-recipes-infer` 提交 `803c3120f483d3ae8d9f73d6d8941de25a2863d7`
- vLLM：`vllm-ascend` 提交 `fb872f96da91de07edbfe22d7d16ccd4d29c6519`

对应文件：

| 算子 | recipe 原始版本 | vllm-ascend 版本 |
| --- | --- | --- |
| Prefill | `cann-recipes-infer/ops/cannbot_dsl/flash_kda.py` | `vllm-ascend/ops/cannbot_dsl/flash_kda.py` |
| Decode/Verify | `cann-recipes-infer/ops/cannbot_dsl/fused_recurrent_kda.py` | `vllm-ascend/ops/cannbot_dsl/fused_recurrent_kda.py` |

## 总体结论

vllm-ascend 中的两个文件来源于 recipe。CANNBot DSL kernel 主体、数学公式、tensor 布局、数据类型、JIT 编译和算子接口均未修改。

源码差异只涉及调优参数的环境变量覆盖和一个 PyTorch 自动加载环境设置：

| 文件 | 实际变化 |
| --- | --- |
| `flash_kda.py` | 删除 `TORCH_DEVICE_BACKEND_AUTOLOAD` 设置；删除 `KDA_GROUP`、`KDA_DV_BASE` 手工覆盖 |
| `fused_recurrent_kda.py` | 删除 `KDA_ROW_BLOCK` 手工覆盖；同步修改报错文字 |

未设置这些环境变量时，recipe 与 vllm-ascend 会选择相同的自动配置，算子行为没有变化。

## flash_kda.py 的差异

### 删除进程级环境设置

recipe 原始版本：

```python
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
```

vllm-ascend 删除了以上代码。算子模块不再在 import 阶段修改整个 Python 进程的 PyTorch backend 自动加载行为。这个变化不涉及 KDA 计算。

### group 只使用自动选型

recipe 原始版本：

```python
group = int(os.environ.get("KDA_GROUP") or get_group_config_bn(bn_total))
```

vllm-ascend 版本：

```python
group = get_group_config_bn(bn_total)
```

recipe 可通过 `KDA_GROUP` 强制指定 chunk group。vllm-ascend 始终根据 batch 与 head 数自动选择。未设置 `KDA_GROUP` 时，两者完全相同。

### dv_base 只使用自动选型

recipe 原始版本：

```python
dv_base = int(
    os.environ.get("KDA_DV_BASE")
    or get_dv_base_config(bn_total, seq_len)
)
```

vllm-ascend 版本：

```python
dv_base = get_dv_base_config(bn_total, seq_len)
```

recipe 可通过 `KDA_DV_BASE` 强制指定 value 维度切分。vllm-ascend 始终根据 batch、head 和序列长度自动选择。未设置该变量时，两者结果相同。

除以上三处外，`flash_kda.py` 没有其他有效代码差异。

## fused_recurrent_kda.py 的差异

### row_block 只使用自动选型

recipe 原始版本：

```python
row_block_env = os.environ.get("KDA_ROW_BLOCK")
row_block = (
    int(row_block_env)
    if row_block_env
    else get_row_block_config(batch * num_value_heads, seq_len)
)
```

vllm-ascend 版本：

```python
row_block = get_row_block_config(
    batch * num_value_heads,
    seq_len,
)
```

recipe 可通过 `KDA_ROW_BLOCK` 强制指定 recurrent kernel 的行分块。vllm-ascend 始终使用自动选型。未设置该变量时，两者得到相同的 `row_block`。

因为不再读取环境变量，文件顶部的 `import os` 也被删除。

### 断言信息调整

recipe 原始版本：

```python
f"KDA_ROW_BLOCK must divide D and be one of 16, 32, 64, got {row_block}"
```

vllm-ascend 版本：

```python
f"row_block must divide D and be one of 16, 32, 64, got {row_block}"
```

这里只修改异常文字，不影响执行逻辑。

除以上内容外，`fused_recurrent_kda.py` 没有其他有效代码差异。

## 对功能和性能的影响

在没有设置 `KDA_GROUP`、`KDA_DV_BASE` 和 `KDA_ROW_BLOCK` 时，两个仓库都调用相同的自动选型函数，因此 kernel 配置、数值结果和预期性能应保持一致。

如果 recipe 的运行脚本显式设置了这些变量，差异才会出现：recipe 使用指定值，vllm-ascend 忽略指定值并继续自动选型。这可能改变编译出的 kernel 配置和性能，但不会改变 KDA 的数学定义。

以上结论来自源码逐行比较。本地没有 Atlas 950 环境，尚未进行 NPU 数值与性能验证。
