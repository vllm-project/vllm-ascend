# test_dispatch_ffn_combine_w4a8 参数说明

## 算子简介

测试 `torch.ops._C_ascend.dispatch_ffn_combine` 算子的正确性与性能。该算子将 MoE（Mixture of Experts）的 **dispatch → GMM1 → SwiGLU → GMM2 → combine** 全流程融合为单个 kernel，采用 **W4A8** 量化方案（INT4 weight × INT8 activation）。

## 参数总览

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `batch_size` | int | 64 | 输入 token 数量 `M`，每张卡处理的 token 总数 |
| `token_hidden_size` | int | 1024 | 隐藏层维度 `K`，即每个 token 的特征向量长度 |
| `moe_intermediate_size` | int | 512 | FFN 中间层维度 `N`（**post-SwiGLU**），GMM1 输出 `2N`，GMM2 输入 `N` |
| `top_k` | int | 8 | 每个 token 选择的 expert 数量 |
| `moe_expert_num` | int | 16 | **全局** expert 总数，单卡 expert 数 = 该值 ÷ `ep_world_size` |
| `ep_world_size` | int | 2 | EP（Expert Parallel）并行度，即 GPU/NPU 卡数 |
| `active_ratio_tensor_list` | int | 1 | `run_tensor_list` 中活跃 token 比例的分母，`active_num = M ÷ N`（1 = 全活跃） |
| `active_ratio_normal` | int | 1 | `run_normal` 中活跃 token 比例的分母，`active_num = M ÷ N`（1 = 全活跃） |
| `output_dir` | str | `./output` | 结果输出目录，profiler trace 也存于此 |
| `profile` | bool | False | 是否启用 `torch_npu.profiler` 采集 NPU 性能数据 |

## 核心 Tensor 形态

以默认参数为基准（`M=64, K=1024, N=512, E_global=16, E_per_rank=8`）：

### 输入端

| Tensor | 形状 | dtype | 说明 |
|--------|------|-------|------|
| `x` | `[M, K]` = `[64, 1024]` | bfloat16 | 输入 token 特征，前 `active_num` 行为 `randn`，其余行置零 |
| `expert_idx` | `[M, top_k]` = `[64, 8]` | int32 | 每行对应一个 token 的 8 个 expert 路由目标，值域 `[0, 15]`，确定性轮询生成 |
| `probs` | `[M, top_k]` = `[64, 8]` | float32 | 每个 expert 选择的 gating 权重，`randn` 随机值 |
| `x_active_mask` | `[M]` = `[64]` | bool | token 活跃掩码，前 `active_num` 个为 `True`，其余 `False` |

### 权重端

| Tensor | 形状 (packed INT32) | 形状 (unpacked INT4) | dtype | 说明 |
|--------|---------------------|----------------------|-------|------|
| `weight1` | `[E, K, 2N/8]` = `[8, 1024, 128]` | `[8, 1024, 1024]` | int32→int4 | GMM1 权重，`K × 2N`，gate+up 合并 |
| `weight2` | `[E, N, K/8]` = `[8, 512, 128]` | `[8, 512, 1024]` | int32→int4 | GMM2 权重，`N × K`，down projection |
| `scale1` | `[E, 2N]` = `[8, 1024]` | — | int64 | GMM1 per-channel 量化 scale |
| `scale2` | `[E, K]` = `[8, 1024]` | — | int64 | GMM2 per-channel 量化 scale |
| `bias1` | `[E, K]` = `[8, 1024]` | — | float32 | GMM1 bias（由 weight1 解包后求和得到） |
| `bias2` | `[E, N]` = `[8, 512]` | — | float32 | GMM2 bias（由 weight2 解包后求和得到） |

### 输出端

| Tensor | 形状 | dtype | 说明 |
|--------|------|-------|------|
| `out` | `[M, K]` = `[64, 1024]` | bfloat16 | 输出结果，与 `x` 同形状 |
| `expert_token_nums` | `[E]` = `[8]` | int32 | 每个 per-rank expert 实际处理的 token 数 |

## 数据流

```
x [M, K]
    │
    ▼
dispatch ──► expand_x [M*topk, K]   (per-expert token 展开)
    │
    ▼
GMM1 (weight1, W4A8) ──► [M*topk, 2N]   (gate + up)
    │
    ▼
SwiGLU ──► [M*topk, N]   (gate 激活 + up)
    │
    ▼
GMM2 (weight2, W4A8) ──► [M*topk, K]   (down projection)
    │
    ▼
combine ──► out [M, K]   (scatter-add 回原始 token)
```

## W4A8 量化方案

- **Weight**: INT32 打包存储，每个 INT32 含 8 个 4-bit signed INT4（值域 [-8, 7]）
- **Activation**: INT8，在 kernel 内部拆分为 high/low INT4 进行双路 INT4×INT4 矩阵乘
- **反量化**: 通过 `scale × (C_high × 16 + C_low + bias)` 恢复浮点结果

## CLI 用法

```bash
# 基础测试（所有默认值）
python test_dispatch_ffn_combine_w4a8.py

# 自定义参数
python test_dispatch_ffn_combine_w4a8.py \
    --batch_size 128 \
    --token_hidden_size 2048 \
    --moe_intermediate_size 1024 \
    --top_k 4 \
    --moe_expert_num 32 \
    --ep_world_size 4

# 性能 profiling
python test_dispatch_ffn_combine_w4a8.py \
    --profile \
    --output_dir ./profiler_results

# 查看 profiler 结果
tensorboard --logdir ./profiler_results
```
