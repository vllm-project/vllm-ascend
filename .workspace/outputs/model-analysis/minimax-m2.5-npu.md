# MiniMax M2.5 NPU 适配分析

> 分析日期：2026-04-28
> 分析范围：vLLM upstream + vLLM-Ascend NPU 适配

## 1. 模型架构

MiniMax M2.5 是标准 GQA + MoE 架构，**不是 MLA**（与 DeepSeek 系列不同）。

```
MiniMaxM2DecoderLayer
  ├── MiniMaxM2Attention          ← 标准 GQA
  │     ├── QKVParallelLinear     ← Q/K/V 联合投影
  │     ├── MiniMaxText01RMSNormTP ← Q/K 各自 RMSNorm（TP 感知）
  │     ├── RotaryEmbedding       ← RoPE
  │     └── Attention             ← 标准 FlashAttention
  │
  └── MiniMaxM2MoE                ← MoE FFN
        ├── gate: ReplicatedLinear (Router)
        └── experts: FusedMoE (num_local_experts 个专家)
```

### 与 DeepSeek V3.2 / GLM5 对比

| 维度 | DeepSeek V3.2 | MiniMax M2.5 |
|------|---------------|-------------|
| 注意力 | MLA + SFA 稀疏 | **标准 GQA** |
| KV Cache | 压缩 c_kv (低秩) | 标准 K/V (head_dim 维) |
| Indexer | Lightning Indexer (TopK) | **无** |
| Q/K Norm | q_a_layernorm / kv_a_layernorm | **MiniMaxText01RMSNormTP (TP感知)** |
| MoE | DeepSeek MoE | FusedMoE |

## 2. vLLM 上游实现

核心文件：`vllm/model_executor/models/minimax_m2.py`

### Attention 原始流程

```python
# minimax_m2.py:227-238
def forward(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    # Q/K 各自做 TP 感知的 RMSNorm
    q, k = MiniMaxText01RMSNormTP.forward_qk(self.q_norm, self.k_norm, q, k)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output
```

### MiniMaxText01RMSNormTP

```python
# linear_attn.py:82-100 — Q/K 独立方差 + TP all_reduce
@staticmethod
def forward_qk(q_norm, k_norm, q, k):
    q_var = q.pow(2).mean(dim=-1, keepdim=True)
    k_var = k.pow(2).mean(dim=-1, keepdim=True)
    if q_norm.tp_world > 1:
        qk_var = torch.cat([q_var, k_var], dim=-1)
        qk_var = tensor_model_parallel_all_reduce(qk_var) / q_norm.tp_world
        q_var, k_var = qk_var.chunk(2, dim=-1)
    q = q * torch.rsqrt(q_var + eps) * q_norm.weight
    k = k * torch.rsqrt(k_var + eps) * k_norm.weight
    return q, k
```

## 3. NPU 路由分发

SGLang 使用 `AttentionBackendRegistry` 做 dispatch，而 vLLM-Ascend 不做路由——它通过 **monkey-patch** 直接修改上游 `MiniMaxM2Attention` 的行为：

```
vllm serve --device npu
  → ModelConfig._verify_quantization (patched)   ← 拦截 fp8
  → ModelConfig._verify_cuda_graph (patched)     ← 设置 HCCL_OP_EXPANSION_MODE
  → MiniMaxM2Model.load_weights (patched)         ← fp8 → bf16 反量化
  → MiniMaxM2Attention.__init__ (patched)         ← k_norm TP 分片
  → MiniMaxM2Attention.forward (patched)          ← 融合 QKV+Norm+RoPE
  → MiniMaxText01RMSNormTP.forward_qk (patched)   ← NPU 快速路径
  → MiniMaxM2MoE.forward (patched)                ← MoE all_reduce 替换
  → MiniMaxM2Model.forward (patched)              ← Eagle3 aux hidden states
```

## 4. 核心数据流

```
hidden_states
    │
    ├─ qkv_proj(hidden_states) → qkv
    │
    ├─[融合算子 split_qkv_tp_rmsnorm_rope]─────────┐
    │  切分 Q/K/V + Q/K 各自 RMSNorm + RoPE          │
    │  输入: qkv, q_norm.weight, k_norm.weight, cos, sin │
    │  输出: q, k, v  (已 norm + RoPE)              │
    │                                                │
    ├─[GQA Attention]────────────────────────────────┤
    │  self.attn(q, k, v)                            │
    │  → FlashAttention / FlashInfer                │
    │                                                │
    ├─ o_proj(attn_output) → output                 │
    │                                                │
    ├─[Residual + PostNorm]─────────────────────────┤
    │                                                │
    └─[MoE FFN]─────────────────────────────────────┘
       gate(hidden_states) → router_logits
       experts(hidden_states, router_logits)
       if tp_size > 1: maybe_all_reduce_tensor_model_parallel()
```

## 5. vLLM-Ascend NPU 适配（6 个 Patch）

### 5.1 Config 层：拦截 FP8 量化验证

```python
# patch_minimax_m2_config.py:74-107
def _patched_verify_quantization(self):
    # 检测 minimax_m2 + fp8 + NPU → 关闭 fp8 量化
    if _should_disable_fp8(self, getattr(self, "quantization", None)):
        cfg.quantization = None  # 后续当 bf16 处理（已反量化加载）
        return
    return _original_verify_quantization(self)
```

同时设置 ACL 图捕获参数：

```python
# patch_minimax_m2_config.py:110-129
def _patched_verify_cuda_graph(self):
    if device == "npu" and model_type == "minimax_m2":
        # 自动设置 HCCL_OP_EXPANSION_MODE=AIV，扩展算子匹配范围
        os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "AIV")
    return _original_verify_cuda_graph(self)
```

### 5.2 MoE all_reduce 替换

```python
# patch_minimax_m2.py:56-68
# 原版用 all_reduce，NPU 上用 maybe_all_reduce（EP 场景下跳过冗余通信）
MiniMaxM2MoE.forward = _patched_moe_forward

### 5.3 Attention：融合 QKV 切分 + RMSNorm + RoPE

```python
# patch_minimax_m2.py:97-119
def _patch_forward(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)
    cos, sin = get_cos_and_sin_slice()
    # 一次 kernel launch 完成 split + Q/K norm + RoPE
    q, k, v = torch.ops.vllm.split_qkv_tp_rmsnorm_rope(
        input=qkv,
        q_weight=self.q_norm.weight,
        k_weight=self.k_norm.weight,
        q_hidden_size=self.q_size,
        kv_hidden_size=self.kv_size,
        head_dim=self.head_dim,
        rotary_dim=getattr(self.rotary_emb, "rotary_dim", self.head_dim),
        eps=self.q_norm.variance_epsilon,
        tp_world=self.q_norm.tp_world,
        cos=cos, sin=sin,
    )
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output
```

同时 hook `__init__` 处理 k_norm 的 TP 分片（当 kv_heads < tp_size 时需要复制）：

```python
# patch_minimax_m2.py:80-92
def _patched_attention_init(self, *args, **kwargs):
    _original_attention_init(self, *args, **kwargs)
    tp_size = get_tensor_model_parallel_world_size()
    self.num_kv_head_replicas = max(1, tp_size // self.total_num_kv_heads)
    if self.total_num_kv_heads < tp_size:
        self.k_norm = MiniMaxText01RMSNormTP(
            ...,
            weight_shard_world_size=self.total_num_kv_heads,
            weight_shard_rank=tp_rank // self.num_kv_head_replicas,
        )
```

### 5.4 FP8 权重反量化加载

```python
# patch_minimax_m2.py:128-196
# NPU 不支持原生 FP8 → 加载时 block-wise dequant 到 bf16
def _dequantize_fp8_block_weight(fp8_weight, weight_scale_inv, block_size):
    n, k = fp8_weight.shape
    block_n, block_k = block_size
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    expanded_scale = weight_scale_inv.repeat_interleave(block_n, dim=0)\
                                     .repeat_interleave(block_k, dim=1)
    expanded_scale = expanded_scale[:n, :k].to(torch.bfloat16)
    return fp8_weight.to(torch.bfloat16) * expanded_scale
```

配套 config 层 patch：检测到 minimax_m2 + fp8 时自动 `cfg.quantization = None`。

### 5.5 Q/K RMSNorm NPU 快速路径

```python
# patch_minimax_m2_linear_attn.py:53-93
def _patched_qk(q_norm, k_norm, q, k):
    if current_platform.device_name == "npu":
        # 硬件加速 npu_rms_norm
        q, q_inv_rms = torch.ops.npu.npu_rms_norm(q, q_norm.weight, eps)
        k, k_inv_rms = torch.ops.npu.npu_rms_norm(k, k_norm.weight, eps)

        if q_norm.tp_world > 1:
            # 本地方差 → all_reduce → 全局方差 → rstd 校正
            q_local_var = ...   # 从 q_inv_rms 反算
            k_local_var = ...
            qk_var = tensor_model_parallel_all_reduce(...) / q_norm.tp_world
            q_global_var, k_global_var = qk_var.chunk(2)

            q = q * (q_global_rstd / q_local_rstd).to(q.dtype)
            k = k * (k_global_rstd / k_local_rstd).to(k.dtype)
        return q, k
```

原理：`npu_rms_norm` 用硬件加速做局部的 normalize，但 TP 场景下需要全局方差做校正。公式为：

```
x_global = x_local * rstd_global / rstd_local
```
其中 `rstd = 1 / sqrt(var + eps)`，`var` 通过 all_reduce 得到全局值。

### 5.6 Linear Attention NPU Kernel（BailingMoE 间接使用）

MiniMax M2.5 主模型不使用 linear attention，但 BailingMoE 借用了 `MiniMaxText01LinearKernel`。
NPU 上需要用 Titan 重写 GPU Triton kernel，关键调整是 UB-safe tiling：

```python
# lightning_attn.py — NPU 友好的 Triton kernel，4 个 kernel：
# _fwd_diag_kernel:      对角块注意力，  CBLOCK=32  (UB ≈ 72KB < 192KB)
# _fwd_kv_parallel:      KV 外积并行计算，CBLOCK=64  (UB ≈ 112KB)
# _fwd_kv_reduce:        块间 KV 前缀归约，UB ≈ 64KB
# _fwd_none_diag_kernel: 非对角块注意力，CBLOCK=64  (UB ≈ 96KB)

BLOCK = 256              # 统一的分块大小
E_FBLOCK = e // 2        # e 维度拆半避免 UB 溢出
```

### 5.7 Eagle3 推测解码支持

```python
# patch_minimax_m2.py:220-289
# 扩展 forward 支持收集中间 hidden states（给 Eagle3 draft model）
def _patched_minimax_m2_forward(self, input_ids, positions, ...):
    for idx, layer in enumerate(self.layers):
        if layer_idx in aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states + residual)
        hidden_states, residual = layer(positions, hidden_states, residual)
    # 返回 (hidden_states, [aux_layer_0, aux_layer_n//2, aux_layer_n-3])
    return hidden_states, aux_hidden_states
```

## 6. 部署配置

关键环境变量和参数：

```bash
# ACL 图捕获优化
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024

# 通信优化
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

# 显存管理
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 推理启动
vllm serve /path/to/MiniMax-M2.5 \
    --enable-expert-parallel \
    --tensor-parallel-size 16 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

### 部署方案

| 方案 | 配置 | 适用场景 |
|------|------|----------|
| A3 单机（短文本） | TP16 DP1 | 3.5k 输入，低延迟 |
| A3 单机（长文本） | TP8 CP2 | 128k 输入 |
| A2 双机 | TP8 DP2 (每机 8 卡) | 190k 超长文本 |

## 7. 与 DeepSeek V3.2 的 NPU 适配对比

| 维度 | DeepSeek V3.2 SFA | MiniMax M2.5 |
|------|-------------------|-------------|
| 注意力类型 | MLA + Sparse (吸收 MQA) | 标准 GQA |
| Indexer | npu_lightning_indexer | 无 |
| NPU 注意力算子 | npu_sparse_flash_attention | Attention (标准 FlashAttn) |
| 融合算子 | MLAPO (qkv+norm+rope+吸收) | split_qkv_tp_rmsnorm_rope |
| Q/K Norm | q_a_layernorm / kv_a_layernorm | MiniMaxText01RMSNormTP |
| 权重格式 | bf16 | FP8 → bf16 反量化 |
| 特殊通信 | CP balance | Expert Parallel + FlashComm |
| 稀疏计算 | TopK 选择 + 稀疏注意力 | 全量注意力 |

## 8. 结论

MiniMax M2.5 在 NPU 上的适配比 DeepSeek V3.2 **简单得多**。核心工作：

1. **FP8 → bf16**（权重量化反量化——适配投入最大的单项工作）
2. **Q/K Norm 的 TP 感知优化**（NPU 快速路径 + 全局方差校正）
3. **算子融合**（split + norm + RoPE）
4. **MoE 通信替换**（all_reduce → maybe_all_reduce + Expert Parallel）
