参考文档：

https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/%EF%BC%88beta%EF%BC%89torch_npu-npu_mla_prolog_v3.md

https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_quantize.md

https://zhuanlan.zhihu.com/p/16730036197

https://zhuanlan.zhihu.com/p/1897225385751585767

---

## MLAPO v3 实现方案澄清

### 1. 当前实现的量化模式

基于 `sfa_v1.py:_sfa_preprocess_decode_v3` 方法，当前 MLAPO v3 采用的是 **部分量化（Partial Quantization）** 场景中的 **kv_cache per-tile 量化** 模式：

| 参数 | 当前值 | 含义 |
|------|--------|------|
| `weight_quant_mode` | 2 | weight_dq、weight_uq_qr、weight_dkv_kr 采用 int8 per-channel 量化 |
| `kv_cache_quant_mode` | 3 | kv_cache 和 kr_cache 采用 per-tile 量化 |
| `query_quant_mode` | 0 | query 输出非量化 |
| `ckvkr_repo_mode` | 1 | kv_cache 和 kr_cache 合并存储 |
| `quant_scale_repo_mode` | 1 | scale 和数据合并存储 |

### 2. 各参数的 dtype 对应关系

| 参数 | dtype | 说明 |
|------|-------|------|
| `token_x` (quanted_hidden_states) | int8 | 经过 `npu_dynamic_quant` 量化的输入 |
| `weight_dq` | int8 | W^DQ 权重，FRACTAL_NZ 格式 |
| `weight_uq_qr` | int8 | W^UQ + W^QR 权重，FRACTAL_NZ 格式 |
| `weight_uk` | bf16 | W^UK 权重，保持 bf16 非量化 |
| `weight_dkv_kr` | int8 | W^DKV + W^KR 权重，FRACTAL_NZ 格式 |
| `kv_cache` | bf16 | 非量化 |
| `kr_cache` | bf16 | 非量化 |
| `dequant_scale_x` | float | token_x 的反量化 scale |
| `dequant_scale_w_dq` | float | weight_dq 的 per-channel scale |
| `dequant_scale_w_uq_qr` | float | weight_uq_qr 的 per-channel scale |
| `dequant_scale_w_dkv_kr` | float | weight_dkv_kr 的 per-channel scale |

### 3. 关键算子调用流程

```python
# Step 1: 合并 kv_cache (ckvkr_repo_mode=1)
# k_nope [B, N, S, Hckv] 和 k_pe [B, N, S, Dr] 合并为 [B, N, S, Hckv+Dr]
kv_cache_combined = torch.cat([k_nope, k_pe], dim=-1)

# Step 2: 对输入 hidden_states 进行动态量化
quanted_hidden_states, quanted_hidden_states_scale = torch_npu.npu_dynamic_quant(hidden_states)

# Step 3: 准备 per-tile 量化 scale
quant_scale_ckv = dequant_scale_w_dkv_kr[:, :self.kv_lora_rank]

# Step 4: 调用 npu_mla_prolog_v3 执行前处理
query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = torch_npu.npu_mla_prolog_v3(
    token_x=quanted_hidden_states,
    weight_dq=self.weight_dq,  # int8, FRACTAL_NZ
    weight_uq_qr=self.weight_uq_qr,  # int8, FRACTAL_NZ
    weight_uk=self.weight_uk,  # bf16
    weight_dkv_kr=self.weight_dkv_kr,  # int8, FRACTAL_NZ
    kv_cache=kv_cache_combined,  # k^C + k^R 合并 cache
    kr_cache=kv_cache_combined,  # 相同 buffer (ckvkr_repo_mode=1)
    cache_index=cache_index.to(torch.int64),
    dequant_scale_x=quanted_hidden_states_scale.unsqueeze(-1),
    dequant_scale_w_dq=self.dequant_scale_w_dq,
    dequant_scale_w_uq_qr=self.dequant_scale_w_uq_qr,
    dequant_scale_w_dkv_kr=self.dequant_scale_w_dkv_kr,
    quant_scale_ckv=quant_scale_ckv,  # per-tile scale
    quant_scale_ckr=None,  # ckvkr_repo_mode=1 时不使用
    smooth_scales_cq=self.smooth_scales_cq,
    actual_seq_len=None,
    k_nope_clip_alpha=None,
    k_pe_clip_alpha=None,
    cache_mode='PA_BSND',
    query_norm_flag=True,
    weight_quant_mode=2,  # 权重量化
    kv_cache_quant_mode=3,  # per-tile 量化
    query_quant_mode=0,   # query 非量化
    ckvkr_repo_mode=1,    # kv_cache 和 kr_cache 合并存储
    quant_scale_repo_mode=1,  # scale 和数据合并存储
    tile_size=128,
    qc_qr_scale=1.0,
    kc_scale=1.0,
)
```

### 4. ckvkr_repo_mode=1 模式说明

当 `ckvkr_repo_mode=1` 时：
- 输入的 `kv_cache` 和 `kr_cache` 需要是合并后的 tensor
- `kv_cache_combined = torch.cat([k_nope, k_pe], dim=-1)`
- 两个参数传入相同的 buffer，算子内部会按 `Hckv` 和 `Dr` 进行拆分

### 5. 返回值使用

- `query`: 标准 Query 输出 (q^N)，用于后续 attention 计算
- `query_rope`: 位置编码 Query 输出 (q^R)，用于 RoPE attention
- `dequant_scale_q_nope`: query 的反量化 scale
- `query_norm`: RmsNorm 后的 query (q^C)
- `dequant_scale_q_norm`: query_norm 的反量化 scale（量化场景下）

### 6. weight_quant_mode=2 的约束条件

根据文档，当 `weight_quant_mode=2` 时：
- 输入要求：
  - `token_x`: bf16（实际代码中先进行了动态量化）
  - `weight_dq`, `weight_uq_qr`, `weight_dkv_kr`: int8 per-channel 量化数据
  - `weight_uk`: bf16（非量化）
  - 必须传入 `dequant_scale_w_dq`, `dequant_scale_w_uq_qr`, `dequant_scale_w_dkv_kr`
- 输出：所有出参返回非量化数据

### 7. 返回值 dtype 汇总

对于当前实现的 `weight_quant_mode=2, kv_cache_quant_mode=3` 场景：

| 返回值 | dtype | shape | 说明 |
|--------|-------|-------|------|
| `query` | bfloat16 | [T, N, Hckv] | 标准 Query 输出 q^N |
| `query_rope` | bfloat16 | [T, N, Dr] | 位置编码 Query 输出 q^R |
| `dequant_scale_q_nope` | float | [0] | 当前场景下无效 |
| `query_norm` | int8 | [T, Hcq] | RmsNorm 后的 q^C（权重量化时输出 int8） |
| `dequant_scale_q_norm` | float | [T, 1] | query_norm 的反量化 scale |

**注意**：`weight_quant_mode=2` 时，`query_norm` 输出为 **int8**，需要配合 `dequant_scale_q_norm` 进行后续计算。

1. **per-tile 量化**：对 KV cache 进行 per-tile 量化，结合 ckvkr_repo_mode=1 合并存储
2. **合并存储**：kv_cache 和 kr_cache 合并为一个 tensor，scale 也与数据合并存储
3. **兼容性好**：通过 ckvkr_repo_mode=1 简化了 KV cache 的存储和读取
