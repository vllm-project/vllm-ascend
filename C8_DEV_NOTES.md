# C8 INT8 KV Cache 适配开发笔记

> 本文件记录了基于 vLLM Ascend（commit `3cc8bf1`）之上为 C8（INT8 KV Cache）量化模型进行适配的全部背景、改动和已知问题，供后续开发者快速上手。

---

## 一、背景

### 目标模型

`Qwen3-235B-A22B-Instruct-2507-w8a8c8-QuaRot`（本地路径 `/mnt/deepseek/lwy/model/Qwen3-235B-A22B-Instruct-2507-w8a8c8-QuaRot/`）

- 量化格式：W8A8C8（权重/激活/KV Cache 均为 INT8）
- KV cache 类型：`"kv_cache_type": "C8"`（见 `quant_model_description.json`）
- KV cache scale/offset：**per-channel**（每个 head_dim 位置一个 scale + offset），shape 为 `[num_kv_heads, head_size]`
- 量化公式：`x_int8 = round(x / scale + offset)`，反量化：`x = (x_int8 - offset) * scale`

### 部署方式

PD 分离（Prefill-Decode Disaggregation），使用 MooncakeConnector 传输 KV Cache：

- **Prefill**：DP2 TP8 EP16，单机 16 卡，脚本 `full-pd/start_prefill.sh`
- **Decode**：DP8 TP2 EP16，单机 16 卡，脚本 `full-pd/start_decode.sh`（`--cudagraph_mode NONE`）

---

## 二、代码改动总览（基于 `3cc8bf1`）

当前 `c8` 分支共 4 个干净 commit：

```
(HEAD)    fix(c8): fix KeyError when loading C8 scales with pipeline parallelism
          fix(c8): fix C8 decode startup and KV transfer issues for PD disaggregation
          fix(quant/moe): fix float32 scale issues for W8A8 linear and MoE
b635c310  feat(quant): add C8 INT8 KV cache support for Qwen/QuaRot models
90da8adf  feat: apply internal codebase fixes on top of 3cc8bf1
```

---

## 三、各文件改动详情

### 1. `vllm_ascend/attention/attention_v1.py`

**新增 C8 核心路径（AscendAttentionBackendImpl）：**

- **`_prepare_c8_scales(layer)`**
  - 懒加载：首次 forward 时按 TP rank 分片 `layer._c8_k_scale / _c8_v_scale / _c8_k_offset / _c8_v_offset`
  - 分片逻辑：当 `tp_size > num_kv_heads` 时，先按 head 数分片再取 1 个 head 数据；否则正常 narrow

- **`_quantize_kv_to_int8(k, v, layer)`**
  - 把 float16 K/V 量化为 INT8 存入 KV cache
  - `k_int8 = round(k / scale + offset).clamp(-128, 127).to(int8)`
  - 调用 `reshape_and_cache_flash` 写入 paged KV cache

- **`_dequant_paged_kv_to_dense(kv_cache, block_table, seq_lens, layer)`**
  - 从 paged INT8 KV cache 中 gather active blocks
  - 反量化回 float16：`(k_int8 - offset) * scale`
  - 返回 dense `[batch, seq_len, num_heads, head_dim]` tensor

- **`forward_c8_fused_infer_attention(...)`**
  - 统一的 C8 attention 入口
  - PrefillNoChunked / ChunkedPrefill：直接量化后存 cache，attention 用 float16
  - DecodeOnly（eager 模式）：`_dequant_paged_kv_to_dense` 后用 FIA V2
  - **注意**：CANN 当前版本 `aclnnFusedInferAttentionScoreV4` 的 `CheckFAIQKV` 拒绝 Q(float16) + K/V(INT8) 混合 dtype，即使设置 `key_quant_mode=0` 也无效 → **C8 decode 必须关闭 graph 模式（`cudagraph_mode=NONE`）**

- **`forward()` 路由**
  - 检测 `layer.c8_kv_cache_enabled` → 路由到 `forward_c8_fused_infer_attention`

### 2. `vllm_ascend/patch/worker/patch_qwen3_moe_c8.py` *(新文件)*

- 拦截 QuaRot 模型的 `load_weights`，将 per-channel `kv_cache_scale` 和 `kv_cache_offset`（shape `[num_kv_heads, head_size]`）赋给 attention layer
- 绕过上游 `numel==1` 断言（上游只支持 per-tensor scale）
- 在 `vllm_ascend/patch/worker/__init__.py` 中注册
- **PP 适配（bugfix）**：`_intercept_c8_scales` 改为：当 `scale_name is not None` 时，无论该层是否属于本 PP 分区，均在拦截器内消费掉，不再 `yield` 给原始 `load_weights`。否则原始函数会对不属于本分区的层做 `params_dict[scale_name]` → `KeyError`。

### 3. `vllm_ascend/quantization/methods/w8a8_dynamic.py`

- 新增 `AscendC8KVCacheAttentionMethod` 类
  - `create_weights`: 注册 `k_cache_scale`, `v_cache_scale`, `k_cache_offset`, `v_cache_offset`（float32 tensor），设置 `layer.c8_kv_cache_enabled = True`
- 修复 `AscendW8A8DynamicLinearMethod.apply()`：`weight_scale.to(torch.float32)`（npu_quant_matmul 对 float16 输出要求 float32 scale）

### 4. `vllm_ascend/quantization/methods/w8a8_static.py`

- 修复 `deq_scale` 初始化为 `float32`（原来对 float16 模型初始化为 int64，导致 checkpoint 值截断为 0，矩阵输出全零）

### 5. `vllm_ascend/quantization/modelslim_config.py`

- 新增 `get_cache_scale(weight_name)` 方法：将模型权重名（如 `...kv_cache_scale`）映射到 vLLM 内部属性名（如 `k_cache_scale`）
- 新增 `qwen3_5_moe` 到 `module_name_mapping` 和 `packed_modules_model_mapping`

### 6. `vllm_ascend/worker/model_runner_v1.py`

- KV cache dtype 设置：检测 `c8_kv_cache_enabled`，将 `kv_cache_spec.dtype = torch.int8`
- 其他内仓 fix：`sync_and_slice_intermediate_tensors`、`_dummy_run` 简化等

### 7. `vllm_ascend/ops/fused_moe/token_dispatcher.py`

- **最终结论**：`global_bs` 公式保持上游原始值 `min(max_num_seqs × 1, 512)`，**未做任何修改**。
- C8 模型（kv_producer，`max_num_batched_tokens=12288`）使用此公式可正常启动，**不会发生 DDR 越界**。
- 曾错误地认为需要将 `global_bs` 扩大到 `max_num_batched_tokens` 以防止 DDR overflow，实测证明这个改动完全不必要，并且会导致 HCCL_BUFFSIZE 暴涨到 ~6.4 GB。

### 8. `vllm_ascend/ops/fused_moe/moe_mlp.py`

- `w1_scale`, `w2_scale`, `swiglu_out_scale` 在 float16 输出时 `.to(float32)`（npu_grouped_matmul 同样要求 float32 scale）

### 9. `vllm_ascend/compilation/passes/norm_quant_fusion_pass.py`

- `AddRMSNormQuantPattern` 等 4 个 replacement 函数：在调用 `torch.ops.npu.npu_add_rms_norm_quant` 前将 `offset` 显式转为 `torch.int32`（CANN 要求 int32 或 bfloat16，不接受 float16）

### 10. `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`

- `reformat_kv_cache_with_fused_op()` 新增 INT8 KV cache 支持：
  - `aclnnTransposeKvCacheByBlock` 不支持 INT8
  - 将 INT8 cache 按 `torch.float16` view（2 个 INT8 byte = 1 float16），调用时 `head_dim // 2`
  - 字节级重排是安全的（该 op 只重排 head 和 block 维度，不在 head_dim 内分割）

### 11. `CMakeLists.txt`

- 移除 torch 版本检查

---

## 四、已知限制 & TODO

| 问题 | 状态 | 说明 |
|------|------|------|
| C8 decode 不支持 graph 模式 | ⚠️ 已绕过 | CANN `CheckFAIQKV` 拒绝混合 dtype；目前用 `cudagraph_mode=NONE`，待 CANN 升级后可实现 shadow float16 cache + graph replay |
| C8 精度：per-channel offset 降级为 per-tensor | ⚠️ 工程妥协 | `_prepare_c8_scales` 用 `mean()` 将 per-channel offset 压成 per-tensor scalar 作为 FIA 的 `dequant_offset`；对于标准 per-channel 量化理论上有精度损失，实测 PD 精度正常 |
| prefill HCCL_BUFFSIZE | ✅ 无问题 | `global_bs` 使用原始公式 `min(max_num_seqs, 512)`，C8 prefill 正常启动，HCCL buffer 极小 |
| `full-pd/` 目录在 `.gitignore` | ℹ️ | 启动脚本本地修改不会被 git 追踪 |

---

## 五、decode 启动脚本关键配置

```bash
# full-pd/start_decode.sh 关键参数
--compilation-config '{"cudagraph_capture_sizes":[...],"cudagraph_mode": "NONE"}'
# 必须关闭 graph 模式，原因见上方"已知限制"
```

---

## 六、参考

- 历史对话记录：`cb0cdb28-335b-4523-aa71-9a025b6dbc54`（含完整 debug 过程）
- 模型配置：`Qwen3-235B-A22B-Instruct-2507-w8a8c8-QuaRot/quant_model_description.json`
- 量化方法入口：`vllm_ascend/quantization/methods/w8a8_dynamic.py` → `AscendC8KVCacheAttentionMethod`
- Attention 路径：`vllm_ascend/attention/attention_v1.py` → `forward_c8_fused_infer_attention`
