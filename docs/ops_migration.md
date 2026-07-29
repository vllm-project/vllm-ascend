# DeepSeek-V4 算子解耦迁移总结

> 本文档记录 vllm-ascend 中 DeepSeek-V4 调用链上所有 `torch.ops._C_ascend.*` 算子迁移到外部 CANN 仓库或替代方案的情况。
> 后续如有新改动，请同步更新本文档。

---

## 1. 背景

### 1.1 目标

将 vllm-ascend 中 DeepSeek-V4 调用链上的所有 `_C_ascend` 算子替换为 CANN 仓库提供的算子或替代方案。仅修改 vllm-ascend，不修改 CANN 仓库。

### 1.2 算子来源与替代方案

| 来源 | 命名空间 | 优先级 |
|------|---------|--------|
| **ops-transformer** | `torch.ops.cann_ops_transformer.*` | 1（优先） |
| **cann-recipes-infer** | `torch.ops.custom.*` | 2 |
| **torch_npu 原生** | `torch_npu.npu_*` | 3 |
| **Python 预计算/分解** | 纯 PyTorch + 上述算子组合 | 4（无外部对应时） |
| **vllm-ascend csrc** | `torch.ops._C_ascend.*` | 保留（确实无替代时） |

### 1.3 兼容性判断原则

- aclnn API 名称和参数一致 → 直接替换
- aclnn API 不同但有 torch binding 且功能等价 → 替换 + 参数适配
- 无 torch binding 但有 aclnn 源码 → 不迁移（不修改 CANN 仓库）
- 完全无外部对应 → Python 预计算或分解为已有算子组合（参考 cann-recipes DSV4 实现）

### 1.4 V4 调用链

```
AscendDeepseekV4ForCausalLM
  → DeepseekV2DecoderLayer
    ├── DeepseekV4Attention → AscendDeepseekSparseAttention (ops/dsa.py)
    │     → DSAAttention → AscendDSABackend → AscendDSAImpl (attention/dsa_v1.py)
    │       或 AscendDSACPImpl (attention/context_parallel/dsa_cp.py)
    ├── DeepseekV4MoE → FusedMoE → AscendMoERunner
    │     → select_experts (ops/fused_moe/experts_selector.py)
    │     → fused_experts (ops/fused_moe/moe_comm_method.py)
    │     → unified_apply_mlp (ops/fused_moe/moe_mlp.py)
    └── RMSNorm (ops/layernorm.py)
```

V4 的 attention 走 **DSA 路径**，不走 SFA/MLA。

### 1.5 约束

- 仅修改 vllm-ascend，不修改 CANN 仓库
- csrc 不清理
- 目标平台：A5 (Ascend 910C / arch35 / `__DAV_C310__`)，同时兼容 A3 (Ascend 910B)

### 1.6 同名算子覆盖原则

同一个 aclnn 算子如果在 vllm-ascend、cann-recipes-infer 或 ops-transformer 的 `.run`/SO 中同名注册，运行时会优先命中 vllm-ascend 已安装的实现，然后才会使用 recipes/transformer 的实现。因此迁移方案不仅要修改 Python 调用命名空间，还必须同步调整 vllm-ascend 编译脚本中的算子列表：

- 最终使用 vllm-ascend 实现：必须在 vllm-ascend 编译脚本中保留该算子，确保对应 `.run` 安装。
- 最终使用 recipes/transformer 实现：必须从 vllm-ascend 编译脚本中移除同名算子，避免 vllm-ascend SO 覆盖外部实现。
- 按设备分流的算子：需要按 SOC 编译列表分别处理；例如 A3/Base 使用 vllm-ascend `compressor`，A5 使用 recipes `custom.compressor`，因此 A3 编译列表保留 `compressor`，A5 编译列表不编译 `compressor`。
- 后续新增/回退任一算子时，必须同时更新 Python 分流、`csrc/build_aclnn.sh` 编译列表和本文档。

---

## 2. 已迁移的算子

### 2.1 Attention — DSA 模块

| # | 算子 | 原 `_C_ascend` | 迁移到 | 调用文件 | 签名调整 |
|---|------|---------------|--------|---------|---------|
| 1 | inplace_partial_rotary_mul | `inplace_partial_rotary_mul` | `cann_ops_transformer.inplace_partial_rotary_mul` | dsa_v1.py, dsa_cp.py (13处) | 无，aclnn API + 参数完全一致 |
| 2 | compressor | `compressor` | 按设备分流：A3/Base 保留 `_C_ascend.compressor`，A5 使用 `custom.compressor` | device_op.py 分流；dsa_v1.py, dsa_cp.py (6处) 调用 `DeviceOperator.dsa_compressor` | keyword args 分隔；A3 回退原 vllm-ascend wrapper，避免 cann-recipes wrapper 对 `state_cache` 连续性/dtype 的额外约束 |
| 3 | npu_quant_lightning_indexer | `npu_vllm_quant_lightning_indexer` | `custom.npu_quant_lightning_indexer` | dsa_v1.py, dsa_cp.py (4处) | 多可选参数 |
| 4 | npu_quant_lightning_indexer_metadata | `npu_vllm_quant_lightning_indexer_metadata` | `custom.npu_quant_lightning_indexer_metadata` | dsa_v1.py, dsa_cp.py (3处) | device 默认值差异 |
| 5 | npu_rms_norm_dynamic_quant | `npu_rms_norm_dynamic_quant` | `custom.npu_rms_norm_dynamic_quant` | dsa_v1.py, dsa_cp.py (4处) | 无 |
| 6 | npu_sparse_attn_sharedkv | `npu_sparse_attn_sharedkv` | `custom.npu_sparse_attn_sharedkv` | device_op.py (BaseAdaptor) | 默认值差异，显式传参 |
| 7 | npu_sparse_attn_sharedkv_metadata | 同上 | `custom.npu_sparse_attn_sharedkv_metadata` | device_op.py (BaseAdaptor) | 同上 |
| 8 | npu_kv_quant_sparse_attn_sharedkv | `npu_kv_quant_sparse_attn_sharedkv` | `custom.npu_kv_quant_sparse_attn_sharedkv` | device_op.py (A5Adaptor) | A5 路径需与 `kv_compress_epilog(quant_mode="fp8_bf16")` 配套 |
| 9 | npu_kv_quant_sparse_attn_sharedkv_metadata | 同上 | `custom.npu_kv_quant_sparse_attn_sharedkv_metadata` | device_op.py (A5Adaptor) | 同上 |
| 10 | scatter_nd_update_asc | `npu_scatter_nd_update_v2` | `torch_npu.npu_scatter_nd_update_` | device_op.py (BaseAdaptor, 7处) | 纯改名；in-place op（3 positional args 不变）；保留 `[:x_flat.shape[0]]` 截取逻辑 |
| 11 | kv_compress_epilog | `kv_compress_epilog` | `cann_ops_transformer.kv_compress_epilog` | device_op.py (A5Adaptor) | A5 custom sparse attention 路径使用 `quant_mode="fp8_bf16"`；cache 保持 4D 不 reshape（tiling 要求 4D `[blockNum, blockSize, 1, headDim]`） |
| 12 | indexer_compress_epilog_v2 | `indexer_compress_epilog_v2` | `cann_ops_transformer.indexer_quant_cache` | device_op.py (A5Adaptor, 3处) | `cache_scale` 必须为 float32（op_def 约束）；cache/cache_scale 保持 4D 不 reshape（tiling `ValidateCache4D` 要求 4D）；参数名 `indexer_full_cache`→`indexer_scale_cache` |

### 2.2 MHC (Hyper-Connection) 模块

| # | 算子 | 原 `_C_ascend` | 迁移到 | 调用文件 | 签名调整 |
|---|------|---------------|--------|---------|---------|
| 13 | npu_hc_pre | `npu_hc_pre_v2` | `cann_ops_transformer.mhc_pre_sinkhorn` | deepseek_v4.py (1处) | 参数改名（`hc_fn`→`phi`, `hc_scale`→`alpha`, `hc_base`→`bias`）；顺序交换（`hc_eps, norm_eps`）；输入 unsqueeze 3D→4D；返回 8 元组取前 3；comb `unflatten(-1, (hc_mult, hc_mult))`+`squeeze(1)`；不同 aclnn API（`aclnnHcPre`→`aclnnMhcPreSinkhorn`） |
| 14 | npu_hc_post | `npu_hc_post` | `cann_ops_transformer.mhc_post` | deepseek_v4.py (1处) | 参数顺序重排 `(x,residual,post,comb)`→`(residual,comb,x,post)`；去掉 unsqueeze/squeeze workaround；不同 aclnn API（`aclnnHcPost`→`aclnnMhcPost`） |

### 2.3 MoE — Gating 模块

| # | 算子 | 原 `_C_ascend` | 迁移到 | 调用文件 | 签名调整 |
|---|------|---------------|--------|---------|---------|
| 15 | moe_gating_top_k_hash | `moe_gating_top_k_hash` | `custom.npu_moe_gating_top_k` | experts_selector.py (1处) | `x`/`k` 改 positional；hash 路径保留 `custom`（torch_npu 不支持 `input_ids`/`tid2eid`） |
| 16 | moe_gating_top_k (非hash) | `moe_gating_top_k` | `torch_npu.npu_moe_gating_top_k` | device_op.py (BaseAdaptor, 1处) | 去掉 `input_ids`/`tid2eid`（非 hash 路径不需要）；ops-transformer 内核原生支持 `renorm=0/1`（无需 Python 后处理） |

### 2.4 MoE — Routing 模块

| # | 算子 | 原 `_C_ascend` | 迁移到 | 调用文件 | 签名调整 |
|---|------|---------------|--------|---------|---------|
| 17 | npu_moe_init_routing_custom | `npu_moe_init_routing_custom` | `torch_npu.npu_moe_init_routing_v2` | device_op.py (BaseAdaptor, 1处) | 纯改名；A5 路径已用此 op 验证可行 |

### 2.5 MoE — Quant/Activation 模块

| # | 算子 | 原 `_C_ascend` | 迁移到 | 调用文件 | 签名调整 |
|---|------|---------------|--------|---------|---------|
| 18 | npu_swiglu_group_quant | `npu_swiglu_group_quant` | `custom.npu_swiglu_group_quant` | device_op.py, fused_moe.py, fused_moe_0_23_0.py (3处) | `topk_weight`→`weight`，`clamp_value`→`clamp_limit`；`quant_mode=2`(MX_QUANT)→`quant_mode=1`(cann MX_QUANT)；输入需 bf16/f16（不能传 int32）；MX_QUANT 时 `round_scale=True` |
| 19 | npu_dequant_swiglu_quant | `npu_dequant_swiglu_quant` | `torch_npu.npu_dequant_swiglu_quant` | fused_moe.py, fused_moe_0_23_0.py, moe_mlp.py (3处) | 去掉 `bias`/`quant_offset`；条件传参：无 `swiglu_limit` 时不传 `swiglu_mode`/`clamp_limit`（用默认 `swiglu_mode=0` 不钳位）；有 `swiglu_limit` 时共享专家 `swiglu_mode=2`（contiguous），路由专家 `swiglu_mode=1`（interleaved） |

### 2.6 MoE — MC2 Communication 模块

| # | 算子 | 原 `_C_ascend` | 迁移到 | 调用文件 | 签名调整 |
|---|------|---------------|--------|---------|---------|
| 20 | dispatch_ffn_combine | `dispatch_ffn_combine` | `super().fused_experts()` (基类非融合路径) | moe_comm_method.py (1处) | 分解为 dispatch + 3阶段 FFN + combine，使用 `npu_moe_distribute_dispatch_v2` + `npu_moe_distribute_combine_v2` |
| 21 | dispatch_gmm_combine_decode | `dispatch_gmm_combine_decode` | `super().fused_experts()` (基类非融合路径) | moe_comm_method.py (1处) | 同上 |

### 2.7 Norm 模块

| # | 算子 | 原 `_C_ascend` | 迁移到 | 调用文件 | 签名调整 |
|---|------|---------------|--------|---------|---------|
| 22 | npu_add_rms_norm_bias | `npu_add_rms_norm_bias` | `torch_npu.npu_add_rms_norm` + `x.add_(bias)` | layernorm.py (2处) | bias 不再融合但功能等价 |

### 2.8 保持当前来源的算子（属第三组 HARD，待后续处理）

以下算子在 ops-transformer 中有对应版本，但因改动量大或参数不兼容暂未切换：

| 算子 | 当前来源 | ops-transformer 对应 | 不切换原因 |
|------|---------|---------------------|-----------|
| `compressor` | A3/Base `_C_ascend.compressor` (18参融合)，A5 `custom.compressor` | `cann_ops_transformer.compressor` (12参，无 norm/rope) | ops-transformer 版不含 RMSNorm+RoPE 融合，拆分 3 步会性能下降（属第三组 HARD） |
| `npu_quant_lightning_indexer` + metadata | `custom.npu_quant_lightning_indexer` | `cann_ops_transformer.quant_lightning_indexer` (V2) | V2 要求 `layout_k="PA_BBND"`，V4 用 `PA_BSND`；需改 KV cache 布局（属第三组 HARD） |
| `npu_sparse_attn_sharedkv` + metadata | `custom.npu_sparse_attn_sharedkv` | `cann_ops_transformer.sparse_flash_mla` | 算子名和参数结构大幅变化，metadata 签名重写（属第三组 HARD） |
| `moe_gating_top_k_hash` | `custom.npu_moe_gating_top_k` | `torch_npu.npu_moe_gating_top_k` | torch_npu 不支持 hash 模式（`input_ids`/`tid2eid`），hash 路径保留 `custom` |

---

## 3. 未迁移的算子

### 3.1 compressor_metadata（回退到 `_C_ascend`，graph capture 不兼容）

> **状态：已回退。** Python 预计算方案与 `UNIFORM_BATCH` graph capture/replay 不兼容。

| 算子 | 原 `_C_ascend` | 调用文件 | 原因 |
|------|---------------|---------|------|
| compressor_metadata | `compressor_metadata` | dsa_v1.py (6处), dsa_cp.py (2处) | 见下方分析 |

**根因分析：**

原始 `_C_ascend.compressor_metadata` 在 forward 路径执行，作为 custom op 注册为 graph node。graph replay 时读取 scheduler 更新的 `start_pos_decode` tensor 的当前值。

Python 预计算在 metadata build 阶段执行，读取 tensor 值后计算结果存为常量。graph replay 时 `build_decode_metadata` 不再调用，预计算结果被冻结为 capture 时的值。导致首 token 正确，后续 token 因 `start_pos` 更新但预计算结果未更新而精度错误。

**结论：** `compressor_metadata` 必须作为 graph node 在 forward 路径执行，不能预计算。csrc 不清理，保留 `_C_ascend` 调用。

### 3.2 MoE — GMM 模块（暂未迁移，保留 `_C_ascend`）

> **状态：暂未迁移。** 3 阶段分解方案在 A3 上调试时遇到 `npu_grouped_matmul` 的 A8W8 per-token quant dtype 不匹配问题（`scale` dtype FLOAT 与 `output_dtype` FLOAT16 不兼容），已回退。

| 算子 | 原 `_C_ascend` | 调用文件 | 原因 |
|------|---------------|---------|------|
| grouped_matmul_swiglu_quant_weight_nz | `grouped_matmul_swiglu_quant_weight_nz` | device_op.py (BaseAdaptor, 1处) | 见下方方案分析 |
| grouped_matmul_swiglu_quant_weight_nz_tensor_list | `grouped_matmul_swiglu_quant_weight_nz_tensor_list` | moe_mlp.py (2处) | 同上 |
| grouped_matmul_swiglu_quant_v2 | `grouped_matmul_swiglu_quant_v2` | moe_mlp.py (1处) | 同上 |

**可行方案（待后续实现）：**

参考 cann-recipes DSV4 `CompressedTensorW8A8Int8MoEGMMMethod`（`compressed_tensors_moe_gmm.py:155-194`）和 `modeling_deepseek.py:442-498`：

1. **gmm1 阶段**：`npu_grouped_matmul` **不传 `scale`/`per_token_scale`**，`output_dtype=torch.int32`
   - cann-recipes W8A8Int8 路径（L159-165）：gmm1 不传 scale，output=int32
   - **关键**：W8A8Int8 路径的 `weight_scale` 是 float32，不能直接传给 `npu_grouped_matmul` 的 `scale` 参数（报 dtype 不匹配）

2. **dequant + swiglu + quant 阶段**：`npu_dequant_swiglu_clamp_quant`，传入 `weight_scale` 做反量化
   - 参数：`x=gmm1_out(int32)`, `weight_scale=w1_scale(float32)`, `activation_scale=pertoken_scale`, `quant_mode=1`(dynamic), `swiglu_mode=0`/`1`

3. **gmm2 阶段**：`npu_grouped_matmul` 传 `scale=[w2_weight_scale]` + `per_token_scale=[swiglu_out_scale]`，`output_dtype=bf16`

**A5 路径不受影响**：`A5DeviceAdaptor.npu_grouped_matmul_swiglu_quant` 使用 `torch_npu.npu_grouped_matmul_swiglu_quant_v2`（torch_npu 原生融合算子），不走 `_C_ascend`。

### 3.3 完全无外部对应（vllm-ascend 独有）

| 算子 | 模块 | aclnn API | 说明 |
|------|------|-----------|------|
| `store_kv_block` / `store_kv_block_pre` | SFA | `aclnnStoreKVBlock` | 两仓库均无（SFA 路径，V4 不走） |
| `batch_matmul_transpose` | SFA | 自研 kernel | 两仓库均无（SFA 路径，V4 不走） |

### 3.4 有 aclnn 源码但无 torch binding（不修改 CANN 仓库）

| 算子 | 模块 | CANN 仓库情况 | 无法迁移原因 |
|------|------|-------------|-----------|
| `grouped_matmul_swiglu_quant_weight_nz` | MoE-GMM | ops-transformer 有 `aclnnGroupedMatmulSwigluQuantWeightNZ` 源码 | 无 torch_extension binding。vllm-ascend 版 aclnn 多 `limited` 参数。3 阶段分解方案见 3.1 节 |
| `grouped_matmul_swiglu_quant_v2` | MoE-GMM | ops-transformer 有 `aclnnGroupedMatmulSwigluQuantWeightNzV2` 源码 | 无 torch_extension binding。vllm-ascend 版 aclnn 多 `swigluLimit` 参数。3 阶段分解方案见 3.1 节 |

---

## 4. 修改的文件列表

| 文件 | 模块 | 改动内容 |
|------|------|---------|
| `vllm_ascend/ops/__init__.py` | 基础 | 添加 `import custom_ops`（注册 `torch.ops.custom.*`） |
| `vllm_ascend/attention/dsa_v1.py` | Attention | inplace_partial_rotary_mul→cann_ops_transformer（延迟导入）；compressor→`DeviceOperator.dsa_compressor`（A3/Base 回退 `_C_ascend`，A5 走 custom）；rms_norm_dynamic_quant/quant_lightning_indexer+metadata→custom；compressor_metadata 保留 `_C_ascend`（graph capture 不兼容） |
| `vllm_ascend/attention/context_parallel/dsa_cp.py` | Attention | 同上 |
| `vllm_ascend/models/deepseek_v4.py` | MHC | hc_pre/hc_post→custom |
| `vllm_ascend/device/device_op.py` | Attention+MoE | 新增 `dsa_compressor` 设备分流（A3/Base `_C_ascend.compressor`，A5 `custom.compressor`）；A5 `npu_kv_quant_sparse_attn_sharedkv(+metadata)`→custom；kv_compress_epilog/indexer_quant_cache→cann_ops_transformer；sparse_attn_sharedkv/scatter_nd_update_asc→custom（含 reshape 适配）；moe_gating_top_k/moe_init_routing/swiglu_group_quant→custom（GMM 模块已回退，保留 `_C_ascend`） |
| `vllm_ascend/ops/fused_moe/experts_selector.py` | MoE-Gating | moe_gating_top_k_hash→custom |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | MoE-Quant | swiglu_group_quant→custom；dequant_swiglu_quant→custom |
| `vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py` | MoE-Quant | 同上 |
| `vllm_ascend/ops/fused_moe/moe_mlp.py` | MoE-GMM | dequant_swiglu_quant→custom（GMM 3 阶段分解已回退） |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py` | MoE-MC2 | dispatch_ffn_combine/dispatch_gmm_combine_decode→基类非融合路径 |
| `vllm_ascend/ops/layernorm.py` | Norm | npu_add_rms_norm_bias→torch_npu.npu_add_rms_norm + bias add |
| `csrc/build_aclnn.sh` | 编译脚本 | A3 (`ascend910_93`) 编译列表保留/新增 `compressor`；A5 (`ascend950`) 编译列表不包含 `compressor`、`kv_compress_epilog`、`kv_quant_sparse_attn_sharedkv(+metadata)`、`indexer_compress_epilog_v2`，避免覆盖外部实现 |

---

## 5. A5 (arch35) 兼容性

所有迁移的算子均验证了 arch35 内核支持。关键路径：

| 路径 | 算子 | arch35 内核 |
|------|------|------------|
| A5 DSA sparse attention | `custom.npu_kv_quant_sparse_attn_sharedkv` | ✅ arch35 |
| A5 KV compress | `cann_ops_transformer.kv_compress_epilog` | ✅ ops-transformer arch35 |
| A5 indexer scatter | `cann_ops_transformer.indexer_quant_cache` | ✅ ops-transformer arch35 |
| A5 MoE gating | `npu_moe_gating_top_k` | ✅ torch_npu 原生 |
| A5 MoE GMM | `torch_npu.npu_grouped_matmul_swiglu_quant_v2` | ✅ torch_npu 原生（未改） |
| A5 MoE MC2 | `npu_moe_distribute_dispatch_v2` + `npu_moe_distribute_combine_v2` | ✅ torch_npu 原生 |
| A5 RMSNorm | `npu_add_rms_norm` | ✅ torch_npu 原生 |
| A5 compressor | `custom.compressor` (wrapper) | ✅ cann-recipes-infer .run |
| A3/Base compressor | `_C_ascend.compressor`（回退） | ✅ vllm-ascend csrc |
| 共享 rotary | `cann_ops_transformer.inplace_partial_rotary_mul` | ✅ arch35 only |
| 共享 hc_pre/post | `custom.npu_hc_pre`/`npu_hc_post` | ✅ `__DAV_C310__` 显式 |
| 共享 compressor_metadata | `_C_ascend.compressor_metadata`（保留） | ✅ vllm-ascend csrc |

---

## 6. 汇总

### 6.1 算子迁移统计

| 状态 | 数量 | 算子 |
|------|:---:|------|
| **已迁移到 ops-transformer** | 6 | inplace_partial_rotary_mul, kv_compress_epilog, indexer_quant_cache, mhc_pre_sinkhorn, mhc_post |
| **已迁移到 ops-nn** | 0 | （暂回退，版本不配套，后续处理） |
| **已迁移到 torch_npu 原生** | 5 | npu_add_rms_norm, scatter_nd_update_, moe_init_routing_v2, moe_gating_top_k(非hash), dequant_swiglu_quant |
| **保留 cann-recipes** | 4 | moe_gating_top_k_hash（hash 路径）, rms_norm_dynamic_quant, swiglu_group_quant, partial_rotary_mul_quant |
| **已迁移到基类非融合路径** | 2 | dispatch_ffn_combine, dispatch_gmm_combine_decode |
| **暂未迁移（graph capture 不兼容）** | 1 | compressor_metadata |
| **暂未迁移（有可行方案）** | 3 | grouped_matmul_swiglu_quant_weight_nz, _tensor_list, _v2 |
| **无法迁移（SFA 路径，V4 不走）** | 3 | store_kv_block, store_kv_block_pre, batch_matmul_transpose |
| **合计** | 25 | |

### 6.2 命名空间分布

| 命名空间 | 算子数 |
|---------|:---:|
| `torch.ops.cann_ops_transformer.*` (ops-transformer) | 5 |
| `torch.ops.cann_ops_nn.*` (ops-nn) | 0（暂回退） |
| `torch.ops.custom.*` (cann-recipes-infer) | 4 + A5 compressor |
| `torch_npu.npu_*` (torch_npu 原生) | 5 |
| 基类非融合路径 | 2 |
| `torch.ops._C_ascend.*` (vllm-ascend csrc，暂未迁移/设备回退) | 4 + A3/Base compressor |
| `torch.ops._C_ascend.*` (vllm-ascend csrc，SFA 路径 V4 不走) | 6 |

### 6.3 V4 路径上 `_C_ascend` 残留

V4 DSA + MoE 调用链上，A5 路径有 **4 个 `_C_ascend` 算子**未迁移（compressor_metadata + GMM 模块 3 个）。A3/Base 路径额外回退 `compressor` 到 `_C_ascend.compressor`，因此 A3/Base 路径当前有 **5 个 `_C_ascend` 算子**残留。

### 6.4 关键技术决策

| 决策 | 原因 |
|------|------|
| 同名算子需同步调整编译列表 | `.run`/SO 中同名 aclnn 算子会优先命中 vllm-ascend 实现；若目标是 recipes/transformer 实现，必须从 vllm-ascend 编译列表移除同名算子，避免覆盖 |
| compressor_metadata 保留 `_C_ascend` | Python 预计算与 `UNIFORM_BATCH` graph capture/replay 不兼容：预计算在 build 时执行，结果冻结为常量；graph replay 时 `start_pos` 已更新但预计算结果未更新，导致首 token 正确后续 token 精度错误。必须作为 graph node 在 forward 路径执行 |
| quant_lightning_indexer 保持 V1 不切 V2 | V2 要求 `layout_q == layout_k`，V4 用不同 layout |
| compressor 按设备分流 | ops-transformer 版 14 参不含 RMSNorm+RoPE 融合；A5 继续使用 cann-recipes `custom.compressor`；A3/Base 使用 cann-recipes wrapper 会触发 `state_cache` 连续性/dtype 约束，先回退 `_C_ascend.compressor` |
| hc_pre/post 保持 cann-recipes 不切 ops-transformer | 不同 aclnn API，语义可能有差异 |
| GMM 模块暂未迁移 | A3 上 `npu_grouped_matmul` A8W8 dtype 不兼容，有可行方案待实现 |
| scatter_nd_update_asc 需 reshape | cann-recipes 要求 2D 输入，vllm-ascend 传 4D |
| NPU 格式兼容（NCHW vs ND） | pad 用 `index_select` + `fill_` 替代 `torch.cat`，避免格式不兼容 |

---

## 7. 算子编译

> 仅编译 vllm-ascend V4 路径实际用到的算子，减少编译时间。编译命令参考 ops-transformer `docs/zh/install/compile.md` 和 ops-nn `build.sh --help`。

### 7.1 ops-transformer 编译（.run 包 + whl 包）

**用到的算子（5 个，A3/A5 统一）：**

```
inplace_partial_rotary_mul,kv_compress_epilog,indexer_quant_cache,mhc_pre_sinkhorn,mhc_post
```

**A3 (ascend910_93)：**

```bash
cd ops-transformer

# 编译 .run 包（仅上述 5 个算子）
bash build.sh --pkg --soc=ascend910_93 \
  --ops=inplace_partial_rotary_mul,kv_compress_epilog,indexer_quant_cache,mhc_pre_sinkhorn,mhc_post

# 安装 .run 包
./build_out/cann-ops-transformer-custom_linux-*.run

# 编译并安装 whl 包（torch_extension，JIT 编译 C++ binding）
cd torch_extension
python3 -m build --wheel -n
pip3 install dist/*.whl --force-reinstall --no-deps
```

**A5 (ascend950)：**

```bash
cd ops-transformer

# 编译 .run 包（仅上述 5 个算子）
bash build.sh --pkg --soc=ascend950 \
  --ops=inplace_partial_rotary_mul,kv_compress_epilog,indexer_quant_cache,mhc_pre_sinkhorn,mhc_post

# 安装 .run 包
./build_out/cann-ops-transformer-custom_linux-*.run

# 编译并安装 whl 包
cd torch_extension
python3 -m build --wheel -n
pip3 install dist/*.whl --force-reinstall --no-deps
```

> **说明**：
> - `.run` 包提供 aclnn API（AscendC 内核），安装到 `$ASCEND_HOME_PATH/opp/vendors/`
> - whl 包提供 torch Python binding（JIT 编译），通过 `import cann_ops_transformer` 注册 `torch.ops.cann_ops_transformer.*` 命名空间
> - `--ops` 参数指定算子目录名，多个用英文逗号分隔，不指定时默认编译全部算子
> - A3 和 A5 的算子列表相同（第三组 HARD 完成后 A3 会增加 compressor/quant_lightning_indexer/sparse_flash_mla）

### 7.2 ops-nn 编译

> **暂不需要**。`swiglu_group_quant` 和 `rms_norm_dynamic_quant` 已回退到 cann-recipes，ops-nn 版本不配套，后续处理。

### 7.3 cann-recipes-infer 编译（hash 路径 moe_gating_top_k + rms_norm_dynamic_quant + swiglu_group_quant）

V4 hash 路径仍使用 `custom.npu_moe_gating_top_k`（ops-transformer 不支持 `input_ids`/`tid2eid`），需编译 cann-recipes：

```bash
cd cann-recipes-infer/ops/ascendc

# 编译 .run 包
bash build.sh -c ascend910_93   # A3
# 或
bash build.sh -c ascend950      # A5

# 安装 .run 包
bash CANN-custom_ops-*.run --install

# 编译并安装 whl 包
cd torch_ops_extension
pip3 install .
```

### 7.4 vllm-ascend 编译（csrc 保留算子）

csrc 仍需编译保留的 `_C_ascend` 算子（`compressor_metadata`、GMM 模块 3 个等）：

```bash
COMPILE_CUSTOM_KERNELS=1 pip install -e .
```

`csrc/build_aclnn.sh` 已按 SOC 版本精简编译列表（A3 保留 22 个，A5 保留 6 个），仅编译无法迁移的算子。

### 7.5 编译顺序

1. ops-transformer .run + whl（提供 aclnn 内核 + Python binding）
2. ops-nn .run + whl
3. cann-recipes-infer .run + whl（hash 路径）
4. vllm-ascend csrc（保留算子）

---

## 8. 架构映射

| 标识 | 硬件 | 宏 |
|------|------|-----|
| arch22 | Ascend 910B (A3) | `__CCE_AICORE__ == 220` |
| arch35 | Ascend 910C (A5) | `__CCE_AICORE__ == 310` / `__DAV_C310__` |
| ascend950 | Ascend 910C (A5) | cann-recipes A5 编译目标 |
