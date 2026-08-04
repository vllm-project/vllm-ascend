# DeepSeek-V4 算子迁移到 CANN 仓库指导

> **文档用途**：基线当前已验证通过的算子迁移改动，供其他人编译和使用。后续迁移工作仍在进行。
> **更新规则**：代码或编译方式有改动时，请同步更新本文档。

---

## 1. 背景与目标

### 1.1 任务

将 vllm-ascend 中 DeepSeek-V4 原生算子（`torch.ops._C_ascend.*`，由 `csrc/` 编译）切换到 CANN 主线算子仓库（ops-transformer、ops-nn），对齐 cann-recipes-infer 仓库中 DeepSeek-V4 的实现。cann-recipes 已基本完成迁移，vllm-ascend 参考其实现进行迁移。

### 1.2 仓库

| 仓库 | 说明 | Python 命名空间 |
|------|------|----------------|
| ops-transformer | CANN 主线算子仓库（非 experimental + experimental） | `torch.ops.cann_ops_transformer.*` |
| ops-nn | CANN 算子仓库（暂未使用） | `torch.ops.cann_ops_nn.*` |
| cann-recipes-infer | 模型推理 recipe 仓库（含 wrapper） | `torch.ops.custom.*` |
| vllm-ascend csrc | vllm-ascend 原生算子（保留必要部分） | `torch.ops._C_ascend.*` |

### 1.3 SO 加载优先级

```
vllm-ascend > experimental > ops-transformer(非experimental) > cann-recipes
```

同名 `OP_ADD` 按此优先级覆盖。因此迁移时必须同步调整 `csrc/build_aclnn.sh` 编译列表，移除已迁移算子，避免 vllm-ascend SO 覆盖外部实现。

### 1.4 约束

- 仅修改 vllm-ascend，不修改 CANN 仓库
- csrc 不清理（保留无法迁移的算子）
- 目标平台 A5 (Ascend 910C / arch35)，兼容 A3 (Ascend 910B / arch22)
- `csrc/build_aclnn.sh` A2 (ascend910b) 列表不动

---

## 2. 当前状态（已验证通过）

> 当前代码已在 A5 环境验证通过。这是**中间态**，部分算子仍使用 cann-recipes wrapper 或 vllm-ascend csrc，后续会继续迁移。

### 2.1 算子总表

| # | 算子 | 进度 | 原 torch API | 当前 torch API | 目标 torch API | kernel 位置 | 关键信息 |
|---|------|:---:|---|---|---|---|---|
| 1 | inplace_partial_rotary_mul | ✅ | `_C_ascend.*` | `cann_ops_transformer.*` | 同当前 | ot 非 exp, posembedding/, arch35 ✅ | 无参数变化 |
| 2 | compressor | ✅ | `_C_ascend.*` (18参融合) | `cann_ops_transformer.*` (12参) + Python `npu_rms_norm` + `inplace_partial_rotary_mul` | 同当前 | ot 非 exp, attention/, arch35 ✅ (op_api 来自 base 包) | 拆融合；需 `kv[..., :head_dim]` 截断；RoPE view 4D；indexer 用 `indexcom_head_dim` |
| 3 | kv_compress_epilog `[A5]` | ✅ | `_C_ascend.*` | `cann_ops_transformer.*` | 同当前 | ot 非 exp, attention/, arch35 ✅ | `quant_mode="fp8_bf16"`；cache 保持 4D |
| 4 | indexer_quant_cache `[A5]` | ✅ | `_C_ascend.indexer_compress_epilog_v2` | `cann_ops_transformer.*` | 同当前 | ot 非 exp, attention/, arch35 ✅ | `cache_scale` float32；cache 4D；参数名 `indexer_full_cache`→`indexer_scale_cache` |
| 5 | mhc_pre_sinkhorn | ✅ | `_C_ascend.npu_hc_pre_v2` | `cann_ops_transformer.*` | 同当前 | ot 非 exp, mhc/, arch35 ✅ | 参数改名+顺序交换；unsqueeze 3D→4D；返回 8 元组取前 3；comb `unflatten`+`squeeze` |
| 6 | mhc_post | ✅ | `_C_ascend.npu_hc_post` | `cann_ops_transformer.*` | 同当前 | ot 非 exp, mhc/, arch35 ✅ | 参数顺序重排；去掉 unsqueeze/squeeze |
| 7 | scatter_nd_update `[A3]` | ✅ | `_C_ascend.npu_scatter_nd_update_v2` | `torch_npu.npu_scatter_nd_update_` | 同当前 | torch_npu pip 包 | 纯改名；保留 `[:x_flat.shape[0]]` 截取 |
| 8 | moe_init_routing_v2 | ✅ | `_C_ascend.npu_moe_init_routing_custom` | `torch_npu.npu_moe_init_routing_v2` | 同当前 | torch_npu pip 包 | 纯改名 |
| 9 | moe_gating_top_k (非hash) | ✅ | `_C_ascend.moe_gating_top_k` | `torch_npu.npu_moe_gating_top_k` | 同当前 | torch_npu pip 包 | 去掉 `input_ids`/`tid2eid` |
| 10 | dequant_swiglu_quant | ✅ | `_C_ascend.npu_dequant_swiglu_quant` | `torch_npu.npu_dequant_swiglu_quant` | 同当前 | torch_npu pip 包 | 条件传参：无 limit `swiglu_mode=0`，有 limit 共享专家 `=2`，路由专家 `=1` |
| 11 | add_rms_norm | ✅ | `_C_ascend.npu_add_rms_norm_bias` | `torch_npu.npu_add_rms_norm` + `x.add_(bias)` | 同当前 | torch_npu pip 包 | bias 不再融合 |
| 12 | dispatch_ffn_combine `[A3]` | ✅ | `_C_ascend.*` | `super().fused_experts()` | 同当前 | 无独立内核 | 基类非融合路径 |
| 13 | dispatch_gmm_combine_decode `[A3]` | ✅ | `_C_ascend.*` | `super().fused_experts()` | 同当前 | 无独立内核 | 同上 |
| 14 | quant_lightning_indexer | 📋 | `_C_ascend.npu_vllm_quant_lightning_indexer` | `custom.npu_quant_lightning_indexer` | `cann_ops_transformer.quant_lightning_indexer` | ot exp, attention/quant_lightning_indexer/, arch35 ✅ | `layout_k` 改 `PA_BBND`；参数改名；`quant_mode` 按 arch 锁定 |
| 15 | quant_lightning_indexer_metadata | 📋 | `_C_ascend.npu_vllm_quant_lightning_indexer_metadata` | `custom.npu_quant_lightning_indexer_metadata` | `cann_ops_transformer.quant_lightning_indexer_metadata` | ot exp, AICPU | 新增 `seqused_k`/`cmp_residual_k`；删除 `device`/`pre_tokens`/`next_tokens` |
| 16 | sparse_attn_sharedkv `[A3]` | ✅ | `_C_ascend.npu_sparse_attn_sharedkv` | `cann_ops_transformer.sparse_flash_mla` | 同当前 | ot 非 exp, attention/sparse_flash_mla/, arch35 ✅ | 算子名变更；`seqused_kv`→`seqused_ori_kv`；`PA_ND`→`PA_BBND`；新增 `seqused_cmp_kv`/`cmp_residual_kv`/`topk_value_mode` |
| 17 | sparse_attn_sharedkv_metadata `[A3]` | ✅ | `_C_ascend.npu_sparse_attn_sharedkv_metadata` | `cann_ops_transformer.sparse_flash_mla_metadata` | 同当前 | ot 非 exp, AICPU | 同上；新增 `max_seqlen_cmp_kv`；删除 `device` |
| 18 | kv_quant_sparse_attn_sharedkv `[A5]` | ✅ | `_C_ascend.npu_kv_quant_sparse_attn_sharedkv` | `cann_ops_transformer.mixed_quant_sparse_flash_mla` | 同当前 | ot 非 exp, attention/mixed_quant_sparse_flash_mla/, arch35 ✅ | A5 路径；`kv_quant_mode`→`quant_mode`；删除 `tile_size` |
| 19 | kv_quant_sparse_attn_sharedkv_metadata `[A5]` | ✅ | `_C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata` | `cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata` | 同当前 | ot 非 exp, AICPU | 同上 |
| 20 | rms_norm_dynamic_quant | ⏳ | `_C_ascend.*` | `custom.npu_rms_norm_dynamic_quant` | A5: `torch_npu.npu_dynamic_quant`；A3: `cann_ops_nn.rms_norm_dynamic_quant` | cann-recipes, src/, 无 arch35 | DSV4 A5 用 `npu_dynamic_quant`，A3 用 `cann_ops_nn.*`；需平台分支；ops-nn 未打包 |
| 21 | swiglu_group_quant `[A5]` | ⏳ | `_C_ascend.*` | `custom.npu_swiglu_group_quant` | `cann_ops_nn.swiglu_group_quant` | cann-recipes, src/, 无 arch35 | DSV4 所有平台都用 `cann_ops_nn.*`；ops-nn 未打包 |
| 22 | moe_gating_top_k_hash | ⏳ | `_C_ascend.moe_gating_top_k_hash` | `custom.npu_moe_gating_top_k` | PyTorch native fallback | cann-recipes, src/, arch35 ✅ | DSV4 hash 路径用纯 PyTorch 实现；需改 vllm hash 路径为 native fallback |
| 23 | compressor_metadata | ❌ | `_C_ascend.*` | 同原 | 同原 | vllm csrc, attention/, 无 arch35 | graph capture 不兼容，无外部对应 |
| 24 | grouped_matmul_swiglu_quant_weight_nz | ❌ | `_C_ascend.*` | 同原 | 同原 | vllm csrc, gmm/, 无 arch35 | 无 torch binding；可行方案：3 阶段分解 |
| 25 | grouped_matmul_swiglu_quant_weight_nz_tensor_list | ❌ | `_C_ascend.*` | 同原 | 同原 | vllm csrc, gmm/, 无 arch35 | vllm 独有 |
| 26 | grouped_matmul_swiglu_quant_v2 | ❌ | `_C_ascend.*` | 同原 | 同原 | vllm csrc, gmm/, arch35 ✅ | vllm 版多 `swigluLimit` 参数 |

### 2.2 迁移统计

| 进度 | 数量 | 说明 |
|------|:---:|------|
| ✅ 完成 | 17 | 已对齐 DSV4，当前=目标 |
| 📋 待执行 | 2 | 当前用 cann-recipes wrapper，目标切到 ops-transformer |
| ⏳ 暂缓 | 3 | 依赖 ops-nn 打包或需改 hash 路径为 native fallback |
| ❌ 阻塞 | 4 | graph capture 不兼容或无外部对应 |
| **合计** | 26 | |

### 2.3 修改的文件

| 文件 | 改动内容 |
|------|---------|
| `vllm_ascend/ops/__init__.py` | 添加 `import custom_ops` |
| `vllm_ascend/attention/dsa_v1.py` | inplace_partial_rotary_mul/compressor→cann_ops_transformer；rms_norm_dynamic_quant/quant_lightning_indexer→custom；compressor_metadata 保留 `_C_ascend` |
| `vllm_ascend/attention/context_parallel/dsa_cp.py` | 同上 |
| `vllm_ascend/models/deepseek_v4.py` | mhc_pre_sinkhorn/mhc_post→cann_ops_transformer |
| `vllm_ascend/device/device_op.py` | compressor→cann_ops_transformer；kv_compress_epilog/indexer_quant_cache→cann_ops_transformer；sparse_flash_mla/mixed_quant_sparse_flash_mla→cann_ops_transformer；scatter_nd_update→torch_npu；moe_gating_top_k/moe_init_routing→torch_npu；swiglu_group_quant→custom |
| `vllm_ascend/ops/fused_moe/experts_selector.py` | moe_gating_top_k_hash→custom |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | swiglu_group_quant→custom；dequant_swiglu_quant→torch_npu |
| `vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py` | 同上 |
| `vllm_ascend/ops/fused_moe/moe_mlp.py` | dequant_swiglu_quant→torch_npu |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py` | dispatch_ffn_combine/dispatch_gmm_combine_decode→基类非融合路径 |
| `vllm_ascend/ops/layernorm.py` | npu_add_rms_norm_bias→torch_npu.npu_add_rms_norm + bias add |
| `csrc/build_aclnn.sh` | A3 编译列表移除已迁移算子；A5 编译列表移除已迁移算子 |

---

## 3. 算子编译指导

> 以下为当前已验证通过的环境的算子打包方式。SO 加载顺序：vllm-ascend > experimental > ops-transformer(非exp) > cann-recipes。

### 3.1 ops-transformer（非 experimental）

**已打包算子（8 个）**：
```
inplace_partial_rotary_mul,kv_compress_epilog,indexer_quant_cache,mhc_pre_sinkhorn,mhc_post,compressor,sparse_flash_mla,mixed_quant_sparse_flash_mla
```

```bash
cd ops-transformer

# A3
bash build.sh --pkg --soc=ascend910_93 \
  --ops=inplace_partial_rotary_mul,kv_compress_epilog,indexer_quant_cache,mhc_pre_sinkhorn,mhc_post,compressor,sparse_flash_mla,mixed_quant_sparse_flash_mla

# A5
bash build.sh --pkg --soc=ascend950 \
  --ops=inplace_partial_rotary_mul,kv_compress_epilog,indexer_quant_cache,mhc_pre_sinkhorn,mhc_post,compressor,sparse_flash_mla,mixed_quant_sparse_flash_mla

# 安装
./build_out/cann-ops-transformer-custom_linux-*.run
```

### 3.2 ops-transformer（experimental）

**已打包算子（2 个）**：
```
quant_lightning_indexer,quant_lightning_indexer_metadata
```

```bash
cd ops-transformer

# A3
bash build.sh --pkg --experimental --vendor_name=experimental --soc=ascend910_93 \
  --ops=quant_lightning_indexer,quant_lightning_indexer_metadata

# A5
bash build.sh --pkg --experimental --vendor_name=experimental --soc=ascend950 \
  --ops=quant_lightning_indexer,quant_lightning_indexer_metadata

# 安装
./build_out/cann-ops-transformer-experimental_linux-*.run
```

> ⚠️ experimental 编译列表**不能**包含 compressor/mhc_post 等与非 experimental 同名 OP_ADD 且参数不兼容的算子。

### 3.3 cann-recipes-infer

**全量打包**（包含所有 cann-recipes 有源码的算子）。

```bash
cd cann-recipes-infer/ops/ascendc

# A3
bash build.sh -c ascend910_93
# A5
bash build.sh -c ascend950

# 安装 .run 包
bash CANN-custom_ops-*.run --install

# 编译并安装 whl 包
cd torch_ops_extension
bash build_and_install.sh
```

### 3.4 ops-nn

> **未打包**。`swiglu_group_quant` 和 `rms_norm_dynamic_quant` 仍用 cann-recipes，ops-nn 版本不配套，后续处理。

### 3.5 ops-transformer whl 包

whl 包是全量编译（包含所有 torch_extension 中注册的算子），不按 `--ops` 选择：

```bash
cd ops-transformer/torch_extension
python3 -m pip install -r requirements.txt
python3 -m build --wheel -n
python3 -m pip install dist/*.whl --force-reinstall --no-deps
```

### 3.6 vllm-ascend（csrc 保留算子）

```bash
COMPILE_CUSTOM_KERNELS=1 pip install -e .
```

`csrc/build_aclnn.sh` 已按 SOC 版本精简编译列表：
- **A2 (ascend910b)**：不动，保留全部原始算子
- **A3 (ascend910_93)**：移除已迁移算子，保留 20 个
- **A5 (ascend950)**：移除已迁移算子，保留 6 个（compressor_metadata、load_index_kv_cache、causal_conv1d、recurrent_gated_delta_rule、chunk_fwd_o、chunk_gated_delta_rule_fwd_h）

### 3.7 编译顺序

1. ops-transformer 非 experimental（.run）
2. ops-transformer experimental（.run，--vendor_name=experimental）
3. cann-recipes-infer（.run + whl）
4. ops-transformer whl 包
5. vllm-ascend csrc

---

## 4. 后续迁移计划

### 4.1 待执行（📋，2 个）

| # | 算子 | 当前 | 目标 | 方案 |
|---|------|------|------|------|
| 14-15 | quant_lightning_indexer + metadata | `custom.*` (wrapper, ot exp kernel) | `cann_ops_transformer.*` | 切换 Python API；`layout_k` 改 `PA_BBND`；参数改名 |

### 4.2 暂缓（⏳，3 个）

| # | 算子 | 遗留问题 |
|---|------|---------|
| 20 | rms_norm_dynamic_quant | DSV4 A5 用 `torch_npu.npu_dynamic_quant`，A3 用 `cann_ops_nn.*`。需平台分支 + ops-nn 打包 |
| 21 | swiglu_group_quant | DSV4 用 `cann_ops_nn.*`。需 ops-nn 打包 |
| 22 | moe_gating_top_k_hash | DSV4 hash 路径用纯 PyTorch 实现。需改 vllm hash 路径为 native fallback |

### 4.3 阻塞（❌，4 个）

| # | 算子 | 原因 |
|---|------|------|
| 23 | compressor_metadata | graph capture 不兼容，无外部对应 |
| 24-26 | GMM 模块 3 个 | 无 torch binding 或 vllm 独有；可行方案：3 阶段分解 |

---

## 5. 关键技术决策

| 决策 | 原因 |
|------|------|
| 同名 OP_ADD 需同步调整 `build_aclnn.sh` | SO 加载优先级：vllm-ascend > experimental > ot(非exp) > cann-recipes |
| compressor_metadata 保留 `_C_ascend` | Python 预计算与 graph capture/replay 不兼容，必须作为 graph node 在 forward 路径执行 |
| compressor 拆融合 norm/rope | ot 版 12 参不含 RMSNorm+RoPE，Python 层用 `npu_rms_norm` + `inplace_partial_rotary_mul` 补偿 |
| experimental `.run` 包需 `--vendor_name=experimental` | 安装到独立目录，避免与非 experimental 同名 OP_ADD 冲突 |
| experimental 编译列表需精确指定 | 不能包含 compressor/mhc_post 等与非 experimental 同名且参数不兼容的算子 |
| dequant_swiglu_quant 条件传参 | 无 `swiglu_limit` 时用默认 `swiglu_mode=0`（不钳位），避免输出错误 |

---

## 6. OP_DEF 冲突审计表

> SO 加载优先级：vllm-ascend > experimental > ops-transformer(非exp) > cann-recipes

| OP_ADD 类名 | vllm | exp | ot | rc | 冲突 | 生效 | 说明 |
|---|:---:|:---:|:---:|:---:|:---:|---|---|
| Compressor | ✅ | ✅18参 | ✅12参 | ❌ | ⚠️ | A5: ot / A2A3: vllm | A5 已移除；A2/A3 需移除 |
| InplacePartialRotaryMul | ✅ | ❌ | ✅ | ✅ | ⚠️ | A5: ot / A2A3: vllm | A5 已移除；A2/A3 需移除 |
| MhcPost | ❌ | ✅ | ✅ | ❌ | ⚠️ | exp / ot | exp 编译列表不含 mhc_post |
| KvCompressEpilog | ✅ | ❌ | ✅ | ✅ | ⚠️ | A5: ot / A2A3: vllm | A5 已移除；A2/A3 需移除 |
| QuantLightningIndexer | ❌ | ✅(已打包) | ✅(有源码未打包) | ❌ | ⚠️ | exp | exp 优先；非 exp 有完整源码但未打包 |
| QuantLightningIndexerMetadata | ❌ | ✅(已打包) | ❌ | ❌ | 无 | exp | 仅 exp 有 |
| RmsNormDynamicQuant | ✅ | ❌ | ❌ | ✅ | ⚠️ | A5: rc / A2A3: vllm | A5 已移除 |
| SparseFlashMla | ❌ | ❌ | ✅(已打包) | ❌ | 无 | ot | 新增非 exp 算子，无同名冲突 |
| MixedQuantSparseFlashMla | ❌ | ❌ | ✅(已打包) | ❌ | 无 | ot | A5 only (arch35)；A3 无 kernel 但 op_def 注册无害 |
| SparseAttnSharedkv | ✅ | ✅(未打包) | ❌ | ❌ | ⚠️ | A2A3: vllm | A5 已移除；exp 已移除；A2/A3 需移除 |
| KvQuantSparseAttnSharedkv | ✅ | ✅(未打包) | ❌ | ❌ | ⚠️ | A2A3: vllm | A5 已移除；exp 已移除；A2/A3 需移除 |
| MoeGatingTopKHash | ✅ | ❌ | ❌ | ✅ | ⚠️ | A5: rc / A2A3: vllm | A5 已移除 |
| MoeGatingTopK | ✅ | ❌ | ✅ | ✅ | ⚠️ | A5: ot / A2A3: vllm | A5 已移除；A2/A3 需移除 |
| SwigluGroupQuant | ✅ | ❌ | ❌ | ✅ | ⚠️ | A5: rc / A2A3: vllm | A5 已移除 |
| ScatterNdUpdate | ✅ | ❌ | ❌ | ✅ | ⚠️ | A5: rc / A2A3: vllm | A5 已移除；A2/A3 需移除 |
| CompressorMetadata | ✅ | ❌ | ❌ | ❌ | 无 | vllm | 保留 |
| IndexerQuantCache | ❌ | ❌ | ✅ | ❌ | 无 | ot | |
| MhcPreSinkhorn | ❌ | ❌ | ✅ | ❌ | 无 | ot | |
| DequantSwigluQuant | ✅ | ❌ | ❌ | ❌ | 无 | vllm | torch_npu 原生，不依赖 OP_ADD |
| AddRmsNormBias | ✅ | ❌ | ❌ | ❌ | 无 | vllm | 保留 |
| MoeInitRoutingCustom | ✅ | ❌ | ❌ | ❌ | 无 | vllm | 保留（A2/A3） |
| MoeInitRoutingV2 | ❌ | ❌ | ✅ | ❌ | 无 | ot | torch_npu 原生调用 |
| GroupedMatmulSwigluQuantWeightNz | ✅ | ❌ | ❌ | ❌ | 无 | vllm | 保留 |
| GroupedMatmulSwigluQuantWeightNzTensorList | ✅ | ❌ | ❌ | ❌ | 无 | vllm | 保留 |

**⚠️ 后续改动检查清单**：新增/回退算子时 → 检查上表冲突 → 确保 `build_aclnn.sh` 移除冲突算子 → 确保 experimental 编译列表不含同名且参数不兼容的算子。

---

## 7. 架构映射

| 标识 | 硬件 | 宏 |
|------|------|-----|
| arch22 | Ascend 910B (A3) | `__CCE_AICORE__ == 220` |
| arch35 | Ascend 910C (A5) | `__CCE_AICORE__ == 310` / `__DAV_C310__` |
| ascend950 | Ascend 910C (A5) | cann-recipes A5 编译目标 |
