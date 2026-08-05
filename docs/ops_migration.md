# DeepSeek-V4 算子迁移到 CANN 仓库指导

> **文档用途**：DSV4 算子迁移改动记录。A5 已验证通过，A3 路径待验证。
> **更新规则**：代码或编译方式有改动时，请同步更新本文档。

---

## 1. 背景与目标

### 1.1 任务

将 vllm-ascend 中 DeepSeek-V4 原生算子（`torch.ops._C_ascend.*`，由 `csrc/` 编译）切换到 CANN 主线算子仓库（ops-transformer、ops-nn），对齐 cann-recipes-infer 仓库中 DeepSeek-V4 的实现。cann-recipes 已基本完成迁移，vllm-ascend 参考其实现进行迁移。

### 1.2 仓库

| 仓库 | 说明 | Python 命名空间 |
|------|------|----------------|
| ops-transformer | CANN 主线算子仓库（非 experimental + experimental） | `torch.ops.cann_ops_transformer.*` |
| ops-nn | CANN 算子仓库（norm/activation） | `torch.ops.cann_ops_nn.*` |
| vllm-ascend csrc | vllm-ascend 原生算子（保留必要部分） | `torch.ops._C_ascend.*` |

### 1.3 SO 加载优先级

```
vllm-ascend > ops-transformer > ops-nn
```

同名 `OP_ADD` 按此优先级覆盖。因此迁移时必须同步调整 `csrc/build_aclnn.sh` 编译列表，移除已迁移算子，避免 vllm-ascend SO 覆盖外部实现。

### 1.4 约束

- 仅修改 vllm-ascend，不修改 CANN 仓库
- csrc 不清理（保留无法迁移的算子）
- 目标平台 A5 (Ascend 910C / arch35)，兼容 A3 (Ascend 910B / arch22)
- `csrc/build_aclnn.sh` A2 (ascend910b) 列表不动

---

## 2. 当前状态

> **A5 已验证通过**，A3 路径待验证。A3-only 算子标记为 🟡（代码已改，尚未在 A3 环境验证）。

### 2.1 算子总表

| # | 算子 | 进度 | 原 torch API | 当前 torch API | 目标 torch API | kernel 位置 | 关键信息 |
|---|------|:---:|---|---|---|---|---|
| 1 | inplace_partial_rotary_mul | ✅ | `_C_ascend.*` | `cann_ops_transformer.*` | 同当前 | ot 非 exp, posembedding/, arch35 ✅ | 无参数变化 |
| 2 | compressor | ✅ | `_C_ascend.*` (18参融合) | `cann_ops_transformer.*` (12参) + Python `npu_rms_norm` + `inplace_partial_rotary_mul` | 同当前 | ot 非 exp, attention/, arch35 ✅ (op_api 来自 base 包) | 拆融合；需 `kv[..., :head_dim]` 截断；RoPE view 4D；indexer 用 `indexcom_head_dim` |
| 3 | kv_compress_epilog `[A5]` | ✅ | `_C_ascend.*` | `cann_ops_transformer.*` | 同当前 | ot 非 exp, attention/, arch35 ✅ | `quant_mode="fp8_bf16"`；cache 保持 4D |
| 4 | indexer_quant_cache `[A5]` | ✅ | `_C_ascend.indexer_compress_epilog_v2` | `cann_ops_transformer.*` | 同当前 | ot 非 exp, attention/, arch35 ✅ | `cache_scale` float32；cache 4D；参数名 `indexer_full_cache`→`indexer_scale_cache` |
| 5 | mhc_pre_sinkhorn | ✅ | `_C_ascend.npu_hc_pre_v2` | `cann_ops_transformer.*` | 同当前 | ot 非 exp, mhc/, arch35 ✅ | 参数改名+顺序交换；unsqueeze 3D→4D；返回 8 元组取前 3；comb `unflatten`+`squeeze` |
| 6 | mhc_post | ✅ | `_C_ascend.npu_hc_post` | `cann_ops_transformer.*` | 同当前 | ot 非 exp, mhc/, arch35 ✅ | 参数顺序重排；去掉 unsqueeze/squeeze |
| 7 | scatter_nd_update `[A3]` | 🟡 | `_C_ascend.npu_scatter_nd_update_v2` | `torch_npu.npu_scatter_nd_update_` | 同当前 | torch_npu pip 包 | 纯改名；保留 `[:x_flat.shape[0]]` 截取；A3 待验证 |
| 8 | moe_init_routing_v2 | ✅ | `_C_ascend.npu_moe_init_routing_custom` | `torch_npu.npu_moe_init_routing_v2` | 同当前 | torch_npu pip 包 | 纯改名 |
| 9 | moe_gating_top_k (非hash) | ✅ | `_C_ascend.moe_gating_top_k` | `torch_npu.npu_moe_gating_top_k` | 同当前 | torch_npu pip 包 | 去掉 `input_ids`/`tid2eid` |
| 10 | dequant_swiglu_quant | ✅ | `_C_ascend.npu_dequant_swiglu_quant` | `torch_npu.npu_dequant_swiglu_quant` | 同当前 | torch_npu pip 包 | 条件传参：无 limit `swiglu_mode=0`，有 limit 共享专家 `=2`，路由专家 `=1` |
| 11 | add_rms_norm | ✅ | `_C_ascend.npu_add_rms_norm_bias` | `torch_npu.npu_add_rms_norm` + `x.add_(bias)` | 同当前 | torch_npu pip 包 | bias 不再融合 |
| 12 | dispatch_ffn_combine `[A3]` | 🟡 | `_C_ascend.*` | `super().fused_experts()` | 同当前 | 无独立内核 | 基类非融合路径；A3 待验证 |
| 13 | dispatch_gmm_combine_decode `[A3]` | 🟡 | `_C_ascend.*` | `super().fused_experts()` | 同当前 | 无独立内核 | 同上；A3 待验证 |
| 14 | quant_lightning_indexer | ✅ | `_C_ascend.npu_vllm_quant_lightning_indexer` | `cann_ops_transformer.quant_lightning_indexer` | 同当前 | ot 非 exp, attention/quant_lightning_indexer_v2/, arch35 ✅ | whl C++ 调 V2 aclnn；`.run` 编译 `quant_lightning_indexer_v2`；`layout_k` 改 `PA_BBND`；`quant_mode` 按 arch 锁定（A5=1 FP8, A3=2 INT8）；`seqused_q` 用 per-batch query token 数 |
| 15 | quant_lightning_indexer_metadata | ✅ | `_C_ascend.npu_vllm_quant_lightning_indexer_metadata` | `cann_ops_transformer.quant_lightning_indexer_metadata` | 同当前 | ot 非 exp, attention/quant_lightning_indexer_v2_metadata/, AICPU | 同上；新增 `seqused_k`/`cmp_residual_k`；删除 `device`/`pre_tokens`/`next_tokens`/`max_seqlen_*` |
| 16 | sparse_attn_sharedkv `[A3]` | 🟡 | `_C_ascend.npu_sparse_attn_sharedkv` | `cann_ops_transformer.sparse_flash_mla` | 同当前 | ot 非 exp, attention/sparse_flash_mla/, arch35 ✅ | 算子名变更；`seqused_kv`→`seqused_ori_kv`；`PA_ND`→`PA_BBND`；A3 待验证 |
| 17 | sparse_attn_sharedkv_metadata `[A3]` | 🟡 | `_C_ascend.npu_sparse_attn_sharedkv_metadata` | `cann_ops_transformer.sparse_flash_mla_metadata` | 同当前 | ot 非 exp, AICPU | 同上；A3 待验证 |
| 18 | kv_quant_sparse_attn_sharedkv `[A5]` | ✅ | `_C_ascend.npu_kv_quant_sparse_attn_sharedkv` | `cann_ops_transformer.mixed_quant_sparse_flash_mla` | 同当前 | ot 非 exp, attention/mixed_quant_sparse_flash_mla/, arch35 ✅ | A5 路径；`kv_quant_mode`→`quant_mode`；删除 `tile_size` |
| 19 | kv_quant_sparse_attn_sharedkv_metadata `[A5]` | ✅ | `_C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata` | `cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata` | 同当前 | ot 非 exp, AICPU | 同上 |
| 20 | rms_norm_dynamic_quant | 🟡 | `_C_ascend.*` | `DeviceOperator.rms_norm_dynamic_quant` | 同当前 | ops-nn, norm/, arch22 ✅ (A3)；A5 无 arch35 | A3: `cann_ops_nn.rms_norm_dynamic_quant`（fused）；A5: `npu_rms_norm`+`npu_dynamic_quant`（拆分）；不传 `smooth_scales`；A3 待验证 |
| 21 | swiglu_group_quant `[A5]` | ✅ | `_C_ascend.*` | `cann_ops_nn.swiglu_group_quant` | 同当前 | ops-nn, activation/, arch35 ✅ | `dst_type=24`（ScalarType）；`quant_mode=1`；`clamp_limit` None→-1.0；A5-only |
| 22 | moe_gating_top_k_hash | ✅ | `_C_ascend.moe_gating_top_k_hash` | `torch_npu.npu_moe_gating_top_k` | 同当前 | ot 非 exp, moe/moe_gating_top_k/, arch35 ✅ | hash 路径用 `torch_npu.npu_moe_gating_top_k`（13 参数，调 `aclnnMoeGatingTopKV2`）；`k_group=1, group_count=1`；需升级 torch_npu 到 13 参数版本 + `exposed: all_version` |
| 23 | compressor_metadata | ❌ | `_C_ascend.*` | 同原 | 同原 | vllm csrc, attention/, 无 arch35 | graph capture 不兼容，无外部对应 |
| 24 | grouped_matmul_swiglu_quant_weight_nz | ❌ | `_C_ascend.*` | 同原 | 同原 | vllm csrc, gmm/, 无 arch35 | 无 torch binding；可行方案：3 阶段分解 |
| 25 | grouped_matmul_swiglu_quant_weight_nz_tensor_list | ❌ | `_C_ascend.*` | 同原 | 同原 | vllm csrc, gmm/, 无 arch35 | vllm 独有 |
| 26 | grouped_matmul_swiglu_quant_v2 | ❌ | `_C_ascend.*` | 同原 | 同原 | vllm csrc, gmm/, arch35 ✅ | vllm 版多 `swigluLimit` 参数 |

### 2.2 迁移统计

| 进度 | 数量 | 说明 |
|------|:---:|------|
| ✅ A5 已验证 | 16 | A5 环境验证通过 |
| 🟡 代码已改，A3 待验证 | 6 | A3-only 或含 A3 路径，尚未在 A3 验证 |
| ❌ 阻塞 | 4 | graph capture 不兼容或无外部对应 |
| **合计** | 26 | |

### 2.3 修改的文件

| 文件 | 改动内容 |
|------|---------|
| `vllm_ascend/ops/__init__.py` | 移除 `import custom_ops`（cann-recipes 不再依赖） |
| `vllm_ascend/attention/dsa_v1.py` | inplace_partial_rotary_mul/compressor→cann_ops_transformer；rms_norm_dynamic_quant→DeviceOperator.rms_norm_dynamic_quant；quant_lightning_indexer→cann_ops_transformer；compressor_metadata 保留 `_C_ascend` |
| `vllm_ascend/attention/context_parallel/dsa_cp.py` | 同上 |
| `vllm_ascend/models/deepseek_v4.py` | mhc_pre_sinkhorn/mhc_post→cann_ops_transformer；cache_dim 608 公式对齐 cann-recipes |
| `vllm_ascend/device/device_op.py` | compressor→cann_ops_transformer；kv_compress_epilog/indexer_quant_cache→cann_ops_transformer；sparse_flash_mla/mixed_quant_sparse_flash_mla→cann_ops_transformer；quant_lightning_indexer quant_mode→DeviceOperator.get_qli_quant_mode()；rms_norm_dynamic_quant→DeviceOperator.rms_norm_dynamic_quant；scatter_nd_update→torch_npu；moe_gating_top_k/moe_init_routing→torch_npu；swiglu_group_quant→cann_ops_nn |
| `vllm_ascend/ops/fused_moe/experts_selector.py` | moe_gating_top_k_hash→torch_npu.npu_moe_gating_top_k（13 参数）；k_group/group_count=1 |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | swiglu_group_quant→cann_ops_nn；dequant_swiglu_quant→torch_npu |
| `vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py` | 同上 |
| `vllm_ascend/ops/fused_moe/moe_mlp.py` | dequant_swiglu_quant→torch_npu |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py` | dispatch_ffn_combine/dispatch_gmm_combine_decode→基类非融合路径 |
| `vllm_ascend/ops/layernorm.py` | npu_add_rms_norm_bias→torch_npu.npu_add_rms_norm + bias add |
| `csrc/build_aclnn.sh` | A3 编译列表移除已迁移算子；A5 编译列表移除已迁移算子 |

---

## 3. 算子编译指导

> SO 加载顺序：vllm-ascend > ops-transformer > ops-nn。

### 3.1 ops-transformer

**算子列表（10 个主算子 + 3 个 metadata）**：
```
inplace_partial_rotary_mul, kv_compress_epilog, indexer_quant_cache,
mhc_pre_sinkhorn, mhc_post, compressor,
sparse_flash_mla, mixed_quant_sparse_flash_mla, quant_lightning_indexer_v2, moe_gating_top_k,
sparse_flash_mla_metadata, mixed_quant_sparse_flash_mla_metadata, quant_lightning_indexer_v2_metadata
```

> ⚠️ metadata 算子**必须手动在 `--ops` 中列出**，不会随主算子自动编译。
>
> ℹ️ whl C++ 扩展调用 V2 aclnn 符号（`aclnnQuantLightningIndexerV2`），`.run` 包必须编译 `quant_lightning_indexer_v2`（不是 V1）。
>
> ℹ️ `--pkg` 会自动构建 `.run` 包 **和** whl 包（`build_torch_extension_whl` 在 `--pkg` 流程中自动执行），whl 是全量编译不按 `--ops` 选择。

```bash
cd ops-transformer

# A3
bash build.sh --pkg --soc=ascend910_93 \
  --ops=inplace_partial_rotary_mul,kv_compress_epilog,indexer_quant_cache,mhc_pre_sinkhorn,mhc_post,compressor,sparse_flash_mla,sparse_flash_mla_metadata,mixed_quant_sparse_flash_mla,mixed_quant_sparse_flash_mla_metadata,quant_lightning_indexer_v2,quant_lightning_indexer_v2_metadata,moe_gating_top_k

# A5
bash build.sh --pkg --soc=ascend950 \
  --ops=inplace_partial_rotary_mul,kv_compress_epilog,indexer_quant_cache,mhc_pre_sinkhorn,mhc_post,compressor,sparse_flash_mla,sparse_flash_mla_metadata,mixed_quant_sparse_flash_mla,mixed_quant_sparse_flash_mla_metadata,quant_lightning_indexer_v2,quant_lightning_indexer_v2_metadata,moe_gating_top_k

# 安装 .run 包（aclnn kernel）
./build_out/*.run

# 安装 whl 包（torch binding，JIT 编译）
python3 -m pip install build_out/*.whl --force-reinstall --no-deps
```

### 3.2 ops-nn

**算子列表（2 个）**：
```
rms_norm_dynamic_quant, swiglu_group_quant
```

> ℹ️ ops-nn torch_extension 与 ops-transformer 架构一致：`cann_ops_nn` namespace + OpBuilder JIT + `aclnn_common.h`。
>
> ℹ️ `rms_norm_dynamic_quant` 有 arch22（A3），无 arch35（A5）→ A5 走 `npu_rms_norm`+`npu_dynamic_quant` 拆分。
>
> ℹ️ `swiglu_group_quant` 仅 arch35（A5），A5-only 算子。
>
> ℹ️ `.run` 包和 whl 包**分开构建**：`--pkg` 只构建 `.run` 包，`--torch_extension` 只构建 whl 包。

```bash
cd ops-nn

# --- .run 包（aclnn kernel）---

# A3 (rms_norm_dynamic_quant: arch22)
bash build.sh --pkg --soc=ascend910_93 --ops=rms_norm_dynamic_quant

# A5 (swiglu_group_quant: arch35)
bash build.sh --pkg --soc=ascend950 --ops=swiglu_group_quant

# 安装 .run 包
./build_out/*.run

# --- whl 包（torch binding，JIT 编译）---
# 整包构建（包含所有 op 的框架 + 自动扫描 torch_extension）
cd torch_extension
python3 -m pip install -r requirements.txt
python3 -m build --wheel -n
python3 -m pip install dist/cann_ops_nn-*.whl --force-reinstall --no-deps
```

### 3.3 torch_npu

> ℹ️ `npu_moe_gating_top_k` 需要升级到 13 参数版本（支持 `input_ids`/`tid2eid` hash 模式）。
>
> ℹ️ 源码 `op_plugin_functions.yaml` 中 `npu_moe_gating_top_k` 缺少 `exposed: all_version`，需手动添加才能暴露到 Python `torch_npu.npu_moe_gating_top_k`。

```bash
cd ascend/pytorch

# 1. 修改 op_plugin_functions.yaml，给 npu_moe_gating_top_k 加 exposed: all_version
#    在 "op_api: all_version" 后添加 "exposed: all_version"

# 2. 编译安装 torch_npu
bash build.sh
pip install dist/torch_npu-*.whl --force-reinstall --no-deps
```

### 3.4 vllm-ascend

```bash
COMPILE_CUSTOM_KERNELS=1 pip install -e .
```

`csrc/build_aclnn.sh` 已按 SOC 版本精简编译列表：
- **A2 (ascend910b)**：不动，保留全部原始算子
- **A3 (ascend910_93)**：移除已迁移算子，保留 17 个
- **A5 (ascend950)**：移除已迁移算子，保留 6 个（compressor_metadata、load_index_kv_cache、causal_conv1d、recurrent_gated_delta_rule、chunk_fwd_o、chunk_gated_delta_rule_fwd_h）

### 3.5 编译顺序

1. ops-transformer（.run + whl）
2. ops-nn（.run + whl）
3. torch_npu（修改 + 编译）
4. vllm-ascend（csrc）

---

## 4. 后续迁移计划

全部可迁移算子已完成（22/26）。剩余 4 个为阻塞状态。

### 4.3 阻塞（❌，4 个）

| # | 算子 | 原因 |
|---|------|------|
| 23 | compressor_metadata | graph capture 不兼容，无外部对应 |
| 24-26 | GMM 模块 3 个 | 无 torch binding 或 vllm 独有；可行方案：3 阶段分解 |

---

## 5. 关键技术决策

| 决策 | 原因 |
|------|------|
| 同名 OP_ADD 需同步调整 `build_aclnn.sh` | SO 加载优先级：vllm-ascend > ops-transformer > ops-nn |
| compressor_metadata 保留 `_C_ascend` | Python 预计算与 graph capture/replay 不兼容，必须作为 graph node 在 forward 路径执行 |
| compressor 拆融合 norm/rope | ot 版 12 参不含 RMSNorm+RoPE，Python 层用 `npu_rms_norm` + `inplace_partial_rotary_mul` 补偿 |
| dequant_swiglu_quant 条件传参 | 无 `swiglu_limit` 时用默认 `swiglu_mode=0`（不钳位），避免输出错误 |
| `rms_norm_dynamic_quant` A5 拆分 | ops-nn 无 arch35 kernel，A5 走 `npu_rms_norm` + `npu_dynamic_quant`，对齐 cann-recipes |
| `swiglu_group_quant` `dst_type=24` | ops-nn `GetAclDataType` 同时支持 ScalarType(24) 和 DType enum(291)，用 24 对齐 cann-recipes |
| `quant_lightning_indexer_v2` 而非 v1 | whl C++ 扩展调 V2 aclnn 符号，`.run` 包必须编译 v2 目录 |
| `sparse_flash_mla` metadata 手动编译 | metadata 不会随主算子自动编译，必须在 `--ops` 中显式列出 |
| torch_npu `npu_moe_gating_top_k` 需升级 | 源码有 13 参数版本（支持 hash），但缺 `exposed: all_version`；需手动添加并重新编译安装 |
| ops-nn whl 整包构建 | `--torch_extension --ops=` 子包方式依赖整包先安装；直接整包构建最简单 |

---

## 6. OP_ADD 冲突注意事项

迁移后大部分同名 OP_ADD 冲突已消除（vllm-ascend csrc 编译列表已移除已迁移算子）。新增/回退算子时仍需检查：

1. 同名 `OP_ADD` 按 SO 加载优先级覆盖（见 1.3 节），vllm-ascend csrc 优先级最高
2. 确保 `csrc/build_aclnn.sh` 不再编译已迁移算子（否则会覆盖外部实现）

---

## 7. 架构映射

| 标识 | 硬件 | 宏 |
|------|------|-----|
| arch22 | Ascend 910B (A3) | `__CCE_AICORE__ == 220` |
| arch35 | Ascend 910C (A5) | `__CCE_AICORE__ == 310` / `__DAV_C310__` |
| ascend950 | Ascend 910C (A5) | cann-recipes A5 编译目标 |
