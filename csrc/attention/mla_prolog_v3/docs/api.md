# MlaPrologV3 API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| vllm-ascend 单算子入口 | `torch.ops._C_ascend.npu_mla_prolog_v3` | 支持 |
| aclnn | `aclnnMlaPrologV3WeightNzGetWorkspaceSize` / `aclnnMlaPrologV3WeightNz` | 支持 |
| Ascend C `<<<>>>` | `mla_prolog_v3<<<blockDim, nullptr, stream>>>` | 支持（诊断/直调；需自备 tiling） |

各入口表达同一套 MLA 前处理融合语义：下采样 → RMSNorm → 上采样 / RoPE → 写入 KV/KR Cache（及可选量化）。  
底层算子名为 **MlaPrologV3**，权重 `weight_dq` / `weight_uq_qr` / `weight_dkv_kr` 需以 **FRACTAL_NZ** 格式传入。

## 2. 公共参数与约束

### 2.0 形状符号

| 符号 | 含义 | 典型/约束值 |
| --- | --- | --- |
| `B` / `S` / `T` | batch / seq / 合轴 token 数（`T=B*S`） | `T≤1M`；允许部分维为 0（空 Tensor） |
| `He` | 隐层宽度 | `{1024,2048,3072,4096,5120,6144,7168,7680,8192}` |
| `Hcq` | Query 压缩维 | `1536` |
| `Hckv` | KV 压缩维 | `512` |
| `D` | Qc 头维 | `128` |
| `Dr` | RoPE 维 | `64` |
| `N` | Query head 数 | `[1, 128]` |
| `Nkv` | KV head 数 | `1` |
| `BlockNum` / `BlockSize` | PA cache 页数/页长 | `BlockSize∈[16,1024]` 且为 16 的倍数 |

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `token_x` | 必选 | 合轴 `(T,He)` 或非合轴 `(B,S,He)` | BF16 / INT8 / FP8_E4M3 / HIF8 | ND | 输入隐状态 |
| `weight_dq` | 必选 | `(He,Hcq)` | 同量化场景 | **FRACTAL_NZ** | \(W^{DQ}\) |
| `weight_uq_qr` | 必选 | `(Hcq,N*(D+Dr))` | 同量化场景 | **FRACTAL_NZ** | \(W^{UQ}\|W^{QR}\) |
| `weight_uk` | 必选 | `(N,D,Hckv)` | BF16 | ND | \(W^{UK}\) |
| `weight_dkv_kr` | 必选 | `(He,Hckv+Dr)` | 同量化场景 | **FRACTAL_NZ** | \(W^{DKV}\|W^{KR}\) |
| `rmsnorm_gamma_cq` | 必选 | `(Hcq,)` | BF16 | ND | Cq RMSNorm \(\gamma\) |
| `rmsnorm_gamma_ckv` | 必选 | `(Hckv,)` | BF16 | ND | Ckv RMSNorm \(\gamma\) |
| `rope_sin` / `rope_cos` | 条件必选 | 合轴 `(T,Dr)` 或非合轴 `(B,S,Dr)`；禁用 RoPE 时传 `nullptr` | BF16 | ND | RoPE 参数；同时非空时启用，同时为空时禁用，混合 null 返回错误 |
| `kv_cache` | 必选（可变） | 见 2.4 CacheMode | BF16 / INT8 / FP8… | ND | \(k^C\) 原地更新 |
| `kr_cache` | 必选（可变） | 见 2.4；`ckvkr_repo_mode=1` 时可为空 | BF16 / INT8 | ND | \(k^R\) 原地更新 |
| `cache_index` | 条件必选 | PA：`(T,)` 或 `(B,S)` 等 | INT64 | ND | PA 写 cache 槽位；取值见 2.4 |
| `dequant_scale_x` | 条件必选 | FULL/MXFP8/FP8/HIF8 必传 | FP32 / FP8_E8M0 | ND | `token_x` 反量化 |
| `dequant_scale_w_dq` | 条件必选 | 同上 | FP32 / FP8_E8M0 | ND | `weight_dq` 反量化 |
| `dequant_scale_w_uq_qr` | 条件必选 | PARTIAL 及以上必传 | FP32 / FP8_E8M0 | ND | `weight_uq_qr` 反量化 |
| `dequant_scale_w_dkv_kr` | 条件必选 | FULL 及以上必传 | FP32 / FP8_E8M0 | ND | `weight_dkv_kr` 反量化 |
| `quant_scale_ckv` / `quant_scale_ckr` | 条件必选 | KV per-channel / per-tensor 等 | FP32 | ND | cache 量化 scale |
| `smooth_scales_cq` | 可选 | `(Hcq,)` 等 | FP32 | ND | Cq 动态量化 smooth |
| `actual_seq_len` | 条件必选 | `(B,)` | INT32 | ND | `PA_BLK_*` 时必传 |
| `k_nope_clip_alpha` | 可选 | 标量/向量 | FP32 | ND | Ckv clip 缩放 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `query` | 合轴 `(T,N,Hckv)` / 非合轴 `(B,S,N,Hckv)` | BF16 / INT8 / FP8… | \(q^N\) |
| `query_rope` | 合轴 `(T,N,Dr)` / 非合轴 `(B,S,N,Dr)` | BF16 | \(q^R\) |
| `dequant_scale_q_nope` | 全量化 + KV per-tensor 时非空，否则空 | FP32 | Query 动态量化 scale |
| `query_norm` | `query_norm_flag=True` 时非空 | BF16 / 量化 dtype | \(c^Q\) |
| `dequant_scale_q_norm` | `query_norm_flag` 且量化时非空 | FP32 / FP8_E8M0 | `query_norm` 反量化 scale |

`kv_cache` / `kr_cache` 为可变输入：按 `cache_index` 原地写入，不作为独立 alias 输出返回。

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `rmsnorm_epsilon_cq` | float | `1e-5` | `>0` | Cq RMSNorm \(\epsilon\) |
| `rmsnorm_epsilon_ckv` | float | `1e-5` | `>0` | Ckv RMSNorm \(\epsilon\) |
| `cache_mode` | str | `"PA_BSND"` | 见 2.4 | cache 布局 |
| `query_norm_flag` | bool | `false` | `{false,true}` | 是否输出 `query_norm` |
| `weight_quant_mode` | int | `0` | `{0,1,2,3,4,5}` | 权重/激活量化模式 |
| `kv_cache_quant_mode` | int | `0` | `{0,1,2,3}` | KV cache 量化模式 |
| `query_quant_mode` | int | `0` | `{0,1}` | Query 量化；per-tensor KV 时需为 1 |
| `ckvkr_repo_mode` | int | `0` | `{0,1}` | 与 `quant_scale_repo_mode` 成对；pertile 必须为 1 |
| `quant_scale_repo_mode` | int | `0` | `{0,1}` | 同上 |
| `tile_size` | int | `128` | pertile 时必须为 `128` | per-token-per-group tile |
| `qc_qr_scale` | float | `1.0` | 有限浮点 | Query 尺度 \(\alpha_q\) |
| `kc_scale` | float | `1.0` | 有限浮点 | Key 尺度 \(\alpha_{kv}\) |

RoPE 开关由 `ropeSin` / `ropeCos` 的 nullity 推导：同时非空 → 开启，同时为空 → 关闭；混合 null 返回参数错误。

#### 量化模式合法组合（`weight_quant_mode` × `kv_cache_quant_mode`）

| wq | 含义 | 合法 kvq |
| --- | --- | --- |
| `0` | 非量化 | `{0}` |
| `1` | PARTIAL（仅 `weight_uq_qr` 量化） | `{0, 2, 3}` |
| `2` | FULL INT8 | `{0, 1, 3}` |
| `3` | MXFP8 | `{0, 1, 3}` |
| `4` | FP8 | `{0, 1, 3}` |
| `5` | HIF8 | `{0, 1, 3}` |

`kvq`：`0` 非量化，`1` per-tensor，`2` per-channel，`3` per-tile。

### 2.4 CacheMode

| `cache_mode` | `token_x` | `kv_cache` / `kr_cache`（非 pertile） | `cache_index` |
| --- | --- | --- | --- |
| `PA_BSND` / `PA_NZ` | `(T,He)` | `(BlockNum,BlockSize,Nkv,Hckv/Dr)` | `(T,)`，值 ∈ `[0, BlockNum*BlockSize)` |
| `PA_BLK_BSND` / `PA_BLK_NZ` | `(T,He)` | 同上 | block 级 index；需 `actual_seq_len` |
| `BSND` | `(B,S,He)` | `(B,S,Nkv,Hckv/Dr)` | `(B,S)` |
| `TND` | `(T,He)` | `(T,Nkv,Hckv/Dr)` | `(T,)` |

pertile（`kvq=3`）时 `ckvkr_repo_mode=quant_scale_repo_mode=1`，`kv_cache` 末维为打包 `Dtile`，`kr_cache` 为空 Tensor。

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnMlaPrologV3WeightNzGetWorkspaceSize(
    const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr,
    const aclTensor *weightUk, const aclTensor *weightDkvKr,
    const aclTensor *rmsnormGammaCq, const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin, const aclTensor *ropeCos,
    aclTensor *kvCacheRef, aclTensor *krCacheRef,
    const aclTensor *cacheIndexOptional,
    const aclTensor *dequantScaleXOptional, const aclTensor *dequantScaleWDqOptional,
    const aclTensor *dequantScaleWUqQrOptional, const aclTensor *dequantScaleWDkvKrOptional,
    const aclTensor *quantScaleCkvOptional, const aclTensor *quantScaleCkrOptional,
    const aclTensor *smoothScalesCqOptional, const aclTensor *actualSeqLenOptional,
    const aclTensor *kNopeClipAlphaOptional,
    double rmsnormEpsilonCq, double rmsnormEpsilonCkv, char *cacheModeOptional,
    int64_t weightQuantMode, int64_t kvCacheQuantMode, int64_t queryQuantMode,
    int64_t ckvkrRepoMode, int64_t quantScaleRepoMode, int64_t tileSize,
    double qcQrScale, double kcScale,
    const aclTensor *queryOut, const aclTensor *queryRopeOut,
    const aclTensor *dequantScaleQNopeOutOptional,
    const aclTensor *queryNormOutOptional, const aclTensor *dequantScaleQNormOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnMlaPrologV3WeightNz(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

`GetWorkspaceSize` 完成参数校验与 executor 创建；第二段在传入 stream 上异步执行。  
`ropeSin` / `ropeCos` 同时非空时启用 RoPE，同时为空时禁用；一个空一个非空时返回参数错误。  
`kvCacheRef` / `krCacheRef` 同时是输入和输出。输入、输出、workspace 和 executor 必须保持有效直到 stream 完成。

### 3.2 调用示例

```cpp
// 按 2.1/2.2 创建 aclTensor；weightDq/UqQr/DkvKr 为 FRACTAL_NZ。
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
ACLNN_CHECK(aclnnMlaPrologV3WeightNzGetWorkspaceSize(
    tokenX, weightDq, weightUqQr, weightUk, weightDkvKr,
    gammaCq, gammaCkv, ropeSin, ropeCos, kvCache, krCache,
    cacheIndex, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, 1e-5, 1e-5, const_cast<char *>("PA_BSND"),
    0, 0, 0, 0, 0, 128, 1.0, 1.0,
    queryOut, queryRopeOut, nullptr, nullptr, nullptr,
    &workspaceSize, &executor));
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnMlaPrologV3WeightNz(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
```

## 4. `torch.ops._C_ascend` API

### 4.1 接口签名

```python
query, query_rope, dequant_scale_q_nope, query_norm, dequant_scale_q_norm = (
    torch.ops._C_ascend.npu_mla_prolog_v3(
        token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
        rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos,
        kv_cache, kr_cache,  # mutable
        *,
        cache_index=None,
        dequant_scale_x=None, dequant_scale_w_dq=None,
        dequant_scale_w_uq_qr=None, dequant_scale_w_dkv_kr=None,
        quant_scale_ckv=None, quant_scale_ckr=None, smooth_scales_cq=None,
        actual_seq_len=None, k_nope_clip_alpha=None,
        rmsnorm_epsilon_cq=1e-5, rmsnorm_epsilon_ckv=1e-5,
        cache_mode="PA_BSND", query_norm_flag=False,
        weight_quant_mode=0, kv_cache_quant_mode=0, query_quant_mode=0,
        ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128,
        qc_qr_scale=1.0, kc_scale=1.0,
    )
)
```

仅在 Ascend950 构建且加载 `vllm_ascend_C` + 自定义 opp 后可用。  
`rope_sin` / `rope_cos` 为必传位置参数：同时非空启用 RoPE，同时为空（`numel()==0`）禁用；不允许一空一非空。  
`token_x` rank=2 为合轴 `(T,He)`，rank=3 为 `(B,S,He)`。  
`kv_cache` / `kr_cache` 原地更新；不需要的 optional 输出以空 Tensor 返回。

NZ 权重可用 `torch_npu.npu_format_cast(w.contiguous(), 29)` 转换。

### 4.2 调用示例（bf16 / PA_BSND）

```python
import torch
import torch_npu

# 需已加载 vllm_ascend_C，并 source 自定义 opp set_env.bash
torch_npu.npu.config.allow_internal_format = True
t, he, n = 2, 1024, 8
hcq, hckv, d, dr = 1536, 512, 128, 64
device, dtype = "npu:0", torch.bfloat16

def rnd(*shape):
    return torch.randn(*shape, device=device, dtype=dtype)

token_x = rnd(t, he)
weight_dq = torch_npu.npu_format_cast(rnd(he, hcq).contiguous(), 29)
weight_uq_qr = torch_npu.npu_format_cast(rnd(hcq, n * (d + dr)).contiguous(), 29)
weight_uk = rnd(n, d, hckv)
weight_dkv_kr = torch_npu.npu_format_cast(rnd(he, hckv + dr).contiguous(), 29)
gamma_cq = torch.ones(hcq, device=device, dtype=dtype)
gamma_ckv = torch.ones(hckv, device=device, dtype=dtype)
rope_cos = rnd(t, dr)
rope_sin = rnd(t, dr)
kv_cache = torch.zeros(2, 128, 1, hckv, device=device, dtype=dtype)
kr_cache = torch.zeros(2, 128, 1, dr, device=device, dtype=dtype)
cache_index = torch.arange(t, device=device, dtype=torch.int64)

query, query_rope, *_ = torch.ops._C_ascend.npu_mla_prolog_v3(
    token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
    gamma_cq, gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache,
    cache_index=cache_index, cache_mode="PA_BSND")
# RoPE disabled: pass empty tensors for both rope inputs
empty_rope = torch.empty(0, device=device, dtype=dtype)
q_no_rope, qr_no_rope, *_ = torch.ops._C_ascend.npu_mla_prolog_v3(
    token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr,
    gamma_cq, gamma_ckv, empty_rope, empty_rope, kv_cache.clone(), kr_cache.clone(),
    cache_index=cache_index, cache_mode="PA_BSND")
torch.npu.synchronize()
# query: [T,N,Hckv], query_rope: [T,N,Dr]；kv/kr_cache 已按 cache_index 写入
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 与序列化 tiling data 必须来自同一组 host tiling。参数顺序与 kernel 定义一致：

```cpp
mla_prolog_v3<<<blockDim, nullptr, stream>>>(
    tokenX, weightDq, weightUqQr, weightUk, weightDkvKr,
    rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos,
    kvCache, krCache, cacheIndex,
    dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr,
    quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
    queryOut, queryRopeOut, kvCacheOut, krCacheOut,
    dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut,
    workspace, tiling);
```

直调通路只作 route/诊断入口；公开 Python / aclnn 负责完整校验。GM 按连续物理布局解释。

## 6. 已知限制

- Torch schema 始终注册，实际可用性取决于 `csrc/build_aclnn.sh` 是否按 **Ascend950** 构建并安装了该自定义算子包。
- `weight_dq` / `weight_uq_qr` / `weight_dkv_kr` 必须为 **FRACTAL_NZ**。
- `Hcq=1536`，`Hckv=512`，`D=128`，`Dr=64`，`Nkv=1`；`He` 仅白名单集合；`N∈[1,128]`。
- `weight_quant_mode` 与 `kv_cache_quant_mode` 必须落在 §2.3 合法表内。
- pertile 要求 `ckvkr_repo_mode=quant_scale_repo_mode=1` 且 `tile_size=128`；`kr_cache` 为空。
- KV per-tensor 时 `query_quant_mode` 必须为 `1`。
- `PA_BLK_*` 需要 `actual_seq_len`；末项语义与合轴 `T` 一致。
- RoPE 开关由 `rope_sin` / `rope_cos` 是否为空决定：同时非空启用，同为空（`numel()==0`）禁用。
- B/S/T/Skv 允许为 0：空 query 时不更新 cache；Skv=0 时正常算 query 但不写 cache。

## 7. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| 必选 tensor、workspaceSize 或 executor 为空 | `ACLNN_ERR_PARAM_NULLPTR` |
| rank/shape/dtype/layout、量化组合或 CacheMode 非法 | `ACLNN_ERR_PARAM_INVALID` / tiling `GRAPH_FAILED` |
| 内部 tensor 创建或 L0 调用失败 | `ACLNN_ERR_INNER_NULLPTR` |
| torch 侧未注册算子（非 950 构建）或输入非法 | `RuntimeError` / `AttributeError` |
