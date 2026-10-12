# MlaPrologV3 API and Usage Examples

> 中文版 / Chinese version: [api.md](./api.md)

## 1. API Overview

| Path | API / Entry | Support |
| --- | --- | --- |
| vllm-ascend custom op | `torch.ops._C_ascend.npu_mla_prolog_v3` | Supported |
| aclnn | `aclnnMlaPrologV3WeightNzGetWorkspaceSize` / `aclnnMlaPrologV3WeightNz` | Supported |
| Ascend C `<<<>>>` | `mla_prolog_v3<<<blockDim, nullptr, stream>>>` | Supported (diagnostics / direct invoke; caller must provide tiling) |

All entries implement the same fused MLA preprocess semantics: down-projection → RMSNorm → up-projection / RoPE → write KV/KR cache (with optional quantization).  
The underlying operator name is **MlaPrologV3**. Weights `weight_dq` / `weight_uq_qr` / `weight_dkv_kr` must be passed in **FRACTAL_NZ** format.

**Platform support**

| Platform | Arch | Notes |
| --- | --- | --- |
| Ascend910B / Ascend910_93 (A2/A3) | arch22 (DAV_2201) | `weight_quant_mode ∈ {0,1,2}` only; no MXFP8/FP8/HIF8 full-quant; no SplitM |
| Ascend950 (A5) | arch35 (DAV_3510) | All quant modes in §2.3; SplitM supported where applicable |

## 2. Common Parameters and Constraints

### 2.0 Shape Symbols

| Symbol | Meaning | Typical / constrained values |
| --- | --- | --- |
| `B` / `S` / `T` | batch / seq / fused token count (`T=B*S`) | `T≤1M`; some dims may be 0 (empty tensor) |
| `He` | hidden size | `{1024,2048,3072,4096,5120,6144,7168,7680,8192}` |
| `Hcq` | Query compression dim | `1536` |
| `Hckv` | KV compression dim | `512` |
| `D` | Qc head dim | `128` |
| `Dr` | RoPE dim | `64` |
| `N` | Query head count | `[1, 128]` |
| `Nkv` | KV head count | `1` |
| `BlockNum` / `BlockSize` | PA cache pages / page length | `BlockSize∈[16,1024]` and multiple of 16 |

### 2.1 Inputs

| Name | Required / optional | Shape | Dtype | Layout | Description |
| --- | --- | --- | --- | --- | --- |
| `token_x` | required | fused `(T,He)` or non-fused `(B,S,He)` | BF16 / INT8 / FP8_E4M3 / HIF8 | ND | Input hidden states |
| `weight_dq` | required | `(He,Hcq)` | per quant scenario | **FRACTAL_NZ** | \(W^{DQ}\) |
| `weight_uq_qr` | required | `(Hcq,N*(D+Dr))` | per quant scenario | **FRACTAL_NZ** | \(W^{UQ}\|W^{QR}\) |
| `weight_uk` | required | `(N,D,Hckv)` | BF16 | ND | \(W^{UK}\) |
| `weight_dkv_kr` | required | `(He,Hckv+Dr)` | per quant scenario | **FRACTAL_NZ** | \(W^{DKV}\|W^{KR}\) |
| `rmsnorm_gamma_cq` | required | `(Hcq,)` | BF16 | ND | Cq RMSNorm \(\gamma\) |
| `rmsnorm_gamma_ckv` | required | `(Hckv,)` | BF16 | ND | Ckv RMSNorm \(\gamma\) |
| `rope_sin` / `rope_cos` | required (may be empty) | fused `(T,Dr)` or non-fused `(B,S,Dr)`; empty when RoPE is off | BF16 | ND | RoPE tables; both non-empty enables RoPE, both empty disables; mixed empty/non-empty is invalid |
| `kv_cache` | required (mutable) | see 2.4 CacheMode | BF16 / INT8 / FP8… | ND | \(k^C\) updated in-place |
| `kr_cache` | required (mutable) | see 2.4; may be empty when `ckvkr_repo_mode=1` | BF16 / INT8 | ND | \(k^R\) updated in-place |
| `cache_index` | conditionally required | PA: `(T,)` or `(B,S)`, etc. | INT64 | ND | PA write slots; values in 2.4 |
| `dequant_scale_x` | conditionally required | required for FULL/MXFP8/FP8/HIF8 | FP32 / FP8_E8M0 | ND | `token_x` dequant scale |
| `dequant_scale_w_dq` | conditionally required | same as above | FP32 / FP8_E8M0 | ND | `weight_dq` dequant scale |
| `dequant_scale_w_uq_qr` | conditionally required | required for PARTIAL and above | FP32 / FP8_E8M0 | ND | `weight_uq_qr` dequant scale |
| `dequant_scale_w_dkv_kr` | conditionally required | required for FULL and above | FP32 / FP8_E8M0 | ND | `weight_dkv_kr` dequant scale |
| `quant_scale_ckv` / `quant_scale_ckr` | conditionally required | KV per-channel / per-tensor, etc. | FP32 | ND | cache quant scales |
| `smooth_scales_cq` | optional | `(Hcq,)`, etc. | FP32 | ND | Cq dynamic-quant smooth |
| `actual_seq_len` | conditionally required | `(B,)` | INT32 | ND | required for `PA_BLK_*` |
| `k_nope_clip_alpha` | optional | scalar / vector | FP32 | ND | Ckv clip scale |

### 2.2 Outputs

| Name | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `query` | fused `(T,N,Hckv)` / non-fused `(B,S,N,Hckv)` | BF16 / INT8 / FP8… | \(q^N\) |
| `query_rope` | fused `(T,N,Dr)` / non-fused `(B,S,N,Dr)` | BF16 | \(q^R\) |
| `dequant_scale_q_nope` | non-empty for full-quant + KV per-tensor; otherwise empty | FP32 | Query dynamic-quant scale |
| `query_norm` | non-empty when `query_norm_flag=True` | BF16 / quant dtype | \(c^Q\) |
| `dequant_scale_q_norm` | non-empty when `query_norm_flag` and quantized | FP32 / FP8_E8M0 | `query_norm` dequant scale |

`kv_cache` / `kr_cache` are mutable inputs: they are written in-place by `cache_index` and are not returned as separate aliased outputs.

### 2.3 Attributes

| Name | Type | Default | Range | Description |
| --- | --- | --- | --- | --- |
| `rmsnorm_epsilon_cq` | float | `1e-5` | `>0` | Cq RMSNorm \(\epsilon\) |
| `rmsnorm_epsilon_ckv` | float | `1e-5` | `>0` | Ckv RMSNorm \(\epsilon\) |
| `cache_mode` | str | `"PA_BSND"` | see 2.4 | cache layout |
| `query_norm_flag` | bool | `false` | `{false,true}` | whether to emit `query_norm` |
| `weight_quant_mode` | int | `0` | `{0,1,2,3,4,5}` (A2/A3: `{0,1,2}` only) | weight / activation quant mode |
| `kv_cache_quant_mode` | int | `0` | `{0,1,2,3}` | KV cache quant mode |
| `query_quant_mode` | int | `0` | `{0,1}` | Query quant; must be `1` for KV per-tensor |
| `ckvkr_repo_mode` | int | `0` | `{0,1}` | paired with `quant_scale_repo_mode`; must be `1` for pertile |
| `quant_scale_repo_mode` | int | `0` | `{0,1}` | same as above |
| `tile_size` | int | `128` | must be `128` for pertile | per-token-per-group tile |
| `qc_qr_scale` | float | `1.0` | finite float | Query scale \(\alpha_q\) |
| `kc_scale` | float | `1.0` | finite float | Key scale \(\alpha_{kv}\) |

RoPE enablement is derived from `ropeSin` / `ropeCos` emptiness: both non-empty → on; both empty → off; mixed empty/non-empty → parameter error. Supported on A2/A3/A5.

On Ascend 950PR/Ascend 950DT, `kv_cache` / `kr_cache` may be non-contiguous on the first axis; all other axes must be contiguous.

#### Legal quant combinations (`weight_quant_mode` × `kv_cache_quant_mode`)

| wq | Meaning | Legal kvq | A2/A3 |
| --- | --- | --- | --- |
| `0` | non-quant | `{0}` | yes |
| `1` | PARTIAL (`weight_uq_qr` only) | `{0, 2, 3}` | yes (`kvq∈{0,2}` typical) |
| `2` | FULL INT8 | `{0, 1, 3}` | yes |
| `3` | MXFP8 | `{0, 1, 3}` | **no** |
| `4` | FP8 | `{0, 1, 3}` | **no** |
| `5` | HIF8 | `{0, 1, 3}` | **no** |

`kvq`: `0` non-quant, `1` per-tensor, `2` per-channel, `3` per-tile.

### 2.4 CacheMode

| `cache_mode` | `token_x` | `kv_cache` / `kr_cache` (non-pertile) | `cache_index` |
| --- | --- | --- | --- |
| `PA_BSND` / `PA_NZ` | `(T,He)` | `(BlockNum,BlockSize,Nkv,Hckv/Dr)` | `(T,)`, values ∈ `[0, BlockNum*BlockSize)` |
| `PA_BLK_BSND` / `PA_BLK_NZ` | `(T,He)` | same as above | block-level index; requires `actual_seq_len` |
| `BSND` | `(B,S,He)` | `(B,S,Nkv,Hckv/Dr)` | `(B,S)` |
| `TND` | `(T,He)` | `(T,Nkv,Hckv/Dr)` | `(T,)` |

For pertile (`kvq=3`), `ckvkr_repo_mode=quant_scale_repo_mode=1`, the last dim of `kv_cache` is packed `Dtile`, and `kr_cache` is an empty tensor.

## 3. aclnn API

### 3.1 Signatures

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

`GetWorkspaceSize` validates parameters and builds the executor; the second API runs asynchronously on the given stream.  
`ropeSin` / `ropeCos` both non-empty enable RoPE; both empty disable RoPE; mixed empty/non-empty returns a parameter error.  
`kvCacheRef` / `krCacheRef` are both inputs and outputs. Inputs, outputs, workspace, and executor must remain valid until the stream completes.

### 3.2 Example

```cpp
// Create aclTensors per §2.1/2.2; weightDq/UqQr/DkvKr must be FRACTAL_NZ.
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

### 4.1 Signature

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

Available after building for Ascend910B / Ascend910_93 (A2/A3) or Ascend950 and loading `vllm_ascend_C` plus the custom opp package.  
`rope_sin` / `rope_cos` are required positional args: both non-empty enables RoPE; both empty (`numel()==0`) disables RoPE; mixed empty/non-empty is invalid (supported on A2/A3/A5).  
`token_x` rank=2 is fused `(T,He)`; rank=3 is `(B,S,He)`.  
`kv_cache` / `kr_cache` are updated in-place; unused optional outputs are returned as empty tensors.  
A2/A3 do not support MXFP8/FP8/HIF8 full-quant or SplitM; `weight_quant_mode` is limited to `{0,1,2}`.

NZ weights can be converted with `torch_npu.npu_format_cast(w.contiguous(), 29)`.

### 4.2 Example (bf16 / PA_BSND)

```python
import torch
import torch_npu

# Requires vllm_ascend_C loaded and custom opp set_env.bash sourced.
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
# query: [T,N,Hckv], query_rope: [T,N,Dr]; kv/kr_cache written by cache_index
```

## 5. Ascend C `<<<>>>` Direct Invoke

`blockDim`, workspace, and serialized tiling data must come from the same host tiling. Argument order matches the kernel definition:

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

Direct invoke is for routing / diagnostics only; public Python / aclnn paths perform full validation. GM buffers are interpreted as contiguous physical layouts.

## 6. Known Limitations

- Torch schema is always registered; runtime availability depends on building and installing the custom opp via `csrc/build_aclnn.sh` for **Ascend910B / Ascend910_93 / Ascend950**.
- `weight_dq` / `weight_uq_qr` / `weight_dkv_kr` must be **FRACTAL_NZ**.
- `Hcq=1536`, `Hckv=512`, `D=128`, `Dr=64`, `Nkv=1`; `He` must be in the whitelist; `N∈[1,128]`.
- `weight_quant_mode` and `kv_cache_quant_mode` must match the legal table in §2.3 (A2/A3: `weight_quant_mode∈{0,1,2}` only).
- Pertile requires `ckvkr_repo_mode=quant_scale_repo_mode=1` and `tile_size=128`; `kr_cache` must be empty.
- For KV per-tensor, `query_quant_mode` must be `1`.
- `PA_BLK_*` requires `actual_seq_len`; the last element must match fused `T`.
- RoPE on/off is controlled by whether `rope_sin` / `rope_cos` are empty: both non-empty enables, both empty (`numel()==0`) disables.
- B/S/T/Skv may be 0: empty query does not update cache; Skv=0 computes query but does not write cache.

## 7. Errors and Return Codes

| Condition | Return code / exception |
| --- | --- |
| Required tensor, workspaceSize, or executor is null | `ACLNN_ERR_PARAM_NULLPTR` |
| Illegal rank/shape/dtype/layout, quant combo, or CacheMode | `ACLNN_ERR_PARAM_INVALID` / tiling `GRAPH_FAILED` |
| Internal tensor create or L0 call failure | `ACLNN_ERR_INNER_NULLPTR` |
| Op not registered (package not built for the target SOC) or illegal torch inputs | `RuntimeError` / `AttributeError` |
