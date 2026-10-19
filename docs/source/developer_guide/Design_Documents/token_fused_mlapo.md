# No-RoPE MLAPO with Token-Fused FlashMLA Cache

## Scope

The Kimi K3 main model can use `cann_ops_transformer.mla_prolog` for no-RoPE
decode while FlashMLA retains its single token-fused cache tensor. This reuses
the existing optional-RoPE binding; no additional V4 binding is required.
The existing `enable_mlapo` configuration controls the route.

The route requires:

- A5 FlashMLA on Model Runner V2, with BF16 activations and cache;
- a main-model MLA layer with `use_mla_rope=False` and compressed Q projection;
- native BF16 weights or matching W8A8-MXFP8 dynamic projection methods;
- pure decode with `0 < T <= MLAPO_MAX_SUPPORTED_TOKENS` (currently 1024);
- no FA/KV quantization, KV NZ, PCP, sharded PCP decode, or chunked Q projection;
- preallocated workspace covering all `T` rows, including graph padding.

The input-width guard follows the supported ProLog projection widths.
Unsupported configurations use the existing preprocessing route. The draft
model, RoPE V3 writers and DSpark context writers retain their current routing.

## Cache and Operator Contract

The formal FlashMLA cache remains `[blocks, 128, 1, 576]`, including any physical
page padding and storage offset. Splitting the final dimension into 512 and 64
creates views with token stride 576. These views cannot be passed directly as
MLAPO output caches: the operator requires dense dimensions within each block.

Only the current decode rows are staged:

```text
AscendMLAImpl.forward
  -> _can_use_flashmla_mlapo
  -> mla_preprocess_only_decode
     -> cann_ops_transformer.mla_prolog
        sin=None, cos=None, model Q/KV RMSNorm epsilons
        dense K512/K64 workspace, local int64 arange indices
     -> trim padded Q heads to the actual local head count
     -> reorg_decode_q
     -> DeviceOperator.reshape_and_cache
        original external_flashmla.slots -> formal token-fused cache
  -> _forward_external_flashmla
```

No-RoPE retains the raw 64-dimensional Q/K components. It does not replace them
with zeros. Native weights use mode 0 without activation quantization;
MXFP8 uses the existing mode 3 activation/weight-scale handling.

Non-DCP layers allocate workspace during weight loading, before KV memory
sizing. Its capacity is bounded by the scheduler token limit and the 1024-row
MLAPO route limit; its block size is the FlashMLA kernel block size, 128.
Weight reloads preserve workspace addresses. There is no history-cache copy or
new device-to-host synchronization in the route.
Reloads with unchanged tensor shape, dtype and device also refresh the NZ
weights, padded UK weights and quantization scales in place. Changes to those
properties require graph recapture. Model RMSNorm epsilons are passed only by
this FlashMLA route; legacy MLAPO calls retain their operator defaults.

DCP reuses `_dcp_current_kv_buffers`, including its actual configured block
size. MLAPO requires workspace blocks between 16 and 1024 tokens, aligned to
16; other block sizes select the fallback. All current rows are produced before
rank-owned history rows are scattered. A slot of `-1` skips the history write.
Where DCP current attention
requires the complete chunk, the same dense rows remain in the preprocess
result. Insufficient workspace for a padded graph bucket selects the existing
fallback before entering MLAPO.

## Validation

CPU composition tests emulate only the CANN kernel and communication. They
check route boundaries, native/MXFP8 admission, dense staging, external slots,
head trimming, epsilon propagation, DCP handoff and storage sentinels. Existing
FlashMLA V3 writer and DCP tests cover compatibility with the other routes.

The NPU integration test is
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_flashmla_mlapo.py`.
It checks the native BF16 path against the complete unfused preprocessing
sequence and reports device timings for both complete sequences. It does not
compare a single Matmul or RMSNorm against the entire fused operator.

Hardware acceptance also requires MXFP8, K3 eager/full-graph multi-step decode,
and DCP with speculative query lengths greater than one. CPU test success is
not evidence of NPU precision, graph replay correctness or a performance gain.
The optional `cann_ops_transformer` installation must expose the no-RoPE
`mla_prolog` contract used by the existing main-model MLAPO path.
