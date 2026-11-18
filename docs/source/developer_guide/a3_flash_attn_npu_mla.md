# A3 FlashAttention NPU MLA integration

This implementation adds the optional FA3 APIs from
[MinghuasLab/flash-attention-npu](https://github.com/MinghuasLab/flash-attention-npu)
to the existing MLA backend. It shares the A5 FlashMLA batch and DCP orchestration,
while preserving the A3 component-major KV cache protocol.

## Source baseline and dependency

- vllm-ascend main: `9005a22a862a95d52025b70fb2bff8eed8a5a5e3`.
- A5 implementation reference:
  [#18742](https://github.com/vllm-project/vllm-ascend/pull/18742), merged into
  main on October 20, 2026. The baseline `attention/flashmla.py` has the same Git
  blob (`fc92170e303acd03f01e7b810197bca5f62ab7bc`) as that merged PR.
- Operator source inspected: `aac4ad2f81d8f915256f48011217078a3cd5b891`.
- Required module: `flash_attn_npu_3`, exporting both `flash_attn_with_kvcache`
  and `get_scheduler_metadata`, with Ascend 910 absorbed MLA and paged-TND support.

Build/install the external operator package for the worker's CANN, PyTorch and
torch_npu stack, using that repository's instructions. The package is optional;
it is imported only after device selection on an eligible A3 worker. Detecting
the two public names does not verify the installed binary's MLA/stride support.
An older wheel exporting these names may still lack the required kernel support.

## Operator and layout mapping

| Contract | A3 FA3 | A5 FlashMLA |
| --- | --- | --- |
| Attention API | `flash_attn_with_kvcache` | `flash_mla` |
| Metadata API | `get_scheduler_metadata` | `flash_mla_metadata` |
| Package | `flash_attn_npu_3` | `cann_ops_transformer.ops` |
| Q passed to attention | RoPE `q[..., 512:]`, NoPE `qv=q[..., :512]` | Fused `[NoPE 512, RoPE 64]` |
| Persistent cache | Separate logical `(NoPE, RoPE)` tensors | Token-fused BBND tensor |
| FA3 K/V arguments | `k_cache=RoPE`, `v_cache=NoPE` | Single fused cache |
| Metadata dimensions | `headdim=64`, `headdim_v=512` | QK 576, V 512 |
| Metadata dtype | Opaque `uint8` | Opaque `int32` |
| External output | TND, converted to NTD | NTD |
| External LSE | NT FP32 | NT FP32 |
| Maximum-length arguments | Static allocated query/KV capacity bounds | Device-derived `-1` |
| Prefill | Existing FIA path | Existing FIA path |

Absorbed MLA computes
`scale * (q_nope @ cache_nope.T + q_rope @ cache_rope.T)` and uses
`cache_nope` as the value. FA3 accepts these two query terms through `qv` and `q`.
The original MLA softmax scale is passed explicitly.

Logical separation does not imply separate physical allocations: an A3 hybrid
page may contain `[all NoPE][all RoPE][padding]` in one raw backing. FA3 receives
the two views and retains their leading physical page strides. Inner component
pages must be packed; the adapter rejects token-fused history views before the
external package could materialize them. Persistent KV writes remain owned by
the existing preprocessing path. Append-KV is not requested from FA3.

## Selection and shared orchestration

A3 uses `HardwareCapability.MLA_FLASH_ATTN_NPU`; A5 retains `MLA_FLASH`.
The flags `use_flash_attn_npu` and `use_flashmla` remain distinct, so installing
FA3 does not request the A5 fused-cache allocation branch.

The A3 path currently requires Model Runner V2, one KV head, latent dimension
512, RoPE dimension 64, FP16/BF16 unquantized KV, no KV-NZ and no PCP. Speculative
decoding retains the existing `dspark` gate. The existing connector compatibility
gate also applies. The MLA backend's kernel block size is 128. A5 retains its
existing Q-head restrictions; those A5-specific restrictions are not imposed
on FA3. Missing optional APIs select FIA before metadata/cache allocation;
failures inside an installed package or running operator are propagated.

`FlashAttnNPUMetadataBuilder` reuses real prefill/decode classification, request
ordering, graph padding, captured buffer reuse and device-length refresh from
`FlashMLAMetadataBuilder`. The existing `flashmla_state` and `external_flashmla`
fields represent this shared scheduled-decode route for either adapter.
One-token prompt tails remain FIA prefill. Prefill and decode can coexist in an
eager batch; FULL graph capture remains restricted to decode-only batches.

FA3 metadata receives the same lengths, causal mode, head counts, page size and
capacity bounds as attention. Query bounds come from allocated token capacity,
and KV bounds from block-table columns times 128. No device `max().item()` is
introduced. `seqused_q` has no counterpart in the public FA3 KV-cache API:
query intervals come from `cu_seqlens_q`, while zero KV lengths represent dead
requests/padding. Empty-row kernel behavior must pass the hardware tests below.

For DCP, history uses all gathered Q heads, rank-local KV lengths and noncausal
attention. DSpark causal verification additionally computes a replicated current
chunk with local Q heads. A3 stores that private chunk as separate packed
components. The existing slot mapping, global-history subtraction before
interleave sharding, A2A exchange, LSE combination and stream ordering are reused.
Both current-cache components are recorded on the side stream.

## Compatibility boundaries

- A5 keeps its package, fused-cache allocation and operator calls.
- A2/310P and Model Runner V1 retain FIA selection.
- A3 does not change its persistent cache representation when FA3 is installed.
- This change does not add sparse/quantized MLA, PCP or other speculative methods.
- Raw PD transfers between A3 component-major and A5 token-fused endpoints are
  incompatible without an explicit layout conversion. Equal page byte sizes
  do not establish compatibility. This change adds no cross-generation converter.
- Do not infer binary compatibility or performance parity from the shared
  framework orchestration. Record the external package revision in deployments.

## Verification

CPU boundary and regression suite, runnable without vLLM/torch_npu:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  --confcutdir=tests/ut/attention \
  tests/ut/attention/test_flash_attn_npu_mla_cpu.py \
  tests/ut/attention/test_flashmla_dcp.py \
  tests/ut/attention/test_flashmla_prolog_v3_cpu.py -q
```

This checks argument/layout mapping, page strides, cache immutability, backend
selection, metadata refresh with stable buffer addresses, A3 DCP ordinary and
speculative numerical orchestration, and the existing A5 DCP/PROLOG behavior.
The kernels, streams and collectives in these tests are emulated on CPU.

The actual hardware profile capability matrix is covered by
`tests/ut/device/test_hardware_profile.py` in the standard UT environment.

On one exclusively allocated A3 device with the external package installed:

```bash
python -m pytest \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_flash_attn_npu_mla.py -q
```

This needs no model weights or network service. It checks FP16/BF16, 8/32/96 Q
heads, causal/noncausal multi-token queries, nontrivial page indices, padded
page strides, zero KV/query rows, graph padding, read-only caches and metadata
refresh during NPUGraph replay against an independent dense CPU reference.
Reserve one whole A3 dual-die package if required by the host's ownership policy.

Hardware operator tests, latency/memory comparison against FIA, distributed
collectives, PD interoperability and full Kimi K3 accuracy/performance validation
must be reported separately. Passing CPU tests establishes none of those results.
