# Native SFA indexer key/scale store

## Description

- **Function**: Publish quantized Indexer keys and scales in one kernel. The
  ordinary path stores INT8 K and converts FP32 scales to FP16. HiSparse stores
  FP8 E4M3FN K and FP32 scales through INT8/INT32 aliases, preserving every bit.
  Quantization stays in the existing backend and is not repeated here.
- **Formula**: For a valid slot `s`, let `p = s // block_size` and
  `t = s % block_size`. Store K at `key_base + p * key_block_stride + t * width`
  and scale at `scale_base + p * scale_block_stride + t`. Offsets are in the
  respective view's elements. The ordinary path converts the scale on store;
  the HiSparse integer views copy its original bits.
- **Algorithm flow**: After connector/layout dependencies, each program loads
  one token's K, scale and slot; masks invalid slots; computes the two page
  addresses and publishes both components.
- **Supported modes**: NPU eager and ACLGraph capture/replay. The ordinary INT8
  path has Atlas A3 operator tests. The HiSparse FP8 page-strided path targets
  qualified Ascend 950PR/950DT environments. Atlas A2 coverage is N/A here.

Source: `vllm_ascend/ops/triton/sfa_indexer_store.py`.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `quant` | Input | Already quantized K, `[rows, width]`; FP8 bytes are passed through an INT8 view | INT8 | ND, contiguous |
| `scales` | Input | Ordinary scales or bit-preserving HiSparse scale aliases | FP32 or INT32 | ND, one element per row |
| `slots` | Input | Destination slot per token | INT32 or INT64 | ND, contiguous |
| `cache` | Output | K cache; physical page gaps are allowed | INT8, including FP8 aliases | Dense token rows |
| `scale_cache` | Output | Scale cache corresponding to K | FP16 or INT32 aliases of FP32 | Dense scale rows |
| `block_size` | Attribute | Tokens per page; default `1` preserves the original dense-slot API | Positive integer | Scalar |
| `key_block_stride` | Attribute | K page stride in K-view elements; default `block_size * width` | Positive integer or `None` | Scalar |
| `scale_block_stride` | Attribute | Scale page stride in scale-view elements; default `block_size` | Positive integer or `None` | Scalar |

The private `_store_indexer_key_scale` kernel additionally receives `capacity`
(runtime slot bound), `KEY_WIDTH` (feature width) and `BLOCK_SIZE` (page tokens).
The wrapper derives these attributes from the inputs and `block_size`.

## Constraints

- Valid slots must be unique within one launch. Negative and out-of-capacity
  slots perform no stores. The kernel does not resolve competing writes.
- K and scale inputs must have one row per slot. Feature width is suitable for
  the kernel's full-row `tl.arange`; the qualified HiSparse width is 128.
- The ordinary backend selects this store for native INT8/FP16 decode writes.
  C4, PCP/DSA-CP and other ordinary backends retain their separate routing.
- HiSparse uses block128, one K head, FP8 E4M3FN K128 and one FP32 scale per
  token. Within each page, all K rows precede the scale region. Consecutive
  physical pages need not be adjacent, so both page strides must be supplied.
- `view(torch.int8)` and `view(torch.int32)` reinterpret storage, not values.
  The HiSparse caller must not convert FP8 values to numeric INT8 or FP32 scales
  to numeric INT32. K/scale offsets and strides must remain aligned to their
  element sizes and within the backing allocation.
- Empty inputs launch no programs. Row count controls the grid, not a fixed
  token limit. Capacity and page strides are runtime, nonspecialized arguments.
- Input/output owners and addresses must outlive queued work and captured
  graphs. The surrounding backend controls stream and connector dependencies.

## Origin and Differences

- **Origin**: Native SFA Indexer key/scale cache publication.
- **Differences**: The ordinary kernel fuses scale conversion and two scatters.
  HiSparse adds explicit page-stride addressing and bit-preserving FP8/FP32
  stores into Core's shared allocation. It does not add quantization or move
  publication ahead of connector/query dependencies.

## Test Cases

The ordinary INT8 test checks exact K bits and FP16 scale results, slot bounds,
both slot dtypes and changing Graph inputs. The HiSparse test checks block128,
K128, physical page gaps, untouched neighboring bytes, padding, changing slots
and FP32 NaN/signed-zero/infinity bit patterns in eager and Graph modes.
These pure publication tests use exact comparison (`rtol=0`, `atol=0`).

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_sfa_dcp.py -k native_quantized
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_hisparse.py -k page_strided_indexer
```

## Example

HiSparse supplies aliases and their actual page strides after binding:

```python
store_indexer_key_scale(
    fp8_values.contiguous().view(torch.int8).reshape(-1, 128),
    fp32_scales.view(torch.int32).reshape(-1),
    slots.reshape(-1),
    key.view(torch.int8),
    scale.view(torch.int32),
    block_size=128,
    key_block_stride=key.stride(0),
    scale_block_stride=scale.stride(0),
)
```
