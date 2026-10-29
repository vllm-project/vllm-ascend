# SFA DCP output/LSE exchange

## Description

- **Function**: Packs partial attention output and FP32 LSE into one bit-preserving payload, exchanges it once, and merges the received contributions in FP32. The registered entry point remains `dcp_a2a_fused`.
- **Formula**: With `R` ranks, `H` gathered heads, `L = H / R` and feature width `D`, each peer receives `T * L * (D / 2 + 1)` INT32 words: two BF16 output values per word and one word per FP32 LSE. No floating-point LSE conversion is performed.
- **Algorithm flow**: Pack the local output words followed by LSE bits for each peer; invoke `all_to_all_single`; create strided views of received output/LSE without a receive-side transpose; invoke [SFA DCP merge](sfa_dcp_merge.md), then convert the final output to BF16.
- **Supported modes**: NPU eager execution and ACLGraph capture/replay, using the same API on all NPU generations. Execution/performance evidence is from Atlas A3.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `output` | Input | Partial output, `[T, H, D]` | BF16 | ND |
| `lse` | Input | Matching LSE, `[T, H, 1]` | FP32 | ND, strided |
| `group` | Attribute | Actual head-scatter process group | HCCL process group | N/A |
| merged output | Output | This rank's slice, `[T, H / R, D]` | FP32 internally, BF16 at registered operator | ND, contiguous |

## Constraints

- Group size must divide `H`. The local merge's reduction vector requires a power-of-two rank dimension.
- Feature width and local head count are derived from the input and group, without D512 or32-head routing limits. BF16 features must form complete INT32 words, and both `D / 2` and the local head count must be powers of two for the full-vector copies. Other layouts retain upstream packing.
- `LOCAL_HEADS` is static vector configuration because it determines `tl.arange` widths, including the T1 pack. Token counts and ordinary strides remain runtime values, not exact-value constexpr keys.
- Empty inputs are excluded. There is no fixed token cap or dependency on scheduler configuration.
- The existing full-vector pack is retained when both `T * (H / R)` and `(H / R) * (D / 2)` fit within16384 32-bit elements. Larger payloads use native head/feature tiles of at most4096 elements and bounded LSE chunks, with tokens distributed across Vector Cores. The payload layout and single All2All are unchanged; this threshold selects tiling rather than rejecting larger shapes.
- Output feature storage must permit INT32 reinterpretation. The exchange normalizes non-unit features, odd outer strides or odd BF16 storage offsets inside the one-collective path; it does not change collective counts based on rank-local storage.
- T1 flattens contiguous output before dtype reinterpretation. This removes odd singleton strides without copying data; `contiguous()` alone can preserve such strides.
- PCP composition, token scatter, returned-LSE and deferred-combine requests retain their existing implementations and semantics.
- Eligibility is selected once at the registered operator. The internal exchange and merge wrappers trust that contract rather than repeat defensive shape/dtype checks.

## Origin and Differences

- **Origin**: The shared `dcp_a2a_fused` pack/All2All/merge pipeline.
- **Differences**: Uses raw INT32 payloads to preserve BF16/FP32 bits and separates the received output/LSE views for the fused merge. Peer head counts, payload sizes and strides come from the input and group, not a fixed DCP8/H64 layout. T1 uses its existing compact packing kernel; multi-token packing keeps a runtime token loop.

## Test Cases

The distributed tests cover T1/2/13/48/191/192, multiple head counts, head-major storage, nonuniform and invalid LSE, graph replay with changing inputs and revisiting older captures. FP32 math is checked against an independent FP64 reference; BF16 adds its dtype-rounding bound. This does not replace or loosen existing strict BF16 tests. Distributed NPU execution has covered DCP2/4/8, not every possible power-of-two group size.

```bash
torchrun --standalone --nproc-per-node=8 -m pytest --noconftest -sv \
  tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_sfa_dcp.py -k exchange
```
