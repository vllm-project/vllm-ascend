# DCP fused All-to-All and LSE combine

## Purpose

`dcp_a2a_fused` exchanges partial attention outputs and FP32 log-sum-exp (LSE)
values across a context-parallel group. It returns the output slice owned by
the local rank after combining contributions from every source rank. The same
operator can defer the merge so a caller can include a local contribution later.

The implementation is in `vllm_ascend/ops/triton/dcp/dcp_a2a.py` and has three
stages:

1. A stride-aware Triton kernel packs the partial output and LSE into one send
   buffer.
2. `torch.distributed.all_to_all_single` exchanges that buffer over the DCP
   HCCL process group.
3. A Triton kernel reconstructs the LSE values and performs a numerically
   stable weighted reduction of the partial outputs.

The individual kernels are described in [pack_dcp_output_lse](pack_dcp_output_lse.md)
and [fused_dcp_lse_combine](fused_dcp_lse_combine.md).

The small native BF16/D512 head-scatter route uses the bit-preserving
[SFA DCP exchange](sfa_dcp_exchange.md) and [local merge](sfa_dcp_merge.md).
It derives the payload from the input and actual group rather than assuming
DCP8/H64. Query layout preparation is described in [query gather preparation](query_gather_prep.md).
The public operator has no scheduler-budget parameter; the backend owns
scheduling and each layout route owns its kernel-specific selection policy.

## Inputs and output

| Argument | Shape or type | Description |
| --- | --- | --- |
| `partial_output` | `[tokens, heads, head_dim]` | BF16, FP16, or FP32 partial attention output. Arbitrary positive strides are supported. |
| `softmax_lse` | `[tokens, heads, 1]`, FP32 | LSE associated with each partial output row. |
| `dcp_size` | positive `int` | Number of ranks in the All-to-All scatter group; with PCP, this can be the TP subgroup size. |
| `scatter_dim` | `0` or `1` | Dimension sharded by the All-to-All: tokens for `0`, heads for `1`. |
| `group_name` | `str` | Unique name of a live vLLM `GroupCoordinator`. |
| `pcp_group_name` | `str` or `None` | Optional PCP group used to gather rank contributions after the scatter. |
| `return_lse` | `bool` | Append merged LSE to an FP32 output for a later merge. |
| `defer_combine` | `bool` | Return the packed receive buffer after exchange instead of merging it. |

The merged result has the same dtype and `head_dim` as `partial_output`. Its
token or head dimension is divided by `dcp_size`, according to `scatter_dim`.
`defer_combine` and `return_lse` cannot both be enabled.

## Packed payload

The send buffer is laid out as
`[dcp_size, local_scatter_size, replicated_size, packed_dim]`. The first
`head_dim` elements contain the partial output. The remaining elements contain
the LSE representation.

- FP32 output stores the FP32 LSE directly in one element.
- BF16 and FP16 output use four elements: a signed base-2 exponent code and
  three base-256 digits containing the FP32 significand.

All four fields are integers in `[-255, 255]`, which are represented exactly by
both BF16 and FP16. Reconstructing the three digits therefore preserves the
original finite FP32 LSE, including nearby values above the FP16 finite range.
Exponent code zero marks NaN and infinity as invalid rank contributions.

## Triton launch strategy

The pack and combine kernels flatten `(token, head)` into one logical row index. The
launch grid is

```python
grid_size = min(num_tokens * num_heads, get_vectorcore_num())
grid = (grid_size,)
```

Each program processes additional rows with a grid-stride loop. This limits
the program count to the physical vector-core count while still covering large
prefill and decode shapes. Each row uses one `BLOCK_D` vector, where
`BLOCK_D = next_power_of_2(head_dim)`.

The scalar combine kernel keeps one FP32 output accumulator and scalar LSE state live.
It first finds the maximum valid LSE, then accumulates
`exp(lse - lse_max) * partial_output`. Invalid LSE rows contribute zero; when
all ranks are invalid, the output row is zero.
The eligible BF16 path batches eight rows per program and masks partial
tiles; other valid shapes use the scalar-row kernels.

## Custom-op registration and graph tracing

The eager implementation resolves `group_name` through vLLM's registered
process groups and executes the HCCL collective. The `fake_impl` registered
with the operator is used only by PyTorch FakeTensor/`torch.compile` shape
propagation. It returns an empty tensor with the local output metadata and does
not execute a collective.

## Validation

The single-card nightly test covers BF16/FP16, both scatter dimensions,
head dimensions 96/128/160/256, non-contiguous inputs, invalid LSE rows, and
finite FP32 LSE values outside FP16 range. The multi-card A3 nightly test starts
a real two-rank HCCL group and invokes the registered custom operator end to
end for both scatter dimensions.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dcp_a2a.py
pytest -sv tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_dcp_a2a.py
```

## Split MLA history/current attention

The split MLA path calls `dcp_a2a_fused(..., defer_combine=True)` on the
communication stream. This returns the packed receive buffer after exchange,
with shape `[source ranks, local heads, tokens, packed D]` for head scatter.
Packing and communication overlap current-token FIA on the main stream.

After the main stream waits for the communication event,
`fused_dcp_lse_combine(..., local_output=current_output,
local_lse=current_lse)` reads both raw FIA tensors using their strides. A single
kernel finds the maximum LSE over all history ranks and the local contribution,
accumulates their FP32 weighted outputs, and normalizes once. Current KV is
counted exactly once. Invalid LSE contributions are masked before multiplication;
fully invalid rows return zero output and, if requested, negative-infinite LSE.
The output dtype follows the receive buffer (FP32 for split MLA).

The default custom-op path still returns the merged output. Deferred mode and
`return_lse=True` are mutually exclusive. The single-card test covers direct
local contributions at DCP sizes 1/2/8, D=96/256/512, both scatter dimensions,
FP32/BF16/FP16 local output, strided inputs and invalid rows. The MLA unit test
checks stream ordering, one final combine, and current-KV multiplicity.

## JIT warmup

`model_executor/warmup/kernel_warmup.py` registers actual
`AscendSFADCPImpl` and `AscendMlaDCPImpl` owners from the runner's static
forward context, including draft layers. Registration runs after cache/group
setup and model dummy runs, inside the shared `JitWarmupRegistry` window and
before registry compilation and ACLGraph capture. Owners supply their own
model dtype, local heads, latent dimension and token capacity. Physical PCP
and TP groups determine scatter/receive geometry, including replicated MTP
whose logical PCP config is disabled. Repeated layer layouts register once.

The compile-only owners in `dcp/dcp_jit.py` and `dcp/dcp_warmup.py` use the same
JITFunction objects and launch-construction functions as runtime. They cover
scalar/batched pack and combine, T1/full-vector/tiled exchange packing, and
fast merge. SFA uses its contiguous native/QSFA output and FP32 max/sum views;
TurboQuant supplies already-computed LSE. MLA additionally covers FIA output,
single-batch BSND LSE views, transposed FlashMLA output/LSE, and deferred history
combine with or without one local contribution. Only the owner's dtype and
geometry are registered. `FROM_STATS`, block sizes, rank unroll and compiler
options follow the production launcher.

Representative token counts cover each dispatch interval and its modulo-16
scalar/alignment classes, including exact-one and integer-width boundaries.
The sampling uses owner capacity, not graph capture sizes, and deduplicates
Triton's scalar-specialization representatives. Received LSE pointer alignment
includes its offset inside the exchange payload. No device payload is allocated
or read, no collective runs, and native `kernel.warmup(grid=(1,))` only compiles.
`enable_jit_warmup=False` retains lazy compilation. Runtime grid, tiles,
communication, payload format and reduction math remain unchanged.

Fast merge's rank stride is now a non-specialized runtime scalar: it is used
only in address arithmetic and still receives the exact `lse_recv.stride(0)`.
Previously it was constexpr and produced one key per token count. This is the
only kernel specialization-policy change in this warmup integration.

The registration contract covers these SFA/MLA head-scatter owners. It does not
claim precompilation for unrelated GQA owners or direct public-operator calls
using token scatter, `return_lse=True`, arbitrary external tensor views or
unregistered dtypes. Those calls retain their existing runtime behavior.

`tests/ut/model_executor/warmup/test_kernel_warmup.py` contains finite-domain
versus exhaustive-token coverage, owner/draft registration and deduplication,
compile-only behavior and alignment checks. The existing
`test_dcp_pre_post.py` single-NPU suite also adds a cold in-memory JIT / registry
warmup / forbidden-recompile regression over new token counts and layouts,
followed by changed-input replay of retained graphs. These tests were added
without execution for this change; NPU compilation, graph replay,
precision and performance validation remain outstanding.

## Follow-up optimization

The existing `head_dim == 512` / `tokens >= 128` pack routing, 8/32-row pack
tiles, eight-row combine tiles, rank-unroll bound, and full-vector versus tiled
exchange thresholds are deliberately retained. Optimizing those implementation
choices requires separate shape-range measurements on A3/A5. In particular,
measure the dynamic merge rank-stride change against the former specialization;
this change does not claim a measured kernel-speed improvement.
