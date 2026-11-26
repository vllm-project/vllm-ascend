# QSA DCP group ownership

Qwen3.8 uses the same interleaved KV ownership as SFA. The compressed indexer
addresses four-token groups, so it uses a group interleave of `I / 4` for a
configured token interleave of `I`.

## Interface and invariants

- The QSA builder passes the token slot mask, request boundaries, and local
  compressed block table to vLLM's `get_compressed_slot_mapping`, also used by
  Kimi. A complete group writes one state on its owner. Replayed and padded
  rows carry slot `-1`.
- `distributed_topk(return_local=True)` returns the global winners owned by
  this rank as compact local group IDs. `prepare_qsa_gbsa_inputs` consumes
  these IDs and uses `dcp_local_length` to compute visibility in original tokens.
- The kernel emits local group IDs in selection order with `-1` padding, local
  group counts and visible token lengths, query offsets, and a page table per row.
- Interleave must be divisible by four. A compressed group and its partial tail
  therefore belong to exactly one rank. The scheduler block size must also be
  divisible by interleave, as required by vLLM's block-table mapping.
- All 512 complete groups can participate. The last three packed columns hold
  global tail token IDs, including when MTP reuses step 0. GBSA converts this
  visibility to a local prefix, so only the owner contributes residual tokens.
  A padded row contributes none, even if its packed buffer still holds a tail.
- Input cache page strides are preserved. Historical KV is never gathered.
- Rows with zero complete groups and zero trailing tokens have output zero and LSE negative infinity
  before the existing DCP output exchange/merge.

For token interleave `I`, world size `D`, rank `r`, and group ID `g`, the owner
is `floor(g / (I/4)) % D`. Its local ID is
`floor(g / (D*I/4)) * (I/4) + g % (I/4)`. Local prefix lengths use full rounds
plus the rank's clipped portion of the final round.

The BF16 LightningIndexerV2 path treats each query as a separate one-query
request. vLLM's `get_dcp_local_seq_lens` converts its global count of visible
complete groups into a rank-local count. The operator uses `mask_mode=0` and
`cmp_ratio=1`. It requests scores and reuses `distributed_topk` to select global
winners in rank-local coordinates. The indexer projection is replicated; main attention query heads are
gathered across DCP. GBSA then returns local output and FP32 LSE for the shared
DCP merge. GDN states and the indexer's raw-key ring retain their existing state
ownership.

## Specialization and validation

World size, interleave, head count, and sparse width specialize the integer
kernel. Request lengths and group IDs are tensor values and do not specialize.
There is no autotuning or new warmup registry. Runtime calls use the public
wrapper. NPU compilation counts and performance still require hardware evidence.

CPU reference tests compare exact ownership, page-slot mapping, local prefix
lengths, the full selection plus its causal tail, and dense attention/LSE equivalence. NPU tests
exercise real Triton compilation and CANN GBSA output/LSE, including an empty
rank. Run:

```bash
pytest tests/ut/models/qwen4_exp/test_dcp.py tests/ut/models/qwen4_exp/test_qsa_upstream_abi.py
pytest tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_qsa_dcp.py
```

The implementation requires PCP=1 and TP divisible by `num_key_value_heads *
DCP`, so every DCP group shares a KV head. It uses BF16 main and compressed
caches. An omitted interleave resolves to 4; explicit settings must preserve
four-token groups. Joint PCP+DCP needs a separate query-layout implementation.
Multi-rank HCCL, ACLGraph replay, real-weight accuracy, and AISBench performance
must be validated before claiming runtime support.
