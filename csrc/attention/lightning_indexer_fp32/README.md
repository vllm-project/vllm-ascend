# LightningIndexer FP32 scores (Ascend 910B)

`torch.ops._C_ascend.npu_lightning_indexer_fp32` is an explicit opt-in
alternative to `npu_lightning_indexer`. Its CANN name is
`LightningIndexerFp32`; its ACLNN entry points are
`aclnnLightningIndexerFp32GetWorkspaceSize` and `aclnnLightningIndexerFp32`.
The legacy operator, its symbols, defaults and query-dtype score output remain
available unchanged.

The new operator publishes the existing internal float TopK scores directly,
without casting through FP16/BF16 and without recomputing QK. Query/key input
types and scoring/selection algorithms are inherited from LightningIndexer.
Indices are `torch.int32`; scores are always `torch.float32`.

```python
indices, scores = torch.ops._C_ascend.npu_lightning_indexer_fp32(
    query, key, weights,
    actual_seq_lengths_query=actual_seq_lengths_query,
    actual_seq_lengths_key=actual_seq_lengths_key,
    layout_query="TND",
    layout_key="TND",
    return_value=True,
)
```

All arguments after `weights` are keyword-only. Defaults match the legacy Torch
API: optional sequence lengths and block table are `None`, both layouts are
`"BSND"`, `sparse_count=2048`, `sparse_mode=3`, both token limits are
`9223372036854775807`, and `return_value=False`.

With `return_value=True`, indices and scores have shape `[B, S, Nk, K]` for
BSND queries or `[T, Nk, K]` for TND queries, where `Nk` is the key head count
and `K` is `sparse_count`. Invalid entries retain index `-1` and score `-inf`
(the FP32 bit pattern `0xff800000`). With `return_value=False`, selection still
runs and the score result is an empty FP32 tensor of shape `[0]`; score stores
are disabled. At the ACLNN boundary, the score output may be null only when
`returnValues=false`, as in the legacy wrapper.

Only Ascend 910B / arch22 is supported. Normal `csrc/build_aclnn.sh` packaging
includes this operator in the 910B package only. The CANN definition advertises
only `ascend910b`, tiling rejects other SoCs, and the kernel rejects other AICore
architectures. This does not enable Ascend 910C (`ascend910_93`) or Ascend 950 /
arch35. The shared LightningIndexer dependency supplies host validation, tiling,
and arch22 computation; it does not expand this operator's target support.

The Meta registration provides output shape/dtype propagation for concrete
supported layouts, including FakeTensor use. It is not a claim of dynamic-shape,
TorchAir conversion, ACL graph or end-to-end compiled-model support.

TopK membership may differ at an **exact** cutoff tie, and valid selected IDs
may be permuted. There is no stable sort or ID tie-break guarantee. Ordered IDs
are diagnostic only. Correctness checks must verify valid unique IDs, valid
counts/sentinels, row/request/causal mapping, inclusion of scores strictly above
the cutoff, and legal tied members. Close scores are not necessarily exact ties.
Consumer numerical equivalence and model quality require separate validation.

CPU contract tests are in `tests/ut/ops/test_lightning_indexer_fp32.py`. They
compile the real Torch adapters with ACLNN execution stubbed, check CPU/Meta/Fake
output contracts and legacy coexistence, and inspect host/kernel/build contracts.
They do not establish device correctness, binary packaging, model accuracy or
performance. A supported CANN/PyTorch build and 910B device regression are needed
before merge.
