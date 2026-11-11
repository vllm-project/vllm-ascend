# Kimi DP-axis projection TP

Kimi K3.1 can keep its attention state on the original DP owner while sharding
projection weights over independent DP-axis groups. This implementation adapts
the common configuration/group and input-gather/head-return layout from
[PR20574](https://github.com/vllm-project/vllm-ascend/pull/20574), pinned at
`66f79fede7980c1e1192796ef0af648fe9a13106`. It extends the Kimi adapter on vllm-ascend main and reuses the existing
KDA core and SFA implementation.

## Configuration

For DP32/EP32 with TP=PCP=DCP=PP=1 on a PD decode node:

```json
{
  "finegrained_tp_config": {
    "embedding_tensor_parallel_size": 8,
    "lmhead_tensor_parallel_size": 8,
    "mlp_tensor_parallel_size": 8,
    "qkv_iproj_tensor_parallel_size": 8,
    "gproj_tensor_parallel_size": 8,
    "oproj_tensor_parallel_size": 8
  },
  "scheduler_config": {"recompute_scheduler_enable": true}
}
```

This is an additional-config fragment, not a complete deployment command. The
existing EP, KV connector and graph settings must be retained. Capture capacity
must cover `min(max_num_batched_tokens, max_num_seqs * (1 + num_speculative_tokens))`.
Invalid Kimi projection configurations fail at startup rather than silently
disabling an explicitly requested shard.

Both 0 and 1 disable a component; a size above 1 must divide DP and the relevant
weight dimensions. All six fields default to 1. Negative sizes are rejected.

| Knob | Weight layout / scope |
| --- | --- |
| embedding | Vocab rows, replacing this module's standard TP layout |
| lmhead | Vocab rows, replacing this module's standard TP layout |
| mlp | Dense FFN gate/up rows and matching down columns; existing shared/routed paths are preserved |
| qkv_iproj | KDA Q/K/V and beta, separately by whole head |
| gproj | KDA and DSA full-rank output gate rows; independent of O |
| oproj | KDA/DSA O input columns, through the existing output projection exchanges |

QKV/G require the mixed Kimi checkpoint format (quantized QKV and floating-point
beta/F/G), full-rank gates, TP=PCP=DCP=PP=1, PD decode and graph dispatch. The
existing O/MLP deployment restrictions continue to apply. Enabling only
embedding/head retains their existing support with standard TP>1; PCP vocabulary
sharding and batch-sharded sampling restrictions are unchanged.

## KDA ownership and execution

`self.tp_size` and `local_num_heads` remain the state TP/PCP domain. They are not
replaced by any projection group's size. QKV/beta are projected for gathered DP
tokens and returned to the owner in Q-major/K-major/V-major order. The owner
continues to own conv weights, A_log/dt_bias, convolution/recurrent state and
the norm/gate parameters.

When QKV or G is sharded, KDA instantiates separate QKV, beta, F_a, F_b and G
layers. F_a/F_b remain owner-local and full-rank F is not precomposed or retained.
Mixed checkpoint keys for these standalone layers are consumed by their own
loaders before upstream packed mappings. MXFP8 QKV weights/scales follow the
same per-Q/K/V output slice.

When both QKV and G are disabled, the existing mixed QKV/full-F BFG path remains
in use, even if O, vocab or dense FFN TP is enabled. G-only, QKV-only and unequal
QKV/G/O group sizes restore the same owner layout before core/gate/O joins.

The new KDA custom op has a caller-owned output buffer and a fake implementation.
It keeps Python Work objects, stream orchestration and state dispatch outside
FX tracing. The existing core is extracted without changing its conv/spec/KV
hooks or kernels. Output G is waited on only after core work has been enqueued.

| Window | Independent resources |
| --- | --- |
| Input gather / owner F | Communication / Cube |
| MX input quant / BF16 beta | Vector / Cube |
| QKV/beta return / G projection | Communication / Cube |
| G return / conv-recurrent path | Communication / Vector |

These are dependency windows, not measured speedups. Two Cube GEMMs or two
Vector kernels are not counted as mutually hidden work. Raw beta is returned
without reducing precision; the existing core prepares its FP32 sigmoid.

## DSA and peripheral modules

DSA starts G input gather on owner hidden before prolog, computes the G shard
after prolog, and launches its return before indexer/SFA work. The final gate
joins owner G after UV. Kimi O restores the original graph capacity before communication, including
NoPE PIECEWISE and zero-token owners; it trims only after the output exchange.
Dummy ranks participate in G and O exchanges. Absorbed
UK/UV, indexing and KV ownership are unchanged.

Kimi dense FFN and LM head use two owner-row chunks. Each executor launches
AG0, AG1, then RS0/RS1 or A2A0/A2A1 in the same order on every peer. This exposes
AG1/GEMM0 and output-return0/GEMM1 windows. Each chunk's output is restored before
concatenating owner rows. One-row capacities use a single chunk. Embedding keeps
its existing fixed-capacity ID-gather/lookup/reduce-scatter path.

The communication stream is owned by each executor, and every outstanding Work
is joined before that module returns. Send tensors remain referenced through
completion. Equal-sized component groups reuse the existing PG cache; teardown
destroys each underlying group only once.

The runner installs a per-execution check at the existing post-DP-dispatch
patch boundary. Both active and idle ranks reject an unsupported eager step
before projection collectives; startup profile and synthetic warmup are exempt.
The callback is restored when execution finishes. This patch can be removed
when the upstream runner provides a common post-dispatch hook.

## Validation boundaries

`tests/ut/ops/test_kimi_finegrained_tp.py` exercises all 64 switch combinations
over 32 simulated DP owners, 27 unequal QKV/G/O size combinations, real linear
loaders, MXFP8 block32 scale slicing, idle state ownership, configuration checks
and the deferred G join. Real SFA/O adapter tests cover uneven and zero-token
NoPE PIECEWISE owners. Sampling tests construct the LM head in a config scope
and exercise logits after that scope exits. The stateful test core is an ownership oracle, not a
replacement for the KDA accuracy tests.

`tests/e2e/pull_request/two_card/test_kimi_finegrained_tp.py` is a small-weight
two-NPU HCCL/MXFP8 test with graph replay. It needs no checkpoint download. It
tests the new executors, not complete-model numerical accuracy or performance.
Full DP32 checkpoint loading, model accuracy, memory peaks, real overlap and
legacy TP/DCP deployment regression still require deployment-level acceptance.
