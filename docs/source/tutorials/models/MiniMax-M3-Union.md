# Optional MiniMax M3 Union prefill

Union is an opt-in sparse-attention prefill path for MiniMax M3 on Ascend950.
The default configuration is off. Other models, decode attention, the indexer,
the selected block sets and top-k are unchanged. This integration is experimental
and requires separately built custom operators; they are not bundled with
vLLM Ascend.

## Operator prerequisites

Build and install **both** matched custom packages using their own instructions:

- [Union GBSA and metadata tutorial](https://gitcode.com/L_Euler/ops-transformer_gbsa/blob/personal%2Fwjc%2Fgbsa_union_sparse/attention/generic_block_sparse_attention/docs/union_%E6%8E%A5%E5%85%A5README.md), reviewed at `bd054f842c9ba33b4c94b234ed24998cb821c51f`.
- [Fused NN UnionSparseMeta operator](https://gitcode.com/cann/ops-nn/pull/12461).

These repositories may require access permission. The integration expects
`cann_ops_nn_custom.union_sparse_meta_launch` and the custom transformer
`generic_block_sparse_attention_metadata` / `generic_block_sparse_attention`
APIs with Union `mask_mode=3`. Exposing similarly named non-Union APIs is not
sufficient. Source the installed vendor environment before starting workers;
ensure `ASCEND_CUSTOM_OPP_PATH` resolves this matched NN/transformer pair before
incompatible packages. The attention code does not rewrite global search paths.

Supported configuration:

- Ascend950 (Arch35), BF16 or FP16 computation and matching KV cache. MXFP8
  weights can be used with BF16 KV; FP8 KV is unsupported by this Union path.
- 128-token cache pages and sparse blocks; one local KV/selector head per TP
  rank, at most 16 local query heads. TP must equal the total selector-head count.
- Union size 8, TND queries, contiguous PA_BBND K/V, sparse pattern 4,
  exact mask mode 3 and softmax precision 0.

Unsupported enabled configurations fail at model construction. The custom
extensions are loaded only for layers routed to Union; default workers do not
need these packages.

## Configuration

Merge one of these objects into the existing `--additional-config` JSON; retain
the other deployment settings from the [MiniMax M3 guide](MiniMax-M3.md).

```json
{"minimax_m3_union": {"mode": "union"}}
```

This selects Union for every sparse prefill layer. To use a frozen policy:

```json
{
  "minimax_m3_union": {
    "mode": "policy",
    "policy_path": "/models/policies/m3-union.json",
    "policy_sha256": "<SHA256 of the exact policy file bytes>"
  }
}
```

The default is `{"minimax_m3_union": {"mode": "off"}}`. File paths must be
accessible to every worker. Compute the digest with `sha256sum`; missing files,
hash mismatches, incomplete tables and incompatible topology fail before
inference. Routes are resolved once at model construction, before graph capture,
and logged with the layer, global selector head, path and policy digest.

## How the frozen policy works

The indexer first produces the same selected block IDs as the native path.
Union groups up to eight adjacent query tokens **within one request** and forms
the union of their selected blocks. An exact per-token mask prevents a token
from attending to blocks selected only by another token, and preserves causal
visibility. Sorting IDs changes their order, not the selected sets.

The policy chooses which attention implementation consumes these unchanged
selections. One layer/selector-head pair always uses either Union or the native
prefill path (`q-gather-kv` with the matching native CANN implementation).
Each TP rank owns one selector head. The same table is used by every DP replica.
The unused implementation is not executed. Decode uses its existing backend.

This is **not per-token dynamic switching**: there is no runtime device-to-host
copy of selections, repetition-rate calculation, file IO or threshold comparison
in attention forward. Different workloads may need different calibrated tables.
A table is immutable for the lifetime of the loaded model; changing its file
does not change the routes of an already loaded engine.

The policy JSON has this schema (a reduced two-layer/two-head example):

```json
{
  "valid": true,
  "routing": "frozen-layer-selector-head",
  "per_token_dynamic": false,
  "selector_modified": false,
  "decisions": {
    "0:0": {"path": "union"},
    "0:1": {"path": "q-gather-kv"},
    "1:0": {"path": "q-gather-kv"},
    "1:1": {"path": "union"}
  }
}
```

Supply exactly every `layer:selector_head` key for the actual text-model layer
count and selector-head count, including unused dense-layer entries. Additional
calibration fields may record provenance but do not change runtime routing.
For example, a 60-layer/four-selector-head model needs 240 entries, rather than
copying the small example above.

Calibrate using real indexer block selections in a separate run. For a full
aligned group of eight tokens selecting K blocks each, `D8 = U / (8 * K)`, where
U is the union size. Lower expansion can make Union attractive; an illustrative
`D8 <= 0.375` corresponds to `U/K <= 3`. This is a starting hypothesis, not an
optimized or universally profitable threshold. R8 (the intersection of the
current and previous eight selection sets divided by K) is a different metric.
Text repetition is not a substitute for either block-selection metric. This
integration neither generates nor recalibrates policies during performance runs.

## Graph-boundary MoE output buffers

Enabling Union or policy also requests explicit MoE output buffers for the
shared-expert DP / SP+SEDP layouts. Allocations are owned by the preceding
captured graph and an opaque operator writes native shared/routed results into
them, preserving the addresses seen by the following captured graph. Native
expert math, routing and communication are retained, with two device copies.
The existing latent-transform SP-overlap entry keeps precedence.
This output-buffer adapter applies to MoE calls in the enabled model, including
decode calls; the decode attention implementation itself is unchanged.
For enabled MiniMax M3 PIECEWISE graphs, the platform registers this opaque
out-op as a splitting boundary, keeping MoE execution between captured graphs.
The non-fused sparse-attention entry is also registered in this opt-in scope.

The adapter is opt-in at the MiniMax model call site. Default MiniMax M3 and
other models keep their existing MoE entry. For comparisons that must include
the same adapter cost in Q, use a policy whose entries all select `q-gather-kv`.
Compare native Q, pure Union and mixed policies on identical data and settings;
this feature does not guarantee a speedup.

## Validation and provenance

The original r79 prototype used the release-branch image with vLLM Ascend
commit `b04f258fa559d9ee630a20e1112084ae5e270172`
(`0.19.1rc2.dev3296+gb04f258fa`), not main. Its changes were file overlays,
not upstream commits. This PR ports the integration onto main rather than
copying the release model implementation. Existing main FP8/decode/indexer
features remain in place.

Run the host tests in a prepared project environment:

```bash
pytest -q tests/ut/models/minimax_m3/test_union_policy.py \
  tests/ut/models/minimax_m3/test_msa_m3_union.py \
  tests/ut/models/minimax_m3/test_msa_m3.py \
  tests/ut/models/minimax_m3/test_minimax_m3.py \
  tests/ut/test_ascend_config.py tests/ut/test_platform.py \
  tests/ut/ops/test_fused_moe.py
```

With the matched custom packages on Ascend950, run the optional eager/NPUGraph
exact-visibility tests:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_minimax_m3_union.py
```

Before merging, validate full-model graph replay and output agreement on main
with the intended CANN/vLLM revisions and TP/DP/EP topology. Benchmark the frozen
workload separately from block-selection capture, report all repetitions and
use total requests divided by total elapsed time. Historical r79 prototype
validation does not establish correctness or performance for this main port.
