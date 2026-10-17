# GDN checkpoint/replay on Ascend

`vllm_ascend.ops.triton.mamba.gdn_recover` provides NPU building blocks for
speculative Gated DeltaNet verification. The opt-in framework path integrates
these kernels with GDN cache allocation, attention metadata and post-sampler
state commit. See the [feature guide](../../user_guide/feature_guide/gdn_recover.md)
for supported configurations and the launch option.

The algorithm follows the existing
[GDN RecoverSSM proposal](https://github.com/vllm-project/vllm/pull/58863)
and its [Qwen3.5 follow-up](https://github.com/vllm-project/vllm/pull/61309).
The contribution here is the Ascend implementation and its storage/metadata
contract, rather than a new recovery algorithm.

## Recurrence and record

For each value head, verification evaluates the original recurrence:

```text
decayed = exp(g) * state
correction = beta * (value - decayed @ key)
state = decayed + correction outer key
output = state @ (query * scale)
```

Q/K normalization and gate/beta activation happen before the primitive.
Verification reads a checkpoint but never writes it. Each token stores the
FP32 correction, normalized key and actual decay used by the recurrence.
Commit replays only the processed input prefix in the same arithmetic order,
then writes the checkpoint once. Floating-point fusion is disabled; equality
with other fused kernels still requires numerical and model-level qualification.

## Tensor contract

| Tensor | Shape | Dtype |
| --- | --- | --- |
| Checkpoint | `[slots, value_heads, value_dim, key_dim]` | FP32 |
| Replay record | `[slots, value_heads, window, value_dim + key_dim + 1]` | FP32 |
| Q/K | `[tokens, qk_heads, key_dim]` | BF16 or FP32 |
| V | `[tokens, value_heads, value_dim]` | BF16 or FP32 |
| Log decay / beta | `[tokens, value_heads]` | BF16 or FP32 |
| Slot indices | `[batch]` | INT32 or INT64 |
| Query offsets | `[batch + 1]`, cumulative packed offsets | INT32 or INT64 |
| Processed prefix lengths | `[batch]` | INT32 or INT64 |

Value heads must be divisible by Q/K heads. Key/value dimensions may differ;
each is in `1..256`. The window is in `1..16` and includes the mandatory input
token. Only the leading slot/token stride may have padding; inner dimensions
must be contiguous. Storage offsets and device pointer products use 64-bit
addressing. No cache materialization or device-to-host metadata read is needed.

All tensors must reside on the same NPU. Slot zero is reserved and negative
indices are padding; those rows emit zero outputs and do not modify cache or
records. Positive indices must be in range and unique within a launch. Offsets
must partition the packed inputs, with each query length at most the record
window. These device-value invariants are the caller's responsibility; Python
checks shapes, dtypes and strides without synchronizing device metadata.
Checkpoint, record, saved history and output regions must not overlap.

## Calls and ordering

1. `gdn_save_conv_history`: save the active slots' pre-verify conv history.
2. Run the speculative convolution and construct normalized Q/K and gates.
3. `gdn_recover_verify`: produce output and replay records, preserving SSM.
4. Sample and compute the number of **processed input tokens** to commit.
5. `gdn_recover_commit` and `gdn_commit_conv_history`: commit the same prefix.
6. Complete both commits before the next read/verify of that request's state.

Zero accepted drafts normally still commits the mandatory input token. The
newly sampled bonus token has not been processed by this verification and must
not be included. A prefix count of zero is available for rollback/testing;
negative/oversized counts are clipped to `[0, query_length]`. Replaying the
same record twice is invalid: the caller owns exactly-once commit ordering.

Convolution primitives accept `[slots, history + window - 1, channels]` and
saved `[slots, history, channels]` tensors with matching BF16/FP32 dtype.
`history` is in `1..16`. After speculative convolution, a positive prefix
compacts `conv[prefix - 1 : prefix - 1 + history]`; zero restores saved history.
The remaining expanded rows and saved history stay unchanged.

The primitives can be captured into an NPU graph after compilation. The
caller supplies persistent buffers and updates their contents before replay;
graph replay does not run Python metadata validation. Graph support at this
level does not establish engine-level post-sampler commit, preemption, prefix
caching, TP/PP or cache-transfer support.

## Tests

```bash
pytest -q tests/ut/ops/test_gdn_recover_contract.py
pytest -q tests/e2e/pull_request/one_card/triton/test_gdn_recover.py
pytest -q tests/e2e/pull_request/one_card/triton/test_gdn_recover_batch.py
pytest -q tests/e2e/pull_request/one_card/triton/test_gdn_recover_conv.py
```

The NPU tests use a direct CPU recurrence oracle independent of replay records.
They cover every prefix, ragged and empty rows, grouped heads, non-square and
maximum dimensions, padded storage, INT32/INT64 metadata, unchanged checkpoint
and unselected bytes, convolution rollback/compaction, dynamic graph replay,
consecutive rounds, slot reuse and a slot-stride product beyond 8GiB.
The large-address test initializes and checks occupied regions only.
The real-convolution test calls the installed FLA prefill/update operators,
checks output against a direct convolution oracle, and verifies history across
speculative commits and ordinary decode transitions.

## Memory accounting

For one FP32 checkpoint and a window of records, SSM plus record bytes are:

```text
4 * value_heads * (value_dim * key_dim
                  + window * (value_dim + key_dim + 1))
```

For 48 value heads, K=V=128 and window=4 this is approximately 3.188MiB,
versus 12MiB for four full FP32 snapshots. Conv history, saved history, page
padding, attention caches and graph workspace are additional costs. Adding
records replaces per-draft SSM reservations in the integrated path. This
reduces state blocks charged per request; the allocator may retain the same
physical pool. Throughput and latency depend on load and require measurement.
