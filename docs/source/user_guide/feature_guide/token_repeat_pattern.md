# Token repetition detection (pattern backend)

> **Pending open PR:** [#21492](https://github.com/vllm-project/vllm-ascend/pull/21492) — docs preview; code not on `main` yet. Sliding-window `token_repeat` is removed in that PR.

`token_repeat` uses a fixed pattern-based rule set with 64 raw-token check
boundaries and a final partial-block flush. There is no sliding-window
algorithm and no `algorithm` selector. The native CPU extension ships with
vllm-ascend (`native_pattern`); no separate install step is required.

Alert thresholds and reason strings live in packaged `native_pattern/pattern.json` (loaded once at detector init; not a `runtime_config` hot-reload field).

**Breaking change:** former sliding-window keys (`window`,
`repeat_sum_threshold`, `min_tokens`, `consecutive_hits`, `ignore_token_ids`,
`algorithm`) are rejected as unknown. Migrate to:

```json
{
  "enabled": true,
  "think_end": -1
}
```

Use `-1` for chat mode, or the model's explicit `</think>` token ID for reasoning.
Tables are built from the served model's `tokenizer.json` during enable-time
warmup. ByteLevel BPE is supported; remote model IDs require tokenizer.json in
the local HF cache. The table stays in memory and is released when the detector
is disabled. Warmup logs vocabulary size and elapsed time, with a warning above
the 5-second target.

Warmup failure is logged once for that enable attempt and leaves detection
unavailable; inference can continue. Fix the tokenizer/resources and
disable/re-enable to retry. Rebinding to a different model rebuilds the tables.

Detection uses the existing CPU ActionQueue and Incident actions. If the queue
is full or stopping, that step's detect job may be dropped (warning logged).

Terminal tails are checked before Store cleanup after CPU jobs drain. Forced
cleanup at the deferred-wave limit can skip the terminal check for outstanding
jobs; this is explicitly logged.

Alert `detail.reason` strings (native evaluation order):

| reason | meaning (short) |
|--------|-----------------|
| `vllm[32,500]x5` / `vllm[32,2000]x3` | exact raw-token period copies |
| `run>=500` | long same-char run |
| `A3_*` / `A5_*` / `P_*` | n-gram density features |
| `wrep_*` | word-repeat share |
| `sdup_*` | sentence-duplicate share |

Focused regression tests are in `tests/ut/runtime_guard/test_token_repeat_pattern.py`.
Native cases require a compiled install and skip when the extension is absent.
