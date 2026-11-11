# Token repetition detection: M4 backend

The default `token_repeat.algorithm` remains `"window"`, preserving existing
single-window configuration. Select `"m4"` for the fixed repetition rules with
64 raw-token check boundaries and a final partial-block flush. M4 requires default
values for the legacy window-scoring parameters; incompatible settings are rejected.

Build the optional CPU extension independently in the serving environment (Linux, C++17):

```bash
pip install ./tools/m4_token_repeat/src
```

Set these fields under `detector.token_repeat` in runtime_config:

```json
{
  "enabled": true,
  "algorithm": "m4",
  "m4_think_end": -1
}
```

Use `-1` for chat mode, or the model's explicit `</think>` token ID for reasoning.
M4 builds its lookup tables directly from the served model's `tokenizer.json` during
initialization or enable-time warmup, before CPU detection batches. ByteLevel BPE
is supported; remote model IDs require tokenizer.json in the local HF cache.
No separately generated NPZ, tokenizer path override, SHA check or model preset is
required. The table stays in memory and is released when the detector is disabled.
Warmup logs vocabulary size and elapsed time, with a warning above the 5-second
target. Measure this on the deployment CPU and tokenizer before enabling in production.

Warmup failure is logged once for that enable attempt and leaves M4 unavailable;
inference can continue. Fix the resource/extension and disable/re-enable to retry.
Rebinding to a different model rebuilds the tables. Drain requests before changing
the model or reasoning boundary.

M4 uses the existing CPU detection queue and Incident actions. Terminal tails are
checked before Store cleanup after CPU jobs drain. Forced cleanup at the deferred-wave
limit can skip the terminal check for outstanding jobs; this is explicitly logged.
Disable clears M4 private state. Shared IO follows runtime_guard's existing
lifecycle, including prompt-preserving cleanup when that lifecycle is available. A native batch failure isolates only failed/untrustworthy states.
Healthy requests resume from their exact consumption receipts; the batch is never replayed.

Re-enabling M4 or changing its reasoning boundary starts monitoring after the current
retained output length. `fired_at` includes that start offset. `finish_only` identifies
an alert emitted by the terminal partial-block check in the Incident report detail.

Focused regression tests are in `tests/ut/runtime_guard/test_m4_production.py`.
Native cases require the optional extension and skip when it is absent; the default
CI does not build this independent extension. Build it locally before running the
native boundary and partial-failure cases.
