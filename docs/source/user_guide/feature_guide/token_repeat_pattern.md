# Token repetition detection (pattern backend)

`token_repeat` uses a fixed pattern-based rule set with 64 raw-token check
boundaries and a final partial-block flush. There is no sliding-window
algorithm and no `algorithm` selector.

Build the optional CPU extension independently in the serving environment (Linux, C++17):

```bash
pip install ./tools/token_repeat_pattern/src
```

Set these fields under `detector.token_repeat` in runtime_config:

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
unavailable; inference can continue. Fix the extension/resources and
disable/re-enable to retry. Rebinding to a different model rebuilds the tables.

Detection uses the existing CPU ActionQueue and Incident actions. Terminal
tails are checked before Store cleanup after CPU jobs drain. Forced cleanup at
the deferred-wave limit can skip the terminal check for outstanding jobs; this
is explicitly logged.

Focused regression tests are in `tests/ut/runtime_guard/test_token_repeat_pattern.py`.
Native cases require the optional extension and skip when it is absent.
