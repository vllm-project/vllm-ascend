# vLLM 0.30.0 bounded replay backport

This opt-in adapter reuses [vLLM PR #56227](https://github.com/vllm-project/vllm/pull/56227)
(head `4909bfff0ed7ee3d9036c3b8334dd54bbd9e0577`) on v0.30.0
(`ced6857afa0ea7b2e3f0846a62e1394e90f15607`). The diff contains only configuration,
scheduler, request payload, KV cache and model-state changes. It retains the
v0.30.0 scheduler lookahead initialization. Ascend supplies its own attention
and slot-padding implementation.

Prepend this directory and the Ascend checkout to `PYTHONPATH` before starting
Python, including tests and the ordinary `vllm serve` command:

```bash
export PYTHONPATH=/workspace/ascend-current/vllm_ascend/patch/compat/vllm030:/workspace/ascend-current:$PYTHONPATH
vllm serve MODEL --enable-prefix-caching --swa-bounded-replay
```

Use `--no-swa-bounded-replay` for the control run. Python's startup hook installs
the adapter in spawned workers as well. Configuration classes are created with
the added fields before Pydantic and serialization schemas are generated.
The loader compiles patched source in memory without writing vLLM source or
bytecode files. It rejects other vLLM versions, late activation, missing source,
and hunks that do not match the installed source. Existing installation-wide
`sitecustomize.py` hooks must be composed explicitly before using this opt-in path.

Remove this compatibility directory when the image pin includes #56227. The
existing platform replay patch then continues to supply the V1 resume payload.

`test_prefix_replay.py.txt` is the unmodified upstream regression test. Copy it
to `tests/v1/core/test_prefix_replay.py` in a v0.30.0 source checkout and run that
test with the compatibility directory first on `PYTHONPATH`. It covers local
and remote prefix hits, replay-only chunks, block alignment and short-hit
fallback. These scheduler tests do not establish model output accuracy or an
NPU prefix-hit-rate improvement; those require the model service E2E run.
