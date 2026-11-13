# AscendStore CPU event validation

Layerwise Memcache offloads now publish CPU BlockStored events after the backend
confirms that all required groups, ranks, and paired DSpark objects are readable.
A single TP/PCP owner publishes the confirmed grains through the sharded event
path; replicated blockwise events retain the existing cross-step TP quorum.
Pending request metadata belongs to each worker, has a limit of 256 requests,
and expires after 300 seconds. No new background thread or device synchronization
is introduced. Readability polling adds backend requests; throughput impact has
not been benchmarked.

The blockwise sender also skips keys without local descriptors and maps partial
PUT results back to the original selected ranges before publishing events.
No public API, configuration option, or environment variable is added.

## Hardware evidence

On 2026-11-13, an Ascend 950DT deployment of DeepSeek-V4-Flash-0731 with prefill
DP4/TP1/PCP1 and decode DP4/TP1 passed the following live checks:

- Prefill ports 7100-7103 and decode ports 7200-7203 returned HTTP 200 on `/health`.
- Motor coordinator reported `ready=true`; both P/D instances were healthy.
- A 6413-token prompt sent through coordinator `/v1/completions` returned HTTP 200.
- Before the request, Conductor reported 32 CPU blocks and zero NPU blocks for
  prefill DP0, matching 4096 tokens at a block size of 128.
- Coordinator selected DP0 with `policy=kv_cache_affinity`, `matched=4096`,
  `hbm=0`, `cpu=4096`, and `disk=0`.
- DP0 `external_prefix_cache_hits_total` increased from 4096 to 8192 and store
  GET-key count increased from 77 to 154. Local prefix-hit count stayed zero.

These results establish actual external KV reuse, beyond an index-only hit.
The request took approximately 0.56 seconds; this is functional evidence, not a
performance benchmark. This deployment used blockwise transfer; earlier layerwise
hardware evidence used prefill PCP4 with decode DP4 and also reused 4096 tokens.
The optional Memcache 1.3.0 parser workaround was loaded in MetaService;
see `tools/memcache_event_compat/README.md`. Evidence from the running release
checkout is not an end-to-end test of the rebased main-branch PR commit.

## Reproduction

Send a stable long prompt through coordinator, wait for offload confirmation,
and query Conductor `/query` with the same tokenizer token IDs, model identity,
and actual event block size. Record each DP's `cpu_blocks`, `npu_blocks`, and
`matched_tokens`. Compare prefill external-prefix-hit and store GET counters
before and after the subsequent coordinator request, and correlate its request
ID with coordinator scheduling logs. A CPU-only test requires a DP with no local
HBM prefix; use an idle instance or a dedicated test deployment. Do not confuse
Conductor CPU-index hits with actual external KV reuse.

## Validation limits

Focused tests use the dependency mocks already provided by the AscendStore suite.
The default unit-test conftest cannot load in this release container against the
PR main-branch source: its vLLM `step3p5` model lacks `_get_layer_type`.
NPU throughput, broader model accuracy, and multi-TP layerwise hardware validation
remain outstanding. The paired-draft and incomplete-readability paths have unit
coverage.

Focused command (107 passed in the validation container):

```bash
python -m pytest -q --confcutdir=tests/ut/distributed/ascend_store \
  tests/ut/distributed/ascend_store/test_layerwise_cpu_events.py \
  tests/ut/distributed/ascend_store/test_kv_transfer.py \
  tests/ut/distributed/ascend_store/test_ascend_store_connector.py
bash tools/memcache_event_compat/build.sh
```

The native parser assertions and all 9 Python RTLD_LOCAL regression cases passed.
Ruff checks and formatting of changed Python files passed.

`timeout 240 bash format.sh ci` exited 124 while installing the Ruff pre-commit
environment. Full repository formatting/Markdown CI remains unverified; no full
CI pass is claimed.
