# GLM-5.3 full-model PD coverage

These cases load complete GLM-5.3 checkpoints on main. They do not override
`num_hidden_layers`, modify checkpoints, or use dummy weights. P uses PCP,
both roles use Model Runner V2 and MooncakeConnectorV2, and neither role enables
DSA CP. The existing measured A3 PP case is unchanged.

## Coverage and scheduling

| Frequency | Hardware / weights | Case suffixes | Entries |
| --- | --- | --- | --- |
| Nightly | A5 W8A8C8, A5 W4A4C8 | `16K-PREFILL`, `16K-DECODE`, `16K-E2E`, `ACCURACY`, `FUNCTION` | 10 |
| Weekly | A3 W8A8C8, A5 W8A8C8, A5 W4A4C8 | `ALL-PREFILL`, `ALL-DECODE`, `ALL-E2E`, `ACCURACY` | 12 |

Names use `GLM-5.3-<quantization>-<hardware>-<suffix>`. Accuracy files are
shared between Nightly and Weekly, so there are 20 YAML files for 22 entries.
The five reduced-layer feature-disable variants per A5 quantization are excluded.

The Weekly-A5 workflow dispatches only the new multi-node matrix. Existing
single-node entries are not activated by that workflow. After the workflow
lands on main, authorized maintainers can select individual entries, for example:

```text
/nightly GLM-5.3-W8A8C8-A5-16K-E2E
/nightly GLM-5.3-W4A4C8-A5-FUNCTION
/weekly GLM-5.3-W8A8C8-A3-ALL-E2E
/weekly GLM-5.3-W4A4C8-A5-ALL-DECODE
```

## Resources and enabled features

| Platform | Allocation | P: nodes 0–1 | D: nodes 2–3 | Cache format |
| --- | --- | --- | --- | --- |
| A3 W8A8C8 | 4 × 16 logical devices | PP2, PCP16, TP1 | DP16, TP2 | INT8 |
| A5 W8A8C8 / W4A4C8 | 4 × 8 devices | PP2, PCP8, TP1 | DP16, TP1 | FP8 |

P retains all 78 layers with a `42,36` PP partition. Both roles enable expert
parallelism, chunked prefill, asynchronous scheduling and C8 caches. P uses eager
execution, FlashComm1 and one MTP token; D uses `FULL_DECODE_ONLY`, recompute
scheduling and five MTP tokens. Both select fused MC2 mode 1 from the existing
main PP configuration. PCP replaces the reference script's DSA CP path.
Flags alone do not establish that an optimization executes on a given platform;
inspect the startup configuration and runtime traces during hardware validation.

All cases, including FUNCTION, initially reserve four nodes. On A5 this is
32 devices, exceeding the scope sheet's 16-card function target. A 16-device
full-model PD function topology is **not validated by this change**. In
particular, the W8A8 checkpoint's documented colocated footprint already spans
two A5 nodes. Validate memory capacity before reducing the allocation.
On A3 the runner counts logical devices; do not equate 64 logical devices with
64 physical cards. The allocation and routing above are the executable contract.

## Workloads and gates

The feature stacking sheet specifies short context at most 16K, long context at
least 64K, and short TPOT 20/50 ms and long TPOT 30/50 ms. The following are
initial concrete workloads within those ranges, not measured capacity points:

| Profile | Input dataset length | Concurrency | Requests per phase | TPOT limit for D / E2E |
| --- | --- | --- | --- | --- |
| Nightly 16K | 16384 | 16 | 32 | Collect only |
| Weekly short 20 ms | 16384 | 16 | 32 | 20 ms |
| Weekly short 50 ms | 16384 | 32 | 64 | 50 ms |
| Weekly long 30 ms | 131072 | 8 | 16 | 30 ms |
| Weekly long 50 ms | 131072 | 16 | 32 | 50 ms |

The short workload uses the existing
`vllm-ascend/GSM8K_prefix0_in16384_bs500_GLM5_3_w8a8c8` dataset; the long
workload uses `vllm-ascend/GSM8K_prefix90_in131072_bs100_glm`. Lengths are dataset
labels; record actual token counts for each checkpoint/tokenizer. The long dataset
has a shared prefix, which only benefits the DECODE cases with P prefix caching.

- PREFILL requests one output token and records TTFT and input throughput. It
  still traverses PD; this is a prefill-dominated workload, not a P-only engine
  microbenchmark. A one-token output cannot measure TPOT, so it has no TPOT gate.
- DECODE enables P prefix caching and repeats each workload after a warmup of
  the same size. It requests 1024 output tokens. Confirm cache hits and cache
  capacity on hardware before interpreting the result as decode dominated;
  warmup alone does not guarantee all requests remain cached.
- E2E requests 1024 output tokens with P prefix caching disabled. PREFILL and
  E2E warm up execution without reusing prompt KV. Each YAML starts a separate
  server lifecycle; Weekly profiles within a YAML share the warmed server.
- Warmup and measurement use closed-loop traffic (`request_rate: 0`). Output
  lengths are requested maxima. The AISBench stream performance template controls
  EOS handling; verify actual generated lengths in the artifacts.
- Weekly D/E2E checks average TPOT against the sheet's targets. Throughput
  baselines are deliberately absent until calibrated; a passing run is not
  evidence of a throughput regression gate. Record input/output throughput,
  TTFT, TPOT and actual token counts, then add measured throughput thresholds.
- ACCURACY evaluates GPQA with 65536 maximum output tokens, temperature 1.0 and
  top-p 0.95. Its initial acceptance floor, 86.92, is derived from the existing
  A3 GLM-5.3 reference (91.92 minus 5 points), not an A5 measurement. Validate this
  floor for both A5 quantizations before merge. No deepSwe10 coverage is claimed.
- FUNCTION sends three arithmetic requests, including a repeat, through the PD
  proxy. It checks exact final answers, normal completion and positive token
  usage under the configured stack. This smoke test does not establish every
  optional optimization, tool calling or long-context correctness.

The `ALL` names map the scope sheet's four RP3 latency/context categories to the
four workloads above. They do **not** claim every RP3 workload or 1M capacity.
The server limit is 200000; 1M acceptance needs its own topology, dataset and
capacity run. The original table does not specify exact output length or batch
size, so the values above must be calibrated, not treated as product requirements.

## Validation before merge

Run the CPU topology/command and semantic-probe tests:

```bash
python -m unittest tests.ut.tools.test_glm53_full_model_cases
bash format.sh ci
```

Then run one FUNCTION and ACCURACY case for each A5 quantization, followed by
the isolated Nightly performance cases and the Weekly profiles on A3 and A5.
Capture vLLM/vLLM-Ascend commits, image digest, CANN/driver versions, checkpoint
revision, device allocation, resolved commands, startup logs, memory use, cache
hit metrics and AISBench artifacts. Confirm all 78 layers, PCP and Model Runner
V2 are active, and DSA CP is absent. Check graph/MTP compatibility on A5 and
calibrate throughput and accuracy gates before treating these cases as regression
coverage. CPU validation cannot replace these NPU runs.
